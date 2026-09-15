"""observe.py -- the event sink. Watches what the framework does; never part of it.

THE RULE. Components emit facts about their own state transitions; the Observer decides what to
record. An Observer must never influence scheduling, and Leaves must never own framework metrics.

WHY NOT TIMERS IN LEAVES. A provider performs an operation; only the framework that invokes it
knows when it was queued, began blocking, became ready and was consumed. `stats.read_s += ...`
inside read() would move measurement semantics into the execution seam, so swapping the provider
would silently change what the numbers mean. The framework brackets the call instead.

INSTRUMENT AT THE OWNERSHIP POINT. Chain owns edges, SlotReady owns readiness, StagingPool owns
leases, LoaderService owns admission. Each emits once, where it lives -- never once per caller, or
the same wait is counted twice under two names.

IDENTITY THAT SURVIVES ASYNC. Every event carries an OpContext (request, step, layer) and every
wait carries a unique `span`. Both are load-bearing rather than decorative: the context is queued
THROUGH the loader with the work, so a read completing on a worker thread still knows which step
asked for it, and the span is what pairs a start with its own end when eight workers are blocked on
the same resource at the same instant. Pairing on (reason, slot, layer) silently overwrote itself.

CRITICAL PATH IS NOT RESOURCE PRESSURE. Loader waits overlap: eight workers each blocked 5 ms on
staging during the same 5 ms of wall time sum to 40 ms, which is real pressure and NOT 40 ms of
anybody's critical path. WaitScope separates them, and the report never adds the two together.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import itertools
import threading
import time


class WaitScope(str, enum.Enum):
    CONSUMER = "consumer"        # the driver itself is blocked: this IS the critical path
    LOADER = "loader"            # a worker is blocked on a resource: pressure, overlapping
    DEPENDENCY = "dependency"    # a declared happens-before edge


class WaitReason(str, enum.Enum):
    EXPERT_DATA = "expert_data"          # consumer: the bytes are not in the arena yet
    GLOBAL_BARRIER = "global_barrier"    # consumer: D3, waiting on reads it does not need
    ENGRAM = "engram"                    # consumer: the second NVMe stream
    CHAIN = "chain"                      # dependency: a declared edge
    NVME_ADMISSION = "nvme_admission"    # loader: allowed to read, not admitted
    STAGING_BUFFER = "staging_buffer"    # loader: no pinned buffer free
    SLOT_READER = "slot_reader"          # loader: a previous reader of this slot is unfinished
    H2D_CAPACITY = "h2d_capacity"        # loader: copy engines saturated
    COMPUTE_BARRIER = "compute_barrier"  # loader: D1, the H2D waits for ALL compute


SCOPE = {
    WaitReason.EXPERT_DATA: WaitScope.CONSUMER,
    WaitReason.GLOBAL_BARRIER: WaitScope.CONSUMER,
    WaitReason.ENGRAM: WaitScope.CONSUMER,
    WaitReason.CHAIN: WaitScope.DEPENDENCY,
    WaitReason.NVME_ADMISSION: WaitScope.LOADER,
    WaitReason.STAGING_BUFFER: WaitScope.LOADER,
    WaitReason.SLOT_READER: WaitScope.LOADER,
    WaitReason.H2D_CAPACITY: WaitScope.LOADER,
    WaitReason.COMPUTE_BARRIER: WaitScope.LOADER,
}


@dataclasses.dataclass(frozen=True, slots=True)
class OpContext:
    """Carried instead of threading step/layer through every signature, and queued WITH the work so
    an async completion still knows who asked for it."""

    request_id: int = 0
    step: int = -1
    layer: int = -1

    def at(self, layer: int) -> "OpContext":
        return OpContext(self.request_id, self.step, layer)


NO_CTX = OpContext()
_span = itertools.count(1)


def next_span() -> int:
    """Unique id for one wait. `itertools.count.__next__` is a single bytecode on CPython, which is
    why it is used instead of a lock in a path that exists to measure locks."""
    return next(_span)


@dataclasses.dataclass(slots=True)
class Event:
    ts_ns: int
    kind: str
    ctx: OpContext = NO_CTX
    key: tuple | None = None
    slot: int = -1
    gen: int = -1
    span: int = 0
    value: float = 0.0
    aux: object = None

    @property
    def step(self) -> int:
        return self.ctx.step

    @property
    def layer(self) -> int:
        return self.ctx.layer


class Observer:
    """Emission sites are guarded by `enabled`, so the null path constructs nothing.

    AN OBSERVER MUST NOT THROW. It is called from the engine's own critical path and from loader
    workers; an exception there would change or abort execution, which breaks the one contract this
    layer has. `safe_emit` enforces it by quarantining an observer that raises, rather than trusting
    every implementation to be careful.
    """

    enabled = True
    failed: BaseException | None = None

    def emit(self, event: Event) -> None:
        raise NotImplementedError

    def safe_emit(self, event: Event) -> None:
        try:
            self.emit(event)
        except BaseException as exc:            # noqa: BLE001 -- deliberately total
            self.failed = exc
            self.enabled = False                # quarantined: the engine continues unaffected

    # GPU ranges go through the same sink. A profiling implementation records CUDA events and
    # resolves them LATER -- never synchronously, which would serialise the stream it measures. An
    # NVTX implementation lives here too, so NVTX never becomes a framework dependency.
    def gpu_begin(self, name: str, ctx: OpContext) -> None:
        pass

    def gpu_end(self, name: str, ctx: OpContext) -> None:
        pass


class NullObserver(Observer):
    enabled = False

    def emit(self, event: Event) -> None:
        pass

    def safe_emit(self, event: Event) -> None:
        pass


class CounterObserver(Observer):
    """Cheap production metrics. Pairs waits by SPAN, so concurrent waits on one resource survive."""

    def __init__(self):
        self.counts: collections.Counter = collections.Counter()
        self.wait_ns: collections.Counter = collections.Counter()      # by WaitReason
        self.wait_n: collections.Counter = collections.Counter()
        self.lead_ns: list = []
        self._open: dict = {}
        self._lk = threading.Lock()

    def emit(self, event: Event) -> None:
        k = event.kind
        with self._lk:
            self.counts[k] += 1
            if k == "wait_start":
                self._open[event.span] = event.ts_ns
            elif k == "wait_end":
                t0 = self._open.pop(event.span, None)
                if t0 is not None:
                    self.wait_ns[event.aux] += event.ts_ns - t0
                    self.wait_n[event.aux] += 1
            elif k == "prefetch_ready_hit" and event.value:
                self.lead_ns.append(event.value)

    def report(self) -> dict:
        """Two axes, never summed together. Consumer/dependency time is critical path; loader time
        is resource pressure across overlapping workers and does not add to a wall clock."""
        by_scope: dict = {}
        for reason, ns in self.wait_ns.items():
            by_scope.setdefault(SCOPE.get(reason, WaitScope.LOADER), {})[reason] = ns / 1e6
        crit = sum(by_scope.get(WaitScope.CONSUMER, {}).values())
        return {
            "counts": dict(self.counts),
            "critical_path_ms": by_scope.get(WaitScope.CONSUMER, {}),
            "dependency_ms": by_scope.get(WaitScope.DEPENDENCY, {}),
            "loader_pressure_ms": by_scope.get(WaitScope.LOADER, {}),
            "critical_path_total_ms": crit,
            "mean_lead_ms": (sum(self.lead_ns) / len(self.lead_ns) / 1e6) if self.lead_ns else 0.0,
        }


class TraceObserver(Observer):
    """Raw events into PER-THREAD rings, merged on drain.

    One shared ring is not thread-safe: `i = self._i; self._i = i + 1; buf[i] = e` is three
    bytecodes and 48 loader threads will interleave them, losing and overwriting events. A global
    lock would instead put contention into the path that exists to measure contention. Per-thread
    rings avoid both; merging by timestamp happens in drain(), off the hot path.
    """

    def __init__(self, capacity: int = 1 << 20):
        self.capacity = capacity
        self._local = threading.local()
        self._rings: list = []
        self._lk = threading.Lock()

    def _ring(self):
        r = getattr(self._local, "ring", None)
        if r is None:
            r = self._local.ring = [[None] * self.capacity, 0]
            with self._lk:
                self._rings.append(r)
        return r

    def emit(self, event: Event) -> None:
        r = self._ring()
        i = r[1]
        r[0][i % self.capacity] = event
        r[1] = i + 1

    @property
    def dropped(self) -> int:
        with self._lk:
            return sum(max(0, r[1] - self.capacity) for r in self._rings)

    def drain(self) -> list:
        out = []
        with self._lk:
            rings = list(self._rings)
        for buf, n in rings:
            if n <= self.capacity:
                out.extend(e for e in buf[:n] if e is not None)
            else:
                s = n % self.capacity
                out.extend(e for e in (buf[s:] + buf[:s]) if e is not None)
        out.sort(key=lambda e: e.ts_ns)
        return out


now_ns = time.perf_counter_ns
