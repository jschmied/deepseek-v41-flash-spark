"""observe.py -- the event sink. Watches what the framework does; never part of it.

THE RULE. Components emit facts about their own state transitions; the Observer decides what to
record. An Observer must never influence scheduling, and Leaves must never own framework metrics.

WHY NOT TIMERS IN LEAVES. A provider performs an operation; only the framework that invokes it
knows when that operation was queued, when it began blocking, when it became ready and when it was
consumed. `stats.read_s += ...` inside read() would move measurement semantics into the execution
seam, so swapping the provider would silently change what the numbers mean. The framework brackets
the call instead.

INSTRUMENT AT THE OWNERSHIP POINT. Chain owns edges, SlotReady owns readiness, StagingPool owns
leases, LoaderService owns admission. Each emits once, where it lives -- never once per caller, or
the same wait is counted twice under two names.

IDENTITY THAT SURVIVES ASYNC. Every expert movement carries (step, layer, key, slot, generation).
The generation is what disambiguates a slot that held five different experts during one request --
it exists for correctness already, and it pays for itself again here. Prefetch additionally carries
prediction_id, source_layer and horizon in `aux`, so a prediction's life is readable directly
rather than inferred from timestamps:

    prediction at L12 -> queued at L12 -> read at L13 -> ready at L14 -> consumed at L16

COST. NullObserver is the default and `enabled` is a class attribute, so every emission site is
guarded by one attribute load and a branch. Nothing is constructed on the null path -- a decode step
crosses these sites several hundred times.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import time


class WaitReason(str, enum.Enum):
    """Why the critical path was blocked. The most valuable axis in this engine: durations of
    operations are much less interesting than which dependency stalled the consumer."""

    EXPERT_DATA = "expert_data"          # the bytes are not in the arena yet
    NVME_ADMISSION = "nvme_admission"    # allowed to read, but not admitted
    STAGING_BUFFER = "staging_buffer"    # no pinned buffer free
    SLOT_READER = "slot_reader"          # a previous reader of this slot has not finished
    COMPUTE_BARRIER = "compute_barrier"  # D1: the H2D waits for ALL compute, not just this slot
    H2D_CAPACITY = "h2d_capacity"        # copy engines saturated
    GLOBAL_BARRIER = "global_barrier"    # D3: waiting on reads this consumer does not need
    ENGRAM = "engram"                    # the second NVMe stream
    CHAIN = "chain"                      # a declared happens-before edge


@dataclasses.dataclass(frozen=True, slots=True)
class OpContext:
    """Carried instead of threading step/layer through every signature. Becomes load-bearing the
    moment more than one request is in flight."""

    request_id: int = 0
    step: int = -1
    layer: int = -1

    def at(self, layer: int) -> "OpContext":
        return OpContext(self.request_id, self.step, layer)


@dataclasses.dataclass(slots=True)
class Event:
    ts_ns: int
    kind: str
    step: int = -1
    layer: int = -1
    key: tuple | None = None
    slot: int = -1
    gen: int = -1
    value: float = 0.0
    aux: object = None


class Observer:
    """Emission sites are guarded by `enabled`, so the null path constructs nothing."""

    enabled = True

    def emit(self, event: Event) -> None:
        raise NotImplementedError

    # GPU ranges go through the same sink. A profiling implementation records CUDA events and
    # resolves them LATER -- never synchronously, which would serialise the stream it is measuring.
    # An NVTX implementation lives here too, so NVTX never becomes a framework dependency.
    def gpu_begin(self, name: str, ctx: OpContext) -> None:
        pass

    def gpu_end(self, name: str, ctx: OpContext) -> None:
        pass


class NullObserver(Observer):
    enabled = False

    def emit(self, event: Event) -> None:
        pass


class CounterObserver(Observer):
    """Cheap production metrics: aggregates on arrival, keeps no history."""

    def __init__(self):
        self.counts: collections.Counter = collections.Counter()
        self.sums: collections.Counter = collections.Counter()
        self.wait_ns: collections.Counter = collections.Counter()   # by WaitReason
        self._open: dict = {}

    def emit(self, event: Event) -> None:
        k = event.kind
        self.counts[k] += 1
        if event.value:
            self.sums[k] += event.value
        if k == "wait_start":
            self._open[(event.aux, event.slot, event.layer)] = event.ts_ns
        elif k == "wait_end":
            t0 = self._open.pop((event.aux, event.slot, event.layer), None)
            if t0 is not None:
                self.wait_ns[event.aux] += event.ts_ns - t0

    def report(self) -> dict:
        total = sum(self.wait_ns.values()) or 1
        return {
            "counts": dict(self.counts),
            "wait_ms": {r: ns / 1e6 for r, ns in self.wait_ns.items()},
            "wait_share": {r: ns / total for r, ns in self.wait_ns.items()},
        }


class TraceObserver(Observer):
    """Raw events into a FIXED-SIZE ring. The hot path appends a tuple and advances an index.

    No formatting, no JSON, no lock: a worker writes its own slot and the index is advanced with a
    single increment under the GIL. Serialisation happens in `drain`, off the hot path. A full ring
    overwrites the oldest events, because dropping history is always better than stalling the thing
    being measured.
    """

    def __init__(self, capacity: int = 1 << 20):
        self.capacity = capacity
        self._buf: list = [None] * capacity
        self._i = 0
        self.dropped = 0

    def emit(self, event: Event) -> None:
        i = self._i
        self._i = i + 1
        if i >= self.capacity:
            self.dropped += 1
            i %= self.capacity
        self._buf[i] = event

    def drain(self) -> list:
        n = min(self._i, self.capacity)
        if self._i <= self.capacity:
            return [e for e in self._buf[:n] if e is not None]
        start = self._i % self.capacity
        return [e for e in (self._buf[start:] + self._buf[:start]) if e is not None]


now_ns = time.perf_counter_ns
