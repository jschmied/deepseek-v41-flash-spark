"""sched.py -- the loader service, the compute stream, and the swappable scheduling policy.

v1 and v2 are ARMS OF THE SAME PROGRAM. The loader below is always a continuous service fed by a
queue; what the arms change is the SCHEDULING POLICY -- whether the driver awaits it immediately,
what it waits on, when the staging lease comes back, and in which order the two halves of the
layer's compute run. That is the only honest way to attribute a difference to a dependency rather
than to two separately written programs.

THE FIVE TOGGLES. The design note names four false dependencies; the fifth is the fork-join shape
itself, separated out because the coordinator is right that "resolve blocks" and "global barrier"
may be the same phenomenon in reality and a table with independent columns would over-credit them.

  shape  resolve_blocks          reserve -> submit -> BLOCK, all in one call. v1's `resolve()` ends
                                 in `list(self.pool.map(...))`, so nothing can issue work except a
                                 call that immediately waits on it.
  D1     compute_barrier_global  the H2D waits for ALL compute to be idle, rather than for the one
                                 slot's previous reader. v1's `stream.wait_stream(compute)`.
  D2     lease_until_completion  the pinned staging lease is held submit -> completion, so a demand
                                 miss can block on a BUFFER while the device is idle.
  D3     global_barrier          the wait is over every pending read rather than the slots this
                                 layer actually needs. v1's `join_pending()`.
WHAT IS NOT A TOGGLE HERE, and why. An earlier revision carried a fifth switch, `moe_before_shared`
(D4): run the routed MoE before the shared expert, so the one piece of expert-independent
post-router work cannot overlap the reads. It has been moved to the LEAF PROVIDER
(`leaves.Leaves.shared_first`). engine/fastdecode.py captures the routed MoE and the shared expert
into ONE graph (B), so reordering them is not a scheduling decision the driver can make -- it needs
graph B captured in two pieces. Leaving it in Policy modelled a capture-time cost as free and would
have over-credited v2.

v1 = all four True. v2 = all four False.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import queue
import threading
import time

from .leaves import Bandwidth, ModelLeaves
from .store import ExpertSlots, SlotArena, SlotReady, StagingPool


def _default_leaves(scale: float):
    """A provider with no trace: read/h2d only. Used by tests that drive the loader directly."""
    return ModelLeaves(calls=(), bw=Bandwidth(scale=scale), scale=scale)


@dataclasses.dataclass(frozen=True)
class Policy:
    resolve_blocks: bool = True
    compute_barrier_global: bool = True
    lease_until_completion: bool = True
    global_barrier: bool = True

    @property
    def name(self) -> str:
        return "v1" if all(dataclasses.astuple(self)) else \
               "v2" if not any(dataclasses.astuple(self)) else "mixed"


V1 = Policy()
V2 = Policy(False, False, False, False)


class ComputeStream:
    """The model's compute stream, as the loader threads can see it.

    Two different questions are fused in v1 and separated here:
      * CORRECTNESS ORDERING -- this slot's previous reader must finish before it is overwritten.
      * BANDWIDTH POLICY     -- when the H2D may run at all.
    `stream.wait_stream(compute)` answers the first by asking the second, which is why it is listed
    as a false dependency.
    """

    def __init__(self):
        self._lk = threading.Lock()
        self._cv = threading.Condition(self._lk)
        self.busy = 0
        self.readers: collections.Counter = collections.Counter()
        self.compute_s = 0.0

    @contextlib.contextmanager
    def run(self, slots=()):
        with self._lk:
            self.busy += 1
            self.readers.update(slots)
            self._cv.notify_all()
        try:
            yield
        finally:
            with self._lk:
                self.busy -= 1
                self.readers.subtract(slots)
                self._cv.notify_all()

    def wait_idle(self, timeout: float = 30.0) -> None:
        with self._lk:
            if not self._cv.wait_for(lambda: self.busy == 0, timeout):
                raise TimeoutError("compute never went idle")

    def wait_slot_free(self, slot: int, timeout: float = 30.0) -> None:
        with self._lk:
            if not self._cv.wait_for(lambda: self.readers[slot] <= 0, timeout):
                raise TimeoutError(f"slot {slot} never stopped being read")


class LoaderService:
    """A CONTINUOUS service fed by a queue -- not a function that is called and awaited.

    Workers run for the life of the engine. The driver puts (key, slot, gen) on the queue and the
    policy decides whether it then waits, and on what. This is the structural change v2 is about:
    it is what makes any lookahead expressible at all. (There is none available at decode without
    prediction -- layer L's ids cannot exist before layer L-1's output -- and phase2.py reports
    exactly that.)
    """

    def __init__(self, arena: SlotArena, slots: ExpertSlots, compute: ComputeStream,
                 policy: Policy, leaves=None, n_workers: int = 48, staging: int = 48,
                 device_queue_depth: int = 8, scale: float = 1.0, fail: set | None = None):
        self.arena = arena
        self.slots = slots
        self.compute = compute
        self.policy = policy
        # The I/O half of the seam. A provider supplies read() and h2d(); the modelled one sleeps,
        # a real one does O_DIRECT into a pinned buffer and a device copy out of it. `bw` is the
        # modelled device and exists only for statistics -- a real provider has none, so every
        # reader of it must tolerate None.
        self.leaves = leaves if leaves is not None else _default_leaves(scale)
        self.bw = getattr(self.leaves, "bw", None)
        self.stage = StagingPool(staging)
        self.ready = SlotReady()
        self.scale = scale
        # Backpressure on DEVICE QUEUE DEPTH, not on thread count: the device saturates near 2
        # concurrent reads, so threads exist to hide per-read latency, not to add bandwidth. Held
        # equal across arms so it cannot confound the dependency table; sweep it separately.
        self.dq = threading.Semaphore(device_queue_depth)
        self.fail = fail if fail is not None else set()
        self.q: queue.Queue = queue.Queue()
        self.h2d_calls = 0
        self.h2d_s = 0.0
        self._lk = threading.Lock()
        self._stop = threading.Event()
        self.workers = [threading.Thread(target=self._worker, daemon=True,
                                         name=f"loader-{i}") for i in range(n_workers)]
        for w in self.workers:
            w.start()

    # ------------------------------------------------------------------ the leaf
    def _load_one(self, key: tuple, slot: int, gen: int) -> None:
        """read -> (handoff) -> compute-order barrier -> H2D -> per-slot event.

        Every step between the lease and the release must be inside the try, or the lease leaks and
        there are only `staging` of them for the life of the process.
        """
        sid = self.stage.acquire()
        released = False
        try:
            if key in self.fail:
                raise IOError(f"injected NVMe failure {key}")
            payload = self.leaves.read(key, sid)
            if not self.policy.lease_until_completion:
                # D2 OFF: the lease covers the READ only. Buffer count stops being coupled to read
                # duration. NOTE the physical caveat, recorded because the skeleton cannot check
                # it: releasing here assumes the H2D no longer reads that pinned buffer, which on
                # the real path needs either a second buffer or a copy the device owns.
                self.stage.release(sid)
                released = True
            if self.policy.compute_barrier_global:
                self.compute.wait_idle()          # D1 ON: wait for ALL compute
            else:
                self.compute.wait_slot_free(slot)  # D1 OFF: only this slot's previous reader
            with self.dq:
                with self.arena.writing(slot, key):
                    t0 = time.perf_counter()
                    self.leaves.h2d(slot, key, payload)
                    dt = time.perf_counter() - t0
            with self._lk:
                self.h2d_calls += 1
                self.h2d_s += dt
            self.ready.set(slot, gen)
        except BaseException as exc:              # noqa: BLE001
            # A failed expert must be UN-MAPPED: its slot holds a torn read, and leaving the key
            # mapped would make the next reserve() count it as a HIT and compute with partial
            # bytes -- silent, and permanent for the life of the process.
            self.slots.forget(key, slot)
            self.ready.set(slot, gen, err=exc)
        finally:
            if not released:
                self.stage.release(sid)

    def _worker(self) -> None:
        while True:
            item = self.q.get()
            try:
                if item is None:
                    return
                self._load_one(*item)
            finally:
                self.q.task_done()

    # ------------------------------------------------------------------ service api
    def submit(self, to_load) -> None:
        for key, slot, gen in to_load:
            self.ready.arm(slot, gen)
        for item in to_load:
            self.q.put(item)

    def wait_slots(self, to_load) -> None:
        """Wait only for the slots THIS consumer needs. Errors are drained, then the first re-raised."""
        err = None
        for key, slot, gen in to_load:
            try:
                self.ready.wait(slot, gen)
            except BaseException as e:            # noqa: BLE001
                if err is None:
                    err = e
        if err is not None:
            raise err

    def wait_all(self) -> None:
        """The global barrier: every pending read, whether or not this consumer needs it."""
        self.q.join()

    def shutdown(self) -> None:
        for _ in self.workers:
            self.q.put(None)
        for w in self.workers:
            w.join(timeout=5)
