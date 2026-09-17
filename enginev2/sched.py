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
  D2     lease_until_completion  the NVMe ADMISSION PERMIT is held submit -> H2D completion, so a
                                 demand miss can block on the right to start a read while the
                                 device is idle. See the ownership note on LoaderService: this is
                                 the permit, NOT the physical pinned buffer, which is always held
                                 until the H2D completes because cudaMemcpyAsync reads out of it.
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
import os
import queue
import threading
import time

from .leaves import Bandwidth, ModelLeaves
from .observe import NO_CTX, Event, NullObserver, WaitReason, next_span, now_ns
from .store import ExpertSlots, SlotArena, SlotReady, StagingPool


@dataclasses.dataclass(frozen=True, slots=True)
class LoadTicket:
    """One queued expert read. Immutable, because it crosses a thread boundary and is read on the
    other side: nothing may edit a ticket after submit().

    It replaces `(prio, seq, (ctx, cause_id, speculative, scored) + tuple(item), prio == 0)` -- a
    tuple spliced from two sources, indexed positionally in _worker (entry[2], entry[3]), and
    carrying the demand flag twice (entry[3] was `prio == 0`, which is `not speculative`, which is
    already inside entry[2]). Two of those four fields existed only to make the tuple sortable.
    """
    ctx: object
    cause_id: int
    speculative: bool
    scored: bool
    key: tuple
    slot: int
    gen: int

    @property
    def demand(self) -> bool:
        return not self.speculative


class _WorkQueue:
    """Demand reads ahead of speculative ones, FIFO within each class.

    That is exactly what the PriorityQueue expressed, at the cost of a monotonic `_seq` whose only
    job was to break priority ties before Python compared the payloads -- and `_seq` had to be
    generated under the service lock. Two deques carry the same policy with no counter, no tuple
    ordering, and no magic priority for the shutdown sentinel, which now simply goes to the front.

    `outstanding` keeps queue.Queue's unfinished_tasks semantics (put +1, done -1), because the
    invariant tests assert that nothing is left pending after a wait -- and a queue that is empty
    while a worker is still inside a read is not at rest.
    """

    def __init__(self):
        self._cv = threading.Condition(threading.Lock())
        self._demand: collections.deque = collections.deque()
        self._spec: collections.deque = collections.deque()
        self.outstanding = 0

    def put(self, ticket) -> None:
        with self._cv:
            (self._demand if ticket is None or ticket.demand else self._spec).append(ticket)
            self.outstanding += 1
            self._cv.notify()

    def put_front(self, ticket) -> None:
        """Shutdown sentinels only: they must overtake queued work, not wait behind it."""
        with self._cv:
            self._demand.appendleft(ticket)
            self.outstanding += 1
            self._cv.notify()

    def get(self):
        with self._cv:
            while not self._demand and not self._spec:
                self._cv.wait()
            return (self._demand or self._spec).popleft()

    def task_done(self) -> None:
        with self._cv:
            self.outstanding -= 1
            self._cv.notify_all()

    def __len__(self) -> int:
        with self._cv:
            return len(self._demand) + len(self._spec)


class _Nothing:
    """Sentinel for "the queue was empty this pass", distinct from the None shutdown sentinel."""


_NOTHING = _Nothing()


class _NullWrite:
    """The synthetic SlotArena exists to catch modelled read/write violations; the real bytes live
    in RealLeaves' own arena. Each h2d was paying two of its locks for bookkeeping nothing reads."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_NULL_WRITE = _NullWrite()


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
                 expert_read_qd: int = 8, h2d_inflight: int = 8, scale: float = 1.0,
                 fail: set | None = None, observer=None):
        self.obs = observer if observer is not None else NullObserver()
        self.arena = arena
        self.slots = slots
        self.compute = compute
        self.policy = policy
        # The I/O half of the seam. A provider supplies read() and h2d(); the modelled one sleeps,
        # a real one does O_DIRECT into a pinned buffer and a device copy out of it. `bw` is the
        # modelled device and exists only for statistics -- a real provider has none, so every
        # reader of it must tolerate None.
        self.leaves = leaves if leaves is not None else _default_leaves(scale)
        # Same decision the driver makes: when the provider orders slot reuse on the DEVICE, the
        # host-side wait and the synthetic SlotArena's write tracking are verification, not
        # synchronisation. ~67 expert reads per step were each taking a compute-stream wait plus
        # two SlotArena locks for bookkeeping the real arena does not use.
        self._dev_orders = (getattr(self.leaves, "device_orders_slot_reuse", False)
                            and os.environ.get("CHECK_INVARIANTS") != "1")
        self.bw = getattr(self.leaves, "bw", None)
        self.stage = self.leaves.make_staging(staging, observer=self.obs)
        self.ready = SlotReady(observer=self.obs)
        self.scale = scale
        # FOUR SEPARATE RESOURCES. They were conflated: one semaphore named `device_queue_depth`
        # sat around the H2D, so it bounded device copies while every worker could enter the NVMe
        # model unthrottled. Harmless at decode, where the workload sits near 2.4 outstanding reads
        # on its own -- and wrong the moment lookahead issues speculative reads, which is the one
        # feature v2 exists for. Caught in review 2026-09-15.
        #
        #   n_workers          threads; they hide per-read latency, they do not add bandwidth
        #   expert_read_qd     how many EXPERT READS may be admitted at once -- see below
        #   staging            physical pinned buffers
        #   h2d_inflight       how many device copies may run at once
        #
        # THIS IS NOT THE DEVICE'S REQUEST QUEUE DEPTH, and the distinction matters for calibrating
        # a real provider. One expert read is fanned into several aligned O_DIRECT chunk reads on
        # `_read_leased`'s inner read_pool, so N admitted expert reads put substantially more than N
        # requests in the device. It was called nvme_qd, which invited exactly that misreading.
        self.read_qd = threading.Semaphore(expert_read_qd)
        self.h2d_sem = threading.Semaphore(h2d_inflight)
        self._read_qd_n, self._h2d_n = expert_read_qd, h2d_inflight
        # D2 off releases the NVMe permit at handoff so another read can start while this buffer
        # waits on its copy. That only buys anything if there are buffers spare to start into.
        if not policy.lease_until_completion and staging <= expert_read_qd:
            raise ValueError(
                f"lease_until_completion=False needs staging ({staging}) > expert_read_qd "
                f"({expert_read_qd}): releasing the permit early cannot help if every buffer is "
                f"already committed")
        self.fail = fail if fail is not None else set()
        # PRIORITY QUEUE: demand reads (0) ahead of speculation (1). Speculation still takes the
        # same expert-read admissions once it starts -- it is deprioritised, never exempted, because a
        # prefetch for layer L+1 and a demand miss on layer L really do contend for one device.
        # Deferred H2D completions. A provider that returns a device event from h2d() puts the
        # completion here instead of blocking its worker on it.
        self._completions: queue.Queue = queue.Queue()
        self.q = _WorkQueue()
        self._cancelled: set = set()
        # What is still IN the queue. cancel() used to count every request as a cancellation even
        # when the read was already running -- inflating the discard statistics -- and its marker
        # then stayed in _cancelled forever, because only a worker checking BEFORE it starts
        # consumes one. Membership here is the truth, and both sides take _lk, so there is no race
        # between a worker dequeuing and a cancel arriving.
        self._queued: set = set()
        # Wrong speculation that is ALREADY RUNNING cannot be un-read, but it must not stay
        # resident: its slot was taken from the cache and it will never be used. The read is paid
        # for either way -- that is the honest cost of being wrong -- but the residency is not.
        self._discard: set = set()
        # READS THAT HAVE STARTED AND WHOSE BYTES HAVE NOT LANDED. cancel() used
        # SlotReady.is_done() for this, which stopped meaning "the write is over" the moment the
        # device-ordered fast path began publishing readiness at H2D ENQUEUE time. A speculative
        # read could then be classified "finished" while its copy was still running: the driver
        # un-mapped the key and cleared the pending marker, the slot became an ordinary eviction
        # victim, and the next reserve() could hand it to another expert whose H2D is issued on a
        # different worker stream -- write-after-write with nothing ordering the two. That is the
        # torn-slot class the I/O hardening closed, reopened by a change in what "ready" means.
        #
        # So cancellation reads THIS, which is owned by _lk and flipped exactly twice: on when a
        # worker commits to the read, off in _complete_h2d once the bytes are down.
        self._running: set = set()
        # ExpertSlots IS NOT THREAD-SAFE -- lru, free_lru and the policy buckets are plain dicts
        # mutated by the driver. Un-mapping from a worker was tolerable while it only happened on
        # the rare error path; routing every discarded prefetch through it made the race routine and
        # produced KeyError in the victim search. Workers therefore only RECORD what to un-map, and
        # the driver drains it between layers, where all other cache mutation already happens.
        self._forget: list = []
        # Reads that actually STARTED, split by kind. `issued` counts submissions, and a queued
        # cancellation means that submission never became a read -- so using it as the denominator
        # of fetch precision understates the predictor. This is the honest denominator, and it
        # stays correct through failures and any future scheduling change.
        self.started_demand = 0
        self.started_spec = 0
        # Scored speculative reads, counted at the SAME ownership point. The scored bit rides with
        # the work item so the denominator is measured rather than reconstructed from outcomes.
        self.started_spec_scored = 0
        # Total outstanding work, for quiesce(). A helper thread wrapped around q.join() leaked one
        # blocked thread per timeout; a counter with its own condition times out cleanly.
        self._inflight = 0
        self._inflight_cv = threading.Condition(threading.Lock())
        # D3's barrier is over DEMAND reads only. v1's join_pending() has no speculation to wait
        # for, so folding prefetches into it would make the v1 arm wait on work v1 never issues --
        # a confound, not a finding. Speculation is deprioritised and uncounted here by design.
        self._demand = 0
        self._demand_cv = threading.Condition(threading.Lock())
        self.h2d_calls = 0
        self.h2d_s = 0.0
        self._lk = threading.Lock()
        self._stop = threading.Event()
        self.workers = [threading.Thread(target=self._worker, daemon=True,
                                         name=f"loader-{i}") for i in range(n_workers)]
        for w in self.workers:
            w.start()
        self.completer = threading.Thread(target=self._completer, daemon=True, name="loader-done")
        self.completer.start()

    # ------------------------------------------------------------------ the leaf
    def _load_one(self, ctx, cause_id: int, spec: bool, scored: bool, key: tuple, slot: int,
                  gen: int, is_demand: bool = False) -> bool:
        """read -> (handoff) -> compute-order barrier -> H2D -> per-slot event.

        Every step between the lease and the release must be inside the try, or the lease leaks and
        there are only `staging` of them for the life of the process.
        """
        # The PHYSICAL pinned buffer. Held until the H2D has completed, under every policy: a real
        # cudaMemcpyAsync reads out of this buffer until its completion event, so handing it to
        # another reader early corrupts the transfer in flight. The previous revision released it
        # at handoff under D2, which made the v2 arm compare against something unbuildable.
        # QUEUED -> RUNNING IS ONE CRITICAL SECTION, and it must stay that way. Splitting it across
        # two acquisitions leaves a window in which this id is in NEITHER `_queued` nor `_running`,
        # and cancel()'s classifier falls through to "finished": the driver then un-maps the key and
        # clears the pending marker, the slot becomes an ordinary eviction victim, and the next
        # reserve() can hand it to another expert while THIS worker goes on to read and H2D into it.
        # That is the same torn-slot failure the published-vs-landed fix closed, one step earlier in
        # the lifecycle -- and that fix is what introduced it. There must be no externally visible
        # state between QUEUED and RUNNING.
        ident = (slot, gen)
        with self._lk:
            self._queued.discard(ident)              # past the point of cancelling
            cancelled = ident in self._cancelled
            self._cancelled.discard(ident)           # consumed either way: no marker outlives its read
            if not cancelled:
                self._running.add(ident)
        if cancelled:
            # The DRIVER already un-mapped this and cleared its pending mark when it cancelled --
            # synchronously, so the slot was available to the very next reserve() rather than to
            # whenever a low-priority queue entry happened to be dequeued. Nothing to do here but
            # drop the stale entry. Touching ExpertSlots from a worker is what this whole deferral
            # exists to avoid.
            return False
        # The PROVIDER acquires the staging buffer, inside read(), and hands back a StagedExpert
        # that owns the lease -- because the bytes and the lease cannot be separated: a real reader
        # returns views ALIASING that pinned buffer. The loader decides only WHEN to release it,
        # which is after h2d has completed, never before.
        staged = None
        permit_held = False
        deferred = False
        try:
            if key in self.fail:
                raise IOError(f"injected NVMe failure {key}")
            if self.obs.enabled and not self.read_qd.acquire(blocking=False):
                sp = next_span()
                self.obs.safe_emit(Event(now_ns(), "wait_start", ctx=ctx, key=key, slot=slot, gen=gen, span=sp,
                                         scored=scored, aux=WaitReason.NVME_ADMISSION))
                self.read_qd.acquire()
                self.obs.safe_emit(Event(now_ns(), "wait_end", ctx=ctx, key=key, slot=slot, gen=gen, span=sp,
                                         scored=scored, aux=WaitReason.NVME_ADMISSION))
            elif not self.obs.enabled:
                self.read_qd.acquire()
            permit_held = True
            # The FRAMEWORK brackets the provider's call. A provider performs the operation; only
            # the caller knows when it was queued, began and ended, so measurement semantics stay
            # put when the provider is swapped.
            if self.obs.enabled:
                self.obs.safe_emit(Event(now_ns(), "nvme_start", ctx=ctx, cause_id=cause_id, key=key, slot=slot, gen=gen,
                                         scored=scored))
            # Counted HERE, not at dequeue: a worker blocked on admission has not started a read,
            # and an injected failure never reaches the device at all. The counter is the
            # denominator of fetch precision, so "committed to read" is not good enough.
            with self._lk:
                if spec:
                    self.started_spec += 1
                    if scored:
                        self.started_spec_scored += 1
                else:
                    self.started_demand += 1
            staged = self.leaves.read(key, self.stage, ctx, scored)
            if self.obs.enabled:
                self.obs.safe_emit(Event(now_ns(), "nvme_end", ctx=ctx, cause_id=cause_id, key=key, slot=slot, gen=gen,
                                         scored=scored))
            if not self.policy.lease_until_completion:
                # D2 OFF: give the ADMISSION back now. Another read may enter the device while this
                # expert's buffer waits on its copy. The buffer itself stays ours until H2D done.
                self.read_qd.release()
                permit_held = False
            # D1 ON is a COMPUTE barrier; D3's wait_all is the global READ barrier. Tagging both
            # GLOBAL_BARRIER made one wait appear under another's name -- the double-counting the
            # ownership-point rule exists to prevent.
            if self._dev_orders:
                # The copy stream already waits on this slot's last reader event inside h2d.
                pass
            reason = (WaitReason.COMPUTE_BARRIER if self.policy.compute_barrier_global
                      else WaitReason.SLOT_READER)
            sp = next_span()
            if self.obs.enabled:
                self.obs.safe_emit(Event(now_ns(), "wait_start", ctx=ctx, key=key, slot=slot,
                                         gen=gen, span=sp, aux=reason, scored=scored))
            try:
                if self._dev_orders:
                    pass                               # the device orders it; see _dev_orders
                elif self.policy.compute_barrier_global:
                    self.compute.wait_idle()          # D1 ON: wait for ALL compute
                else:
                    self.compute.wait_slot_free(slot)  # D1 OFF: only this slot's previous reader
            finally:
                if self.obs.enabled:
                    self.obs.safe_emit(Event(now_ns(), "wait_end", ctx=ctx, key=key, slot=slot,
                                             gen=gen, span=sp, aux=reason, scored=scored))
            if self.obs.enabled and not self.h2d_sem.acquire(blocking=False):
                sp = next_span()
                self.obs.safe_emit(Event(now_ns(), "wait_start", ctx=ctx, key=key, slot=slot, gen=gen, span=sp,
                                         scored=scored, aux=WaitReason.H2D_CAPACITY))
                self.h2d_sem.acquire()
                self.obs.safe_emit(Event(now_ns(), "wait_end", ctx=ctx, key=key, slot=slot, gen=gen, span=sp,
                                         scored=scored, aux=WaitReason.H2D_CAPACITY))
            elif not self.obs.enabled:
                self.h2d_sem.acquire()
            # ISSUE the copy. A provider that completes synchronously returns None and everything
            # below runs on this thread, exactly as before. One that returns a HANDLE has only
            # ENQUEUED the copy: the worker must not sit on it, because waiting for its own copy is
            # a buffer-lifetime dependency, not a compute dependency, and it keeps a thread that
            # could be issuing the next NVMe read parked on a device event instead.
            wctx = _NULL_WRITE if self._dev_orders else self.arena.writing(slot, key)
            wctx.__enter__()
            if self.obs.enabled:
                self.obs.safe_emit(Event(now_ns(), "h2d_start", ctx=ctx,
                                         cause_id=cause_id, key=key, slot=slot, gen=gen,
                                         scored=scored))
            t0 = time.perf_counter()
            try:
                handle = self.leaves.h2d(slot, key, staged)
            except BaseException:
                wctx.__exit__(None, None, None)
                self.h2d_sem.release()
                raise
            if handle is None:
                self._complete_h2d(None, wctx, staged, key, slot, gen, ctx, cause_id, scored, t0)
                staged = None                      # the completion owns the lease now
            else:
                deferred = True
                if self._dev_orders:
                    # EARLY READINESS. The copy is enqueued and its event published, so the driver
                    # may reserve and queue the graph now; leaves.await_copies() puts the event on
                    # the compute stream before that graph reads the slot, and the DEVICE enforces
                    # completion. This takes a host block off the front of every layer -- the old
                    # path was worker -> completer -> synchronize -> ready.set -> driver wakes,
                    # about three thread handoffs per expert, all to learn something CUDA already
                    # knows. Only for a provider that declares device_orders_slot_reuse, because
                    # only that provider implements await_copies.
                    self.ready.set(slot, gen)
                self._completions.put(
                    (handle, wctx, staged, key, slot, gen, ctx, cause_id, scored, t0, is_demand))
                staged = None
        except BaseException as exc:              # noqa: BLE001
            # A failed expert must be UN-MAPPED: its slot holds a torn read, and leaving the key
            # mapped would make the next reserve() count it as a HIT and compute with partial
            # bytes -- silent, and permanent for the life of the process. Deferred to the driver
            # for the same reason as above; drain_forgets() runs before the next reserve().
            with self._lk:
                # Leave _running here too, or a read that died before its H2D stays "in flight"
                # forever and every later cancel() of that slot is misclassified as running.
                self._running.discard((slot, gen))
                self._discard.discard((slot, gen))
                self._forget.append((key, slot, gen, False))
            self.ready.set(slot, gen, err=exc)
            self.ready.set_landed(slot, gen)     # nothing is writing this slot any more
        finally:
            if not deferred:
                if permit_held:
                    self.read_qd.release()
                # The write is over (done or failed): the slot may be evicted again.
                self.slots.clear_pending(slot, gen)
                if staged is not None:
                    staged.release()      # after the copy, always, and exactly once
            elif permit_held and not self.policy.lease_until_completion:
                pass                      # already released at handoff
            # A DEFERRED copy owns all three: the completer releases the permit, clears the pending
            # mark and returns the buffer when the device says the bytes have landed. Doing any of
            # it here would hand the slot or the buffer to someone else mid-copy.
        return deferred

    def _complete_h2d(self, handle, wctx, staged, key, slot, gen, ctx, cause_id, scored, t0,
                      is_demand: bool = False):
        """Everything that may only happen once the bytes have LANDED.

        Split out of _load_one so the inline path (a provider whose h2d has already completed on
        return) and the deferred path (one that handed back a device event) run the same code and
        cannot drift apart. `handle` is None for the inline path; otherwise it is waited on here,
        on the completer thread, never on a worker.
        """
        try:
            if handle is not None:
                handle.synchronize()
            dt = time.perf_counter() - t0
            if self.obs.enabled:
                self.obs.safe_emit(Event(now_ns(), "h2d_end", ctx=ctx, cause_id=cause_id,
                                         key=key, slot=slot, gen=gen, value=dt, scored=scored))
            with self._lk:
                self.h2d_calls += 1
                self.h2d_s += dt
            if not (handle is not None and self._dev_orders):
                self.ready.set(slot, gen)     # already published at enqueue on the fast path
            self.ready.set_landed(slot, gen)  # the bytes are down: the slot may be recycled
            # ONE critical section, because these two facts must change together. Leaving _running
            # and reading _discard separately leaves a window in which cancel() sees neither the
            # running mark nor a landed slot, calls the read "finished", and un-maps a slot whose
            # copy this thread has not yet finished accounting for.
            with self._lk:
                self._running.discard((slot, gen))
                drop = (slot, gen) in self._discard
                self._discard.discard((slot, gen))
                if drop:
                    # Was this speculation discarded while it was in flight? The read is paid for;
                    # refuse it the cache slot. Nothing ever waits on a speculative read, so
                    # un-mapping after completion cannot strand a consumer.
                    self._forget.append((key, slot, gen, False))
        except BaseException as exc:              # noqa: BLE001
            with self._lk:
                self._running.discard((slot, gen))   # or a failed copy is "running" forever
                self._discard.discard((slot, gen))
                self._forget.append((key, slot, gen, False))
            self.ready.set(slot, gen, err=exc)
            self.ready.set_landed(slot, gen)         # nothing is writing this slot any more
        finally:
            wctx.__exit__(None, None, None)
            self.h2d_sem.release()
            if self.policy.lease_until_completion:
                self.read_qd.release()
            self.slots.clear_pending(slot, gen)
            if staged is not None:
                staged.release()
            if handle is not None:
                # DEFERRED: the worker already went back to the queue, so the two counters that say
                # "this read is still happening" were NOT decremented there. They belong here --
                # `_inflight` is what quiesce() waits on and `_demand` is D3's demand barrier, and
                # dropping either while the copy is still in flight reports the loader at rest with
                # bytes still landing. That is exactly the class of bug the exact-logits gate
                # catches only intermittently, so it is fixed by construction instead.
                if is_demand:
                    with self._demand_cv:
                        self._demand -= 1
                        self._demand_cv.notify_all()
                with self._inflight_cv:
                    self._inflight -= 1
                    self._inflight_cv.notify_all()

    # How long the completer sleeps when copies are outstanding but none has finished. Copies are
    # milliseconds, so this retires within a fraction of one and costs a few thousand wakeups per
    # second at most. It is a queue.get() timeout, not a sleep, so a new submission wakes it early.
    _COMPLETER_POLL_S = 2e-4
    _completer_in_order = os.environ.get("DSV41_COMPLETER_INORDER") == "1"

    def _completer(self) -> None:
        """Retire copies in the order they FINISH, not the order they were issued.

        The previous version called handle.synchronize() on each item in submission order. The
        copies run on per-worker streams and finish out of order, so if copy #1 was slow while #2
        was already done, the completer sat on #1 and released neither #2's staging buffer nor its
        H2D permit. With h2d_inflight=2 that can halve the effective copy depth -- a scheduling
        bubble, not Python overhead, which is why it is worth fixing even though the lock-shaving
        A/B (job 370) came back a null.

        The first attempt at this fix reproduced the same defect one step along: it took ONE item,
        found it unfinished, and parked in synchronize() on it -- without looking at the queue,
        where a finished copy was already waiting. So the rule is: take everything the queue has
        before deciding anything, and never block on a specific event. `pending` is bounded by
        h2d_inflight, so the scan is over two or three items.
        """
        pending: list = []
        while True:
            try:
                item = self._completions.get(
                    timeout=None if not pending else self._COMPLETER_POLL_S)
            except queue.Empty:
                item = _NOTHING
            shutdown = False
            while item is not _NOTHING:                 # drain the burst, do not sip
                if item is None:
                    shutdown = True
                    self._completions.task_done()
                else:
                    pending.append(item)
                try:
                    item = self._completions.get_nowait()
                except queue.Empty:
                    break
            if self._completer_in_order and pending:
                # MEASUREMENT ARM ONLY (DSV41_COMPLETER_INORDER=1): the pre-2026-09-16 behaviour,
                # kept so the out-of-order change has a counterfactual rather than a before/after
                # across two builds. Retire strictly in submission order, blocking on each.
                it = pending.pop(0)
                if it[0] is not None:
                    it[0].synchronize()
                try:
                    self._complete_h2d(*it)
                finally:
                    self._completions.task_done()
            for it in [it for it in pending if it[0] is None or it[0].query()]:
                pending.remove(it)
                try:
                    self._complete_h2d(*it)
                finally:
                    self._completions.task_done()
            if shutdown:
                for it in pending:                      # nothing may be dropped on the way out:
                    if it[0] is not None:               # _complete_h2d releases the staging buffer
                        it[0].synchronize()             # and the H2D permit
                    self._complete_h2d(*it)
                    self._completions.task_done()
                return

    def _worker(self) -> None:
        while True:
            t = self.q.get()
            try:
                if t is None:
                    return
                deferred = False
                try:
                    deferred = bool(self._load_one(
                        t.ctx, t.cause_id, t.speculative, t.scored, t.key, t.slot, t.gen,
                        is_demand=t.demand))
                finally:
                    if t.demand and not deferred:
                        with self._demand_cv:
                            self._demand -= 1
                            self._demand_cv.notify_all()
                if not deferred:
                    with self._inflight_cv:
                        self._inflight -= 1
                        self._inflight_cv.notify_all()
            finally:
                self.q.task_done()

    # ------------------------------------------------------------------ service api
    def submit(self, to_load, speculative: bool = False, ctx=NO_CTX, cause_id: int = 0,
               scored: bool = True) -> None:
        # Protect before queueing, never after: between the two a worker can already be writing.
        self.slots.mark_pending(to_load)
        for key, slot, gen in to_load:
            self.ready.arm(slot, gen, ctx, cause_id, scored)
        if self.obs.enabled:
            for key, slot, gen in to_load:
                self.obs.safe_emit(Event(now_ns(), "load_queued", ctx=ctx, key=key, slot=slot,
                                         gen=gen, cause_id=cause_id, scored=scored,
                                         aux="spec" if speculative else "demand"))
        with self._inflight_cv:
            self._inflight += len(to_load)
        if not speculative:
            with self._demand_cv:
                self._demand += len(to_load)
        with self._lk:
            for key, slot, gen in to_load:
                self._queued.add((slot, gen))
                # the context travels WITH the work: a completion on a worker thread must still
                # know which request and step asked for it.
                self.q.put(LoadTicket(ctx, cause_id, speculative, scored, key, slot, gen))

    def cancel(self, items) -> list:
        """Discard wrong speculation in whichever of its three states it is in.

        An earlier version only cancelled QUEUED work and let a running or finished wrong prefetch
        stay mapped in the LRU -- which contradicts the policy it implements and quietly hands the
        predictor arms cache residency they did not earn. The three states:

          queued    -> mark the stale entry AND queue the un-map now: the driver recovers the slot
                       on its next drain, not whenever a low-priority entry is dequeued
          running   -> cannot be un-read, so mark it: un-mapped when the H2D ends
          finished  -> un-map now

        WHY QUEUED IS NOT "THE WORKER WILL HANDLE IT". Speculation is deliberately lower priority,
        so a cancelled wrong prediction could sit in the queue holding a mapped slot and a pending
        mark for an unbounded time -- protected from eviction exactly while demand needed it. The
        driver therefore recovers it immediately and the queue entry becomes inert.

        The read cost of a running one is still paid in full. Only the residency is refused.
        -> [(key, slot, gen, state)] with state in {"queued", "running", "finished"}.

        PER ITEM, not three totals: the caller may need to account only a SUBSET of what it
        cancels. Statistics and physical behaviour must be separable -- an attempt that is not
        being scored still has to be discarded, or the experiment quietly changes the policy it is
        measuring.
        """
        out = []
        with self._lk:
            for key, slot, gen in items:
                if (slot, gen) in self._queued:
                    self._cancelled.add((slot, gen))
                    self._queued.discard((slot, gen))
                    # rollback: the read never started, so the displaced tenant's bytes are intact
                    self._forget.append((key, slot, gen, True))
                    out.append((key, slot, gen, "queued"))
                elif (slot, gen) in self._running:
                    # IN FLIGHT -- the read or its copy is still happening. Collected at completion
                    # instead, because un-mapping now would release the slot while it is being
                    # written. NOT `ready.is_done`: that means PUBLISHED since the device-ordered
                    # fast path, and a published copy is still moving bytes.
                    self._discard.add((slot, gen))
                    out.append((key, slot, gen, "running"))
                else:
                    self._forget.append((key, slot, gen, False))   # landed; the slot WAS overwritten
                    out.append((key, slot, gen, "finished"))
        return out

    def quiesce(self, timeout: float = 60.0) -> None:
        """Wait until nothing is queued or in flight, or raise.

        Speculative reads are NOT counted by wait_all() -- that is D3's demand barrier -- so
        anything inspecting the cache at rest needs this, and so does an orderly shutdown. The
        timeout is honoured: an earlier version accepted one and then called an unbounded join(),
        which is worse than not offering it.
        """
        with self._inflight_cv:
            if not self._inflight_cv.wait_for(lambda: self._inflight == 0, timeout):
                raise TimeoutError(
                    f"loader did not quiesce within {timeout}s ({self._inflight} outstanding)")

    def drain_forgets(self) -> int:
        """Apply pending un-maps. MUST be called from the driver thread, before reserve()."""
        with self._lk:
            pending, self._forget = self._forget, []
        for key, slot, gen, rollback in pending:
            if rollback:
                self.slots.rollback_speculative(key, slot, gen)
            else:
                self.slots.forget(key, slot)
            self.slots.clear_pending(slot, gen)
        return len(pending)

    def wait_slots(self, to_load, ctx=NO_CTX, scored: bool = True) -> None:
        """Wait only for the slots THIS consumer needs. Errors are drained, then the first re-raised."""
        err = None
        for key, slot, gen in to_load:
            try:
                self.ready.wait(slot, gen, ctx=ctx, key=key, scored=scored)
            except BaseException as e:            # noqa: BLE001
                if err is None:
                    err = e
        if err is not None:
            raise err

    def wait_all(self, timeout: float = 60.0, ctx=NO_CTX, scored: bool = True) -> None:
        """The global barrier: every pending DEMAND read, whether or not this consumer needs it.

        Instrumented here because this is where D3 actually costs something. It was invisible, and
        its absence made the wait table read as if nothing blocked at all -- every later per-slot
        wait found its data already there, because this had drained the queue first.
        """
        with self._demand_cv:
            if self._demand == 0:
                return
            sp = next_span()
            if self.obs.enabled:
                self.obs.safe_emit(Event(now_ns(), "wait_start", ctx=ctx, span=sp, scored=scored,
                                         value=self._demand, aux=WaitReason.GLOBAL_BARRIER))
            try:
                if not self._demand_cv.wait_for(lambda: self._demand == 0, timeout):
                    raise TimeoutError("global barrier never drained")
            finally:
                if self.obs.enabled:
                    self.obs.safe_emit(Event(now_ns(), "wait_end", ctx=ctx, span=sp, scored=scored,
                                             aux=WaitReason.GLOBAL_BARRIER))

    def shutdown(self, drain: bool = True, timeout: float = 60.0) -> None:
        """Orderly by default: finish what is queued, apply pending un-maps, THEN stop the workers.

        The stop sentinels are priority -1, so they outrank demand (0) and speculation (1) -- a
        shutdown that just posts them lets workers exit with work still queued. Harmless while the
        leaves were sleeps; not harmless once a worker owns a pinned buffer and an in-flight CUDA
        copy. Pass drain=False to abort instead: queued work is cancelled first, running work is
        still allowed to finish, and only then do the workers stop.
        """
        if drain:
            # PROPAGATES. Falling through to the sentinels on timeout reverted to exactly the unsafe
            # behaviour this method exists to remove -- priority -1 outranks the unfinished queue --
            # and with real pinned buffers and in-flight CUDA copies, returning from a failed drain
            # is not a good failure mode. Use drain=False to abort deliberately.
            self.quiesce(timeout)
            self.drain_forgets()
        else:
            with self._lk:
                self._cancelled.update(self._queued)
                self._queued.clear()
            try:
                self.quiesce(timeout)
            except TimeoutError:
                pass
        for _ in self.workers:
            self.q.put_front(None)
        for w in self.workers:
            w.join(timeout=5)
        # The completer stops LAST. quiesce() above already waited for every completion (that is
        # what moving the _inflight decrement into _complete_h2d bought), but a worker can enqueue
        # one right up to its sentinel, so the sentinel for this thread goes in after theirs.
        self._completions.put(None)
        self.completer.join(timeout=5)
