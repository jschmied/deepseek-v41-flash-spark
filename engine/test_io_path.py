"""test_io_path.py -- the expert loader's I/O and async machinery, pinned by invariant.

`engine/test_expert_store.py` stubs `_load_into_slot` wholesale, so it proves things about the
slot BOOKKEEPING and nothing at all about the machinery underneath it: the two thread pools, the
pinned-buffer leases, the exception paths, or what any of that does when a consumer is reading the
arena at the same time. That machinery has already produced one silent corruption (the per-call
`used` set, fixed with `_pending_slots`), it is the single largest remaining decode lever
(async issue + compute overlap removes 49-70 % of modelled decode blocking wait against 9-23 % for
a PERFECT oracle on top of it, and we move 2.78 of a measured 6.8 GB/s), and the next round of work
is going to make it do MORE, on the decode path, which is the one shape nothing here covers.

So this file drives the REAL `_read_leased`: the real leases, the real `pool`/`read_pool` split,
the real chunk arithmetic, the real `resolve`/`join_pending` state machine. Only two things are
faked, and neither is on the async path:

  * `_shard` returns a stub with an fd that ENCODES (layer, expert) and a hand-built two-run
    layout, so no NVMe and no checkpoint is needed (the on-disk reader's byte-exactness is
    test_expert_io.py's job, against safetensors, on the box).
  * `_pread_chunk` fills the pinned staging buffer with a 4-byte tag of that fd's key instead of
    reading. Every view handed to the sink is checked against the tag, so a staging buffer handed
    to two experts at once shows up as wrong bytes and not as a silent pass.

The H2D is modelled by `SlotArena`, which knows who is writing a slot and who is reading it and
records every overlap -- that is the "model the consumer explicitly" part, and it is what makes
"no torn slot" an assertion rather than a hope.

Run:  python -m engine.test_io_path            (all)
      python -m engine.test_io_path lease      (substring filter, for mutation checking)

Needs a CUDA context for the pinned staging buffers (io_threads x 17.7 MB, so every store here is
built with 4-8 io threads and closed as soon as its test ends). It does NOT load the model, does
not touch MODEL_DIR, does not open the GPU arena, and runs in seconds.
"""

from __future__ import annotations

import contextlib
import os
import struct
import sys
import tempfile
import threading
import time

# Import-time env hygiene. DSV41_IO_THREADS is the one that would silently WIN over the constructor
# argument (`io_threads = int(os.environ.get("DSV41_IO_THREADS", io_threads))`), and a store built
# with 48 io threads here would allocate 850 MB of pinned memory per test on a box that is usually
# holding a job. The rest would reroute a test into a real cache or a real log file.
for _v in ("DSV41_IO_THREADS", "DSV41_CB3_CACHE", "DSV41_ROUTE_LOG", "DSV41_ROUTE_SYNC",
           "DSV41_EVICT_POLICY", "DSV41_UNSAFE_NO_COMPUTE_WAIT"):
    os.environ.pop(_v, None)

import torch

from engine import experts as E

ALIGN = E.ALIGN

# ---------------------------------------------------------------- the fake NVMe side

_FD_KEY: dict[int, tuple] = {}      # stub fd -> (layer, expert); the fd is how _pread_chunk knows
_FAIL_READ: set[tuple] = set()      # keys whose read must raise (injected NVMe failure)
_GATE: threading.Event | None = None  # when set-able, reads block on it: freezes loads mid-flight
_REAL_PREAD = E._pread_chunk


def _tag(key: tuple) -> bytes:
    """4 bytes identifying (layer, expert). Four and not one because a single byte would collide
    once a test uses more than 255 distinct experts, and the decode-shaped test uses ~360."""
    return struct.pack("<HH", key[0], key[1])


def _fake_pread(fd: int, view: memoryview, off: int, need: int) -> None:
    """Stand-in for the O_DIRECT read: stamp the key's tag instead of fetching bytes.

    Every run start and every member offset below is a multiple of 4 and every chunk offset is a
    multiple of ALIGN, so the tag's phase is the same in every chunk of a run and a member view can
    be checked at its head and its tail.
    """
    key = _FD_KEY[fd]
    if _GATE is not None:
        _GATE.wait(30)
    if key in _FAIL_READ:
        raise IOError(f"injected NVMe failure reading {key} at {off}")
    t = _tag(key)
    view[:need] = (t * (need // 4 + 1))[:need]


class StubShard:
    """The two attributes `_read_leased` touches on a ShardFile: `fd` and `expert_runs`.

    The layout is the real shape -- one small scale run and one large weight run, both starting
    unaligned so `alo`/`ahi` and the past-EOF tail are exercised -- at 1/32 of the real size, which
    is what keeps a 400-expert test in single-digit seconds.
    """

    SCALE_AT, SCALE_N = 8192 + 4, 4096          # 3 x 4 KiB scales, unaligned start
    WEIGHT_AT, WEIGHT_N = 1048576 + 8, 65536    # 3 x 64 KiB weights, unaligned start

    def __init__(self, fd: int):
        self.fd = fd
        a = self.SCALE_AT
        scales = (a, a + 3 * self.SCALE_N,
                  [(1, 0, self.SCALE_N), (3, self.SCALE_N, self.SCALE_N),
                   (5, 2 * self.SCALE_N, self.SCALE_N)])
        b = self.WEIGHT_AT
        weights = (b, b + 3 * self.WEIGHT_N,
                   [(0, 0, self.WEIGHT_N), (2, self.WEIGHT_N, self.WEIGHT_N),
                    (4, 2 * self.WEIGHT_N, self.WEIGHT_N)])
        self._runs = [scales, weights]

    def expert_runs(self, prefix: str):
        return self._runs


# ---------------------------------------------------------------- the arena / consumer model

class SlotArena:
    """The GPU arena as the host can see it: who is writing a slot, who is reading it, what it holds.

    Every overlap is RECORDED rather than raised, because an assertion inside a pool worker only
    reaches the test as a future's exception at join time and loses which two parties collided.
    Each test asserts `violations == []` itself -- except the one that asserts the opposite.
    """

    def __init__(self, slots: int):
        self.slots = slots
        self.device = "cpu"
        self._lk = threading.Lock()
        self.content: dict[int, tuple] = {}      # slot -> key, as the writes actually land
        self._writing: dict[int, tuple] = {}
        self._reading: dict[int, int] = {}
        self.violations: list[tuple] = []

    @contextlib.contextmanager
    def writing(self, slot: int, key: tuple):
        with self._lk:
            other = self._writing.get(slot)
            if other is not None:
                self.violations.append(("write-write", slot, key, other))
            if self._reading.get(slot):
                self.violations.append(("write-during-read", slot, key))
            self._writing[slot] = key
        try:
            yield
        finally:
            with self._lk:
                self._writing.pop(slot, None)
                self.content[slot] = key

    @contextlib.contextmanager
    def reading(self, slot: int):
        with self._lk:
            w = self._writing.get(slot)
            if w is not None:
                self.violations.append(("read-during-write", slot, w))
            self._reading[slot] = self._reading.get(slot, 0) + 1
        try:
            yield self.content.get(slot)
        finally:
            with self._lk:
                self._reading[slot] -= 1


def consume(arena: SlotArena, slots, hold: float = 0.0):
    """The MoE kernel reading its experts out of the arena. Returns what each slot held."""
    out = []
    for s in slots:
        with arena.reading(s) as content:
            if hold:
                time.sleep(hold)
            out.append(content)
    return out


# ---------------------------------------------------------------- store construction

_OPEN: list = []


def make_store(transient_slots: int = 16, lru_slots: int = 8, io_threads: int = 4,
               read_threads: int = 3, read_chunk_mb: float = 0.0625,
               evict_policy: str | None = None, jitter: float = 0.006, seed: int = 1):
    """A store whose reads are faked at the fd and whose H2D is faked at the arena. Everything
    between those two points -- leases, both pools, chunking, resolve/join state -- is the real code.

    read_chunk_mb defaults to 64 KiB so the 192 KiB weight run becomes 3 pieces on `read_pool` plus
    the one the io worker runs inline: without a split there is nothing for the second pool to do
    and the deadlock invariant would be vacuous.
    """
    d = tempfile.mkdtemp(prefix="dsv41-io-path-")
    arena = SlotArena(transient_slots + lru_slots)
    st = E.ExpertStore(d, {"weight_map": {}}, arena, n_layers=64,
                       transient_slots=transient_slots, io_threads=io_threads,
                       read_threads=read_threads, read_chunk_mb=read_chunk_mb,
                       evict_policy=evict_policy)
    shards: dict[tuple, StubShard] = {}
    lk = threading.Lock()

    def _shard(name: str):
        p = name.split(".")
        key = (int(p[1]), int(p[4]))
        with lk:
            sh = shards.get(key)
            if sh is None:
                fd = 1000 + len(_FD_KEY)
                _FD_KEY[fd] = key
                sh = shards[key] = StubShard(fd)
        return sh

    st._shard = _shard
    st.test_arena = arena
    st.test_submitted: list = []
    st.test_lock = threading.Lock()

    def load(key: tuple, slot: int, prefix: str | None = None):
        with st.test_lock:
            st.test_submitted.append((key, slot))

        def sink(views):
            want = _tag(key)
            for i, v in enumerate(views):
                head = v[:4].numpy().tobytes()
                tail = v[-4:].numpy().tobytes()
                if head != want or tail != want:
                    arena.violations.append(("staging-mixup", slot, key, i, head, tail))
            # the H2D window: the slot is being written for exactly as long as this block runs.
            # Deterministic per key, so completion order is jittered but reproducible -- a shared
            # random.Random across io threads would not be.
            with arena.writing(slot, key):
                time.sleep(((key[0] * 7 + key[1] * 13 + seed) % 11) / 11.0 * jitter)
            return slot

        return st._read_leased(key[0], key[1], prefix, sink)

    st._load_into_slot = load
    _OPEN.append(st)
    return st, arena


def close_all():
    """Shut both pools of every store this test built and drop its pinned buffers."""
    global _GATE
    _GATE = None
    _FAIL_READ.clear()
    while _OPEN:
        st = _OPEN.pop()
        st.pool.shutdown(wait=True)
        if isinstance(st.read_pool, _OnePool):
            st.read_pool = st.read_pool.pool
        st.read_pool.shutdown(wait=True)
        st.stage_mv = []
        st.stage = []


def ids(*experts):
    return torch.tensor([list(experts)], dtype=torch.int32)


def leases(st):
    """(free list length, semaphore value, distinct entries). All three are io_threads at rest."""
    free = list(st.stage_free)
    return len(free), st.stage_sem._value, len(set(free))


def assert_leases_at_rest(st, where: str):
    n, sem, distinct = leases(st)
    assert n == st.io_threads, f"{where}: {st.io_threads - n} pinned buffer(s) LEAKED ({n} free)"
    assert sem == st.io_threads, f"{where}: semaphore says {sem}, io_threads is {st.io_threads}"
    assert distinct == n, f"{where}: a buffer was released twice -- free list {st.stage_free}"


class _OnePool:
    """`read_pool` pointed back at `pool`: the arrangement the two-pool comment says deadlocks.

    Records the piece-jobs so the test can cancel them and unwedge the process afterwards -- the
    real code's `f.result()` has no timeout, so without the cancel the io workers would sit there
    forever and ThreadPoolExecutor's atexit handler would hang the interpreter on the way out.
    """

    def __init__(self, pool):
        self.pool = pool
        self.futs: list = []
        self.lk = threading.Lock()

    def submit(self, fn, *a):
        f = self.pool.submit(fn, *a)
        with self.lk:
            self.futs.append(f)
        return f

    def cancel_all(self) -> int:
        with self.lk:
            fs = list(self.futs)
        return sum(1 for f in fs if f.cancel())


def prespawn(pool, n: int) -> None:
    """Force `pool` to create all n of its worker threads before the experiment starts.

    ThreadPoolExecutor spawns lazily -- one thread per submit, while `len(_threads) < max_workers`.
    On a COLD pool that hides the one-pool deadlock completely: the io task's own piece-submit is
    what spawns the next worker, and that worker then runs the piece. The test was 50/50 on this
    until it pre-spawned. A serving process has had all 48 io threads for hours by the time any of
    this matters, which is the state modelled here -- and the state in which the deadlock is
    permanent rather than lucky.
    """
    b = threading.Barrier(n + 1)
    for _ in range(n):
        pool.submit(b.wait)
    b.wait(10)


def run_with_timeout(fn, timeout: float):
    """(finished, error). Runs fn on a daemon thread; the caller decides what a timeout means."""
    box: dict = {}

    def body():
        try:
            box["r"] = fn()
        except BaseException as e:              # noqa: BLE001 -- the caller inspects it
            box["e"] = e

    t = threading.Thread(target=body, daemon=True)
    t.start()
    t.join(timeout)
    return (not t.is_alive()), box.get("e"), t


# ================================================================ 1. no torn slot

def test_no_torn_slot_under_deferred_submission():
    """No slot's bytes are read by a consumer while a write to it is in flight, and a window that
    cannot be served without recycling a live slot REFUSES instead of overlapping two writes."""
    # (a) within capacity: four chunks of one layer, deferred, jittered completion.
    st, arena = make_store(transient_slots=12, lru_slots=8)
    expect: dict[int, tuple] = {}
    for chunk in [(3, 7, 11), (7, 4, 19), (19, 23, 3), (5, 11, 31)]:
        sl = st.resolve(0, ids(*chunk), prefill=True, defer=True).tolist()[0]
        for e, s in zip(chunk, sl):
            expect[s] = (0, e)
    pending_slots = [s for _, _, s in st._pending]
    assert len(set(pending_slots)) == len(pending_slots), \
        f"two in-flight loads share a slot: {sorted(pending_slots)}"
    st.join_pending()
    got = consume(arena, sorted(expect))
    assert arena.violations == [], arena.violations
    assert got == [expect[s] for s in sorted(expect)], (got, expect)
    assert_leases_at_rest(st, "after a deferred layer")
    n_a = len(expect)

    # (b) over capacity, with every read frozen so nothing can retire: the ring cannot serve the
    # second resolve without taking a slot an in-flight read owns. It must raise there and then.
    global _GATE
    _GATE = threading.Event()
    st2, arena2 = make_store(transient_slots=8, lru_slots=8)
    st2.resolve(0, ids(1, 2, 3, 4, 5, 6), prefill=True, defer=True)
    raised = None
    try:
        st2.resolve(0, ids(10, 11, 12, 13, 14, 15), prefill=True, defer=True)
    except RuntimeError as e:
        raised = e
    live = [s for _, _, s in st2._pending]
    assert len(set(live)) == len(live), \
        f"a slot with a read in flight was handed to a second expert: {sorted(live)}"
    assert raised is not None and "transient ring exhausted" in str(raised), raised
    _GATE.set()
    st2.join_pending()
    assert arena2.violations == [], arena2.violations
    assert_leases_at_rest(st2, "after the refused resolve")
    print(f"  no torn slot: {n_a} slots written under jitter, 0 overlaps; "
          f"ring exhaustion refuses instead of recycling  OK")


def test_no_torn_slot_decode_lru_eviction():
    """The LRU half of the arena must protect in-flight slots too, under BOTH evict policies.

    Decode misses go to `_lru_slot_for`, not to the transient ring, and a deferred decode resolve
    puts its key in self.lru the moment the slot is assigned -- 13.8 MB before the data is there.
    Nothing in production defers on the decode path TODAY, which is precisely why this has to be
    pinned before the next change does: at 5,328 LRU slots a fresh entry is MRU under "lru" and
    scores age 0 (the minimum) under age_over_freq, so the hole is invisible exactly the way the
    transient-ring one was at TRANSIENT_SLOTS=400 against 362 experts.
    """
    global _GATE
    for policy in ("lru", "age_over_freq"):
        _GATE = threading.Event()
        st, arena = make_store(transient_slots=8, lru_slots=4, evict_policy=policy)
        st.resolve(0, ids(1, 2), prefill=False, defer=True)
        st.resolve(0, ids(3, 4), prefill=False, defer=True)
        raised = None
        try:
            st.resolve(0, ids(5, 6), prefill=False, defer=True)
        except RuntimeError as e:
            raised = e
        live = [s for _, _, s in st._pending]
        assert len(set(live)) == len(live), \
            f"[{policy}] eviction gave away a slot whose read is in flight: {sorted(live)}"
        assert raised is not None and "LRU exhausted" in str(raised), (policy, raised)
        _GATE.set()
        st.join_pending()
        assert arena.violations == [], (policy, arena.violations)
        assert_leases_at_rest(st, f"[{policy}] after the refused decode resolve")
        _GATE = None
    print("  no torn slot: LRU eviction refuses an in-flight slot under lru and age_over_freq  OK")


# ================================================================ 2. lease conservation

def test_lease_conservation_normal_and_under_failure():
    """`stage_free` and `stage_sem` come back to io_threads after every path, including exceptions.

    Note the ONLY invariant that holds at every instant is the weaker one asserted by the sampler:
    `_lease` acquires the semaphore and THEN pops the free list, so between those two statements
    the list is one longer than the semaphore. Exact agreement is an at-rest property.
    """
    st, arena = make_store(transient_slots=24, lru_slots=8, io_threads=4)
    assert_leases_at_rest(st, "fresh store")

    stop = threading.Event()
    bad: list = []

    def sampler():
        while not stop.is_set():
            free = list(st.stage_free)
            if len(free) > st.io_threads or len(set(free)) != len(free) \
                    or st.stage_sem._value > st.io_threads:
                bad.append((len(free), st.stage_sem._value, sorted(free)))
            time.sleep(0.0005)

    s = threading.Thread(target=sampler, daemon=True)
    s.start()
    st.resolve(0, ids(*range(20)), prefill=True, defer=True)
    st.join_pending()
    stop.set()
    s.join(5)
    assert bad == [], f"free list / semaphore inconsistent mid-flight: {bad[:3]}"
    assert_leases_at_rest(st, "after 20 clean loads")

    # (a) the read raises: _read_leased's finally owns the release
    _FAIL_READ.update({(1, 4), (1, 9)})
    st.resolve(1, ids(*range(12)), prefill=True, defer=True)
    err = None
    try:
        st.join_pending()
    except IOError as e:
        err = e
    assert err is not None and "injected NVMe failure" in str(err), err
    assert_leases_at_rest(st, "after 2 of 12 reads raised")
    _FAIL_READ.clear()

    # (b) the SINK raises: the H2D half, past the read, still inside the lease
    real_load = st._load_into_slot

    def exploding_sink(key, slot, prefix=None):
        if key[1] % 3 == 0:
            return st._read_leased(key[0], key[1], prefix,
                                   lambda v: (_ for _ in ()).throw(RuntimeError(f"H2D died {key}")))
        return real_load(key, slot, prefix)

    st._load_into_slot = exploding_sink
    st.resolve(2, ids(*range(9)), prefill=True, defer=True)
    err = None
    try:
        st.join_pending()
    except RuntimeError as e:
        err = e
    assert err is not None and "H2D died" in str(err), err
    assert_leases_at_rest(st, "after 3 of 9 sinks raised")
    print(f"  lease conservation: {st.io_threads}/{st.io_threads} buffers back after clean loads, "
          f"read failures and sink failures; {len(bad)} mid-flight inconsistencies  OK")


# ================================================================ 3. the two pools

def test_two_pools_do_not_deadlock():
    """`read_pool` saturated to a single worker; `pool` must still complete every expert."""
    st, arena = make_store(transient_slots=24, lru_slots=8, io_threads=4, read_threads=1)
    st.resolve(0, ids(*range(16)), prefill=True, defer=True)
    done, err, _ = run_with_timeout(st.join_pending, 60)
    assert done, "16 experts did not finish in 60 s with read_threads=1 -- the pools are wedged"
    assert err is None, err
    assert arena.violations == [], arena.violations
    assert len(arena.content) == 16, arena.content
    assert_leases_at_rest(st, "after a saturated read_pool")
    print("  two pools: read_pool pinned to 1 worker, 16 experts still completed  OK")


def test_one_pool_deadlocks():
    """The reverse arrangement, so the comment above `self.pool` is a CHECKED claim and not folklore.

    An io task submits its pieces to `read_pool` and then blocks on them. Point `read_pool` back at
    `pool` and hand it as many experts as there are workers: every worker is inside an io task
    waiting for a piece that can only run on a worker. Nothing retires. This is why there are two
    pools, and the cost of getting it wrong is a hung server, not a slow one.
    """
    st, arena = make_store(transient_slots=16, lru_slots=8, io_threads=4, read_threads=4)
    prespawn(st.pool, 4)
    rec = _OnePool(st.pool)
    st.read_pool = rec
    st.resolve(0, ids(0, 1, 2, 3), prefill=True, defer=True)
    done, err, t = run_with_timeout(st.join_pending, 3.0)
    assert not done, ("one pool did NOT deadlock: 4 experts on 4 shared workers completed. "
                      "If this is now true the two-pool split may be removable -- measure it.")
    assert arena.content == {}, f"a slot landed despite the deadlock: {arena.content}"
    n = rec.cancel_all()
    assert n > 0, "nothing was cancellable -- the pieces had started, so this was not the deadlock"
    t.join(30)
    assert not t.is_alive(), "cancelling the queued pieces did not unwedge the io workers"
    # the unwind is itself an exception path: every lease must have come back
    assert_leases_at_rest(st, "after the deadlock was cancelled")
    print(f"  one pool: 4 experts on 4 shared workers made no progress in 3 s "
          f"({n} pieces queued behind them, never scheduled)  OK")


# ================================================================ 4. join completeness

def test_join_pending_completes_even_when_a_load_raises():
    """After join_pending(): every future done, `_pending_slots` empty, `_pending_layer` cleared.

    The straight `for f: f.result()` returned through the FIRST exception with all three still
    populated. The damage was never at the failure -- it was one request later, when the ring had
    permanently lost those slots and a healthy resolve died with "transient ring exhausted", or
    when the next layer's deferred resolve tripped the span-layers assert. And the reads of that
    layer were still landing in an arena the caller had been told was quiesced.
    """
    st, arena = make_store(transient_slots=16, lru_slots=8)
    _FAIL_READ.update({(3, 2), (3, 5)})
    st.resolve(3, ids(0, 1, 2, 3, 4, 5, 6), prefill=True, defer=True)
    futs = [f for f, _, _ in st._pending]
    err = None
    try:
        st.join_pending()
    except IOError as e:
        err = e
    assert err is not None, "join_pending() swallowed a failed load"
    assert all(f.done() for f in futs), \
        f"{sum(not f.done() for f in futs)} of {len(futs)} futures still running after join_pending()"
    assert st._pending == [], st._pending
    assert st._pending_slots == set(), \
        f"slots marked pending forever after a failure: {sorted(st._pending_slots)}"
    assert st._pending_layer is None, st._pending_layer
    # and the failed experts must not be reachable as hits: their slots hold a torn read
    for k in ((3, 2), (3, 5)):
        assert k not in st.transient_map and k not in st.lru, \
            f"{k} still maps to a slot whose read raised -- the next resolve() counts a HIT on garbage"
    _FAIL_READ.clear()
    # the store is still usable: the same layer resolves and joins cleanly afterwards
    st.resolve(3, ids(2, 5, 8), prefill=True, defer=True)
    st.join_pending()
    assert arena.violations == [], arena.violations
    assert_leases_at_rest(st, "after recovering from a failed join")
    print("  join completeness: 2 of 7 loads raised, all 7 futures done, pending state empty, "
          "failed keys un-mapped, store still usable  OK")


# ================================================================ 5. cross-layer safety

def test_cross_layer_defer_asserts_in_every_mix():
    """Deferred resolves spanning a layer boundary must assert, prefill or decode or mixed.

    test_expert_store.py covers prefill->prefill. The mixed cases matter because the two halves of
    the arena take different paths to a slot (`_transient_slot_for` vs `_lru_slot_for`) and only
    one of them used to know about `_pending_slots` at all.
    """
    cases = [(True, True), (True, False), (False, True), (False, False)]
    for first, second in cases:
        st, arena = make_store(transient_slots=16, lru_slots=16)
        st.resolve(0, ids(1, 2), prefill=first, defer=True)
        raised = None
        try:
            st.resolve(1, ids(3, 4), prefill=second, defer=True)
        except AssertionError as e:
            raised = e
        assert raised is not None and "span layers" in str(raised), (first, second, raised)
        st.join_pending()
        assert arena.violations == [], arena.violations
    # and the legal mix -- prefill then decode INSIDE one layer -- must be accepted and disjoint
    st, arena = make_store(transient_slots=16, lru_slots=16)
    a = st.resolve(7, ids(1, 2, 3), prefill=True, defer=True).tolist()[0]
    b = st.resolve(7, ids(4, 5), prefill=False, defer=True).tolist()[0]
    assert set(a).isdisjoint(b), (a, b)
    st.join_pending()
    assert arena.violations == [], arena.violations
    assert len(arena.content) == 5, arena.content
    print(f"  cross-layer: all 4 prefill/decode mixes assert; prefill+decode inside one layer "
          f"is accepted with disjoint slots  OK")


# ================================================================ 6. decode-shaped concurrency

def test_decode_shaped_concurrency():
    """40 layers, a handful of routed experts each, ~1.7 misses per layer -- the untested regime.

    Every existing store test is prefill-shaped: hundreds of misses in one call, one layer, the
    transient ring. Decode is the opposite -- a couple of misses per layer, 40 layers deep, through
    the LRU, with an eviction under way on most of them. That is the shape the async work is going
    to target, so the invariants have to be pinned HERE, not only where the batch is large enough
    to hide an ordering mistake.

    The consumer runs where the engine runs it: after join_pending() and before the next layer's
    resolve (engine/model.py -- moe_apply is between them).
    """
    n_layers, steps = 40, 4
    # 5 hot experts per layer (200 keys) that must stay resident + 2 cold per layer drawn from a
    # rotating pool of 4 (160 keys) that must mostly not. The capacity has to exceed ONE STEP's
    # distinct accesses (280) or LRU degenerates on a cyclic trace and every access misses -- which
    # is a statement about LRU, not about this loader, and would make the shape claim meaningless.
    st, arena = make_store(transient_slots=8, lru_slots=300, io_threads=6, read_threads=4)
    hot = {L: [(L * 11 + i) % 24 for i in range(5)] for L in range(n_layers)}
    misses0 = st.stats["misses"]
    for step in range(steps):
        for L in range(n_layers):
            # one cold expert per layer that is new every step (a guaranteed miss) plus a second
            # on 70 % of them: 1.7 misses/layer, the measured decode rate
            cold = [100 + (L * 7 + step * 13) % 97]
            if (L + step) % 10 < 7:
                cold.append(250 + (L * 11 + step * 17) % 89)
            route = hot[L] + cold
            slots = st.resolve(L, ids(*route), prefill=False, defer=True)
            st.join_pending()
            got = consume(arena, slots.tolist()[0], hold=0.0002)
            for e, k in zip(route, got):
                assert k == (L, e), f"layer {L} expert {e} read slot holding {k}"
        if step == 0:                       # first pass is the warm-up; measure the steady state
            misses0 = st.stats["misses"]
    steady = (st.stats["misses"] - misses0) / ((steps - 1) * n_layers)
    assert arena.violations == [], arena.violations
    assert 1.4 <= steady <= 2.1, \
        f"{steady:.2f} misses/layer is not the decode shape (~1.7) -- retune, or this proves nothing"
    assert st._pending == [] and st._pending_slots == set()
    assert_leases_at_rest(st, "after 40 layers x 4 decode steps")
    print(f"  decode shape: {n_layers} layers x {steps} steps, {steady:.2f} misses/layer steady "
          f"state, {st.stats['hits']} hits, 0 overlaps, every slot read held its own expert  OK")


def test_decode_shaped_concurrency_deferred_across_chunks():
    """The same shape, but with SEVERAL deferred resolves inside each layer before the join.

    This is what a micro-batched or speculative decode step looks like from the store's side, and
    it is the arrangement in which `_lru_slot_for` has to know about `_pending_slots` -- pass 1 of
    the second resolve sees the first resolve's keys as residents and the eviction search sees
    their slots as candidates.
    """
    n_layers, steps = 40, 4
    st, arena = make_store(transient_slots=8, lru_slots=300, io_threads=6, read_threads=4)
    misses0 = 0
    for step in range(steps):
        for L in range(n_layers):
            groups = [[(L * 11 + i) % 24 for i in range(3)],
                      [(L * 11 + i) % 24 for i in range(3, 5)]
                      + [100 + (L * 7 + step * 13) % 97]
                      + ([250 + (L * 11 + step * 17) % 89] if (L + step) % 10 < 7 else [])]
            handed = []
            for g in groups:
                handed += list(zip(g, st.resolve(L, ids(*g), prefill=False, defer=True).tolist()[0]))
            live = [s for _, _, s in st._pending]
            assert len(set(live)) == len(live), \
                f"layer {L}: two in-flight decode loads share a slot {sorted(live)}"
            st.join_pending()
            for e, s in handed:
                with arena.reading(s) as k:
                    assert k == (L, e), f"layer {L} expert {e} -> slot {s} holds {k}"
        if step == 0:
            misses0 = st.stats["misses"]
    steady = (st.stats["misses"] - misses0) / ((steps - 1) * n_layers)
    assert arena.violations == [], arena.violations
    assert 1.4 <= steady <= 2.1, f"{steady:.2f} misses/layer is not the decode shape (~1.7)"
    assert_leases_at_rest(st, "after chunked decode resolves")
    print(f"  decode shape: 2 deferred resolves per layer x {n_layers} layers x {steps} steps, "
          f"{steady:.2f} misses/layer, no slot shared by two in-flight loads  OK")


# ================================================================ the stream barrier

def test_compute_barrier_is_taken_before_the_arena_write():
    """`stream.wait_stream(compute)` must happen between the read and the H2D, and only the
    documented gate may remove it.

    Removing the barrier measures 14.5 % SLOWER, which nobody has explained, so the temptation to
    "simplify" it away is real. Run with fake streams so this needs no CUDA context of its own and
    can assert the ORDER, which is the part that matters: the wait is recorded in the sink, AFTER
    the ~5 ms read, not when the lease was taken.
    """
    st, arena = make_store(transient_slots=8, lru_slots=8, io_threads=2)
    order: list = []

    class FakeStream:
        def wait_stream(self, other):
            order.append(("wait", other))

        def synchronize(self):
            order.append(("sync", None))

    class FakeCB3:
        record = 8192

        def read_into(self, mv, layer, expert):
            order.append(("read", (layer, expert)))
            mv[:] = b"\x5a" * len(mv)

        def load_slot(self, arena_, slot, buf, non_blocking=True):
            order.append(("h2d", slot))

    compute = object()
    st.cb3_cache = FakeCB3()
    st._copy_stream = FakeStream
    old_cur, old_ctx = torch.cuda.current_stream, torch.cuda.stream
    torch.cuda.current_stream = lambda *a, **k: compute
    torch.cuda.stream = lambda s: contextlib.nullcontext()
    try:
        E.ExpertStore._load_into_slot(st, (0, 5), 3)
        got = [k for k, _ in order]
        assert got == ["read", "wait", "h2d", "sync"], got
        assert order[1][1] is compute, order[1]
        # ...and the gate, which is the only sanctioned way to drop it
        order.clear()
        E.UNSAFE_NO_COMPUTE_WAIT = True
        E.ExpertStore._load_into_slot(st, (0, 6), 4)
        assert [k for k, _ in order] == ["read", "h2d", "sync"], order
    finally:
        E.UNSAFE_NO_COMPUTE_WAIT = False
        torch.cuda.current_stream, torch.cuda.stream = old_cur, old_ctx
    assert_leases_at_rest(st, "after the barrier probe")
    print("  stream barrier: wait_stream(compute) sits between the read and the arena write, "
          "and only DSV41_UNSAFE_NO_COMPUTE_WAIT removes it  OK")


def test_slot_reuse_across_layers_is_ordered_by_the_stream_barrier_alone():
    """DOCUMENTS A HAZARD, and fails if that stops being true.

    join_pending() clears `_pending_slots`, so from the host's point of view the next layer may
    reuse any slot immediately. But the consumer is a KERNEL: the MoE of layer L is still executing
    on the compute stream long after moe_apply returned on the host, and layer L+1's H2D into the
    same slot is issued from an io thread on a different stream. Nothing in resolve(), `used`,
    `_pending_slots` or join_pending() orders those two -- `stream.wait_stream(compute)` is the
    ONLY thing that does, which is exactly why removing it "for speed" corrupts an in-flight slot.

    Here the consumer is held inside its read while the next layer's write lands, and the arena
    records the overlap. The assertion is that the overlap IS observed: if a future change makes
    the host side protect this too, this test fails and the wait_stream comment has to be revisited.
    """
    st, arena = make_store(transient_slots=8, lru_slots=8, io_threads=4)
    first = st.resolve(0, ids(*range(8)), prefill=True, defer=True).tolist()[0]
    st.join_pending()
    slot0 = first[0]
    started, release = threading.Event(), threading.Event()

    def still_computing():
        with arena.reading(slot0):          # layer 0's MoE, still on the compute stream
            started.set()
            release.wait(10)

    t = threading.Thread(target=still_computing, daemon=True)
    t.start()
    assert started.wait(5)
    st.resolve(1, ids(*range(8)), prefill=True, defer=True)   # wraps the ring onto layer 0's slots
    st.join_pending()
    release.set()
    t.join(10)
    kinds = {k[0] for k in arena.violations}
    assert "write-during-read" in kinds, (
        "layer 1's H2D no longer overlaps a slot layer 0's consumer is reading. Host-side "
        "protection was added somewhere -- revisit the wait_stream comment in _copy_stream(), "
        f"it may no longer be the only ordering. violations={arena.violations}")
    assert arena.content[slot0] == (1, 0), arena.content[slot0]
    assert_leases_at_rest(st, "after the cross-layer overlap")
    print(f"  cross-layer slot reuse: layer 1 wrote slot {slot0} while layer 0's consumer held it "
          f"-- host-side bookkeeping does not order this, wait_stream(compute) does  OK")


TESTS = (
    test_no_torn_slot_under_deferred_submission,
    test_no_torn_slot_decode_lru_eviction,
    test_lease_conservation_normal_and_under_failure,
    test_two_pools_do_not_deadlock,
    test_one_pool_deadlocks,
    test_join_pending_completes_even_when_a_load_raises,
    test_cross_layer_defer_asserts_in_every_mix,
    test_decode_shaped_concurrency,
    test_decode_shaped_concurrency_deferred_across_chunks,
    test_compute_barrier_is_taken_before_the_arena_write,
    test_slot_reuse_across_layers_is_ordered_by_the_stream_barrier_alone,
)


def main(argv) -> int:
    E._pread_chunk = _fake_pread
    want = argv[1] if len(argv) > 1 else ""
    chosen = [t for t in TESTS if want in t.__name__]
    print(f"expert-loader I/O path invariants ({len(chosen)} tests):")
    fails = 0
    for fn in chosen:
        t0 = time.perf_counter()
        try:
            fn()
        except BaseException as exc:            # noqa: BLE001 -- report all, then fail once
            fails += 1
            import traceback
            print(f"  FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            print(f"        ({fn.__name__}, {time.perf_counter() - t0:.1f}s)")
            close_all()
    E._pread_chunk = _REAL_PREAD
    print("ALL OK" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
