"""store.py -- slot bookkeeping, the two eviction policies, staging leases, per-slot readiness.

Everything here is REAL: a real LRU ordering, a real transient ring, a real count-bucketed
age/(1+count) victim search, a real semaphore-guarded pinned-buffer pool, real per-slot events.
Only what a slot CONTAINS is fake (a key tag instead of 13.7 MB of FP4).

The eviction logic is ported from engine/experts.py rather than reinvented, because it is
measured-good: LRU 92.67 % hit / 64.6 fetches per decode step, age/(1+count) 94.07 % / 52.3. Those
are the numbers enginev2/phase1.py has to reproduce before anything else in this package is
allowed to mean something.
"""

from __future__ import annotations

import collections
import contextlib
import threading

from .observe import NO_CTX, Event, NullObserver, WaitReason, next_span, now_ns

N_EXPERTS = 384
# ---------------------------------------------------------------------------
# Eviction: an interface, not a flag.
# ---------------------------------------------------------------------------
#
# This used to be a two-valued string dispatched through `if self._afq:` at six sites, so a third
# policy meant editing all six. A policy is now an object owning its own bookkeeping, and the one
# thing that makes the swap safe is that phase1 is an EXACT reproduction gate: lru must still give
# 92.67 % / 64.6 / 849 and age_over_freq 94.07 % / 52.3 / 687, bit for bit.
#
# `victim(residents, protected)` returns the KEY to evict, or None if it cannot. `protected` covers
# both slots promised earlier in this call and slots with a write in flight -- a policy never needs
# to know which, only that it may not touch them.


class EvictionPolicy:
    """Bookkeeping for one cache region. All hooks are called with the store's lock held."""

    name = "abstract"

    def on_hit(self, key: tuple, slot: int, clock: int) -> None:
        """A resident was used. `clock` is the store's logical time (decode resolves only)."""

    def on_insert(self, key: tuple, slot: int, clock: int) -> None:
        """A key became resident in `slot` because something DEMANDED it."""

    def on_restore(self, key: tuple, slot: int) -> None:
        """A key that was evicted is resident again and NOTHING about it changed.

        Distinct from on_admit: a restore must not touch age or count at all, because the eviction
        it undoes never physically happened.
        """
        self.on_admit(key, slot, 0)

    def on_admit(self, key: tuple, slot: int, clock: int) -> None:
        """A key became resident SPECULATIVELY -- nothing has used it.

        It must become evictable (a policy that cannot see it can never evict it, which leaks the
        slot) without receiving any usage credit. Default: treat it as an insert, which is correct
        for policies that carry no usage history.
        """
        self.on_insert(key, slot, clock)

    def on_drop(self, key: tuple) -> None:
        """A key stopped being resident (evicted, or un-mapped after a torn read)."""

    def victim(self, residents: "collections.OrderedDict", protected: frozenset,
               clock: int) -> tuple | None:
        raise NotImplementedError


class LRUPolicy(EvictionPolicy):
    """Least-recently-used. The residents dict is already in LRU order, so the head wins."""

    name = "lru"

    def victim(self, residents, protected, clock):
        for k, slot in residents.items():
            if slot not in protected:
                return k
        return None


class AgeOverFreqPolicy(EvictionPolicy):
    """EXACT global argmax of age/(1+count), in O(#distinct use counts).

    Within a count bucket the count is constant, so that bucket's maximum age is its LRU head -- the
    global winner is the best of the bucket heads. ~228 comparisons per eviction at 5,328 slots,
    verified bit-identical to the brute-force scan offline. This is what makes 94.07 % / 52.3
    affordable at all.

    `_use_count` and `_last_acc` survive eviction ON PURPOSE, exactly as the offline replay's
    per-key stats do: an expert that comes back from NVMe comes back with its history, and that is
    the only way a use count means anything when 5,328 slots cover 15,360 pairs.
    """

    name = "age_over_freq"

    def __init__(self):
        self._use_count: dict[tuple, int] = {}
        self._last_acc: dict[tuple, int] = {}
        self._buckets: dict[int, collections.OrderedDict] = {}

    def _touch(self, key: tuple, slot: int, clock: int) -> None:
        old = self._use_count.get(key, 0)
        b = self._buckets.get(old)
        if b is not None:
            b.pop(key, None)
            if not b:
                del self._buckets[old]
        new = old + 1
        self._use_count[key] = new
        self._last_acc[key] = clock
        self._buckets.setdefault(new, collections.OrderedDict())[key] = slot

    on_hit = _touch
    on_insert = _touch

    def on_admit(self, key: tuple, slot: int, clock: int) -> None:
        """Register in the bucket for the count it ALREADY has, and age from now.

        A speculative insert that bumped the count was the bug: a wrong prefetch permanently
        credited that expert with a use it never had, and on_drop preserves _use_count on purpose,
        so the phantom survived eviction and made the expert harder to evict every time it came
        back. That is precisely the eviction signal the predictor experiments must not rewrite.
        Ageing from `clock` is deliberate too -- an unused speculative entry should age normally
        from arrival, so it is evicted early rather than looking ancient or looking fresh.
        """
        c = self._use_count.get(key, 0)
        # Materialise the count even when it is zero. on_drop() looks the key up in _use_count to
        # find which bucket to remove it from and returns early when it is absent -- so admitting
        # without writing it left the key in bucket 0 forever after eviction, and the victim search
        # later returned a key no longer in `lru`. Writing 0 is not a usage credit: it records the
        # count this key already had.
        self._use_count.setdefault(key, c)
        self._last_acc[key] = clock
        self._buckets.setdefault(c, collections.OrderedDict())[key] = slot

    def on_drop(self, key: tuple) -> None:
        c = self._use_count.get(key)
        if c is None:
            return
        b = self._buckets.get(c)
        if b is not None:
            b.pop(key, None)
            if not b:
                del self._buckets[c]

    def on_restore(self, key: tuple, slot: int) -> None:
        # Re-register at the count AND the age it already had; an undone eviction leaves no trace.
        c = self._use_count.get(key, 0)
        self._buckets.setdefault(c, collections.OrderedDict())[key] = slot

    def victim(self, residents, protected, clock):
        best = best_key = None
        for c, b in self._buckets.items():
            for k, s in b.items():
                if s in protected:
                    continue
                score = (clock - self._last_acc.get(k, 0)) / (1.0 + c)
                rank = (score, -self._last_acc.get(k, 0))
                if best_key is None or rank > best_key:
                    best_key, best = rank, k
                break                             # only the head of each bucket can win
        return best


EVICT_POLICIES = {"lru": LRUPolicy, "age_over_freq": AgeOverFreqPolicy}


def make_policy(spec) -> EvictionPolicy:
    """Accept a name or an EvictionPolicy instance -- the latter is the plug point."""
    if isinstance(spec, EvictionPolicy):
        return spec
    if spec not in EVICT_POLICIES:
        raise ValueError(f"policy {spec!r} not in {sorted(EVICT_POLICIES)} and not an EvictionPolicy")
    return EVICT_POLICIES[spec]()


class SlotArena:
    """The arena as the host can see it: who writes a slot, who reads it, what it holds.

    Lifted in spirit from engine/test_io_path.py's SlotArena. Overlaps are RECORDED, not raised:
    an assertion inside a worker reaches the caller as a future's exception at join time and loses
    which two parties collided.
    """

    def __init__(self, slots: int):
        self.slots = slots
        self._lk = threading.Lock()
        self.content: dict[int, tuple] = {}
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


class StagingPool:
    """Pinned staging buffers, leased one per in-flight expert read.

    The acquire order is deliberately v1's: semaphore FIRST, then pop the free list. Between those
    two statements the free list is one longer than the semaphore, so exact agreement is an AT-REST
    property only -- engine/test_io_path.py's sampler asserts the weaker mid-flight invariant and
    this pool must keep that true, because the v2 tests re-assert it.

    A leaked lease is unrecoverable: there are only `n` for the life of the process and a handful
    of leaks means every miss blocks forever with the NVMe idle. Hence release-in-finally, always.

    Each lease owns a real BUFFER, not just a ticket. That is what makes zero-copy testable: a
    provider must hand the H2D views that ALIAS this memory, and `same_buffer()` can prove it did
    not quietly clone. A real provider swaps these for `torch.empty(n, pin_memory=True)`; identity
    is checked by object, so the test survives the substitution.
    """

    def __init__(self, n: int, nbytes: int = 4096, observer=None):
        self.obs = observer if observer is not None else NullObserver()
        self.n = n
        self.nbytes = nbytes
        self._sem = threading.Semaphore(n)
        self._lk = threading.Lock()
        self.free = list(range(n))
        self.peak_in_use = 0
        # Small by default: identity is what the contract tests need, not 48 x 13.8 MB.
        self._buf = [bytearray(nbytes) for _ in range(n)]
        self._leased: set[int] = set()

    def buffer(self, sid: int) -> memoryview:
        """The pinned buffer for this lease. Only valid while the lease is held."""
        if sid not in self._leased:
            raise RuntimeError(f"staging buffer {sid} accessed while not leased")
        return memoryview(self._buf[sid])

    def same_buffer(self, sid: int, view) -> bool:
        """True if `view` ALIASES this lease's buffer rather than being a copy of it.

        The invariant is storage identity plus containment, not any one language's buffer protocol.
        `view.obj is buf` only answers it for memoryview(bytearray); a real provider hands back a
        pinned torch.Tensor view, which shares storage with its base without exposing that
        relationship the same way -- the check would then report non-aliasing on memory that is in
        fact the same bytes. So the pool asks the BUFFER, and a provider that supplies real pinned
        memory supplies the matching predicate with it.
        """
        buf = self._buf[sid]
        probe = getattr(buf, "aliases", None)          # provider-supplied buffers answer for themselves
        if callable(probe):
            return bool(probe(view))
        if hasattr(view, "data_ptr") and hasattr(buf, "data_ptr"):
            # torch: same storage, and the view lies inside the leased range
            try:
                if view.untyped_storage().data_ptr() != buf.untyped_storage().data_ptr():
                    return False
                lo = buf.data_ptr()
                return lo <= view.data_ptr() < lo + buf.numel() * buf.element_size()
            except Exception:
                return False
        return getattr(view, "obj", None) is buf       # memoryview over the model's bytearray

    def in_use(self, sid: int) -> bool:
        with self._lk:
            return sid in self._leased

    def acquire(self, ctx=NO_CTX) -> int:
        # The lease is the ownership point for STAGING_BUFFER waits: whoever blocks here is blocked
        # on a pinned buffer, whatever they meant to do with it.
        # Atomic probe: testing a private _value and then acquiring is racy in both directions --
        # another worker can take or release a permit in between, so real waits are missed and
        # false ones recorded. try-acquire answers exactly the question being asked.
        if self.obs.enabled and not self._sem.acquire(blocking=False):
            sp = next_span()
            self.obs.safe_emit(Event(now_ns(), "wait_start", ctx=ctx, span=sp,
                                     aux=WaitReason.STAGING_BUFFER))
            self._sem.acquire()
            self.obs.safe_emit(Event(now_ns(), "wait_end", ctx=ctx, span=sp,
                                     aux=WaitReason.STAGING_BUFFER))
        elif not self.obs.enabled:
            self._sem.acquire()
        with self._lk:
            sid = self.free.pop()
            self._leased.add(sid)
            self.peak_in_use = max(self.peak_in_use, self.n - len(self.free))
            return sid

    def release(self, sid: int) -> None:
        with self._lk:
            if sid not in self._leased:
                raise RuntimeError(f"staging buffer {sid} released twice")
            self._leased.discard(sid)
            self.free.append(sid)
        self._sem.release()

    def at_rest(self) -> bool:
        with self._lk:
            return len(self.free) == self.n and self._sem._value == self.n and \
                len(set(self.free)) == self.n

    @property
    def sem_value(self) -> int:
        return self._sem._value


class SlotReady:
    """Per-slot readiness, generation-tagged -- the skeleton's stand-in for one CUDA event per slot.

    v1 has no such thing: its only readiness signal is join_pending(), a barrier over EVERY pending
    read. The generation tag is what makes a per-slot wait safe when a slot is recycled: waiting on
    (slot, gen) cannot be satisfied by a previous tenant's completion.

    READINESS IS EXACT, NOT MONOTONIC. An earlier revision kept `slot -> highest generation
    completed` and waited on `>=`, which is wrong in both directions and was caught in review:

      * `set()` assigned unconditionally, so if gen 2 completed before gen 1 the recorded readiness
        REGRESSED from 2 to 1.
      * `>=` let a waiter for gen 1 be released by gen 2's completion -- at which point the slot
        holds gen 2's expert, not the one that waiter was promised. Silent wrong data.
      * `arm()` popped the slot's entry, so arming gen 2 stranded a live gen-1 waiter at -1.

    A completion is therefore recorded against the exact (slot, generation) pair and a waiter is
    released only by its own. Nothing is pruned: with pending-slot ownership in ExpertSlots there is
    at most one write in flight per slot, so the set grows with fetches, not with time, and this is
    a skeleton. The real engine needs none of it -- one CUDA event per slot, re-recorded per write,
    is exact for the same reason and is naturally recycled.
    """

    def __init__(self, observer=None):
        self.obs = observer if observer is not None else NullObserver()
        self._lk = threading.Lock()
        self._cv = threading.Condition(self._lk)
        self._done: set[tuple] = set()       # (slot, generation) pairs that have completed
        self._ts: dict[tuple, int] = {}      # when each became ready, for prefetch lead time
        # arm() records the PRODUCER's context -- who submitted this read. That is the right owner
        # for nvme/h2d/slot_ready, and the WRONG one for a wait: a speculative read armed at layer
        # 12 for layer 16 would attribute layer 16's blocking to layer 12. The consumer passes its
        # own context to wait(); only cause_id is recovered from here, because the cause really is
        # the prediction that issued the read.
        self._ctx: dict[tuple, tuple] = {}   # (slot, gen) -> (producer_ctx, cause_id)
        self._err: dict[tuple, BaseException] = {}

    def arm(self, slot: int, gen: int, ctx=NO_CTX, cause_id: int = 0) -> None:
        # Still clears nothing -- clearing here is what stranded waiters before. It only RECORDS
        # who this generation belongs to, so later events can carry it.
        if ctx is not NO_CTX or cause_id:
            self._ctx[(slot, gen)] = (ctx, cause_id)

    def set(self, slot: int, gen: int, err: BaseException | None = None) -> None:
        if self.obs.enabled:
            c, cause = self._ctx.get((slot, gen), (NO_CTX, 0))
            self.obs.safe_emit(Event(now_ns(), "slot_ready", ctx=c, slot=slot, gen=gen,
                                     cause_id=cause, aux=err))
        with self._lk:
            self._done.add((slot, gen))
            self._ts[(slot, gen)] = now_ns()
            if err is not None:
                self._err[(slot, gen)] = err
            self._cv.notify_all()

    def is_done(self, slot: int, gen: int) -> bool:
        with self._lk:
            return (slot, gen) in self._done

    def ready_ts(self, slot: int, gen: int):
        """When (slot, gen) became ready, or None if it has not. Lets the driver tell a prefetch
        that ARRIVED from one that was merely mapped and is still in flight."""
        with self._lk:
            return self._ts.get((slot, gen))

    def wait(self, slot: int, gen: int, timeout: float | None = None, ctx=NO_CTX,
             key: tuple | None = None) -> None:
        with self._lk:
            blocked = (slot, gen) not in self._done
            sp = next_span()
            _producer, cause = self._ctx.get((slot, gen), (NO_CTX, 0))
            if blocked and self.obs.enabled:
                # ctx is the CONSUMER: the layer actually blocked. cause_id is the producer's
                # prediction, so "who waited" and "what caused the wait" stay separable.
                # the key travels with the wait: without it a trace cannot say WHICH expert the
                # consumer was blocked on, only which slot, and slots are recycled.
                self.obs.safe_emit(Event(now_ns(), "wait_start", ctx=ctx, key=key, slot=slot,
                                         gen=gen, span=sp, cause_id=cause,
                                         aux=WaitReason.EXPERT_DATA))
            try:
                if not self._cv.wait_for(lambda: (slot, gen) in self._done, timeout):
                    raise TimeoutError(f"slot {slot} gen {gen} never became ready")
            finally:
                if blocked and self.obs.enabled:
                    self.obs.safe_emit(Event(now_ns(), "wait_end", ctx=ctx, key=key, slot=slot,
                                             gen=gen, span=sp, cause_id=cause,
                                             aux=WaitReason.EXPERT_DATA))
            err = self._err.get((slot, gen))
        if err is not None:
            raise err


class ExpertSlots:
    """Slot reservation: LRU region + transient ring, and the pluggable victim policy.

    Ported from engine/experts.py `resolve` passes 1 and 2. The `used` set is load-bearing and is
    the reason it is ported rather than rewritten: reserving residents BEFORE allocating for misses
    is what stops a later miss running the ring over a slot an earlier hit in the same call already
    holds. That bug gave two experts one slot, the second load overwrote the first, and moe_forward's
    `y[t] +=` silently dropped a contribution.
    """

    def __init__(self, lru_slots: int, transient_slots: int, policy="lru"):
        # `policy` is a name or an EvictionPolicy instance. The instance form is the plug point:
        # a new policy implements on_hit / on_insert / on_drop / victim and needs no edit here.
        self.evict = make_policy(policy)
        assert transient_slots >= 8, "transient ring too small"
        self.lru_slots = lru_slots
        self.transient_slots = transient_slots
        self.n_slots = lru_slots + transient_slots
        self.policy = self.evict.name

        self.lru: collections.OrderedDict[tuple, int] = collections.OrderedDict()
        self.slot_key: dict[int, tuple] = {}
        self.free_lru = list(range(lru_slots))
        self.transient_ring = list(range(lru_slots, self.n_slots))
        self.transient_pos = 0
        self.transient_map: dict[tuple, int] = {}
        self.gen: dict[int, int] = {}                      # slot -> generation, bumped on every write
        # PENDING-WRITE OWNERSHIP. A slot whose write is still in flight must not be handed to a
        # second writer, and generations do NOT provide this: a generation tag protects a CONSUMER
        # from reading a stale tenant, it does not serialise two producers. v1 has `_pending_slots`
        # for exactly this; the skeleton had no equivalent, so a later reserve() could pick a slot
        # mid-write. Decode never exposed it -- one layer is in flight at a time -- but lookahead,
        # the whole point of v2, breaks it immediately. Caught in review 2026-09-15.
        self._pending: dict[int, int] = {}                 # slot -> generation of the in-flight write
        # Which resident a SPECULATIVE reservation displaced. A speculation cancelled before it read
        # anything never overwrote the slot, so the old tenant's bytes are still there and the
        # eviction can be undone at zero cost -- see rollback_speculative(). Without this a wrong
        # prediction still raised later demand misses even when its read was cancelled.
        self._displaced: dict[tuple, tuple] = {}           # (slot, gen) -> displaced key
        self._pending_lk = threading.Lock()

        # age/(1+count). Both dicts survive eviction ON PURPOSE, exactly as the offline replay's
        # per-key stats do: an expert that comes back from NVMe comes back with its history, and
        # that is the only way a use count means anything when 5,328 slots cover 15,360 pairs.
        self._clock = 0
        self.hits = 0
        self.misses = 0
        self.restored = 0
        self._last_victim = None
        self.prefill_misses = 0

    # ---------------------------------------------------------------- pending writes
    def mark_pending(self, to_load) -> None:
        """Called by the loader at submit. A pending slot is non-evictable until `clear_pending`."""
        with self._pending_lk:
            for _key, slot, gen in to_load:
                self._pending[slot] = gen

    def clear_pending(self, slot: int, gen: int) -> None:
        """Called when that write has completed (or failed). Only clears its OWN generation, so a
        late completion cannot unprotect a newer write that has since been armed on the slot."""
        with self._pending_lk:
            if self._pending.get(slot) == gen:
                del self._pending[slot]

    def pending_slots(self) -> frozenset:
        with self._pending_lk:
            return frozenset(self._pending)

    def pending_gen(self, slot: int):
        with self._pending_lk:
            return self._pending.get(slot)

    # ---------------------------------------------------------------- allocation
    def _lru_slot_for(self, key: tuple, used: frozenset, touch: bool = True) -> int:
        self._last_victim = None
        if self.free_lru:
            slot = self.free_lru.pop()
        else:
            victim = self.evict.victim(self.lru, used, self._clock)
            self._last_victim = victim
            if victim is None:
                raise RuntimeError(
                    "no evictable LRU slot: every resident is in use by this call or has a write "
                    "in flight -- raise lru_slots, or bound how far lookahead may run ahead")
            slot = self.lru.pop(victim)
            self.slot_key.pop(slot, None)
            self.evict.on_drop(victim)
        self.lru[key] = slot
        self.slot_key[slot] = key
        if touch:
            self.evict.on_insert(key, slot, self._clock)
        else:
            self.evict.on_admit(key, slot, self._clock)
        return slot

    def _transient_slot_for(self, key: tuple, used: frozenset) -> int:
        for _ in range(self.transient_slots):
            slot = self.transient_ring[self.transient_pos]
            self.transient_pos = (self.transient_pos + 1) % self.transient_slots
            if slot in used:
                continue
            old = self.slot_key.get(slot)
            if old is not None:
                self.transient_map.pop(old, None)
            self.transient_map[key] = slot
            self.slot_key[slot] = key
            return slot
        raise RuntimeError("transient ring exhausted: every slot is in use or mid-write")

    def reserve(self, layer: int, uniq, prefill: bool) -> tuple[dict, list]:
        """Host-only bookkeeping: assign every requested expert a slot, return the misses to load.

        Returns (slot_of, to_load, to_wait). to_load is [(key, slot, gen)] to ISSUE; to_wait is
        already-in-flight writes (a prefetch, or another layer's read) this consumer must still
        wait on. No I/O happens here and no
        blocking -- which is the point: in v1 `slots` is already correct the moment this returns,
        and the only thing the wait afterwards provides is DATA readiness. v2 exploits that; v1
        does not.
        """
        slot_of: dict[int, int] = {}
        to_load: list = []
        to_wait: list = []
        used: set[int] = set()
        for e in uniq:
            key = (layer, e)
            s = self.lru.get(key)
            if s is None:
                s = self.transient_map.get(key)
            else:
                self.lru.move_to_end(key)
                if not prefill:
                    # DECODE hits only. A prefill chunk touches nearly every expert of a layer, so
                    # letting it write the counter would push every resident up by one per chunk
                    # and drown the decode signal the policy was fitted on.
                    self._clock += 1
                    self.evict.on_hit(key, s, self._clock)
            if s is not None:
                slot_of[e] = s
                used.add(s)
                self.hits += 1
                # RESIDENT IS NOT THE SAME AS READY. A speculative reserve maps its key the moment
                # it allocates a slot, so a prefetch whose H2D is still in flight looks exactly
                # like a hit here. Waiting on nothing would compute against a half-written slot --
                # the same silent-wrong-data failure as a torn read counted as a HIT. Anything with
                # a write outstanding goes on the wait list even though it is not a new fetch.
                g = self.pending_gen(s)
                if g is not None:
                    to_wait.append((key, s, g))
        # Slots with a write in flight from an EARLIER call are protected exactly as slots promised
        # within this call are. Read once: a slot can only leave this set (a completion), and losing
        # that race costs one extra protected slot for one call, never a mid-write reassignment.
        inflight = self.pending_slots()
        for e in uniq:
            if e in slot_of:
                continue
            key = (layer, e)
            if prefill:
                self.prefill_misses += 1
                s = self._transient_slot_for(key, frozenset(used) | inflight)
            else:
                self.misses += 1
                self._clock += 1
                s = self._lru_slot_for(key, frozenset(used) | inflight)
            slot_of[e] = s
            used.add(s)
            g = self.gen[s] = self.gen.get(s, 0) + 1
            to_load.append((key, s, g))
        assert len(set(slot_of.values())) == len(slot_of), "slot collision in reserve()"
        return slot_of, to_load, to_wait

    def reserve_speculative(self, keys) -> list:
        """Allocate slots for PREDICTED keys without perturbing any cache statistic.

        A speculative touch must not count as a hit, must not advance the age/(1+count) clock and
        must not bump a use count -- otherwise the predictor rewrites the miss stream it is being
        measured against, and phase1's exact reproduction becomes meaningless. So this calls
        on_insert (the key really does become resident) and nothing else.

        Refusal is normal, not an error: with pending-write protection the store can legitimately
        have no free slot, and a prefetcher that cannot place a key simply does not get it. The
        caller is told how many were refused.
        """
        to_load, refused = [], 0
        used = set()
        inflight = self.pending_slots()
        for key in keys:
            if key in self.lru or key in self.transient_map:
                continue                                   # already resident: nothing to do
            try:
                # touch=False: speculation makes a key resident, it does not USE it.
                s = self._lru_slot_for(key, frozenset(used) | inflight, touch=False)
            except RuntimeError:
                refused += 1
                continue
            used.add(s)
            g = self.gen[s] = self.gen.get(s, 0) + 1
            if self._last_victim is not None:
                self._displaced[(s, g)] = self._last_victim
            to_load.append((key, s, g))
        return to_load, refused

    def rollback_speculative(self, key: tuple, slot: int, gen: int) -> bool:
        """Undo a speculative reservation whose read NEVER STARTED.

        Only sound in that case: the arena still holds the displaced tenant's bytes, so restoring
        its mapping restores real data. Returns True if a tenant was put back.
        """
        self.forget(key, slot)
        victim = self._displaced.pop((slot, gen), None)
        if victim is None or self.lru.get(victim) is not None:
            return False
        try:
            self.free_lru.remove(slot)
        except ValueError:
            return False                      # the slot was taken again: too late, and that is fine
        self.lru[victim] = slot
        # ...at the LRU END, not the MRU end. `lru[k] = v` appends, which would promote the tenant
        # we just evicted to most-recently-used -- an undone eviction that left a large trace, and
        # it made the rollback arm measurably WORSE than no rollback. It was the least-recently-used
        # entry when it was chosen as victim, so that is where it goes back.
        self.lru.move_to_end(victim, last=False)
        self.slot_key[slot] = victim
        self.evict.on_restore(victim, slot)
        self.restored += 1
        return True

    def forget(self, key: tuple, slot: int) -> None:
        """Un-map a key whose load did not complete: the slot holds a torn read.

        Leaving it mapped would make the next reserve() count it as a HIT and compute with whatever
        partial bytes landed -- silent, and permanent for the life of the process.
        """
        if self.lru.get(key) == slot:
            del self.lru[key]
            self.free_lru.append(slot)
            self.evict.on_drop(key)
        if self.transient_map.get(key) == slot:
            del self.transient_map[key]
        if self.slot_key.get(slot) == key:
            del self.slot_key[slot]

    @property
    def hit_rate(self) -> float:
        tot = self.hits + self.misses + self.prefill_misses
        return self.hits / tot if tot else 0.0
