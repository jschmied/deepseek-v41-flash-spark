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

N_EXPERTS = 384
EVICT_POLICIES = ("lru", "age_over_freq")


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
    """

    def __init__(self, n: int):
        self.n = n
        self._sem = threading.Semaphore(n)
        self._lk = threading.Lock()
        self.free = list(range(n))
        self.peak_in_use = 0

    def acquire(self) -> int:
        self._sem.acquire()
        with self._lk:
            sid = self.free.pop()
            self.peak_in_use = max(self.peak_in_use, self.n - len(self.free))
            return sid

    def release(self, sid: int) -> None:
        with self._lk:
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

    def __init__(self):
        self._lk = threading.Lock()
        self._cv = threading.Condition(self._lk)
        self._done: set[tuple] = set()       # (slot, generation) pairs that have completed
        self._err: dict[tuple, BaseException] = {}

    def arm(self, slot: int, gen: int) -> None:
        # Deliberately a no-op on state. Clearing anything here is what stranded waiters before.
        pass

    def set(self, slot: int, gen: int, err: BaseException | None = None) -> None:
        with self._lk:
            self._done.add((slot, gen))
            if err is not None:
                self._err[(slot, gen)] = err
            self._cv.notify_all()

    def wait(self, slot: int, gen: int, timeout: float | None = None) -> None:
        with self._lk:
            if not self._cv.wait_for(lambda: (slot, gen) in self._done, timeout):
                raise TimeoutError(f"slot {slot} gen {gen} never became ready")
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

    def __init__(self, lru_slots: int, transient_slots: int, policy: str = "lru"):
        if policy not in EVICT_POLICIES:
            raise ValueError(f"policy {policy!r} not in {EVICT_POLICIES}")
        assert transient_slots >= 8, "transient ring too small"
        self.lru_slots = lru_slots
        self.transient_slots = transient_slots
        self.n_slots = lru_slots + transient_slots
        self.policy = policy
        self._afq = policy == "age_over_freq"

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
        self._pending_lk = threading.Lock()

        # age/(1+count). Both dicts survive eviction ON PURPOSE, exactly as the offline replay's
        # per-key stats do: an expert that comes back from NVMe comes back with its history, and
        # that is the only way a use count means anything when 5,328 slots cover 15,360 pairs.
        self._clock = 0
        self._use_count: dict[tuple, int] = {}
        self._last_acc: dict[tuple, int] = {}
        self._buckets: dict[int, collections.OrderedDict] = {}
        self.hits = 0
        self.misses = 0
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

    # ---------------------------------------------------------------- age/(1+count)
    def _afq_touch(self, key: tuple, slot: int) -> None:
        old = self._use_count.get(key, 0)
        b = self._buckets.get(old)
        if b is not None:
            b.pop(key, None)
            if not b:
                del self._buckets[old]
        new = old + 1
        self._use_count[key] = new
        self._last_acc[key] = self._clock
        self._buckets.setdefault(new, collections.OrderedDict())[key] = slot

    def _afq_drop(self, key: tuple) -> None:
        c = self._use_count.get(key)
        if c is None:
            return
        b = self._buckets.get(c)
        if b is not None:
            b.pop(key, None)
            if not b:
                del self._buckets[c]

    def _afq_victim(self, used: frozenset) -> tuple:
        """EXACT global argmax of (age)/(1+count) over residents, in O(#distinct use counts).

        Within a count bucket the count is constant, so the maximum of age/(1+count) is that
        bucket's LRU head -- the global winner is the best of the bucket heads. ~228 comparisons
        per eviction at 5,328 slots, verified bit-identical to the brute-force scan offline. This
        is what makes 94.07 % / 52.3 fetches affordable at all.
        """
        best = None
        best_key = None
        for c, b in self._buckets.items():
            for k, s in b.items():
                if s in used:
                    continue                      # protected: promised earlier in THIS call
                score = (self._clock - self._last_acc.get(k, 0)) / (1.0 + c)
                rank = (score, -self._last_acc.get(k, 0))
                if best_key is None or rank > best_key:
                    best_key, best = rank, k
                break                             # only the head of each bucket can win
        return best

    # ---------------------------------------------------------------- allocation
    def _lru_slot_for(self, key: tuple, used: frozenset) -> int:
        if self.free_lru:
            slot = self.free_lru.pop()
        else:
            if self._afq:
                victim = self._afq_victim(used)
            else:
                victim = None
                for k in self.lru:
                    if self.lru[k] not in used:
                        victim = k
                        break
            if victim is None:
                raise RuntimeError(
                    "no evictable LRU slot: every resident is in use by this call or has a write "
                    "in flight -- raise lru_slots, or bound how far lookahead may run ahead")
            slot = self.lru.pop(victim)
            self.slot_key.pop(slot, None)
            if self._afq:
                self._afq_drop(victim)
        self.lru[key] = slot
        self.slot_key[slot] = key
        if self._afq:
            self._afq_touch(key, slot)
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

        Returns (slot_of, to_load) where to_load is [(key, slot, gen)]. No I/O happens here and no
        blocking -- which is the point: in v1 `slots` is already correct the moment this returns,
        and the only thing the wait afterwards provides is DATA readiness. v2 exploits that; v1
        does not.
        """
        slot_of: dict[int, int] = {}
        to_load: list = []
        used: set[int] = set()
        for e in uniq:
            key = (layer, e)
            s = self.lru.get(key)
            if s is None:
                s = self.transient_map.get(key)
            else:
                self.lru.move_to_end(key)
                if self._afq and not prefill:
                    # DECODE hits only. A prefill chunk touches nearly every expert of a layer, so
                    # letting it write the counter would push every resident up by one per chunk
                    # and drown the decode signal the policy was fitted on.
                    self._clock += 1
                    self._afq_touch(key, s)
            if s is not None:
                slot_of[e] = s
                used.add(s)
                self.hits += 1
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
        return slot_of, to_load

    def forget(self, key: tuple, slot: int) -> None:
        """Un-map a key whose load did not complete: the slot holds a torn read.

        Leaving it mapped would make the next reserve() count it as a HIT and compute with whatever
        partial bytes landed -- silent, and permanent for the life of the process.
        """
        if self.lru.get(key) == slot:
            del self.lru[key]
            self.free_lru.append(slot)
            if self._afq:
                self._afq_drop(key)
        if self.transient_map.get(key) == slot:
            del self.transient_map[key]
        if self.slot_key.get(slot) == key:
            del self.slot_key[slot]

    @property
    def hit_rate(self) -> float:
        tot = self.hits + self.misses + self.prefill_misses
        return self.hits / tot if tot else 0.0
