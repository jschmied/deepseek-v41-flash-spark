"""test_evict_policy.py -- DSV41_EVICT_POLICY: the default is untouched, age_over_freq is EXACT.

Eviction policy is numerically invisible. It decides which experts are RESIDENT, never which
weights are used -- `resolve()` fetches a miss before anybody reads the expert -- so the same
prompt produces the same logits under either policy and no quality gate can see a bug here. A
wrong victim shows up only as a worse hit rate, which is indistinguishable from a different prompt.
That is why these tests are the gate instead:

  (a) under `lru` the store must produce the SAME slot for the same access, the same eviction
      sequence and the same final LRU order as a from-scratch reimplementation of the shipped
      path, and must not write one byte of the new bookkeeping;
  (b) under `age_over_freq` the victim the bucket-head shortcut picks must equal a brute-force
      argmax of age/(1 + use count) over EVERY resident, at every one of a few thousand evictions
      -- that is the exactness claim the offline study makes (`~/ds41-queue/eviction_oracle.py`:
      "hits 299,853 fetches 63,449 : IDENTICAL" against the full scan), and it is the claim that
      justifies ranking the whole cache instead of the 32-64 coldest (3-5 % of the Belady gap at
      K=32/64, 31 % at K=all);
  (c) the count buckets stay consistent with the LRU across `_promote_transient`'s pointer swap
      and across re-loads of an evicted expert;
  (d) no slot is ever handed to two experts -- the invariant resolve() asserts.

`_load_into_slot` is stubbed exactly as in test_expert_store.py: no NVMe, no model, no GPU work
beyond the pinned staging buffers the constructor allocates.

Run:  python -m engine.test_evict_policy
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import threading
from collections import OrderedDict

# the module reads these at import time; a stale one from the serving env would open a real cache
for _v in ("DSV41_CB3_CACHE", "DSV41_ROUTE_LOG", "DSV41_UNSAFE_NO_COMPUTE_WAIT",
           "DSV41_EVICT_POLICY"):
    os.environ.pop(_v, None)
os.environ["DSV41_IO_THREADS"] = "2"

import torch

from engine import experts as E


class FakeArena:
    """Only what ExpertStore's constructor and our recorder touch."""

    def __init__(self, slots: int):
        self.slots = slots
        self.device = "cpu"


def make_store(policy: str, lru_slots: int = 24, transient_slots: int = 8, io_threads: int = 2,
               seed: int = 0xC0FFEE, load_delay: float = 0.002):
    d = tempfile.mkdtemp(prefix="dsv41-evict-test-")
    arena = FakeArena(transient_slots + lru_slots)
    st = E.ExpertStore(d, {"weight_map": {}}, arena, n_layers=4, transient_slots=transient_slots,
                       io_threads=io_threads, evict_policy=policy)
    assert st.lru_slots == lru_slots, (st.lru_slots, lru_slots)
    assert st.evict_policy == policy
    st.test_content = {}            # slot -> key, as the loads actually land
    st.test_lock = threading.Lock()
    st.test_evictions = []          # keys that left the LRU, in order
    rng = random.Random(seed)

    def fake_load(key, slot, prefix=None):
        with st.test_lock:
            delay = rng.random() * load_delay
            before = st.test_content.get(slot)
        if delay:
            threading.Event().wait(delay)
        with st.test_lock:
            now = st.test_content.get(slot)
            assert now == before, (
                f"slot {slot} was overwritten while {key}'s read was in flight: {before} -> {now}")
            st.test_content[slot] = key
        return slot

    st._load_into_slot = fake_load

    # record which key left the LRU on every reservation; at most one can, and `before - after` is
    # cheap at these sizes and does not care HOW the victim was chosen.
    real_slot_for = st._lru_slot_for

    def rec_slot_for(key, used=frozenset()):
        before = set(st.lru)
        slot = real_slot_for(key, used)
        st.test_evictions.extend(sorted(before - set(st.lru)))
        return slot

    st._lru_slot_for = rec_slot_for
    real_promote = st._promote_transient

    def rec_promote(key, slot, used):
        before = set(st.lru)
        ok = real_promote(key, slot, used)
        st.test_evictions.extend(sorted(before - set(st.lru)))
        return ok

    st._promote_transient = rec_promote
    return st


def ids(experts):
    return torch.tensor([list(experts)], dtype=torch.int32)


# ---------------------------------------------------------------- the workload

def route_stream(n_calls: int, seed: int, n_layers: int = 4, n_experts: int = 40, k: int = 6,
                 skew: float = 1.0):
    """A decode route stream with a per-layer hot set, which is the only shape that makes this
    policy testable: uniform routing puts every resident in the count-1 bucket and the bucket
    structure degenerates to plain LRU. Weights are 1/rank over a per-layer permutation, so counts
    spread over ~10 buckets at these sizes -- the same qualitative shape as the real trace, where
    the winning score needs both a wide count spread and a wide age spread to differ from LRU.
    """
    rng = random.Random(seed)
    order = {L: rng.sample(range(n_experts), n_experts) for L in range(n_layers)}
    w = [1.0 / (i + 1) ** skew for i in range(n_experts)]
    for i in range(n_calls):
        L = i % n_layers
        ex = set()
        while len(ex) < k:
            ex.add(order[L][rng.choices(range(n_experts), weights=w, k=1)[0]])
        yield L, sorted(ex)


# ---------------------------------------------------------------- (a) the default path

class RefLRU:
    """Today's policy, written from scratch against the shipped code path: an OrderedDict, hits
    move_to_end, a miss pops the oldest entry whose slot is not promised to this same call (the
    rest are parked and put back oldest-first), free slots come off the END of the free list."""

    def __init__(self, lru_slots: int):
        self.lru: OrderedDict = OrderedDict()
        self.free = list(range(lru_slots))
        self.evictions: list = []

    def warm(self, keys):
        for k in keys:
            self.lru[k] = self.free.pop()

    def resolve(self, layer: int, experts) -> dict:
        used: set = set()
        slot_of: dict = {}
        uniq = sorted(set(experts))          # resolve() routes np.unique(), i.e. sorted distinct
        for e in uniq:                       # pass 1: residents, reserved before any allocation
            s = self.lru.get((layer, e))
            if s is not None:
                self.lru.move_to_end((layer, e))
                slot_of[e] = s
                used.add(s)
        for e in uniq:                       # pass 2: misses
            if e in slot_of:
                continue
            if self.free:
                slot = self.free.pop()
            else:
                parked = []
                while True:
                    ok, s = self.lru.popitem(last=False)
                    if s not in used:
                        self.evictions.append(ok)
                        slot = s
                        break
                    parked.append((ok, s))
                for k2, s2 in reversed(parked):
                    self.lru[k2] = s2
                    self.lru.move_to_end(k2, last=False)
            self.lru[(layer, e)] = slot
            slot_of[e] = slot
            used.add(slot)
        return slot_of


def test_lru_default_is_unchanged():
    """`lru` must be the shipped path, not a re-derivation of it: same slots, same victims, same
    final order -- and none of the new bookkeeping may be written at all."""
    st = make_store("lru")
    ref = RefLRU(st.lru_slots)
    warm = [(0, e) for e in range(10)] + [(1, e) for e in range(6)]
    st.warm_start(warm, log=lambda *a: None)
    ref.warm(warm)
    n = 0
    for L, ex in route_stream(1500, seed=11):
        got = st.resolve(L, ids(ex), prefill=False).tolist()[0]
        slot_of = ref.resolve(L, ex)          # ONCE -- a second call is six more move_to_end()s
        want = [slot_of[e] for e in ex]
        assert got == want, f"call {n} layer {L} {ex}: slots {got} != reference {want}"
        assert list(st.lru.items()) == list(ref.lru.items()), (
            f"call {n}: LRU order diverged, so the next victim would differ")
        n += 1
    assert st.test_evictions == ref.evictions, (
        f"{sum(a != b for a, b in zip(st.test_evictions, ref.evictions))} of "
        f"{len(ref.evictions)} victims differ")
    assert list(st.lru.items()) == list(ref.lru.items()), "final LRU order differs"
    assert not st._buckets and not st._use_count and not st._last_acc, "lru mode wrote afq state"
    assert st.stats["evictions"] == st.stats["evict_cmps"] == 0
    print(f"  lru: {n} resolves, {len(ref.evictions)} evictions, same slot for every access, "
          f"same victim order, same final LRU, zero afq state  OK")


# ---------------------------------------------------------------- (b) exactness

def brute_victim(st, used, avoid: int = -1):
    """The answer the bucket-head shortcut has to reproduce: argmax over ALL residents of
    age/(1 + use count), ties to the older entry. Same (score, -last_acc) key as the study."""
    now = st._clock
    best, best_key = None, None
    for k, sl in st.lru.items():
        if sl in used or sl == avoid:
            continue
        key = ((now - st._last_acc[k]) / (1.0 + st._use_count[k]), -st._last_acc[k])
        if best is None or key > best_key:
            best, best_key = k, key
    return best


def check_exact_rational(st, victim, used, avoid: int = -1):
    """The float score could in principle round two different rationals together and hand the
    tie-break the wrong entry. Cross-multiply instead -- integers, no rounding: no eligible
    resident may have a STRICTLY larger age/(1+count) than the victim."""
    now = st._clock
    av, cv = now - st._last_acc[victim], 1 + st._use_count[victim]
    for k, sl in st.lru.items():
        if sl in used or sl == avoid:
            continue
        a, c = now - st._last_acc[k], 1 + st._use_count[k]
        assert a * cv <= av * c, (
            f"victim {victim} scores {av}/{cv} but {k} scores {a}/{c} -- float rounding picked a "
            f"non-maximal entry")


def instrument_exactness(st):
    """Wrap `_afq_victim` so every single eviction is checked against the brute-force argmax."""
    real = st._afq_victim
    st.test_checked = 0
    st.test_heads = []

    def checked(used, avoid=-1):
        c0 = st.stats["evict_cmps"]
        v = real(used, avoid)
        st.test_heads.append(st.stats["evict_cmps"] - c0)
        b = brute_victim(st, used, avoid)
        assert v == b, (f"bucket-head victim {v} != brute-force argmax {b} at clock {st._clock} "
                        f"({len(st.lru)} residents, {len(st._buckets)} buckets)")
        if v is not None:
            check_exact_rational(st, v, used, avoid)
        st.test_checked += 1
        return v

    st._afq_victim = checked
    return st


def run_afq(seed: int = 11, calls: int = 1500, lru_slots: int = 24, n_experts: int = 40,
            n_layers: int = 4, check: bool = True, load_delay: float = 0.002):
    st = make_store("age_over_freq", lru_slots=lru_slots, load_delay=load_delay)
    if check:
        instrument_exactness(st)
    st.warm_start([(0, e) for e in range(10)] + [(1, e) for e in range(6)], log=lambda *a: None)
    for L, ex in route_stream(calls, seed=seed, n_experts=n_experts, n_layers=n_layers):
        st.resolve(L, ids(ex), prefill=False)
    return st


def test_afq_victim_is_the_exact_argmax():
    """(b) the exactness claim, over a few thousand accesses, not a handful, at two cache sizes:
    one where nearly every resident has its own count bucket and one where they are shared."""
    for lru_slots, n_experts, calls in ((24, 40, 1500), (256, 200, 2000)):
        st = run_afq(lru_slots=lru_slots, n_experts=n_experts, calls=calls, load_delay=0.0)
        assert st.test_checked > 2000, f"only {st.test_checked} evictions -- workload too weak"
        heads = st.test_heads
        print(f"  age_over_freq @ {lru_slots} slots: {st.test_checked} evictions, every victim == "
              f"brute-force argmax over all {len(st.lru)} residents (exact rationals too), "
              f"{sum(heads) / len(heads):.1f} candidates/eviction, {len(st._buckets)} buckets  OK")


def test_comparisons_per_eviction_stay_sublinear():
    """The reason for the bucket structure at all: the score keeps improving up to ranking EVERY
    resident, and that must not cost a scan of every resident. The study measures 226 comparisons
    per eviction at 3,000 slots; this drives the same 3,000 with a trace-shaped workload (40 layers
    x 384 experts, 6 per call) and requires the same order of magnitude, not 3,000.
    """
    st = run_afq(lru_slots=3000, n_experts=384, n_layers=40, calls=8000, check=False,
                 load_delay=0.0, seed=5)
    st_cmps = st.stats["evict_cmps"] / max(1, st.stats["evictions"])
    assert st.stats["evictions"] > 10_000, st.stats["evictions"]
    assert st_cmps < 0.2 * len(st.lru), (
        f"{st_cmps:.0f} comparisons per eviction against {len(st.lru)} residents -- the bucket "
        f"shortcut has degenerated into a scan")
    print(f"  {st.stats['evictions']:,} evictions at {len(st.lru):,} residents: "
          f"{st_cmps:.0f} comparisons each ({len(st._buckets)} live count buckets, "
          f"{100 * st_cmps / len(st.lru):.1f} % of a full scan)  OK")


def test_afq_differs_from_lru():
    """A shortcut that quietly degenerated to LRU would pass (b) forever. It must not: the policy
    only pays for itself where it evicts something else."""
    b = make_store("lru", load_delay=0.0)
    b.warm_start([(0, e) for e in range(10)] + [(1, e) for e in range(6)], log=lambda *a_: None)
    for L, ex in route_stream(1500, seed=11):
        b.resolve(L, ids(ex), prefill=False)
    a = run_afq(load_delay=0.0)
    same = sum(x == y for x, y in zip(a.test_evictions, b.test_evictions))
    diff = len(b.test_evictions) - same
    assert diff > 0.05 * len(b.test_evictions), (
        f"age_over_freq picked the LRU victim {same}/{len(b.test_evictions)} times -- it has "
        f"collapsed to LRU and (b) proves nothing")
    print(f"  age_over_freq chose a different victim than lru at {diff}/{len(b.test_evictions)} "
          f"evictions ({100 * diff / len(b.test_evictions):.0f} %)  OK")


# ---------------------------------------------------------------- (c) bookkeeping consistency

def check_invariants(st, where: str):
    seen = {}
    for c, b in st._buckets.items():
        assert b, f"{where}: empty bucket {c} left behind -- the victim scan pays for it every time"
        for k, sl in b.items():
            assert k not in seen, f"{where}: {k} in buckets {seen[k]} and {c}"
            seen[k] = c
            assert st.lru.get(k) == sl, f"{where}: bucket says {k}->{sl}, LRU says {st.lru.get(k)}"
            assert st._use_count[k] == c, f"{where}: {k} counted {st._use_count[k]}, bucket {c}"
            assert k in st._last_acc, f"{where}: {k} has no last use"
    assert seen.keys() == st.lru.keys(), (
        f"{where}: {len(set(st.lru) - set(seen))} residents are in no bucket (never evictable) and "
        f"{len(set(seen) - set(st.lru))} bucket entries are not resident")
    # (d) one slot, one owner -- across both pools
    slots = list(st.lru.values())
    assert len(set(slots)) == len(slots), f"{where}: two LRU keys share a slot"
    assert not set(slots) & set(st.transient_ring), f"{where}: a slot is in the LRU and the ring"
    assert len(set(st.transient_ring)) == len(st.transient_ring), f"{where}: duplicate ring slot"


def test_buckets_survive_promotion_and_reload():
    """(c) prefill fills the ring, decode promotes out of it (a pointer swap that moves SLOTS
    between the two pools, never keys), and evicted experts come back with their old count."""
    st = make_store("age_over_freq", lru_slots=24, transient_slots=8)
    st.warm_start([(0, e) for e in range(4)], log=lambda *a: None)
    rng = random.Random(7)
    reloads = 0
    seen_evicted = set()
    for i in range(400):
        L = i % 4
        pf = (i % 5 == 0)
        ex = sorted({rng.randrange(20) for _ in range(rng.randint(3, 6))})
        before = set(st.lru)
        st.resolve(L, ids(ex), prefill=pf)
        if not pf:
            for e in ex:
                k = (L, e)
                if k in seen_evicted and k in st.lru and k not in before:
                    reloads += 1
                    assert st._use_count[k] >= 2, f"{k} came back with a reset count"
        seen_evicted |= (before - set(st.lru))
        check_invariants(st, f"call {i} (prefill={pf})")
    assert st.stats["promoted"] > 0, "no promotion happened -- the pointer swap was never exercised"
    assert reloads > 20, f"only {reloads} re-loads of an evicted expert -- count survival untested"
    print(f"  buckets consistent over 400 mixed resolves: {st.stats['promoted']} promotions, "
          f"{reloads} re-loads kept their use count, {len(st._buckets)} live buckets  OK")


def test_no_slot_handed_out_twice():
    """(d) the invariant resolve() asserts, under both policies and with deferred submission --
    the fake loader also fails if a slot is rewritten while a read is in flight."""
    for policy in ("lru", "age_over_freq"):
        st = make_store(policy, lru_slots=24, transient_slots=16)
        rng = random.Random(3)
        last = None
        for i in range(300):
            L = i % 4
            if last is not None and L != last:
                st.join_pending()
            ex = sorted({rng.randrange(30) for _ in range(rng.randint(4, 8))})
            slots = st.resolve(L, ids(ex), prefill=(i % 3 == 0), defer=True).tolist()[0]
            assert len(set(slots)) == len(slots), f"{policy}: slot collision {slots}"
            last = L
        st.join_pending()
        check_invariants(st, policy) if policy == "age_over_freq" else None
        owners = {}
        for sl, k in st.slot_key.items():
            assert sl not in owners, f"{policy}: slot {sl} has two owners"
            owners[sl] = k
    print(f"  no slot handed out twice under either policy (deferred, 300 resolves each)  OK")


# ---------------------------------------------------------------- the mutation check

def _broken_victim(self, used, avoid=-1):
    """MUTANT: the first count bucket's best head instead of the best of ALL bucket heads.

    This is the shortcut done wrong in the most plausible way -- it is still exact within one
    bucket, still returns an eligible resident, still cheap. Only the comparison ACROSS counts is
    gone. If test (b) still passes against this, test (b) is worthless.
    """
    now = self._clock
    fallback = None
    for c, b in self._buckets.items():
        for k, sl in b.items():
            if sl in used or sl == avoid:
                continue
            if fallback is None:
                fallback = k
            if c == next(iter(self._buckets)):
                self.stats["evictions"] += 1
                return k
            break
    self.stats["evictions"] += 1
    return fallback


def mutation_check():
    """Break the bucket-head shortcut on purpose and require test (b) to FAIL."""
    good = E.ExpertStore._afq_victim
    E.ExpertStore._afq_victim = _broken_victim
    try:
        run_afq(load_delay=0.0)
    except AssertionError as exc:
        msg = str(exc).splitlines()[0]
        print(f"  MUTANT (first bucket only) -> test (b) FAILS as it must: {msg[:110]}")
        return True
    finally:
        E.ExpertStore._afq_victim = good
    print("  MUTANT PASSED test (b) -- the exactness test does not test exactness")
    return False


if __name__ == "__main__":
    print("DSV41_EVICT_POLICY: default unchanged, age_over_freq exact:")
    fails = 0
    for fn in (test_lru_default_is_unchanged,
               test_afq_victim_is_the_exact_argmax,
               test_comparisons_per_eviction_stay_sublinear,
               test_afq_differs_from_lru,
               test_buckets_survive_promotion_and_reload,
               test_no_slot_handed_out_twice):
        try:
            fn()
        except Exception as exc:                      # noqa: BLE001 -- report all, then fail once
            fails += 1
            print(f"  FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    if not mutation_check():
        fails += 1
    print("ALL OK" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
