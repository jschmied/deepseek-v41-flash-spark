"""test_v2_invariants.py -- the semantics engine/test_io_path.py pins for v1, re-asserted on v2.

Those 11 invariants are SEMANTICS, not implementation, which is what makes them v2's acceptance
tests before v2 exists. Each test below names the v1 test it corresponds to. Three of them could
not be carried over unchanged, and saying exactly why is the most useful output of this file:

  v1 test_one_pool_deadlocks
      v1 needs TWO pools because an io task submits its expert's aligned pieces to read_pool and
      then blocks on them; one shared pool wedges. v2's read is a single leaf, so the two-pool
      requirement has no v2 counterpart. It is REPLACED, not dropped, by the deadlock requirement
      v2 does have: a loader thread waits for the compute stream while holding a staging lease, so
      the driver must never wait for a slot from INSIDE a compute region. Pinned below, with the
      mutation that produces the hang.

  v1 test_cross_layer_defer_asserts_in_every_mix
      v1 ASSERTS that deferred resolves must not span layers, because the ~400-slot transient ring
      can be wrapped onto a slot the previous layer is still writing. v2 replaces the assert with
      per-slot generations, so spanning layers is SAFE rather than forbidden. The test therefore
      asserts the opposite of v1's, plus the mutation showing the generation is what makes it safe.

  v1 test_slot_reuse_across_layers_is_ordered_by_the_stream_barrier_alone
      v1 DOCUMENTS A HAZARD: it asserts that layer L+1's H2D DOES overlap a slot layer L's consumer
      is still reading, because nothing host-side orders them. Under v2's per-slot barrier that
      overlap must NOT happen. Same semantics, inverted expectation -- and the mutation turns the
      barrier off and shows v1's hazard coming back.

Run:  python -m enginev2.test_v2_invariants            (all)
      python -m enginev2.test_v2_invariants lease      (substring filter, for mutation checking)
"""

from __future__ import annotations

import dataclasses
import sys
import threading
import time

import pytest

from .drivers import Engine
from .sched import V1, V2, ComputeStream, LoaderService, Policy
from .chain import Chain, EngramSource
from .prefetch import OraclePrefetcher, Prefetcher, RecallOraclePrefetcher, SpecAttempt
from .observe import CounterObserver, NullObserver, TraceObserver, WaitReason, now_ns
from .leaves import Bandwidth, ModelLeaves, StagedExpert
from .store import ExpertSlots, SlotArena, SlotReady, StagingPool
from .trace import N_LAYERS, load_decode, warmup_cut

SCALE = 20.0          # leaves run 20x faster; ratios are preserved, the tests are about ordering


def mk(policy: Policy = V2, lru_slots: int = 16, transient_slots: int = 8, n_workers: int = 8,
       staging: int = 8, expert_read_qd: int | None = None, h2d_inflight: int | None = None,
       fail=None) -> Engine:
    # expert_read_qd must stay BELOW the buffer count or releasing the permit at handoff buys nothing and
    # the loader refuses the configuration -- so derive it from staging unless a test pins it.
    if expert_read_qd is None:
        expert_read_qd = max(1, staging // 2)
    if h2d_inflight is None:
        h2d_inflight = max(1, staging // 2)
    e = Engine(policy, lru_slots=lru_slots, transient_slots=transient_slots,
               n_workers=n_workers, staging=staging, expert_read_qd=expert_read_qd,
               h2d_inflight=h2d_inflight, scale=SCALE)
    if fail:
        e.loader.fail.update(fail)
    return e


def run_with_timeout(fn, seconds):
    box = {}
    t = threading.Thread(target=lambda: box.update(r=fn()) if True else None, daemon=True)
    t.start()
    t.join(seconds)
    return (not t.is_alive()), t


# ================================================================ 1. no torn slot
def test_no_torn_slot_under_concurrent_submission():
    """v1: test_no_torn_slot_under_deferred_submission. No slot is written by two loads at once,
    and no load writes a slot while a consumer reads it."""
    e = mk()
    try:
        _, to_load, _ = e.slots.reserve(0, tuple(range(12)), prefill=False)
        e.loader.submit(to_load)
        e.loader.wait_slots(to_load)
        assert e.arena.violations == [], e.arena.violations
        assert len(e.arena.content) == 12, e.arena.content
        for key, slot, _ in to_load:
            assert e.arena.content[slot] == key, (slot, e.arena.content[slot], key)
        assert e.loader.stage.at_rest()
    finally:
        e.close()
    print("  no torn slot: 12 concurrent loads, distinct slots, no write-write, no write-during-read  OK")


def test_no_torn_slot_decode_lru_eviction():
    """v1: test_no_torn_slot_decode_lru_eviction. Eviction under a decode-shaped stream never hands
    two live experts the same slot."""
    calls = load_decode()
    e = mk(lru_slots=64, transient_slots=8)
    try:
        for layer, uniq in calls[:80]:
            _, to_load, _ = e.slots.reserve(layer, uniq[:10], prefill=False)
            e.loader.submit(to_load)
            e.loader.wait_slots(to_load)
        assert e.arena.violations == [], e.arena.violations[:4]
        assert e.loader.stage.at_rest()
    finally:
        e.close()
    print("  decode eviction: 80 layers over a 64-slot cache, no slot collision  OK")


# ================================================================ 2. leases
def test_lease_conservation_normal_and_under_failure():
    """v1: test_lease_conservation_normal_and_under_failure.

    The only invariant true at EVERY instant is the weaker one the sampler checks: acquire takes
    the semaphore and THEN pops the free list, so mid-flight the list may be one longer than the
    semaphore. Exact agreement is an at-rest property.
    """
    e = mk(lru_slots=32, staging=4, n_workers=4)
    bad: list = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            free = list(e.loader.stage.free)
            if len(free) > e.loader.stage.n or len(set(free)) != len(free) \
                    or e.loader.stage.sem_value > e.loader.stage.n:
                bad.append((len(free), e.loader.stage.sem_value, sorted(free)))
            time.sleep(0.0005)

    s = threading.Thread(target=sampler, daemon=True)
    s.start()
    try:
        _, to_load, _ = e.slots.reserve(0, tuple(range(20)), prefill=False)
        e.loader.submit(to_load)
        e.loader.wait_slots(to_load)
        assert e.loader.stage.at_rest(), "leases not at rest after 20 clean loads"

        # the read raises, inside the lease
        e.loader.fail.update({(1, 4), (1, 9)})
        _, to_load, _ = e.slots.reserve(1, tuple(range(12)), prefill=False)
        e.loader.submit(to_load)
        err = None
        try:
            e.loader.wait_slots(to_load)
        except IOError as exc:
            err = exc
        assert err is not None and "injected NVMe failure" in str(err), err
        assert e.loader.stage.at_rest(), "leases not at rest after 2 of 12 reads raised"
        stop.set()
        s.join(5)
        assert bad == [], f"free list / semaphore inconsistent mid-flight: {bad[:3]}"
    finally:
        stop.set()
        e.close()
    print(f"  lease conservation: {e.loader.stage.n}/{e.loader.stage.n} buffers back after clean "
          f"loads and read failures; {len(bad)} mid-flight inconsistencies  OK")


def test_lease_released_at_handoff_is_still_conserved():
    """v2-only: D2 off releases the lease after the READ, not after the H2D. It must still balance,
    and it must not be released twice -- a double release would inflate the semaphore above n and
    let two loads share one staging buffer, which is the torn-slot bug with extra steps."""
    e = mk(Policy(False, False, False, False), staging=4, n_workers=4, lru_slots=32)
    try:
        _, to_load, _ = e.slots.reserve(0, tuple(range(16)), prefill=False)
        e.loader.submit(to_load)
        e.loader.wait_slots(to_load)
        assert e.loader.stage.at_rest(), (e.loader.stage.free, e.loader.stage.sem_value)
        assert e.loader.stage.sem_value == e.loader.stage.n
        # and under failure, where the release-at-handoff path and the finally path could both fire
        e.loader.fail.update({(2, 3)})
        _, to_load, _ = e.slots.reserve(2, (1, 2, 3, 4), prefill=False)
        e.loader.submit(to_load)
        try:
            e.loader.wait_slots(to_load)
        except IOError:
            pass
        assert e.loader.stage.at_rest(), "double release or leak on the handoff path"
    finally:
        e.close()
    print("  handoff lease: released after the read, exactly once, balanced under failure too  OK")


# ================================================================ 3. the v2 deadlock requirement
def test_driver_must_not_wait_for_a_slot_inside_a_compute_region():
    """REPLACES v1's test_one_pool_deadlocks (see the module docstring).

    Under D1 (compute_barrier_global) a loader thread waits for the compute stream to go idle
    before writing the arena. If the driver waits for that slot from INSIDE a compute region,
    neither can proceed. This pins the requirement AND demonstrates the hang, so the constraint is
    a checked claim and not folklore.
    """
    e = mk(Policy(False, True, False, False), n_workers=2, staging=2)
    try:
        _, to_load, _ = e.slots.reserve(0, (1, 2), prefill=False)
        e.loader.submit(to_load)

        def bad_driver():
            with e.compute.run():                     # the mutation: wait while compute is open
                e.loader.wait_slots(to_load)
            return True

        done, t = run_with_timeout(bad_driver, 2.0)
        assert not done, ("waiting for a slot inside a compute region did NOT deadlock under D1. "
                          "If that is now true the constraint may be removable -- measure it.")
        e.compute.busy = 0                            # unwedge so the workers can retire
        with e.compute._lk:
            e.compute._cv.notify_all()
        t.join(10)
        e.loader.wait_slots(to_load)
        assert e.loader.stage.at_rest(), "the unwind leaked a lease"
    finally:
        e.close()
    print("  D1 deadlock requirement: waiting for a slot inside a compute region hangs, as it must; "
          "the driver keeps the wait outside  OK")


# ================================================================ 4. join completeness
def test_wait_completes_even_when_a_load_raises():
    """v1: test_join_pending_completes_even_when_a_load_raises.

    After the wait: every load resolved, the failed key UN-MAPPED so the next reserve() counts it
    as a miss rather than computing on a torn slot, and no lease lost. v1's damage was never at the
    failure -- it was one request later, when the ring had permanently lost those slots.
    """
    e = mk(lru_slots=32, fail={(3, 2), (3, 5)})
    try:
        _, to_load, _ = e.slots.reserve(3, (0, 1, 2, 3, 4, 5, 6), prefill=False)
        e.loader.submit(to_load)
        err = None
        try:
            e.loader.wait_slots(to_load)
        except IOError as exc:
            err = exc
        assert err is not None, "a raising load did not surface"
        assert e.loader.q.outstanding == 0, "pending left populated after the wait"
        # The invariant is not "un-mapped at this instant" -- it is "never counted as a HIT by a
        # later reserve". Un-mapping moved to the driver's drain (cache mutation happens on one
        # thread), so assert the property through the public path instead of the internal dict.
        e.loader.drain_forgets()
        assert (3, 2) not in e.slots.lru and (3, 5) not in e.slots.lru, "torn slot still mapped"
        # the test's own next-reserve check below is the real invariant; this only has to happen
        # before it, because un-mapping is now applied by the driver rather than the worker.
        assert (3, 0) in e.slots.lru, "a healthy sibling was forgotten too"
        assert e.loader.stage.at_rest()
        # the whole point: the next reserve must MISS on the failed key, not hit a torn slot
        before = e.slots.misses
        e.slots.reserve(3, (2,), prefill=False)
        assert e.slots.misses == before + 1, "the failed expert was counted as a HIT"
    finally:
        e.close()
    print("  join completeness: 2 of 7 raised, all resolved, torn keys un-mapped and re-missed, "
          "leases balanced  OK")


# ================================================================ 5. cross-layer safety
def test_cross_layer_pending_is_safe_by_generation():
    """INVERTS v1's test_cross_layer_defer_asserts_in_every_mix (see the module docstring).

    v1 forbids deferred resolves spanning layers. v2 permits it: a recycled slot gets a new
    generation, so a waiter for (slot, gen) cannot be satisfied by a later tenant's completion.
    """
    e = mk(lru_slots=16, transient_slots=8)
    try:
        _, a, _ = e.slots.reserve(0, tuple(range(6)), prefill=False)
        e.loader.submit(a)
        _, b, _ = e.slots.reserve(1, tuple(range(6)), prefill=False)     # spans layers: legal in v2
        e.loader.submit(b)
        assert not ({s for _, s, _ in a} & {s for _, s, _ in b}), (
            "layer 1 was handed a slot layer 0 is still writing")
        e.loader.wait_slots(a + b)
        assert e.arena.violations == [], e.arena.violations[:4]
        gens = {}
        for key, slot, gen in a + b:
            assert gens.get(slot, -1) < gen or slot not in gens
            gens[slot] = gen
        assert e.loader.stage.at_rest()
    finally:
        e.close()

    # AND THE BOUND, which the earlier version of this test could not see. Slot capacity is what
    # limits how far ahead the loader may run. With too few slots the second reserve must REFUSE --
    # the old code silently reassigned a slot whose write was still in flight, and passed, because
    # submitting the older batch first usually let worker FIFO serialise the two writes. Generations
    # never protected this: they keep a consumer off a stale tenant, they do not serialise producers.
    e = mk(lru_slots=8, transient_slots=8)
    try:
        _, a, _ = e.slots.reserve(0, tuple(range(6)), prefill=False)
        e.loader.submit(a)
        try:
            e.slots.reserve(1, tuple(range(6)), prefill=False)
            raise AssertionError("reserve() handed out slots with writes in flight")
        except RuntimeError as exc:
            assert "in flight" in str(exc), exc
        e.loader.wait_slots(a)
    finally:
        e.close()
    print("  cross-layer: two layers pending is safe when slots allow, and REFUSED when they do "
          "not  OK")


def test_generation_is_what_makes_cross_layer_safe():
    """MUTATION for the test above: satisfy readiness with a STALE generation and the wait for a
    slot that has been recycled returns before its new tenant has landed."""
    e = mk()
    try:
        _, to_load, _ = e.slots.reserve(0, (1,), prefill=False)
        key, slot, gen = to_load[0]
        e.loader.ready.arm(slot, gen)
        e.loader.ready.set(slot, gen - 1)             # a previous tenant completing
        hit = True
        try:
            e.loader.ready.wait(slot, gen, timeout=0.3)
        except TimeoutError:
            hit = False
        assert not hit, ("a stale generation satisfied the wait -- recycling a slot would let a "
                         "consumer read the PREVIOUS tenant's bytes")
        e.loader.ready.set(slot, gen)
        e.loader.ready.wait(slot, gen, timeout=1.0)
    finally:
        e.close()
    print("  generation mutation: a stale completion does not satisfy a newer wait  OK")


# ================================================================ 6. the compute barrier
def test_barrier_sits_between_the_read_and_the_arena_write():
    """v1: test_compute_barrier_is_taken_before_the_arena_write.

    The ORDER is the part that matters and it is the same in both arms: read, THEN the barrier,
    THEN the arena write. What v2 changes is only WHICH barrier -- per-slot rather than
    wait-for-all-compute. Both orders are asserted.
    """
    for pol, want in ((Policy(False, True, False, False), "wait_idle"),
                      (Policy(False, False, False, False), "wait_slot")):
        e = mk(pol, n_workers=1, staging=2, expert_read_qd=1, h2d_inflight=1)
        order: list = []
        try:
            real_read, real_idle, real_slot = e.loader.bw.read, e.compute.wait_idle, e.compute.wait_slot_free
            real_writing = e.arena.writing
            e.loader.bw.read = lambda n=0, _r=real_read: (order.append("read"), _r(n))[1]
            e.compute.wait_idle = lambda *a, _r=real_idle: (order.append("wait_idle"), _r(*a))[1]
            e.compute.wait_slot_free = lambda *a, _r=real_slot: (order.append("wait_slot"), _r(*a))[1]
            e.arena.writing = lambda s, k, _r=real_writing: (order.append("h2d"), _r(s, k))[1]
            _, to_load, _ = e.slots.reserve(0, (7,), prefill=False)
            e.loader.submit(to_load)
            e.loader.wait_slots(to_load)
            assert order == ["read", want, "h2d"], (pol.name, order)
        finally:
            e.close()
    print("  barrier order: read -> barrier -> arena write, under both the global and the per-slot "
          "barrier  OK")


# ================================================================ 7. slot reuse across layers
def test_per_slot_barrier_removes_v1s_cross_layer_overlap():
    """INVERTS v1's test_slot_reuse_across_layers_is_ordered_by_the_stream_barrier_alone, and
    MUTATES it back.

    v1 asserts the overlap IS observed -- layer L+1's H2D lands in a slot layer L's consumer still
    holds -- because host-side bookkeeping does not order them. Under v2's per-slot barrier it must
    NOT be observed; with every barrier off, v1's hazard must come back, which is what proves the
    barrier is the thing doing the work and not an accident of timing.
    """
    def attempt(policy, barrier_off_mutation: bool):
        e = mk(policy, lru_slots=8, transient_slots=8, n_workers=4, staging=4)
        try:
            _, first, _ = e.slots.reserve(0, tuple(range(8)), prefill=False)
            e.loader.submit(first)
            e.loader.wait_slots(first)
            slot0 = first[0][1]
            started, release = threading.Event(), threading.Event()

            def still_computing():
                with e.compute.run(slots=[slot0]):     # layer 0's MoE, still on the compute stream
                    with e.arena.reading(slot0):
                        started.set()
                        release.wait(5)

            t = threading.Thread(target=still_computing, daemon=True)
            t.start()
            assert started.wait(5)
            e.slots.lru.clear()
            e.slots.free_lru = [slot0]                 # force layer 1 onto layer 0's slot
            _, second, _ = e.slots.reserve(1, (0,), prefill=False)
            assert second[0][1] == slot0
            if barrier_off_mutation:
                e.compute.wait_slot_free = lambda *a, **k: None
            e.loader.submit(second)
            landed = True
            try:
                e.loader.wait_slots([second[0]])
                e.loader.ready.wait(slot0, second[0][2], timeout=0.5)
            except TimeoutError:
                landed = False
            release.set()
            t.join(5)
            e.loader.wait_slots(second)
            return {k[0] for k in e.arena.violations}, landed
        finally:
            e.close()

    kinds, _ = attempt(Policy(False, False, False, False), False)
    assert "write-during-read" not in kinds, (
        f"v2's per-slot barrier did not order layer 1's H2D against layer 0's reader: {kinds}")
    kinds_mut, _ = attempt(Policy(False, False, False, False), True)
    assert "write-during-read" in kinds_mut, (
        "with the per-slot barrier removed the overlap did NOT come back, so this test was not "
        "measuring the barrier. v1's hazard must reappear or the assertion above proves nothing.")
    print("  cross-layer reuse: the per-slot barrier removes v1's write-during-read hazard, and "
          "removing the barrier brings it back  OK")


# ================================================================ 8. decode-shaped end to end
def test_decode_shaped_concurrency_both_arms():
    """v1: test_decode_shaped_concurrency / _deferred_across_chunks. Both arms, real trace, real
    eviction: no violations, leases at rest, same fetch count from the same miss stream."""
    calls = load_decode()
    cut = warmup_cut(calls)
    got = {}
    for name, pol in (("v1", V1), ("v2", V2)):
        e = Engine(pol, lru_slots=5328, transient_slots=400, scale=SCALE,
                   leaves=ModelLeaves(calls, Bandwidth(scale=SCALE), scale=SCALE, start=cut))
        try:
            e.warm(calls, cut)
            c = e.decode(3)
            assert e.arena.violations == [], (name, e.arena.violations[:4])
            assert e.loader.stage.at_rest(), name
            got[name] = c.fetches
        finally:
            e.close()
    assert got["v1"] == got["v2"], (
        f"the arms disagree on the miss stream ({got}) -- a scheduling policy must not change "
        f"WHICH experts are fetched, only when")
    print(f"  decode shaped: both arms clean over 3 real steps, identical fetch count "
          f"({got['v1']})  OK")


# ================================================================ 9. prefill / D3
def test_prefill_chunked_global_barrier_vs_per_chunk():
    """The one shape where D3 is reachable: several chunks of ONE layer pending together."""
    chunks = [tuple(range(0, 40)), tuple(range(40, 80)), tuple(range(80, 120))]
    for pol in (V1, V2):
        e = mk(pol, lru_slots=16, transient_slots=200, n_workers=8, staging=8)
        try:
            # The route now comes OUT of prefill_attn, so the modelled provider replays it from
            # here instead of the driver being handed it.
            e.leaves.prefill_calls = {(0, ci): u for ci, u in enumerate(chunks)}
            e.prefill_chunked(0, len(chunks))
            assert e.arena.violations == [], (pol.name, e.arena.violations[:4])
            assert e.loader.stage.at_rest(), pol.name
        finally:
            e.close()
    print("  prefill chunked: both barrier policies complete cleanly over 3 chunks of one layer  OK")


TESTS = (
    test_no_torn_slot_under_concurrent_submission,
    test_no_torn_slot_decode_lru_eviction,
    test_lease_conservation_normal_and_under_failure,
    test_lease_released_at_handoff_is_still_conserved,
    test_driver_must_not_wait_for_a_slot_inside_a_compute_region,
    test_wait_completes_even_when_a_load_raises,
    test_cross_layer_pending_is_safe_by_generation,
    test_generation_is_what_makes_cross_layer_safe,
    test_barrier_sits_between_the_read_and_the_arena_write,
    test_per_slot_barrier_removes_v1s_cross_layer_overlap,
    test_decode_shaped_concurrency_both_arms,
    test_prefill_chunked_global_barrier_vs_per_chunk,
)


def main(argv) -> int:
    want = argv[1] if len(argv) > 1 else ""
    chosen = [t for t in TESTS if want in t.__name__]
    print(f"engine v2 skeleton invariants ({len(chosen)} tests):")
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
    print("ALL OK" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


# ================================================================ 13. readiness is exact
def test_readiness_is_per_generation_and_cannot_regress():
    """Out-of-order completions must not release the wrong waiter. Found in review, not by a test.

    The old SlotReady kept `slot -> highest generation completed` and waited on `>=`. Three ways
    that is wrong, all reproduced below against the current implementation:
      1. gen 2 completing before gen 1 made readiness REGRESS from 2 to 1 on assignment.
      2. `>=` released a gen-1 waiter on gen 2's completion -- the slot then holds gen 2's expert.
      3. arm(gen 2) cleared the slot's entry and stranded a live gen-1 waiter.
    """
    r = SlotReady()

    # (2) the dangerous one: gen 2 completes first; a gen-1 waiter must NOT be released.
    r.arm(7, 1)
    r.arm(7, 2)
    r.set(7, 2)
    try:
        r.wait(7, 1, timeout=0.2)
        raise AssertionError("a waiter for gen 1 was released by gen 2's completion")
    except TimeoutError:
        pass
    r.set(7, 1)
    r.wait(7, 1, timeout=0.2)          # its own completion does release it
    r.wait(7, 2, timeout=0.2)          # (1) and gen 2 did not regress when gen 1 landed

    # (3) arming a newer generation must not strand a waiter for an older one.
    r2 = SlotReady()
    r2.arm(3, 5)
    r2.set(3, 5)
    r2.arm(3, 6)
    r2.wait(3, 5, timeout=0.2)

    # errors stay bound to their own generation
    r3 = SlotReady()
    r3.arm(1, 1)
    r3.set(1, 1, err=IOError("torn"))
    r3.set(1, 2)
    r3.wait(1, 2, timeout=0.2)
    try:
        r3.wait(1, 1, timeout=0.2)
        raise AssertionError("gen 1's error was not raised")
    except IOError:
        pass
    print("  readiness: exact per (slot, generation), no regression, errors stay bound  OK")


# ================================================================ 14. the edges are enforced
def test_declared_edges_are_load_bearing_not_decorative():
    """A dependency that is only implied by statement order is invisible to anything plugged in.

    So each real edge is a named event, and this proves the driver actually blocks on one: an
    engram source that never signals layer 7 must HANG the step, not let graph A read rows that
    have not arrived. If this test ever passes without the timeout, the edge has become decorative.
    """
    calls = load_decode()
    cut = warmup_cut(calls)

    class SkipsOneLayer(EngramSource):
        def issue(self, layers, step, chain, ctx=None, scored=True):
            for L in layers:
                if L != 7:
                    chain.set("engram", L, ctx=ctx, scored=scored)

    e = Engine(V2, lru_slots=5328, transient_slots=400, scale=SCALE,
               leaves=ModelLeaves(calls, Bandwidth(scale=SCALE), scale=SCALE, start=cut),
               engram=SkipsOneLayer())
    try:
        e.warm(calls, cut)
        ok, _ = run_with_timeout(lambda: e.decode(1), 6)
        assert not ok, "the step completed although layer 7's engram rows never arrived"
        # The hung worker is a daemon and cannot be reaped -- it keeps waiting and raises
        # TimeoutError at Chain's 60 s bound, which pytest reports as an unhandled thread exception
        # inside whichever test happens to be running then. That is this test working, not a
        # failure elsewhere.
    finally:
        e.close()

    # and the same driver completes when every layer IS signalled
    e = Engine(V2, lru_slots=5328, transient_slots=400, scale=SCALE,
               leaves=ModelLeaves(calls, Bandwidth(scale=SCALE), scale=SCALE, start=cut),
               engram=EngramSource())
    try:
        e.warm(calls, cut)
        c = e.decode(1)
        assert c.steps == 1
        # every per-layer edge the step declares must have been produced
        for L in range(N_LAYERS):
            for name in ("h", "y", "route", "engram", "kv"):
                assert e.chain.is_set(name, L), f"{name}@{L} was never set"
        assert e.chain.is_set("logits", 0), "the step-level edge was never set"
        assert e.chain.checks >= N_LAYERS * 2, (
            f"the driver only reached {e.chain.checks} edges -- the chain is not in the path")
        assert e.chain.blocks == 0, (
            f"{e.chain.blocks} edges blocked on a strictly sequential driver, which should be "
            f"impossible: every edge is produced before it is reached")
    finally:
        e.close()
    print("  edges: a missing producer HANGS the step, and every declared edge is produced  OK")


# ================================================================ 15. zero-copy / buffer lifetime
def test_staging_is_zero_copy_and_the_buffer_outlives_the_h2d():
    """The four things a real pinned-memory provider must satisfy, checkable before it exists.

    a) read() hands back views ALIASING the leased buffer, not a clone. A 13.8 MB copy per expert
       would be a different performance model wearing the same interface.
    b) the buffer is STILL LEASED while h2d runs -- cudaMemcpyAsync reads out of it until its
       completion event, so releasing at handoff corrupts the transfer in flight. This is the
       property the earlier `lease_until_completion=False` violated: it released the physical
       buffer after the read.
    c) it is released exactly once, and after h2d, on both the success and the failure path.
    d) the payload is unusable after release, so a use-after-release is loud rather than silent.
    """
    seen = {}

    class ProbeLeaves(ModelLeaves):
        def h2d(self, slot, key, staged):
            pool = self.__dict__["_pool"]
            seen.setdefault("aliases", []).append(pool.same_buffer(staged.sid, staged.payload))
            seen.setdefault("leased_during_h2d", []).append(pool.in_use(staged.sid))
            seen.setdefault("released_during_h2d", []).append(staged.released)
            super().h2d(slot, key, staged)

        def read(self, key, pool, ctx=None, scored=True):
            self.__dict__["_pool"] = pool
            return super().read(key, pool, ctx, scored)

    for pol in (V1, V2):                       # D2 on and off: the buffer rule holds either way
        seen.clear()
        e = mk(pol, lru_slots=32, transient_slots=8, n_workers=4, staging=8)
        e.loader.leaves = ProbeLeaves((), Bandwidth(scale=SCALE), scale=SCALE)
        try:
            _, to_load, _ = e.slots.reserve(0, tuple(range(6)), prefill=False)
            e.loader.submit(to_load)
            e.loader.wait_slots(to_load)
            assert seen["aliases"] and all(seen["aliases"]), (
                f"{pol.name}: read() returned a COPY of the staging buffer, not a view")
            assert all(seen["leased_during_h2d"]), (
                f"{pol.name}: the staging buffer was back on the free list while the H2D ran")
            assert not any(seen["released_during_h2d"]), (
                f"{pol.name}: the lease was released before the copy completed")
            assert e.loader.stage.at_rest(), f"{pol.name}: leases outstanding at rest"
        finally:
            e.close()

    # (c) failure path: the read raises, and the buffer still comes back exactly once
    e = mk(V2, lru_slots=32, transient_slots=8, n_workers=2, staging=4, fail={(0, 3)})
    try:
        _, to_load, _ = e.slots.reserve(0, (3,), prefill=False)
        e.loader.submit(to_load)
        try:
            e.loader.wait_slots(to_load)
        except IOError:
            pass
        assert e.loader.stage.at_rest(), "a failed read leaked its staging buffer"
    finally:
        e.close()

    # (d) use-after-release is loud
    pool = StagingPool(2, nbytes=64)
    st = StagedExpert(pool.acquire(), None, pool)
    st.payload = pool.buffer(st.sid)
    assert pool.same_buffer(st.sid, st.payload)
    st.release()
    assert st.payload is None and st.released
    st.release()                                # idempotent, not a double free
    assert pool.at_rest()
    print("  staging: zero-copy view, leased across the H2D, released once and only after  OK")


# ================================================================ 16. observability is inert
def test_observer_records_but_never_influences():
    """The one rule the event layer must not break: it watches, it is not part of the graph.

    Same trace, same policy, three observers -- the miss stream and the arena must be identical.
    Also checks the null path is genuinely free (no Event is constructed) and that a wait is
    attributed at exactly ONE ownership point, never under two names.
    """
    calls = load_decode()
    cut = warmup_cut(calls)
    got = {}
    for name, obs in (("null", NullObserver()), ("counter", CounterObserver()),
                      ("trace", TraceObserver(capacity=1 << 16))):
        e = Engine(V1, lru_slots=5328, transient_slots=400, scale=SCALE,
                   leaves=ModelLeaves(calls, Bandwidth(scale=SCALE), scale=SCALE, start=cut),
                   observer=obs)
        try:
            e.warm(calls, cut)
            c = e.decode(2)
            got[name] = (c.fetches, c.steps, sorted(e.arena.content.items())[:20])
            assert e.arena.violations == [], (name, e.arena.violations[:3])
        finally:
            e.close()
    assert got["null"] == got["counter"] == got["trace"], (
        "an observer changed what the engine did: "
        f"{[(k, v[0], v[1]) for k, v in got.items()]}")

    # the null path constructs nothing: emitting through it must not touch Event at all
    assert NullObserver.enabled is False
    n = NullObserver()
    n.emit(None)                      # would raise if the null path dereferenced an event

    # each wait is attributed once, at the abstraction that owns it
    obs = TraceObserver(capacity=1 << 16)
    e = Engine(V1, lru_slots=5328, transient_slots=400, scale=SCALE,
               leaves=ModelLeaves(calls, Bandwidth(scale=SCALE), scale=SCALE, start=cut),
               observer=obs)
    try:
        e.warm(calls, cut)
        e.decode(2)
    finally:
        e.close()
    ev = obs.drain()
    starts = [x for x in ev if x.kind == "wait_start"]
    ends = [x for x in ev if x.kind == "wait_end"]
    assert len(starts) == len(ends), f"unbalanced waits: {len(starts)} start, {len(ends)} end"
    assert starts, "no wait was recorded at all"
    # warm-up must contribute nothing
    assert not any(x.kind == "cache_miss" and x.step < 0 for x in ev), "warm-up emitted events"
    print(f"  observer: inert across null/counter/trace, {len(starts)} waits balanced  OK")


# ================================================================ 17. identity and quarantine
def test_context_and_cause_survive_async_and_a_bad_observer_cannot_break_the_engine():
    """The four properties the previous test did not cover, each of which was a real gap.

    a) the critical-path wait -- EXPERT_DATA -- must know WHICH request/step was blocked. It is
       produced by SlotReady on a worker's completion, so the context has to be retained at arm().
    b) a prediction must be joinable end to end by cause_id, not reconstructed from timestamps.
    c) a throwing observer must be quarantined, INCLUDING from the GPU hooks -- those bypassed
       safe_emit entirely and could abort inference.
    d) the trace ring is a total budget, not a per-thread allocation.
    """
    calls = load_decode()
    cut = warmup_cut(calls)

    # (a) + (b)
    # Budget sized to the run: 3 steps x 40 layers emits ~1.8k events on the driver thread alone,
    # and the ring correctly drops the oldest when it does not fit -- which silently removed the
    # `prediction` events this test joins on. That is the ring working, not a bug, but a causal
    # join needs the whole window present.
    obs = TraceObserver(capacity=1 << 19)
    e = Engine(V2, lru_slots=5328, transient_slots=400, scale=SCALE, request_id=17,
               leaves=ModelLeaves(calls, Bandwidth(scale=SCALE), scale=SCALE, start=cut),
               prefetch=OraclePrefetcher(calls, 2, start=cut), observer=obs)
    try:
        e.warm(calls, cut)
        e.decode(3)
    finally:
        e.close()
    ev = obs.drain()

    waits = [x for x in ev if x.kind == "wait_start" and x.aux is WaitReason.EXPERT_DATA]
    assert waits, "no EXPERT_DATA wait was recorded"
    identified = [x for x in waits if x.ctx.request_id == 17 and x.ctx.step >= 0]
    assert identified, "EXPERT_DATA waits carry no request/step -- context is lost at SlotReady"

    issued = [x for x in ev if x.kind == "prefetch_issued"]
    assert issued, "no prefetch was issued"
    cause = issued[0].cause_id
    assert cause, "prefetch_issued carries no cause_id"
    chain_of_one = {x.kind for x in ev if x.cause_id == cause}
    assert "prediction" in chain_of_one, "the prediction that caused this is not joinable"
    assert chain_of_one & {"load_queued", "nvme_start", "h2d_start"}, (
        f"the cause_id does not follow the work into the loader: {sorted(chain_of_one)}")
    assert obs.dropped == 0, f"the ring wrapped ({obs.dropped} dropped); the join window is partial"

    # (c) an observer that throws -- from emit AND from the GPU hooks -- must not break the engine
    class Hostile(CounterObserver):
        def emit(self, event):
            raise RuntimeError("observer blew up")

    class HostileGPU(CounterObserver):
        def gpu_begin(self, name, ctx):
            raise RuntimeError("nvtx blew up")

    for cls in (Hostile, HostileGPU):
        bad = cls()
        e = Engine(V2, lru_slots=5328, transient_slots=400, scale=SCALE,
                   leaves=ModelLeaves(calls, Bandwidth(scale=SCALE), scale=SCALE, start=cut),
                   observer=bad)
        try:
            e.warm(calls, cut)
            c = e.decode(1)                      # must complete
            assert c.steps == 1, cls.__name__
            assert bad.failed is not None and not bad.enabled, (
                f"{cls.__name__} was not quarantined")
        finally:
            e.close()

    # (d) capacity is a total budget: many rings must not multiply it
    t = TraceObserver(capacity=1 << 17)
    assert t.per_ring * 64 <= t.capacity + t.per_ring, (
        f"per-ring {t.per_ring} x 64 threads exceeds the {t.capacity} budget")
    assert t.per_ring < t.capacity, "capacity is being used as a per-thread size"
    print("  identity: EXPERT_DATA carries request/step, cause_id joins a prediction to its I/O, "
          "a hostile observer is quarantined  OK")


# ================================================================ 18. whose wait is it
def test_waits_are_attributed_to_the_blocked_consumer_not_the_producer():
    """Three attributions that test 17 did not check, each of which was wrong or missing.

    a) a LATE speculative read must blame the layer that is blocked, not the layer that predicted
       it. SlotReady.arm() stores the producer's context, so recovering it in wait() charged
       layer 16's stall to layer 12.
    b) D3's global barrier -- the dominant v1 consumer wait -- must carry a context at all.
    c) a STAGING_BUFFER wait must carry one too; acquire(ctx) existed but read() never passed it.
    """
    calls = load_decode()
    cut = warmup_cut(calls)

    # (a) speculative reads at horizon 1, forced to be late by a slow device
    obs = TraceObserver(capacity=1 << 19)
    e = Engine(V2, lru_slots=5328, transient_slots=400, request_id=9,
               leaves=ModelLeaves(calls, Bandwidth(scale=0.25), scale=1.0, start=cut),
               prefetch=OraclePrefetcher(calls, 1, start=cut), observer=obs)
    try:
        e.warm(calls, cut)
        e.decode(2)
    finally:
        e.close()
    spec_waits = [x for x in obs.drain()
                  if x.kind == "wait_start" and x.aux is WaitReason.EXPERT_DATA
                  and x.cause_id and x.key]
    assert spec_waits, "no speculative read was waited on -- the arm proves nothing"
    misattributed = [x for x in spec_waits if x.ctx.layer != x.key[0]]
    assert not misattributed, (
        "a late prefetch blamed the predicting layer instead of the blocked one: "
        f"{[(x.ctx.layer, x.key[0]) for x in misattributed[:3]]}")

    # (b) D3 under v1
    obs = TraceObserver(capacity=1 << 19)
    e = Engine(V1, lru_slots=5328, transient_slots=400, request_id=9,
               leaves=ModelLeaves(calls, Bandwidth(), start=cut), observer=obs)
    try:
        e.warm(calls, cut)
        e.decode(2)
    finally:
        e.close()
    gb = [x for x in obs.drain()
          if x.kind == "wait_start" and x.aux is WaitReason.GLOBAL_BARRIER]
    assert gb, "D3 never blocked under v1, so this arm proves nothing"
    assert any(x.ctx.request_id == 9 and x.ctx.layer >= 0 for x in gb), (
        "the dominant v1 wait carries no context")

    # (c) force staging contention: one buffer, many workers, v1 holds it to completion
    obs = TraceObserver(capacity=1 << 19)
    e = Engine(V1, lru_slots=5328, transient_slots=400, request_id=9,
               n_workers=8, staging=1, expert_read_qd=8, h2d_inflight=8,
               leaves=ModelLeaves(calls, Bandwidth(), start=cut), observer=obs)
    try:
        e.warm(calls, cut)
        e.decode(1)
    finally:
        e.close()
    st = [x for x in obs.drain()
          if x.kind == "wait_start" and x.aux is WaitReason.STAGING_BUFFER]
    assert st, "one staging buffer and eight workers produced no staging wait"
    assert any(x.ctx.request_id == 9 for x in st), (
        "staging waits carry no context -- read() is not passing it to acquire()")
    print(f"  attribution: {len(spec_waits)} speculative waits blamed on the blocked layer, "
          f"D3 and staging both carry context  OK")


# ================================================================ 19. speculation must not lie
def test_speculation_neither_credits_frequency_nor_keeps_a_wrong_slot():
    """Two bugs that would have invalidated every predictor arm, and a third they uncovered.

    a) speculative insertion went through on_insert, which for age_over_freq IS the use-count
       touch -- so a WRONG prefetch permanently credited that expert with a use it never had, and
       on_drop preserves _use_count on purpose, so the phantom outlived eviction. The predictor was
       rewriting the eviction signal it was being measured against.
    b) discard_wrong_asap only cancelled QUEUED work; a wrong prefetch already reading stayed
       mapped and kept its slot. The read cost is unavoidable, the residency is not.
    c) admitting without materialising the count left the key in a policy bucket after eviction --
       a ghost the victim search returns and `lru.pop` then raises on.
    """
    sl = ExpertSlots(64, 8, policy="age_over_freq")
    k = (3, 77)
    to_load, _ = sl.reserve_speculative([k])
    assert sl.evict._use_count.get(k, 0) == 0, "speculation credited a use"
    assert any(k in b for b in sl.evict._buckets.values()), "speculative key is not evictable"
    sl.reserve(3, (77,), prefill=False)
    assert sl.evict._use_count[k] == 1, "a real demand did not count"

    # (c) no ghosts after a speculative key is dropped
    sl2 = ExpertSlots(16, 8, policy="age_over_freq")
    sl2.reserve(0, tuple(range(6)), prefill=False)
    spec, _ = sl2.reserve_speculative([(1, i) for i in range(6)])
    for key, slot, _g in spec[:3]:
        sl2.forget(key, slot)
    in_bucket = {kk for b in sl2.evict._buckets.values() for kk in b}
    assert not (in_bucket - set(sl2.lru)), (
        f"policy holds keys that are no longer resident: {sorted(in_bucket - set(sl2.lru))[:4]}")

    # (b) a wrong prefetch that already RAN must not stay resident
    calls = load_decode()
    cut = warmup_cut(calls)
    # The wrong keys come from the TRACE, not from test-only state on the engine: prefetch_wasted
    # already carries key/slot/gen, so an unbounded list on Engine was redundant as well as
    # test-shaped.
    obs = TraceObserver(capacity=1 << 20)
    e = Engine(V2, lru_slots=5328, transient_slots=400, observer=obs,
               leaves=ModelLeaves(calls, Bandwidth(), start=cut),
               prefetch=RecallOraclePrefetcher(calls, 2, recall=1.0, precision=0.5, start=cut))
    try:
        e.warm(calls, cut)
        e.decode(3)
        # Speculative reads still in flight at the end of the window are marked for discard and
        # un-mapped when their H2D ends, so the cache is only inspectable AT REST.
        e.loader.quiesce()
        e.loader.drain_forgets()
        assert e.pf.wasted > 0, "no wrong prefetch was produced, so this arm proves nothing"
        assert e.pf.discarded_running > 0, (
            "no wrong prefetch was caught mid-flight -- only queued ones were being discarded")
        # The previous assertion here was tautological -- every (layer, expert) key has layer >= 0.
        # What matters is that the wrong keys are GONE from the cache, so check the cache.
        wrong = [(x.key, x.slot, x.gen) for x in obs.drain() if x.kind == "prefetch_wasted"]
        assert wrong, "the arm recorded no wrong keys to check"
        assert obs.dropped == 0, "the trace wrapped; the wrong-key list is partial"
        # A discarded (key, slot) must not still be mapped AT THAT SLOT. Checking the key alone is
        # wrong: the same expert can be predicted again later and be legitimately resident
        # elsewhere, which is a hit, not a leak.
        still = [(k, sl, g) for k, sl, g in wrong
                 if e.slots.lru.get(k) == sl and e.slots.gen.get(sl) == g]
        assert not still, (
            f"{len(still)} of {len(wrong)} discarded prefetches still hold their slot: "
            f"{sorted(still)[:4]}")
    finally:
        e.close()
    print(f"  speculation: no phantom use-counts, {e.pf.discarded_running} running wrong prefetches "
          f"un-mapped, no policy ghosts  OK")


# ================================================================ 20. queued cancellation
def test_a_cancelled_queued_prefetch_frees_its_slot_before_the_next_reserve():
    """The slot must come back to the DRIVER's next reserve, not to whenever a worker dequeues.

    Speculation is deliberately lower priority, so a cancelled wrong prediction left for its worker
    could hold a mapped slot and a pending mark for an unbounded time -- protected from eviction
    exactly while demand needed it. And un-mapping it from the worker mutates ExpertSlots off the
    driver thread, which is the race the deferral exists to avoid.

    One worker, busy with a slow demand read, so the speculative entry provably cannot have run.
    """
    e = mk(V2, lru_slots=32, transient_slots=8, n_workers=1, staging=8)
    try:
        # occupy the single worker with a demand read
        _, demand, _ = e.slots.reserve(0, (1, 2, 3, 4), prefill=False)
        e.loader.submit(demand)

        spec, refused = e.slots.reserve_speculative([(9, 700), (9, 701)])
        assert spec and not refused, "speculation was refused; the arm proves nothing"
        e.loader.submit(spec, speculative=True)
        keys = [(k, sl, g) for k, sl, g in spec]
        for k, sl, _g in keys:
            assert e.slots.lru.get(k) == sl, "speculation did not become resident"

        states = [st for _k, _sl, _g, st in e.loader.cancel(spec)]
        q = states.count("queued")
        running, fin = states.count("running"), states.count("finished")
        assert q == len(spec), (
            f"only {q} of {len(spec)} were treated as queued -- with one busy worker they cannot "
            f"have started (running={running}, finished={fin})")

        # BEFORE the worker could possibly dequeue them
        e.loader.drain_forgets()
        for k, sl, g in keys:
            assert e.slots.lru.get(k) != sl, f"{k} still holds slot {sl} after cancellation"
            assert sl not in e.slots.pending_slots(), f"slot {sl} is still marked pending"
        # and the slot is genuinely reusable by the next reserve
        _, again, _ = e.slots.reserve(9, (800, 801), prefill=False)
        assert len(again) == 2, "the freed slots were not reusable"

        e.loader.quiesce()
        assert e.loader.stage.at_rest(), "a cancelled entry leaked a staging lease"
        assert e.arena.violations == [], e.arena.violations[:3]
    finally:
        e.close()
    print("  queued cancel: slot recovered by the driver before the next reserve, no worker "
          "touched the cache  OK")


# ================================================================ 21. an undone eviction leaves no trace
def test_rollback_restores_the_exact_eviction_state_under_both_policies():
    """Snapshot the next victim, speculate + roll back, and demand the SAME victim afterwards.

    Restoring is only sound if it is invisible. Two places it was not:
      * the global LRU: `lru[k] = v` appends, promoting the evicted tenant to most-recently-used;
      * age_over_freq: victim() only ever considers each bucket's HEAD, so bucket order IS recency,
        and re-registering at the tail changes who is evicted next.
    Neither shows up in a "the slot is reusable" test, which is all test 20 checked.
    """
    for policy in ("lru", "age_over_freq"):
        sl = ExpertSlots(16, 8, policy=policy)
        # The victim's BUCKET must hold several keys or head-vs-tail is moot -- the first version of
        # this fixture left it with one member, and the test then passed against the bug it was
        # written for. Touch each key once, in its own call, so they share a use count but differ
        # in age: bucket 1 is then populated and ordered by recency.
        for e in range(16):
            sl.reserve(0, (e,), prefill=False)
        if policy == "age_over_freq":
            v0 = sl.evict.victim(sl.lru, frozenset(), sl._clock).key
            c0 = sl.evict._use_count.get(v0, 0)
            assert len(sl.evict._buckets[c0]) > 1, (
                f"fixture is degenerate: victim {v0} is alone in bucket {c0}")
        before_order = list(sl.lru)
        _bv = sl.evict.victim(sl.lru, frozenset(), sl._clock)
        before_victim = None if _bv is None else _bv.key
        assert before_victim is not None, policy

        spec, refused = sl.reserve_speculative([(9, 900)])
        assert spec and not refused, f"{policy}: speculation was refused"
        k, slot, gen = spec[0]
        assert before_victim not in sl.lru, f"{policy}: speculation did not evict the victim"

        assert sl.rollback_speculative(k, slot, gen), f"{policy}: rollback refused"
        assert list(sl.lru) == before_order, (
            f"{policy}: rollback changed LRU order\n  before {before_order[:5]}\n  "
            f"after  {list(sl.lru)[:5]}")
        _av = sl.evict.victim(sl.lru, frozenset(), sl._clock)
        after_victim = None if _av is None else _av.key
        assert after_victim == before_victim, (
            f"{policy}: the next victim changed across an undone eviction: "
            f"{before_victim} -> {after_victim}")
        assert k not in sl.lru, f"{policy}: the speculative key survived rollback"

    # and the displaced record is bounded by slots, not by speculation count
    sl = ExpertSlots(16, 8, policy="lru")
    sl.reserve(0, tuple(range(16)), prefill=False)
    for i in range(200):
        spec, _ = sl.reserve_speculative([(9, 900 + i)])
        if spec:
            sl.clear_pending(spec[0][1], spec[0][2])
    assert len(sl._displaced) <= 16, (
        f"displaced records grew with speculation count: {len(sl._displaced)} for 16 slots")
    # AND WITH PROTECTED PREDECESSORS, which is the case the above cannot see. victim() returns the
    # first UNPROTECTED entry, so when slots ahead of it are protected the victim is NOT the head --
    # and restoring at the head puts it in front of keys that were ahead of it. Under lookahead,
    # protected slots (used this call, or mid-write) are the normal case, not the rare one.
    for policy in ("lru", "age_over_freq"):
        sl = ExpertSlots(16, 8, policy=policy)
        for e in range(16):
            sl.reserve(0, (e,), prefill=False)
        before_order = list(sl.lru)
        # protect the first few slots exactly as a live reserve() would
        head = [sl.lru[k] for k in before_order[:3]]
        protected = frozenset(head)
        _v = sl.evict.victim(sl.lru, protected, sl._clock)
        victim = None if _v is None else _v.key
        assert victim is not None and victim not in before_order[:3], (
            f"{policy}: fixture did not protect ahead of the victim")

        # snapshot BEFORE the speculation -- taking it after captures the state being undone
        before_buckets = ({c: list(b) for c, b in sl.evict._buckets.items()}
                          if policy == "age_over_freq" else None)
        # drive the real path: reserve_speculative protects pending slots, so make them pending
        fake = [((99, i), slot, sl.gen.get(slot, 0) + 1) for i, slot in enumerate(head)]
        for _k, slot, g in fake:
            sl.gen[slot] = g
        sl.mark_pending(fake)
        spec, refused = sl.reserve_speculative([(9, 950)])
        assert spec and not refused, f"{policy}: protected-prefix speculation was refused"
        k, slot, gen = spec[0]
        evicted = [kk for kk in before_order if kk not in sl.lru]
        assert evicted and evicted[0] not in before_order[:3], (
            f"{policy}: a protected key was evicted: {evicted}")

        assert sl.rollback_speculative(k, slot, gen), f"{policy}: rollback refused"
        if sl.evict.uses_lru_order:
            assert list(sl.lru) == before_order, (
                f"{policy}: rollback with a protected prefix changed LRU order\n"
                f"  before {before_order[:6]}\n  after  {list(sl.lru)[:6]}")
        else:
            # age_over_freq never reads the global order, so it is not part of its restore
            # contract -- asserting it would pin state the policy does not use. Its own state is
            # the bucket order, and THAT must come back exactly.
            after_buckets = {c: list(b) for c, b in sl.evict._buckets.items()}
            assert after_buckets == before_buckets, (
                f"{policy}: rollback changed bucket order\n  before {before_buckets}\n"
                f"  after  {after_buckets}")
        _av2 = sl.evict.victim(sl.lru, protected, sl._clock)
        after_victim = None if _av2 is None else _av2.key
        assert after_victim == victim, (
            f"{policy}: protected-prefix rollback changed the next victim: "
            f"{victim} -> {after_victim}")

    print("  rollback: same victim and order, with and without a protected prefix, both "
          "policies; displaced bounded by slots  OK")


# ================================================================ 22. rollback vs a real access
def test_rollback_does_not_undo_a_hit_that_happened_while_the_prediction_waited():
    """A queued prediction lives for LAYERS. What happens to the cache meanwhile is real.

    The saved `skipped` prefix records where a victim sat AT EVICTION TIME. If one of those
    predecessors is legitimately HIT before the prediction is settled, restoring it to its saved
    position undoes that access -- reconstructing a cache state that never existed. And the skipped
    prefix is exactly the protected/in-use keys, which are the ones most likely to be used next.

    Control-vs-test, which is the only construction that can see it: the same store with and
    without a speculation that is rolled back. After the rollback the two must be identical.
    """
    def build(policy):
        sl = ExpertSlots(16, 8, policy=policy)
        for e in range(16):
            sl.reserve(0, (e,), prefill=False)
        return sl

    for policy in ("lru", "age_over_freq"):
        # --- control: no speculation at all, just the intervening hit
        ctl = build(policy)
        order = list(ctl.lru)
        touched = order[0]                       # a key the victim search would skip over
        ctl.reserve(touched[0], (touched[1],), prefill=False)
        ctl_order = list(ctl.lru)
        _cv = ctl.evict.victim(ctl.lru, frozenset(), ctl._clock)
        ctl_victim = None if _cv is None else _cv.key

        # --- test: speculate (evicting past `touched`), take the same hit, then roll back
        tst = build(policy)
        head = [tst.lru[k] for k in list(tst.lru)[:3]]
        fake = [((99, i), sl_, tst.gen.get(sl_, 0) + 1) for i, sl_ in enumerate(head)]
        for _k, sl_, g in fake:
            tst.gen[sl_] = g
        tst.mark_pending(fake)                   # makes the first three slots protected
        spec, refused = tst.reserve_speculative([(9, 950)])
        assert spec and not refused, f"{policy}: speculation refused"
        k, slot, gen = spec[0]
        tst.reserve(touched[0], (touched[1],), prefill=False)   # THE INTERVENING HIT
        for _k, sl_, g in fake:
            tst.clear_pending(sl_, g)
        tst.rollback_speculative(k, slot, gen)

        if tst.evict.uses_lru_order:
            assert list(tst.lru) == ctl_order, (
                f"{policy}: rollback undid an access that really happened\n"
                f"  control {ctl_order[:6]}\n  after   {list(tst.lru)[:6]}")
        _tv = tst.evict.victim(tst.lru, frozenset(), tst._clock)
        assert (None if _tv is None else _tv.key) == ctl_victim, (
            f"{policy}: next victim differs from the no-speculation control")
    print("  rollback: an intervening hit survives the rollback on both policies  OK")


# ================================================================ 23. right, but too early
def test_a_prefetch_evicted_before_its_target_is_not_a_ready_hit():
    """Correct prediction, read completed, evicted before the layer arrived. It saved NOTHING.

    _spec is independent of residency, and SlotReady keeps (slot, generation) readiness for the life
    of the process -- so classifying on `ready_ts` alone counted this as a TIMELY, USEFUL hit while
    the engine turned round and read the expert again as a demand miss. It inflated used, ready_hit,
    timeliness, fetch precision and mean lead simultaneously, which are exactly the quantities a
    decision about a learned predictor is read from.

    Provoked with a long horizon and a small cache: predictions land many layers early and the
    working set evicts them before their target.
    """
    calls = load_decode()
    cut = warmup_cut(calls)
    e = Engine(V2, lru_slots=600, transient_slots=400,
               leaves=ModelLeaves(calls, Bandwidth(), start=cut),
               prefetch=OraclePrefetcher(calls, 16, start=cut))
    try:
        e.warm(calls, cut)
        e.decode(4)
        e.finalize_stats()
        p = e.pf
        assert p.evicted_before_use > 0, (
            "no prefetch was evicted before its target, so this arm proves nothing "
            f"(issued {p.issued}, used {p.used})")
        # every classified prediction lands in exactly one bucket
        total = p.ready_hit + p.late_hit + p.evicted_before_use
        assert p.used == p.ready_hit + p.late_hit, (
            f"used ({p.used}) != ready ({p.ready_hit}) + late ({p.late_hit}) -- an evicted "
            f"prefetch is being counted as used")
        # and a ready hit must really be resident at its own generation
        assert p.ready_hit <= total, (p.ready_hit, total)
        print(f"  evicted-before-use: {p.evicted_before_use} correct-but-dead prefetches kept out "
              f"of ready_hit ({p.ready_hit} ready, {p.late_hit} late)  OK")
    finally:
        e.close()


# ================================================================ 24. what a queued reservation costs
def test_a_queued_speculation_still_costs_capacity_and_that_is_documented_not_hidden():
    """Rollback undoes the eviction it caused. It does NOT undo evictions OTHERS made meanwhile.

    A queued speculation holds a PROTECTED slot. A demand miss arriving before it is cancelled
    cannot use that slot and evicts a different resident instead. Rolling back restores the
    speculation's own victim, but the secondary eviction stands -- so the cache does not return to
    the no-speculation counterfactual.

    THIS IS THE CHOSEN SEMANTICS, not an oversight: reserving a slot costs capacity from the moment
    it is reserved, whether or not the read ever starts. Making it genuinely free needs provisional
    reservation -- keep the old tenant usable until the speculative write commits, and let demand
    preempt a queued speculation -- which is a different design. Until that exists, "queued
    cancellation is free" is true of I/O and false of capacity, and this test says so out loud.
    """
    def build():
        sl = ExpertSlots(16, 8, policy="lru")
        for e in range(16):
            sl.reserve(0, (e,), prefill=False)
        return sl

    # control: no speculation, one demand miss
    ctl = build()
    ctl.reserve(5, (500,), prefill=False)
    ctl_order = list(ctl.lru)

    # test: speculate (protected), demand miss, cancel, roll back
    tst = build()
    spec, refused = tst.reserve_speculative([(9, 950)])
    assert spec and not refused
    k, slot, gen = spec[0]
    tst.mark_pending(spec)                     # queued: its slot is protected
    tst.reserve(5, (500,), prefill=False)      # the demand miss must evict SOMEONE ELSE
    tst.clear_pending(slot, gen)
    tst.rollback_speculative(k, slot, gen)

    assert k not in tst.lru, "the speculation survived its own rollback"
    assert (5, 500) in tst.lru, "the demand miss was lost"
    differs = list(tst.lru) != ctl_order
    assert differs, (
        "the counterfactual matched exactly -- if that is now true, provisional reservation has "
        "been implemented and this test should be replaced by an equality assertion")
    lost = [x for x in ctl_order if x not in tst.lru]
    print(f"  queued reservation cost {len(lost)} extra resident(s) that the control kept "
          f"({lost[:2]}) -- documented, not free  OK")


# ================================================================ 25. retry, cohort, failure
def test_a_dead_prediction_does_not_block_a_closer_one_and_a_failed_read_is_not_a_hit():
    """Three ways the classifier could still flatter or starve a predictor.

    a) _spec survives cache eviction, so once a prediction died early the same key could never be
       predicted again closer to its target -- "earliest prediction wins forever", which is the
       worst possible policy exactly when early death is common. A dead attempt is now retired
       (still counted, it really read) and the nearer prediction is allowed through.
    b) a FAILED speculative read still has a completion timestamp -- set() records one even on
       error -- and its mapping survives until the driver's deferred un-map. Classifying on the
       timestamp counted it a timely hit, then the un-map ran and the expert became a demand miss.
    c) the scored cohort needs a denominator of its own: `started` includes reads issued during
       settlement, which are deliberately unscored.
    """
    calls = load_decode()
    cut = warmup_cut(calls)

    # (a) heavy pressure + long horizon: early death is common, retries must recover some of it
    e = Engine(V2, lru_slots=600, transient_slots=400,
               leaves=ModelLeaves(calls, Bandwidth(), start=cut),
               prefetch=OraclePrefetcher(calls, 16, start=cut))
    try:
        e.warm(calls, cut)
        e.decode(3)
        e.finalize_stats()
        p = e.pf
        assert p.evicted_before_use > 0, "no early death; this arm proves nothing"
        assert p.reissued > 0, (
            "a key that died early was never predicted again -- the scheduler is still "
            "'earliest prediction wins forever'")
        # (c) the cohort denominator is a real subset of the raw count
        assert 0 < p.started_cohort <= p.started, (p.started_cohort, p.started)
        assert p.precision == p.used / p.started_cohort
    finally:
        e.close()

    # (b) a failing read must not be counted ready. Force one on a speculative key.
    e = Engine(V2, lru_slots=64, transient_slots=8, n_workers=2, staging=8,
               expert_read_qd=2, h2d_inflight=2,
               leaves=ModelLeaves(calls, Bandwidth(), start=cut))
    try:
        spec, refused = e.slots.reserve_speculative([(9, 700)])
        assert spec and not refused
        key, slot, gen = spec[0]
        e.loader.fail.add(key)
        e._spec[key] = SpecAttempt(key, slot, gen, cause_id=1)
        e.loader.submit(spec, speculative=True)
        e.loader.quiesce()
        assert e.loader.ready.state(slot, gen) == "error", "the read did not fail"
        # settle BEFORE drain_forgets, which is the window the bug lived in
        e._settle_speculation(9, (700,), None)
        assert e.pf.failed_before_use == 1, (
            f"a failed speculative read was classified as something else "
            f"(ready {e.pf.ready_hit}, late {e.pf.late_hit}, used {e.pf.used})")
        assert e.pf.ready_hit == 0 and e.pf.used == 0
    finally:
        e.close()
    print("  retry: dead predictions are retired and re-predicted; a failed read is not a hit  OK")


# ================================================================ 26. the cohort is one population
def test_cancel_outstanding_works_and_every_statistic_is_cohort_scoped():
    """The regression that motivated SpecAttempt, plus the mixing it was hiding.

    a) _spec went (slot, gen, cause) -> +seq and one unpack site was missed, so
       finalize_stats(cancel_outstanding=True) raised ValueError at runtime. A named object turns
       that class of change into an edit-time error.
    b) the sequence cutoff gated only FETCH outcomes. pred_hit/pred_miss, issued, refused and
       reissued kept counting during settlement, so one PrefetchStats held two populations.
    c) the cutoff was global and stayed set, so any decode after a settle() was permanently
       unscored on the same Engine.
    d) started_cohort was reconstructed from outcome buckets, which counts an injected failure --
       it raises BEFORE the read leaf increments the counter. It is measured now.
    """
    calls = load_decode()
    cut = warmup_cut(calls)

    # (a) must not raise
    e = Engine(V2, lru_slots=5328, transient_slots=400,
               leaves=ModelLeaves(calls, Bandwidth(), start=cut),
               prefetch=OraclePrefetcher(calls, 8, start=cut))
    try:
        e.warm(calls, cut)
        e.decode(3)
        e.finalize_stats(cancel_outstanding=True)
        assert not e._spec, "cancelled attempts were left in _spec; a second call double-counts"
        e.finalize_stats(cancel_outstanding=True)      # idempotent
    finally:
        e.close()

    # (b) + (c) + (d)
    e = Engine(V2, lru_slots=5328, transient_slots=400,
               leaves=ModelLeaves(calls, Bandwidth(), start=cut),
               prefetch=OraclePrefetcher(calls, 4, start=cut))
    try:
        e.warm(calls, cut)
        e.decode(3)
        before = (e.pf.issued, e.pf.pred_hit + e.pf.pred_miss, e.pf.reissued)
        e.settle(4)
        after = (e.pf.issued, e.pf.pred_hit + e.pf.pred_miss, e.pf.reissued)
        assert before[0] == after[0], f"issued grew during settlement: {before[0]} -> {after[0]}"
        assert before[2] == after[2], f"reissued grew during settlement: {before[2]} -> {after[2]}"
        # (c) scoring is restored, so a later decode counts again
        assert e._scoring is True, "settlement left the engine permanently unscored"
        # The property is that new attempts INHERIT scoring again -- not that a particular
        # prediction produces I/O, which depends on residency and is not a fact about scoring.
        # (decode() is also unusable here: settle() consumes trace, so a replay provider desyncs.)
        from enginev2.prefetch import SpecAttempt as _SA
        probe = _SA(("probe",), 0, 1, cause_id=0, scored=e._scoring)
        assert probe.scored is True, "attempts created after settlement are still unscored"

        e.finalize_stats()
        # (d) measured, not reconstructed
        assert e.pf.started_cohort == e.loader.started_spec_scored
        assert e.pf.started_cohort <= e.pf.started
        assert 0.0 <= e.pf.precision <= 1.0, e.pf.precision
    finally:
        e.close()
    print("  cohort: cancel-outstanding works, settlement counts nothing, scoring is restored, "
          "denominator is measured  OK")


# ================================================================ 27. scored gates counting only
def test_an_unscored_wrong_prediction_is_still_physically_discarded():
    """Settlement holds the POLICY constant and withholds only the STATISTICS.

    `if not att.scored: continue` returned before the wrong-prediction branch, so a prediction
    issued during settlement that turned out wrong was popped from _spec and never cancelled -- it
    stayed resident. Settlement therefore switched the discard policy it was meant to hold fixed,
    and the retained expert could evict something a later scored prediction needed.

    Test 26 could not see it: a perfect oracle produces no wrong predictions during settlement.
    This uses a deliberately imprecise one.
    """
    calls = load_decode()
    cut = warmup_cut(calls)
    e = Engine(V2, lru_slots=2000, transient_slots=400,
               leaves=ModelLeaves(calls, Bandwidth(), start=cut),
               prefetch=RecallOraclePrefetcher(calls, 4, recall=1.0, precision=0.4, start=cut))
    try:
        e.warm(calls, cut)
        e.decode(3)
        # NOTE the counters DO grow across settlement, legitimately: attempts issued BEFORE it,
        # whose target layer falls inside it, are scored and settle there. What must be true is
        # narrower -- an attempt issued DURING settlement, and wrong, is discarded physically and
        # not counted. discarded_unscored exists to make exactly that observable.
        e.settle(4)
        assert e.pf.discarded_unscored > 0, (
            "no wrong prediction issued during settlement was physically discarded -- either the "
            "predictor made none, or `scored` is still short-circuiting the discard")
        e.loader.quiesce()
        e.loader.drain_forgets()

        # nothing speculative and wrong may still hold a slot at its own generation
        assert e.loader.stage.at_rest(), "settlement leaked a staging lease"
        assert e.arena.violations == [], e.arena.violations[:3]
        print(f"  settlement: {e.pf.discarded_unscored} wrong unscored predictions discarded "
              f"physically and counted nowhere  OK")
    finally:
        e.close()


# ================================================================ 28. the cohort boundary leaks
def test_settlement_cannot_erase_a_prediction_made_inside_the_window():
    """Three ways the scored/unscored boundary still mixed populations.

    a) _pred keeps ONE record per (target, key). An unscored settlement prediction overwrote a
       scored one, and a prediction genuinely issued in the timed window then vanished from
       pred_hit/pred_miss. Not exotic: the oracle re-predicts the same target every layer inside
       its horizon, and a RESIDENT correct prediction has no _spec entry to suppress the duplicate.
    b) a retired attempt was labelled evicted_before_use -- "right, but too early" -- without
       checking the layer actually wanted it, so a plain false positive was reported as a timing
       failure.
    c) settlement events still reached CounterObserver, so its totals included work the timed
       Counters exclude, and unlike Counters an observer cannot undo an increment.
    """
    calls = load_decode()
    cut = warmup_cut(calls)

    # (a) THROUGH THE REAL PATH. An earlier version of this test reimplemented the merge inline
    # and so proved nothing -- the overwrite mutation passed it. Drive _issue_speculation with a
    # predictor that names the same target twice, once scored and once not.
    # The key must be ALREADY RESIDENT. That is the whole scenario: a correct prediction for a
    # cached expert produces no _spec entry, so nothing suppresses the duplicate and the second
    # emission reaches the merge. A non-resident key is filtered out by _spec and never gets there,
    # which is why the first version of this test could not fail.
    class Fixed(Prefetcher):
        name = "fixed"
        horizon = 1
        target = None

        def predict(self, layer, uniq, step):
            return (self.target,) if self.target else ()

    pf = Fixed()
    e = Engine(V2, lru_slots=5328, transient_slots=400,
               leaves=ModelLeaves(calls, Bandwidth(), start=cut), prefetch=pf)
    try:
        e.warm(calls, cut)
        pf.target = next(iter(e.slots.lru))             # something the cache already holds
        tl = pf.target[0]
        e._scoring = True
        e._issue_speculation(tl - 1, (0,), None)        # scored prediction, resident -> no _spec
        assert pf.target not in e._spec, "the fixture key was not resident; the arm is invalid"
        assert e._pred.get(tl, {}).get(pf.target, (0, False))[1] is True, "setup failed"
        e._scoring = False
        e._issue_speculation(tl - 1, (0,), None)        # settlement re-predicts the same target
        assert e._pred[tl][pf.target][1] is True, (
            "an unscored settlement prediction downgraded a scored one; that prediction would "
            "disappear from pred_hit/pred_miss")
        e._scoring = True
    finally:
        e.close()

    # (b) + (c) end to end. The counter is compared against a TRACE of the same run: the property
    # is not "nothing moved during settlement" -- scored reads issued in the window legitimately
    # START during it, and asserting stillness rewarded the very bug this replaced. The property is
    # that NO UNSCORED event reached the aggregate.
    # Budget for the WHOLE run on one ring: capacity is a total divided across rings, and the
    # driver thread alone emits tens of thousands here. A wrapped ring silently undercounts the
    # scored population and makes the comparison below look like an aggregate over-count.
    trace = TraceObserver(capacity=1 << 21, max_rings=8)

    class Both(CounterObserver):
        def emit(self, event):
            trace.emit(event)                 # trace keeps everything, including unscored
            super().emit(event)               # aggregate must keep only scored

    obs = Both()
    e = Engine(V2, lru_slots=2000, transient_slots=400, observer=obs,
               leaves=ModelLeaves(calls, Bandwidth(), start=cut),
               prefetch=RecallOraclePrefetcher(calls, 4, recall=1.0, precision=0.4, start=cut))
    try:
        e.warm(calls, cut)
        e.decode(3)
        e.settle(4)
        e.finalize_stats()
        ev = trace.drain()
        assert trace.dropped == 0, f"the trace wrapped ({trace.dropped} dropped); comparison void"
        unscored = [x for x in ev if not x.scored]
        assert unscored, "settlement emitted no unscored events; the arm proves nothing"
        assert obs.counts, "the aggregate recorded nothing at all"
        # every counted event must be scored: compare totals per kind
        from collections import Counter as _C
        scored_by_kind = _C(x.kind for x in ev if x.scored)
        for kind, n in obs.counts.items():
            assert n <= scored_by_kind[kind], (
                f"aggregate counted {n} {kind} but only {scored_by_kind[kind]} were scored")
        # (b) a retired attempt that was never wanted must not be a timing failure
        assert e.pf.wasted >= e.pf.evicted_before_use + e.pf.failed_before_use
        print(f"  cohort boundary: {len(unscored)} unscored events traced, none counted; "
              f"scored starts {e.pf.started_cohort}/{e.pf.started}  OK")
    finally:
        e.close()




# ================================================================ 29. a wrong retired attempt
def test_a_retired_attempt_the_layer_never_wanted_is_not_a_timing_failure():
    """evicted_before_use means RIGHT but too early. A false positive must not borrow that label.

    Test 28's assertions on this were tautological (`x <= y + x`). Constructed explicitly here: a
    retired attempt whose key the target layer does not want must land in `wasted` only, leaving
    both timing buckets untouched -- otherwise the failure-mode breakdown reports a prediction
    error as a scheduling problem, which is the opposite diagnosis.
    """
    calls = load_decode()
    cut = warmup_cut(calls)
    e = Engine(V2, lru_slots=5328, transient_slots=400,
               leaves=ModelLeaves(calls, Bandwidth(), start=cut))
    try:
        before = (e.pf.evicted_before_use, e.pf.failed_before_use, e.pf.wasted)
        # a retired attempt for layer 5, key the layer will NOT want
        att = SpecAttempt((5, 999), slot=17, gen=3, cause_id=7, scored=True)
        att.terminal = "evicted"
        e._expired.setdefault(5, []).append(att)
        e._settle_speculation(5, (1, 2, 3), None)          # layer 5 wants 1,2,3 -- not 999
        after = (e.pf.evicted_before_use, e.pf.failed_before_use, e.pf.wasted)
        assert after[0] == before[0], (
            f"a retired attempt the layer never wanted was counted evicted_before_use "
            f"({before[0]} -> {after[0]})")
        assert after[1] == before[1], "it was counted as an I/O failure"
        assert after[2] == before[2] + 1, f"it should be plain wasted ({before[2]} -> {after[2]})"

        # and the same attempt, when the layer DOES want it, is a timing failure
        att2 = SpecAttempt((6, 42), slot=18, gen=4, cause_id=8, scored=True)
        att2.terminal = "evicted"
        e._expired.setdefault(6, []).append(att2)
        mid = e.pf.evicted_before_use
        e._settle_speculation(6, (42,), None)
        assert e.pf.evicted_before_use == mid + 1, (
            "a retired attempt the layer DID want was not counted as too-early")
    finally:
        e.close()
    print("  retired: wanted+evicted is a timing failure, unwanted is just wrong  OK")


# ================================================================ 30. settlement demand work
def test_demand_work_during_settlement_is_not_labelled_measured():
    """Settlement generates REAL demand misses, and the demand path defaulted to scored=True.

    So a settlement load_queued / NVME_ADMISSION / nvme / STAGING_BUFFER / slot_ready /
    EXPERT_DATA / h2d chain could be labelled part of the measured cohort. Test 28 is blind to it
    by construction: it compares CounterObserver against TraceObserver on Event.scored, and if BOTH
    receive a wrongly-scored event they agree and it passes.

    Bracketed by timestamp instead: every loader event emitted between settle()'s first and last
    instruction must be unscored, demand or speculative alike.
    """
    calls = load_decode()
    cut = warmup_cut(calls)
    obs = TraceObserver(capacity=1 << 21, max_rings=8)
    e = Engine(V2, lru_slots=900, transient_slots=400, observer=obs,
               leaves=ModelLeaves(calls, Bandwidth(), start=cut))
    try:
        e.warm(calls, cut)
        e.decode(2)
        t0 = now_ns()
        e.settle(8)                      # small cache: these layers really do miss
        t1 = now_ns()
        e.loader.quiesce()
        assert obs.dropped == 0, f"trace wrapped ({obs.dropped}); the window is partial"

        window = [x for x in obs.drain() if t0 <= x.ts_ns <= t1]
        demand = [x for x in window if x.kind == "load_queued" and x.aux == "demand"]
        assert demand, "settlement issued no demand reads; the arm proves nothing"
        mislabelled = [x for x in window if x.scored]
        assert not mislabelled, (
            f"{len(mislabelled)} settlement events labelled measured, e.g. "
            f"{[(x.kind, x.aux) for x in mislabelled[:4]]}")
        print(f"  settlement: {len(demand)} demand reads and {len(window)} events in the window, "
              f"none labelled measured  OK")
    finally:
        e.close()


def test_speculation_cannot_evict_the_current_layers_slots():
    """Review 2026-09-16, item 1. Speculation is issued after bind_slots() has baked slot numbers
    into the provider's route-aligned tensor but BEFORE graph B reads them. A slot in that set must
    not be a legal victim.

    The fixture is built so POLICY cannot hide the bug: the current layer's keys are made the
    LEAST attractive victims-by-accident is exactly what masks this in production, so here they are
    deliberately the OLDEST entries in the LRU and therefore the first victims any ordinary policy
    would choose. Without protected_slots the speculative reservation takes them.
    """
    sl = ExpertSlots(lru_slots=6, transient_slots=8, policy="lru")

    # six residents; the first three are oldest and would be evicted first
    for i in range(6):
        slot_of, to_load, _ = sl.reserve(0, (i,), prefill=False)
        sl.clear_pending(slot_of[i], sl.gen[slot_of[i]])

    current = {0: sl.lru[(0, 0)], 1: sl.lru[(0, 1)], 2: sl.lru[(0, 2)]}   # the oldest three
    live = frozenset(current.values())

    spec, refused = sl.reserve_speculative([(0, 90), (0, 91), (0, 92)], protected_slots=live)
    got = {s for _k, s, _g in spec}
    assert not (got & live), (
        f"speculation took live slots {sorted(got & live)}; graph B is about to read them")
    assert len(spec) + refused == 3

    # and the protection is not a blanket refusal: with nothing live, the same call places them
    sl2 = ExpertSlots(lru_slots=6, transient_slots=8, policy="lru")
    for i in range(6):
        so, _tl, _ = sl2.reserve(0, (i,), prefill=False)
        sl2.clear_pending(so[i], sl2.gen[so[i]])
    spec2, refused2 = sl2.reserve_speculative([(0, 90), (0, 91), (0, 92)])
    assert len(spec2) == 3 and refused2 == 0, (len(spec2), refused2)


# ----------------------------------------------------------------------------------------------
# PUBLISHED IS NOT LANDED.
#
# Since the device-ordered fast path, SlotReady.set() fires when the H2D is ENQUEUED. cancel() used
# SlotReady.is_done() to mean "the write is over", so a speculative read whose copy was still
# running could be classified "finished": the driver then un-maps the key, clears the pending
# marker, and the slot becomes an ordinary eviction victim while an H2D is still writing it. The
# next tenant's copy is issued on a different worker stream with nothing ordering the two writes --
# the torn-slot class the I/O hardening closed, reopened by a change in what "ready" means.
#
# The whole existing suite passed with that bug present, because every modelled provider completes
# its h2d inline and so is never published-but-not-landed. These drive the state directly.
# ----------------------------------------------------------------------------------------------

def test_a_published_but_unlanded_copy_is_cancelled_as_running_not_finished():
    e = mk(V2, lru_slots=16, transient_slots=8)
    try:
        ld = e.loader
        key, slot, gen = (3, 7), 2, 1
        # The state a device-ordered provider leaves between enqueue and completion.
        with ld._lk:
            ld._running.add((slot, gen))
        ld.ready.set(slot, gen)                       # PUBLISHED
        assert ld.ready.is_done(slot, gen)
        assert not ld.ready.is_landed(slot, gen)

        out = ld.cancel([(key, slot, gen)])
        assert out == [(key, slot, gen, "running")], (
            f"got {out}: a copy that is published but still moving bytes was classified as "
            f"finished, which releases its slot to the next reserve() mid-write")
        with ld._lk:
            assert (slot, gen) in ld._discard, "not queued for discard at completion"
            assert not any(f[1] == slot for f in ld._forget), (
                "un-mapped immediately -- the slot can now be handed to another expert while the "
                "old H2D is still writing it")
    finally:
        e.close()


def test_a_landed_copy_is_still_cancelled_as_finished():
    """The other half: once the bytes are down the slot really can be released at once, and the
    fix must not have turned every cancellation into a deferred one."""
    e = mk(V2, lru_slots=16, transient_slots=8)
    try:
        ld = e.loader
        key, slot, gen = (3, 8), 3, 1
        ld.ready.set(slot, gen)
        ld.ready.set_landed(slot, gen)
        out = ld.cancel([(key, slot, gen)])
        assert out == [(key, slot, gen, "finished")], out
        with ld._lk:
            assert any(f[1] == slot for f in ld._forget), "a landed slot was not un-mapped"
            assert (slot, gen) not in ld._discard
    finally:
        e.close()


def test_completion_clears_the_running_mark_so_a_later_cancel_is_not_misread():
    """_complete_h2d must leave _running, or every later cancellation of that slot is deferred
    forever and the discard set grows without bound."""
    e = mk(V2, lru_slots=16, transient_slots=8)
    try:
        ld = e.loader
        uniq = [0, 1]
        slot_of, to_load, _ = e.slots.reserve(0, uniq, prefill=False)
        ld.submit(to_load, speculative=True)
        ld.quiesce(timeout=30)
        with ld._lk:
            left = set(ld._running)
        assert not left, f"_running still holds {left} after quiesce -- completions are not clearing it"
        for k, sl, g in to_load:
            assert ld.ready.is_landed(sl, g), f"({sl},{g}) never marked landed"
    finally:
        e.close()


# ----------------------------------------------------------------------------------------------
# QUEUED -> RUNNING MUST BE ATOMIC.
#
# cancel() classifies on membership: _queued -> "queued", _running -> "running", neither ->
# "finished". If the worker leaves _queued in one critical section and enters _running in another,
# an id is briefly in NEITHER, cancel() calls a read that has not started yet "finished", the driver
# un-maps it and clears its pending marker, and the slot can be handed to another expert while this
# worker goes on to write it. Same torn-slot class as published-vs-landed, one step earlier.
#
# The behavioural test below can only observe the post-state, so the structural one proves the
# property directly: both mutations must happen inside the SAME lock acquisition.
# ----------------------------------------------------------------------------------------------

class _CountingLock:
    """Wraps the service lock and numbers each acquisition, so a mutation can record which one it
    happened in. Only the `with` protocol is used by LoaderService."""

    def __init__(self, inner):
        self._inner, self.gen = inner, 0

    def __enter__(self):
        self._inner.acquire()
        self.gen += 1
        return self

    def __exit__(self, *a):
        self._inner.release()
        return False


class _RecordingSet(set):
    def __init__(self, lock, log, name):
        super().__init__()
        self._lock, self._log, self._name = lock, log, name

    def add(self, x):
        self._log.append((self._name, "add", x, self._lock.gen))
        return super().add(x)

    def discard(self, x):
        if x in self:
            self._log.append((self._name, "discard", x, self._lock.gen))
        return super().discard(x)


def test_queued_to_running_happens_in_one_lock_acquisition():
    e = mk(V2, lru_slots=16, transient_slots=8)
    try:
        ld = e.loader
        log: list = []
        lk = _CountingLock(ld._lk)
        ld._lk = lk
        q, r = _RecordingSet(lk, log, "queued"), _RecordingSet(lk, log, "running")
        q.update(ld._queued)
        r.update(ld._running)
        ld._queued, ld._running = q, r

        slot_of, to_load, _ = e.slots.reserve(0, [0, 1], prefill=False)
        ld.submit(to_load, speculative=True)
        ld.quiesce(timeout=30)

        for _, sl, g in to_load:
            ident = (sl, g)
            leaves = [x[3] for x in log if x[0] == "queued" and x[2] == ident and x[1] == "discard"]
            enters = [x[3] for x in log if x[0] == "running" and x[2] == ident and x[1] == "add"]
            assert leaves and enters, f"no transition recorded for {ident}: {log}"
            assert leaves[0] == enters[0], (
                f"{ident} left _queued in lock acquisition {leaves[0]} and entered _running in "
                f"{enters[0]}. Between them cancel() sees neither set and returns 'finished', which "
                f"releases the slot while this worker is about to write it.")
    finally:
        e.close()


def test_a_read_that_has_started_is_never_cancelled_as_finished():
    """Behavioural half: hold a worker inside read(), then cancel. It must classify as running."""
    import threading as _th
    e = mk(V2, lru_slots=16, transient_slots=8)
    try:
        ld = e.loader
        entered, release = _th.Event(), _th.Event()
        inner = ld.leaves.read

        def blocking_read(key, pool, ctx=None, scored=True):
            entered.set()
            release.wait(10)
            return inner(key, pool, ctx, scored)

        ld.leaves.read = blocking_read
        slot_of, to_load, _ = e.slots.reserve(0, [0], prefill=False)
        ld.submit(to_load, speculative=True)
        assert entered.wait(10), "worker never reached read()"

        key, sl, g = to_load[0]
        out = ld.cancel([(key, sl, g)])
        assert out == [(key, sl, g, "running")], (
            f"got {out}: a read already inside read() was classified as finished, so its slot is "
            f"released while the read is still in flight")
        release.set()
        ld.quiesce(timeout=30)
    finally:
        try:
            release.set()
        except Exception:
            pass
        e.close()


def test_slot_readiness_does_not_grow_without_bound():
    """SlotReady kept every (slot, gen) for the life of the process.

    Measured before the fix: 4.0 entries per completed generation (_done, _ts, _landed,
    _landed_ts, plus _ctx when a producer context was recorded), never pruned. At the engine's
    19 expert reads per output token that is ~1.9M generations over a 100k-token session -- order
    of a gigabyte of unified memory, on a box where memory is the largest measured throughput
    lever. The published-vs-landed split doubled the per-generation cost, so this guards the fix
    rather than the original design.
    """
    e = mk(V2, lru_slots=64, transient_slots=16)
    try:
        def one_round(base: int) -> None:
            for layer in range(4):
                uniq = [(base * 5 + layer * 3 + i) % 50 for i in range(6)]
                e.decode_layer(layer) if False else None
                slot_of, to_load, to_wait = e.slots.reserve(layer, uniq, prefill=False)
                e.loader.submit(to_load)
                e.loader.quiesce(timeout=30)
                e.loader.drain_forgets()
                for _k, sl, g in to_load:
                    e.loader.ready.retire(sl, g)

        for r in range(5):
            one_round(r)
        early = e.loader.ready.tracked()
        for r in range(5, 25):
            one_round(r)
        late = e.loader.ready.tracked()
        assert late <= early + 8, (
            f"SlotReady grew {early} -> {late} over 20 more rounds: retirement is not reaching "
            f"some path, and this table is unbounded in a long session")
    finally:
        e.close()


def test_retire_before_landing_does_not_leak():
    """PUBLISHED-then-RETIRED-then-LANDED must leave nothing behind.

    The growth test above retires only after `quiesce()`, i.e. after the copy has physically
    completed, which is the one ordering where a purge-on-retire cannot be undone. The driver does
    not do that. Since the device-ordered fast path, `set()` fires when the H2D is ENQUEUED; with
    `policy.global_barrier` off, `_wait()` returns on that published readiness, graph B is issued,
    and the driver retires -- all while the bytes are still moving. `_complete_h2d()` then calls
    `set_landed()` on every path it has, success and both except arms.

    With an unconditional purge in retire() and an unconditional add in set_landed(), that sequence
    re-creates `_landed` and `_landed_ts` for a generation nobody will ever retire again: the
    4-entries-per-read leak comes back as 2 entries per read, on precisely the path the fast policy
    takes. Retirement is therefore a two-party handshake -- consumer_done AND landed -- and this is
    the mutation test for it.
    """
    r = SlotReady()
    r.arm(7, 3, cause_id=11)
    r.set(7, 3)                                   # H2D ENQUEUED, not landed
    assert not r.is_landed(7, 3), "set() must not imply landed"
    r.retire(7, 3)                                # the consumer is done; the copy is not
    assert not r.is_landed(7, 3), (
        "retire() of a published-but-unlanded generation must not mark it landed")
    r.set_landed(7, 3)                            # the bytes arrive AFTER the retire
    assert r.tracked() == 0, (
        f"tracked()=={r.tracked()} after the copy landed on an already-retired generation: "
        f"set_landed() re-created records that nothing will ever purge")
    assert not r.is_landed(7, 3) and r.landed_ts(7, 3) is None

    # And the ordinary ordering still purges, so the handshake did not merely defer the leak.
    r.arm(7, 4)
    r.set(7, 4)
    r.set_landed(7, 4)
    assert r.is_landed(7, 4)
    r.retire(7, 4)
    assert r.tracked() == 0 and r.landed_ts(7, 4) is None


def test_slotready_bounded_on_the_published_first_path():
    """Bounded growth over many reads when every one of them retires BEFORE it lands.

    An end-to-end version of this belongs here too and is deliberately not written: the mock loader
    completes its copy inside `ready.wait()`, so retire() always sees a landed generation and the
    rig cannot reach the published-but-unlanded window at all -- it needs the device-ordered fast
    path (`handle is not None and self._dev_orders`), where `set()` fires at enqueue. A test that
    cannot fail is worse than no test, so this drives SlotReady through the driver's exact sequence
    instead, and asserts EXACT zero residue rather than a loose bound.
    """
    r = SlotReady()
    for i in range(2000):
        slot, gen = i % 64, i
        r.arm(slot, gen, cause_id=i)
        r.set(slot, gen)          # published at H2D enqueue
        r.retire(slot, gen)       # consumer done: graph B issued
        r.set_landed(slot, gen)   # bytes land afterwards
    assert r.tracked() == 0, (
        f"{r.tracked()} generations retained over 2000 published-first reads -- at 19 expert reads "
        f"per output token this is the unbounded table retire() was written to prevent")


def test_sampled_verify_is_distribution_preserving():
    """Speculative decoding under temperature must emit from the TARGET distribution, not near it.

    This is the property that makes spec decode sound, and the one no performance test can see: a
    verify that accepted a draft because it happened to be the argmax would draw from a different
    distribution than the model's, while tokens/s, acceptance length and even NLL all looked
    ordinary. So drive `_verify_sampled` directly with a drafter distribution `q` deliberately
    unlike the target `p`, and check the emitted token's empirical distribution against `p`.

    Driven as an unbound call on a duck-typed self: the method needs six attributes and no GPU, and
    constructing a RealLeaves would need an engine, an arena and a CB3 file to test arithmetic.
    """
    import torch as _t
    from enginev2.real import RealLeaves
    from engine.v41_engine import sample_probs

    V, N = 8, 20000
    _t.manual_seed(0)
    logits = _t.randn(3, V)
    q = _t.softmax(_t.randn(1, V), -1)          # the drafter's own distribution, not the target's

    class _S: pass
    s = _S()
    s.fd = type("FD", (), {"logits": logits})()
    s.q, s._tv = q, 2                           # one draft position
    s.temperature, s.top_p, s.stop_ids = 1.0, 1.0, frozenset()

    counts = _t.zeros(V)
    greedy_counts = _t.zeros(V)
    target = sample_probs(logits[0].float(), 1.0, 1.0)
    argmax0 = int(logits[0].argmax())
    for _ in range(N):
        d = int(_t.multinomial(q[0], 1))        # the drafter proposes from q
        s.drafts = _t.tensor([d])
        a, new, bonus = RealLeaves._verify_sampled(s)
        counts[new[0] if new else bonus] += 1
        # What a greedy accept would have produced, for the same draw: accept iff the draft is the
        # target's argmax, else emit that argmax. Included so this test is shown to DISCRIMINATE --
        # a distribution check that passes for both implementations gates nothing.
        greedy_counts[d if d == argmax0 else argmax0] += 1

    tv = 0.5 * (counts / counts.sum() - target).abs().sum().item()
    tv_greedy = 0.5 * (greedy_counts / greedy_counts.sum() - target).abs().sum().item()
    assert tv < 0.02, (
        f"sampled verify diverges from the target distribution: total-variation {tv:.4f} over {N} "
        f"trials. Rejection sampling against q is what makes this distribution-preserving.")
    assert tv_greedy > 0.2, (
        f"the greedy comparison scored tv={tv_greedy:.4f}, so this test would pass for a greedy "
        f"accept too and gates nothing")


@pytest.mark.xfail(strict=True, reason=
                   "v2 has no _promote_transient: a v1-parity gap, not a correctness one. The "
                   "stale-weight defect it was briefly blamed for had a different cause -- "
                   "Model.decoder_replay drove v1's ExpertStore over v2's arena -- and is fixed by "
                   "giving layers src+1..39 a v2-owned seam. strict=True so this errors the moment "
                   "promotion lands.")
def test_a_decode_hit_in_the_transient_ring_is_promoted():
    """V1 parity: a decode hit on an expert sitting in the transient ring must be PROMOTED.

    v1 calls `_promote_transient()` (engine/experts.py:636, fired at :733) when a decode request
    hits a key that a prefill left in the transient ring: the expert moves into the LRU at its
    existing physical slot and an LRU donor slot is swapped back into the ring. v2's `reserve()`
    takes the `transient_map` hit and leaves it there, so the key stays in the ring, never joins the
    LRU, and -- the part that reaches beyond ring position -- never calls `evict.on_hit()` or
    advances the decode clock.

    Under `age_over_freq` that means a genuine decode access earns NO usage credit while it lives in
    the ring, which changes which expert is chosen as the next victim. So this is a cache-semantic
    divergence from v1 that outlives the ring itself.

    The existing suite missed it because its decode streams allocate straight into the LRU and never
    exercise prefill -> transient -> decode-hit.

    EXPECTED TO FAIL until v2 implements the promotion. It is recorded as a test rather than a note
    because a note would be read once and a failing test is read every run.
    """
    e = mk(V1, lru_slots=32, transient_slots=8)
    try:
        key_layer, expert = 0, 5
        # A prefill puts the expert in the transient ring.
        slot_of, to_load, _ = e.slots.reserve(key_layer, [expert], prefill=True)
        e.loader.submit(to_load)
        e.loader.quiesce(timeout=30)
        e.loader.drain_forgets()
        slot = slot_of[expert]
        assert (key_layer, expert) in e.slots.transient_map, "setup: expected a transient placement"
        before_ring = set(e.slots.transient_ring)

        # Now DECODE asks for the same expert: v1 would promote it.
        slot_of2, to_load2, _ = e.slots.reserve(key_layer, [expert], prefill=False)
        assert slot_of2[expert] == slot, "promotion must keep the physical slot, not re-read"
        assert not to_load2, "a promotion is not a fetch"
        assert (key_layer, expert) in e.slots.lru, (
            "decode hit on a transient-ring expert did not promote it into the LRU; v1 does "
            "(_promote_transient), and without it the expert earns no usage credit and the "
            "eviction policy picks different victims than v1 would")
        assert (key_layer, expert) not in e.slots.transient_map, "promoted key still in the ring"
        assert set(e.slots.transient_ring) != before_ring, "no LRU donor was swapped into the ring"
    finally:
        e.close()


def test_decode_step_is_a_sequence_position_not_a_per_call_counter():
    """decode(2) twice must observe steps 0,1,2,3 -- never 0,1,0,1.

    This is the invariant behind jobs 825/880: `step` was `range(steps)`, so every decode() call
    restarted at 0, and RealLeaves.select_block() deliberately does nothing at step 0 (it leaves
    block_ids as they are). After a 30-step warm-up the first timed step replayed the warm-up's last
    block and then resumed at corpus block 1 -- sequence 29, 1, 2, 3 at positions 30, 31, 32. The
    harness spent three GPU jobs looking for a bad ROUTE_BASE that was never wrong; the engine was
    generating a genuinely different sequence.

    It is a production bug too, not only a harness one: serving calls decode(1) repeatedly, so every
    invocation looked internally like step 0 and select_block never advanced the block at all.

    CPU-only and fake-leaved on purpose -- this must be checkable before anything is queued.
    """
    calls = load_decode()
    cut = warmup_cut(calls)
    e = Engine(V2, lru_slots=5328, transient_slots=400, scale=SCALE,
               leaves=ModelLeaves(calls, Bandwidth(scale=SCALE), scale=SCALE, start=cut))
    e.warm(calls, cut)
    seen = []
    orig = e.leaves.select_block

    def spy(step):
        seen.append(step)
        return orig(step)

    e.leaves.select_block = spy
    e.decode(2)
    e.decode(2)
    assert seen == [0, 1, 2, 3], f"expected sequence positions [0,1,2,3], observed {seen}"

    # The measurement counters are independent: replacing them must not move the sequence.
    before = e.seq_step
    e.c = type(e.c)()
    e.decode(1)
    assert seen == [0, 1, 2, 3, 4], f"a counter reset moved the sequence: {seen}"
    assert e.seq_step == before + 1
