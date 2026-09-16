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

from .drivers import Engine
from .sched import V1, V2, ComputeStream, LoaderService, Policy
from .chain import Chain, EngramSource
from .prefetch import OraclePrefetcher, RecallOraclePrefetcher
from .observe import CounterObserver, NullObserver, TraceObserver, WaitReason
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
        assert e.loader.q.unfinished_tasks == 0, "pending left populated after the wait"
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
            e.prefill_chunked(0, chunks)
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
        def issue(self, layers, step, chain):
            for L in layers:
                if L != 7:
                    chain.set("engram", L)

    e = Engine(V2, lru_slots=5328, transient_slots=400, scale=SCALE,
               leaves=ModelLeaves(calls, Bandwidth(scale=SCALE), scale=SCALE, start=cut),
               engram=SkipsOneLayer())
    try:
        e.warm(calls, cut)
        ok, _ = run_with_timeout(lambda: e.decode(1), 6)
        assert not ok, "the step completed although layer 7's engram rows never arrived"
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

        def read(self, key, pool, ctx=None):
            self.__dict__["_pool"] = pool
            return super().read(key, pool, ctx)

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

        q, running, fin = e.loader.cancel(spec)
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
