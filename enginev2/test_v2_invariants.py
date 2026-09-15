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
from .leaves import Bandwidth, ModelLeaves
from .store import ExpertSlots, SlotArena, SlotReady
from .trace import load_decode, warmup_cut

SCALE = 20.0          # leaves run 20x faster; ratios are preserved, the tests are about ordering


def mk(policy: Policy = V2, lru_slots: int = 16, transient_slots: int = 8, n_workers: int = 8,
       staging: int = 8, nvme_qd: int | None = None, h2d_inflight: int | None = None,
       fail=None) -> Engine:
    # nvme_qd must stay BELOW the buffer count or releasing the permit at handoff buys nothing and
    # the loader refuses the configuration -- so derive it from staging unless a test pins it.
    if nvme_qd is None:
        nvme_qd = max(1, staging // 2)
    if h2d_inflight is None:
        h2d_inflight = max(1, staging // 2)
    e = Engine(policy, lru_slots=lru_slots, transient_slots=transient_slots,
               n_workers=n_workers, staging=staging, nvme_qd=nvme_qd,
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
        assert (3, 2) not in e.slots.lru and (3, 5) not in e.slots.lru, "torn slot still mapped"
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
        e = mk(pol, n_workers=1, staging=2, nvme_qd=1, h2d_inflight=1)
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
