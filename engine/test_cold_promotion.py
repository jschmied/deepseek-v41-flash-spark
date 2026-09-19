"""Hazards of the cold-pool promotion lifecycle. No CUDA: this is the bookkeeping, not the copies.

Each test is one way the design corrupts data if a rule is dropped, and the rules are the four in
cold_promotion.py's docstring.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.cold_promotion import PromotionPool, ColdSlotBusy

K1, K2, K3 = (0, 11), (0, 22), (1, 33)


def test_hot_slot_is_not_resident_until_the_copy_lands():
    """Rule 1. The whole point: an enqueued promotion must not look like residency."""
    p = PromotionPool(2)
    i, g = p.reserve(K1, hot_slot=500)
    p.cold_ready(i, g)
    assert p.cold_of(K1) == (i, g), "before landing the expert must be read from the cold slot"
    assert p.hot_pending(K1) == 500
    p.compute_done(i, g)
    assert p.cold_of(K1) == (i, g), "compute finishing does not make the hot slot resident"
    p.promo_done(i, g)
    assert p.cold_of(K1) is None, "after landing the cold slot is no longer the source"
    assert p.hot_pending(K1) is None


def test_hot_destination_is_protected_while_the_copy_is_outstanding():
    """Rule 2. The destination holds no key, so only this set keeps eviction off it."""
    p = PromotionPool(2)
    i1, g1 = p.reserve(K1, hot_slot=500)
    i2, g2 = p.reserve(K2, hot_slot=501)
    assert p.pending_hot() == {500, 501}
    p.cold_ready(i1, g1); p.compute_done(i1, g1); p.promo_done(i1, g1)
    assert p.pending_hot() == {501}, "a landed promotion stops protecting its destination"


def test_cold_slot_needs_both_parties_in_either_order():
    """Rule 3. Whichever arrives second does the release; neither alone may."""
    for order in ("compute-first", "promo-first"):
        p = PromotionPool(1)
        i, g = p.reserve(K1, hot_slot=500)
        p.cold_ready(i, g)
        if order == "compute-first":
            p.compute_done(i, g)
            assert p.free_slots() == 0, f"{order}: released on compute alone"
            p.promo_done(i, g)
        else:
            p.promo_done(i, g)
            assert p.free_slots() == 0, f"{order}: released on promotion alone"
            p.compute_done(i, g)
        assert p.free_slots() == 1, f"{order}: not released once both arrived"


def test_a_stale_completion_cannot_release_the_next_tenant():
    """Rule 4. The recycle window is a few layers, so this is a live hazard, not a theoretical one."""
    p = PromotionPool(1)
    i, g = p.reserve(K1, hot_slot=500)
    p.cold_ready(i, g); p.compute_done(i, g); p.promo_done(i, g)
    j, g2 = p.reserve(K2, hot_slot=501)
    assert (j, g2) == (i, g + 1), "the same slot is reused with a bumped generation"
    for fn in ("compute_done", "promo_done", "cold_ready", "fail"):
        try:
            getattr(p, fn)(i, g)
        except RuntimeError as e:
            assert "stale generation" in str(e), f"{fn}: wrong error {e}"
        else:
            raise AssertionError(f"{fn} accepted a stale generation and could release K2's slot")
    assert p.cold_of(K2) == (j, g2), "K2 is still in flight after the stale completions"
    assert p.free_slots() == 0


def test_exhaustion_is_explicit_not_silent_reuse():
    p = PromotionPool(2)
    a = p.reserve(K1, 500); b = p.reserve(K2, 501)
    try:
        p.reserve(K3, 502)
    except ColdSlotBusy:
        pass
    else:
        raise AssertionError("a third reservation was handed a slot that is still owed work")
    assert p.stats["busy"] == 1
    assert set(p.pending_cold()) == {a[0], b[0]}


def test_failure_returns_the_key_and_still_waits_for_compute():
    """A failed promotion does not tell us the cold kernel has stopped reading the slot."""
    p = PromotionPool(1)
    i, g = p.reserve(K1, hot_slot=500)
    p.cold_ready(i, g)
    assert p.fail(i, g) == K1, "fail must name the key so the caller can undo its own bookkeeping"
    assert p.free_slots() == 0, "released before the cold kernel was known to be finished"
    p.compute_done(i, g)
    assert p.free_slots() == 1
    assert p.stats["failed"] == 1 and p.stats["promoted"] == 0


def test_double_reserve_of_one_key_is_refused():
    p = PromotionPool(2)
    p.reserve(K1, 500)
    try:
        p.reserve(K1, 501)
    except RuntimeError as e:
        assert "already in flight" in str(e)
    else:
        raise AssertionError("one key held two cold slots; its second promotion would race the first")


def test_compute_before_the_read_landed_is_refused():
    p = PromotionPool(1)
    i, g = p.reserve(K1, 500)
    try:
        p.compute_done(i, g)
    except RuntimeError as e:
        assert "before cold_ready" in str(e)
    else:
        raise AssertionError("the cold phase was allowed to consume bytes that had not arrived")


def test_model_class_structure_is_intact():
    """A structural guard, because ast.parse is not one.

    Inserting a module-level def between Model's methods terminates the class body and silently moves
    every following method out of it -- 13 of them, including forward(), block() and moe(). That
    parses cleanly and breaks the engine at import-time-plus-one. This asserts the shape instead.
    """
    import ast, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tree = ast.parse(open(os.path.join(root, "engine", "model.py")).read())
    classes = {n.name: [f.name for f in n.body if isinstance(f, ast.FunctionDef)]
               for n in tree.body if isinstance(n, ast.ClassDef)}
    model = classes.get("Model")
    assert model is not None, "engine/model.py has no Model class"
    for must in ("forward", "block", "moe", "moe_apply", "dspark_draft", "decoder_replay"):
        assert must in model, f"Model.{must} is missing -- the class body was terminated early"
    assert len(model) >= 20, f"Model has only {len(model)} methods; expected 20+"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"  ok  {f.__name__}")
    print(f"  {len(fns)}/{len(fns)} passed")
