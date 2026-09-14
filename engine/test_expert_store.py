"""test_expert_store.py -- store-level determinism for deferred (EARLY_SUBMIT) resolves.

The engine-level A/B (test_layer_major.py) can only say "the tokens came out the same on this
prompt". That is not enough to trust `resolve(defer=True)`: the failure it guards against is a slot
handed to a second expert while the first one's 13.8 MB read is still landing in it, which is
silent, data-dependent, and invisible whenever the transient ring happens to be larger than a
layer's distinct expert count. TRANSIENT_SLOTS=400 vs ~362 distinct experts is exactly that
"happens to be", so the engine test would pass over a broken store.

So this test drives the store directly, with `_load_into_slot` replaced by a recorder that sleeps a
pseudo-random amount before writing a known byte pattern into a fake arena. No NVMe, no GPU work,
no model -- the only thing under test is the slot bookkeeping in resolve()/_transient_slot_for()/
join_pending(), which is where the bug class lives.

Run:  python -m engine.test_expert_store
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import threading

# the module reads these at import time; a stale one from the serving env would open a real cache
for _v in ("DSV41_CB3_CACHE", "DSV41_ROUTE_LOG", "DSV41_UNSAFE_NO_COMPUTE_WAIT"):
    os.environ.pop(_v, None)
os.environ["DSV41_IO_THREADS"] = "4"

import torch

from engine import experts as E


class FakeArena:
    """Only what ExpertStore's constructor and our recorder touch."""

    def __init__(self, slots: int):
        self.slots = slots
        self.device = "cpu"


def make_store(transient_slots: int, lru_slots: int = 8, io_threads: int = 4,
               seed: int = 0xC0FFEE):
    d = tempfile.mkdtemp(prefix="dsv41-store-test-")
    arena = FakeArena(transient_slots + lru_slots)
    st = E.ExpertStore(d, {"weight_map": {}}, arena, n_layers=4,
                       transient_slots=transient_slots, io_threads=io_threads)
    # the recorder: what a slot HOLDS (key), in completion order, with a jittered delay so
    # completions interleave differently from submission order.
    st.test_content = {}            # slot -> key, as the loads actually land
    st.test_loads = []              # (key, slot) in submission order
    st.test_order = []              # (key, slot) in COMPLETION order
    st.test_lock = threading.Lock()
    rng = random.Random(seed)

    def fake_load(key, slot, prefix=None):
        with st.test_lock:
            st.test_loads.append((key, slot))
            delay = rng.random() * 0.02
        # a real load owns the slot for the whole of this window; anything that writes the same
        # slot meanwhile is the corruption we are looking for.
        with st.test_lock:
            before = st.test_content.get(slot)
        threading.Event().wait(delay)
        with st.test_lock:
            now = st.test_content.get(slot)
            assert now == before, (
                f"slot {slot} was overwritten while {key}'s read was in flight: "
                f"{before} -> {now}")
            st.test_content[slot] = key
            st.test_order.append((key, slot))
        return slot

    st._load_into_slot = fake_load
    return st


def ids(*experts):
    return torch.tensor([list(experts)], dtype=torch.int32)


# ---------------------------------------------------------------- the tests

def test_overlapping_ids_load_once():
    """Two chunks of one layer that share experts must produce ONE physical load per expert."""
    st = make_store(transient_slots=16)
    st.resolve(0, ids(1, 2, 3), prefill=True, defer=True)
    st.resolve(0, ids(3, 4, 1), prefill=True, defer=True)
    st.join_pending()
    loaded = [k for k, _ in st.test_loads]
    assert sorted(loaded) == [(0, 1), (0, 2), (0, 3), (0, 4)], loaded
    assert len(loaded) == len(set(loaded)), f"expert loaded twice: {loaded}"
    print(f"  overlapping ids -> one load each: {len(loaded)} loads for 4 distinct experts  OK")


def test_out_of_order_completion_identical_arena():
    """Completion order is jittered; the resulting slot->expert map must not depend on it."""
    seq = [(0, (5, 9, 2, 7)), (0, (7, 1, 5)), (0, (3, 9, 8))]
    maps = []
    orders = []
    for seed in (1, 2, 3):
        st = make_store(transient_slots=16, seed=seed)
        for layer, ex in seq:
            st.resolve(layer, ids(*ex), prefill=True, defer=True)
        st.join_pending()
        maps.append(dict(st.test_content))
        orders.append(list(st.test_order))
    assert maps[0] == maps[1] == maps[2], maps
    assert any(o != orders[0] for o in orders[1:]) or len(orders[0]) < 2, \
        "completion order never varied -- the test is not exercising what it claims"
    print(f"  out-of-order completion -> identical arena ({len(maps[0])} slots)  OK")


def test_union_larger_than_ring_raises():
    """A layer whose distinct experts exceed the ring must RAISE, never recycle a pending slot."""
    st = make_store(transient_slots=8)
    st.resolve(0, ids(1, 2, 3, 4, 5, 6), prefill=True, defer=True)
    try:
        st.resolve(0, ids(10, 11, 12, 13, 14, 15), prefill=True, defer=True)
    except RuntimeError as e:
        assert "transient ring exhausted" in str(e), e
        print(f"  union > ring -> RuntimeError, not corruption  OK")
        st.join_pending()
        return
    st.join_pending()
    raise AssertionError("resolve() recycled a slot whose read was still in flight")


def test_deferred_matches_blocking():
    """defer=True and defer=False must give the same slot tensors and the same final arena."""
    seq = [(0, (5, 9, 2)), (0, (9, 4)), (1, (2, 5)), (1, (7, 2, 1))]
    out = {}
    for mode in (False, True):
        st = make_store(transient_slots=16)
        slots = []
        last_layer = None
        for layer, ex in seq:
            if mode and last_layer is not None and layer != last_layer:
                st.join_pending()          # the contract: join at a layer boundary
            slots.append(st.resolve(layer, ids(*ex), prefill=True, defer=mode).tolist())
            last_layer = layer
        if mode:
            st.join_pending()
        out[mode] = (slots, dict(st.test_content))
    assert out[False][0] == out[True][0], (out[False][0], out[True][0])
    assert out[False][1] == out[True][1], (out[False][1], out[True][1])
    print(f"  deferred == blocking (slot maps and arena identical)  OK")


def test_cross_layer_defer_asserts():
    """Deferred resolves spanning two layers without a join must assert, not silently proceed."""
    st = make_store(transient_slots=16)
    st.resolve(0, ids(1, 2), prefill=True, defer=True)
    try:
        st.resolve(1, ids(3, 4), prefill=True, defer=True)
    except AssertionError as e:
        assert "span layers" in str(e), e
        print(f"  deferred resolve across a layer boundary -> AssertionError  OK")
        st.join_pending()
        return
    st.join_pending()
    raise AssertionError("resolve() accepted deferred resolves spanning two layers")


if __name__ == "__main__":
    print("store-level determinism for resolve(defer=True):")
    fails = 0
    for fn in (test_overlapping_ids_load_once,
               test_out_of_order_completion_identical_arena,
               test_union_larger_than_ring_raises,
               test_deferred_matches_blocking,
               test_cross_layer_defer_asserts):
        try:
            fn()
        except Exception as exc:                      # noqa: BLE001 -- report all, then fail once
            fails += 1
            print(f"  FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print("ALL OK" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
