"""test_ring_lending.py -- the transient ring is lent to the LRU during decode, and taken back.

WHY THE CHANGE EXISTS. The ring is sized for PREFILL: one chunk touches nearly every expert of a
layer, and streaming that through the LRU would evict the decode working set. During decode it is
idle -- 400 slots of a 4,565-slot arena doing nothing. Measured on the box (job 235, ARENA_GB
60/66/72 = 3,750/4,165/4,581 LRU slots): every +415 slots is -14.0 % NVMe and +12 % decode tok/s,
linear across both steps.

WHY IT COULD NOT BE A KNOB. transient_slots=8 starts and then raises `transient ring exhausted` on
the first prefill chunk (job 230). The ring is load-bearing at any small size, so the only way to
give decode those slots is to lend and reclaim them around prefill.

WHAT THIS PINS, none of which the e2e A/B can show:
  * lending is off unless DSV41_RING_TO_LRU=1 -- the shipped default must be untouched;
  * a lent slot is really allocatable by the LRU;
  * reclaim un-maps whatever decode put in the ring, so a later prefill cannot hit a slot the ring
    is about to reuse -- the torn-slot-counted-as-a-HIT failure class (1e293b7);
  * last prefill's transient_map is dropped at lend time, for the same reason;
  * a slot with a write still in flight is never lent;
  * lend/reclaim are idempotent, because the hooks sit on paths that can repeat.

Run:  python -m pytest engine/test_ring_lending.py -q
"""

from __future__ import annotations

import os
import tempfile

for _v in ("DSV41_CB3_CACHE", "DSV41_ROUTE_LOG", "DSV41_UNSAFE_NO_COMPUTE_WAIT",
           "DSV41_EVICT_POLICY", "DSV41_RING_TO_LRU"):
    os.environ.pop(_v, None)
os.environ["DSV41_IO_THREADS"] = "2"

from engine import experts as E


class FakeArena:
    def __init__(self, slots: int):
        self.slots = slots
        self.device = "cpu"


def make_store(lend: bool, lru_slots: int = 24, transient_slots: int = 8, policy: str = "lru"):
    os.environ["DSV41_RING_TO_LRU"] = "1" if lend else "0"
    try:
        d = tempfile.mkdtemp(prefix="dsv41-ring-test-")
        return E.ExpertStore(d, {"weight_map": {}}, FakeArena(transient_slots + lru_slots),
                             n_layers=4, transient_slots=transient_slots, io_threads=2,
                             evict_policy=policy)
    finally:
        os.environ.pop("DSV41_RING_TO_LRU", None)


def test_off_by_default_is_bit_identical():
    st = make_store(lend=False)
    free_before = list(st.free_lru)
    assert st.lend_ring_to_lru() == 0, "lending happened with DSV41_RING_TO_LRU unset"
    assert list(st.free_lru) == free_before, "the free list moved with lending off"
    assert st.reclaim_ring() == 0


def test_lent_slots_are_allocatable_and_come_back():
    st = make_store(lend=True, lru_slots=24, transient_slots=8)
    ring = set(st.transient_ring)
    assert st.lend_ring_to_lru() == 8
    assert ring <= set(st.free_lru), "lent slots are not in the LRU free list"

    # fill the whole enlarged region; without the ring this would need an eviction
    for e in range(32):
        st._lru_slot_for((0, e), frozenset())
    used_ring = {s for k, s in st.lru.items() if s in ring}
    assert used_ring, "decode never actually landed in a lent slot"

    dropped = st.reclaim_ring()
    assert dropped == len(used_ring), f"reclaim dropped {dropped}, expected {len(used_ring)}"
    assert not (ring & set(st.free_lru)), "ring slots stayed in the LRU free list"
    assert not any(s in ring for s in st.lru.values()), (
        "a key still maps to a ring slot after reclaim -- the next prefill would overwrite it "
        "while reserve() still counts it a HIT")
    for s in ring:
        assert st.slot_key.get(s) is None, f"slot {s} still has a key after reclaim"


def test_lend_drops_the_previous_prefills_transient_map():
    st = make_store(lend=True)
    st._transient_slot_for((1, 7), frozenset())
    assert st.transient_map, "fixture did not populate the ring"
    st.lend_ring_to_lru()
    assert not st.transient_map, (
        "last prefill's transient mapping survived lending; a decode reserve() could count it a "
        "HIT on a slot the LRU is about to reuse")


def test_a_slot_with_a_write_in_flight_is_never_lent():
    st = make_store(lend=True, transient_slots=8)
    busy = st.transient_ring[3]
    st._pending_slots.add(busy)
    try:
        lent = st.lend_ring_to_lru()
        assert lent == 7, f"lent {lent}, expected 7 (one slot mid-write)"
        assert busy not in st.free_lru, "a slot with a write in flight was handed to the LRU"
    finally:
        st._pending_slots.discard(busy)


def test_lend_and_reclaim_are_idempotent():
    st = make_store(lend=True, transient_slots=8)
    assert st.lend_ring_to_lru() == 8
    assert st.lend_ring_to_lru() == 0, "a second lend double-counted the ring"
    assert len(st.free_lru) == len(set(st.free_lru)), "the free list has duplicates"
    st.reclaim_ring()
    assert st.reclaim_ring() == 0, "a second reclaim did work"
    assert not (set(st.transient_ring) & set(st.free_lru))
