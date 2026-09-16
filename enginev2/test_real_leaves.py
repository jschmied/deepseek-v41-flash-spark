"""test_real_leaves.py -- the real leaves, stages 1-3: pinned staging, CB3 read, CB3 H2D.

Needs the box: a GPU and ~/dsv41-cb3/experts-cb3-s3.bin (211 GB). Skipped without them.

WHAT IS NON-CIRCULAR HERE, and what is not. The read is checked against an INDEPENDENT buffered
pread of the same offset -- a different syscall path, no pinning, no O_DIRECT -- so it cannot pass
by calling the code that wrote it. The H2D is checked for the nine WEIGHT planes the same way: they
are stored in the arena's own layout, so each is a byte-for-byte slice of the record.

The three SCALE planes are expanded on the device rather than copied, so nothing byte-level can
check them here. They are covered by the stage-5 gate instead, which is much stronger: job 255
established that prefill is bit-deterministic (eager vs eager was exactly 0.000e+00), so
RealLeaves-vs-v1 final logits must be EXACTLY equal, not merely close.

Run:  python -m pytest enginev2/test_real_leaves.py -q
"""

from __future__ import annotations

import os

import pytest
import torch

from .real import ALIGN, PinnedStagingPool, RealLeaves

CB3 = os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin")
needs_box = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.exists(CB3)),
    reason="needs a GPU and the CB3 cache")


@pytest.fixture(scope="module")
def leaves():
    RealLeaves._v1_on_path()
    from cb3_moe import CB3ArenaV2
    arena = CB3ArenaV2(4, "cuda")
    rl = RealLeaves(CB3, arena)
    yield rl
    rl.close()


@needs_box
def test_the_pool_is_pinned_and_aligned(leaves):
    """O_DIRECT requires an ALIGN-aligned destination and torch.empty promises nothing, so the
    buffer is over-allocated and the view starts at the first aligned address inside it."""
    pool = PinnedStagingPool(2, leaves.record)
    try:
        for sid in range(2):
            assert pool._buf[sid].is_pinned(), f"buffer {sid} is not page-locked"
            assert pool.aligned(sid), f"buffer {sid} view is not {ALIGN}-aligned"
            assert pool._view[sid].numel() == leaves.record
        assert pool.bytes_pinned == 2 * (leaves.record + ALIGN)
    finally:
        assert pool.at_rest()


@needs_box
def test_the_read_is_zero_copy_and_byte_exact(leaves):
    pool = PinnedStagingPool(2, leaves.record)
    key = (7, 123)
    st = leaves.read(key, pool)
    try:
        assert pool.same_buffer(st.sid, st.payload), (
            "read() returned a COPY of the staging buffer, not a view")
        assert pool.in_use(st.sid), "the lease was not held across the read"
        got = bytes(st.payload.numpy())
        off = leaves.cache.offset(*key)
        with open(CB3, "rb") as f:                 # independent path: buffered, unpinned
            f.seek(off)
            want = f.read(leaves.record)
        assert got == want, f"the O_DIRECT read differs from a buffered read at offset {off}"
    finally:
        st.release()
    assert pool.at_rest(), "the lease was not returned"


@needs_box
def test_the_h2d_puts_the_record_in_the_arena(leaves):
    pool = PinnedStagingPool(2, leaves.record)
    key, slot = (7, 123), 2
    st = leaves.read(key, pool)
    record = bytes(st.payload.numpy())
    leaves.h2d(slot, key, st)
    st.release()
    assert pool.at_rest()
    for name in ("w1_lo", "w1_hi", "w1_cb", "w3_lo", "w3_hi", "w3_cb",
                 "w2_lo", "w2_hi", "w2_cb"):
        lo, hi = leaves.cache.planes[name]
        dst = getattr(leaves.arena, name)[slot]
        got = dst.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        assert got == record[lo:lo + len(got)], f"arena plane {name} differs from the record"


@needs_box
def test_a_failed_read_returns_its_lease(leaves):
    """The lease must come back on the error path or the pool drains and every miss blocks with the
    device idle -- v1's ExpertStore carries the same comment for the same reason."""
    pool = PinnedStagingPool(1, leaves.record)
    with pytest.raises(Exception):
        leaves.read((99999, 0), pool)              # out of range: the pread loop must fail
    assert pool.at_rest(), "a failed read leaked its pinned buffer"
