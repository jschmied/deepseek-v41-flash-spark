"""CB3RecordArena must compute exactly what CB3ArenaV2 computes, and must not disturb it.

The record-major layout exists so a promotion is one contiguous copy and an O_DIRECT read lands in
its final slot. Neither is worth anything if the arithmetic moves, so this is the gate on that.
"""
import os, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
import torch
import cb3_moe as C3
from cb3_moe import DIM, PLANE_ORDER
from engine.codebook_sim import CodebookSim

WEIGHTS = ("w1_lo", "w1_hi", "w1_cb", "w3_lo", "w3_hi", "w3_cb", "w2_lo", "w2_hi", "w2_cb")
SCALES = ("s1", "s3", "s2")


def _fill(a, slots, gen):
    for nm in WEIGHTS:
        getattr(a, nm).random_(0, 255, generator=gen)
    # Sane UE8M0 exponents. Uniform 0..255 gives 2**127, every output becomes NaN and NaN != NaN
    # makes every comparison below vacuously false -- which is exactly how job 1030 wasted a run.
    for nm in SCALES:
        getattr(a, nm).random_(124, 131, generator=gen)


def _copy_into_record(plane, rec, slots):
    for nm in PLANE_ORDER:
        src = getattr(plane, nm).reshape(slots, -1)
        for s in range(slots):
            rec.slot_view(s, nm).view(-1).copy_(src[s])


def test_layout_matches_the_pack_manifest():
    """The record must equal the on-disk payload, or O_DIRECT placement is not possible."""
    off, per, payload = C3.plane_layout(packed_scales=True)
    assert payload == C3.CB3_BYTES_PER_SLOT_PACKED, (payload, C3.CB3_BYTES_PER_SLOT_PACKED)
    man = "/home/jschmied/dsv41-cb3/experts-cb3-s3.bin.json"
    if os.path.exists(man):
        import json
        m = json.load(open(man))
        assert payload == m["payload_bytes"], (payload, m["payload_bytes"])
        for nm, (b0, _) in off.items():
            assert b0 == m["planes"][nm][0], f"{nm}: {b0} != {m['planes'][nm][0]}"
    off_u, _, pay_u = C3.plane_layout(packed_scales=False)
    assert pay_u == C3.CB3_BYTES_PER_SLOT, (pay_u, C3.CB3_BYTES_PER_SLOT)


def test_record_arena_is_bitwise_identical_to_plane_major():
    slots = 16
    gen = torch.Generator(device="cuda").manual_seed(4)
    plane = C3.CB3ArenaV2(slots, "cuda"); plane.sim = CodebookSim(3, "cuda")
    _fill(plane, slots, gen)
    rec = C3.CB3RecordArena(slots, "cuda", packed_scales=False)
    rec.sim = plane.sim
    _copy_into_record(plane, rec, slots)
    assert C3._rstride(plane) == 0 and C3._rstride(rec) == rec.rstride
    for T in (1, 6):
        x = torch.randn(T, DIM, dtype=torch.bfloat16, device="cuda", generator=gen)
        sl = torch.randint(0, slots, (T, 6), dtype=torch.int32, device="cuda", generator=gen)
        w = torch.rand(T, 6, dtype=torch.float32, device="cuda", generator=gen)
        w = w / w.sum(-1, keepdim=True)
        a = C3.moe_forward_v3(x, sl, w, plane)
        b = C3.moe_forward_v3(x, sl, w, rec)
        assert torch.isfinite(a.float()).all(), "test data overflowed; comparison would be vacuous"
        assert torch.equal(a, b), f"T={T}: record-major differs, max|d| " \
                                  f"{(a.float()-b.float()).abs().max().item():.4g}"


def test_promotion_copy_reproduces_the_source_slot():
    slots = 8
    gen = torch.Generator(device="cuda").manual_seed(5)
    plane = C3.CB3ArenaV2(slots, "cuda"); plane.sim = CodebookSim(3, "cuda")
    _fill(plane, slots, gen)
    cold = C3.CB3RecordArena(slots, "cuda", packed_scales=False)
    hot = C3.CB3RecordArena(slots, "cuda", packed_scales=False)
    _copy_into_record(plane, cold, slots)
    cold.promote_into(hot, dst_slot=3, slot=5, non_blocking=False)
    torch.cuda.synchronize()
    assert torch.equal(hot.record(3), cold.record(5)), "the promotion copy is not byte-exact"
    for nm in PLANE_ORDER:
        assert torch.equal(hot.slot_view(3, nm), cold.slot_view(5, nm)), nm


def test_shipped_plane_major_path_is_unchanged():
    """_rstride must read 0 off an arena that has never heard of it, so the shipped arithmetic is
    byte-identical to what it was before RSTRIDE existed."""
    slots = 8
    gen = torch.Generator(device="cuda").manual_seed(6)
    a = C3.CB3ArenaV2(slots, "cuda"); a.sim = CodebookSim(3, "cuda")
    _fill(a, slots, gen)
    assert C3._rstride(a) == 0
    x = torch.randn(1, DIM, dtype=torch.bfloat16, device="cuda", generator=gen)
    w = torch.ones(1, 6, device="cuda") / 6
    o0 = C3.moe_forward_v3(x, torch.zeros(1, 6, dtype=torch.int32, device="cuda"), w, a)
    o7 = C3.moe_forward_v3(x, torch.full((1, 6), 7, dtype=torch.int32, device="cuda"), w, a)
    assert not torch.equal(o0, o7), "slot resolution is broken: every slot reads slot 0"
    assert torch.isfinite(o0.float()).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"  ok  {f.__name__}")
    print(f"  {len(fns)}/{len(fns)} passed")
