"""Does splitting the routed MoE into a resident phase and a missing phase change the result?

THE PLAN THIS TESTS. Jobs 400/405 put graph B at 69.4 ms/step of device time and job 460/465 put
PAIR residency at 94-95 % -- so most of the MoE could run while this layer's misses are still
loading, instead of after them. The split needs no Triton change: `build_routing_small` gives one
BM-block per distinct slot, residency is a property of the slot, and both kernels already early-out
on `if slot < 0: return`. So masking `block_slot` to -1 runs exactly the complementary half, twice,
into the same `parts` buffer, with the same `block_pair`, the same tiling and the same arithmetic.

If that is bitwise-identical to one launch, the split is safe and only scheduling remains. If it is
not, the plan is dead and no amount of engine work rescues it.

The arena is filled with random bytes on purpose: this is an EQUALITY test between two ways of
running the same kernels over the same bytes, not a numerical-correctness test. Valid weights are
not required and the full checkpoint is not on this box. Scales are pinned to one exponent so the
swiglu cannot produce NaN, which would make `torch.equal` meaningless (NaN != NaN).
"""
import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cb3_moe as C3
from cb3_moe import DIM as _DIM  # noqa: F401                                            # noqa: E402
from fp4_moe import DIM, _pick_bm, build_routing_small          # noqa: E402

torch.manual_seed(0)
S = 64
arena = C3.CB3ArenaV2(S, "cuda")
for name in ("w1_lo", "w1_hi", "w1_cb", "w3_lo", "w3_hi", "w3_cb", "w2_lo", "w2_hi", "w2_cb"):
    t = getattr(arena, name)
    t.copy_(torch.randint(0, 256, t.shape, dtype=torch.uint8, device="cuda"))
for name in ("s1", "s2", "s3"):
    getattr(arena, name).fill_(127)          # one exponent: keeps the swiglu finite

fails = 0
for T, K in ((6, 6), (1, 6), (5, 6)):
    x = (torch.randn(T, DIM, device="cuda") * 0.5).to(torch.bfloat16)
    slots = torch.stack([torch.randperm(S, device="cuda")[:K] for _ in range(T)]).to(torch.int32)
    wgt = torch.rand(T, K, device="cuda")

    ref = C3.moe_forward_v3(x, slots, wgt, arena)

    # --- the split, done exactly as the engine would ---
    P, BM = T * K, _pick_bm(T * K)
    block_slot, block_pair, NB = build_routing_small(slots, BM)
    # Residency is a slot property, so a block is wholly resident or wholly missing.
    resident_slots = torch.zeros(S, dtype=torch.bool, device="cuda")
    resident_slots[torch.randperm(S, device="cuda")[: int(S * 0.95)]] = True   # ~95 %, as measured
    blk_res = torch.where(block_slot >= 0, resident_slots[block_slot.long().clamp(min=0)],
                          torch.zeros_like(block_slot, dtype=torch.bool))
    bs_res = torch.where(blk_res, block_slot, torch.full_like(block_slot, -1))
    bs_mis = torch.where(blk_res, torch.full_like(block_slot, -1), block_slot)

    out = C3.moe_forward_v3_split(x, slots, wgt, arena, block_slot=block_slot,
                                  block_pair=block_pair, NB=NB, BM=BM,
                                  phase_masks=(bs_res, bs_mis))
    same = torch.equal(ref, out)

    # THE FORM THE ENGINE WILL ACTUALLY USE: two masked SLOT tensors, caller-owned h and parts
    # shared across the phases, one reduction at the end. This is what survives CUDA-graph capture,
    # because the two phases land in different graphs and must share the same memory.
    hbuf = torch.empty((P, C3.INTER), dtype=torch.bfloat16, device="cuda")
    pbuf = torch.empty((P, C3.DIM), dtype=torch.float32, device="cuda")
    C3.moe_v3_phase(x, slots, wgt, arena, hbuf, pbuf, routing=(bs_res, block_pair, NB))
    C3.moe_v3_phase(x, slots, wgt, arena, hbuf, pbuf, routing=(bs_mis, block_pair, NB))
    out2 = C3.moe_v3_reduce(pbuf, T, K)
    same2 = torch.equal(ref, out2)
    print(f"        phase API (shared h/parts, one reduce): exact match {same2}  "
          f"max |delta| {(ref.float() - out2.float()).abs().max():.3e}")
    fails += 0 if same2 else 1
    n_res = int(blk_res.sum())
    print(f"  T={T} K={K} P={P} BM={BM} NB={NB}  blocks {n_res} resident / {NB - n_res} missing  "
          f"exact match {same}  max |delta| {(ref.float() - out.float()).abs().max():.3e}")
    fails += 0 if same else 1

print("  SPLIT IS BITWISE" if not fails else f"  SPLIT DIVERGES on {fails} case(s)")
sys.exit(1 if fails else 0)
