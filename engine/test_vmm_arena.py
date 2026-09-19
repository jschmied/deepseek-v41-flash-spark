"""The real CB3 kernel on a VMM host-NUMA arena: bitwise, and at what cost.

The streaming probe put a VMM host-NUMA range at 1.024x cudaMalloc. That distinction has already
mattered once -- the pinned pool measured 96.5 % of device on a stream and 0.92-1.10x on the real
250-register gather kernel -- so this runs the actual kernel at the actual shapes.

If it is bitwise and within ~10 %, the arena can be the O_DIRECT destination permanently and the
promotion disappears rather than being scheduled better.
"""
import os, statistics, sys
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")]
import torch
import cb3_moe as C3
from cb3_moe import DIM, PLANE_ORDER
from engine.codebook_sim import CodebookSim
from engine.vmm_alloc import HostNumaAlloc

SLOTS, KTOP, REPS = 384, 6, 40
WEIGHTS = ("w1_lo", "w1_hi", "w1_cb", "w3_lo", "w3_hi", "w3_cb", "w2_lo", "w2_hi", "w2_cb")


def main():
    gen = torch.Generator(device="cuda").manual_seed(19)
    dev = C3.CB3RecordArena(SLOTS, "cuda", packed_scales=False)
    dev.sim = CodebookSim(3, "cuda")
    for nm in WEIGHTS:
        dev.plane(nm)  # touch
    dev.buf.random_(0, 255, generator=gen)
    # sane UE8M0 exponents, or every output is NaN and the comparison is vacuous
    for nm in ("s1", "s3", "s2"):
        for s in range(SLOTS):
            dev.slot_view(s, nm).random_(124, 131, generator=gen)

    alloc = HostNumaAlloc(SLOTS * dev.rstride)
    print(f"  VMM granularity {alloc.granularity/2**20:.0f} MiB, arena "
          f"{alloc.nbytes/2**30:.2f} GiB, 4096-aligned {alloc.tensor.data_ptr() % 4096 == 0}")
    vmm = C3.CB3RecordArena(SLOTS, "cuda", packed_scales=False, buf=alloc.tensor)
    vmm.sim = dev.sim
    vmm.buf.copy_(dev.buf)
    assert torch.equal(vmm.buf.to("cuda"), dev.buf), "the two arenas do not hold the same bytes"
    assert C3._rstride(vmm) == vmm.rstride == C3._rstride(dev)

    shapes = [("T=1", 1, "normal"), ("T=6", 6, "normal"),
              ("T=6 high-distinct", 6, "distinct"), ("T=24", 24, "normal")]
    bad = 0
    for tag, T, mode in shapes:
        P = T * KTOP
        bm = C3._pick_bm(P)
        calls = []
        for _ in range(6):
            x = torch.randn(T, DIM, dtype=torch.bfloat16, device="cuda", generator=gen)
            if mode == "distinct":
                sl = torch.randperm(SLOTS, device="cuda", generator=gen)[:P].to(torch.int32).view(T, KTOP)
            else:
                sl = torch.randint(0, SLOTS, (T, KTOP), dtype=torch.int32, device="cuda", generator=gen)
            w = torch.rand(T, KTOP, dtype=torch.float32, device="cuda", generator=gen)
            calls.append((x, sl, w / w.sum(-1, keepdim=True)))
        x, sl, w = calls[0]
        a = C3.moe_forward_v3(x, sl, w, dev, block_m=bm)
        b = C3.moe_forward_v3(x, sl, w, vmm, block_m=bm)
        fin = torch.isfinite(a.float()).all().item()
        eq = torch.equal(a, b)
        if not (fin and eq):
            bad += 1
            print(f"    {tag}: finite {fin} bitwise {eq} "
                  f"max|d| {(a.float()-b.float()).abs().max().item():.4g}")

        def bench(arena):
            for c in calls[:3]:
                C3.moe_forward_v3(c[0], c[1], c[2], arena, block_m=bm)
            torch.cuda.synchronize()
            per = []
            for i in range(REPS):
                c = calls[i % len(calls)]
                s0 = torch.cuda.Event(True); e0 = torch.cuda.Event(True)
                s0.record(); C3.moe_forward_v3(c[0], c[1], c[2], arena, block_m=bm); e0.record()
                torch.cuda.synchronize(); per.append(s0.elapsed_time(e0))
            return statistics.median(sorted(per))
        md, mv = bench(dev), bench(vmm)
        print(f"  {tag:20s} device {md:8.3f} ms   VMM host-NUMA {mv:8.3f} ms  "
              f"({mv/md-1:+6.1%})   bitwise {eq}")
    alloc.free()
    print(f"  VMM ARENA " + ("FAIL" if bad else "VERIFIED (bitwise at every shape)"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
