"""The promotion into a PLANE-MAJOR arena, byte for byte against the shipped loader.

This path had no gate. engine/test_cold_path_real.py promotes record-major -> record-major, which is
ONE contiguous copy; the engine's hot arena is still plane-major, so ColdPool.promote takes a
different branch entirely -- twelve scatter copies -- and that branch is exactly the thing that could
make a mapping point at bytes which are not the expert's.

Reference is CB3Cache.load_slot, the loader the ordinary miss path uses. Both start from the same pack
record; all twelve planes must match.
"""
import os, sys
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")]
import torch
import cb3_moe as C3
from cb3_moe import PLANE_ORDER
from engine.cb3_cache import CB3Cache
from engine.cold_pool import ColdPool
from engine.codebook_sim import CodebookSim

PACK = os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin")
N = 20


def main():
    torch.manual_seed(77)
    cache = CB3Cache(PACK)
    pool = ColdPool(PACK, n_slots=8)
    ref = C3.CB3ArenaV2(4, "cuda", packed_scales=True); ref.sim = CodebookSim(3, "cuda")
    got = C3.CB3ArenaV2(4, "cuda", packed_scales=True); got.sim = ref.sim
    print(f"  pack {pool.n_records:,} records; reference CB3Cache.load_slot vs ColdPool.promote "
          f"into a PLANE-MAJOR arena ({type(ref).__name__}, packed_scales={ref.packed_scales})")
    gen = torch.Generator().manual_seed(77)
    bad = 0
    for i in range(N):
        rec = int(torch.randint(0, pool.n_records, (1,), generator=gen))
        layer, expert = divmod(rec, pool.n_experts_per_layer)
        # A: the shipped loader, from a staged CPU copy of the record
        staged = torch.empty(cache.record, dtype=torch.uint8)
        cache.read_into(memoryview(staged.numpy()), layer, expert)
        cache.load_slot(ref, 0, staged, non_blocking=False)
        # B: O_DIRECT into a cold slot, then promote into the plane-major arena
        cslot, cgen = pool.fetch((layer, expert), hot_slot=1)
        pool.promo.compute_done(cslot, cgen)
        _s, _g, ev = pool.promote((layer, expert), got, stream=None)
        torch.cuda.synchronize()
        pool.promo.promo_done(_s, _g)
        for nm in PLANE_ORDER:
            a, b = getattr(ref, nm)[0], getattr(got, nm)[1]
            if not torch.equal(a, b):
                n = int((a != b).sum())
                print(f"    MISMATCH rec {rec} (L{layer} e{expert}) plane {nm}: "
                      f"{n:,} of {a.numel():,} bytes differ")
                bad += 1
        if i == 0:
            print(f"    first record L{layer} e{expert}: all 12 planes "
                  f"{'match' if bad == 0 else 'DIFFER'}")
    pool.close()
    print(f"  {N} records checked, {bad} plane mismatches")
    print("  PLANE-MAJOR PROMOTION " + ("FAIL" if bad else "VERIFIED"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
