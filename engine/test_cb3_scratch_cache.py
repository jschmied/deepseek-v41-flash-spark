"""test_cb3_scratch_cache.py -- the layer-scoped unpack cache must change speed, never numbers.

`moe_forward_prefill` unpacks a CB3 expert into packed FP4 before the FP4 MoE kernels run, once per
(layer, chunk) call. Layer-major prefill runs ~7 chunks over nearly the same expert set, so the
2026-09-14 nsys trace shows up to 48,384 expert-unpacks where 7,600 are needed -- 1.61 TB of traffic
and 7.44 s, the largest single GPU consumer in a 40.3 s GPU-busy prefill.

DSV41_CB3_SCRATCH_SLOTS keeps a chunk's unpacked experts alive for the rest of the layer. That is
only safe if two things hold, and both are what this file checks:

  * the cached path is BIT-IDENTICAL to the batched one -- it is the same kernel over the same
    bytes, so anything other than bit-identical is a bug, not a tolerance;
  * a scratch entry dies the moment its arena slot is given to another expert. Miss that and a
    chunk silently computes with the previous expert's weights -- no exception, just wrong tokens.

Run:  python -m engine.test_cb3_scratch_cache
"""

from __future__ import annotations

import os
import sys

os.environ.pop("DSV41_CB3_SCRATCH_SLOTS", None)

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import cb3_moe as C3          # noqa: E402
import fp4_moe as F4          # noqa: E402
from engine.codebook_sim import CodebookSim   # noqa: E402

S = 24          # experts in the test arena
T = 40          # tokens per "chunk" -- P = 240, above PREFILL_MIN_P so the prefill path is taken
K = 6


def build_arena(seed: int = 0):
    torch.manual_seed(seed)
    a = C3.CB3ArenaV2(S, "cuda")
    a.sim = CodebookSim(3, "cuda")
    ref = F4.ExpertArena(1, "cuda")
    shapes = [(ref.w1.shape[1:], ref.s1.shape[1:]), (ref.w2.shape[1:], ref.s2.shape[1:]),
              (ref.w3.shape[1:], ref.s3.shape[1:])]
    for s in range(S):
        g = []
        for wsh, ssh in shapes:
            g.append(torch.randint(0, 255, wsh, dtype=torch.uint8, device="cuda"))
            g.append(torch.randint(120, 136, ssh, dtype=torch.uint8, device="cuda"))
        a.load_slot(s, g[0], g[1], g[2], g[3], g[4], g[5])
    return a


def chunks(seed: int = 1):
    """Three chunks of one layer with heavily overlapping expert sets, like a real layer."""
    torch.manual_seed(seed)
    out = []
    for c in range(3):
        lo = 0 if c < 2 else 4                 # chunk 3 drops the first four experts
        slots = torch.randint(lo, S, (T, K), dtype=torch.int32, device="cuda")
        x = (torch.randn(T, F4.DIM, device="cuda") * 0.5).to(torch.bfloat16)
        w = torch.rand(T, K, device="cuda")
        out.append((x, slots, w))
    return out


def run(arena, cs, scratch_slots: int):
    """One layer: three chunks, with the unpack accounted."""
    C3.SCRATCH_SLOTS = scratch_slots
    arena._scratch = None                       # force a fresh scratch at this size
    arena.scratch_epoch()
    n = {"calls": 0, "experts": 0}
    orig = C3._unpack_into

    def counted(ar, src, sc, dst=None):
        n["calls"] += 1
        n["experts"] += int(src.numel())
        return orig(ar, src, sc, dst)

    C3._unpack_into = counted
    try:
        ys = [C3.moe_forward_prefill(x, sl, w, arena) for x, sl, w in cs]
    finally:
        C3._unpack_into = orig
    torch.cuda.synchronize()
    return ys, n


def test_identical_and_fewer_unpacks():
    a = build_arena()
    cs = chunks()
    y_off, n_off = run(a, cs, 0)
    y_on, n_on = run(a, cs, 64)
    for i, (u, v) in enumerate(zip(y_off, y_on)):
        assert torch.equal(u, v), f"chunk {i}: cached output differs from batched"
    assert n_on["experts"] < n_off["experts"], (n_off, n_on)
    print(f"  3 chunks of one layer: batched unpacks {n_off['experts']} experts in "
          f"{n_off['calls']} calls, cached {n_on['experts']} in {n_on['calls']} -- "
          f"{n_off['experts'] / n_on['experts']:.2f}x less, output bit-identical  OK")


def test_reload_invalidates():
    """Overwrite an arena slot between chunks: the cache must not serve the old expert."""
    a = build_arena()
    cs = chunks()
    x, slots, w = cs[0]

    C3.SCRATCH_SLOTS = 64
    a._scratch = None
    a.scratch_epoch()
    C3.moe_forward_prefill(x, slots, w, a)              # populates the cache

    victim = int(slots[0, 0])
    torch.manual_seed(99)
    ref = F4.ExpertArena(1, "cuda")
    shapes = [(ref.w1.shape[1:], ref.s1.shape[1:]), (ref.w2.shape[1:], ref.s2.shape[1:]),
              (ref.w3.shape[1:], ref.s3.shape[1:])]
    g = []
    for wsh, ssh in shapes:
        g.append(torch.randint(0, 255, wsh, dtype=torch.uint8, device="cuda"))
        g.append(torch.randint(120, 136, ssh, dtype=torch.uint8, device="cuda"))
    a.load_slot(victim, g[0], g[1], g[2], g[3], g[4], g[5])   # a different expert now lives there
    assert victim not in a._scratch_of, "load_slot did not drop the stale scratch entry"
    y_cached = C3.moe_forward_prefill(x, slots, w, a)

    C3.SCRATCH_SLOTS = 0                                # the batched path cannot be stale
    a._scratch = None
    a.scratch_epoch()
    y_fresh = C3.moe_forward_prefill(x, slots, w, a)
    assert torch.equal(y_cached, y_fresh), \
        "after reloading an arena slot the cached path served the OLD expert's weights"
    print(f"  arena slot {victim} reloaded mid-layer -> cache dropped it, output matches fresh  OK")


def test_scratch_overflow_is_correct():
    """More experts than scratch slots: must fall back, still exactly right."""
    a = build_arena()
    cs = chunks()
    y_ref, _ = run(a, cs, 0)
    y_small, n = run(a, cs, 8)     # 8 < the ~20 distinct experts a chunk touches
    for i, (u, v) in enumerate(zip(y_ref, y_small)):
        assert torch.equal(u, v), f"chunk {i}: undersized scratch changed the output"
    print(f"  scratch smaller than a chunk's expert set -> falls back, bit-identical  OK")


if __name__ == "__main__":
    print("CB3 layer-scoped unpack cache:")
    fails = 0
    for fn in (test_identical_and_fewer_unpacks, test_reload_invalidates,
               test_scratch_overflow_is_correct):
        try:
            fn()
        except Exception as exc:                        # noqa: BLE001
            fails += 1
            print(f"  FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print("ALL OK" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
