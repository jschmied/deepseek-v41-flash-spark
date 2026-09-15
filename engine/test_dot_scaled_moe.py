"""test_dot_scaled_moe.py -- A/B of the DSV41_DOT_SCALED arm of tools/fp4_moe.py.

The gated arm replaces the software FP4 decode (`cvt.rn.f16x2.e2m1x2` + prmt + a per-32-K-group
multiply on the [BM, BN] fp32 partial) with one `tl.dot_scaled` per 128-K step. Bit-identical output
is NOT expected -- the UE8M0 factor moves from the accumulator onto the weight operand and the
tensor core sums the four groups itself -- so the bar is: the new path must be no further from
`moe_forward_reference` (dequant + torch bf16 GEMM) than the current path is.

Random packed-FP4 experts only: no checkpoint, ~19 MB per expert slot, so this runs anywhere.

Run:  python engine/test_dot_scaled_moe.py
      python engine/test_dot_scaled_moe.py --iters 30 --slots 12
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from fp4_linear import fp4_linear, quantize_to_fp4  # noqa: E402
from fp4_moe import DIM, INTER, ExpertArena, moe_forward, moe_forward_reference  # noqa: E402

TOPK = 6


def fill_random_experts(arena: ExpertArena, gen: torch.Generator) -> None:
    """Uniformly random nibbles + a spread of UE8M0 exponents. The spread is the point: a constant
    scale would hide any error in how an arm applies it.

    Exponent base 120 (2^-7), not 127: the FP4 grid has std ~2.5, so at 2^0 a K=5120 dot has std
    ~180 and every token saturates the SwiGLU clamp at +-10, which makes the whole test a comparison
    of two clamped constants. 2^-7 puts |gate| around 1.4, i.e. where a real expert sits."""
    for t in (arena.w1, arena.w3, arena.w2):
        t.copy_(torch.randint(0, 256, t.shape, generator=gen, dtype=torch.int32).to(torch.uint8))
    for t in (arena.s1, arena.s3, arena.s2):
        t.copy_((120 + torch.randint(-2, 3, t.shape, generator=gen, dtype=torch.int32)).to(torch.uint8))
    torch.cuda.synchronize()


def routing(T: int, n_slots: int, gen: torch.Generator, device) -> tuple[torch.Tensor, torch.Tensor]:
    slots = torch.stack([torch.randperm(n_slots, generator=gen)[:TOPK] for _ in range(T)]).to(torch.int32)
    w = torch.rand((T, TOPK), generator=gen)
    return slots.to(device), (w / w.sum(1, keepdim=True)).to(device)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def bench(fn, iters: int) -> float:
    """Median ms per call over `iters` timed calls, each bracketed by its own CUDA events."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out)


def dense_ab(gen: torch.Generator, device, iters: int) -> None:
    """Same A/B for tools/fp4_linear._fp4_linear_kernel (1.64 s of the prefill), three real
    projection shapes: wkv (small N), an ffn w1/w3 and a wo_b-sized square."""
    # fp32 reference, not a bf16 torch GEMM: at M=6 cuBLAS picks a different kernel per N and the
    # bf16 reference then agrees with the Triton output to 5e-8 for one shape and 2e-3 for the next,
    # which says nothing about the two arms. Against fp32 both arms show their real distance, which
    # is the bf16 store (~2e-3) for every shape.
    print("\n== dense fp4_linear (x @ W^T), vs W.dequant() in fp32")
    print(f"{'N':>6} {'K':>6} {'M':>6} {'rel old':>10} {'rel new':>10} {'old ms':>9} {'new ms':>9} {'speedup':>8}")
    for N, K in ((512, 5120), (2304, 5120), (5120, 5120)):
        ref = (torch.randn((N, K), generator=gen) * 0.02).to(device)
        W = quantize_to_fp4(ref)
        wd = W.dequant()
        for M in (6, 2048):
            x = torch.randn((M, K), generator=gen).to(torch.bfloat16).to(device)
            y_old = fp4_linear(x, W, scaled=False)
            y_new = fp4_linear(x, W, scaled=True)
            assert not torch.isnan(y_new.float()).any(), f"NaN at N={N} M={M}"
            y_ref = x.float() @ wd.float().T
            ms_old = bench(lambda: fp4_linear(x, W, scaled=False), iters)
            ms_new = bench(lambda: fp4_linear(x, W, scaled=True), iters)
            print(f"{N:>6} {K:>6} {M:>6} {rel(y_old, y_ref):>10.4e} {rel(y_new, y_ref):>10.4e} "
                  f"{ms_old:>9.3f} {ms_new:>9.3f} {ms_old / ms_new:>7.2f}x")
            assert rel(y_new, y_ref) < rel(y_old, y_ref) * 1.05, f"dense dot_scaled is worse at N={N} M={M}"
        del ref, W, wd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--slots", type=int, default=8)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        raise SystemExit(0)
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(4242)
    arena = ExpertArena(args.slots, device)
    fill_random_experts(arena, gen)
    print(f"arena: {args.slots} random slots x {arena.bytes_per_slot / 1e6:.2f} MB, "
          f"DIM={DIM} INTER={INTER} TOPK={TOPK}")

    # ------------------------------------------------------------------ accuracy vs the reference
    print("\n== distance from moe_forward_reference (dequant + torch bf16 GEMM)")
    print(f"{'T':>5} {'rel old':>10} {'rel new':>10} {'new-vs-old':>11} "
          f"{'maxabs n-o':>11} {'maxrel n-o':>11} {'max|ref|':>9}")
    worse = []
    for T in (1, 6, 64, 512):
        x = torch.randn((T, DIM), generator=gen).to(torch.bfloat16).to(device)
        slots, w = routing(T, args.slots, gen, device)
        y_old = moe_forward(x, slots, w, arena, scaled=False)
        y_new = moe_forward(x, slots, w, arena, scaled=True)
        y_ref = moe_forward_reference(x, slots, w, arena)
        assert not torch.isnan(y_new.float()).any() and not torch.isinf(y_new.float()).any(), f"NaN/Inf at T={T}"
        r_old, r_new = rel(y_old, y_ref), rel(y_new, y_ref)
        d = (y_new.float() - y_old.float()).abs()
        denom = y_old.float().abs().clamp_min(1e-3)  # |y| spans ~1e-4..1e1; a bare ratio is all noise
        print(f"{T:>5} {r_old:>10.4e} {r_new:>10.4e} {rel(y_new, y_old):>11.4e} "
              f"{d.max().item():>11.4e} {(d / denom).max().item():>11.4e} {y_ref.float().abs().max().item():>9.3f}")
        if r_new > r_old * 1.05:
            worse.append((T, r_old, r_new))
    assert not worse, f"dot_scaled is further from the reference than the current kernel: {worse}"
    print("accuracy: PASS (new path is no further from the reference than the old one)")

    # ------------------------------------------------------------------ determinism
    # Same property the [TOPK, T, DIM] parts buffer exists for (see the comment above
    # _moe_down_kernel): a token's output may not depend on how many tokens share the call.
    print("\n== chunk invariance of the dot_scaled arm")
    T = 512
    x = torch.randn((T, DIM), generator=gen).to(torch.bfloat16).to(device)
    slots, w = routing(T, args.slots, gen, device)
    full = moe_forward(x, slots, w, arena, scaled=True)
    same = all(torch.equal(moe_forward(x, slots, w, arena, scaled=True), full) for _ in range(3))
    bad = [m for m in (1, 3, 6, 7, 17, 64, 148, 256, 300)
           if not torch.equal(moe_forward(x[:m], slots[:m], w[:m], arena, scaled=True), full[:m])]
    print(f"  run-to-run bit-identical: {same}")
    print(f"  prefix bit-identical for every call size: {not bad}" + (f" (differs at M={bad})" if bad else ""))
    assert same and not bad, "dot_scaled arm is not chunk-invariant"
    print("chunk invariance: PASS")

    # ------------------------------------------------------------------ timing
    print(f"\n== timing (CUDA events, median of {args.iters} calls)")
    print(f"{'T':>5} {'old ms':>9} {'new ms':>9} {'speedup':>8} {'old TFLOPS':>11} {'new TFLOPS':>11}")
    for T in (1, 6, 64, 512, 2048):
        x = torch.randn((T, DIM), generator=gen).to(torch.bfloat16).to(device)
        slots, w = routing(T, args.slots, gen, device)
        ms_old = bench(lambda: moe_forward(x, slots, w, arena, scaled=False), args.iters)
        ms_new = bench(lambda: moe_forward(x, slots, w, arena, scaled=True), args.iters)
        flop = 3 * 2 * (T * TOPK) * INTER * DIM  # w1 + w3 + w2, 2 flop per MAC
        print(f"{T:>5} {ms_old:>9.3f} {ms_new:>9.3f} {ms_old / ms_new:>7.2f}x "
              f"{flop / ms_old / 1e9:>11.2f} {flop / ms_new / 1e9:>11.2f}")
    # ------------------------------------------------------------------ the gate itself
    import fp4_moe
    want = os.environ.get("DSV41_DOT_SCALED", "0") == "1"
    assert fp4_moe.DOT_SCALED is want, f"DSV41_DOT_SCALED={os.environ.get('DSV41_DOT_SCALED')} -> {fp4_moe.DOT_SCALED}"
    x = torch.randn((64, DIM), generator=gen).to(torch.bfloat16).to(device)
    slots, w = routing(64, args.slots, gen, device)
    assert torch.equal(moe_forward(x, slots, w, arena), moe_forward(x, slots, w, arena, scaled=want))
    print(f"\ngate: DSV41_DOT_SCALED={'1' if want else '0'} -> "
          f"{'dot_scaled' if want else 'software decode'} arm, and the default call matches it")

    dense_ab(gen, device, args.iters)
    print(f"\npeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
