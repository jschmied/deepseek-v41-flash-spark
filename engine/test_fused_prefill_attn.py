"""Unit test for the fused prefill attention: tools/decode_attn.prefill_attention against the
eager chain in engine/model.py `Model._softmax_attn` that it replaces under
DSV41_ATTN_FUSED_PREFILL=1.

The reference is the real method, called on a stand-in object that only carries `args.head_dim`,
so the test cannot drift away from the code it is guarding.

It will not be bit-identical and is not expected to be: the eager path computes the PV product as
an fp32 einsum, the kernel has to hand `tl.dot` a tensor-core dtype and so carries the
probabilities as a bf16 high/low pair (PV_SPLIT). That is a softmax reassociation plus ~16 mantissa
bits instead of 24, against a bf16 output whose own rounding step is 2^-8. So the test measures the
error and holds it under 2e-3 relative to the tensor's scale, and additionally requires:
  * no NaN and no Inf anywhere (the padding rows are the trap: all-masked scores are -inf, the max
    clamps to -1e30, and only the sink term keeps 0/inf = 0 out of NaN territory), and
  * an all-masked row is EXACTLY zero on both paths, not merely small.

Run:  python engine/test_fused_prefill_attn.py     (needs a GPU; allocates a few hundred MB)
"""
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from engine import model as M  # noqa: E402
from decode_attn import prefill_attention  # noqa: E402

fails = []

# (T, H, D, N, what it is there for)
SHAPES = [
    (7, 8, 64, 40, "tiny, T < ATTN_TILE"),
    (64, 32, 128, 128, "T exactly one ATTN_TILE"),
    (130, 16, 256, 200, "T not a multiple of ATTN_TILE (2 full tiles + 2 rows)"),
    (65, 8, 576, 136, "head dim 512+64, exercises the DA/DB split"),
    (192, 64, 512, 640, "the real prefill shape (H=64 D=512 N=640)"),
]


def make(T, H, D, N, seed):
    """Random inputs with the three mask rows that matter: all-masked (a padding row), fully
    visible, and one with a single visible key."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, H, D, generator=g, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, N, D, generator=g, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(T, N, generator=g, device="cuda") < 0.7
    dead = T // 2
    mask[dead] = False          # padding row: every key masked
    mask[0] = True              # nothing masked
    mask[-1] = False
    mask[-1, N // 3] = True     # exactly one visible key
    sink = torch.randn(H, generator=g, device="cuda", dtype=torch.float32) * 2.0
    return q, kv, mask, sink, dead


def reference(q, kv, mask, sink, D):
    """The eager path, called as the engine calls it (gate off)."""
    assert not M.ATTN_FUSED_PREFILL
    stub = types.SimpleNamespace(args=types.SimpleNamespace(head_dim=D))
    return M.Model._softmax_attn(stub, q, kv, mask, sink)


def kernel_launches(fn):
    """CUDA kernel launches of one call, from the profiler's device-side events."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events()
               if str(getattr(e, "device_type", "")) == "DeviceType.CUDA")


def median_ms(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
print(f"ATTN_TILE={M.ATTN_TILE}  gate DSV41_ATTN_FUSED_PREFILL="
      f"{os.environ.get('DSV41_ATTN_FUSED_PREFILL', '0')}\n")

for i, (T, H, D, N, why) in enumerate(SHAPES):
    q, kv, mask, sink, dead = make(T, H, D, N, seed=1234 + i)
    scale = D ** -0.5
    ref = reference(q, kv, mask, sink, D)
    got = prefill_attention(q, kv, mask, sink, scale)

    r, g = ref.float(), got.float()
    d = (g - r).abs()
    max_abs = float(d.max())
    scale_r = float(r.abs().max())
    max_rel = max_abs / scale_r if scale_r else 0.0
    norm_rel = float((g - r).norm() / r.norm()) if float(r.norm()) else 0.0
    # how far apart in bf16 steps: the two paths round independently, so 1 step is the floor
    ulps = (got.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
    bad = int((ulps > 1).sum())

    finite = bool(torch.isfinite(got).all() and torch.isfinite(ref).all())
    zero_ref = bool((ref[dead] == 0).all())
    zero_got = bool((got[dead] == 0).all())

    ok = max_rel < 2e-3 and finite and zero_ref and zero_got
    print(f"[{'PASS' if ok else 'FAIL'}] T={T:<4d} H={H:<3d} D={D:<4d} N={N:<4d}  {why}")
    print(f"        max|d| {max_abs:.3e}   max|d|/max|ref| {max_rel:.3e}   "
          f"||d||/||ref|| {norm_rel:.3e}   max|ref| {scale_r:.3f}")
    print(f"        bf16 steps apart: >1 step on {bad}/{ref.numel()} elements "
          f"({100.0 * bad / ref.numel():.2f} %)   finite: {finite}")
    print(f"        all-masked row {dead}: exactly zero  ref={zero_ref} kernel={zero_got}")
    if not ok:
        fails.append(f"{T}x{H}x{D}x{N}")

    if (T, H, D, N) == (192, 64, 512, 640):
        nk_ref = kernel_launches(lambda: reference(q, kv, mask, sink, D))
        nk_new = kernel_launches(lambda: prefill_attention(q, kv, mask, sink, scale))
        t_ref = median_ms(lambda: reference(q, kv, mask, sink, D))
        t_new = median_ms(lambda: prefill_attention(q, kv, mask, sink, scale))
        print(f"        launches/call: eager {nk_ref}  fused {nk_new}  "
              f"({nk_ref / max(nk_new, 1):.0f}x fewer)")
        print(f"        median GPU time/call: eager {t_ref:.3f} ms  fused {t_new:.3f} ms  "
              f"({t_ref / t_new:.2f}x)  [box is shared, treat as indicative]")
    print()

# The gate itself: with it on, `_softmax_attn` must return the kernel's result and nothing else.
q, kv, mask, sink, dead = make(96, 64, 512, 640, seed=99)
stub = types.SimpleNamespace(args=types.SimpleNamespace(head_dim=512))
M.ATTN_FUSED_PREFILL = True
try:
    gated = M.Model._softmax_attn(stub, q, kv, mask, sink)
finally:
    M.ATTN_FUSED_PREFILL = False
same = bool(torch.equal(gated, prefill_attention(q, kv, mask, sink, 512 ** -0.5)))
print(f"[{'PASS' if same else 'FAIL'}] DSV41_ATTN_FUSED_PREFILL routes _softmax_attn to the kernel "
      f"(bit-identical: {same})")
if not same:
    fails.append("gate")

print()
print("FAILURES: " + ", ".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
