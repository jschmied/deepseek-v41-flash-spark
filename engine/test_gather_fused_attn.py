"""Unit test for the gather-fused prefill attention: tools/decode_attn.prefill_attention_gather
against tools/decode_attn.prefill_attention, the kernel it extends.

`prefill_attention` is the arm DSV41_ATTN_FUSED_PREFILL=1 already selects, and
engine/test_fused_prefill_attn.py validates IT against the eager `Model._softmax_attn`. So the bar
here is different and much harder: the gather-fused kernel changes only how a key ROW is addressed
(ring row (pos - 127 + j) % RING, compressed row idx[t, j], instead of row j of a pre-gathered
[T, 640, 512] copy), not one step of the arithmetic -- same order of accumulation, same PV split,
same sink. So it must be BIT-IDENTICAL to `prefill_attention` on the same inputs, and this test
asserts equality, not a tolerance. The error against the eager path is printed too, but only so the
two tests are comparable; it is inherited from `prefill_attention`, unchanged.

The reference builds `kv_all` the way engine/model.py `attention` does, through the real
`Model._window_positions`, so the test cannot drift away from the code it is guarding.

What the cases are for:
  * the real prefill shape (T>=192, H=64, D=512, 128 window + 512 compressed),
  * ring WRAPAROUND -- a small ring where pos % RING wraps inside a single query's 128-row window,
    which is the one piece of index arithmetic that only exists in the kernel,
  * a head dim that is not a power of two (the DA/DB split),
  * a layer with no compressed side at all (w.ratio == 0 -> idx is None),
  * an all-masked row, which must be EXACTLY zero and not NaN on both paths.

Run:  python engine/test_gather_fused_attn.py    (needs a GPU; allocates a few hundred MB)
"""
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from engine import model as M  # noqa: E402
from decode_attn import prefill_attention, prefill_attention_gather  # noqa: E402

fails = []

# (T, H, D, window, k, n_c, RING, S, what it is there for)
CASES = [
    (192, 64, 512, 128, 512, 2048, 4096, 0, "real prefill shape, chunk at position 0"),
    (192, 64, 512, 128, 512, 2048, 4096, 1907, "real shape, S not a multiple of anything"),
    (192, 64, 512, 128, 512, 2048, 256, 200, "WRAPAROUND: pos % RING wraps inside the window"),
    (256, 64, 512, 128, 512, 2048, 384, 300, "WRAPAROUND, chunk > half the ring"),
    (65, 8, 576, 128, 64, 512, 4096, 3, "head dim 512+64 (DA/DB split), short chunk"),
    (192, 64, 512, 128, 0, 0, 4096, 640, "no compressed side (a w.ratio == 0 layer)"),
    (96, 8, 512, 128, 512, 3, 4096, 700, "n_c=3 < index_topk: _pad_topk's all -1 tail, ROW_MASK=1"),
]


def make(T, H, D, window, k, n_c, ring_size, S, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, H, D, generator=g, device="cuda", dtype=torch.bfloat16)
    ring = torch.randn(ring_size, D, generator=g, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, generator=g, device="cuda", dtype=torch.float32) * 2.0
    pos = torch.arange(S, S + T, device="cuda")

    # the window mask exactly as `attention` builds it (win_lo = 0)
    stub = types.SimpleNamespace(args=types.SimpleNamespace(window_size=window), dev="cuda")
    wpos = M.Model._window_positions(stub, pos)          # [T, window], -1 where out of range
    wmask = wpos >= 0

    if k:
        ckv = torch.randn(n_c, D, generator=g, device="cuda", dtype=torch.bfloat16)
        # the indexer's output: k distinct rows per query, sorted ascending, with a -1 tail --
        # `_pad_topk` pads to index_topk and the rows past compress_lens are set to -1, and both
        # land at the end because the topk is sorted ascending first
        idx = torch.rand(T, n_c, generator=g, device="cuda").argsort(dim=-1)[:, :k].sort(-1).values
        if idx.size(1) < k:   # n_c < index_topk early in a prompt: `_pad_topk` fills with -1
            idx = torch.cat([idx, idx.new_full((T, k - idx.size(1)), -1)], dim=1)
        keep = torch.randint(k // 2, k + 1, (T, 1), generator=g, device="cuda")
        idx = torch.where(torch.arange(k, device="cuda")[None, :] < keep, idx,
                          torch.full_like(idx, -1))
        cmask = idx >= 0
        mask = torch.cat([wmask, cmask], dim=1)
    else:
        ckv, idx = None, None
        mask = wmask

    # one fully-masked row: the trap is that its scores are all -inf, so only the clamp + the sink
    # term keep 0/inf = 0 out of NaN territory
    dead = T // 2
    mask[dead] = False
    return q, ring, ckv, idx, pos, wpos, mask, sink, dead


def materialise(ring, ckv, idx, wpos, ring_size):
    """`kv_all` exactly as engine/model.py `attention` builds it, before this change."""
    wkv = ring[wpos.clamp_min(0) % ring_size]
    if idx is None:
        return wkv
    return torch.cat([wkv, ckv[idx.clamp_min(0)]], dim=1)


def median_ms(fn, reps=15):
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
print(f"triton block: BLOCK_H={os.environ.get('DSV41_PREFILL_ATTN_BLOCK_H', 16)} "
      f"BLOCK_N={os.environ.get('DSV41_PREFILL_ATTN_BLOCK_N', 32)}\n")

for i, (T, H, D, window, k, n_c, ring_size, S, why) in enumerate(CASES):
    q, ring, ckv, idx, pos, wpos, mask, sink, dead = make(
        T, H, D, window, k, n_c, ring_size, S, seed=4321 + i)
    scale = D ** -0.5

    kv_all = materialise(ring, ckv, idx, wpos, ring_size)
    ref = prefill_attention(q, kv_all, mask, sink, scale)            # the validated fused kernel
    got = prefill_attention_gather(q, ring, ckv, idx, S, mask, sink, scale, window, ring_size)

    same = bool(torch.equal(ref, got))
    d = (got.float() - ref.float()).abs()
    max_abs = float(d.max())
    scale_r = float(ref.float().abs().max())
    max_rel = max_abs / scale_r if scale_r else 0.0
    finite = bool(torch.isfinite(got).all())
    nan = int(torch.isnan(got).sum())
    zero_dead = bool((got[dead] == 0).all())

    ok = same and finite and not nan and zero_dead
    print(f"[{'PASS' if ok else 'FAIL'}] T={T:<4d} H={H:<3d} D={D:<4d} win={window} k={k:<4d} "
          f"RING={ring_size:<5d} S={S:<5d}  {why}")
    print(f"        vs prefill_attention on the same KV: bit-identical={same}  "
          f"max|d| {max_abs:.3e}  max|d|/max|ref| {max_rel:.3e}")
    print(f"        finite={finite}  NaN={nan}  all-masked row {dead} exactly zero={zero_dead}")
    if not ok:
        fails.append(f"{T}x{H}x{D} RING={ring_size} S={S}")

    # the error the fused arm carries against eager, for comparison with
    # engine/test_fused_prefill_attn.py -- inherited, not introduced here
    if i < 3:
        assert not M.ATTN_FUSED_PREFILL
        est = types.SimpleNamespace(args=types.SimpleNamespace(head_dim=D))
        eager = M.Model._softmax_attn(est, q, kv_all, mask, sink)
        e = (got.float() - eager.float()).abs()
        se = float(eager.float().abs().max())
        print(f"        vs the EAGER path: max|d| {float(e.max()):.3e}  "
              f"max|d|/max|ref| {float(e.max()) / se:.3e}  "
              f"||d||/||ref|| {float((got.float() - eager.float()).norm() / eager.float().norm()):.3e}")
        del eager

    del kv_all, ref, got
    torch.cuda.empty_cache()
    print()

# -- timing ------------------------------------------------------------------------------------
# Two questions, measured separately because they need different methods.
# 1) Does addressing the rows in-kernel make the K-LOOP slower than reading a pre-gathered kv_all?
#    Strict ABAB interleave, one timed call per arm per rep: the box is shared, and blocked A/B
#    (15 reps of one arm, then the other) reads anywhere from 0.79x to 1.13x on the same code --
#    the spread is measurement order and allocator churn, not the kernel.
# 2) What the engine actually pays per layer per chunk, gather + cat + K-loop. Timed on its own
#    because it allocates ~2x kv_all every call, which perturbs anything interleaved with it.
def abab(fa, fb, reps=60):
    def once(fn):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        return a.elapsed_time(b)

    for _ in range(10):
        fa(); fb()
    torch.cuda.synchronize()
    ta, tb = [], []
    for _ in range(reps):          # one timed call each, strictly alternating
        ta.append(once(fa)); tb.append(once(fb))
    ta.sort(); tb.sort()
    return ta[reps // 2], tb[reps // 2]


print("-- timing (shared box, indicative) --")
for T in (192, 1024):
    H, D, window, k, n_c, ring_size, S = 64, 512, 128, 512, 4096, 4096, 0
    q, ring, ckv, idx, pos, wpos, mask, sink, dead = make(
        T, H, D, window, k, n_c, ring_size, S, seed=555)
    scale = D ** -0.5
    kv_all = materialise(ring, ckv, idx, wpos, ring_size)
    t_k, t_g = abab(lambda: prefill_attention(q, kv_all, mask, sink, scale),
                    lambda: prefill_attention_gather(
                        q, ring, ckv, idx, S, mask, sink, scale, window, ring_size))
    t_f = median_ms(lambda: prefill_attention(
        q, materialise(ring, ckv, idx, wpos, ring_size), mask, sink, scale))
    print(f"   T={T:<5d} kv_all {kv_all.numel() * 2 / 2**20:6.0f} MiB    "
          f"K-loop only {t_k:7.3f} ms | gather-fused {t_g:7.3f} ms ({t_g / t_k:.3f}x the K-loop) | "
          f"gather+cat+K-loop {t_f:7.3f} ms ({t_f / t_g:.2f}x end to end)")
    del kv_all, q, ring, ckv, idx, mask
    torch.cuda.empty_cache()
print()

# The plumbing in engine/model.py: `Model._gather_attn` must feed the kernel the model's own scale,
# window and RING, and nothing else.
T, H, D, window, k, n_c, ring_size, S = 96, 64, 512, 128, 512, 2048, 4096, 777
q, ring, ckv, idx, pos, wpos, mask, sink, dead = make(T, H, D, window, k, n_c, ring_size, S, 7)
stub = types.SimpleNamespace(args=types.SimpleNamespace(head_dim=D, window_size=window))
old_ring = M.RING
M.RING = ring_size
try:
    via_model = M.Model._gather_attn(stub, q, ring, ckv, idx, S, mask, sink)
finally:
    M.RING = old_ring
direct = prefill_attention_gather(q, ring, ckv, idx, S, mask, sink, D ** -0.5, window, ring_size)
eq = bool(torch.equal(via_model, direct))
mat = bool(torch.equal(via_model, prefill_attention(
    q, materialise(ring, ckv, idx, wpos, ring_size), mask, sink, D ** -0.5)))
print(f"[{'PASS' if eq and mat else 'FAIL'}] Model._gather_attn plumbing: == direct call {eq}, "
      f"== materialised fused path {mat}")
if not (eq and mat):
    fails.append("_gather_attn plumbing")

print()
print("FAILURES: " + ", ".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
