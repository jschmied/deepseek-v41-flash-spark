"""
decode_attn.py -- one Triton kernel for the sinked softmax attention of the decode path, and
(via `prefill_attention` at the bottom of this file) of the prefill path as well: both are T
independent single-query attentions over each token's own gathered key set, so one kernel serves
both and only the launch shape differs.

Shape of the problem (engine/fastdecode.py `_attention`): T query tokens (6 in a verify block,
5 in a DSpark draft), h = 64 query heads, and a per-token key set that every head of that token
shares. That key set comes in two pieces which the caller used to `torch.cat` into one [T, n, d]
tensor: the 128 sliding-window rows gathered from the layer's ring, and either the 512 rows the
CSA2 indexer selected from the compressed cache or (in the draft) the T draft keys themselves.
d = head_dim = 512 (the last 64 dims carry RoPE). The torch version ran this as two fp32 SIMT
batched GEMMs plus the masked_fill / exp / sum elementwise passes around them.

The kernel takes the two pieces as two base pointers plus the boundary n1 and walks them as one key
axis, so the `torch.cat` disappears (it wrote and re-read ~3.9 MB per layer, ~1.3 ms per step). The
second piece may also be a stride-0 broadcast view, which is what the draft's window is, so that
expand is not materialised either. The mask stays a single [T, n1+n2] tensor -- catting the two
masks costs 3.8 kB per layer and keeps the indexing simple.

The math is flash-decoding of exactly what the torch path did:
  scores = q . k * head_dim**-0.5, masked to -inf,
  m      = max_n scores, clamped to >= -1e30 (an all-masked row keeps a finite m),
  denom  = sum_n exp(scores - m) + exp(sink_h - m),
  o      = sum_n exp(scores - m) / denom * k.
Q and K are read in bf16 and every dot accumulates in fp32; the scores and the softmax never touch
bf16. The PV product would, since `tl.dot` needs a tensor-core dtype for both operands, so the
probability matrix is split into a bf16 high part and a bf16 remainder and the two are accumulated
separately (PV_SPLIT=1) -- that keeps ~16 mantissa bits of the probabilities instead of 8, which
matters because the probabilities are dense (~640 keys, so a single bf16 rounding of p costs about
as much accuracy as the final bf16 rounding of o).

d is handled as DA + DB, two powers of two, so that a head dim which is not itself a power of two
(e.g. 576) does not have to be padded up to 1024: the accumulator is the register-hungry part. For
this checkpoint d = 512 and DB = 0, so the second block is compiled away.

With T = 6 and 64 heads there are only T * 64/BLOCK_H = 24 programs, half the 48 SMs of a GB10, so
the key axis is also split SPLIT ways; the partial (acc, m, l) triples are combined by a second,
cheap kernel. Both kernels have static shapes, allocate nothing beyond their outputs and do no host
synchronisation, so the pair captures into a CUDA graph.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

NEG = -1e30
_NEG = tl.constexpr(-1e30)  # module-level constexpr so the jitted kernels may read it


@triton.jit
def _seg(KV, MB, lo, hi, base, q1, q2, m_i, l_i, acc1, acc2, stride_kn, scale,
         IDXP, pos_t, RING_SZ, W_LAST, ROW_MASK,
         DA: tl.constexpr, DB: tl.constexpr, DBP: tl.constexpr,
         BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, PV_SPLIT: tl.constexpr,
         MODE: tl.constexpr):
    """Online-softmax pass over the key rows [lo, hi) of one segment. `base` is subtracted from the
    global key index to get the row inside this segment; the mask is always indexed globally.

    MODE says how that segment-local index becomes a row of KV:
      0  KV is this token's own [N, D] slice, row = index               (the materialised path)
      1  KV is the layer's shared window ring, row = (pos_t - W_LAST + index) % RING_SZ
      2  KV is the shared compressed cache, row = IDXP[index]
    1 and 2 are the gather-fused prefill: the caller hands over the ring and the compressed cache
    themselves instead of a [T, 640, 512] bf16 copy of them, which is 1.34 GB at T = 2048 (2.7 GB
    at its construction peak) and is what sets the prefill chunk ceiling. The arithmetic is the
    same as engine/model.py does on the host (`_window_positions().clamp_min(0) % RING` and
    `sh.ckv[idx.clamp_min(0)]`), key by key.
    Coalescing does not suffer -- measured, the K-loop gets FASTER, not slower: 5.65 ms vs 7.02 ms
    at T = 2048 (0.80x, ABAB-interleaved x40). MODE 1 rows are an affine ramp, so a BLOCK_N = 32
    block reads 32 consecutive ring rows, and a MODE 2 row is 512 contiguous bf16 (1 kB) whatever
    `idx` says; and the tables are small and shared, so every query hits the same 4 MB ring and the
    same 4 MB compressed cache, where kv_all is 1.34 GB streamed once with no reuse at all."""
    da = tl.arange(0, DA)
    db = tl.arange(0, DBP)
    for n0 in range(lo, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nm = offs_n < hi
        j = offs_n - base
        if MODE == 1:
            rows = tl.maximum(pos_t - W_LAST + j, 0) % RING_SZ
        else:
            if MODE == 2:
                # idx = -1 means "no compressed row" and the mask drops the column, so ANY in-bounds
                # row is correct here (its score is forced to -inf, so p is 0 and 0 * k = 0 exactly
                # -- the output is bit-identical whatever is loaded). The host gather used
                # `clamp_min(0)`, i.e. row 0, and that turns out to be the worst choice for the
                # kernel: 32 lanes of a block reading the SAME 1 kB row is 2.2x SLOWER than 32
                # scattered rows (measured at T=1024: 7.89 ms with every idx clamped to 0, 2.82 ms
                # with distinct rows, against 3.55 ms for the pre-gathered K-loop). It is not a
                # corner case -- the -1 tail is the whole tail of `_pad_topk` whenever a query sees
                # fewer than index_topk compressed positions, which is every layer of the first
                # ~512*ratio positions of a prompt. So spread the dead columns over distinct rows
                # instead. ROW_MASK is 2^floor(log2(n_c)) - 1, so `j & ROW_MASK` is in bounds by
                # construction and is distinct for every column as soon as n_c >= BLOCK_N -- and it
                # is a single AND. A modulo would be correct too and was 0.79 ms/call SLOWER at
                # T=1024 than the cliff it fixes (4.39 vs 3.60 ms): 64-bit integer division is not
                # cheap here. With the AND the tail is free: 0.805x of the K-loop at T=2048 with a
                # 25 % -1 tail, 0.802x with none.
                r = tl.load(IDXP + j, mask=nm, other=0)
                rows = tl.where(r >= 0, r, j & ROW_MASK)
            else:
                rows = j
        kp = KV + rows[:, None] * stride_kn
        k1 = tl.load(kp + da[None, :], mask=nm[:, None], other=0.0)
        s = tl.dot(q1, tl.trans(k1), out_dtype=tl.float32)
        if DB > 0:
            k2 = tl.load(kp + (DA + db)[None, :], mask=nm[:, None], other=0.0)
            s += tl.dot(q2, tl.trans(k2), out_dtype=tl.float32)
        else:
            k2 = tl.zeros((BLOCK_N, DBP), tl.bfloat16)
        s = s * scale
        keep = tl.load(MB + offs_n, mask=nm, other=0) != 0
        s = tl.where(keep[None, :] & nm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_new = tl.maximum(m_new, _NEG)  # all-masked rows keep a finite max, as the torch path does
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc1 = acc1 * alpha[:, None]
        acc2 = acc2 * alpha[:, None]
        ph = p.to(tl.bfloat16)
        acc1 += tl.dot(ph, k1, out_dtype=tl.float32)
        if PV_SPLIT:
            pl = (p - ph.to(tl.float32)).to(tl.bfloat16)
            acc1 += tl.dot(pl, k1, out_dtype=tl.float32)
        if DB > 0:
            acc2 += tl.dot(ph, k2, out_dtype=tl.float32)
            if PV_SPLIT:
                acc2 += tl.dot(pl, k2, out_dtype=tl.float32)
        m_i = m_new
    return m_i, l_i, acc1, acc2


@triton.jit
def _dattn_kernel(Q, KV1, KV2, MSK, SINK, OUT, MP, LP,
                  T, H, N, N1,
                  stride_qt, stride_qh, stride_k1t, stride_k1n, stride_k2t, stride_k2n, stride_mt,
                  stride_ot, stride_oh, stride_os,
                  stride_pt, stride_ph,
                  scale,
                  IDX, stride_it, POS_BASE, RING_SZ, W_LAST, ROW_MASK,
                  DA: tl.constexpr, DB: tl.constexpr, DBP: tl.constexpr, BLOCK_H: tl.constexpr,
                  BLOCK_N: tl.constexpr, SPLIT: tl.constexpr, FINAL: tl.constexpr,
                  PV_SPLIT: tl.constexpr, TWO: tl.constexpr,
                  MODE1: tl.constexpr = 0, MODE2: tl.constexpr = 0):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    da = tl.arange(0, DA)
    db = tl.arange(0, DBP)

    qb = Q + pid_t * stride_qt + offs_h[:, None] * stride_qh
    q1 = tl.load(qb + da[None, :], mask=h_mask[:, None], other=0.0)
    if DB > 0:  # head dims that are not a power of two (D = DA + DB); compile-time branch
        q2 = tl.load(qb + (DA + db)[None, :], mask=h_mask[:, None], other=0.0)
    else:
        q2 = tl.zeros((BLOCK_H, DBP), tl.bfloat16)

    per = tl.cdiv(N, SPLIT)
    n_lo = pid_s * per
    n_hi = tl.minimum(n_lo + per, N)

    m_i = tl.full((BLOCK_H,), _NEG, tl.float32)
    l_i = tl.zeros((BLOCK_H,), tl.float32)
    acc1 = tl.zeros((BLOCK_H, DA), tl.float32)
    acc2 = tl.zeros((BLOCK_H, DBP), tl.float32)
    mb = MSK + pid_t * stride_mt
    # the gather modes address shared tables, so the per-token part of the row is the query's
    # absolute position (prefill positions are contiguous: pos = POS_BASE + t) and `idx`'s row
    idxp = IDX + pid_t * stride_it
    pos_t = POS_BASE + pid_t

    m_i, l_i, acc1, acc2 = _seg(KV1 + pid_t * stride_k1t, mb, n_lo, tl.minimum(n_hi, N1), 0,
                                q1, q2, m_i, l_i, acc1, acc2, stride_k1n, scale,
                                idxp, pos_t, RING_SZ, W_LAST, ROW_MASK,
                                DA, DB, DBP, BLOCK_H, BLOCK_N, PV_SPLIT, MODE1)
    if TWO:
        m_i, l_i, acc1, acc2 = _seg(KV2 + pid_t * stride_k2t, mb, tl.maximum(n_lo, N1), n_hi, N1,
                                    q1, q2, m_i, l_i, acc1, acc2, stride_k2n, scale,
                                    idxp, pos_t, RING_SZ, W_LAST, ROW_MASK,
                                    DA, DB, DBP, BLOCK_H, BLOCK_N, PV_SPLIT, MODE2)

    if FINAL:
        sink = tl.load(SINK + offs_h, mask=h_mask, other=0.0).to(tl.float32)
        denom = l_i + tl.exp(sink - m_i)  # all-masked row: exp(sink + 1e30) = inf -> o = 0
        ob = OUT + pid_t * stride_ot + offs_h[:, None] * stride_oh
        tl.store(ob + da[None, :], (acc1 / denom[:, None]).to(tl.bfloat16), mask=h_mask[:, None])
        if DB > 0:
            tl.store(ob + (DA + db)[None, :], (acc2 / denom[:, None]).to(tl.bfloat16), mask=h_mask[:, None])
    else:
        ob = OUT + pid_t * stride_ot + offs_h[:, None] * stride_oh + pid_s * stride_os
        tl.store(ob + da[None, :], acc1, mask=h_mask[:, None])
        if DB > 0:
            tl.store(ob + (DA + db)[None, :], acc2, mask=h_mask[:, None])
        pp = pid_t * stride_pt + offs_h * stride_ph + pid_s
        tl.store(MP + pp, m_i, mask=h_mask)
        tl.store(LP + pp, l_i, mask=h_mask)


@triton.jit
def _dattn_combine(ACC, MP, LP, SINK, OUT, H, D,
                   stride_at, stride_ah, stride_as, stride_ot, stride_oh,
                   stride_pt, stride_ph,
                   SPLIT: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    t = pid // H
    h = pid % H
    si = tl.arange(0, SPLIT)
    d = tl.arange(0, BD)
    dm = d < D
    pp = t * stride_pt + h * stride_ph + si
    m_s = tl.load(MP + pp)
    l_s = tl.load(LP + pp)
    m = tl.maximum(tl.max(m_s), _NEG)
    wgt = tl.exp(m_s - m)
    sink = tl.load(SINK + h).to(tl.float32)
    denom = tl.sum(l_s * wgt) + tl.exp(sink - m)
    acc = tl.load(ACC + t * stride_at + h * stride_ah + si[:, None] * stride_as + d[None, :],
                  mask=dm[None, :], other=0.0)
    o = tl.sum(acc * wgt[:, None], 0) / denom
    tl.store(OUT + t * stride_ot + h * stride_oh + d, o.to(tl.bfloat16), mask=dm)


def _split_d(d: int):
    da = 1 << (d.bit_length() - 1)
    if da == d:
        return d, 0
    db = d - da
    assert db & (db - 1) == 0, f"head dim {d} is not a sum of two powers of two"
    return da, db


BLOCK_H = int(os.environ.get("DSV41_ATTN_BLOCK_H", 16))
BLOCK_N = int(os.environ.get("DSV41_ATTN_BLOCK_N", 32))
N_SPLIT = int(os.environ.get("DSV41_ATTN_SPLIT", 2))
PV_SPLIT = int(os.environ.get("DSV41_ATTN_PV_SPLIT", 1))
NUM_WARPS = int(os.environ.get("DSV41_ATTN_WARPS", 4))
NUM_STAGES = int(os.environ.get("DSV41_ATTN_STAGES", 2))


def decode_attention(q: torch.Tensor, kv1: torch.Tensor, kv2: torch.Tensor | None,
                     mask: torch.Tensor, sink: torch.Tensor, scale: float,
                     split: int | None = None, pv_split: int | None = None,
                     block_n: int | None = None, block_h: int | None = None,
                     num_warps: int | None = None, num_stages: int | None = None) -> torch.Tensor:
    """q bf16 [T, H, D]; kv1 bf16 [T, N1, D] and optional kv2 bf16 [T, N2, D] (either may be a
    stride-0 broadcast along T); mask bool [T, N1+N2]; sink fp32 [H] -> o bf16 [T, H, D].

    block_h/num_warps/num_stages exist because prefill and decode want different shapes out of the
    same kernel: at T = 6 the grid is starved and the launch wants to be small, at T = 2048 there
    are 8192 programs and the per-token KV row set (640 x 512 bf16 = 640 kB) should be read by as
    few programs as possible. Defaults are the decode ones, so no decode call changes."""
    T, H, D = q.shape
    N1 = kv1.shape[1]
    N2 = 0 if kv2 is None else kv2.shape[1]
    N = N1 + N2
    assert kv1.shape[2] == D and mask.shape == (T, N)
    assert q.stride(-1) == 1 and kv1.stride(-1) == 1 and mask.stride(-1) == 1
    assert q.dtype == torch.bfloat16 and kv1.dtype == torch.bfloat16
    if kv2 is not None:
        assert kv2.shape[2] == D and kv2.stride(-1) == 1 and kv2.dtype == torch.bfloat16
    DA, DB = _split_d(D)   # head_dim 512 is a power of two -> DB = 0 and the second block vanishes
    DBP = max(DB, 16)
    bn = BLOCK_N if block_n is None else block_n
    bh = BLOCK_H if block_h is None else block_h
    nw = NUM_WARPS if num_warps is None else num_warps
    ns = NUM_STAGES if num_stages is None else num_stages
    pv = PV_SPLIT if pv_split is None else pv_split
    sp = N_SPLIT if split is None else split
    sp = max(1, min(sp, triton.cdiv(N, bn)))
    if sp & (sp - 1):  # the combine kernel indexes the splits with a power-of-two arange
        sp = 1 << (sp.bit_length() - 1)
    msk = mask if mask.dtype == torch.int8 else mask.view(torch.int8)  # metadata-only, graph-safe
    k2 = kv1 if kv2 is None else kv2
    s2t, s2n = k2.stride(0), k2.stride(1)
    o = torch.empty(T, H, D, dtype=torch.bfloat16, device=q.device)
    grid = (triton.cdiv(H, bh), T, sp)
    args = (q, kv1, k2, msk, sink)
    common = dict(DA=DA, DB=DB, DBP=DBP, BLOCK_H=bh, BLOCK_N=bn, PV_SPLIT=pv,
                  TWO=1 if kv2 is not None else 0, num_warps=nw, num_stages=ns)
    if sp == 1:
        _dattn_kernel[grid](*args, o, o, o, T, H, N, N1,
                            q.stride(0), q.stride(1), kv1.stride(0), kv1.stride(1), s2t, s2n,
                            msk.stride(0), o.stride(0), o.stride(1), 0, 0, 0, scale,
                            q, 0, 0, 1, 0, 1,  # MODE 0: no index tensor, no ring arithmetic
                            SPLIT=1, FINAL=1, **common)
        return o
    acc = torch.empty(T, H, sp, D, dtype=torch.float32, device=q.device)
    mp = torch.empty(T, H, sp, dtype=torch.float32, device=q.device)
    lp = torch.empty(T, H, sp, dtype=torch.float32, device=q.device)
    _dattn_kernel[grid](*args, acc, mp, lp, T, H, N, N1,
                        q.stride(0), q.stride(1), kv1.stride(0), kv1.stride(1), s2t, s2n,
                        msk.stride(0), acc.stride(0), acc.stride(1), acc.stride(2),
                        mp.stride(0), mp.stride(1), scale,
                        q, 0, 0, 1, 0, 1,  # MODE 0: no index tensor, no ring arithmetic
                        SPLIT=sp, FINAL=0, **common)
    BD = 1 << (D - 1).bit_length()
    _dattn_combine[(T * H,)](acc, mp, lp, sink, o, H, D,
                             acc.stride(0), acc.stride(1), acc.stride(2), o.stride(0), o.stride(1),
                             mp.stride(0), mp.stride(1),
                             SPLIT=sp, BD=BD, num_warps=4, num_stages=1)
    return o


# ------------------------------------------------------------------- prefill entry point
# Prefill (engine/model.py `_softmax_attn`) is the SAME problem as decode, only wider: every query
# token still carries its own [N, D] key set (the gathered window plus the compressed rows), so it
# is a batched attention with batch = T, not a shared-KV flash attention. That means this kernel
# serves it unchanged; only the launch shape differs.
#
# Why it is worth a gate at all: the eager chain is einsum -> masked_fill -> amax -> clamp -> sub ->
# exp -> sum -> exp(sink) -> add -> div -> einsum -> cast, per 64-row query tile. In the nsys trace
# of a 79 s prefill, `elementwise_kernel` alone is 224,977 launches (44.7 % of every launch in the
# run) carrying ~12 us of GPU work behind ~90 us of host launch cost; host launch time totals 13.4 s
# of those 79 s. This collapses the whole chain to one launch per call.
#
# The one prefill-specific choice is SPLIT = 1. The key axis is split in decode only because T = 6
# leaves 24 programs on 48 SMs; at T = 2048 the grid is already T * H/BLOCK_H = 8192 programs, so
# splitting would buy nothing and cost the combine kernel plus a [T, H, SPLIT, D] fp32 scratch
# buffer (1.6 GB at the real shape).
#
# The tile itself is swept, not inherited: at T=512, H=64, D=512, N=640 (median of 15, box shared)
#   BLOCK_H 16 / BLOCK_N  32 / 4 warps  1.93 ms   <- default
#   BLOCK_H 16 / BLOCK_N  32 / 8 warps  2.29 ms
#   BLOCK_H 16 / BLOCK_N  64 / 8 warps  2.46 ms
#   BLOCK_H 16 / BLOCK_N  64 / 4 warps  2.57 ms
#   BLOCK_H 32 / BLOCK_N  32 / 4 warps  2.46 ms
# and every (BLOCK_H, BLOCK_N) above 16x32 except 32x32 fails to compile at all: with D = 512 the
# k1 tile alone is BLOCK_N x 512 bf16, so 32x64 already asks for 100 kB of shared memory. Bigger
# head blocks would re-read each token's 640 kB key set fewer times, but they are not reachable;
# what makes 16 acceptable anyway is the axis order (h, t, s), which launches the H/BLOCK_H
# programs of one token adjacently so the re-reads hit in L2. The prefill arm carries its own
# constants because the decode ones are tuned against a 24-program grid, not this one.
#
# The kernel is also chunk-invariant for free, which the torch path had to buy with ATTN_TILE: a
# program reduces over the key axis of ONE token in a fixed order, so a token's output does not
# depend on how many other tokens are in the call. No padding to a fixed tile is needed.
PF_BLOCK_H = int(os.environ.get("DSV41_PREFILL_ATTN_BLOCK_H", 16))
PF_BLOCK_N = int(os.environ.get("DSV41_PREFILL_ATTN_BLOCK_N", 32))
PF_WARPS = int(os.environ.get("DSV41_PREFILL_ATTN_WARPS", 4))
PF_STAGES = int(os.environ.get("DSV41_PREFILL_ATTN_STAGES", 2))


def prefill_attention(q: torch.Tensor, kv_all: torch.Tensor, mask: torch.Tensor,
                      sink: torch.Tensor, scale: float) -> torch.Tensor:
    """The fused form of engine/model.py `_softmax_attn`.

    q bf16 [T, H, D]; kv_all bf16 [T, N, D]; mask bool [T, N] (True = visible); sink fp32 [H]
    -> o bf16 [T, H, D].  One kernel launch, no host synchronisation, nothing allocated but `o`.

    An all-masked (padding) row keeps the torch path's behaviour exactly: its scores are -inf, m is
    clamped to -1e30, l is 0, and exp(sink - (-1e30)) is +inf, so 0/inf = 0 and not NaN -- the same
    clamp and the same sink term, in `_dattn_kernel`.
    """
    if q.stride(-1) != 1:
        q = q.contiguous()
    if kv_all.stride(-1) != 1:
        kv_all = kv_all.contiguous()
    if mask.stride(-1) != 1:
        mask = mask.contiguous()
    return decode_attention(q, kv_all, None, mask, sink, scale, split=1,
                            block_n=PF_BLOCK_N, block_h=PF_BLOCK_H,
                            num_warps=PF_WARPS, num_stages=PF_STAGES)


# ------------------------------------------------------- gather-fused prefill entry point
# `prefill_attention` still takes the [T, 640, 512] bf16 `kv_all` its caller builds with a ring
# gather, an indexer gather and a torch.cat. At T = 2048 that tensor is 1.34 GB and the peak while
# it is built is ~2.7 GB (the 0.27 GB window rows and the 1.07 GB compressed rows are both still
# alive when the 1.34 GB concatenation is allocated) -- the biggest activation in a prefill chunk,
# and the reason engine/model.py's MAX_CHUNK comment names activation memory as the ceiling
# (DSV41_PREFILL_CHUNK=8192 refuses to start today). Building it is not free either: across a run,
# CatArrayBatchedCopy* is 2.15 s and vectorized_gather_kernel 1.31 s of a 40 s GPU-busy prefill.
#
# None of it has to exist. Each of the two pieces is a pure function of something already on the
# device:
#   * window   row = (pos_t - (W-1) + j) % RING, so the ring plus the query's position is enough --
#              no index tensor at all, and `_window_positions` + its clamp + its modulo + its
#              gather all disappear from the host side.
#   * compressed  row = idx[t, j], the indexer's own output, which the host had to materialise only
#              because the kernel could not follow a pointer.
# `_seg` already walked the two pieces with separate base pointers, so this is a row-addressing
# mode per segment (MODE 1 / MODE 2 above) and nothing else: the softmax, the PV split and the
# sink term are the same code, so the result is bit-identical to `prefill_attention` on the same
# inputs -- the test asserts that, not a tolerance.
#
# The mask stays the caller's [T, N] bool (1.3 MB at T=2048) rather than being re-derived here from
# win_lo and compress_lens: it is 0.05 % of what kv_all cost, and keeping it keeps the SWA bounded
# replay semantics in one place.
def prefill_attention_gather(q: torch.Tensor, ring: torch.Tensor, ckv: torch.Tensor | None,
                             idx: torch.Tensor | None, pos0: int, mask: torch.Tensor,
                             sink: torch.Tensor, scale: float, window: int,
                             ring_size: int) -> torch.Tensor:
    """`prefill_attention` without the materialised kv_all.

    q bf16 [T, H, D]; ring bf16 [RING, D] (this layer's window ring, already written);
    ckv bf16 [n_c, D] and idx int64 [T, k] (the indexer's absolute compressed rows, -1 = none), or
    both None for a layer with no compressed side; pos0 = absolute position of query row 0 (prefill
    positions are contiguous); mask bool [T, window + k]; sink fp32 [H] -> o bf16 [T, H, D].

    One kernel launch, no host synchronisation, nothing allocated but `o`.
    """
    T, H, D = q.shape
    N1 = window
    N2 = 0 if idx is None else idx.shape[1]
    N = N1 + N2
    assert ring.shape[1] == D and ring.dtype == torch.bfloat16 and ring.stride(1) == 1
    assert mask.shape == (T, N), (mask.shape, (T, N))
    assert q.dtype == torch.bfloat16 and q.is_cuda
    if q.stride(-1) != 1:
        q = q.contiguous()
    if mask.stride(-1) != 1:
        mask = mask.contiguous()
    if idx is not None:
        assert ckv is not None and ckv.shape[1] == D and ckv.dtype == torch.bfloat16
        assert ckv.stride(1) == 1 and idx.stride(1) == 1
        assert ckv.shape[0] >= 1  # ROW_MASK below needs a non-empty table to land a dead column in
    DA, DB = _split_d(D)
    DBP = max(DB, 16)
    msk = mask if mask.dtype == torch.int8 else mask.view(torch.int8)
    k2 = ring if ckv is None else ckv
    ix = q if idx is None else idx
    o = torch.empty(T, H, D, dtype=torch.bfloat16, device=q.device)
    grid = (triton.cdiv(H, PF_BLOCK_H), T, 1)
    # stride_k1t / stride_k2t are 0: unlike the materialised path both segments read a table
    # SHARED by every query row, and the per-row part is pos_t / idx[t] instead.
    _dattn_kernel[grid](q, ring, k2, msk, sink, o, o, o, T, H, N, N1,
                        q.stride(0), q.stride(1), 0, ring.stride(0), 0, k2.stride(0),
                        msk.stride(0), o.stride(0), o.stride(1), 0, 0, 0, scale,
                        ix, 0 if idx is None else idx.stride(0), pos0, ring_size, window - 1,
                        0 if ckv is None else (1 << (ckv.shape[0].bit_length() - 1)) - 1,
                        DA=DA, DB=DB, DBP=DBP, BLOCK_H=PF_BLOCK_H, BLOCK_N=PF_BLOCK_N,
                        SPLIT=1, FINAL=1, PV_SPLIT=PV_SPLIT, TWO=1 if idx is not None else 0,
                        MODE1=1, MODE2=2, num_warps=PF_WARPS, num_stages=PF_STAGES)
    return o


def decode_attention_ref(q, kv, mask, sink, scale):
    """The torch path this replaces (engine/fastdecode.py `_attention`), for tests."""
    scores = torch.einsum("thd,tnd->thn", q.float(), kv.float()) * scale
    scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
    mx = scores.amax(dim=-1, keepdim=True).clamp_min(NEG)
    p = torch.exp(scores - mx)
    denom = p.sum(-1, keepdim=True) + torch.exp(sink[None, :, None] - mx)
    return torch.einsum("thn,tnd->thd", p / denom, kv.float()).to(torch.bfloat16)
