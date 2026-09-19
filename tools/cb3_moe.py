"""
cb3_moe.py -- grouped MoE kernel for experts stored in the CB3 format (tools/cb3.py): 3-bit
per-row-codebook indices in a plane layout + the original UE8M0 group scales. The kernel rebuilds the
packed-FP4 byte tile of each 128-K quad in registers (index -> codebook nibble via one variable
shift of a per-row 32-bit codebook word) and then reuses the FP4 kernel's decode/dot path
(tools/fp4_moe.py: `_split4`, `_chunk_dot` with the hardware e2m1 -> f16 cvt). Routing, block layout
and the down/scatter scheme are shared with fp4_moe.

Per slot: w1/w3 lo [2304, 1280] + hi [2304, 640] + cb [2304, 8] + s [2304, 160];
          w2    lo [5120, 576]  + hi [5120, 288] + cb [5120, 8] + s [5120, 72]  -> 14.45 MB (3.07 bpw).
"""

from __future__ import annotations

import contextlib
import os

import torch
import scale_codec as SC   # the packer the builder and the engine share
import triton
import triton.language as tl

import fp4_moe as F4
from fp4_moe import DIM, INTER, _chunk_dot, _split4, _ue8m0, build_routing, build_routing_small, _pick_bm  # noqa: F401
from cb3 import dequant_cb2, dequant_cb3, fp4_to_cb2, fp4_to_cb3

# nsys attribution only, same gate/helper as engine/model.py (kept local: tools/ has no import of
# engine/). Off by default; `if not NVTX: yield; return` means the disabled path never touches
# torch.cuda.nvtx.
NVTX = os.environ.get("DSV41_NVTX", "0") == "1"


@contextlib.contextmanager
def nvtx_range(name: str):
    if not NVTX:
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

SG1, SG2 = DIM // 32, INTER // 32
# lo + hi + 8 codebook bytes per row, plus the unchanged UE8M0 scales: 14,454,784 B vs FP4's
# 18,800,640 (0.769x), i.e. 3.26-3.28 bit/weight.
CB3_BYTES_PER_SLOT = (2 * (INTER * (DIM // 4 + DIM // 8 + 8) + INTER * SG1)
                      + DIM * (INTER // 4 + INTER // 8 + 8) + DIM * SG2)
# Packed scale planes shrink the slot by exactly the three scale planes' difference: 4.936 % more
# slots for the same arena bytes. The engine divides its arena budget by this, so it MUST follow the
# layout or the capacity gain never materialises.
CB3_BYTES_PER_SLOT_PACKED = (CB3_BYTES_PER_SLOT
                             - 2 * INTER * (SG1 - (1 + SG1 * 3 // 8))
                             - DIM * (SG2 - (1 + SG2 * 3 // 8)))


class CB3Arena:
    # PACKED SCALE PLANES (change A). False keeps the historical layout -- one UE8M0 byte per group
    # -- so every existing caller, checkpoint path and test is untouched. True stores the file's own
    # `ue8m0-3bit-rowbase-v1`: a u8 row base plus 3 bits per group. Nine of the twelve planes are
    # already byte-identical to the on-disk record; the scale planes are the entire 679,936 B/slot
    # difference, so packing them is +4.936 % capacity (5,949 -> 6,243 slots at 86 GB).
    #
    # Gated locally before any of this reached an engine: registers (up 250->254 with 0 spills
    # either way, down 128->148 with its 4 existing spills ELIMINATED), bitwise identical on
    # T=1 / T=6 / T=6 high-distinct / T=24, and latency-neutral.
    packed_scales = False

    def __init__(self, slots: int, device: torch.device | str = "cuda", packed_scales: bool = False):
        self.slots = slots
        self.device = torch.device(device)
        self.packed_scales = packed_scales
        sg1 = packed_row_bytes(SG1) if packed_scales else SG1
        sg2 = packed_row_bytes(SG2) if packed_scales else SG2
        u8 = dict(dtype=torch.uint8, device=self.device)
        self.w1_lo = torch.empty((slots, INTER, DIM // 4), **u8)
        self.w1_hi = torch.empty((slots, INTER, DIM // 8), **u8)
        self.w1_cb = torch.empty((slots, INTER, 8), **u8)
        self.s1 = torch.empty((slots, INTER, sg1), **u8)
        self.w3_lo = torch.empty((slots, INTER, DIM // 4), **u8)
        self.w3_hi = torch.empty((slots, INTER, DIM // 8), **u8)
        self.w3_cb = torch.empty((slots, INTER, 8), **u8)
        self.s3 = torch.empty((slots, INTER, sg1), **u8)
        self.w2_lo = torch.empty((slots, DIM, INTER // 4), **u8)
        self.w2_hi = torch.empty((slots, DIM, INTER // 8), **u8)
        self.w2_cb = torch.empty((slots, DIM, 8), **u8)
        self.s2 = torch.empty((slots, DIM, sg2), **u8)
        self.sim = None  # engine.codebook_sim.CodebookSim(3), set by the caller

    @property
    def bytes_per_slot(self) -> int:
        return sum(t[0].numel() for t in (self.w1_lo, self.w1_hi, self.w1_cb, self.s1, self.w3_lo, self.w3_hi,
                                          self.w3_cb, self.s3, self.w2_lo, self.w2_hi, self.w2_cb, self.s2))

    def load_slot(self, slot: int, w1, s1, w2, s2, w3, s3, non_blocking: bool = False) -> None:
        """Takes packed-FP4 CPU tensors as read from the checkpoint (same signature as the FP4 arena)
        and converts them to CB3 on the GPU on the way in."""
        assert self.sim is not None, "CB3Arena.sim must be a CodebookSim(3)"
        # Only CB3ArenaV2 implements the packed path (load, dequant, and the v3 kernels). This base
        # class would write one byte per group into a 61-byte row and read it back as if nothing had
        # happened -- silent wrong numbers rather than a crash, which is the worst failure shape
        # available. Refuse instead.
        assert not self.packed_scales, (
            "CB3Arena (v1 layout) has no packed-scale path; use CB3ArenaV2")
        dev = self.device
        for (w, s, lo_t, hi_t, cb_t, s_t) in ((w1, s1, self.w1_lo, self.w1_hi, self.w1_cb, self.s1),
                                               (w3, s3, self.w3_lo, self.w3_hi, self.w3_cb, self.s3),
                                               (w2, s2, self.w2_lo, self.w2_hi, self.w2_cb, self.s2)):
            wg = w.view(torch.uint8).to(dev, non_blocking=non_blocking)
            sg = s.view(torch.uint8).to(dev, non_blocking=non_blocking)
            lo, hi, cb = fp4_to_cb3(wg, sg, self.sim)
            lo_t[slot].copy_(lo); hi_t[slot].copy_(hi); cb_t[slot].copy_(cb); s_t[slot].copy_(sg)

    def dequant_slot(self, slot: int):
        w1 = dequant_cb3(self.w1_lo[slot], self.w1_hi[slot], self.w1_cb[slot], self.s1[slot])
        w2 = dequant_cb3(self.w2_lo[slot], self.w2_hi[slot], self.w2_cb[slot], self.s2[slot])
        w3 = dequant_cb3(self.w3_lo[slot], self.w3_hi[slot], self.w3_cb[slot], self.s3[slot])
        return w1, w2, w3


@triton.jit
def _cb3_pack_quad(lo_row, hi_row, cbword, BN: tl.constexpr):
    """Rebuild the packed-FP4 byte tile [BN, 64] of one 128-K quad from the plane layout.
    lo_row / hi_row: [BN, 1] pointers at the quad's first lo byte (32 per quad) / hi byte (16 per quad);
    cbword: [BN, 1] int32 with the row's 8 codebook nibbles. Byte j of the tile holds codes 2j, 2j+1:
    their low bits sit in lo[j // 2] (at 4*(j%2) and +2), their high bits in hi[j // 4] (bit 2*(j%4), +1).
    Gather loads (L1-served duplicates) instead of register reshapes."""
    j = tl.arange(0, 64)[None, :]
    lo_e = tl.load(lo_row + j // 2).to(tl.int32)  # [BN, 64]
    hi_e = tl.load(hi_row + j // 4).to(tl.int32)  # [BN, 64]
    sh0 = (j % 2) * 4
    bh = (j % 4) * 2
    idx0 = ((lo_e >> sh0) & 3) | (((hi_e >> bh) & 1) << 2)
    idx1 = ((lo_e >> (sh0 + 2)) & 3) | (((hi_e >> (bh + 1)) & 1) << 2)
    nib0 = (cbword >> (idx0 * 4)) & 15
    nib1 = (cbword >> (idx1 * 4)) & 15
    return (nib0 | (nib1 << 4)).to(tl.uint8)


@triton.jit
def _cb3_quad_dot(x_base, xk, mask_m, lo_row, hi_row, cbword, s_ptr, BN: tl.constexpr):
    packed = _cb3_pack_quad(lo_row, hi_row, cbword, BN)
    c0, c1, c2, c3 = _split4(packed, BN, 16)
    s = tl.load(s_ptr)  # [BN, 4]
    sa, sb = tl.split(tl.permute(tl.reshape(s, [BN, 2, 2]), [0, 2, 1]))
    s0, s1 = tl.split(sa)
    s2, s3 = tl.split(sb)
    acc = _chunk_dot(x_base, xk, mask_m, c0, s0)
    acc += _chunk_dot(x_base + 32, xk, mask_m, c1, s1)
    acc += _chunk_dot(x_base + 64, xk, mask_m, c2, s2)
    acc += _chunk_dot(x_base + 96, xk, mask_m, c3, s3)
    return acc


@triton.jit
def _cbword(cb_ptr, BN: tl.constexpr):
    cb = tl.load(cb_ptr).to(tl.int32)  # [BN, 8]
    return tl.sum(cb << (tl.arange(0, 8) * 4)[None, :], axis=1)[:, None]  # [BN, 1]


@triton.jit
def _cb3_up_kernel(
    x_ptr, lo1_ptr, hi1_ptr, cb1_ptr, s1_ptr, lo3_ptr, hi3_ptr, cb3_ptr, s3_ptr, h_ptr,
    wgt_ptr, block_slot_ptr, block_pair_ptr,
    stride_x, stride_h, limit,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    # Packed rows are [base u8][3 B per 8 groups]; unpacked are one byte per group.
    SSTRIDE: tl.constexpr = (1 + SG * 3 // 8) if PACKED else SG
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    tok = (offs_m // TOPK).to(tl.int64)
    offs_n = nb * BN + tl.arange(0, BN)
    offs_q = tl.arange(0, 4)
    offs_c = tl.arange(0, 8)
    x_base = x_ptr + tok[:, None] * stride_x
    xk = 2 * tl.arange(0, 16)[None, :]
    lo1 = lo1_ptr + slot * SLO + offs_n[:, None] * KL
    hi1 = hi1_ptr + slot * SHI + offs_n[:, None] * KH
    lo3 = lo3_ptr + slot * SLO + offs_n[:, None] * KL
    hi3 = hi3_ptr + slot * SHI + offs_n[:, None] * KH
    s1t = s1_ptr + slot * SSC + offs_n[:, None] * SSTRIDE + offs_q[None, :]
    s3t = s3_ptr + slot * SSC + offs_n[:, None] * SSTRIDE + offs_q[None, :]
    cw1 = _cbword(cb1_ptr + slot * SCB + offs_n[:, None] * 8 + offs_c[None, :], BN)
    cw3 = _cbword(cb3_ptr + slot * SCB + offs_n[:, None] * 8 + offs_c[None, :], BN)
    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    acc_u = tl.zeros([BM, BN], dtype=tl.float32)
    for q in range(0, SG // 4):
        acc_g += _cb3_quad_dot(x_base + q * 128, xk, mask_m[:, None], lo1 + q * 32, hi1 + q * 16, cw1, s1t + q * 4, BN)
        acc_u += _cb3_quad_dot(x_base + q * 128, xk, mask_m[:, None], lo3 + q * 32, hi3 + q * 16, cw3, s3t + q * 4, BN)
    gate = tl.minimum(acc_g, limit)
    up = tl.minimum(tl.maximum(acc_u, -limit), limit)
    wgt = tl.load(wgt_ptr + offs_m, mask=mask_m, other=0.0)
    h = gate * tl.sigmoid(gate) * up * wgt[:, None]
    tl.store(h_ptr + offs_m[:, None] * stride_h + offs_n[None, :], h.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _cb3_down_kernel(
    h_ptr, lo2_ptr, hi2_ptr, cb2_ptr, s2_ptr, y_ptr,
    block_slot_ptr, block_pair_ptr,
    stride_h, stride_y,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NTOK: tl.constexpr,
):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    # Packed rows are [base u8][3 B per 8 groups]; unpacked are one byte per group.
    SSTRIDE: tl.constexpr = (1 + SG * 3 // 8) if PACKED else SG
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = nb * BN + tl.arange(0, BN)
    offs_q = tl.arange(0, 4)
    offs_c = tl.arange(0, 8)
    h_base = h_ptr + offs_m[:, None].to(tl.int64) * stride_h
    xk = 2 * tl.arange(0, 16)[None, :]
    lo2 = lo2_ptr + slot * SLO + offs_n[:, None] * KL
    hi2 = hi2_ptr + slot * SHI + offs_n[:, None] * KH
    s2t = s2_ptr + slot * SSC + offs_n[:, None] * SSTRIDE + offs_q[None, :]
    cw2 = _cbword(cb2_ptr + slot * SCB + offs_n[:, None] * 8 + offs_c[None, :], BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for q in range(0, SG // 4):
        acc += _cb3_quad_dot(h_base + q * 128, xk, mask_m[:, None], lo2 + q * 32, hi2 + q * 16, cw2, s2t + q * 4, BN)
    row = ((offs_m % TOPK) * NTOK + offs_m // TOPK).to(tl.int64)
    tl.store(y_ptr + row[:, None] * stride_y + offs_n[None, :], acc, mask=mask_m[:, None])


_UP_CFG = {16: (128, 4, 1), 32: (128, 4, 1), 64: (64, 4, 1)}
_DOWN_CFG = {16: (128, 8, 2), 32: (128, 4, 2), 64: (128, 8, 2)}


def moe_forward(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena: CB3Arena,
                swiglu_limit: float = 10.0, block_m: int | None = None) -> torch.Tensor:
    assert x.dtype == torch.bfloat16 and x.shape[1] == DIM and x.is_contiguous()
    T, K = slots.shape
    P = T * K
    dev = x.device
    BM = block_m or _pick_bm(P)
    bn1, nw1, ns1 = _UP_CFG[BM]
    bn2, nw2, ns2 = _DOWN_CFG[BM]
    block_slot, block_pair, NB = build_routing(slots, arena.slots, BM)
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    h = torch.empty((P, INTER), dtype=torch.bfloat16, device=dev)
    parts = torch.empty((P, DIM), dtype=torch.float32, device=dev)
    _cb3_up_kernel[(NB, INTER // bn1)](
        x, arena.w1_lo, arena.w1_hi, arena.w1_cb, arena.s1, arena.w3_lo, arena.w3_hi, arena.w3_cb, arena.s3, h,
        wgt, block_slot, block_pair, x.stride(0), h.stride(0), float(swiglu_limit),
        TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, num_warps=nw1, num_stages=ns1)
    _cb3_down_kernel[(NB, DIM // bn2)](
        h, arena.w2_lo, arena.w2_hi, arena.w2_cb, arena.s2, parts, block_slot, block_pair,
        h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T, num_warps=nw2, num_stages=ns2)
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)


if __name__ == "__main__":
    import json, os, sys, time
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
    from codebook_sim import CodebookSim
    from safetensors import safe_open
    md = os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
    idx = json.load(open(f"{md}/model.safetensors.index.json"))["weight_map"]
    f = safe_open(f"{md}/{idx['layers.0.ffn.experts.0.w1.weight']}", "pt", device="cpu")
    torch.manual_seed(0)
    S = 32
    arena = CB3Arena(S, "cuda"); arena.sim = CodebookSim(3, "cuda")
    t0 = time.time()
    for s in range(S):
        p = f"layers.0.ffn.experts.{s}."
        arena.load_slot(s, f.get_tensor(p + "w1.weight"), f.get_tensor(p + "w1.scale"), f.get_tensor(p + "w2.weight"),
                        f.get_tensor(p + "w2.scale"), f.get_tensor(p + "w3.weight"), f.get_tensor(p + "w3.scale"))
    torch.cuda.synchronize(); print(f"loaded+converted {S} experts in {time.time() - t0:.1f}s ({arena.bytes_per_slot / 1e6:.2f} MB/slot)")
    for T in (1, 6, 64, 512):
        x = (torch.randn(T, DIM, device="cuda") * 0.5).to(torch.bfloat16)
        slots = torch.stack([torch.randperm(S, device="cuda")[:6] for _ in range(T)]).to(torch.int32)
        w = torch.rand(T, 6, device="cuda")
        y = moe_forward(x, slots, w, arena)
        r = F4.moe_forward_reference(x, slots, w, arena)  # uses arena.dequant_slot -> CB3 dequant
        rel = float((y.float() - r.float()).norm() / r.float().norm())
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(10): moe_forward(x, slots, w, arena)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 10
        n_exp = len(torch.unique(slots))
        print(f"T={T:3d} experts={n_exp:2d}: rel err {rel:.2e}  {dt * 1e3:6.2f} ms  {n_exp * arena.bytes_per_slot / dt / 1e9:5.0f} GB/s of CB3 bytes")


# ============================================================================ v2
# Kernel over the v2 CB3 layout (tools/cb3.py): one [BN, 64] lo load plus one [BN, 32] hi load per
# 256-weight block, split in registers into four and two [BN, 16] tiles, and every scale group's
# packed-FP4 byte tile built with shifts and masks only -- no gathers, no reshapes, no shared memory.
# From there it is the FP4 kernel's path verbatim (`_chunk_dot` with the hardware e2m1 cvt).
import cb3 as CB3  # noqa: E402
from cb3 import dequant_cb3_v2, fp4_to_cb3_v2  # noqa: E402


@triton.jit
def _split2(t, BN: tl.constexpr, W: tl.constexpr):
    a, b = tl.split(tl.permute(tl.reshape(t, [BN, 2, W]), [0, 2, 1]))
    return a, b


@triton.jit
def _split8(s, BN: tl.constexpr):
    a, b = _split2(s, BN, 4)
    aa, ab = tl.split(tl.permute(tl.reshape(a, [BN, 2, 2]), [0, 2, 1]))
    ba, bb = tl.split(tl.permute(tl.reshape(b, [BN, 2, 2]), [0, 2, 1]))
    s0, s1 = tl.split(aa)
    s2, s3 = tl.split(ab)
    s4, s5 = tl.split(ba)
    s6, s7 = tl.split(bb)
    return s0, s1, s2, s3, s4, s5, s6, s7


@triton.jit
def _grp_packed(Lk, Hm, cw, sh: tl.constexpr, hb: tl.constexpr):
    """One scale group's packed-FP4 byte tile [BN, 16] from its lo sub-tile, hi sub-tile and the
    row's 32-bit codebook word. Byte e holds the group's K offsets 2e (low nibble) and 2e+1 (high),
    which is exactly what `_chunk_dot` expects."""
    l = Lk.to(tl.int32)
    h = Hm.to(tl.int32)
    ie = ((l >> sh) & 3) | (((h >> hb) & 1) << 2)
    io = ((l >> (sh + 2)) & 3) | (((h >> (hb + 1)) & 1) << 2)
    ne = (cw >> (ie * 4)) & 15
    no = (cw >> (io * 4)) & 15
    return (ne | (no << 4)).to(tl.uint8)


@triton.jit
def _cb3v2_block_dot(x_base, xk, mask_m, lo_ptr, hi_ptr, s_ptr, cw, BN: tl.constexpr):
    """256 logical K = 8 scale groups: 64 lo bytes + 32 hi bytes + 8 scale bytes per row."""
    L = tl.load(lo_ptr)   # [BN, 64]
    H = tl.load(hi_ptr)   # [BN, 32]
    L0, L1, L2, L3 = _split4(L, BN, 16)
    H0, H1 = _split2(H, BN, 16)
    s0, s1, s2, s3, s4, s5, s6, s7 = _split8(tl.load(s_ptr), BN)
    acc = _chunk_dot(x_base, xk, mask_m, _grp_packed(L0, H0, cw, 0, 0), s0)
    acc += _chunk_dot(x_base + 32, xk, mask_m, _grp_packed(L0, H0, cw, 4, 2), s1)
    acc += _chunk_dot(x_base + 64, xk, mask_m, _grp_packed(L1, H0, cw, 0, 4), s2)
    acc += _chunk_dot(x_base + 96, xk, mask_m, _grp_packed(L1, H0, cw, 4, 6), s3)
    acc += _chunk_dot(x_base + 128, xk, mask_m, _grp_packed(L2, H1, cw, 0, 0), s4)
    acc += _chunk_dot(x_base + 160, xk, mask_m, _grp_packed(L2, H1, cw, 4, 2), s5)
    acc += _chunk_dot(x_base + 192, xk, mask_m, _grp_packed(L3, H1, cw, 0, 4), s6)
    acc += _chunk_dot(x_base + 224, xk, mask_m, _grp_packed(L3, H1, cw, 4, 6), s7)
    return acc


@triton.jit
def _cb3v2_up_kernel(
    x_ptr, lo1_ptr, hi1_ptr, cb1_ptr, s1_ptr, lo3_ptr, hi3_ptr, cb3_ptr, s3_ptr, h_ptr,
    wgt_ptr, block_slot_ptr, block_pair_ptr,
    stride_x, stride_h, limit,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    # Packed rows are [base u8][3 B per 8 groups]; unpacked are one byte per group.
    SSTRIDE: tl.constexpr = (1 + SG * 3 // 8) if PACKED else SG
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    tok = (offs_m // TOPK).to(tl.int64)
    offs_n = nb * BN + tl.arange(0, BN)
    x_base = x_ptr + tok[:, None] * stride_x
    xk = 2 * tl.arange(0, 16)[None, :]
    lo1 = lo1_ptr + slot * (N * KL) + offs_n[:, None] * KL + tl.arange(0, 64)[None, :]
    hi1 = hi1_ptr + slot * (N * KH) + offs_n[:, None] * KH + tl.arange(0, 32)[None, :]
    lo3 = lo3_ptr + slot * (N * KL) + offs_n[:, None] * KL + tl.arange(0, 64)[None, :]
    hi3 = hi3_ptr + slot * (N * KH) + offs_n[:, None] * KH + tl.arange(0, 32)[None, :]
    s1t = s1_ptr + slot * (N * SSTRIDE) + offs_n[:, None] * SSTRIDE + tl.arange(0, 8)[None, :]
    s3t = s3_ptr + slot * (N * SSTRIDE) + offs_n[:, None] * SSTRIDE + tl.arange(0, 8)[None, :]
    cw1 = _cbword(cb1_ptr + slot * (N * 8) + offs_n[:, None] * 8 + tl.arange(0, 8)[None, :], BN)
    cw3 = _cbword(cb3_ptr + slot * (N * 8) + offs_n[:, None] * 8 + tl.arange(0, 8)[None, :], BN)
    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    acc_u = tl.zeros([BM, BN], dtype=tl.float32)
    for b in range(0, K // 256):
        acc_g += _cb3v2_block_dot(x_base + b * 256, xk, mask_m[:, None], lo1 + b * 64, hi1 + b * 32, s1t + b * 8, cw1, BN)
        acc_u += _cb3v2_block_dot(x_base + b * 256, xk, mask_m[:, None], lo3 + b * 64, hi3 + b * 32, s3t + b * 8, cw3, BN)
    gate = tl.minimum(acc_g, limit)
    up = tl.minimum(tl.maximum(acc_u, -limit), limit)
    wgt = tl.load(wgt_ptr + offs_m, mask=mask_m, other=0.0)
    h = gate * tl.sigmoid(gate) * up * wgt[:, None]
    tl.store(h_ptr + offs_m[:, None] * stride_h + offs_n[None, :], h.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _cb3v2_down_kernel(
    h_ptr, lo2_ptr, hi2_ptr, cb2_ptr, s2_ptr, y_ptr,
    block_slot_ptr, block_pair_ptr,
    stride_h, stride_y,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NTOK: tl.constexpr,
):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    # Packed rows are [base u8][3 B per 8 groups]; unpacked are one byte per group.
    SSTRIDE: tl.constexpr = (1 + SG * 3 // 8) if PACKED else SG
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = nb * BN + tl.arange(0, BN)
    h_base = h_ptr + offs_m[:, None].to(tl.int64) * stride_h
    xk = 2 * tl.arange(0, 16)[None, :]
    lo2 = lo2_ptr + slot * (N * KL) + offs_n[:, None] * KL + tl.arange(0, 64)[None, :]
    hi2 = hi2_ptr + slot * (N * KH) + offs_n[:, None] * KH + tl.arange(0, 32)[None, :]
    s2t = s2_ptr + slot * (N * SSTRIDE) + offs_n[:, None] * SSTRIDE + tl.arange(0, 8)[None, :]
    cw2 = _cbword(cb2_ptr + slot * (N * 8) + offs_n[:, None] * 8 + tl.arange(0, 8)[None, :], BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for b in range(0, K // 256):
        acc += _cb3v2_block_dot(h_base + b * 256, xk, mask_m[:, None], lo2 + b * 64, hi2 + b * 32, s2t + b * 8, cw2, BN)
    row = ((offs_m % TOPK) * NTOK + offs_m // TOPK).to(tl.int64)
    tl.store(y_ptr + row[:, None] * stride_y + offs_n[None, :], acc, mask=mask_m[:, None])


UNPACK_BATCH = int(os.environ.get("DSV41_CB3_UNPACK_BATCH", 32))
PREFILL_MODE = os.environ.get("DSV41_CB3_PREFILL", "fp4")   # "fp4" = unpack fallback, "direct" = CB3 kernel
PREFILL_MIN_P = int(os.environ.get("DSV41_CB3_PREFILL_MIN_P", 65))
# Scratch slots kept across the chunks of ONE layer. 0 = the old per-call behaviour.
#
# Why this exists. `moe_forward_prefill` unpacks the experts of one (layer, chunk) call, and
# layer-major prefill runs ~7 chunks over nearly the same expert set -- a layer has ~362 distinct
# experts and every chunk touches almost all of them. Measured in the 2026-09-14 nsys trace:
# 1512 `_unpack_into` calls x <=32 experts = up to 48,384 expert-unpacks where 21 x 362 = 7,600 are
# needed, a 6.4x redundancy. At 33.25 MB per unpack (read 14.45 CB3 + write 18.80 FP4) that is
# 1.61 TB, and `_cb3_unpack_kernel` was the largest single GPU consumer in the profile at 7.44 s of
# a 40.3 s GPU-busy prefill (18.5 %) -- running at ~90 % of this box's 240 GB/s stream rate, so the
# only way to make it cheaper is to do less of it.
#
# This is the same re-read that layer-major already removed for NVMe (74.2 % of loads were an
# expert the layer had fetched in an earlier chunk), still fully present in a buffer 17x larger
# than the NVMe traffic it was measured against.
#
# Holding a layer's union costs scratch: 400 slots x 18.80 MB = 7.5 GB, ~3.4 pp of expert
# residency. That is the trade the A/B has to beat.
# 384 (a layer's expert set), not 0: measured -3.70 s of a 58.6 s prefill, and it still wins by
# 3.7 s at EQUAL TOTAL MEMORY -- arena 82.8 GB with no cache (58.1 s) against arena 75.6 GB plus
# 7.2 GB of scratch (54.4 s), i.e. after paying 3.3 pp of expert residency and ~7 GB more NVMe.
# The engine's auto-sizer subtracts this scratch before sizing the arena (see v41_engine); it is
# allocated lazily at the first prefill, which is why it must be reserved up front.
SCRATCH_SLOTS = int(os.environ.get("DSV41_CB3_SCRATCH_SLOTS", 384))


class CB3ArenaV2(CB3Arena):
    """Same tensors as CB3Arena, v2 bit layout inside them."""

    def fp4_scratch(self, slots: int):
        """A packed-FP4 arena the prefill path unpacks into. Allocated once and reused; at the
        default batch of 32 it is 0.6 GB, at SCRATCH_SLOTS=400 it is 7.5 GB."""
        sc = getattr(self, "_scratch", None)
        want = max(slots, UNPACK_BATCH, SCRATCH_SLOTS)
        if sc is None or sc.slots < want:
            self._scratch = sc = F4.ExpertArena(want, self.device)
            self.scratch_epoch()
        return sc

    # ------------------------------------------------- the layer-scoped unpack cache
    # `_scratch_of` maps an ARENA slot to the scratch slot holding its unpacked FP4 copy. It is
    # valid only while that arena slot still holds the same expert, so every write to a slot drops
    # its entry (see `load_slot` here and CB3Cache.load_slot), and `scratch_epoch()` drops the lot
    # at a layer boundary. Without both, a chunk would read another expert's weights -- silently.
    def scratch_epoch(self) -> None:
        """Forget the layer's unpacked experts. Call at a layer boundary."""
        sc = getattr(self, "_scratch", None)
        self._scratch_of: dict[int, int] = {}
        self._scratch_free: list[int] = list(range(sc.slots)) if sc is not None else []

    def invalidate_scratch(self, slot: int) -> None:
        """The expert in arena slot `slot` changed; its unpacked copy is stale."""
        m = getattr(self, "_scratch_of", None)
        if m:
            s = m.pop(int(slot), None)
            if s is not None:
                self._scratch_free.append(s)

    def load_slot(self, slot: int, w1, s1, w2, s2, w3, s3, non_blocking: bool = False, sim=None) -> None:
        """`sim` overrides the arena's own CodebookSim for this slot only.

        A CodebookSim with fewer than 3 bits produces a codebook the packer repeats up to 8 entries
        (cb3._pad_codebook), so the slot keeps its size and its kernel and carries the arithmetic of
        the narrower format. That is how a 2-bit tier is measured for quality before it is built."""
        sim = sim or self.sim
        assert sim is not None, "CB3ArenaV2.sim must be a CodebookSim(3)"
        self.invalidate_scratch(slot)   # this slot's unpacked FP4 copy is about to be stale
        dev = self.device
        for (w, s, lo_t, hi_t, cb_t, s_t) in ((w1, s1, self.w1_lo, self.w1_hi, self.w1_cb, self.s1),
                                              (w3, s3, self.w3_lo, self.w3_hi, self.w3_cb, self.s3),
                                              (w2, s2, self.w2_lo, self.w2_hi, self.w2_cb, self.s2)):
            wg = w.view(torch.uint8).to(dev, non_blocking=non_blocking)
            sg = s.view(torch.uint8).to(dev, non_blocking=non_blocking)
            lo, hi, cb = fp4_to_cb3_v2(wg, sg, sim)
            lo_t[slot].copy_(lo); hi_t[slot].copy_(hi); cb_t[slot].copy_(cb)
            # A packed arena stores the file's own codec, so the checkpoint path packs here. The
            # codec is exactly lossless on this checkpoint (scale_codec's survey: intra-row exponent
            # range never exceeds 7 over all 149,422,080 rows), and job 790 re-checked that on 87
            # real RESIDENT planes rather than trusting the file, which round-trips by construction.
            s_t[slot].copy_(SC.pack_torch(sg) if self.packed_scales else sg)

    def _s(self, plane, groups: int):
        """Scales as the reference path wants them: one byte per group, whatever the arena stores."""
        return SC.unpack_torch(plane, groups) if self.packed_scales else plane

    def dequant_slot(self, slot: int):
        w1 = dequant_cb3_v2(self.w1_lo[slot], self.w1_hi[slot], self.w1_cb[slot],
                            self._s(self.s1[slot], SG1))
        w2 = dequant_cb3_v2(self.w2_lo[slot], self.w2_hi[slot], self.w2_cb[slot],
                            self._s(self.s2[slot], SG2))
        w3 = dequant_cb3_v2(self.w3_lo[slot], self.w3_hi[slot], self.w3_cb[slot],
                            self._s(self.s3[slot], SG1))
        return w1, w2, w3


def moe_forward_v2(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena: CB3ArenaV2,
                   swiglu_limit: float = 10.0, block_m: int | None = None,
                   cfg_up=None, cfg_down=None) -> torch.Tensor:
    # The _cb3v2_* kernels have no PACKED constexpr -- only the v3 pair does. A packed arena
    # here would read 61-byte rows as if they were 160-byte ones and return plausible garbage.
    assert not _packed(arena), "moe_forward_v2 has no packed-scale path; use moe_forward_v3"
    assert x.dtype == torch.bfloat16 and x.shape[1] == DIM and x.is_contiguous()
    T, K = slots.shape
    P = T * K
    dev = x.device
    BM = block_m or _pick_bm(P)
    bn1, nw1, ns1 = cfg_up or _UP_CFG[BM]
    bn2, nw2, ns2 = cfg_down or _DOWN_CFG[BM]
    block_slot, block_pair, NB = build_routing(slots, arena.slots, BM)
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    h = torch.empty((P, INTER), dtype=torch.bfloat16, device=dev)
    parts = torch.empty((P, DIM), dtype=torch.float32, device=dev)
    _cb3v2_up_kernel[(NB, INTER // bn1)](
        x, arena.w1_lo, arena.w1_hi, arena.w1_cb, arena.s1, arena.w3_lo, arena.w3_hi, arena.w3_cb, arena.s3, h,
        wgt, block_slot, block_pair, x.stride(0), h.stride(0), float(swiglu_limit),
        TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, num_warps=nw1, num_stages=ns1)
    _cb3v2_down_kernel[(NB, DIM // bn2)](
        h, arena.w2_lo, arena.w2_hi, arena.w2_cb, arena.s2, parts, block_slot, block_pair,
        h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T, num_warps=nw2, num_stages=ns2)
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)


# ---------------------------------------------------------------------------- v3: PTX decode
# The v2 kernel is instruction-bound, not gather-bound: Triton computes uint8 tile arithmetic one
# byte per 32-bit register, so rebuilding one scale group's packed-FP4 tile costs ~10 instructions
# per weight (2 index builds x 6 ops, 2 codebook lookups x 3, 1 pack x 2, on [BN, 16] tiles).
#
# v3 does the same arithmetic in inline PTX with pack=4, i.e. four bytes per instruction:
#   lo2  = (L32 >> sh) & 0x03030303          two ops for four weights
#   hib  = (H32 >> hb) & 0x01010101
#   ie   = lo2 | (hib << 2)                  the 3-bit index, one per byte lane
#   sel  = compact(ie)                       byte lanes -> nibbles (4 ops), because...
#   code = prmt.b32(cbA, cbB, sel)           ...prmt selects one of 8 bytes per output byte with a
#                                            nibble selector: the codebook lookup is ONE instruction
#   packed = code_even | (code_odd << 4)
# 24 PTX instructions per 8 weights = 3 per weight, against ~10 register ops per weight in v2.
# The codebook arrives as two uint8 tiles whose every 4-element group is entries 0..3 / 4..7 of the
# row's codebook, so that pack=4 hands prmt exactly the two source registers it needs.

def _cb3_asm(sh: int, hb: int) -> str:
    """9 instructions per half (four weights), 20 per invocation (eight weights).

    Two lop3 fusions carry the weight: `a | (b & c)` (immLut 0xF8) folds the high-bit mask and the
    or into one instruction, and `(a | b) & c` (immLut 0xA8) does the same for the first step of the
    byte-lane -> nibble compaction. The high bit is pre-positioned by shifting H by hb-2 so it lands
    at bit 2 of its byte lane, which is where the 3-bit index wants it.
    """
    def half(shift, bit, out):
        # put the wanted high bit at bit 2 of each byte lane
        mv = f"shr.b32 b, $2, {bit - 2};" if bit >= 2 else f"shl.b32 b, $2, {2 - bit};"
        return f"""
shr.b32 a, $1, {shift};
and.b32 a, a, 0x03030303;
{mv}
lop3.b32 ie, a, b, 0x04040404, 0xF8;
shr.b32 t, ie, 4;
lop3.b32 r, ie, t, 0x00FF00FF, 0xA8;
shr.b32 t, r, 8;
or.b32  r, r, t;
prmt.b32 {out}, $3, $4, r;"""
    return "{\n.reg .b32 a, b, ie, t, r, ne, no;" + half(sh, hb, "ne") + half(sh + 2, hb + 1, "no") + """
shl.b32 no, no, 4;
or.b32  $0, ne, no;
}
"""


_ASM_00 = tl.constexpr(_cb3_asm(0, 0))
_ASM_42 = tl.constexpr(_cb3_asm(4, 2))
_ASM_04 = tl.constexpr(_cb3_asm(0, 4))
_ASM_46 = tl.constexpr(_cb3_asm(4, 6))


@triton.jit
def _grp_packed_ptx(Lk, Hm, A, B, ASM: tl.constexpr):
    return tl.inline_asm_elementwise(ASM, "=r,r,r,r,r", [Lk, Hm, A, B],
                                     dtype=tl.uint8, is_pure=True, pack=4)


@triton.jit
def _split16(s, BN: tl.constexpr):
    a, b, c, d = _split4(s, BN, 4)
    aa, ab = tl.split(tl.permute(tl.reshape(a, [BN, 2, 2]), [0, 2, 1]))
    ba, bb = tl.split(tl.permute(tl.reshape(b, [BN, 2, 2]), [0, 2, 1]))
    ca, cb = tl.split(tl.permute(tl.reshape(c, [BN, 2, 2]), [0, 2, 1]))
    da, db = tl.split(tl.permute(tl.reshape(d, [BN, 2, 2]), [0, 2, 1]))
    s0, s1 = tl.split(aa); s2, s3 = tl.split(ab)
    s4, s5 = tl.split(ba); s6, s7 = tl.split(bb)
    s8, s9 = tl.split(ca); s10, s11 = tl.split(cb)
    s12, s13 = tl.split(da); s14, s15 = tl.split(db)
    return s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15


@triton.jit
def _pair_dot(x_base, xk, mask_m, Lk, Hm, A, B, sA, sB, KPAR: tl.constexpr):
    """The two scale groups that share one 16-byte lo sub-tile (k), using the low or the high half
    of its hi sub-tile's bits depending on k's parity."""
    if KPAR == 0:
        p0 = _grp_packed_ptx(Lk, Hm, A, B, _ASM_00)
        p1 = _grp_packed_ptx(Lk, Hm, A, B, _ASM_42)
    else:
        p0 = _grp_packed_ptx(Lk, Hm, A, B, _ASM_04)
        p1 = _grp_packed_ptx(Lk, Hm, A, B, _ASM_46)
    acc = _chunk_dot(x_base, xk, mask_m, p0, sA)
    acc += _chunk_dot(x_base + 32, xk, mask_m, p1, sB)
    return acc


# --------------------------------------------------------------- packed scale planes (change A)
# The arena stores UE8M0 group scales one byte per group; the on-disk CB3 record stores them as
# `ue8m0-3bit-rowbase-v1` -- one u8 row base plus 3 bits per group, eight groups to a 24-bit LE
# word. That difference is the ENTIRE 679,936 B/slot gap between the 14,454,784 B arena slot and the
# 13,774,848 B record (nine of twelve planes are byte-identical): 4.936 % of capacity, 5,949 ->
# 6,243 slots at 86 GB, which job 715's measured slope prices at ~+16 % decode.
#
# Gates already passed: job 790 packed and unpacked 87 real RESIDENT planes with 0 mismatches --
# stronger than the manifest, which round-trips by construction because the file is built with this
# codec. Job 810: bitwise identical, +0.3 % standalone. Job 860: ZERO register cost in this
# consumption shape, shared memory down, and hoisting the row base made no difference, so the
# simpler per-block form is used here.
#
# THE OPEN GATE: _cb3v3_up_kernel compiles at n_regs=250 of a 255 cap with 0 spills, and
# _cb3v3_down_kernel already spills 4 (job 850). A 40-register probe cannot predict a 250-register
# kernel, so the decisive number is these kernels' own n_regs under PACKED=True.
#
# NOT part of this change: making the arena slot byte-identical to the record so a miss is one H2D.
# That needs record-major backing storage and per-slot strides in every kernel -- a separate change
# with its own benchmark. A stands on capacity alone.


def _packed(arena) -> bool:
    """Does this arena hold packed scale planes? Declared by the ARENA, not by a module global, so a
    packed arena and an unpacked one can coexist in one process -- which is exactly what the bitwise
    gate needs, and what a module flag would have made impossible."""
    return bool(getattr(arena, "packed_scales", False))


def packed_row_bytes(groups: int) -> int:
    """Bytes per row of a packed scale plane: one u8 base plus 3 bits per group."""
    assert groups % 8 == 0, groups
    return 1 + groups * 3 // 8


@triton.jit
def _scales_blk(s_row, blk, N_S: tl.constexpr, PACKED: tl.constexpr, BN: tl.constexpr):
    """Block `blk`'s N_S scales as a [BN, N_S] uint8 tile of UE8M0 exponents.

    `s_row` is the ROW pointer in BOTH layouts, not a pre-offset one: the packed form needs the row
    base byte at offset 0 as well as the block's own bytes, and only the row pointer addresses both.
    """
    if PACKED:
        base = tl.load(s_row).to(tl.int32)                       # [BN, 1] -- one byte per ROW
        g = tl.arange(0, N_S)[None, :]
        o = 1 + blk * (N_S // 8 * 3) + (g // 8) * 3
        b0 = tl.load(s_row + o).to(tl.int32)
        b1 = tl.load(s_row + o + 1).to(tl.int32)
        b2 = tl.load(s_row + o + 2).to(tl.int32)
        w = b0 | (b1 << 8) | (b2 << 16)                          # 24-bit LE, value i at bit 3*i
        return (((w >> ((g % 8) * 3)) & 7) + base).to(tl.uint8)
    return tl.load(s_row + blk * N_S + tl.arange(0, N_S)[None, :])


@triton.jit
def _cb3v3_block_dot(x_base, xk, mask_m, lo_ptr, hi_ptr, s_row, blk, A, B,
                     BN: tl.constexpr, BW: tl.constexpr, PACKED: tl.constexpr):
    """One packing block: BW logical K from a [BN, BW/4] lo tile and a [BN, BW/8] hi tile.
    BW=512 -> 128 B + 64 B row tiles (218 / 185 GB/s on GB10); BW=256 -> 64 B + 32 B, and the 32 B
    hi tile caps at 101 GB/s, which is why 512 is used wherever K allows it."""
    if BW == 512:
        L = tl.load(lo_ptr)   # [BN, 128]
        H = tl.load(hi_ptr)   # [BN, 64]
        La, Lb, Lc, Ld = _split4(L, BN, 32)
        L0, L1 = _split2(La, BN, 16)
        L2, L3 = _split2(Lb, BN, 16)
        L4, L5 = _split2(Lc, BN, 16)
        L6, L7 = _split2(Ld, BN, 16)
        H0, H1, H2, H3 = _split4(H, BN, 16)
        s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15 = \
            _split16(_scales_blk(s_row, blk, 16, PACKED, BN), BN)
        acc = _pair_dot(x_base, xk, mask_m, L0, H0, A, B, s0, s1, 0)
        acc += _pair_dot(x_base + 64, xk, mask_m, L1, H0, A, B, s2, s3, 1)
        acc += _pair_dot(x_base + 128, xk, mask_m, L2, H1, A, B, s4, s5, 0)
        acc += _pair_dot(x_base + 192, xk, mask_m, L3, H1, A, B, s6, s7, 1)
        acc += _pair_dot(x_base + 256, xk, mask_m, L4, H2, A, B, s8, s9, 0)
        acc += _pair_dot(x_base + 320, xk, mask_m, L5, H2, A, B, s10, s11, 1)
        acc += _pair_dot(x_base + 384, xk, mask_m, L6, H3, A, B, s12, s13, 0)
        acc += _pair_dot(x_base + 448, xk, mask_m, L7, H3, A, B, s14, s15, 1)
    else:
        L = tl.load(lo_ptr)   # [BN, 64]
        H = tl.load(hi_ptr)   # [BN, 32]
        L0, L1, L2, L3 = _split4(L, BN, 16)
        H0, H1 = _split2(H, BN, 16)
        s0, s1, s2, s3, s4, s5, s6, s7 = _split8(_scales_blk(s_row, blk, 8, PACKED, BN), BN)
        acc = _pair_dot(x_base, xk, mask_m, L0, H0, A, B, s0, s1, 0)
        acc += _pair_dot(x_base + 64, xk, mask_m, L1, H0, A, B, s2, s3, 1)
        acc += _pair_dot(x_base + 128, xk, mask_m, L2, H1, A, B, s4, s5, 0)
        acc += _pair_dot(x_base + 192, xk, mask_m, L3, H1, A, B, s6, s7, 1)
    return acc


@triton.jit
def _cb_ab(cb_ptr, BN: tl.constexpr):
    """The row's codebook as two [BN, 16] uint8 tiles: every 4-element group is entries 0..3 / 4..7,
    so pack=4 delivers them to prmt as two source registers. One (L1-served) load per program."""
    j = tl.arange(0, 16)[None, :]
    return tl.load(cb_ptr + (j % 4)), tl.load(cb_ptr + 4 + (j % 4))


@triton.jit
def _cb3v3_up_kernel(
    x_ptr, lo1_ptr, hi1_ptr, cb1_ptr, s1_ptr, lo3_ptr, hi3_ptr, cb3_ptr, s3_ptr, h_ptr,
    wgt_ptr, block_slot_ptr, block_pair_ptr,
    stride_x, stride_h, limit,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NB512: tl.constexpr, NB256: tl.constexpr, PACKED: tl.constexpr = False,
    RSTRIDE: tl.constexpr = 0):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    # Packed rows are [base u8][3 B per 8 groups]; unpacked are one byte per group.
    SSTRIDE: tl.constexpr = (1 + SG * 3 // 8) if PACKED else SG
    # RSTRIDE = 0 is the shipped PLANE-MAJOR arena: each plane is its own [slots, ...] tensor, so the
    # slot stride is that plane's own size. RSTRIDE = record bytes is the RECORD-MAJOR arena: one
    # buffer of [slots, RECORD], every plane pointer pre-offset to its position inside the record, so
    # all planes share one slot stride. Record-major is what lets an O_DIRECT read land in the final
    # location instead of being scattered into twelve tensors.
    SLO: tl.constexpr = RSTRIDE if RSTRIDE else (N * KL)
    SHI: tl.constexpr = RSTRIDE if RSTRIDE else (N * KH)
    SCB: tl.constexpr = RSTRIDE if RSTRIDE else (N * 8)
    SSC: tl.constexpr = RSTRIDE if RSTRIDE else (N * SSTRIDE)
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    tok = (offs_m // TOPK).to(tl.int64)
    offs_n = nb * BN + tl.arange(0, BN)
    x_base = x_ptr + tok[:, None] * stride_x
    xk = 2 * tl.arange(0, 16)[None, :]
    lo1 = lo1_ptr + slot * (N * KL) + offs_n[:, None] * KL
    hi1 = hi1_ptr + slot * (N * KH) + offs_n[:, None] * KH
    lo3 = lo3_ptr + slot * (N * KL) + offs_n[:, None] * KL
    hi3 = hi3_ptr + slot * (N * KH) + offs_n[:, None] * KH
    s1t = s1_ptr + slot * (N * SSTRIDE) + offs_n[:, None] * SSTRIDE
    s3t = s3_ptr + slot * (N * SSTRIDE) + offs_n[:, None] * SSTRIDE
    A1, B1 = _cb_ab(cb1_ptr + slot * (N * 8) + offs_n[:, None] * 8, BN)
    A3, B3 = _cb_ab(cb3_ptr + slot * (N * 8) + offs_n[:, None] * 8, BN)
    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    acc_u = tl.zeros([BM, BN], dtype=tl.float32)
    l1a = lo1 + tl.arange(0, 128)[None, :]; h1a = hi1 + tl.arange(0, 64)[None, :]; s1a = s1t + tl.arange(0, 16)[None, :]
    l3a = lo3 + tl.arange(0, 128)[None, :]; h3a = hi3 + tl.arange(0, 64)[None, :]; s3a = s3t + tl.arange(0, 16)[None, :]
    for b in range(0, NB512):
        acc_g += _cb3v3_block_dot(x_base + b * 512, xk, mask_m[:, None], l1a + b * 128, h1a + b * 64, s1t, b, A1, B1, BN, 512, PACKED)
        acc_u += _cb3v3_block_dot(x_base + b * 512, xk, mask_m[:, None], l3a + b * 128, h3a + b * 64, s3t, b, A3, B3, BN, 512, PACKED)
    if NB256 > 0:
        o: tl.constexpr = NB512 * 512
        l1b = lo1 + (NB512 * 128 + tl.arange(0, 64))[None, :]; h1b = hi1 + (NB512 * 64 + tl.arange(0, 32))[None, :]
        s1b = s1t + (NB512 * 16 + tl.arange(0, 8))[None, :]
        l3b = lo3 + (NB512 * 128 + tl.arange(0, 64))[None, :]; h3b = hi3 + (NB512 * 64 + tl.arange(0, 32))[None, :]
        s3b = s3t + (NB512 * 16 + tl.arange(0, 8))[None, :]
        for b in range(0, NB256):
            acc_g += _cb3v3_block_dot(x_base + o + b * 256, xk, mask_m[:, None], l1b + b * 64, h1b + b * 32, s1t, NB512 * 2 + b, A1, B1, BN, 256, PACKED)
            acc_u += _cb3v3_block_dot(x_base + o + b * 256, xk, mask_m[:, None], l3b + b * 64, h3b + b * 32, s3t, NB512 * 2 + b, A3, B3, BN, 256, PACKED)
    gate = tl.minimum(acc_g, limit)
    up = tl.minimum(tl.maximum(acc_u, -limit), limit)
    wgt = tl.load(wgt_ptr + offs_m, mask=mask_m, other=0.0)
    h = gate * tl.sigmoid(gate) * up * wgt[:, None]
    tl.store(h_ptr + offs_m[:, None] * stride_h + offs_n[None, :], h.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _cb3v3_down_kernel(
    h_ptr, lo2_ptr, hi2_ptr, cb2_ptr, s2_ptr, y_ptr,
    block_slot_ptr, block_pair_ptr,
    stride_h, stride_y,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NTOK: tl.constexpr, NB512: tl.constexpr, NB256: tl.constexpr, PACKED: tl.constexpr = False,
    RSTRIDE: tl.constexpr = 0):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    # Packed rows are [base u8][3 B per 8 groups]; unpacked are one byte per group.
    SSTRIDE: tl.constexpr = (1 + SG * 3 // 8) if PACKED else SG
    # See _cb3v3_up_kernel: RSTRIDE = 0 keeps the shipped plane-major strides exactly.
    SLO: tl.constexpr = RSTRIDE if RSTRIDE else (N * KL)
    SHI: tl.constexpr = RSTRIDE if RSTRIDE else (N * KH)
    SCB: tl.constexpr = RSTRIDE if RSTRIDE else (N * 8)
    SSC: tl.constexpr = RSTRIDE if RSTRIDE else (N * SSTRIDE)
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = nb * BN + tl.arange(0, BN)
    h_base = h_ptr + offs_m[:, None].to(tl.int64) * stride_h
    xk = 2 * tl.arange(0, 16)[None, :]
    lo2 = lo2_ptr + slot * (N * KL) + offs_n[:, None] * KL
    hi2 = hi2_ptr + slot * (N * KH) + offs_n[:, None] * KH
    s2t = s2_ptr + slot * (N * SSTRIDE) + offs_n[:, None] * SSTRIDE
    A2, B2 = _cb_ab(cb2_ptr + slot * (N * 8) + offs_n[:, None] * 8, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    l2a = lo2 + tl.arange(0, 128)[None, :]; h2a = hi2 + tl.arange(0, 64)[None, :]; s2a = s2t + tl.arange(0, 16)[None, :]
    for b in range(0, NB512):
        acc += _cb3v3_block_dot(h_base + b * 512, xk, mask_m[:, None], l2a + b * 128, h2a + b * 64, s2t, b, A2, B2, BN, 512, PACKED)
    if NB256 > 0:
        o: tl.constexpr = NB512 * 512
        l2b = lo2 + (NB512 * 128 + tl.arange(0, 64))[None, :]; h2b = hi2 + (NB512 * 64 + tl.arange(0, 32))[None, :]
        s2b = s2t + (NB512 * 16 + tl.arange(0, 8))[None, :]
        for b in range(0, NB256):
            acc += _cb3v3_block_dot(h_base + o + b * 256, xk, mask_m[:, None], l2b + b * 64, h2b + b * 32, s2t, NB512 * 2 + b, A2, B2, BN, 256, PACKED)
    row = ((offs_m % TOPK) * NTOK + offs_m // TOPK).to(tl.int64)
    tl.store(y_ptr + row[:, None] * stride_y + offs_n[None, :], acc, mask=mask_m[:, None])


# Measured on GB10 with 18 real layer-0 experts at decode size (T=6, top-6): (BN, num_warps,
# num_stages). num_warps=8 is deliberately not used anywhere: at BN=32 it is both slower AND
# produces wrong results (rel err ~4 instead of 4.3e-3), i.e. ptxas/Triton miscompiles the inline
# asm at that warp count. Keep num_warps=4 unless someone re-validates.
CB3_UP_CFG = {16: (32, 4, 3), 32: (32, 4, 3), 64: (32, 4, 3)}
CB3_DOWN_CFG = {16: (32, 4, 3), 32: (32, 4, 3), 64: (32, 4, 3)}


def moe_forward_v3(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena: CB3ArenaV2,
                   swiglu_limit: float = 10.0, block_m: int | None = None,
                   cfg_up=None, cfg_down=None) -> torch.Tensor:
    assert x.dtype == torch.bfloat16 and x.shape[1] == DIM and x.is_contiguous()
    T, K = slots.shape
    P = T * K
    if P >= PREFILL_MIN_P and PREFILL_MODE == "fp4" and block_m is None and cfg_up is None:
        return moe_forward_prefill(x, slots, weights, arena, swiglu_limit)
    dev = x.device
    BM = block_m or _pick_bm(P)
    bn1, nw1, ns1 = cfg_up or CB3_UP_CFG[BM]
    bn2, nw2, ns2 = cfg_down or CB3_DOWN_CFG[BM]
    if P <= 64:  # decode-sized call: pure-torch routing (static shapes, graph-capturable)
        block_slot, block_pair, NB = build_routing_small(slots, BM)
    else:
        block_slot, block_pair, NB = build_routing(slots, arena.slots, BM)
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    h = torch.empty((P, INTER), dtype=torch.bfloat16, device=dev)
    parts = torch.empty((P, DIM), dtype=torch.float32, device=dev)
    _cb3v3_up_kernel[(NB, INTER // bn1)](
        x, arena.w1_lo, arena.w1_hi, arena.w1_cb, arena.s1, arena.w3_lo, arena.w3_hi, arena.w3_cb, arena.s3, h,
        wgt, block_slot, block_pair, x.stride(0), h.stride(0), float(swiglu_limit),
        TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, NB512=CB3.block_plan(DIM)[0], NB256=CB3.block_plan(DIM)[1], num_warps=nw1, num_stages=ns1, PACKED=_packed(arena))
    _cb3v3_down_kernel[(NB, DIM // bn2)](
        h, arena.w2_lo, arena.w2_hi, arena.w2_cb, arena.s2, parts, block_slot, block_pair,
        h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T, NB512=CB3.block_plan(INTER)[0], NB256=CB3.block_plan(INTER)[1], num_warps=nw2, num_stages=ns2, PACKED=_packed(arena))
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)


# ---------------------------------------------------------------------------- prefill: unpack to FP4
# CB3's decode work is paid once per PAIR BLOCK, its byte saving once per expert. At decode there is
# one block per expert and CB3 wins (0.79x the FP4 time); at prefill shapes there are one to twelve,
# the decode dominates, and CB3 measured 2.3-6.7x the FP4 time depending on the block size. So a
# prefill-sized call does not run the CB3 kernel at all: it unpacks the experts it needs back into
# packed FP4 codes -- bit-exact, since the CB3 codes ARE a subset of the FP4 grid -- a batch at a
# time into a small scratch arena, and runs the ordinary FP4 kernel over them. One unpack of an
# expert costs a read of its 14.45 MB and a write of 18.80; the FP4 kernel then behaves exactly as it
# does today. `DSV41_CB3_PREFILL=direct` forces the CB3 kernel instead (for measurement).

@triton.jit
def _unpack_pair(out_base, Lk, Hm, A, B, KPAR: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr):
    """Store the two scale groups of one lo sub-tile as 2 x 16 packed-FP4 bytes."""
    j = tl.arange(0, 16)[None, :]
    if KPAR == 0:
        p0 = _grp_packed_ptx(Lk, Hm, A, B, _ASM_00)
        p1 = _grp_packed_ptx(Lk, Hm, A, B, _ASM_42)
    else:
        p0 = _grp_packed_ptx(Lk, Hm, A, B, _ASM_04)
        p1 = _grp_packed_ptx(Lk, Hm, A, B, _ASM_46)
    tl.store(out_base + j, p0)
    tl.store(out_base + 16 + j, p1)


@triton.jit
def _cb3_unpack_kernel(LO, HI, CB, OUT, SRC, DST, N,
                       KL: tl.constexpr, KH: tl.constexpr, KB: tl.constexpr,
                       BN: tl.constexpr, NB512: tl.constexpr, NB256: tl.constexpr):
    b = tl.program_id(0)          # which entry of SRC/DST
    nb = tl.program_id(1)         # row block
    src = tl.load(SRC + b).to(tl.int64)
    dst = tl.load(DST + b).to(tl.int64)   # scratch slot to write; the layer cache picks free ones
    offs_n = nb * BN + tl.arange(0, BN)
    lo = LO + src * (N * KL) + offs_n[:, None] * KL
    hi = HI + src * (N * KH) + offs_n[:, None] * KH
    out = OUT + dst * (N * KB) + offs_n[:, None] * KB
    A, Bc = _cb_ab(CB + src * (N * 8) + offs_n[:, None] * 8, BN)
    for j in range(0, NB512):
        L = tl.load(lo + (j * 128 + tl.arange(0, 128))[None, :])
        H = tl.load(hi + (j * 64 + tl.arange(0, 64))[None, :])
        La, Lb, Lc, Ld = _split4(L, BN, 32)
        L0, L1 = _split2(La, BN, 16)
        L2, L3 = _split2(Lb, BN, 16)
        L4, L5 = _split2(Lc, BN, 16)
        L6, L7 = _split2(Ld, BN, 16)
        H0, H1, H2, H3 = _split4(H, BN, 16)
        o = out + j * 256
        _unpack_pair(o, L0, H0, A, Bc, 0, BN, KB)
        _unpack_pair(o + 32, L1, H0, A, Bc, 1, BN, KB)
        _unpack_pair(o + 64, L2, H1, A, Bc, 0, BN, KB)
        _unpack_pair(o + 96, L3, H1, A, Bc, 1, BN, KB)
        _unpack_pair(o + 128, L4, H2, A, Bc, 0, BN, KB)
        _unpack_pair(o + 160, L5, H2, A, Bc, 1, BN, KB)
        _unpack_pair(o + 192, L6, H3, A, Bc, 0, BN, KB)
        _unpack_pair(o + 224, L7, H3, A, Bc, 1, BN, KB)
    if NB256 > 0:
        for j in range(0, NB256):
            L = tl.load(lo + (NB512 * 128 + j * 64 + tl.arange(0, 64))[None, :])
            H = tl.load(hi + (NB512 * 64 + j * 32 + tl.arange(0, 32))[None, :])
            L0, L1, L2, L3 = _split4(L, BN, 16)
            H0, H1 = _split2(H, BN, 16)
            o = out + NB512 * 256 + j * 128
            _unpack_pair(o, L0, H0, A, Bc, 0, BN, KB)
            _unpack_pair(o + 32, L1, H0, A, Bc, 1, BN, KB)
            _unpack_pair(o + 64, L2, H1, A, Bc, 0, BN, KB)
            _unpack_pair(o + 96, L3, H1, A, Bc, 1, BN, KB)


def _unpack_into(arena, src_slots: torch.Tensor, scratch, dst_slots: torch.Tensor | None = None) -> None:
    """CB3 slots `src_slots` (int32 [B]) -> packed-FP4 scratch slots `dst_slots` (default 0..B-1).

    Scales are copied rather than unpacked when the arena is UNPACKED: the UE8M0 bytes are then the
    same in both formats. A PACKED arena must EXPAND them -- the FP4 scratch always holds one byte
    per group, so a verbatim copy would drop 61-byte rows into 160-byte ones. That shape mismatch is
    how the bitwise gate found this site at all: moe_forward_v3 delegates here whenever
    P >= PREFILL_MIN_P, so it sits on the prefill path even though the decode kernels never reach it. An
    explicit `dst_slots` is what lets the layer-scoped cache leave already-unpacked experts alone
    and drop the new ones into whatever slots are free.
    """
    with nvtx_range("moe.unpack"):
        B = src_slots.numel()
        if dst_slots is None:
            dst_slots = torch.arange(B, dtype=torch.int32, device=src_slots.device)
        BN = 64
        for (lo, hi, cb, N, K, out, s_src, s_dst) in (
                (arena.w1_lo, arena.w1_hi, arena.w1_cb, INTER, DIM, scratch.w1, arena.s1, scratch.s1),
                (arena.w3_lo, arena.w3_hi, arena.w3_cb, INTER, DIM, scratch.w3, arena.s3, scratch.s3),
                (arena.w2_lo, arena.w2_hi, arena.w2_cb, DIM, INTER, scratch.w2, arena.s2, scratch.s2)):
            n512, n256 = CB3.block_plan(K)
            _cb3_unpack_kernel[(B, triton.cdiv(N, BN))](
                lo, hi, cb, out, src_slots, dst_slots, N, KL=K // 4, KH=K // 8, KB=K // 2,
                BN=BN, NB512=n512, NB256=n256, num_warps=4, num_stages=2)
            if _packed(arena):
                src = s_src[src_slots.long()]                       # [B, rows, packed_row_bytes]
                s_dst[dst_slots.long()] = SC.unpack_torch(
                    src.reshape(-1, src.shape[-1]), s_dst.shape[-1]).reshape(src.shape[0], src.shape[1], -1)
            else:
                s_dst[dst_slots.long()] = s_src[src_slots.long()]


def moe_v3_phase(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena: CB3ArenaV2,
                 h: torch.Tensor, parts: torch.Tensor, swiglu_limit: float = 10.0,
                 block_m: int | None = None, routing=None, cfg_up=None, cfg_down=None) -> None:
    """One PHASE of a split routed MoE: run up+down for `slots` into CALLER-OWNED h and parts.

    Split out of moe_forward_v3 for the resident-first path, where the two phases are separated in
    time by this layer's expert reads and therefore live in two different CUDA graphs. Two things
    follow from that and both are why this signature looks the way it does:

      * h and parts are the CALLER'S. moe_forward_v3 allocates them per call, which is fine inside
        one graph but gives the two phases different buffers. They must be the same memory.
      * there is NO reduction here. Doing `parts.view(K,T,DIM).sum(dim=0)` per phase and adding the
        results is NOT the same number: summing six terms as (p0+p2+p3)+(p1+p4+p5) differs from
        p0+p1+p2+p3+p4+p5 in float. The single fixed-order reduction stays in moe_v3_reduce, called
        once after the last phase.

    MASK THE BLOCK LIST, NOT THE SLOTS. Passing `slots` with the other phase's entries set to -1
    looks equivalent and is not: build_routing_small assumes "a slot never has more than BM pairs at
    this size", which holds for real slots but not for the -1 sentinel group. With 36 pairs and BM
    16, masking out 25 resident pairs puts 25 entries in block 0 and `blk * BM + rank` runs past its
    16 slots into block 1's pair list. tools/test_moe_split_bitwise.py caught that as a 1.3e6 delta.

    So the caller builds ONE routing from the full slots and passes `block_slot` already masked --
    the other phase's blocks set to -1, which both kernels early-out on. `block_pair` is shared and
    identical across phases, every pair is computed exactly once, and which block a pair sits in
    never changes its own accumulation over K.
    """
    assert x.dtype == torch.bfloat16 and x.shape[1] == DIM and x.is_contiguous()
    T, K = slots.shape
    P = T * K
    BM = block_m or _pick_bm(P)
    bn1, nw1, ns1 = cfg_up or CB3_UP_CFG[BM]
    bn2, nw2, ns2 = cfg_down or CB3_DOWN_CFG[BM]
    if routing is None:
        block_slot, block_pair, NB = build_routing_small(slots, BM)
    else:
        block_slot, block_pair, NB = routing
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    _cb3v3_up_kernel[(NB, INTER // bn1)](
        x, arena.w1_lo, arena.w1_hi, arena.w1_cb, arena.s1, arena.w3_lo, arena.w3_hi, arena.w3_cb,
        arena.s3, h, wgt, block_slot, block_pair, x.stride(0), h.stride(0), float(swiglu_limit),
        TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, NB512=CB3.block_plan(DIM)[0],
        NB256=CB3.block_plan(DIM)[1], num_warps=nw1, num_stages=ns1, PACKED=_packed(arena))
    _cb3v3_down_kernel[(NB, DIM // bn2)](
        h, arena.w2_lo, arena.w2_hi, arena.w2_cb, arena.s2, parts, block_slot, block_pair,
        h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T,
        NB512=CB3.block_plan(INTER)[0], NB256=CB3.block_plan(INTER)[1],
        num_warps=nw2, num_stages=ns2, PACKED=_packed(arena))


def moe_v3_reduce(parts: torch.Tensor, T: int, K: int) -> torch.Tensor:
    """The one fixed-order reduction, unchanged from moe_forward_v3 and called once per layer."""
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)


def moe_forward_v3_split(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor,
                         arena: CB3ArenaV2, swiglu_limit: float = 10.0, *,
                         block_slot, block_pair, NB: int, BM: int, phase_masks,
                         cfg_up=None, cfg_down=None) -> torch.Tensor:
    """moe_forward_v3 run in PHASES over the same routing, for the resident-first split.

    The point is to start the resident pairs while this layer's misses are still loading. Jobs
    400/405 put graph B at 69.4 ms/step of device time and jobs 460/465 put PAIR residency at
    94-95 %, so most of that work does not depend on the reads it currently waits behind.

    NO KERNEL CHANGE IS NEEDED, which is what makes this safe. `build_routing_small` emits one
    BM-block per distinct slot and residency is a property of the slot, so every block is wholly
    resident or wholly missing. Both kernels already early-out on `if slot < 0: return`. So a phase
    is just `block_slot` with the other phase's blocks set to -1: same `block_pair`, same NB, same
    tiling, same per-pair arithmetic. Every pair is computed exactly once, in the same program
    geometry it would have had unsplit, so the result is bitwise identical -- see
    tools/test_moe_split_bitwise.py, which is the gate on that claim.

    `phase_masks` is a sequence of masked block_slot tensors, applied in order. The caller is
    expected to wait for the expert reads between them; nothing here enforces that, because the
    ordering belongs to the driver and this function must stay graph-capturable.
    """
    assert x.dtype == torch.bfloat16 and x.shape[1] == DIM and x.is_contiguous()
    T, K = slots.shape
    P = T * K
    dev = x.device
    bn1, nw1, ns1 = cfg_up or CB3_UP_CFG[BM]
    bn2, nw2, ns2 = cfg_down or CB3_DOWN_CFG[BM]
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    # Allocated once and written across the phases: every pair lands in exactly one of them, so the
    # buffer is fully defined by the end for the same reason it is in the unsplit path.
    h = torch.empty((P, INTER), dtype=torch.bfloat16, device=dev)
    parts = torch.empty((P, DIM), dtype=torch.float32, device=dev)
    for mask in phase_masks:
        _cb3v3_up_kernel[(NB, INTER // bn1)](
            x, arena.w1_lo, arena.w1_hi, arena.w1_cb, arena.s1, arena.w3_lo, arena.w3_hi,
            arena.w3_cb, arena.s3, h, wgt, mask, block_pair, x.stride(0), h.stride(0),
            float(swiglu_limit), TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1,
            NB512=CB3.block_plan(DIM)[0], NB256=CB3.block_plan(DIM)[1],
            num_warps=nw1, num_stages=ns1, PACKED=_packed(arena))
        _cb3v3_down_kernel[(NB, DIM // bn2)](
            h, arena.w2_lo, arena.w2_hi, arena.w2_cb, arena.s2, parts, mask, block_pair,
            h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T,
            NB512=CB3.block_plan(INTER)[0], NB256=CB3.block_plan(INTER)[1],
            num_warps=nw2, num_stages=ns2, PACKED=_packed(arena))
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)


def moe_forward_prefill(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor,
                        arena: CB3ArenaV2, swiglu_limit: float = 10.0,
                        batch: int | None = None) -> torch.Tensor:
    """Prefill-sized call over a CB3 arena, via the FP4 kernel and a batched unpack scratch."""
    T, K = slots.shape
    P = T * K
    dev = x.device
    batch = batch or UNPACK_BATCH
    uniq = torch.unique(slots)
    uniq = uniq[uniq >= 0].to(torch.int32)
    n = int(uniq.numel())
    batch = min(batch, n)
    scratch = arena.fp4_scratch(batch)
    inv = torch.full((arena.slots,), -1, dtype=torch.int32, device=dev)
    ar = torch.arange(batch, dtype=torch.int32, device=dev)
    BM = _pick_bm(P)
    bn1, nw1, ns1 = F4._UP_CFG[BM]
    bn2, nw2, ns2 = F4._DOWN_CFG[BM]
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    # h and parts are written exactly once per (token, k) pair across the batches -- every pair's
    # expert is in exactly one batch -- so neither needs zeroing and the reduction runs once.
    h = torch.empty((P, INTER), dtype=torch.bfloat16, device=dev)
    parts = torch.empty((P, DIM), dtype=torch.float32, device=dev)
    # The layer-scoped cache turns the batch loop into a single pass whenever the scratch can hold
    # this call's whole expert set: experts already unpacked by an earlier chunk of the same layer
    # keep their scratch slot and are not touched, and only the new ones are unpacked -- in ONE
    # launch rather than one per batch of 32. Falls back to the original per-call batching when the
    # scratch is too small (SCRATCH_SLOTS=0, the old default).
    cache = getattr(arena, "_scratch_of", None) if SCRATCH_SLOTS > 0 else None
    if cache is not None and scratch.slots >= n:
        new_sel = [int(v) for v in uniq.tolist() if int(v) not in cache]
        if len(new_sel) > len(arena._scratch_free):
            arena.scratch_epoch()                       # cannot fit: start the epoch over
            cache = arena._scratch_of
            new_sel = [int(v) for v in uniq.tolist()]
        if new_sel:
            dst = [arena._scratch_free.pop() for _ in new_sel]
            _unpack_into(arena, torch.tensor(new_sel, dtype=torch.int32, device=dev), scratch,
                         torch.tensor(dst, dtype=torch.int32, device=dev))
            cache.update(zip(new_sel, dst))
        inv.fill_(-1)
        src_t = uniq.long()
        inv[src_t] = torch.tensor([cache[int(v)] for v in uniq.tolist()],
                                  dtype=torch.int32, device=dev)
        s2 = torch.where(slots >= 0, inv[slots.long().clamp_min(0)], slots.to(torch.int32))
        block_slot, block_pair, NB = build_routing(s2, scratch.slots, BM)
        with nvtx_range("moe.kernels"):
            F4._moe_up_kernel[(NB, INTER // bn1)](
                x, scratch.w1, scratch.s1, scratch.w3, scratch.s3, h, wgt, block_slot, block_pair,
                x.stride(0), h.stride(0), float(swiglu_limit),
                TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, num_warps=nw1, num_stages=ns1)
            F4._moe_down_kernel[(NB, DIM // bn2)](
                h, scratch.w2, scratch.s2, parts, block_slot, block_pair, h.stride(0), parts.stride(0),
                TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T, num_warps=nw2, num_stages=ns2)
        return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)

    for i in range(0, n, batch):
        sel = uniq[i:i + batch]
        b = int(sel.numel())
        _unpack_into(arena, sel, scratch)
        inv.fill_(-1)
        inv[sel.long()] = ar[:b]
        s2 = torch.where(slots >= 0, inv[slots.long().clamp_min(0)], slots.to(torch.int32))
        block_slot, block_pair, NB = build_routing(s2, b, BM)
        with nvtx_range("moe.kernels"):
            F4._moe_up_kernel[(NB, INTER // bn1)](
                x, scratch.w1, scratch.s1, scratch.w3, scratch.s3, h, wgt, block_slot, block_pair,
                x.stride(0), h.stride(0), float(swiglu_limit),
                TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, num_warps=nw1, num_stages=ns1)
            F4._moe_down_kernel[(NB, DIM // bn2)](
                h, scratch.w2, scratch.s2, parts, block_slot, block_pair, h.stride(0), parts.stride(0),
                TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T, num_warps=nw2, num_stages=ns2)
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)


# ============================================================================= CB2: the 2-bit tier
# CB2 is CB3 with the high-bit plane left out (tools/cb3.py): per row a codebook of four FP4 grid
# codes and two bits per weight, in the same v2 bit positions, with the same UE8M0 scales.
# 9.99 MB per expert against CB3's 14.45 (0.691x) and FP4's 18.80 (0.531x).
#
# Everything below is the v3 CB3 path with the `hi` loads and the high-bit arithmetic removed:
#   * the PTX decoder loses three instructions per four weights (no high-bit shift, no lop3 fusion
#     of it) and the codebook lookup needs only ONE prmt source register, because an index of 0..3
#     never names a byte above lane 3;
#   * `_pair_dot` no longer depends on the sub-tile's parity, which is what selected the hi bits;
#   * the row tiles the kernel loads keep their widths -- 128 B per 512-weight block, 64 B per
#     256-weight block -- so the tile-width cliff that made CB3 reach bandwidth still applies.

CB2_BYTES_PER_SLOT = (2 * (INTER * (DIM // 4 + 4) + INTER * SG1)
                      + DIM * (INTER // 4 + 4) + DIM * SG2)


class CB2ArenaV2:
    """Same shape as CB3ArenaV2 without the hi planes, and with a 4-entry codebook per row."""

    def __init__(self, slots: int, device: torch.device | str = "cuda"):
        self.slots = slots
        self.device = torch.device(device)
        u8 = dict(dtype=torch.uint8, device=self.device)
        self.w1_lo = torch.empty((slots, INTER, DIM // 4), **u8)
        self.w1_cb = torch.empty((slots, INTER, 4), **u8)
        self.s1 = torch.empty((slots, INTER, SG1), **u8)
        self.w3_lo = torch.empty((slots, INTER, DIM // 4), **u8)
        self.w3_cb = torch.empty((slots, INTER, 4), **u8)
        self.s3 = torch.empty((slots, INTER, SG1), **u8)
        self.w2_lo = torch.empty((slots, DIM, INTER // 4), **u8)
        self.w2_cb = torch.empty((slots, DIM, 4), **u8)
        self.s2 = torch.empty((slots, DIM, SG2), **u8)
        self.sim = None  # engine.codebook_sim.CodebookSim(2), set by the caller

    @property
    def bytes_per_slot(self) -> int:
        return sum(t[0].numel() for t in (self.w1_lo, self.w1_cb, self.s1, self.w3_lo, self.w3_cb,
                                          self.s3, self.w2_lo, self.w2_cb, self.s2))

    def fp4_scratch(self, slots: int):
        sc = getattr(self, "_scratch", None)
        if sc is None or sc.slots < slots:
            self._scratch = sc = F4.ExpertArena(max(slots, UNPACK_BATCH), self.device)
        return sc

    def load_slot(self, slot: int, w1, s1, w2, s2, w3, s3, non_blocking: bool = False, sim=None) -> None:
        sim = sim or self.sim
        assert sim is not None and sim.bits == 2, "CB2ArenaV2.sim must be a CodebookSim(2)"
        dev = self.device
        for (w, s, lo_t, cb_t, s_t) in ((w1, s1, self.w1_lo, self.w1_cb, self.s1),
                                        (w3, s3, self.w3_lo, self.w3_cb, self.s3),
                                        (w2, s2, self.w2_lo, self.w2_cb, self.s2)):
            wg = w.view(torch.uint8).to(dev, non_blocking=non_blocking)
            sg = s.view(torch.uint8).to(dev, non_blocking=non_blocking)
            lo, cb = fp4_to_cb2(wg, sg, sim)
            lo_t[slot].copy_(lo); cb_t[slot].copy_(cb); s_t[slot].copy_(sg)

    def dequant_slot(self, slot: int):
        return (dequant_cb2(self.w1_lo[slot], self.w1_cb[slot], self.s1[slot]),
                dequant_cb2(self.w2_lo[slot], self.w2_cb[slot], self.s2[slot]),
                dequant_cb2(self.w3_lo[slot], self.w3_cb[slot], self.s3[slot]))


def _cb2_asm(sh: int) -> str:
    """7 instructions per half (four weights), 16 per invocation (eight weights), against CB3's 20.

    The byte-lane -> nibble compaction is `_cb3_asm`'s, unchanged: `(a | (a >> 4)) & 0x00FF00FF`
    then `r | (r >> 8)` leaves the four 2-bit indices as the four low nibbles, which is the selector
    `prmt` wants. `prmt` takes the codebook register twice because an index of 0..3 only ever names
    a byte of the first source."""
    def half(shift, out):
        return f"""
shr.b32 a, $1, {shift};
and.b32 a, a, 0x03030303;
shr.b32 t, a, 4;
lop3.b32 r, a, t, 0x00FF00FF, 0xA8;
shr.b32 t, r, 8;
or.b32  r, r, t;
prmt.b32 {out}, $2, $2, r;"""
    return "{\n.reg .b32 a, t, r, ne, no;" + half(sh, "ne") + half(sh + 2, "no") + """
shl.b32 no, no, 4;
or.b32  $0, ne, no;
}
"""


_ASM2_0 = tl.constexpr(_cb2_asm(0))
_ASM2_4 = tl.constexpr(_cb2_asm(4))


@triton.jit
def _grp_packed_ptx2(Lk, A, ASM: tl.constexpr):
    return tl.inline_asm_elementwise(ASM, "=r,r,r", [Lk, A],
                                     dtype=tl.uint8, is_pure=True, pack=4)


@triton.jit
def _cb2_a(cb_ptr, BN: tl.constexpr):
    """The row's 4-entry codebook as one [BN, 16] uint8 tile whose every 4-element group is entries
    0..3, so pack=4 hands `prmt` exactly the source register it needs."""
    j = tl.arange(0, 16)[None, :]
    return tl.load(cb_ptr + (j % 4))


@triton.jit
def _pair_dot2(x_base, xk, mask_m, Lk, A, sA, sB):
    """The two scale groups that share one 16-byte lo sub-tile. Unlike CB3's `_pair_dot` there is no
    parity argument: the parity only ever chose which half of the hi sub-tile's bits to read."""
    p0 = _grp_packed_ptx2(Lk, A, _ASM2_0)
    p1 = _grp_packed_ptx2(Lk, A, _ASM2_4)
    acc = _chunk_dot(x_base, xk, mask_m, p0, sA)
    acc += _chunk_dot(x_base + 32, xk, mask_m, p1, sB)
    return acc


@triton.jit
def _cb2_block_dot(x_base, xk, mask_m, lo_ptr, s_ptr, A, BN: tl.constexpr, BW: tl.constexpr):
    """One packing block: BW logical K from a [BN, BW/4] lo tile. BW=512 -> a 128 B row tile,
    BW=256 -> 64 B; both are above the 64 B width at which this box's strided reads reach
    bandwidth, which the CB3 round measured."""
    if BW == 512:
        L = tl.load(lo_ptr)   # [BN, 128]
        La, Lb, Lc, Ld = _split4(L, BN, 32)
        L0, L1 = _split2(La, BN, 16)
        L2, L3 = _split2(Lb, BN, 16)
        L4, L5 = _split2(Lc, BN, 16)
        L6, L7 = _split2(Ld, BN, 16)
        s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15 = _split16(tl.load(s_ptr), BN)
        acc = _pair_dot2(x_base, xk, mask_m, L0, A, s0, s1)
        acc += _pair_dot2(x_base + 64, xk, mask_m, L1, A, s2, s3)
        acc += _pair_dot2(x_base + 128, xk, mask_m, L2, A, s4, s5)
        acc += _pair_dot2(x_base + 192, xk, mask_m, L3, A, s6, s7)
        acc += _pair_dot2(x_base + 256, xk, mask_m, L4, A, s8, s9)
        acc += _pair_dot2(x_base + 320, xk, mask_m, L5, A, s10, s11)
        acc += _pair_dot2(x_base + 384, xk, mask_m, L6, A, s12, s13)
        acc += _pair_dot2(x_base + 448, xk, mask_m, L7, A, s14, s15)
    else:
        L = tl.load(lo_ptr)   # [BN, 64]
        L0, L1, L2, L3 = _split4(L, BN, 16)
        s0, s1, s2, s3, s4, s5, s6, s7 = _split8(tl.load(s_ptr), BN)
        acc = _pair_dot2(x_base, xk, mask_m, L0, A, s0, s1)
        acc += _pair_dot2(x_base + 64, xk, mask_m, L1, A, s2, s3)
        acc += _pair_dot2(x_base + 128, xk, mask_m, L2, A, s4, s5)
        acc += _pair_dot2(x_base + 192, xk, mask_m, L3, A, s6, s7)
    return acc


@triton.jit
def _cb2_up_kernel(
    x_ptr, lo1_ptr, cb1_ptr, s1_ptr, lo3_ptr, cb3_ptr, s3_ptr, h_ptr,
    wgt_ptr, block_slot_ptr, block_pair_ptr,
    stride_x, stride_h, limit,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NB512: tl.constexpr, NB256: tl.constexpr,
):
    KL: tl.constexpr = K // 4
    SG: tl.constexpr = K // 32
    # Packed rows are [base u8][3 B per 8 groups]; unpacked are one byte per group.
    SSTRIDE: tl.constexpr = (1 + SG * 3 // 8) if PACKED else SG
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    tok = (offs_m // TOPK).to(tl.int64)
    offs_n = nb * BN + tl.arange(0, BN)
    x_base = x_ptr + tok[:, None] * stride_x
    xk = 2 * tl.arange(0, 16)[None, :]
    lo1 = lo1_ptr + slot * (N * KL) + offs_n[:, None] * KL
    lo3 = lo3_ptr + slot * (N * KL) + offs_n[:, None] * KL
    s1t = s1_ptr + slot * (N * SSTRIDE) + offs_n[:, None] * SSTRIDE
    s3t = s3_ptr + slot * (N * SSTRIDE) + offs_n[:, None] * SSTRIDE
    A1 = _cb2_a(cb1_ptr + slot * (N * 4) + offs_n[:, None] * 4, BN)
    A3 = _cb2_a(cb3_ptr + slot * (N * 4) + offs_n[:, None] * 4, BN)
    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    acc_u = tl.zeros([BM, BN], dtype=tl.float32)
    l1a = lo1 + tl.arange(0, 128)[None, :]; s1a = s1t + tl.arange(0, 16)[None, :]
    l3a = lo3 + tl.arange(0, 128)[None, :]; s3a = s3t + tl.arange(0, 16)[None, :]
    for b in range(0, NB512):
        acc_g += _cb2_block_dot(x_base + b * 512, xk, mask_m[:, None], l1a + b * 128, s1a + b * 16, A1, BN, 512)
        acc_u += _cb2_block_dot(x_base + b * 512, xk, mask_m[:, None], l3a + b * 128, s3a + b * 16, A3, BN, 512)
    if NB256 > 0:
        o: tl.constexpr = NB512 * 512
        l1b = lo1 + (NB512 * 128 + tl.arange(0, 64))[None, :]; s1b = s1t + (NB512 * 16 + tl.arange(0, 8))[None, :]
        l3b = lo3 + (NB512 * 128 + tl.arange(0, 64))[None, :]; s3b = s3t + (NB512 * 16 + tl.arange(0, 8))[None, :]
        for b in range(0, NB256):
            acc_g += _cb2_block_dot(x_base + o + b * 256, xk, mask_m[:, None], l1b + b * 64, s1b + b * 8, A1, BN, 256)
            acc_u += _cb2_block_dot(x_base + o + b * 256, xk, mask_m[:, None], l3b + b * 64, s3b + b * 8, A3, BN, 256)
    gate = tl.minimum(acc_g, limit)
    up = tl.minimum(tl.maximum(acc_u, -limit), limit)
    wgt = tl.load(wgt_ptr + offs_m, mask=mask_m, other=0.0)
    h = gate * tl.sigmoid(gate) * up * wgt[:, None]
    tl.store(h_ptr + offs_m[:, None] * stride_h + offs_n[None, :], h.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _cb2_down_kernel(
    h_ptr, lo2_ptr, cb2_ptr, s2_ptr, y_ptr,
    block_slot_ptr, block_pair_ptr,
    stride_h, stride_y,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NTOK: tl.constexpr, NB512: tl.constexpr, NB256: tl.constexpr,
):
    KL: tl.constexpr = K // 4
    SG: tl.constexpr = K // 32
    # Packed rows are [base u8][3 B per 8 groups]; unpacked are one byte per group.
    SSTRIDE: tl.constexpr = (1 + SG * 3 // 8) if PACKED else SG
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = nb * BN + tl.arange(0, BN)
    h_base = h_ptr + offs_m[:, None].to(tl.int64) * stride_h
    xk = 2 * tl.arange(0, 16)[None, :]
    lo2 = lo2_ptr + slot * (N * KL) + offs_n[:, None] * KL
    s2t = s2_ptr + slot * (N * SSTRIDE) + offs_n[:, None] * SSTRIDE
    A2 = _cb2_a(cb2_ptr + slot * (N * 4) + offs_n[:, None] * 4, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    l2a = lo2 + tl.arange(0, 128)[None, :]; s2a = s2t + tl.arange(0, 16)[None, :]
    for b in range(0, NB512):
        acc += _cb2_block_dot(h_base + b * 512, xk, mask_m[:, None], l2a + b * 128, s2a + b * 16, A2, BN, 512)
    if NB256 > 0:
        o: tl.constexpr = NB512 * 512
        l2b = lo2 + (NB512 * 128 + tl.arange(0, 64))[None, :]; s2b = s2t + (NB512 * 16 + tl.arange(0, 8))[None, :]
        for b in range(0, NB256):
            acc += _cb2_block_dot(h_base + o + b * 256, xk, mask_m[:, None], l2b + b * 64, s2b + b * 8, A2, BN, 256)
    row = ((offs_m % TOPK) * NTOK + offs_m // TOPK).to(tl.int64)
    tl.store(y_ptr + row[:, None] * stride_y + offs_n[None, :], acc, mask=mask_m[:, None])


CB2_UP_CFG = {16: (32, 4, 3), 32: (32, 4, 3), 64: (32, 4, 3)}
CB2_DOWN_CFG = {16: (32, 4, 3), 32: (32, 4, 3), 64: (32, 4, 3)}


def moe_forward_cb2(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena: CB2ArenaV2,
                    swiglu_limit: float = 10.0, block_m: int | None = None,
                    cfg_up=None, cfg_down=None) -> torch.Tensor:
    """Decode-shaped MoE over a CB2 arena. Same contract as `moe_forward_v3`."""
    assert x.dtype == torch.bfloat16 and x.shape[1] == DIM and x.is_contiguous()
    T, K = slots.shape
    P = T * K
    if P >= PREFILL_MIN_P and PREFILL_MODE == "fp4" and block_m is None and cfg_up is None:
        return moe_forward_cb2_prefill(x, slots, weights, arena, swiglu_limit)
    dev = x.device
    BM = block_m or _pick_bm(P)
    bn1, nw1, ns1 = cfg_up or CB2_UP_CFG[BM]
    bn2, nw2, ns2 = cfg_down or CB2_DOWN_CFG[BM]
    if P <= 64:
        block_slot, block_pair, NB = build_routing_small(slots, BM)
    else:
        block_slot, block_pair, NB = build_routing(slots, arena.slots, BM)
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    h = torch.empty((P, INTER), dtype=torch.bfloat16, device=dev)
    parts = torch.empty((P, DIM), dtype=torch.float32, device=dev)
    _cb2_up_kernel[(NB, INTER // bn1)](
        x, arena.w1_lo, arena.w1_cb, arena.s1, arena.w3_lo, arena.w3_cb, arena.s3, h,
        wgt, block_slot, block_pair, x.stride(0), h.stride(0), float(swiglu_limit),
        TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1,
        NB512=CB3.block_plan(DIM)[0], NB256=CB3.block_plan(DIM)[1],
        num_warps=nw1, num_stages=ns1)
    _cb2_down_kernel[(NB, DIM // bn2)](
        h, arena.w2_lo, arena.w2_cb, arena.s2, parts, block_slot, block_pair,
        h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T,
        NB512=CB3.block_plan(INTER)[0], NB256=CB3.block_plan(INTER)[1],
        num_warps=nw2, num_stages=ns2)
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)


# ---------------------------------------------------------------------------- CB2 prefill unpack
@triton.jit
def _unpack_pair2(out_base, Lk, A, BN: tl.constexpr):
    j = tl.arange(0, 16)[None, :]
    tl.store(out_base + j, _grp_packed_ptx2(Lk, A, _ASM2_0))
    tl.store(out_base + 16 + j, _grp_packed_ptx2(Lk, A, _ASM2_4))


@triton.jit
def _cb2_unpack_kernel(LO, CB, OUT, SRC, N,
                       KL: tl.constexpr, KB: tl.constexpr,
                       BN: tl.constexpr, NB512: tl.constexpr, NB256: tl.constexpr):
    b = tl.program_id(0)
    nb = tl.program_id(1)
    src = tl.load(SRC + b).to(tl.int64)
    offs_n = nb * BN + tl.arange(0, BN)
    lo = LO + src * (N * KL) + offs_n[:, None] * KL
    out = OUT + b.to(tl.int64) * (N * KB) + offs_n[:, None] * KB
    A = _cb2_a(CB + src * (N * 4) + offs_n[:, None] * 4, BN)
    for j in range(0, NB512):
        L = tl.load(lo + (j * 128 + tl.arange(0, 128))[None, :])
        La, Lb, Lc, Ld = _split4(L, BN, 32)
        L0, L1 = _split2(La, BN, 16)
        L2, L3 = _split2(Lb, BN, 16)
        L4, L5 = _split2(Lc, BN, 16)
        L6, L7 = _split2(Ld, BN, 16)
        o = out + j * 256
        _unpack_pair2(o, L0, A, BN)
        _unpack_pair2(o + 32, L1, A, BN)
        _unpack_pair2(o + 64, L2, A, BN)
        _unpack_pair2(o + 96, L3, A, BN)
        _unpack_pair2(o + 128, L4, A, BN)
        _unpack_pair2(o + 160, L5, A, BN)
        _unpack_pair2(o + 192, L6, A, BN)
        _unpack_pair2(o + 224, L7, A, BN)
    if NB256 > 0:
        for j in range(0, NB256):
            L = tl.load(lo + (NB512 * 128 + j * 64 + tl.arange(0, 64))[None, :])
            L0, L1, L2, L3 = _split4(L, BN, 16)
            o = out + NB512 * 256 + j * 128
            _unpack_pair2(o, L0, A, BN)
            _unpack_pair2(o + 32, L1, A, BN)
            _unpack_pair2(o + 64, L2, A, BN)
            _unpack_pair2(o + 96, L3, A, BN)


def _unpack_into_cb2(arena, src_slots: torch.Tensor, scratch) -> None:
    B = src_slots.numel()
    BN = 64
    for (lo, cb, N, K, out, s_src, s_dst) in (
            (arena.w1_lo, arena.w1_cb, INTER, DIM, scratch.w1, arena.s1, scratch.s1),
            (arena.w3_lo, arena.w3_cb, INTER, DIM, scratch.w3, arena.s3, scratch.s3),
            (arena.w2_lo, arena.w2_cb, DIM, INTER, scratch.w2, arena.s2, scratch.s2)):
        n512, n256 = CB3.block_plan(K)
        _cb2_unpack_kernel[(B, triton.cdiv(N, BN))](
            lo, cb, out, src_slots, N, KL=K // 4, KB=K // 2,
            BN=BN, NB512=n512, NB256=n256, num_warps=4, num_stages=2)
        s_dst[:B].copy_(s_src[src_slots.long()])


def moe_forward_cb2_prefill(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor,
                            arena: CB2ArenaV2, swiglu_limit: float = 10.0,
                            batch: int | None = None) -> torch.Tensor:
    """Prefill-sized call over a CB2 arena: unpack to packed FP4 in batches (bit-exact, the CB2
    codes are a subset of the FP4 grid) and run the FP4 kernel, exactly as the CB3 path does."""
    T, K = slots.shape
    P = T * K
    dev = x.device
    batch = batch or UNPACK_BATCH
    uniq = torch.unique(slots)
    uniq = uniq[uniq >= 0].to(torch.int32)
    n = int(uniq.numel())
    batch = min(batch, n)
    scratch = arena.fp4_scratch(batch)
    inv = torch.full((arena.slots,), -1, dtype=torch.int32, device=dev)
    ar = torch.arange(batch, dtype=torch.int32, device=dev)
    BM = _pick_bm(P)
    bn1, nw1, ns1 = F4._UP_CFG[BM]
    bn2, nw2, ns2 = F4._DOWN_CFG[BM]
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    h = torch.empty((P, INTER), dtype=torch.bfloat16, device=dev)
    parts = torch.empty((P, DIM), dtype=torch.float32, device=dev)
    for i in range(0, n, batch):
        sel = uniq[i:i + batch]
        b = int(sel.numel())
        _unpack_into_cb2(arena, sel, scratch)
        inv.fill_(-1)
        inv[sel.long()] = ar[:b]
        s2 = torch.where(slots >= 0, inv[slots.long().clamp_min(0)], slots.to(torch.int32))
        block_slot, block_pair, NB = build_routing(s2, b, BM)
        F4._moe_up_kernel[(NB, INTER // bn1)](
            x, scratch.w1, scratch.s1, scratch.w3, scratch.s3, h, wgt, block_slot, block_pair,
            x.stride(0), h.stride(0), float(swiglu_limit),
            TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, num_warps=nw1, num_stages=ns1)
        F4._moe_down_kernel[(NB, DIM // bn2)](
            h, scratch.w2, scratch.s2, parts, block_slot, block_pair,
            h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T,
            num_warps=nw2, num_stages=ns2)
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)
