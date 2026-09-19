#!/usr/bin/env python3
"""Lossless 3-bit codec for DeepSeek-V4.1 UE8M0 group scales.

Measured over all 40 layers, 149,422,080 rows (notes/data/scale-survey-20260913.txt): the exponent
range inside one output row never exceeds 7. So a row stores one 8-bit base plus 3 bits per group,
exactly, with no escape path. 2 bits would NOT be lossless -- range <= 3 fails on 0.2% of layer 39.

THAT SURVEY IS LAYERS 0-39 ONLY, and the bound does not hold outside it. Job 985 tried to pack the
DSpark draft experts (mtp.0/1/2) and layers 0 and 1 went through clean, 128 experts each, while mtp.2
raised on a row of intra-row range 8. So "lossless on this checkpoint" means lossless on the ROUTED
experts, not on the checkpoint: any new tensor family has to be surveyed before it is packed, and the
raise below is the thing that catches it. A CB3 draft arena therefore keeps its scales unpacked
(engine/v41_engine.py, DSV41_DRAFT_CB3) -- which costs it nothing, because a resident arena never
reads a record off disk.

Layout per row: [base u8][ceil(groups*3/8) bytes], groups packed 8-at-a-time as a 24-bit
little-endian word, value i at bit 3*i. Both real group counts are multiples of 8 (160 -> 60 B,
72 -> 27 B), so no partial word ever occurs.

Both a numpy and a torch implementation, because the builder packs on the host and the engine
unpacks on the device.
"""
import numpy as np

CODEC = "ue8m0-3bit-rowbase-v1"
BITS = 3

SHIFTS = np.arange(8, dtype=np.uint32) * 3


def packed_row_bytes(groups: int) -> int:
    assert groups % 8 == 0, f"group count {groups} must be a multiple of 8"
    return 1 + groups * 3 // 8


def pack(x: np.ndarray) -> np.ndarray:
    """x: uint8 [rows, groups] of UE8M0 exponents -> uint8 [rows, 1 + groups*3//8]."""
    assert x.dtype == np.uint8 and x.ndim == 2
    rows, groups = x.shape
    assert groups % 8 == 0
    base = x.min(axis=1)
    d = (x.astype(np.int16) - base[:, None].astype(np.int16))
    if d.max() > 7:
        raise ValueError(f"intra-row range {d.max()} > 7; this row is not representable in 3 bits")
    w = (d.astype(np.uint32).reshape(rows, groups // 8, 8) << SHIFTS).sum(axis=2, dtype=np.uint32)
    b = np.stack([w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF], axis=2).astype(np.uint8)
    return np.concatenate([base[:, None], b.reshape(rows, groups * 3 // 8)], axis=1)


def unpack(p: np.ndarray, groups: int) -> np.ndarray:
    """Inverse of `pack`. p: uint8 [rows, 1 + groups*3//8] -> uint8 [rows, groups]."""
    rows = p.shape[0]
    base, b = p[:, :1], p[:, 1:].reshape(rows, groups // 8, 3).astype(np.uint32)
    w = b[:, :, 0] | (b[:, :, 1] << 8) | (b[:, :, 2] << 16)
    d = ((w[:, :, None] >> SHIFTS) & 7).astype(np.uint8).reshape(rows, groups)
    return (d + base).astype(np.uint8)


_SH_CACHE = {}


def unpack_torch(p, groups: int):
    """Device-side inverse, for the engine's load path. p: uint8 [rows, 1+groups*3//8] on any device."""
    import torch
    rows = p.shape[0]
    base = p[:, :1].to(torch.int16)
    b = p[:, 1:].reshape(rows, groups // 8, 3).to(torch.int32)
    w = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    sh = _SH_CACHE.get(p.device)
    if sh is None:
        sh = _SH_CACHE[p.device] = torch.arange(8, device=p.device, dtype=torch.int32) * 3
    d = ((w.unsqueeze(-1) >> sh) & 7).reshape(rows, groups).to(torch.int16)
    return (d + base).to(torch.uint8)


def pack_torch(x):
    """Device-side forward, for the engine's CHECKPOINT load path. x: uint8 [rows, groups].

    The builder packs on the host with `pack`; a packed ARENA also has to pack experts arriving from
    the FP4 checkpoint, which are already on the device. Mirrors `pack` exactly -- same row base,
    same 24-bit little-endian word, same bit positions -- and `test_pack_torch_matches_numpy` in the
    gate asserts byte equality against it rather than assuming.
    """
    import torch
    assert x.dtype == torch.uint8 and x.ndim == 2
    rows, groups = x.shape
    assert groups % 8 == 0, groups
    base = x.min(dim=1, keepdim=True).values                      # [rows, 1]
    d = (x.to(torch.int16) - base.to(torch.int16))
    mx = int(d.max())
    if mx > 7:
        raise ValueError(f"intra-row range {mx} > 7; this row is not representable in 3 bits")
    sh = _SH_CACHE.get(x.device)
    if sh is None:
        sh = _SH_CACHE[x.device] = torch.arange(8, device=x.device, dtype=torch.int32) * 3
    w = (d.to(torch.int32).reshape(rows, groups // 8, 8) << sh).sum(dim=2)   # [rows, groups//8]
    b = torch.stack([w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF], dim=2).to(torch.uint8)
    return torch.cat([base, b.reshape(rows, groups * 3 // 8)], dim=1)
