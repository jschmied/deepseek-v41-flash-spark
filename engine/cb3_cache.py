"""Native on-disk CB3 expert cache: read a slot without touching the FP4 checkpoint.

Without this, an LRU miss reads 18,800,640 B of packed FP4 by O_DIRECT, ships it to the GPU and
repacks it to CB3 to fill a 14,454,784 B slot. With it, a miss is one aligned 13,774,848 B read whose
bytes are already the arena's layout -- 26.7 % fewer bytes, one contiguous extent instead of the
checkpoint's two runs 585 MB apart, and no `fp4_to_cb3_v2` per miss.

Scales are stored 3 bits per group plus a 1-byte row base. That is exactly lossless on this
checkpoint: the intra-row UE8M0 range never exceeds 7 over all 149,422,080 rows of all 40 layers
(deepseek-moe-gb10/notes/data/scale-survey-20260913.txt). Two bits would not be.

Built by deepseek-moe-gb10/tools/cb3_cache_build.py, which writes the manifest this reads.
"""
from __future__ import annotations

import json
import os

import torch

ALIGN = 4096


class CB3Cache:
    """Reader for the expert-major cache file. Record i = layer*n_experts + expert."""

    def __init__(self, path: str, device: torch.device | str = "cuda"):
        self.path = path
        self.man = json.load(open(path + ".json"))
        self.record = int(self.man["record_bytes"])
        self.n_experts = int(self.man["n_experts"])
        self.planes = {k: tuple(v) for k, v in self.man["planes"].items()}
        self.groups = {k: int(v) for k, v in self.man["scale_groups"].items()}
        assert int(self.man["scale_bits"]) == 3, "only the 3-bit scale codec is implemented"
        assert self.record % ALIGN == 0
        self.device = torch.device(device)
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self._sh = torch.arange(8, device=self.device, dtype=torch.int32) * 3

    def close(self):
        if getattr(self, "fd", None) is not None:
            os.close(self.fd)
            self.fd = None

    def offset(self, layer: int, expert: int) -> int:
        return (layer * self.n_experts + expert) * self.record

    def read_into(self, view: memoryview, layer: int, expert: int) -> None:
        """O_DIRECT-read one record into `view`, an aligned slice of a pinned buffer.

        One `pread` loop, not six: the record is one contiguous, page-aligned extent by construction,
        which is the whole point of the format. `view` must be at least `record` bytes.
        """
        off = self.offset(layer, expert)
        got = 0
        while got < self.record:
            r = os.preadv(self.fd, [view[got:self.record]], off + got)
            if r <= 0:
                raise IOError(f"short read at {off}+{got}/{self.record}")
            got += r

    def _unpack_scales(self, buf: torch.Tensor, name: str, rows: int) -> torch.Tensor:
        g = self.groups[name]
        p = buf.reshape(rows, 1 + g * 3 // 8)
        base = p[:, :1].to(torch.int16)
        b = p[:, 1:].reshape(rows, g // 8, 3).to(torch.int32)
        w = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
        d = ((w.unsqueeze(-1) >> self._sh) & 7).reshape(rows, g).to(torch.int16)
        return (d + base).to(torch.uint8)

    def load_slot(self, arena, slot: int, staged: torch.Tensor, non_blocking: bool = False) -> None:
        """Copy one cached record from `staged` (a uint8 CPU tensor holding the record) into `slot`.

        The planes are stored in the arena's own order, so each is a slice and a `copy_`; only the
        three scale planes need expanding, which is a handful of integer ops on the device and
        replaces the per-miss `fp4_to_cb3_v2` the FP4 path pays.
        """
        for name in ("w1_lo", "w1_hi", "w1_cb", "w3_lo", "w3_hi", "w3_cb", "w2_lo", "w2_hi", "w2_cb"):
            lo, hi = self.planes[name]
            dst = getattr(arena, name)[slot]
            dst.copy_(staged[lo:hi].view_as(dst.reshape(-1)).reshape(dst.shape),
                      non_blocking=non_blocking)
        for name, rows in (("s1", arena.s1.shape[1]), ("s3", arena.s3.shape[1]),
                           ("s2", arena.s2.shape[1])):
            lo, hi = self.planes[name]
            src = staged[lo:hi].to(self.device, non_blocking=non_blocking)
            getattr(arena, name)[slot].copy_(self._unpack_scales(src, name, rows),
                                             non_blocking=non_blocking)
