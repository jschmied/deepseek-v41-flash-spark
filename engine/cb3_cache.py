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
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import scale_codec as SC  # noqa: E402  the writer uses the same module, by design

ALIGN = 4096
FORMAT_VERSION = 1


class CB3Cache:
    """Reader for the expert-major cache file. Record i = layer*n_experts + expert."""

    def __init__(self, path: str, device: torch.device | str = "cuda"):
        self.path = path
        self.man = json.load(open(path + ".json"))
        self.record = int(self.man["record_bytes"])
        self.n_experts = int(self.man["n_experts"])
        self.planes = {k: tuple(v) for k, v in self.man["planes"].items()}
        self.groups = {k: int(v) for k, v in self.man["scale_groups"].items()}
        v = int(self.man.get("format_version", 0))
        if v != FORMAT_VERSION:
            raise ValueError(f"{path}: record format v{v}, this engine reads v{FORMAT_VERSION}. "
                             f"Rebuild with tools/cb3_cache_build.py.")
        if self.man.get("codec") != SC.CODEC:
            raise ValueError(f"{path}: scale codec {self.man.get('codec')!r} != {SC.CODEC!r}")
        assert int(self.man["scale_bits"]) == SC.BITS
        assert self.record % ALIGN == 0
        self.device = torch.device(device)
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)

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
        return SC.unpack_torch(buf.reshape(rows, SC.packed_row_bytes(g)), g)

    def load_slot(self, arena, slot: int, staged: torch.Tensor, non_blocking: bool = False) -> None:
        """Copy one cached record from `staged` (a uint8 CPU tensor holding the record) into `slot`.

        The planes are stored in the arena's own order, so each is a slice and a `copy_`; only the
        three scale planes need expanding, which is a handful of integer ops on the device and
        replaces the per-miss `fp4_to_cb3_v2` the FP4 path pays.
        """
        inv = getattr(arena, "invalidate_scratch", None)
        if inv is not None:
            inv(slot)   # the layer-scoped unpack cache must not keep the outgoing expert
        for name in ("w1_lo", "w1_hi", "w1_cb", "w3_lo", "w3_hi", "w3_cb", "w2_lo", "w2_hi", "w2_cb"):
            lo, hi = self.planes[name]
            dst = getattr(arena, name)[slot]
            dst.copy_(staged[lo:hi].view_as(dst.reshape(-1)).reshape(dst.shape),
                      non_blocking=non_blocking)
        packed = bool(getattr(arena, "packed_scales", False))
        for name, rows in (("s1", arena.s1.shape[1]), ("s3", arena.s3.shape[1]),
                           ("s2", arena.s2.shape[1])):
            lo, hi = self.planes[name]
            dst = getattr(arena, name)[slot]
            if packed:
                # THE POINT OF CHANGE A. The record already holds `ue8m0-3bit-rowbase-v1`, so a
                # packed arena wants those bytes verbatim: three device-side `_unpack_scales` calls
                # per miss disappear, and the plane shrinks 160 -> 61 B/row (s1/s3) and 72 -> 28
                # (s2). This does NOT make the slot one contiguous H2D -- the arena is still twelve
                # plane-major tensors, and record-major storage is a separate change.
                dst.copy_(staged[lo:hi].view_as(dst.reshape(-1)).reshape(dst.shape),
                          non_blocking=non_blocking)
            else:
                src = staged[lo:hi].to(self.device, non_blocking=non_blocking)
                dst.copy_(self._unpack_scales(src, name, rows), non_blocking=non_blocking)
