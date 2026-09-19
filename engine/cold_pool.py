"""A pinned pool of record-major expert slots, filled by O_DIRECT straight off the pack.

This is the fetch half of the cold path; engine/cold_promotion.py is the lifetime half. Together
they implement: read a missing expert directly into its final mapped slot, compute from it there,
then promote it into the device hot arena as one contiguous copy.

WHY O_DIRECT INTO THE SLOT IS POSSIBLE AT ALL. With packed scales the arena slot is byte-identical to
the pack's record payload (13,773,312 B), and the pack pads each record to 13,774,848 = 3363 x 4096.
So file offsets, transfer lengths and slot offsets are all 4096-aligned and an expert needs no
transform between disk and kernel. With unpacked scales the slot is 14,454,784 and none of this works
-- the scale planes would have to be expanded on the way in.

The pool is allocated ONCE. Decode runs in CUDA graphs with baked base pointers, so a pool that grew
or moved would invalidate them; the slot count is a construction parameter for that reason.
"""
from __future__ import annotations

import json
import os
import time

import torch

from engine.cold_promotion import PromotionPool


class ColdPool:
    def __init__(self, pack_path: str, n_slots: int = 64, arena_cls=None):
        man = json.load(open(pack_path + ".json"))
        self.record_bytes = int(man["record_bytes"])      # padded, what the file strides by
        self.payload = int(man["payload_bytes"])          # what a slot holds
        self.n_records = os.path.getsize(pack_path) // self.record_bytes
        self.n_experts_per_layer = int(man["n_experts"])
        if arena_cls is None:
            import cb3_moe as C3
            arena_cls = C3.CB3RecordArena
        self.arena = arena_cls(n_slots, device="cpu", packed_scales=True, pinned=True)
        if self.arena.payload != self.payload:
            raise ValueError(f"arena payload {self.arena.payload} != pack payload {self.payload}")
        if self.arena.rstride % 4096:
            raise ValueError(f"slot stride {self.arena.rstride} is not 4096-aligned; O_DIRECT needs it")
        if self.arena.rstride != self.record_bytes:
            raise ValueError(f"slot stride {self.arena.rstride} != pack record stride "
                             f"{self.record_bytes}; a whole padded record must fit a slot exactly")
        self.promo = PromotionPool(n_slots)
        self.fd = os.open(pack_path, os.O_RDONLY | os.O_DIRECT)
        base = self.arena.buf.data_ptr()
        if base % 4096:
            raise RuntimeError(f"pinned base is not page aligned ({base % 4096}); O_DIRECT needs it")
        self._mv = memoryview(self.arena.buf.numpy())
        self.stats = {"reads": 0, "bytes": 0, "promotions": 0, "read_s": 0.0}
        self._events: dict = {}          # key -> (slot, gen, promo_event, compute_event)
        self.last_read_s = 0.0

    # ------------------------------------------------------------------ fetch
    def record_index(self, layer: int, expert: int) -> int:
        """The pack is record i = layer * n_experts + expert, per engine/cb3_cache.py."""
        return layer * self.n_experts_per_layer + expert

    def reserve(self, key: tuple, hot_slot: int) -> tuple[int, int]:
        """Take a cold slot. Host-side only -- no I/O, so it is safe to do for every miss of a layer
        before any read starts, which is what lets the reads run concurrently."""
        return self.promo.reserve(key, hot_slot)

    def read_into(self, key: tuple, slot: int, gen: int) -> None:
        """Do the O_DIRECT read for an already-reserved slot. Called on an I/O WORKER.

        Split from reserve() because the first version did both inline in resolve()'s miss loop, which
        serialised the reads: the ordinary path submits them to a pool and gets io_threads of
        concurrency, and doing a blocking preadv per miss threw that away. Nothing here touches CUDA,
        so it is safe off the main thread.
        """
        layer, expert = key
        off = slot * self.arena.rstride
        _t = time.perf_counter()
        try:
            got = os.preadv(self.fd, [self._mv[off:off + self.record_bytes]],
                            self.record_index(layer, expert) * self.record_bytes)
        except OSError:
            # A read that never delivered bytes has no compute party and never will, so the slot must
            # be released now rather than waiting for one.
            self.promo.abort_before_compute(slot, gen)
            raise
        dt = time.perf_counter() - _t
        self.last_read_s = dt
        self.stats["read_s"] += dt
        if got != self.record_bytes:
            self.promo.abort_before_compute(slot, gen)
            raise IOError(f"short read for {key}: {got} of {self.record_bytes}")
        self.stats["reads"] += 1
        self.stats["bytes"] += got
        self.promo.cold_ready(slot, gen)

    def fetch(self, key: tuple, hot_slot: int) -> tuple[int, int]:
        """reserve + read, for callers that do not need concurrency (tests)."""
        slot, gen = self.reserve(key, hot_slot)
        self.read_into(key, slot, gen)
        return slot, gen

    def event_of(self, key: tuple):
        return self._events.get(key)

    # ------------------------------------------------------------------ promotion
    def promote(self, key: tuple, hot_arena, stream=None):
        """Issue the contiguous cold -> hot copy. Returns (slot, gen, event); the caller marks
        promo_done when the EVENT LANDS, never when it is enqueued."""
        ent = self.promo._inflight.get(key)
        if ent is None:
            raise RuntimeError(f"{key} has no cold slot in flight")
        slot, gen, hot_slot = ent
        ctx = torch.cuda.stream(stream) if stream is not None else torch.cuda.stream(
            torch.cuda.current_stream())
        with ctx:
            if getattr(hot_arena, "rstride", 0) == self.arena.rstride:
                # Both record-major: ONE contiguous copy, which is the point of the layout.
                self.arena.promote_into(hot_arena, hot_slot, slot, non_blocking=True)
            else:
                # Plane-major destination: twelve scatters, the very thing record-major removes.
                # Kept so the cold path can be gated on without converting the main arena first --
                # correct, slower, and the promotion cost measured this way is an upper bound.
                import cb3_moe as _C3
                for nm in _C3.PLANE_ORDER:
                    getattr(hot_arena, nm)[hot_slot].view(-1).copy_(
                        self.arena.slot_view(slot, nm).view(-1), non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
        self.stats["promotions"] += 1
        return slot, gen, ev

    def close(self):
        if getattr(self, "fd", None) is not None:
            os.close(self.fd)
            self.fd = None
