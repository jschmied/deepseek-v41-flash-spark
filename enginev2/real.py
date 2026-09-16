"""real.py -- the real leaves. Swaps the modelled provider for CB3 on NVMe and the engine's graphs.

THE SEAM IS THE ONLY THING THAT CHANGES. Engine, LoaderService, ExpertSlots, Chain and the observer
are untouched: this implements Leaves and nothing else. If that turns out to be untrue, the skeleton
was wrong about where the boundary is and that is worth knowing before anything else.

WHAT IS ALREADY BUILT AND TESTED IN v1, and is therefore called rather than rewritten:
  CB3Cache.read_into(view, layer, expert)  one O_DIRECT pread loop into an aligned pinned slice
  CB3Cache.load_slot(arena, slot, staged)  the record into the arena, expanding the three scales
Both come from engine/cb3_cache.py. The record is 13,774,848 B, page-aligned by construction.

WHAT THIS FILE OWNS: the pinned pool (O_DIRECT needs ALIGN-aligned addresses, which torch does not
promise), and the mapping from the skeleton's Leaves contract onto those two calls.

Stage 1 and 2 of the bring-up. Graph A/B are not here yet -- see notes at the bottom.
"""

from __future__ import annotations

import os
import sys

import torch

from .leaves import Leaves, StagedExpert

ALIGN = 4096


class PinnedStagingPool:
    """Real pinned buffers, handed out one per in-flight read, with ALIGN-aligned views.

    Two things the modelled pool did not have to care about:
      * O_DIRECT requires the destination address to be ALIGN-aligned, and torch.empty gives no
        such promise -- so each buffer is over-allocated by ALIGN and the view starts at the first
        aligned address inside it. v1's ExpertStore does the same `(-base_addr) % ALIGN` dance.
      * pin_memory is not free. `n` buffers of `record` bytes is n x 13.1 MiB of page-locked host
        memory that the driver cannot page out; at 48 leases that is 630 MiB, which on this box is
        real money against the arena. The count is the caller's to choose and is reported.

    The lease protocol is deliberately v1's and the skeleton's: semaphore first, then the free list,
    so exact agreement between the two is an AT-REST property only.
    """

    def __init__(self, n: int, record: int, device: str = "cuda"):
        import threading
        self.n = n
        self.record = record
        self.nbytes = record + ALIGN
        self._sem = threading.Semaphore(n)
        self._lk = threading.Lock()
        self.free = list(range(n))
        self.peak_in_use = 0
        self._leased: set[int] = set()
        self._buf: list = []
        self._view: list = []
        for _ in range(n):
            b = torch.empty(self.nbytes, dtype=torch.uint8, pin_memory=True)
            off = (-b.data_ptr()) % ALIGN
            self._buf.append(b)
            self._view.append(b[off:off + record])
        self.bytes_pinned = n * self.nbytes

    # --- the skeleton's StagingPool surface -------------------------------------------------
    def acquire(self, ctx=None, scored: bool = True) -> int:
        self._sem.acquire()
        with self._lk:
            sid = self.free.pop()
            self._leased.add(sid)
            self.peak_in_use = max(self.peak_in_use, self.n - len(self.free))
            return sid

    def release(self, sid: int) -> None:
        with self._lk:
            if sid not in self._leased:
                raise RuntimeError(f"staging buffer {sid} released twice")
            self._leased.discard(sid)
            self.free.append(sid)
        self._sem.release()

    def buffer(self, sid: int) -> torch.Tensor:
        if sid not in self._leased:
            raise RuntimeError(f"staging buffer {sid} accessed while not leased")
        return self._view[sid]

    def same_buffer(self, sid: int, view) -> bool:
        """Storage identity plus containment -- the predicate the skeleton's contract test asks
        for, answered for real tensors instead of memoryviews."""
        base = self._view[sid]
        try:
            if view.untyped_storage().data_ptr() != base.untyped_storage().data_ptr():
                return False
            lo = base.data_ptr()
            return lo <= view.data_ptr() < lo + base.numel() * base.element_size()
        except Exception:
            return False

    def in_use(self, sid: int) -> bool:
        with self._lk:
            return sid in self._leased

    def at_rest(self) -> bool:
        with self._lk:
            return len(self.free) == self.n and self._sem._value == self.n and \
                len(set(self.free)) == self.n

    @property
    def sem_value(self) -> int:
        return self._sem._value

    def aligned(self, sid: int) -> bool:
        return self._view[sid].data_ptr() % ALIGN == 0


class RealLeaves(Leaves):
    """CB3 on NVMe for the I/O half. Graph A/B land in stage 4.

    `shared_first` stays False: engine/fastdecode.py captures the shared expert inside graph B, and
    claiming otherwise would be claiming a second capture that does not exist.
    """

    shared_first = False

    V1 = os.path.expanduser("~/git/deepseek-v41-flash-spark")

    @staticmethod
    def _v1_on_path() -> None:
        """v1's tools/ do SIBLING imports (`import fp4_moe`), so tools/ has to be on the path in
        its own right -- importing `tools.cb3_moe` as a package raises ModuleNotFoundError. The
        same trap as job 195's transition_prefetch."""
        for p in (RealLeaves.V1, os.path.join(RealLeaves.V1, "tools")):
            if p not in sys.path:
                sys.path.insert(0, p)

    def __init__(self, cb3_path: str, arena, device: str = "cuda"):
        self._v1_on_path()
        from engine.cb3_cache import CB3Cache
        self.cache = CB3Cache(cb3_path, device)
        self.arena = arena
        self.record = self.cache.record

    def close(self):
        self.cache.close()

    # --- I/O half ---------------------------------------------------------------------------
    def read(self, key: tuple, pool, ctx=None, scored: bool = True) -> StagedExpert:
        """One record, O_DIRECT, straight into the leased pinned buffer. Zero copy: the payload is
        the pool's own view, so the StagedExpert owns the lease until the H2D has completed."""
        sid = pool.acquire(ctx, scored) if ctx is not None else pool.acquire(scored=scored)
        try:
            view = pool.buffer(sid)
            layer, expert = key
            self.cache.read_into(memoryview(view.numpy()), layer, expert)
        except BaseException:
            pool.release(sid)
            raise
        return StagedExpert(sid, view, pool)

    def h2d(self, slot: int, key: tuple, staged: StagedExpert) -> None:
        """The record into the arena slot. Synchronous by contract -- the loader releases the
        pinned buffer the moment this returns, so the copy must have completed."""
        self.cache.load_slot(self.arena, slot, staged.payload, non_blocking=False)
        torch.cuda.synchronize()
