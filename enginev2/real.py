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

Stages 1-4: the I/O half plus the graph half. Graph A and graph B are replayed from
engine/fastdecode.py's captures; the DRIVER owns the resolve between them, which is the whole point
-- v1's step() calls gA, _resolve, gB in one loop, and v2 splits the resolve out so a scheduler can
sit there.
"""

from __future__ import annotations

import os
import sys
import time

import torch

from .leaves import Leaves, RouteResult, StagedExpert

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
        import threading
        self._ctr = threading.Lock()
        self._tls = threading.local()
        # PER-SLOT READER LIFETIME, on the device. ComputeStream.run(slots) models a slot as being
        # read for the lifetime of the Python call, which is true for a modelled leaf and FALSE
        # here: layer_b replays a graph and returns while the GPU is still reading the arena. So
        # the host-side wait_slot_free() sees readers[] already at zero and D1-off would let a copy
        # overwrite a slot the current graph is mid-read of. One event per LAYER, shared by every
        # slot that layer bound -- thousands of events are not needed and not wanted.
        #
        # Initialised HERE, not in attach(): h2d() reads it, and the I/O half of this provider is
        # used without a graph half at all (test_real_leaves drives read/h2d directly). Putting it
        # in attach() made that path raise AttributeError -- caught by that test, 2026-09-16.
        self._last_reader: dict = {}
        self._reader_lk = threading.Lock()
        self._bound_slots: frozenset = frozenset()
        self.reader_waits = 0
        self._v1_on_path()
        from engine.cb3_cache import CB3Cache
        self.cache = CB3Cache(cb3_path, device)
        self.arena = arena
        self.record = self.cache.record
        # PROVIDER counters, the counterpart of v1's ExpertStore.stats. The engine's own numbers,
        # not /proc/diskstats: a byte counted here is a byte this provider asked the device for.
        self.read_bytes = 0
        self.read_s = 0.0
        # h2d timing is the LOADER's to measure now: the provider only enqueues, so a duration
        # taken here would be the enqueue cost, not the copy's.

    def close(self):
        self.cache.close()

    def make_staging(self, n: int, observer=None) -> "PinnedStagingPool":
        """Real pinned buffers, one record wide. n x 13.1 MiB of page-locked host memory, which is
        why the count is worth choosing rather than inheriting the skeleton's 48."""
        return PinnedStagingPool(n, self.record)

    # --- I/O half ---------------------------------------------------------------------------
    def read(self, key: tuple, pool, ctx=None, scored: bool = True) -> StagedExpert:
        """One record, O_DIRECT, straight into the leased pinned buffer. Zero copy: the payload is
        the pool's own view, so the StagedExpert owns the lease until the H2D has completed."""
        sid = pool.acquire(ctx, scored) if ctx is not None else pool.acquire(scored=scored)
        try:
            view = pool.buffer(sid)
            layer, expert = key
            t0 = time.perf_counter()
            self.cache.read_into(memoryview(view.numpy()), layer, expert)
            dt = time.perf_counter() - t0
            with self._ctr:
                self.read_bytes += self.record
                self.read_s += dt
        except BaseException:
            pool.release(sid)
            raise
        return StagedExpert(sid, view, pool)

    def h2d(self, slot: int, key: tuple, staged: StagedExpert):
        """ENQUEUE the record into the arena slot and hand back a device event.

        The pinned buffer may not be reused until the copy has read it, but that is a BUFFER
        LIFETIME dependency, not a compute one, and blocking the calling worker on it parks a
        thread that could be issuing the next NVMe read. So this returns a cuda.Event and the
        loader's completer releases the lease when the device says the bytes have landed.

        Two earlier revisions of this method were both barriers wearing a copy's clothes:
        non_blocking=False plus torch.cuda.synchronize() (device-wide, taken from a loader thread,
        so compute waited on every H2D and every H2D waited on compute), then non_blocking=True
        plus stream.synchronize() (narrower, still a host block on this worker). Neither is a
        dependency of the model.
        """
        # PER-THREAD COPY STREAM, which is what v1 does (experts.py `_load_into_slot`): issue the
        # twelve plane copies async on a stream this worker owns, then wait on THAT stream. The
        # first revision used non_blocking=False plus torch.cuda.synchronize(), a device-wide
        # barrier taken from a loader thread -- so every H2D also waited for the model's compute
        # and, worse, made compute wait for it. That is not a copy cost, it is a barrier cost, and
        # it was being reported as H2D time.
        st = getattr(self._tls, "stream", None)
        if st is None:
            st = self._tls.stream = torch.cuda.Stream()
            self._tls.events = []
        # THE REAL per-slot write-after-read edge. Not wait_stream(compute), which would take
        # everything queued on the compute stream up to NOW -- including work queued after this
        # read began, which is the defect v1 carries (experts.py records its event inside the sink,
        # after the ~5 ms read). This waits for ONE event: the last graph that actually read this
        # slot.
        with self._reader_lk:
            ev_prev = self._last_reader.get(slot)
        if ev_prev is not None:
            st.wait_event(ev_prev)
            self.reader_waits += 1
        with torch.cuda.stream(st):
            self.cache.load_slot(self.arena, slot, staged.payload, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(st)
        return ev


    # --- graph half ---------------------------------------------------------------------------
    #
    # v1's step() is: prologue, then per layer { gA[L].replay(); _resolve(L); gB[L].replay() },
    # then gF, then the KV epilogue. v2 keeps exactly that order and takes over only the middle --
    # _resolve becomes the driver's reserve/submit/wait, which is what lets a scheduler exist there
    # at all.
    #
    # The graphs are captured PER PARITY (S % 2) because the ratio-2 compressor grouping depends on
    # it, so begin_step selects the pair for this step and captures on first sight.

    def attach(self, engine, block_ids, engram_rows=None, next_block=None) -> "RealLeaves":
        """Bind to a live V41Engine. The engine owns the weights, caches and captured graphs; this
        provider only replays them."""
        self.eng = engine
        self.fd = engine.fast
        self.block_ids = block_ids
        self.engram_rows = engram_rows or {}
        # Multi-step decode needs a rule for the NEXT block. v1's is draft + verify; a harness that
        # wants both arms to see the same token sequence supplies its own, identical rule to both.
        # There is no default: a provider asked for more than one step without one is a bug, not a
        # silently-repeated block.
        self.next_block = next_block
        # The engram source, if one is attached. It owns the reads; this owns the dequant+H2D at
        # the consumer, because to_device() makes CUDA calls and may not run on a reader thread.
        self.engram = None
        self.engram_ablated = 0
        self._S = int(self.fd.c.len)
        self._gA = self._gB = self._gF = None
        return self

    def select_block(self, step: int) -> None:
        """This step's input block. Must run before EngramSource.issue() hashes it."""
        if step:
            if self.next_block is None:
                raise RuntimeError("multi-step decode needs attach(next_block=...); without it "
                                   "every step would replay the same block")
            self.block_ids = self.next_block(self.fd.logits, step)

    def begin_step(self, step: int) -> None:
        """The prologue no layer owns. Mirrors fastdecode.step() up to the layer loop."""
        import torch as _t
        fd, a = self.fd, self.fd.a
        S = self._S
        fd.ids.copy_(self.block_ids)
        fd.pos.copy_(S + _t.arange(fd.ids.numel(), device=fd.dev))
        # ENGRAM. With no source attached this ZEROES the rows, which is engram_ablate -- a
        # different model, not a neutral default. It was the silent state of every v2 measurement
        # before a real source existed, so it is now named and counted rather than implied.
        if self.engram is None:
            for L in fd.eg_rows:
                rows = self.engram_rows.get(L)
                if rows is None:
                    fd.eg_rows[L].zero_()
                    self.engram_ablated += 1
                else:
                    fd.eg_rows[L].copy_(rows)
        fd.h.copy_(fd.W.embed[fd.ids].unsqueeze(1).expand(-1, a.hc_mult, -1))
        fd.pre_mix.copy_(fd._premix0)
        parity = S % 2
        fd.prepare_pending_buffers()
        if parity not in fd.graphs:
            fd.capture(parity)
            fd.prepare_pending_buffers()   # capture's warm-up overwrites the buffers
        self._gA, self._gB, self._gF = fd.graphs[parity][0], fd.graphs[parity][1], fd.graphs[parity][2]

    def layer_a(self, layer: int) -> RouteResult:
        """Graph A: attention + HC + router. The expert ids come OUT of it.

        `opaque` is the router's own index tensor -- bind_slots needs its SHAPE and ORDER, not just
        the unique ids, because moe_fn consumes a slot tensor shaped like route_idx.
        """
        # EDGE: graph A reads eg_rows[L]. The raw NVMe read was waited on by the driver's
        # chain.wait("engram", L); the dequantize and H2D are CUDA work and belong here, on the
        # consumer, which is exactly where v1 does them (`eg_rows[L].copy_(finish(*fut.result()))`
        # inside its layer loop).
        if self.engram is not None:
            self.engram.deliver(layer, self.fd)
        self._gA[layer].replay()
        idx = self.fd.route_idx
        flat = idx.flatten().tolist()          # the one unavoidable D2H: the cache lookup is on the host
        return RouteResult(uniq=tuple(sorted(set(flat))), opaque=idx, flat_cpu=flat)

    def bind_slots(self, route: RouteResult, slot_of: dict) -> None:
        """Give graph B its route-aligned slot tensor. This is v1's `self.slots.copy_(slots)`, with
        the mapping coming from v2's ExpertSlots instead of v1's store."""
        import torch as _t
        idx = route.opaque
        # Reuse the host copy layer_a already paid for; a second .tolist() is a second D2H, and
        # host synchronisation between layers is exactly what leaves the NVMe pipe empty.
        flat = route.flat_cpu if route.flat_cpu is not None else idx.flatten().tolist()
        self.fd.slots.copy_(_t.tensor([slot_of[e] for e in flat], dtype=self.fd.slots.dtype,
                                      device=self.fd.slots.device).view_as(self.fd.slots))
        self._bound_slots = frozenset(slot_of.values())

    def layer_b(self, layer: int, route: RouteResult) -> None:
        """Graph B: routed MoE + shared expert + HC residual, over slots already resident.

        replay() QUEUES the graph; it does not run it. So the arena slots it reads stay live on the
        device after this returns, and the only honest statement of "this slot is free again" is an
        event recorded after the replay on the same stream. h2d() waits on it before overwriting.
        """
        self._gB[layer].replay()
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        with self._reader_lk:
            for sl in self._bound_slots:
                self._last_reader[sl] = ev

    def step_other(self) -> None:
        """The final head graph."""
        if self._gF is not None:
            self._gF.replay()

    def end_step(self, step: int) -> None:
        """KV bookkeeping for Caches.rollback, and advance the cache length."""
        fd = self.fd
        S, T = self._S, fd.ids.numel()
        for L in fd.kvl_buf:
            before = fd.c.pending.get(L)
            fd.c._chunk_inputs[L] = (S, fd.kvl_buf[L], fd.sc_buf[L], before)
            if S % 2 == 1:
                fd.c.pending[L] = (fd.kvl_buf[L][T - 1].clone(), fd.sc_buf[L][T - 1].clone())
            else:
                fd.c.pending[L] = None
        fd.c.len = S + T
        self._S = fd.c.len


class RealEngramSource:
    """The SECOND NVMe stream, for real: engine/engram.py's two tables.

    The skeleton's EngramSource completes immediately and exists only so the edge is real. This one
    does what v1 does per step, in v1's order:

        hashes = hash_state(block, pos)      GPU
        h_np   = hashes.cpu().numpy()        D2H HERE, before any graph is queued -- a .cpu() later
                                             would wait for the whole step
        futs   = {L: eg_pool.submit(tables[L].read_raw, h_np[:, li, :])}   host-only, thread-safe
        ...                                  each layer's rows dequantized and copied at its graph A

    THE SPLIT IS NOT COSMETIC. `read_raw` is host-only and safe on a pool thread; `to_device`
    dequantizes and copies and therefore makes CUDA calls, so it runs on the consumer. The source
    signals `chain.set("engram", L)` when the RAW READ lands, and `deliver()` does the CUDA half at
    graph A. Signalling after to_device instead would put CUDA work on a reader thread and hide the
    edge inside it.

    Only `engram_layer_ids` have rows. Every other layer is signalled immediately, because the edge
    must still exist for it -- a driver that waits on "engram" for layer 7 must not hang.

    ~144 lookups per layer per step, 264 B each, deduped against a process cache: ~0.2 % of expert
    byte traffic but a large number of tiny reads, so it competes for IOPS and host CPU rather than
    for bandwidth. Whether that matters is measurable now and was not before.
    """

    name = "real"

    def __init__(self, engine, leaves):
        self.eng = engine
        self.leaves = leaves
        self.tables = engine.tables
        self.pool = engine.eg_pool
        self.hash_state = engine.model.hash_state
        self.layer_ids = tuple(engine.args.engram_layer_ids)
        self._futs: dict = {}
        self.rows_read = 0
        self.steps = 0

    def issue(self, layers, step: int, chain, ctx=None, scored: bool = True) -> None:
        import numpy as np  # noqa: F401
        fd = self.leaves.fd
        block = self.leaves.block_ids
        pos = int(self.leaves._S)
        hashes = self.hash_state(block[None], pos)[0]
        h_np = hashes.cpu().numpy()                       # D2H before any graph is queued
        self._futs = {}
        for li, L in enumerate(self.layer_ids):
            fut = self.pool.submit(self.tables[L].read_raw, h_np[:, li, :])
            self._futs[L] = fut
            # The edge is signalled from a callback so the DRIVER only ever waits, and a layer
            # whose rows are already cached is released without a round trip.
            fut.add_done_callback(
                lambda _f, _L=L: chain.set("engram", _L, ctx=ctx, scored=scored))
        for L in layers:
            if L not in self._futs:
                # no rows for this layer; the edge is still real, and still carries the cohort --
                # an edge set during settlement with scored defaulted to True is a settlement
                # event wearing a measured event's label
                chain.set("engram", L, ctx=ctx, scored=scored)
        self.steps += 1

    def deliver(self, layer: int, fd) -> None:
        """The CUDA half, at the consumer. Raises if the read failed rather than feeding graph A
        stale rows -- a silent stale row is a quality loss with no symptom."""
        fut = self._futs.get(layer)
        if fut is None:
            return
        raw, inv, shape = fut.result()
        fd.eg_rows[layer].copy_(self.tables[layer].to_device(raw, inv, shape))
        self.rows_read += int(shape[0]) if hasattr(shape, "__getitem__") else 0
