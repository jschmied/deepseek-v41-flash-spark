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
        import queue
        self.n = n
        self.record = record
        self.nbytes = record + ALIGN
        # ONE ownership transfer per read. A semaphore AND a lock-guarded free list were doing a
        # single job: hand out a buffer id, take it back. A queue is that job. get() blocks exactly
        # as the semaphore did, and the id IS the permit, so the two can no longer disagree.
        # This is the one place where real ownership synchronisation remains -- a pinned buffer
        # cannot be reused until its H2D has read it -- so it keeps a real primitive, just one.
        self._free: "queue.SimpleQueue[int]" = queue.SimpleQueue()
        for i in range(n):
            self._free.put(i)
        self._out = 0                  # outstanding, for peak_in_use only; see the note below
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
        sid = self._free.get()         # blocks until a buffer is returned, as the semaphore did
        self._leased.add(sid)          # set.add/discard are atomic under the GIL
        self._out += 1                 # statistics only: a lost race costs one sample of a peak
        self.peak_in_use = max(self.peak_in_use, self._out)
        return sid

    def release(self, sid: int) -> None:
        if sid not in self._leased:
            raise RuntimeError(f"staging buffer {sid} released twice")
        self._leased.discard(sid)
        self._out -= 1
        self._free.put(sid)

    @property
    def free(self) -> int:
        """Count, not a list: the queue owns the ids now. Used only in assertion messages."""
        return self._free.qsize()

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
        return sid in self._leased

    def at_rest(self) -> bool:
        """Every buffer back, nothing leased. With one queue there is no second structure to
        disagree with the first -- the old form checked the free list, the semaphore count AND the
        list for duplicates, three things that could only diverge because they were three things."""
        return self._free.qsize() == self.n and not self._leased

    @property
    def sem_value(self) -> int:
        """Available buffers. Kept under the old name so the tests read the same quantity."""
        return self._free.qsize()

    def aligned(self, sid: int) -> bool:
        return self._view[sid].data_ptr() % ALIGN == 0


class RealLeaves(Leaves):
    """CB3 on NVMe for the I/O half. Graph A/B land in stage 4.

    `shared_first` FOLLOWS THE CAPTURE. engine/fastdecode.py puts the shared expert inside graph B
    unless DSV41_SHARED_FIRST=1, in which case it captures a third graph per layer that reads only
    `y`. Declaring shared_first without that capture would be claiming a fork that does not exist,
    which is why this mirrors the engine's own flag instead of being set independently.
    """

    shared_first = os.environ.get("DSV41_SHARED_FIRST") == "1"
    # DEVICE time per graph, which is what bounds the overlap. The host-phase profiler cannot answer
    # this: `layer_b` there is an ENQUEUE and reads ~2 ms, while `layer_a` is a wall span containing
    # graph A's own work AND the drain of the previous layer's graph B. Only graph B's work can be
    # moved into the read window -- A(L) must finish before the layer's expert ids exist -- so the
    # realisable overlap is bounded by B, not by total GPU time.
    graph_timing = os.environ.get("DSV41_GRAPH_TIMING") == "1"
    # MIRRORS engine/fastdecode.py's flag, for the same reason shared_first does: declaring the
    # split without the capture would claim graphs that do not exist.
    resident_first = os.environ.get("DSV41_RESIDENT_FIRST") == "1"
    # layer_b records a CUDA event and h2d waits on it per slot, so the DEVICE orders slot reuse
    # and the host-side ComputeStream is redundant here -- see Leaves.device_orders_slot_reuse.
    device_orders_slot_reuse = True

    # THIS checkout, not a second one. This used to be a hardcoded ~/git/deepseek-v41-flash-spark,
    # which is a different working tree of the SAME repo on a DIFFERENT branch -- so the flags below
    # mirrored an engine/fastdecode.py that this branch did not contain, and every real-graph number
    # it produced described code that was never committed here. `check_engine_capture` is the guard
    # that makes such a substitution impossible to make silently again.
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @staticmethod
    def _engine_on_path() -> None:
        """tools/ has to be on the path in its OWN right, because those modules do SIBLING imports
        (`import fp4_moe`) -- importing `tools.cb3_moe` as a package raises ModuleNotFoundError.
        The same trap as job 195's transition_prefetch."""
        for p in (RealLeaves.ROOT, os.path.join(RealLeaves.ROOT, "tools")):
            if p not in sys.path:
                sys.path.insert(0, p)

    # Back-compat alias: the name predates the fix above and is still called by tests.
    _v1_on_path = _engine_on_path

    @classmethod
    def check_engine_capture(cls, fd) -> str:
        """Refuse to run a flag whose graphs the attached engine cannot have captured.

        `shared_first` and `resident_first` MIRROR engine/fastdecode.py rather than being set
        independently, which is only safe while the engine on the path is the one this branch
        ships. It was not: enginev2/real.py bound fd.graphs_shared / fd.graphs_split /
        fd.slot_missing while this branch's own engine/fastdecode.py defined none of the three,
        and the substitution was invisible because both paths are called `engine`. Fail here,
        at attach, naming the file -- not 40 layers later inside a capture."""
        missing = [n for f, n in ((cls.shared_first, "graphs_shared"),
                                  (cls.resident_first, "graphs_split"),
                                  (cls.resident_first, "slot_missing")) if f and not hasattr(fd, n)]
        where = getattr(sys.modules.get(type(fd).__module__), "__file__", "?")
        if missing:
            raise RuntimeError(
                f"RealLeaves mirrors DSV41_SHARED_FIRST={int(cls.shared_first)} "
                f"DSV41_RESIDENT_FIRST={int(cls.resident_first)}, but the attached "
                f"{type(fd).__name__} from {where} has no {', '.join(missing)}. That engine cannot "
                f"have captured the graphs these flags claim. Check which checkout is on sys.path.")
        return where

    def __init__(self, cb3_path: str, arena, device: str = "cuda"):
        import threading
        # Per-worker counters. read_bytes/read_s are statistics, and a lock around them serialises
        # the I/O workers over something no decision reads. Each worker accumulates in its own
        # thread-local slot; the reporting properties sum them.
        self._stats_tls = threading.local()
        self._stats_all: list = []
        self._stats_lk = threading.Lock()     # taken ONCE per worker, at first use
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
        # ONE writer (layer_b), many readers (h2d). A slot cannot be targeted by a new H2D before
        # its reader event is stored: the driver protects the current layer's slots via
        # _layer_slots until graph B has been issued, and layer_b records the event as part of
        # issuing it. So there is no legal interleaving where a reader sees a stale entry for a
        # slot it is about to overwrite, and a plain list read under the GIL is enough.
        # _copy_event is the opposite case -- many writers, one drainer -- and keeps its lock.
        self._last_reader: list = []
        # Copies ENQUEUED but not yet complete: slot -> the event recorded after the H2D. The
        # driver drains these through await_copies() before the graph that reads those slots, so
        # completion becomes a GPU dependency instead of a host block. Written by loader threads
        # and drained by the driver, so it keeps a lock -- unlike _last_reader, which has exactly
        # one writer (layer_b) and is only read.
        self._copy_event: dict = {}
        self.copies_awaited = 0
        self._reader_lk = threading.Lock()
        self._bound_slots: frozenset = frozenset()
        self.reader_waits = 0
        self._engine_on_path()
        from engine.cb3_cache import CB3Cache
        self.cache = CB3Cache(cb3_path, device)
        self.arena = arena
        self._arena_slots = int(getattr(arena, "slots", 0) or 0)
        self.record = self.cache.record
        # PROVIDER counters, the counterpart of v1's ExpertStore.stats. The engine's own numbers,
        # not /proc/diskstats: a byte counted here is a byte this provider asked the device for.
        # read_bytes / read_s are now properties summing the per-worker counters below.
        # h2d timing is the LOADER's to measure now: the provider only enqueues, so a duration
        # taken here would be the enqueue cost, not the copy's.

    @property
    def read_bytes(self) -> int:
        return sum(c[0] for c in self._stats_all)

    @property
    def read_s(self) -> float:
        return sum(c[1] for c in self._stats_all)

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
            c = getattr(self._stats_tls, "c", None)
            if c is None:
                c = self._stats_tls.c = [0, 0.0]
                with self._stats_lk:
                    self._stats_all.append(c)
            c[0] += self.record
            c[1] += dt
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
        lr = self._last_reader
        ev_prev = lr[slot] if slot < len(lr) else None
        if ev_prev is not None:
            st.wait_event(ev_prev)
            self.reader_waits += 1
        with torch.cuda.stream(st):
            self.cache.load_slot(self.arena, slot, staged.payload, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(st)
        # PUBLISH, do not wait. The loader may now call this slot ready: the bytes are not there
        # yet, but the event that says when they will be is. await_copies() puts that dependency on
        # the compute stream before the graph reads the slot.
        with self._reader_lk:
            self._copy_event[slot] = ev
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

    def attach(self, engine, block_ids, engram_rows=None, next_block=None, spec=False,
               temperature=0.0, first_token=None, stop_ids=()) -> "RealLeaves":
        """Bind to a live V41Engine. The engine owns the weights, caches and captured graphs; this
        provider only replays them."""
        self.eng = engine
        self.fd = engine.fast
        # Startup provenance, printed once: which engine/fastdecode.py is actually bound.
        self._engine_file = self.check_engine_capture(self.fd)
        self.block_ids = block_ids
        self.engram_rows = engram_rows or {}
        # Multi-step decode needs a rule for the NEXT block. v1's is draft + verify; a harness that
        # wants both arms to see the same token sequence supplies its own, identical rule to both.
        # There is no default: a provider asked for more than one step without one is a bug, not a
        # silently-repeated block.
        self.next_block = next_block
        # SPECULATIVE MODE. attach(spec=True) drives the engine the way the server does -- DSpark
        # draft, verify, rollback -- instead of taking a caller-supplied block. Without it every
        # steps/s this branch produces is a proxy: the cache advances by the whole 6 tokens (100 %
        # acceptance) and nothing is ever committed, so there is no tokens/s to report.
        self.spec = False
        self.temperature = 0.0
        self.tok = None
        self.drafts = None
        self.tokens_out = 0          # tokens actually COMMITTED -- what tok/s means
        self.accepted = []           # per step, so accept_len is measured rather than assumed
        self.stop_ids = frozenset()
        # The engram source, if one is attached. It owns the reads; this owns the dequant+H2D at
        # the consumer, because to_device() makes CUDA calls and may not run on a reader thread.
        self.engram = None
        self.engram_ablated = 0
        self._S = int(self.fd.c.len)
        self.spec = bool(spec)
        self.temperature = float(temperature)
        self.stop_ids = frozenset(stop_ids)
        if self.spec:
            if first_token is None:
                raise RuntimeError("spec mode needs first_token: the drafter conditions on it")
            self.tok = int(first_token)
            self._tv = int(self.fd.ids.numel())
        self._gA = self._gB = self._gF = None
        self._gS = None
        self._gt_ev = None
        self._gt_seen = None
        self.gt_ms = {"A": 0.0, "B": 0.0, "S": 0.0, "B1": 0.0}
        self.gt_steps = 0
        return self

    def select_block(self, step: int) -> None:
        """This step's input block. Must run before EngramSource.issue() hashes it."""
        if self.spec:
            # v1's order exactly: draft first, then block = [accepted token | drafts]. The
            # drafter's graph replays here, so its ~12.8 ms and its three MTP layers are INSIDE
            # the measurement rather than omitted from it.
            self.drafts, _q = self.fd.draft(self.tok, int(self.fd.c.len) - 1, self.temperature)
            self.block_ids = torch.cat(
                [torch.tensor([self.tok], device=self.fd.dev), self.drafts])
            return
        if step:
            if self.next_block is None:
                raise RuntimeError("multi-step decode needs attach(next_block=...); without it "
                                   "every step would replay the same block")
            self.block_ids = self.next_block(self.fd.logits, step)

    def begin_step(self, step: int) -> None:
        """The prologue no layer owns. Mirrors fastdecode.step() up to the layer loop."""
        import torch as _t
        fd, a = self.fd, self.fd.a
        # READ THE POSITION FRESH. Verify rolls the cache back to pos + a + 1, so a cached _S is
        # already wrong by the next step under speculation.
        self._S = int(fd.c.len)
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
        self._gS = fd.graphs_shared.get(parity) if self.shared_first else None
        self._gB1 = self._gB2 = None
        if self.resident_first:
            _sp = fd.graphs_split.get(parity)
            if not _sp or not _sp[0]:
                raise RuntimeError(
                    "resident_first is on but engine/fastdecode.py captured no split graphs. "
                    "DSV41_RESIDENT_FIRST must be set for BOTH -- the engine reads it at import "
                    "time, so setting it later leaves gB as None and the MoE never runs.")
            self._gB1, self._gB2 = _sp
        if self.graph_timing and self._gt_ev is None:
            # Two events per graph per layer, reused every step. Recorded on the compute stream and
            # read ONCE per step in gt_drain(), so no synchronisation is added inside the step.
            n = len(self._gA)
            self._gt_ev = {k: [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                               for _ in range(n)] for k in ("A", "B", "S", "B1")}
            self._gt_seen = {k: [False] * n for k in ("A", "B", "S", "B1")}
        if self.shared_first and not self._gS:
            raise RuntimeError(
                "shared_first is on but engine/fastdecode.py captured no shared-expert graphs. "
                "DSV41_SHARED_FIRST must be set for BOTH -- the engine reads it at import time, so "
                "setting it after the module is loaded silently gives graph B the unsplit path "
                "while the driver skips the shared expert entirely, which is wrong output, not a "
                "slow one.")

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
        if self.graph_timing:
            self._gt_ev["A"][layer][0].record()
            self._gA[layer].replay()
            self._gt_ev["A"][layer][1].record()
            self._gt_seen["A"][layer] = True
        else:
            self._gA[layer].replay()
        idx = self.fd.route_idx
        flat = idx.flatten().tolist()          # the one unavoidable D2H: the cache lookup is on the host
        return RouteResult(uniq=tuple(sorted(set(flat))), opaque=idx, flat_cpu=flat)

    def gt_drain(self) -> None:
        """Read the step's graph timings. Called ONCE per step, after the step's work is complete.

        elapsed_time() requires the events to have completed, so this is a synchronisation point --
        which is why it exists only under DSV41_GRAPH_TIMING and why the numbers it produces are
        device time for the graphs, not a wall-clock budget for the step. Compare them against each
        other, never against a profiled step's wall.
        """
        if not self.graph_timing or self._gt_ev is None:
            return
        torch.cuda.synchronize()
        for k, evs in self._gt_ev.items():
            for i, (a, b) in enumerate(evs):
                if self._gt_seen[k][i]:
                    self.gt_ms[k] += a.elapsed_time(b)
                    self._gt_seen[k][i] = False
        self.gt_steps += 1

    def gt_report(self) -> str:
        if not self.graph_timing or not self.gt_steps:
            return ""
        n = self.gt_steps
        tot = sum(self.gt_ms.values())
        out = [f"  graph device time over {n} steps (per step):"]
        split = bool(self.gt_ms.get("B1"))
        for k, label in (("A", "graph A  attention+HC+router"),
                         ("B1", "graph B1 routed MoE, RESIDENT half (pre-wait)"),
                         ("B", "graph B2 routed MoE, missing half + rest" if split
                               else "graph B  routed MoE+shared+HC residual"),
                         ("S", "graph S  shared expert alone")):
            if self.gt_ms[k]:
                out.append(f"    {label:<40} {self.gt_ms[k] / n:7.2f} ms  {self.gt_ms[k] / tot * 100:5.1f}%")
        out.append(f"    {'TOTAL':<40} {tot / n:7.2f} ms")
        if split:
            out.append("    B1 and S run BEFORE the wait; A(L) never can, and B2 needs the bytes. "
                       "TOTAL includes B1 -- without it the split appears to delete device time it "
                       "only moved.")
        else:
            out.append("    Only graph B can move into the read window: A(L) must finish before "
                       "this layer's expert ids exist.")
        return "\n".join(out)

    def layer_b_resident(self, layer: int, missing_slots) -> None:
        """Phase 1 of the split MoE: the pairs whose experts are ALREADY RESIDENT.

        Called by the driver BEFORE it waits on this layer's reads -- that position is the whole
        point, and it is the lesson from shared_first, whose seam sat between the two wait branches
        and therefore overlapped nothing for three jobs running.

        `missing_slots` are the arena slots this layer is still waiting for. They go into the
        engine's static `slot_missing` flags, which the captured graph reads to mask its half of the
        shared routing; writing them here is the only host work the split adds per layer.
        """
        fd = self.fd
        fd.slot_missing.zero_()
        if missing_slots:
            idx = torch.as_tensor(list(missing_slots), dtype=torch.long, device=fd.slot_missing.device)
            fd.slot_missing[idx] = True
        # TIMED SEPARATELY, and it has to be. With the split, the events labelled "B" bracket gB2
        # alone -- the missing half -- so a run with DSV41_RESIDENT_FIRST=1 and
        # DSV41_GRAPH_TIMING=1 used to under-report device time by exactly the piece the split
        # moves, and gt_report presented the remainder as the total.
        if self.graph_timing:
            self._gt_ev["B1"][layer][0].record()
            self._gB1[layer].replay()
            self._gt_ev["B1"][layer][1].record()
            self._gt_seen["B1"][layer] = True
        else:
            self._gB1[layer].replay()

    def shared(self, layer: int) -> None:
        """Graph S: the shared expert, into fastdecode's sh_out buffer.

        Called by the driver AFTER the route is resolved and the reads are submitted, but BEFORE it
        waits on them -- which is the whole point: this is the only compute in the layer that does
        not depend on the expert bytes, so it is the only thing that can fill the wait.
        """
        if self.graph_timing:
            self._gt_ev["S"][layer][0].record()
            self._gS[layer].replay()
            self._gt_ev["S"][layer][1].record()
            self._gt_seen["S"][layer] = True
        else:
            self._gS[layer].replay()

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

    def await_copies(self, slots) -> int:
        """Put this layer's outstanding copies on the compute stream as a GPU dependency.

        Called by the driver after the loader reports the slots ready and before graph B reads
        them. Readiness now means "H2D enqueued, event published", so without this the graph could
        read a slot mid-copy -- which is why the loader only publishes early for a provider that
        declares device_orders_slot_reuse and therefore implements this.
        """
        n = 0
        cur = torch.cuda.current_stream()
        with self._reader_lk:
            evs = [self._copy_event.pop(s) for s in set(slots) if s in self._copy_event]
        for ev in evs:
            cur.wait_event(ev)
            n += 1
        self.copies_awaited += n
        return n

    def layer_b(self, layer: int, route: RouteResult) -> None:
        """Graph B: routed MoE + shared expert + HC residual, over slots already resident.

        replay() QUEUES the graph; it does not run it. So the arena slots it reads stay live on the
        device after this returns, and the only honest statement of "this slot is free again" is an
        event recorded after the replay on the same stream. h2d() waits on it before overwriting.
        """
        _g = self._gB2[layer] if self.resident_first else self._gB[layer]
        if self.graph_timing:
            self._gt_ev["B"][layer][0].record()
            _g.replay()
            self._gt_ev["B"][layer][1].record()
            self._gt_seen["B"][layer] = True
        else:
            _g.replay()
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        if len(self._last_reader) < self._arena_slots:
            self._last_reader = [None] * self._arena_slots
        for sl in self._bound_slots:
            self._last_reader[sl] = ev          # single writer; see __init__

    def step_other(self) -> None:
        """The final head graph."""
        if self._gF is not None:
            self._gF.replay()

    def end_step(self, step: int) -> None:
        """KV bookkeeping for Caches.rollback, and advance the cache length."""
        self.gt_drain()          # no-op unless DSV41_GRAPH_TIMING; the step's graphs are done here
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
        if not self.spec:
            return
        # VERIFY, as v41_engine does it: greedy accept over the leading drafts, one 7-wide D2H,
        # then roll the cache back to what was actually committed. This is where steps stop being
        # free -- a step commits a + 1 tokens, not T.
        am = fd.logits.argmax(-1)
        acc = am[:self._tv - 1].eq(self.drafts).to(torch.int32).cumprod(0)
        a_n = int(acc.sum())
        cand = am.tolist()
        new_toks, bonus = cand[:a_n], cand[a_n]
        for j, t in enumerate(new_toks):
            if t in self.stop_ids:
                a_n, new_toks, bonus = j + 1, new_toks[:j + 1], None
                break
        fd.c.rollback(S + a_n + 1)
        self._S = int(fd.c.len)
        self.accepted.append(a_n)
        self.tokens_out += a_n + (1 if bonus is not None else 0)
        self.tok = bonus if bonus is not None else (new_toks[-1] if new_toks else self.tok)


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
