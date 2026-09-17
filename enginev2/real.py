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

_V41REF = None   # tools/v41_ref, imported once the engine path is set up

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
        # PER-REQUEST STATE, DECLARED HERE. attach() resets these on every request, but a caller
        # may legitimately touch them before the first attach -- V2Engine snapshots counters at the
        # top of generate() and prefills before it can know the first token, so it cannot attach in
        # spec mode any earlier. Job 560 died twice on exactly that: once on `accepted`, once on
        # `eng`. An earlier "fix" anchored on `self.accepted = []` and landed INSIDE attach(), which
        # is where the line it matched already was, so it changed nothing.
        self.eng = None
        self.fd = None
        # ENGRAM IS PROVIDER-LIFETIME, NOT REQUEST STATE. attach() used to null it, and V2Engine
        # installs the source at construction and then attaches -- so the source was wired, wiped,
        # and wiped again on every request, and layer_a's `if self.engram is not None: deliver(...)`
        # never ran. The driver still issued the reads and waited on them; begin_step then ZEROED
        # the rows, which is engram_ablate -- a different model, not a neutral default. Jobs 560 and
        # 565 decoded that way.
        self.engram = None
        self.engram_ablated = 0
        # Optional per-step verify trace. Set to a list and end_step appends one record per step:
        # everything the accept decision saw and everything it produced. A token-sequence
        # comparison cannot tell "the target computed something different" from "the same target
        # was committed differently" -- two autoregressive trajectories that resynchronise look
        # exactly like an insertion. This can.
        self.trace_verify = None
        self._tv_tok = None
        # Optional PER-LAYER trace of one decode step: h in, route, bound slots, h out. Set to a
        # list and layer_a/bind_slots/layer_b append to it. Checksums are computed ON DEVICE and
        # kept as 0-dim tensors -- no D2H inside the 40-layer loop, because a sync there would
        # change the host scheduling that is under diagnosis. Read them after the step.
        self.trace_layers = None
        self.accepted: list = []
        self.last_burst: list = []
        self.hist: list = []
        self.grammar = None
        self.penalties = None
        self.tok = None
        self.tokens_out = 0
        self._engine_on_path()
        global _V41REF
        if _V41REF is None:
            import v41_ref as _V41REF        # tools/ is on the path, sibling-import style
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
               temperature=0.0, top_p=1.0, seed=None, first_token=None,
               stop_ids=()) -> "RealLeaves":
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
        self.top_p = 1.0
        self.q = None            # the drafter's distribution for this step's drafts
        self.tok = None
        self.drafts = None
        self.tokens_out = 0          # tokens actually COMMITTED -- what tok/s means
        self.accepted = []           # per step, so accept_len is measured rather than assumed
        # The tokens THIS step committed, in order. end_step already computes them and used to
        # keep only the last one; a server has to yield the burst, not the survivor.
        self.last_burst: list = []
        # INITIALISED HERE TOO, not only in attach(). A caller may read the counters before the
        # first attach -- V2Engine snapshots them at the top of generate() -- and job 560 died on
        # exactly that: AttributeError before the engine had served one token.
        self.accepted: list = []
        # Optional decoding gate (server/tool_grammar.py). Two calls and nothing else: the driver
        # masks the verify block's rows here, the caller observes the committed tokens.
        self.grammar = None
        # Optional Penalties (engine/v41_engine.py). Same two-call shape as the gate, and applied at
        # the same point for the same reason. `hist` is the caller's running token list, which the
        # cycle breaker and the n-gram ban both read.
        self.penalties = None
        self.hist: list = []
        self.stop_ids = frozenset()
        # The engram source, if one is attached. It owns the reads; this owns the dequant+H2D at
        # the consumer, because to_device() makes CUDA calls and may not run on a reader thread.
        # engram is NOT reset here: it belongs to the provider, not to the request.
        self._S = int(self.fd.c.len)
        self.spec = bool(spec)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        # NO RESEED HERE. Seeding belongs at the start of a generation, before the first token is
        # sampled -- attach() runs after it. Reseeding here also restarted the RNG stream mid-
        # generation, which is not v1's semantics. V2Engine.generate() seeds.
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
            # KEEP q. It is the drafter's own distribution over each draft position, and
            # rejection sampling is defined against it -- throwing it away is what forced the
            # greedy-only verify below. v1 keeps it for exactly this reason.
            self.drafts, self.q = self.fd.draft(self.tok, int(self.fd.c.len) - 1, self.temperature)
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
        tl = self.trace_layers
        if tl is not None:
            _h = self.fd.h
            tl.append({"L": layer, "h_in": (_h.float().sum().detach().clone(),
                                            _h.float().abs().sum().detach().clone())})
        if self.graph_timing:
            self._gt_ev["A"][layer][0].record()
            self._gA[layer].replay()
            self._gt_ev["A"][layer][1].record()
            self._gt_seen["A"][layer] = True
        else:
            self._gA[layer].replay()
        idx = self.fd.route_idx
        if tl is not None:
            tl[-1]["route"] = idx.detach().clone()
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

    # ------------------------------------------------------------------ prefill
    # A TRANSCRIPTION of engine/model.py::encoder_prefill_layer_major, split at the point where it
    # resolves. That method is the shipped path (DSV41_LAYER_MAJOR=1, DSV41_SWA_REPLAY=1, neither
    # set in .env so both default on), and the split falls exactly where the driver needs it:
    # everything before `store.resolve` is per-chunk attention that PRODUCES the route, everything
    # after is per-chunk MoE that CONSUMES the slots. v1 keeps the between-state in locals of one
    # function; here the driver sits in the middle, so it lives in `self._pf`.
    #
    # The one deliberate difference: v1 resolves the layer ONCE over `torch.cat(idxs)`, this
    # resolves per chunk through the driver's reserve/submit. The ring still holds a layer's expert
    # set for the whole layer, so each expert is still read once per layer -- that is the property
    # the transpose exists for and it is preserved. Bitwise equality against v1 is the gate.

    def begin_prefill(self, ids, S0: int = 0):
        """Allocate the per-chunk state a layer-major pass carries. Call once per prompt."""
        m = self.eng.model
        a = m.args
        from engine.model import MAX_CHUNK, Shared
        P = ids.size(0)
        assert m.c.len == S0, (m.c.len, S0)
        bounds = [(lo, min(lo + MAX_CHUNK, P)) for lo in range(0, P, MAX_CHUNK)]
        hashes = [m.hash_state(ids[lo:hi][None], S0 + lo)[0] if m.hash_state is not None else None
                  for lo, hi in bounds]
        H, PM = [], []
        for lo, hi in bounds:
            h = m.W.embed[ids[lo:hi]].unsqueeze(1).repeat(1, a.hc_mult, 1)
            pm = torch.zeros(hi - lo, a.hc_mult, device=m.dev)
            pm[:, 0] = 1.0
            H.append(h); PM.append(pm)
        self._pf = dict(
            ids=ids, S0=S0, P=P, bounds=bounds, hashes=hashes, H=H, PM=PM,
            # One Shared per CHUNK, living ACROSS layers: a kv-source layer fills ckv/ik/ratio and
            # the layers above reuse it. A fresh one per (layer, chunk) trips _compressed's
            # `sh.ratio == r` assert at the first layer that reuses -- v1 records this trap.
            SH=[Shared() for _ in bounds],
            pend={S0 + lo: {} for lo, _ in bounds},
            tails=[None] * len(bounds),
            last_L=a.candidate_source_layer,
            n_experts=a.n_routed_experts,
            layer=None, ys={}, post={}, wts={}, idxs={},
        )
        return len(bounds)

    def prefill_attn(self, layer: int, chunk: int) -> RouteResult:
        pf = self._pf
        m = self.eng.model
        a = m.args
        if pf["layer"] != layer:                 # first chunk of a new layer
            pf["layer"] = layer
            pf["ys"].clear(); pf["post"].clear(); pf["wts"].clear(); pf["idxs"].clear()
        w = m.W.layers[layer]
        lo, hi = pf["bounds"][chunk]
        S0 = pf["S0"]
        h = pf["H"][chunk]
        if layer in m.W.engram:
            li = list(a.engram_layer_ids).index(layer)
            rows = m.engram_rows(layer, pf["hashes"][chunk][:, li, :])
            h = _V41REF.engram_forward(h, rows, m.W.engram[layer], a)
        # Prompt-cache checkpoint pieces. Layer-major never has the whole stack at one position, so
        # each layer's compressor state is collected as it crosses each boundary and assembled in
        # finish_prefill(). Without this the prompt cache is silently dead for the next turn.
        pv = m.c.pending.get(layer)
        pf["pend"][S0 + lo][layer] = None if pv is None else (pv[0].clone(), pv[1].clone())
        sh = pf["SH"][chunk]
        h, attn_pre = m.block_attn(h, pf["PM"][chunk], w, layer, S0 + lo, sh, m.c.win[layer],
                                   m.freqs_c if w.ratio else m.freqs_w)
        y, ffn_pre, ffn_post, ffn_comb = m.block_ffn_in(h, attn_pre, w)
        pf["PM"][chunk] = ffn_pre
        pf["ys"][chunk] = y
        pf["post"][chunk] = (h, ffn_post, ffn_comb)
        i_c, w_c = m.moe_route(y, w, layer, pf["n_experts"])
        pf["idxs"][chunk] = i_c
        pf["wts"][chunk] = w_c
        if layer == pf["last_L"]:
            n = min(a.window_size, hi - lo)
            pf["tails"][chunk] = (sh.topk[-n:],
                                  sh.candidates[-n:] if sh.candidates is not None else None)
        flat = i_c.flatten().tolist()
        return RouteResult(uniq=tuple(sorted(set(flat))), opaque=i_c, flat_cpu=flat)

    def prefill_moe(self, layer: int, chunk: int, route: RouteResult, slot_of: dict) -> None:
        pf = self._pf
        m = self.eng.model
        assert pf["layer"] == layer, (pf["layer"], layer)
        i_c = route.opaque
        # Shaped like the route, not like the de-duplicated slot list: moe_apply consumes one slot
        # per SELECTION, so order and multiplicity are functional. route.flat_cpu is the D2H this
        # provider already paid for in prefill_attn.
        slots = torch.as_tensor([slot_of[e] for e in route.flat_cpu],
                                dtype=torch.long, device=i_c.device).view_as(i_c)
        y = pf["ys"][chunk]
        out = m.moe_apply(y, slots, pf["wts"][chunk], m.W.layers[layer], self.arena)
        resid, ffn_post, ffn_comb = pf["post"][chunk]
        pf["H"][chunk] = _V41REF.hc_post(out, resid, ffn_post, ffn_comb)

    def finish_prefill(self) -> None:
        """Assemble the prompt-cache checkpoints, advance c.len, and seed the SWA replay buffer.

        v1 does all of this after its layer loop; it is not optional bookkeeping -- skipping the
        checkpoint assembly leaves a later turn with nothing to resume from.
        """
        pf = self._pf
        m = self.eng.model
        from engine.model import Shared
        for n, per_layer in pf["pend"].items():
            if len(per_layer) == pf["last_L"] + 1:      # every encoder layer passed this boundary
                m.c._ckpt[n] = dict(per_layer)
        m.c.len = pf["S0"] + pf["P"]
        m.stats["tokens"] += pf["P"]
        for ci, (lo, hi) in enumerate(pf["bounds"]):
            sh = Shared()
            sh.topk, sh.candidates = pf["tails"][ci]
            n = pf["tails"][ci][0].size(0)
            m._rep_keep(pf["H"][ci][-n:], pf["PM"][ci][-n:], sh, pf["S0"] + hi - n, n)

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
        tl = self.trace_layers
        if tl is not None and tl and tl[-1]["L"] == layer:
            tl[-1]["slots"] = self.fd.slots.detach().clone()
        if self.graph_timing:
            self._gt_ev["B"][layer][0].record()
            _g.replay()
            self._gt_ev["B"][layer][1].record()
            self._gt_seen["B"][layer] = True
        else:
            _g.replay()
        if tl is not None and tl and tl[-1]["L"] == layer:
            _h = self.fd.h
            tl[-1]["h_out"] = (_h.float().sum().detach().clone(),
                               _h.float().abs().sum().detach().clone())
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

    def _verify_sampled(self):
        """Rejection sampling over this step's drafts. Returns (n_accepted, new_tokens, bonus).

        One `sample_probs` per examined position, as v1 does -- the loop breaks at the first
        rejection, so it is at most n_draft calls and usually fewer. `sample_probs` is imported from
        the v1 engine rather than reimplemented: it is a pure function of logits and applies
        temperature and nucleus in a specific order, and a subtly different one here would change
        the output distribution without changing anything measurable.
        """
        from engine.v41_engine import sample_probs
        fd = self.fd
        lg, q, drafts = fd.logits, self.q, self.drafts
        n_draft = self._tv - 1
        a, new, bonus = 0, [], None
        for i in range(n_draft):
            pt = sample_probs(lg[i].float(), self.temperature, self.top_p)
            d = int(drafts[i])
            r = torch.rand((), device=pt.device)
            if bool(r < (pt[d] / q[i][d].clamp_min(1e-20)).clamp(max=1.0)):
                a += 1
                new.append(d)
                if d in self.stop_ids:
                    return a, new, None      # an accepted stop ends the block: no bonus token
            else:
                resid = (pt - q[i]).clamp_min(0)
                if float(resid.sum()) <= 0:
                    # q dominates p everywhere the draft could land. Falling back to p keeps the
                    # step productive instead of emitting nothing; v1 does the same.
                    resid = pt
                bonus = int(torch.multinomial(resid / resid.sum(), 1))
                return a, new, bonus
        # Every draft accepted and none of them a stop: the bonus comes from the row AFTER the last
        # accepted draft, which is why the verify block is one wider than the draft.
        pt = sample_probs(lg[a if a < n_draft else n_draft].float(), self.temperature, self.top_p)
        return a, new, int(torch.multinomial(pt, 1))

    def discard_tail(self, n: int) -> None:
        """Un-commit the last `n` tokens of the step just finished.

        `V2Engine` truncates the EMITTED burst at max_tokens, but the step has already committed the
        whole speculative burst -- so without this the engine's cache holds tokens the caller never
        received. Harmless while every request begins with `rollback(0)`, and a real defect the
        moment prompt-cache reuse lands: the resumed cache would carry text no client ever saw.

        THIS DOES NOT UNDO THE EXPERT READS. Those bytes moved, and the residency they perturbed
        stays perturbed. So this fixes cache correctness, NOT the request-to-request tail variation
        seen in job 585 -- do not credit it with that.

        One position per token, which holds because a step's cache advance equals its burst length
        on the path that can be truncated (max_tokens reached, so no stop token shortened it).
        """
        if n <= 0:
            return
        fd = self.fd
        fd.c.rollback(int(fd.c.len) - int(n))
        self._S = int(fd.c.len)

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
        # VERIFY. Two paths, and the split is v1's: greedy decoding keeps the lean one (one 7-wide
        # argmax D2H, no per-position sampling), temperature > 0 takes the SAMPLED one.
        #
        # The sampled path is not "greedy plus noise". Accepting a draft because it happens to be
        # the argmax would draw from a different distribution than the model's, and nothing
        # downstream could detect it -- tokens/s, acceptance length and even NLL all look ordinary
        # while the output distribution is quietly wrong. So it is rejection sampling against the
        # drafter's own q: accept draft d with probability min(1, p[d]/q[d]), and on rejection draw
        # the bonus from the residual (p - q)+ renormalised. That is the standard construction and
        # it is what makes speculative decoding distribution-preserving.
        # Reference: engine/v41_engine.py:1028-1051.
        # GRAMMAR MASK BEFORE THE DECISION, not after. The gate masks row i for the state after
        # block[0..i], so it has to see the verify block's rows while they are still logits -- once
        # a draft has been accepted the choice is made. Masking leaves the gate's own state
        # untouched, which is why speculation that is rolled back needs no undo.
        # PENALTIES FIRST, GRAMMAR LAST. Both write the same rows, and the gate's -inf is a
        # legality statement that a penalty must not be able to soften. v1 orders them the same way.
        if self.penalties is not None:
            self.penalties.apply(fd.logits, self.hist)
        if self.grammar is not None:
            self.grammar.mask_rows(fd.logits, [int(self.tok)] + self.drafts.tolist())
        self._tv_tok = self.tok
        if self.temperature > 0:
            a_n, new_toks, bonus = self._verify_sampled()
        else:
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
        self.last_burst = list(new_toks) + ([bonus] if bonus is not None else [])
        if self.trace_verify is not None:
            lg = fd.logits.float()
            top2 = lg.topk(2, dim=-1)
            self.trace_verify.append({
                "S": int(S),
                "tok": int(self._tv_tok),
                "drafts": self.drafts.tolist(),
                "argmax": lg.argmax(-1).tolist(),
                # top1 - top2 per row: a target that is nondeterministic in the last ulp can only
                # flip an argmax where this is tiny, so it separates a real disagreement from noise.
                "margin": [round(float(x), 6) for x in (top2.values[:, 0] - top2.values[:, 1])],
                "a_n": int(a_n),
                "bonus": bonus,
                "burst": list(self.last_burst),
                "c_len": int(fd.c.len),
            })
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
