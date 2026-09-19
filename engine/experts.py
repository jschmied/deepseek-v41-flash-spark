"""
experts.py -- the routed-expert store for one-box serving.

15,360 routed experts x 18.8 MB (FP4 + UE8M0 scales) = 288.8 GB do not fit next to everything
else, so the experts live in three places:

  * the ARENA: a fixed number of GPU slots holding packed FP4 experts exactly as stored in the
    checkpoint (no re-quantization). Sized at start-up from the memory that is left.
  * the LRU: a map (layer, expert) -> slot, least-recently-used eviction, warm-started from a
    routing trace so the hottest experts are resident before the first request.
  * NVMe: every expert is read straight out of its layer's safetensors shard with O_DIRECT
    preadv (no page-cache pollution, ~5.5 GB/s with 8+ reads in flight on this box), into a
    pinned staging buffer, then copied into its slot. Two runs per expert, not six: see
    `ShardFile.expert_runs`. Each run is split into `read_chunk_mb` aligned pieces issued in
    parallel on a second thread pool, because a decode layer misses only ~4 experts and two
    serial 18.8 MB reads cannot keep the device busy on their own (see NOTES "Speed work").

Prefill chunks touch almost every expert of a layer; letting them stream through the LRU would
evict the hot set each prompt. So misses during prefill go through a small TRANSIENT ring of
slots instead, and only decode misses enter the LRU.

WHICH resident the LRU gives up is a second, separate lever (DSV41_EVICT_POLICY). An offline
replay of a real decode route trace (~/ds41-queue/eviction_oracle.py) puts numbers on it, at the
shipped 5,328 LRU slots:

    lru            (default)  92.67 % hit   64.6 fetches/token   849 MiB/token
    age_over_freq             94.07 %       52.3                 687 MiB/token   (-19 % NVMe)
    Belady (not implementable) 97.17 %       25.0                 328

`age_over_freq` evicts the MAXIMUM of age / (1 + use count) -- recency discounted by how often the
expert has been wanted. It recovers 31.1 % of the LRU-to-Belady gap, is stable at 3,000 slots
(29.3 %) and across 10/25/50 % train splits, and is exact rather than sampled: see `_afq_victim`.
Plain LFU is NOT a substitute -- it also wins at small candidate sets but INVERTS to -10 % when
allowed to rank the whole cache, because an unweighted count protects stale hot entries forever.
The age numerator is what carries the policy.
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

ALIGN = 4096
# every counter reset between requests (engine/v41_engine.py::_reset) lives here
# Prefill I/O oracle recorder. Writes one JSON line per resolve() call so the re-read structure of
# a prompt can be replayed offline without the GPU. Never on by default.
_RL = os.environ.get("DSV41_ROUTE_LOG")
ROUTE_LOG = open(_RL, "w", buffering=1 << 16) if _RL else None

# DSV41_ROUTE_SYNC: synchronize BEFORE the resolve() timer starts, so the GPU wait lands in its own
# counter instead of inside route_s. Diagnostic only -- it adds a full-device sync per layer, which
# is exactly what the pipelined design wants to remove, so never leave it on for a timing number.
ROUTE_SYNC = os.environ.get("DSV41_ROUTE_SYNC", "0") == "1"

# DIAGNOSTIC ONLY, AND UNSAFE. Skips the `stream.wait_stream(compute)` that each loader thread does
# before overwriting a slot. That barrier exists because the previous layer's MoE kernel may still
# be reading the slot -- a PER-SLOT dependency, implemented as "wait for all compute". It was free
# under chunk-major, where resolve() blocked and nothing was on the compute stream; under
# EARLY_SUBMIT it makes every H2D wait for the attention it was supposed to overlap with, parks the
# worker, and stalls the read pool. This gate removes it to find out how much of the measured 1.03x
# is that barrier rather than real contention. It can corrupt an in-flight slot. Never for serving.
UNSAFE_NO_COMPUTE_WAIT = os.environ.get("DSV41_UNSAFE_NO_COMPUTE_WAIT", "0") == "1"

# DSV41_EVICT_POLICY: which LRU resident is given up when the LRU is full (see the module docstring
# for the replayed numbers). "lru" is the default and is the code path this file has always run --
# not a re-derivation of it, the same branches. "age_over_freq" is the only alternative. Read per
# store so a test can hold both policies in one process; the constructor argument wins.
EVICT_POLICIES = ("lru", "age_over_freq")

ZERO_STATS = {"hits": 0, "misses": 0, "prefill_misses": 0, "bytes_read": 0, "read_s": 0.0,
              "resolve_s": 0.0, "route_s": 0.0, "load_s": 0.0, "lease_s": 0.0, "h2d_s": 0.0,
              "sync_s": 0.0, "load_submit_s": 0.0, "load_wait_s": 0.0,
              "loads": 0, "promoted": 0,
              # evict_cmps / evictions = candidates examined per eviction. The whole point of the
              # bucket structure is that this stays ~O(distinct use counts) and not O(lru_slots);
              # if it ever approaches lru_slots the shortcut has degenerated. Both stay 0 under
              # the default policy, which never scores anything.
              "evictions": 0, "evict_cmps": 0}
W13_SHAPE = (2304, 2560)
S13_SHAPE = (2304, 160)
W2_SHAPE = (5120, 1152)
S2_SHAPE = (5120, 72)
NAMES = ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")
EXPERT_BYTES = 3 * (2304 * 2560 + 2304 * 160)


class ShardFile:
    """One safetensors shard: header spans + an O_DIRECT fd."""

    def __init__(self, path: str):
        self.path = path
        self._runs: dict[str, list] = {}
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        hdr.pop("__metadata__", None)
        self.base = 8 + n
        self.spans = {k: (self.base + v["data_offsets"][0], self.base + v["data_offsets"][1]) for k, v in hdr.items()}
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self.fd_buffered = os.open(path, os.O_RDONLY)

    def expert_span(self, prefix: str):
        """Byte span covering the 6 tensors of one expert."""
        s = [self.spans[prefix + n] for n in NAMES]
        lo, hi = min(a for a, _ in s), max(b for _, b in s)
        return lo, hi, s

    def expert_runs(self, prefix: str):
        """The 6 tensors of one expert grouped into maximal contiguous file ranges.

        These shards store ALL the scale tensors near the front and all the weight tensors far
        behind them, but within each group the three tensors of an expert are adjacent. So an
        expert is exactly two runs -- a 1.1 MB scale run and a 17.7 MB weight run -- not six, and
        not one. Reading it as six separate `preadv`s costs six O_DIRECT round trips; at a large
        arena a decode step misses only about one expert per layer, so nothing else is in flight to
        hide that latency and the read rate collapses from ~4.7 GB/s to ~0.6 GB/s.

        Returns [(file_lo, file_hi, [(name_index, offset_in_run, nbytes), ...]), ...].
        """
        cached = self._runs.get(prefix)
        if cached is not None:
            return cached
        spans = [self.spans[prefix + n] for n in NAMES]
        order = sorted(range(len(spans)), key=lambda i: spans[i][0])
        runs = []
        for i in order:
            a, b = spans[i]
            if runs and runs[-1][1] == a:
                lo, _, members = runs[-1]
                members.append((i, a - lo, b - a))
                runs[-1] = (lo, b, members)
            else:
                runs.append((a, b, [(i, 0, b - a)]))
        self._runs[prefix] = runs
        return runs


def _pread_chunk(fd: int, view: memoryview, off: int, need: int) -> None:
    """O_DIRECT-read `need` bytes at file offset `off` into `view` (an aligned slice of a pinned
    buffer). The request length is always the full aligned slice -- O_DIRECT rejects unaligned
    lengths -- and the loop stops as soon as the bytes that actually exist have arrived, which is
    what makes the aligned tail of the last tensor in a shard safe."""
    got = 0
    while got < need:
        r = os.preadv(fd, [view[got:]], off + got)
        if r <= 0:
            raise IOError(f"short read at {off}+{got}/{need}")
        got += r


class ExpertStore:
    def __init__(self, model_dir: str, index: dict, arena, n_layers: int, transient_slots: int = 400,
                 io_threads: int = 48, mtp_prefix: str | None = None, read_threads: int | None = None,
                 read_chunk_mb: float | None = None, evict_policy: str | None = None):
        self.model_dir = model_dir
        self.arena = arena  # tools.fp4_moe.ExpertArena or a compatible object with .slots and load_slot_bytes
        self.n_slots = arena.slots
        self.transient_slots = transient_slots
        self.lru_slots = self.n_slots - transient_slots
        assert self.lru_slots > 0
        # a prefill chunk can touch all 384 experts of a layer; a smaller ring is only safe when every routable
        # expert is resident (pruned all-resident mode). resolve() asserts on slot collisions either way.
        assert transient_slots >= 8, 'transient ring too small'
        self.shards: dict[str, ShardFile] = {}
        self.index = index["weight_map"]
        self.lru: OrderedDict[tuple, int] = OrderedDict()  # (layer, expert) -> slot
        self.slot_key: dict[int, tuple] = {}
        self.free_lru = list(range(self.lru_slots))
        self.transient_ring = list(range(self.lru_slots, self.n_slots))
        self.transient_index = {s: i for i, s in enumerate(self.transient_ring)}
        self.transient_pos = 0
        self.transient_map: dict[tuple, int] = {}
        # See lend_ring_to_lru(). Off unless DSV41_RING_TO_LRU=1, so the shipped default is
        # bit-identical to before this existed.
        self._ring_lending = os.environ.get("DSV41_RING_TO_LRU", "0") == "1"
        self._ring_lent = False
        self._pending: list = []          # (future, key, slot) from resolve(defer=True)
        self._pending_slots: set[int] = set()   # slots those futures are still writing
        self._pending_layer: int | None = None
        # --- eviction policy (see the module docstring). Everything below is dead weight under
        # "lru": _afq is False, nothing is ticked, counted or bucketed, and _lru_slot_for takes the
        # branch it has always taken.
        self.evict_policy = (evict_policy or os.environ.get("DSV41_EVICT_POLICY", "lru")).strip()
        if self.evict_policy not in EVICT_POLICIES:
            raise ValueError(f"DSV41_EVICT_POLICY={self.evict_policy!r} not in {EVICT_POLICIES}")
        self._afq = self.evict_policy == "age_over_freq"
        self._clock = 0                          # one tick per DECODE access, hit or miss
        # Both survive eviction on purpose, exactly as the offline replay's per-key stats do: an
        # expert that comes back from NVMe comes back with its history, and that is the only way a
        # use count means anything when 5,328 slots have to cover 15,360 pairs. Bounded by the pair
        # count, so ~15k small ints at worst.
        self._use_count: dict[tuple, int] = {}
        self._last_acc: dict[tuple, int] = {}
        # count -> {key: slot} in LRU order, RESIDENTS ONLY, empty buckets deleted. The victim
        # search reads only the head of each bucket; see _afq_victim.
        self._buckets: dict[int, OrderedDict] = {}
        io_threads = int(os.environ.get("DSV41_IO_THREADS", io_threads))
        # Optional native CB3 cache (engine/cb3_cache.py). When present a miss is one aligned
        # 13,774,848 B read whose bytes are already the arena's layout, instead of 18,800,640 B of
        # packed FP4 in two runs plus an fp4_to_cb3_v2 repack on the GPU.
        self.cb3_cache = None
        _cc = os.environ.get("DSV41_CB3_CACHE")
        if _cc:
            from .cb3_cache import CB3Cache
            self.cb3_cache = CB3Cache(_cc, arena.device if hasattr(arena, "device") else "cuda")
        if read_threads is None:
            read_threads = int(os.environ.get("DSV41_READ_THREADS", 96))
        if read_chunk_mb is None:
            read_chunk_mb = float(os.environ.get("DSV41_READ_CHUNK_MB", 4))
        self.io_threads = io_threads
        self.read_threads = read_threads
        self.read_chunk = int(read_chunk_mb * 1024 * 1024) // ALIGN * ALIGN
        # Two pools on purpose. `pool` runs one task per expert (it owns a pinned staging buffer for
        # the whole read + H2D); `read_pool` runs the individual aligned pieces of that expert's two
        # file runs. A single pool would deadlock as soon as every worker sat waiting for a piece
        # that has no worker left to run it.
        #
        # ON THE CB3 CACHE PATH `read_pool` AND `read_chunk` ARE DEAD. `_load_into_slot` dispatches to
        # `_load_into_slot_cached` whenever `cb3_cache is not None`, and that path issues ONE
        # `read_into` for the whole 13,774,848 B record -- the format exists precisely so a miss is one
        # contiguous extent. The piece splitting above belongs to `_read_leased()`, the FP4/safetensors
        # path, which the shipped configuration no longer uses. This comment used to describe the
        # splitting as the live mechanism, and that is what made job 965 look like a sensible
        # experiment: it swept DSV41_READ_THREADS and DSV41_READ_CHUNK_MB across four arms that were
        # all identical. Preflight now refuses that sweep while DSV41_CB3_CACHE is set.
        #
        # `io_threads` IS live on both paths, and it is not only NVMe concurrency: it sets the worker
        # count, the staging-buffer count, the worker-local copy streams and H2D concurrency together.
        # Job 970 measured 48 -> 2 as a median +9 % (3 of 3 paired), but that A/B cannot attribute the
        # win to read latency alone -- only job 945's raw O_DIRECT curve makes that the leading
        # explanation.
        self.pool = ThreadPoolExecutor(io_threads, thread_name_prefix="expert-io")
        self.read_pool = ThreadPoolExecutor(max(1, read_threads), thread_name_prefix="expert-read")
        self.lock = threading.Lock()
        # pinned, aligned staging buffers, one per io thread
        self.stage = [torch.empty(EXPERT_BYTES + 8 * ALIGN, dtype=torch.uint8, pin_memory=True) for _ in range(io_threads)]
        self.stage_mv = [memoryview(b.numpy()) for b in self.stage]
        self.stage_free = list(range(io_threads))
        self.stage_sem = threading.Semaphore(io_threads)
        self._tls = threading.local()
        self.n_experts = 384
        self.stats = dict(ZERO_STATS)

    # ------------------------------------------------------------------ io
    # ---------------------------------------------------------------- cold pool (DSV41_COLD_POOL)
    # Off unless a pool is attached. When it is, a DECODE miss is read by O_DIRECT straight into a
    # mapped record slot and computed there, and the hot slot it was promised becomes valid only when
    # the promotion copy lands. See notes/host-mapped-arena.md and engine/cold_promotion.py.
    cold = None

    def attach_cold_pool(self, pool) -> None:
        self.cold = pool
        self.cold_this_call: dict[int, int] = {}     # expert id -> COLD slot, for the current call
        self._cold_inflight: list = []               # (key, cold_slot, gen, event) awaiting landing
        self.stats.setdefault("cold_fetches", 0)
        self.stats.setdefault("cold_reuses", 0)
        self.stats.setdefault("cold_promotions", 0)
        self.stats.setdefault("cold_full", 0)
        # PROMOTION-INDUCED BLOCKING is the number this whole design lives or dies on: the copy is
        # allowed to cost bandwidth, it is not allowed to make the critical path wait. These name the
        # three ways it could.
        self.stats.setdefault("cold_reap_s", 0.0)        # time polling for landed promotions
        self.stats.setdefault("cold_reap_calls", 0)
        self.stats.setdefault("cold_inflight_max", 0)    # high-water of outstanding promotions
        self.stats.setdefault("cold_split_layers", 0)    # layers that ran two phases
        self.stats.setdefault("cold_promote_s", 0.0)     # host time issuing the copies
        self.stats.setdefault("cold_reuse_wait_s", 0.0)
        self.stats.setdefault("cold_read_fail", 0)
        # PER-REQUEST RESET. _reset_stats() zeroes ZERO_STATS only, so a counter that is not in it
        # accumulates across requests while APPEARING in per-request last_stats -- which would have
        # made a timing report attribute five prompts' cold traffic to each of them.
        for k in ("cold_fetches", "cold_reuses", "cold_promotions", "cold_full", "cold_split_layers",
                  "cold_inflight_max", "cold_reap_calls", "cold_read_fail"):
            ZERO_STATS.setdefault(k, 0)
        for k in ("cold_reap_s", "cold_promote_s", "cold_reuse_wait_s"):
            ZERO_STATS.setdefault(k, 0.0)

    def cold_reap(self) -> int:
        """Mark every promotion whose event has LANDED. Called at layer boundaries.

        Polling `Event.query()` rather than synchronising is the whole point: the copy is allowed to
        be in flight, it is only residency that must wait for it. An entry that has not landed stays
        in the list and its hot slot stays protected.
        """
        self.stats["cold_reap_calls"] += 1
        if not self._cold_inflight:
            return 0
        _t = time.perf_counter()
        keep, n = [], 0
        for key, slot, gen, ev, ev_c in self._cold_inflight:
            # BOTH events, and neither is optional. ev_c says the cold-phase kernel has finished
            # reading this slot; ev says the promotion copy has landed. The slot is released only when
            # the state machine has heard from both, which is what makes recycling it safe.
            if ev_c.query() and ev.query():
                self.cold.promo.compute_done(slot, gen)
                self.cold.promo.promo_done(slot, gen)
                self.cold._events.pop(key, None)
                self.stats["cold_promotions"] += 1
                n += 1
            else:
                keep.append((key, slot, gen, ev, ev_c))
        self._cold_inflight = keep
        self.stats["cold_reap_s"] += time.perf_counter() - _t
        return n

    def cold_finish_layer(self, hot_arena, stream=None) -> None:
        """The cold phase has been ENQUEUED; arrange promotion so that it cannot race it.

        ENQUEUED IS NOT EXECUTED, and getting that wrong here corrupted output (job 1085: prompt 3
        diverged at token 5, and misses moved 23,302 -> 23,046). The first version called
        `compute_done` on this line -- immediately after `moe_forward_cold_split` returned, which only
        queues kernels -- and issued the copy on a side stream that waited on nothing. So the
        promotion could land, release the cold slot, and a fresh O_DIRECT read could overwrite it while
        the previous cold-phase kernel was still reading it. That is precisely the hazard the two-party
        handshake exists to prevent, defeated by lying to it about one of the parties.

        So: one event recorded on the COMPUTE stream after the phase; the promotion stream waits on it
        before copying; and `compute_done` is marked only when that event has actually landed, polled
        in `cold_reap` alongside the copy's own event. Ordering is now enforced by the device, and the
        release still needs both parties.
        """
        if not self.cold or not self.cold_this_call:
            return
        _t = time.perf_counter()
        self.stats["cold_split_layers"] += 1
        # DSV41_COLD_SYNC=1 removes ALL asynchrony from the cold path: the cold phase is completed
        # and each promotion copy finished before the layer returns. A diagnostic, not a mode -- if a
        # divergence survives it, the fault is in the split or the residency logic, not in ordering.
        _sync = os.environ.get("DSV41_COLD_SYNC", "0") == "1"
        if _sync:
            torch.cuda.current_stream().synchronize()
        ev_compute = torch.cuda.Event()
        ev_compute.record(torch.cuda.current_stream())
        if stream is not None:
            # the copy may not begin until the kernel that reads these slots has finished
            stream.wait_event(ev_compute)
        for e, cslot in self.cold_this_call.items():
            key = (self._cold_layer, e)
            ent = self.cold.promo._inflight.get(key)
            if ent is None:                      # already landed and released
                continue
            slot, gen, _hot = ent
            _s, _g, ev = self.cold.promote(key, hot_arena, stream=stream)
            self._cold_inflight.append((key, _s, _g, ev, ev_compute))
            self.cold._events[key] = (_s, _g, ev, ev_compute)
        if _sync:
            if stream is not None:
                stream.synchronize()
            torch.cuda.synchronize()
            self.cold_reap()
        if len(self._cold_inflight) > self.stats["cold_inflight_max"]:
            self.stats["cold_inflight_max"] = len(self._cold_inflight)
        self.cold_this_call = {}
        self.stats["cold_promote_s"] += time.perf_counter() - _t

    def _shard(self, name: str) -> ShardFile:
        f = self.index[name]
        if f not in self.shards:
            self.shards[f] = ShardFile(os.path.join(self.model_dir, f))
        return self.shards[f]

    def _lease(self) -> int:
        self.stage_sem.acquire()
        with self.lock:
            return self.stage_free.pop()

    def _release(self, sid: int) -> None:
        with self.lock:
            self.stage_free.append(sid)
        self.stage_sem.release()

    def _read_leased(self, layer: int, expert: int, prefix: str | None, sink):
        """O_DIRECT-read one expert into a pinned staging buffer and call ``sink(views)`` while the
        buffer is still leased. ``views`` are 6 uint8 tensors that ALIAS the pinned buffer, so the
        sink must be done with them before it returns.

        The expert's two contiguous file runs (see `ShardFile.expert_runs`) are cut into
        `self.read_chunk`-sized aligned pieces and all but the first are handed to `read_pool`, so
        one expert alone keeps ~5 requests in flight. A decode layer misses ~4 experts; at two
        serial reads each the queue depth was ~4-8 and the device only gave ~2.6 GB/s of its
        5.5 GB/s, which is the whole reason decode was slower than its byte count implies.
        """
        p = prefix or f"layers.{layer}.ffn.experts.{expert}."
        sh = self._shard(p + "w1.weight")
        runs = sh.expert_runs(p)
        t_lease = time.perf_counter()
        sid = self._lease()
        self.stats["lease_s"] += time.perf_counter() - t_lease
        try:
            # inside the try: everything between _lease() and the try must be infallible or the
            # lease leaks, and there are only io_threads (48) of them for the life of the process --
            # a handful of leaks and every miss blocks forever in _lease() with the NVMe idle.
            buf = self.stage[sid]
            mv = self.stage_mv[sid]
            base_addr = buf.data_ptr()
            cur = (-base_addr) % ALIGN
            out = [None] * len(NAMES)
            jobs = []
            nbytes = 0
            t0 = time.perf_counter()
            for (a, b, members) in runs:
                alo = a - a % ALIGN
                ahi = (b + ALIGN - 1) // ALIGN * ALIGN
                n = ahi - alo
                step = self.read_chunk if 0 < self.read_chunk < n else n
                off = 0
                while off < n:
                    m = min(step, n - off)
                    need = min(b - alo - off, m)  # the aligned tail may run past EOF
                    if need > 0:
                        jobs.append((sh.fd, mv[cur + off: cur + off + m], alo + off, need))
                    off += m
                nbytes += n
                base = cur + (a - alo)
                for (i, o, nb) in members:
                    out[i] = buf[base + o: base + o + nb]
                cur += n
            futs = [self.read_pool.submit(_pread_chunk, *j) for j in jobs[1:]]
            _pread_chunk(*jobs[0])
            for f in futs:
                f.result()
            assert all(t is not None for t in out)
            self.stats["bytes_read"] += nbytes
            self.stats["read_s"] += time.perf_counter() - t0
            self.stats["loads"] += 1
            return sink(out)
        finally:
            self._release(sid)

    def read_expert(self, layer: int, expert: int, prefix: str | None = None):
        """The 6 tensors (CPU uint8) of one expert, copied out of the staging buffer."""
        return self._read_leased(layer, expert, prefix, lambda v: [t.clone() for t in v])

    def _copy_stream(self):
        """One CUDA stream per io thread.

        The arena copy must not run on the default stream: every worker would then have to
        synchronise the stream the model is computing on, once per miss. On its own stream a worker
        only has to (a) wait on the compute stream -- the previous layer's MoE kernel may still be
        reading the slot we are about to overwrite -- and (b) synchronise its own stream before
        releasing the pinned buffer.

        (a) is NOT "whatever was queued when the lease started", which is what this said until the
        deferred path existed and is worth being precise about, because the difference is the whole
        of the unexplained 14.5 %: `wait_stream` records its event where it is CALLED, and it is
        called in the sink, after the ~5 ms read. Under EARLY_SUBMIT the caller spends that read
        queueing the rest of the layer's attention onto the compute stream, so the H2D waits for
        attention that was queued AFTER the load was submitted -- precisely the work it was meant
        to overlap with. See DSV41_UNSAFE_NO_COMPUTE_WAIT at the top of this file.

        `compute` is `torch.cuda.current_stream()` read on an io WORKER thread, and PyTorch's
        current stream is thread-local: it is the device's default stream, which is where the model
        computes today (engine/fastdecode.py is the only other stream user and it joins back). A
        future change that moves compute onto a non-default stream would silently turn this barrier
        into a no-op against a stream nobody uses -- the slot corruption it guards would come back
        with no error and no flag flipped.
        """
        st = getattr(self._tls, "stream", None)
        if st is None:
            st = self._tls.stream = torch.cuda.Stream()
        return st

    def _load_into_slot(self, key: tuple, slot: int, prefix: str | None = None):
        """Read one expert straight from NVMe into its arena slot.

        The pinned staging buffer is handed to `arena.load_slot` directly instead of being cloned
        first: the clone was a second 18.8 MB CPU memcpy per expert AND it made the H2D copy run
        from pageable memory, which PyTorch has to stage through a bounce buffer of its own.
        """
        if self.cb3_cache is not None:
            return self._load_into_slot_cached(key, slot)
        stream = self._copy_stream()
        compute = torch.cuda.current_stream()  # capture OUTSIDE the `with`, where it is still ours

        def sink(v):
            t0 = time.perf_counter()
            w1, s1, w2, s2, w3, s3 = v
            # per-expert codebook width for a codebook arena (engine/codebook_sim.py): a key with an
            # entry in `cb_bits` is packed with that width's CodebookSim instead of the arena's own.
            kw = {}
            cb = getattr(self, "cb_bits", None)
            if cb is not None:
                b = cb.get(key)
                if b:
                    kw["sim"] = self.cb_sims[b]
            with torch.cuda.stream(stream):
                if not UNSAFE_NO_COMPUTE_WAIT:
                    stream.wait_stream(compute)
                self.arena.load_slot(slot, w1.view(*W13_SHAPE), s1.view(*S13_SHAPE), w2.view(*W2_SHAPE),
                                     s2.view(*S2_SHAPE), w3.view(*W13_SHAPE), s3.view(*S13_SHAPE),
                                     non_blocking=True, **kw)
                sim = getattr(self, "requant", None)  # simulated low-bit format (engine/codebook_sim.py)
                if sim is not None:
                    bits = sim.get(key)
                    if bits:
                        self.requant_sims[bits].requant_slot(self.arena, slot)
            stream.synchronize()  # the staging buffer is leased to another expert right after
            self.stats["h2d_s"] += time.perf_counter() - t0
            return slot

        return self._read_leased(key[0], key[1], prefix, sink)

    def _load_into_slot_cached(self, key: tuple, slot: int):
        """The same contract as `_load_into_slot`, reading the native CB3 cache instead.

        Keeps the staging lease and the copy stream exactly as the FP4 path does, so the only
        differences are the byte count (13,774,848 vs 18,800,640), one extent instead of two runs,
        and no `fp4_to_cb3_v2`: the record is already in the arena's layout and only the 3-bit
        scale planes are expanded, on the device.
        """
        c = self.cb3_cache
        stream = self._copy_stream()
        compute = torch.cuda.current_stream()
        t_lease = time.perf_counter()
        sid = self._lease()
        self.stats["lease_s"] += time.perf_counter() - t_lease
        try:
            buf = self.stage[sid]
            mv = self.stage_mv[sid]
            cur = (-buf.data_ptr()) % ALIGN
            t0 = time.perf_counter()
            c.read_into(mv[cur:cur + c.record], key[0], key[1])
            self.stats["bytes_read"] += c.record
            self.stats["read_s"] += time.perf_counter() - t0
            self.stats["loads"] += 1
            t1 = time.perf_counter()
            with torch.cuda.stream(stream):
                if not UNSAFE_NO_COMPUTE_WAIT:
                    stream.wait_stream(compute)
                c.load_slot(self.arena, slot, buf[cur:cur + c.record], non_blocking=True)
            stream.synchronize()
            self.stats["h2d_s"] += time.perf_counter() - t1
            return slot
        finally:
            self._release(sid)

    # ------------------------------------------------------------------ cache policy
    # The three helpers below are the whole of DSV41_EVICT_POLICY=age_over_freq. They are called
    # only when self._afq; under the default policy none of this state is ever written.

    def _afq_tick(self) -> int:
        """One clock tick per decode access, hit or miss, BEFORE the victim is scored.

        Age is measured in accesses, not tokens or wall time -- the same monotone counter the
        offline replay increments once per (layer, expert) access, so a resident's rank here is the
        rank the replay gave it. The tick has to happen before the scan because the score is
        (now - last_acc) / (1 + count) and a +1 on `now` is NOT uniform across buckets: it moves a
        count-1 entry by 1 and a count-9 entry by 0.1.
        """
        self._clock += 1
        return self._clock

    def _afq_touch(self, key: tuple, slot: int) -> None:
        """Record an access to `key`, now resident in `slot`: count +1, age 0, MRU of its bucket.

        EVERY key that enters self.lru must pass through here (both _lru_slot_for and
        _promote_transient do). A resident missing from the buckets would be invisible to the victim
        search and could never be evicted -- a slow leak of the arena, not an exception.
        """
        c = self._use_count.get(key, 0)
        if c:                                  # was it resident under its old count? (not after an
            b = self._buckets.get(c)           # eviction: the count survives, the bucket entry does not)
            if b is not None and b.pop(key, None) is not None and not b:
                del self._buckets[c]
        c += 1
        self._use_count[key] = c
        self._last_acc[key] = self._clock
        self._buckets.setdefault(c, OrderedDict())[key] = slot

    # ------------------------------------------------------------------ transient ring lending
    #
    # The ring exists for PREFILL: one chunk touches nearly every expert of a layer, and letting
    # that stream through the LRU would evict the decode working set. During DECODE it is idle --
    # 400 slots of a 4,565-slot arena doing nothing.
    #
    # Measured on the box (job 235, ARENA_GB 60/66/72 = 3,750/4,165/4,581 LRU slots, warm rep):
    # every +415 slots is -14.0 % NVMe traffic and +12 % decode tok/s, linear across both steps.
    # One ring is 400 slots, so this is worth about that.
    #
    # Shrinking the ring instead does NOT work and is not an alternative: transient_slots=8 starts
    # and then raises `transient ring exhausted` on the first prefill chunk (job 230).
    #
    # OFF BY DEFAULT. Set DSV41_RING_TO_LRU=1 to enable.

    def lend_ring_to_lru(self) -> int:
        """Prefill is over: hand the idle ring to the LRU region. -> slots lent."""
        if self._ring_lent or not self._ring_lending:
            return 0
        # Last prefill's transient mappings are dead -- decode never reads them, and leaving them
        # mapped would let a decode reserve() count one as a HIT on a slot the LRU is about to
        # reuse.
        for key, slot in list(self.transient_map.items()):
            if self.slot_key.get(slot) == key:
                del self.slot_key[slot]
        self.transient_map.clear()
        lent = [s for s in self.transient_ring if s not in self._pending_slots]
        self.free_lru.extend(lent)
        self._ring_lent = True
        return len(lent)

    def reclaim_ring(self) -> int:
        """Prefill is starting: take the ring back, evicting whatever decode put in it."""
        if not self._ring_lent:
            return 0
        ring = set(self.transient_ring)
        dropped = 0
        for key, slot in list(self.lru.items()):
            if slot in ring:
                del self.lru[key]
                if self.slot_key.get(slot) == key:
                    del self.slot_key[slot]
                if self._afq:
                    self._afq_drop(key)
                dropped += 1
        self.free_lru = [s for s in self.free_lru if s not in ring]
        self.transient_pos = 0
        self._ring_lent = False
        return dropped

    def _afq_drop(self, key: tuple) -> None:
        """`key` has left the LRU. Its count and last use stay; only the bucket entry goes."""
        c = self._use_count.get(key)
        if not c:
            return
        b = self._buckets.get(c)
        if b is not None and b.pop(key, None) is not None and not b:
            del self._buckets[c]

    def _afq_victim(self, used: set | frozenset, avoid: int = -1):
        """The resident maximising age / (1 + use count) -- EXACT, without scanning the LRU.

        Residents are bucketed by use count and each bucket is kept in LRU order. Inside a bucket
        the count is constant, so the score is monotone in age and the bucket's oldest entry IS that
        bucket's maximum; the global maximum is therefore the best of the bucket heads. Replaying
        the real decode trace through this store at the shipped 5,328 slots: 227.8 comparisons per
        eviction over 28,502 evictions, because the cache only ever holds ~400 distinct use counts.
        A brute-force argmax would be 5,328, and the offline replay checked the two against each
        other: hits 299,853 fetches 63,449, identical. It matters that this is exact and not a
        sample of the cold end -- the same score recovers only ~3-5 % of the Belady gap when it may
        rank the 32 or 64 coldest and 31 % when it ranks all of them.

        `used` (and `avoid`) are slots promised to another expert of the SAME resolve() call. They
        cannot be evicted, so an ineligible head is stepped over WITHIN its bucket rather than
        skipping the bucket: the next eligible entry is still that bucket's maximum among the
        entries we are allowed to take, which keeps the argmax exact under the constraint. It is
        also free in practice -- 0 skips in those 28,502 evictions, because `used` is ~14 slots that
        were all touched in this very call and are therefore the youngest things in the cache.

        Returns None when every resident is protected -- the caller decides what that means.
        """
        now = self._clock
        best = None
        best_score = 0.0
        best_acc = 0
        cmps = 0
        for c, b in self._buckets.items():
            for k, sl in b.items():             # LRU order: first ELIGIBLE entry is this bucket's max
                cmps += 1
                if sl in used or sl == avoid:
                    continue
                acc = self._last_acc[k]
                score = (now - acc) / (1.0 + c)
                # Tie-break on the older entry, the replay's (score, -last_acc) sort key. last_acc is
                # unique per resident (one tick per access), so the argmax never depends on dict order.
                if best is None or score > best_score or (score == best_score and acc < best_acc):
                    best, best_score, best_acc = k, score, acc
                break
        self.stats["evictions"] += 1
        self.stats["evict_cmps"] += cmps
        return best

    def _protect_pending(self, used: set | frozenset) -> set | frozenset:
        """`used` widened with the slots a deferred read is still landing in.

        `_transient_slot_for` has skipped `_pending_slots` since the prefill-side bug; the LRU half
        of the arena never did, and a deferred DECODE resolve is the same shape: the key goes into
        self.lru the moment the slot is assigned, 13.8 MB before the data is there, so a second
        deferred resolve inside the same layer is free to pick that slot as its eviction victim and
        overwrite a read in flight. It does not happen at the shipped 5,328 LRU slots -- a fresh
        entry is MRU under "lru" and scores age 0 (the minimum) under age_over_freq -- which is
        exactly the "invisible because the numbers happen to be large" that cost us the first one.
        Costs one set union per miss ONLY while something is pending; today nothing defers on the
        decode path, so it is a single empty-set test per miss.
        """
        pend = self._pending_slots
        if self.cold is not None:
            # A promotion destination holds no key yet, so nothing else in the eviction path keeps
            # its hands off it -- and being unmapped is exactly what makes it attractive to a victim
            # search. Handing it to another expert mid-copy would corrupt both.
            hot = self.cold.promo.pending_hot()
            if hot:
                pend = pend | hot
        if not pend:
            return used
        return set(used) | pend

    def _lru_slot_for(self, key: tuple, used: set | frozenset = frozenset()) -> int:
        """Reserve an LRU slot for `key` (evicting if needed). Caller loads it.
        `used` holds the slots already promised to other experts of the SAME resolve() call; they
        must never be evicted, or two experts would end up sharing one slot."""
        used = self._protect_pending(used)
        if self._afq:
            self._afq_tick()
        if self.free_lru:
            slot = self.free_lru.pop()
        elif self._afq:
            victim = self._afq_victim(used)
            if victim is None:                 # every resident is promised to this same call
                raise RuntimeError("LRU exhausted: more experts in one call than lru_slots")
            slot = self.lru.pop(victim)
            self._afq_drop(victim)
            self.slot_key.pop(slot, None)
        else:
            parked = []
            while True:
                if not self.lru:
                    raise RuntimeError("LRU exhausted: more experts in one call than lru_slots")
                old_key, slot = self.lru.popitem(last=False)
                if slot not in used:
                    self.slot_key.pop(slot, None)
                    break
                parked.append((old_key, slot))
            for k, s in reversed(parked):  # put the protected entries back, oldest first
                self.lru[k] = s
                self.lru.move_to_end(k, last=False)
        self.lru[key] = slot
        self.slot_key[slot] = key
        if self._afq:
            self._afq_touch(key, slot)
        return slot

    def _transient_slot_for(self, key: tuple, used: set | frozenset = frozenset()) -> int:
        """Next slot of the transient ring, skipping any slot already promised in this call.

        `_pending_slots` is skipped too, and that is what makes `resolve(defer=True)` safe. `used`
        is local to ONE resolve call, so a sequence of deferred resolves inside a layer each start
        with an empty `used` and would otherwise be free to recycle a slot whose read is still in
        flight -- silently, with no exception. Invisible at TRANSIENT_SLOTS=400 against a layer's
        ~362 distinct experts; immediate at the 8 that env.example documents.
        """
        n = len(self.transient_ring)
        for _ in range(n):
            slot = self.transient_ring[self.transient_pos % n]
            self.transient_pos += 1
            if slot not in used and slot not in self._pending_slots:
                break
        else:
            raise RuntimeError(
                f"transient ring exhausted: {len(used)} promised here + "
                f"{len(self._pending_slots)} still loading > transient_slots={self.transient_slots}. "
                f"Deferred submission needs the ring to hold a layer's distinct experts.")
        old = self.slot_key.pop(slot, None)
        if old is not None:
            self.transient_map.pop(old, None)
        prev = self.transient_map.get(key)
        if prev is not None and prev != slot:  # stale mapping from an earlier, recycled slot
            self.slot_key.pop(prev, None)
        self.transient_map[key] = slot
        self.slot_key[slot] = key
        return slot

    def _promote_transient(self, key: tuple, slot: int, used: set) -> bool:
        """Give a transient-ring slot to the LRU without re-reading its 18.8 MB.

        A prefill chunk loads almost every expert of a layer into the transient ring; when decode
        then routes to one of those the old code counted a hit, used the slot, and left it in the
        ring -- so the ring wrapped over it a few requests later and the expert was read again even
        though it was demonstrably hot at decode time. The ring is only a list of slot ids, so the
        fix is a pointer swap: this slot joins the LRU where it lies, and an LRU victim's slot takes
        its place in the ring.
        """
        i = self.transient_index.get(slot)
        if i is None:
            return False
        # the DONOR must not be a slot a deferred read is still writing: the swap hands it to the
        # transient ring, and the ring's own _pending_slots guard would then be looking at a slot
        # that is no longer pending-by-position. Nothing is corrupted today (the write still lands
        # in the same slot), but the read is wasted and the arena holds a key nobody maps.
        used = self._protect_pending(used)
        if self._afq:
            # A promotion is a decode access to a resident expert (a hit, in the replay's terms) AND
            # an eviction, because the donor slot the ring gets back has to come from the LRU. Tick
            # first, like the miss path: the donor is scored against this access.
            self._afq_tick()
        if self.free_lru:
            donor = self.free_lru.pop()
        elif self._afq:
            victim = self._afq_victim(used, avoid=slot)
            if victim is None:
                return False                   # no donor -> stay in the ring, exactly as before
            donor = self.lru.pop(victim)
            self._afq_drop(victim)
            self.slot_key.pop(donor, None)
        else:
            donor, parked = None, []
            while self.lru:
                k2, s2 = self.lru.popitem(last=False)
                if s2 not in used and s2 != slot:
                    self.slot_key.pop(s2, None)
                    donor = s2
                    break
                parked.append((k2, s2))
            for k2, s2 in reversed(parked):
                self.lru[k2] = s2
                self.lru.move_to_end(k2, last=False)
            if donor is None:
                return False
        self.transient_ring[i] = donor
        self.transient_index.pop(slot, None)
        self.transient_index[donor] = i
        self.transient_map.pop(key, None)
        self.lru[key] = slot
        self.slot_key[slot] = key
        if self._afq:
            # The pointer swap moves SLOTS between the two pools, never keys, so the buckets only
            # ever hear about `key` joining the LRU in `slot`. The donor left the LRU above.
            self._afq_touch(key, slot)
        self.stats["promoted"] += 1
        return True

    def resolve(self, layer: int, experts: torch.Tensor, prefill: bool,
                defer: bool = False) -> torch.Tensor:
        """experts: int tensor [T, K] of expert ids for `layer`. Returns the slot ids [T, K],
        loading misses (in parallel) first.

        `defer=True` returns as soon as the slots are assigned and leaves the reads in flight, for a
        caller that has other work to do first -- measured: after chunk 0 of a layer has routed,
        85.4 % of that layer's entire expert set is already known, and there is ~2.4x more remaining
        attention than the reads need. The caller owns `join_pending()`.
        """
        if ROUTE_SYNC:
            # The first statement below is a BLOCKING .to("cpu"), so without this the device wait is
            # charged to route_s and reads as host bookkeeping. Measured offline, the actual host
            # work in this function is 0.010 ms per call; route_s is 12.7 ms per layer. This moves
            # the difference somewhere it can be named.
            _t = time.perf_counter()
            torch.cuda.synchronize()
            self.stats["sync_s"] += time.perf_counter() - _t
        t_res = time.perf_counter()
        # One device->host copy, and the set/LUT work in numpy on the host. The old path ran
        # torch.unique on the GPU, synchronised on .tolist(), built a 384-entry LUT, copied that
        # back up and gathered it there: two extra launches and a second sync per layer, 40 layers
        # per token, for 36 numbers.
        ex = experts.to("cpu", dtype=torch.int32, non_blocking=False).numpy()
        uniq = np.unique(ex)
        if self.cold is not None:
            # Retire any promotion that has LANDED before this layer decides residency, so an expert
            # promoted during the previous layer counts as an ordinary resident here rather than
            # being fetched again.
            self.cold_reap()
            self.cold_this_call = {}
            self._cold_layer = layer
        slot_of = {}
        to_load = []
        cold_jobs = []      # (key, expert, hot_slot, cold_slot, gen) -- reads submitted together
        used: set[int] = set()  # slots already promised in this call -- never recycle one of them
        # pass 1: residents. Reserving them before any allocation is what keeps a later miss from
        # running the transient ring over a slot an earlier hit is already using (which used to
        # give two experts the same slot: the second load overwrote the first expert's weights and
        # the duplicate index in moe_forward's `y[t] +=` dropped one contribution).
        for e in uniq.tolist():
            key = (layer, e)
            if self.cold is not None:
                ent = self.cold.promo._inflight.get(key)
                if ent is not None:
                    # An expert wanted again while its promotion is outstanding. The first version
                    # read it from the cold slot a SECOND time, which the lifecycle cannot represent:
                    # it models one compute party and one promotion party, so a second cold consumer
                    # would issue a second promotion for the same (slot, gen) and the first pair's
                    # completion could release the slot while the second kernel still read it.
                    # Instead: make the compute stream WAIT for the promotion, then use the hot slot
                    # as an ordinary resident. Routing is uniqued so this cannot happen inside one
                    # layer; a reuse is ~a model traversal later, by which time the event is usually
                    # already complete and the wait is free.
                    _cslot, _gen, hslot = ent
                    evs = self.cold.event_of(key)
                    if evs is not None:
                        torch.cuda.current_stream().wait_event(evs[2])
                        _t0 = time.perf_counter()
                        self.stats["cold_reuse_wait_s"] += time.perf_counter() - _t0
                    else:
                        # promotion not issued yet (same layer is impossible, but be explicit):
                        # fall through and read it cold, which is safe because no promotion exists.
                        self.cold_this_call[e] = _cslot
                    slot_of[e] = hslot
                    used.add(hslot)
                    self.stats["hits"] += 1
                    self.stats["cold_reuses"] += 1
                    continue
            s = self.lru.get(key)
            if s is None:
                s = self.transient_map.get(key)
                if s is not None and not prefill:
                    self._promote_transient(key, s, used)
            else:
                self.lru.move_to_end(key)
                if self._afq and not prefill:
                    # DECODE hits only. A prefill chunk touches almost every expert of a layer, so
                    # letting it write the use counter would push nearly every resident up by one
                    # per chunk and drown the decode signal the policy was fitted on -- the same
                    # reason prefill misses go to the transient ring instead of the LRU. The
                    # move_to_end above is unchanged either way; it is the key->slot map's order.
                    self._afq_tick()
                    self._afq_touch(key, s)
            if s is not None:
                slot_of[e] = s
                used.add(s)
                self.stats["hits"] += 1
        # pass 2: misses
        for e in uniq.tolist():
            if e in slot_of:
                continue
            key = (layer, e)
            if prefill:
                self.stats["prefill_misses"] += 1
                s = self._transient_slot_for(key, used)
            else:
                self.stats["misses"] += 1
                s = self._lru_slot_for(key, used)
                if self.cold is not None:
                    from engine.cold_promotion import ColdSlotBusy
                    try:
                        cslot, cgen = self.cold.reserve(key, s)
                    except ColdSlotBusy:
                        # The pool running dry is a real condition -- a layer can want up to 36
                        # distinct experts -- and it must degrade to the ordinary path, not fail.
                        self.stats["cold_full"] += 1
                    else:
                        # RESERVE HERE, READ LATER. Doing the blocking preadv inline serialised the
                        # cold reads: the ordinary path submits to self.pool and gets io_threads of
                        # concurrency, and reading per miss on the main thread threw that away -- three
                        # misses became ~3 x 2.4 ms instead of overlapping. The reads are submitted
                        # together below.
                        cold_jobs.append((key, e, s, cslot, cgen))
                        slot_of[e] = s
                        used.add(s)
                        continue
            slot_of[e] = s
            used.add(s)
            to_load.append((key, s))
        assert len(set(slot_of.values())) == len(slot_of), "slot collision in resolve()"
        if ROUTE_LOG is not None:
            # One line per resolve() call: which experts this (layer, chunk) wanted, and which of
            # them were not resident. That is everything the prefill I/O oracles need -- the
            # re-read is `uniq` seen again in a later chunk of the SAME layer, and the floor is
            # the union of `miss` over a layer. Off unless DSV41_ROUTE_LOG is set; the cost is a
            # few hundred short lines for a whole prompt.
            self._route_i = getattr(self, "_route_i", 0) + 1
            ROUTE_LOG.write(json.dumps({"i": self._route_i, "L": layer, "pf": int(prefill),
                                        "uniq": uniq.tolist(),
                                        "miss": sorted(e for e, _ in ((k[1], v) for k, v in to_load))})
                            + "\n")
        if cold_jobs:
            # Same pool, same concurrency the ordinary path gets. Failures release their own cold slot
            # (abort_before_compute) and must also un-map the hot slot, which _lru_slot_for published
            # before the read was attempted.
            t_cold = time.perf_counter()
            futs = [(self.pool.submit(self.cold.read_into, k, cs, g), k, e, hs, cs)
                    for (k, e, hs, cs, g) in cold_jobs]
            for fu, k, e, hs, cs in futs:
                try:
                    fu.result()
                except Exception as exc:                      # noqa: BLE001
                    self._forget(k, hs)
                    slot_of.pop(e, None)
                    used.discard(hs)
                    self.stats["cold_read_fail"] += 1
                    raise IOError(f"cold read failed for {k}: {exc!r}") from exc
                self.cold_this_call[e] = cs
                self.stats["cold_fetches"] += 1
                # The cold path reads exactly the record the ordinary path reads; leaving it out of
                # bytes_read made nvme_gb fall 168.9 -> 27.6 GB and looked like a 6x win.
                self.stats["bytes_read"] += self.cold.record_bytes
                self.stats["loads"] = self.stats.get("loads", 0) + 1
            # charged as LOAD time, not route time: it is NVMe, and route_s is host bookkeeping
            self.stats["load_s"] += time.perf_counter() - t_cold
            self.stats["read_s"] += self.cold.stats["read_s"] - getattr(self, "_cold_read_s0", 0.0)
            self._cold_read_s0 = self.cold.stats["read_s"]
        lut = np.full(self.n_experts, -1, dtype=np.int32)
        for e, s in slot_of.items():
            lut[e] = s
        slots = torch.from_numpy(lut[ex.astype(np.intp)]).to(experts.device)
        self.stats["route_s"] += time.perf_counter() - t_res
        if to_load:
            t0 = time.perf_counter()
            if defer:
                # Hand the futures back instead of collapsing them. `slots` is ALREADY correct --
                # assignment happened synchronously in passes 1 and 2 above and `used` guarantees
                # nothing else claims those slots -- so the only thing the wait provides is data
                # readiness. The caller must `join_pending()` before the next resolve() on this
                # store: the transient ring is ~400 slots and one layer wants up to ~384, so a
                # second resolve could hand out a slot this one is still writing.
                assert self._pending_layer in (None, layer), (
                    f"deferred resolves span layers {self._pending_layer} and {layer}; "
                    f"join_pending() must be called at a layer boundary")
                self._pending_layer = layer
                self._pending_slots.update(sl for _, sl in to_load)
                # (future, key, slot): join_pending() needs the key to un-map an expert whose read
                # raised, and the slot to hand back
                self._pending += [(self.pool.submit(self._load_into_slot, k, sl), k, sl)
                                  for k, sl in to_load]
                self.stats["load_submit_s"] += time.perf_counter() - t0
            else:
                list(self.pool.map(lambda ks: self._load_into_slot(*ks), to_load))
            self.stats["load_s"] += time.perf_counter() - t0
        self.stats["resolve_s"] += time.perf_counter() - t_res
        return slots

    def join_pending(self):
        """Block until every read submitted with `defer=True` has landed.

        Each io worker already `stream.synchronize()`s its own copy stream before releasing its
        pinned staging lease, so a resolved future IS a completion guarantee -- no CUDA events are
        needed for this to be safe.

        EVERY future is waited on even when one raises, and the pending bookkeeping is cleared
        before the first error is re-raised. The plain `for f: f.result()` this replaced returned
        through the first exception with `_pending`/`_pending_slots`/`_pending_layer` still
        populated: those slots were then blocked forever (the ring lost them permanently) and the
        rest of the layer's reads were still in flight, writing into an arena the caller believed
        was quiesced. The visible symptom was a LATER, unrelated request dying with "transient ring
        exhausted" or "deferred resolves span layers" -- the wrong error, in the wrong place, one
        request after the real one.

        A failed expert is also un-mapped (`_forget`). Its slot holds a torn read, and leaving the
        key in self.lru / transient_map would make the next resolve() count it as a HIT and compute
        with whatever partial bytes landed -- silent, and permanent for the life of the process.
        """
        if not self._pending:
            return
        t0 = time.perf_counter()
        pending, self._pending = self._pending, []
        err = None
        for f, key, slot in pending:
            try:
                f.result()
            except BaseException as e:      # noqa: BLE001 -- drain the rest, then re-raise the first
                if err is None:
                    err = e
                self._forget(key, slot)
        self._pending_slots.clear()
        self._pending_layer = None
        # NOT load_s. With deferred submission the overlapped part of a load disappears from every
        # counter, so load_s would silently mean different things depending on `defer`. This is the
        # residual the overlap failed to hide -- the number the scheduler is judged on.
        self.stats["load_wait_s"] += time.perf_counter() - t0
        if err is not None:
            raise err

    def _forget(self, key: tuple, slot: int) -> None:
        """Un-map `key` from `slot` after a load that did not complete.

        The slot is left holding a torn read, so it must not stay reachable as a hit. An LRU slot
        goes back to free_lru (it belongs to neither pool otherwise -- a promoted transient slot has
        already left transient_index, so free_lru is its only home); a transient-ring slot is left
        in the ring, where the next wrap overwrites it anyway.
        """
        if self.lru.get(key) == slot:
            del self.lru[key]
            self.free_lru.append(slot)
            if self._afq:
                self._afq_drop(key)
        if self.transient_map.get(key) == slot:
            del self.transient_map[key]
        if self.slot_key.get(slot) == key:
            del self.slot_key[slot]

    def warm_start(self, ranked_keys: list[tuple], log=print):
        """Fill the LRU with `ranked_keys` (most important first) up to capacity."""
        keys = [k for k in ranked_keys[: self.lru_slots]]
        t0 = time.time()
        jobs = []
        for k in keys:
            jobs.append((k, self._lru_slot_for(k)))
        done = 0
        for _ in self.pool.map(lambda ks: self._load_into_slot(*ks), jobs):
            done += 1
            if done % 500 == 0:
                log(f"warm start {done}/{len(jobs)} experts, {self.stats['bytes_read'] / 1e9:.1f} GB, {time.time() - t0:.0f}s")
        per_slot = getattr(self.arena, "bytes_per_slot", EXPERT_BYTES)
        log(f"warm start done: {len(jobs)} experts resident ({len(jobs) * per_slot / 1e9:.1f} GB, "
            f"{self.stats['bytes_read'] / 1e9:.1f} GB read) in {time.time() - t0:.0f}s")

    def hit_rate(self):
        h, m = self.stats["hits"], self.stats["misses"]
        return h / max(1, h + m)


def category_counts(trace_stats_json: str, profile: str, n_experts: int = 384,
                    n_layers: int = 40) -> dict[int, np.ndarray]:
    """Per-layer expert histogram restricted to one corpus category.

    `coverage.json` only carries the mixed histogram (`counts`) plus the two coverage *curves*, so a
    workload-specific hot set has to be recomputed from the raw traces the stats were made from:
    `results/<name>/trace/layer<L>.npz` with `indices` [tokens, 6] and `category` [tokens].
    """
    # Preferred source: the per-category histogram written into coverage.json itself, so a plain
    # checkout can build a keep-set without the raw trace arrays next to it.
    try:
        d = json.load(open(trace_stats_json))
        pl = d.get("per_layer") or {}
        got = {int(L): np.asarray(v[f"counts_{profile}"], dtype=np.float64)
               for L, v in pl.items() if f"counts_{profile}" in v}
        if len(got) >= n_layers:
            return got
    except Exception:  # noqa: BLE001 - fall through to the raw arrays
        pass
    trace_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(trace_stats_json))), "trace")
    out: dict[int, np.ndarray] = {}
    if not os.path.isdir(trace_dir):
        return out
    import glob
    import re
    for path in glob.glob(os.path.join(trace_dir, "layer*.npz")):
        L = int(re.search(r"layer(\d+)", os.path.basename(path)).group(1))
        z = np.load(path)
        idx, cat = z["indices"], z["category"]
        sel = idx[cat.astype("U") == profile]
        if sel.size == 0:
            continue
        out[L] = np.bincount(sel.reshape(-1).astype(np.int64), minlength=n_experts).astype(np.float64)
    return out


def rank_from_trace(trace_stats_json: str, n_layers: int = 40, fallback_uniform: bool = True,
                    profile: str = "mixed") -> list[tuple]:
    """(layer, expert) ranked by frequency from tools/expert_stats.py coverage.json. Layers not in the
    trace get their experts appended in a round-robin so every layer has some residents.

    `profile` picks which slice of the traced corpus ranks the experts: "mixed" (the whole corpus,
    the default and what the coverage.json histogram is), "coding" or "general". The coding and
    general top-25% sets overlap by only 0.18-0.31 Jaccard, so the profile is a real lever on the
    hit rate of a workload that is all one kind.
    """
    ranked = []
    counts = {}
    if profile and profile != "mixed":
        counts = category_counts(trace_stats_json, profile)
    if not counts:
        try:
            d = json.load(open(trace_stats_json))
            for L, v in d["per_layer"].items():
                counts[int(L)] = np.array(v["counts"], dtype=np.float64)
        except Exception:  # noqa: BLE001
            pass
    known = sorted(counts)
    if known:
        # normalize per layer so a layer with more traced tokens is not favoured
        keys = []
        for L in known:
            c = counts[L] / counts[L].sum()
            keys += [(float(c[e]), L, e) for e in range(384)]
        keys.sort(reverse=True)
        ranked = [(L, e) for _, L, e in keys]
    missing = [L for L in range(n_layers) if L not in counts]
    if missing and fallback_uniform:
        # untraced layers: interleave a uniform share so the LRU can learn them
        share = [(L, e) for e in range(384) for L in missing]
        # interleave: after every traced key, one untraced key
        out = []
        it = iter(share)
        for k in ranked:
            out.append(k)
            try:
                out.append(next(it))
            except StopIteration:
                pass
        out += list(it)
        ranked = out
    return ranked
