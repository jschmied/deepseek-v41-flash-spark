"""drivers.py -- request input -> prefill (chunked) and decode (stepped), over the replayed router.

The decode driver is the one the profile measured, so it is the one the dependency table is built
from. The prefill driver exists because D3 (`global_barrier`) is STRUCTURALLY UNREACHABLE at decode
and only a chunked driver can exercise it -- see the note on `decode_step`.
"""

from __future__ import annotations

import dataclasses
import time

from .leaves import Bandwidth, ModelLeaves
from .prefetch import PrefetchStats, Prefetcher
from .sched import ComputeStream, LoaderService, Policy
from .store import ExpertSlots, SlotArena
from .trace import N_LAYERS


@dataclasses.dataclass
class Counters:
    steps: int = 0
    wall_s: float = 0.0
    blocked_s: float = 0.0        # driver time inside a wait for expert data
    compute_s: float = 0.0        # driver time inside a modelled compute leaf
    fetches: int = 0
    mean_inflight: float = 0.0
    achieved_gbs: float = 0.0
    device_busy_s: float = 0.0

    @property
    def steps_per_s(self) -> float:
        return self.steps / self.wall_s if self.wall_s else 0.0


class Engine:
    def __init__(self, policy: Policy, evict: str = "lru", lru_slots: int = 5328,
                 transient_slots: int = 400, n_workers: int = 48, staging: int = 48,
                 nvme_qd: int = 8, h2d_inflight: int = 8, scale: float = 1.0,
                 leaves=None, calls=(), prefetch=None, discard_wrong_asap: bool = True):
        self.policy = policy
        self.slots = ExpertSlots(lru_slots, transient_slots, policy=evict)
        self.arena = SlotArena(self.slots.n_slots)
        self.compute = ComputeStream()
        # THE SEAM. Everything the engine actually does -- router, MoE, NVMe read, H2D -- is behind
        # this one object. The default is the modelled provider; a real one implements the same four
        # methods against engine/fastdecode.py's two graphs and engine/experts.py's `_read_leased`.
        self.leaves = leaves if leaves is not None else ModelLeaves(
            calls, Bandwidth(scale=scale), scale=scale)
        self.loader = LoaderService(self.arena, self.slots, self.compute, policy,
                                    leaves=self.leaves, n_workers=n_workers, staging=staging,
                                    nvme_qd=nvme_qd, h2d_inflight=h2d_inflight, scale=scale)
        self.scale = scale
        # THE PREFETCH SEAM. The default predicts nothing, which is the honest baseline: both arms
        # run this identical driver and differ only in what predict() returns. Nothing is granted
        # to either side.
        self.prefetch = prefetch if prefetch is not None else Prefetcher()
        self.discard_wrong_asap = discard_wrong_asap
        self.pf = PrefetchStats()
        self._spec: dict[tuple, tuple] = {}     # key -> (slot, gen) still speculative, not yet used
        self.c = Counters()

    def close(self):
        self.loader.shutdown()

    # ------------------------------------------------------------------ compute helpers
    def _compute(self, fn, slots=()):
        """Run one compute leaf inside the compute stream. `slots` are the arena slots it READS --
        that is what lets the loader honour a per-slot ordering instead of a global barrier."""
        t0 = time.perf_counter()
        with self.compute.run(slots):
            r = fn()
        self.c.compute_s += time.perf_counter() - t0
        return r

    def _wait(self, to_load):
        t0 = time.perf_counter()
        try:
            if self.policy.global_barrier:
                self.loader.wait_all()
            self.loader.wait_slots(to_load)
        finally:
            self.c.blocked_s += time.perf_counter() - t0

    # ------------------------------------------------------------------ decode
    def decode_layer(self, layer: int) -> None:
        """One layer of one decode step.

        NOTE ON D3, recorded because it changes what the table can say. At decode the driver cannot
        reach layer L+1's router until layer L's MoE has produced its output, so the pending set
        NEVER holds more than the current layer's reads. `global_barrier` and per-slot waiting are
        therefore the SAME WAIT at decode, and D3's decode contribution is structurally zero rather
        than measured-small. It is exercised in prefill_chunked(), where several chunks of one layer
        are in flight together.
        """
        # Graph A. The expert ids come OUT of it -- they are not handed to the driver. This is the
        # line that makes this an engine and not a replay of someone else's route.
        uniq = self._compute(lambda: self.leaves.layer_a(layer))

        # What did speculation actually buy, and what did it cost? Settled BEFORE reserve(), while
        # the previous prediction for this layer is still distinguishable from a real residency.
        self._settle_speculation(layer, uniq)

        slot_of, to_load, to_wait = self.slots.reserve(layer, uniq, prefill=False)
        self.c.fetches += len(to_load)
        self.loader.submit(to_load)
        # A prefetch that has not landed yet is waited on exactly like a demand read. It is not
        # counted as a fetch -- it was already counted when it was issued.
        to_load = to_load + to_wait

        # Speculation is queued AFTER this layer's demand reads, at lower priority, so a demand
        # miss never sits behind a prefetch for a layer we have not reached.
        self._issue_speculation(layer, uniq)

        if self.policy.resolve_blocks:
            # v1: resolve() ends in list(pool.map(...)). Nothing issues work except a call that
            # immediately blocks on it, which is why the device cannot be kept busy at any thread
            # count. Everything after this point runs with the loader already drained.
            self._wait(to_load)

        # The shared expert is expert-INdependent, so a provider that captured it outside graph B
        # can run it here, while the reads fly. One that did not has it inside layer_b, where it
        # cannot overlap anything -- see leaves.Leaves.shared_first.
        if self.leaves.shared_first:
            self._compute(lambda: self.leaves.shared(layer))

        if not self.policy.resolve_blocks:
            self._wait(to_load)

        # Graph B: routed MoE + HC residual (+ the shared expert, unless it ran above).
        reads = sorted(set(slot_of.values()))
        self._compute(lambda: self.leaves.layer_b(layer, reads), slots=reads)

    # ------------------------------------------------------------------ prefetch bookkeeping
    def _settle_speculation(self, layer: int, uniq) -> None:
        """Score the predictions that were made for THIS layer, then drop the ones that missed."""
        want = {(layer, e) for e in uniq}
        wrong = []
        for key in [k for k in self._spec if k[0] == layer]:
            slot, gen = self._spec.pop(key)
            if key in want:
                self.pf.used += 1               # resident when demanded: the prediction paid
            else:
                self.pf.wasted += 1
                wrong.append((key, slot, gen))
        if wrong and self.discard_wrong_asap:
            # A wrong prefetch holds a slot AND a pending write, so it blocks eviction as well as
            # occupying capacity. Cancelling recovers the slot for reads that are already known to
            # be needed. Reads already in flight are not interrupted -- that cost is real and stays.
            self.pf.cancelled += self.loader.cancel(wrong)

    def _issue_speculation(self, layer: int, uniq) -> None:
        self.prefetch.observe(layer, uniq, self.c.steps)
        keys = self.prefetch.predict(layer, uniq, self.c.steps)
        if not keys:
            return
        keys = [k for k in keys if k not in self._spec]
        to_load, refused = self.slots.reserve_speculative(keys)
        self.pf.refused += refused
        if not to_load:
            return
        for key, slot, gen in to_load:
            self._spec[key] = (slot, gen)
        self.pf.issued += len(to_load)
        self.loader.submit(to_load, speculative=True)

    def decode(self, steps: int) -> Counters:
        """Run `steps` decode steps. Where the expert ids come from is the provider's business."""
        t0 = time.perf_counter()
        for _ in range(steps):
            for layer in range(N_LAYERS):
                self.decode_layer(layer)
            self.c.steps += 1
        self.c.wall_s = time.perf_counter() - t0
        bw = self.loader.bw
        if bw is not None:                      # a real provider models no device; leave at 0.0
            self.c.mean_inflight = bw.mean_inflight
            self.c.achieved_gbs = bw.achieved_gbs
            self.c.device_busy_s = bw.busy_s
        return self.c

    def warm(self, calls, upto: int) -> None:
        """Bring the cache to the state the scored window starts in, with no I/O and no timing."""
        for layer, uniq in calls[:upto]:
            _, to_load, _ = self.slots.reserve(layer, uniq, prefill=False)
            for key, slot, gen in to_load:
                self.arena.content[slot] = key
                self.loader.ready.set(slot, gen)

    # ------------------------------------------------------------------ prefill
    def prefill_chunked(self, layer: int, chunks) -> None:
        """Chunked prefill over one layer: the only shape where D3 is reachable.

        v1 defers each chunk's reads and does the chunk's attention while they fly, then joins at
        the layer boundary -- `join_pending()` is a barrier over EVERY chunk's reads even though
        chunk k's MoE needs only chunk k's experts.
        """
        # THE ISSUE SCHEDULE IS IDENTICAL IN BOTH ARMS. An earlier revision routed, submitted and
        # ran chunk k's MoE before it reserved chunk k+1 in the D3-off arm, while the D3-on arm
        # submitted every chunk first -- so the two arms differed in WHEN reads were issued as well
        # as in what they waited for, and D3 was not an isolated toggle. Caught in review
        # 2026-09-15. Both arms now route and submit every chunk up front; the only difference is
        # the readiness condition before each chunk's FFN.
        pending = []
        per_chunk = []
        for uniq in chunks:
            slot_of, to_load, to_wait = self.slots.reserve(layer, uniq, prefill=True)
            to_load = to_load + to_wait
            self.c.fetches += len(to_load)
            self.loader.submit(to_load)
            pending.extend(to_load)
            per_chunk.append((to_load, sorted(set(slot_of.values()))))
            self._compute(lambda: self.leaves.prefill_attn(layer))

        for to_load, reads in per_chunk:
            # D3 ON: this chunk's FFN waits for EVERY chunk's reads. OFF: only its own.
            self._wait(pending if self.policy.global_barrier else to_load)
            self._compute(lambda r=reads: self.leaves.prefill_moe(layer, r), slots=reads)
