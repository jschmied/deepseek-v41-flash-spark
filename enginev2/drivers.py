"""drivers.py -- request input -> prefill (chunked) and decode (stepped), over the replayed router.

The decode driver is the one the profile measured, so it is the one the dependency table is built
from. The prefill driver exists because D3 (`global_barrier`) is STRUCTURALLY UNREACHABLE at decode
and only a chunked driver can exercise it -- see the note on `decode_step`.
"""

from __future__ import annotations

import dataclasses
import time

from .leaves import C_DEP, C_IND, C_OTHER, C_PRE, delay
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
                 device_queue_depth: int = 8, scale: float = 1.0):
        self.policy = policy
        self.slots = ExpertSlots(lru_slots, transient_slots, policy=evict)
        self.arena = SlotArena(self.slots.n_slots)
        self.compute = ComputeStream()
        self.loader = LoaderService(self.arena, self.slots, self.compute, policy,
                                    n_workers=n_workers, staging=staging,
                                    device_queue_depth=device_queue_depth, scale=scale)
        self.scale = scale
        self.c = Counters()

    def close(self):
        self.loader.shutdown()

    # ------------------------------------------------------------------ compute helpers
    def _compute(self, seconds: float, slots=()):
        t0 = time.perf_counter()
        with self.compute.run(slots):
            delay(seconds / self.scale)
        self.c.compute_s += time.perf_counter() - t0

    def _wait(self, to_load):
        t0 = time.perf_counter()
        try:
            if self.policy.global_barrier:
                self.loader.wait_all()
            self.loader.wait_slots(to_load)
        finally:
            self.c.blocked_s += time.perf_counter() - t0

    # ------------------------------------------------------------------ decode
    def decode_layer(self, layer: int, uniq) -> None:
        """One layer of one decode step.

        NOTE ON D3, recorded because it changes what the table can say. At decode the driver cannot
        reach layer L+1's router until layer L's MoE has produced its output, so the pending set
        NEVER holds more than the current layer's reads. `global_barrier` and per-slot waiting are
        therefore the SAME WAIT at decode, and D3's decode contribution is structurally zero rather
        than measured-small. It is exercised in prefill_chunked(), where several chunks of one layer
        are in flight together.
        """
        self._compute(C_PRE)                                    # attention + HC, pre-router
        slot_of, to_load = self.slots.reserve(layer, uniq, prefill=False)
        self.c.fetches += len(to_load)
        self.loader.submit(to_load)

        if self.policy.resolve_blocks:
            # v1: resolve() ends in list(pool.map(...)). Nothing issues work except a call that
            # immediately blocks on it, which is why the device cannot be kept busy at any thread
            # count. Everything after this point runs with the loader already drained.
            self._wait(to_load)

        if not self.policy.moe_before_shared:
            self._compute(C_IND)                                # D4 OFF: shared expert first

        if not self.policy.resolve_blocks:
            self._wait(to_load)

        self._compute(C_DEP, slots=sorted(set(slot_of.values())))   # routed MoE

        if self.policy.moe_before_shared:
            self._compute(C_IND)                                # D4 ON: too late to overlap

        self._compute(C_OTHER)

    def decode(self, calls, steps: int, start: int = 0) -> Counters:
        """`calls` is the decode trace; one step is N_LAYERS consecutive calls."""
        t0 = time.perf_counter()
        for s in range(steps):
            base = start + s * N_LAYERS
            for j in range(N_LAYERS):
                layer, uniq = calls[base + j]
                self.decode_layer(layer, uniq)
            self.c.steps += 1
        self.c.wall_s = time.perf_counter() - t0
        self.c.mean_inflight = self.loader.bw.mean_inflight
        self.c.achieved_gbs = self.loader.bw.achieved_gbs
        self.c.device_busy_s = self.loader.bw.busy_s
        return self.c

    def warm(self, calls, upto: int) -> None:
        """Bring the cache to the state the scored window starts in, with no I/O and no timing."""
        for layer, uniq in calls[:upto]:
            _, to_load = self.slots.reserve(layer, uniq, prefill=False)
            for key, slot, gen in to_load:
                self.arena.content[slot] = key
                self.loader.ready.set(slot, gen)

    # ------------------------------------------------------------------ prefill
    def prefill_chunked(self, layer: int, chunks, hold_per_chunk: float = C_PRE) -> None:
        """Chunked prefill over one layer: the only shape where D3 is reachable.

        v1 defers each chunk's reads and does the chunk's attention while they fly, then joins at
        the layer boundary -- `join_pending()` is a barrier over EVERY chunk's reads even though
        chunk k's MoE needs only chunk k's experts.
        """
        pending = []
        for uniq in chunks:
            slot_of, to_load = self.slots.reserve(layer, uniq, prefill=True)
            self.c.fetches += len(to_load)
            self.loader.submit(to_load)
            pending.extend(to_load)
            self._compute(hold_per_chunk)
            if not self.policy.global_barrier:
                self._wait(to_load)                             # only this chunk's experts
                self._compute(C_DEP, slots=sorted(set(slot_of.values())))
        if self.policy.global_barrier:
            self._wait(pending)                                 # the barrier over all chunks
            self._compute(C_DEP * len(chunks))
