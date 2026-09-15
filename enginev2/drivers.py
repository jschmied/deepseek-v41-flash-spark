"""drivers.py -- request input -> prefill (chunked) and decode (stepped), over the replayed router.

The decode driver is the one the profile measured, so it is the one the dependency table is built
from. The prefill driver exists because D3 (`global_barrier`) is STRUCTURALLY UNREACHABLE at decode
and only a chunked driver can exercise it -- see the note on `decode_step`.
"""

from __future__ import annotations

import dataclasses
import time

from .chain import Chain, EngramSource
from .leaves import Bandwidth, ModelLeaves
from .observe import NO_CTX, Event, NullObserver, OpContext, next_span, now_ns
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
                 expert_read_qd: int = 8, h2d_inflight: int = 8, scale: float = 1.0,
                 leaves=None, calls=(), prefetch=None, discard_wrong_asap: bool = True,
                 engram=None, observer=None, request_id: int = 0):
        self.policy = policy
        # ONE observer, threaded to every component. Each emits facts about its OWN transitions;
        # the observer decides what to record and never influences scheduling.
        self.obs = observer if observer is not None else NullObserver()
        self.request_id = request_id
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
                                    expert_read_qd=expert_read_qd, h2d_inflight=h2d_inflight,
                                    scale=scale, observer=self.obs)
        self.scale = scale
        # THE PREFETCH SEAM. The default predicts nothing, which is the honest baseline: both arms
        # run this identical driver and differ only in what predict() returns. Nothing is granted
        # to either side.
        self.prefetch = prefetch if prefetch is not None else Prefetcher()
        self.discard_wrong_asap = discard_wrong_asap
        self.pf = PrefetchStats()
        # Every real happens-before edge of a step, named and waited on even where the for-loop
        # would have provided it. An edge that is only implicit is invisible to a plugged-in
        # component -- see chain.py.
        self.chain = Chain(observer=self.obs)
        self.engram = engram if engram is not None else EngramSource()
        self._spec: dict[tuple, tuple] = {}     # key -> (slot, gen, cause_id) still speculative
        # Every key the predictor NAMED, including ones already resident. Needed because precision
        # over predictions and precision over fetches are different numbers (see PrefetchStats):
        # a correct prediction that was already cached never becomes a fetch, so scoring only the
        # fetches would credit the predictor with none of its cheap hits.
        self._pred: dict[int, dict] = {}       # target layer -> {key: cause_id}
        # Every (key, slot) speculation got wrong, kept so a test can assert they really left the
        # cache rather than assert that a counter moved. Identity is (key, slot, GENERATION): the
        # same key can be predicted again later and be legitimately resident, even in the same
        # slot, and the generation is what tells that apart from a leak. It is the same
        # disambiguation the readiness events needed.
        self.wrong_keys_for_test: list = []
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

    def _wait(self, to_load, ctx=NO_CTX):
        """The consumer's own context goes in, not the producer's: for a speculative read these
        differ, and the wait belongs to whoever is blocked."""
        t0 = time.perf_counter()
        try:
            if self.policy.global_barrier:
                self.loader.wait_all(ctx=ctx)
            self.loader.wait_slots(to_load, ctx=ctx)
        finally:
            self.c.blocked_s += time.perf_counter() - t0

    # ------------------------------------------------------------------ decode
    def decode_layer(self, layer: int, ctx: OpContext | None = None) -> None:
        """One layer of one decode step.

        NOTE ON D3, recorded because it changes what the table can say. At decode the driver cannot
        reach layer L+1's router until layer L's MoE has produced its output, so the pending set
        NEVER holds more than the current layer's reads. `global_barrier` and per-slot waiting are
        therefore the SAME WAIT at decode, and D3's decode contribution is structurally zero rather
        than measured-small. It is exercised in prefill_chunked(), where several chunks of one layer
        are in flight together.
        """
        # FIRST statement: the edge waits below already use it, so constructing it afterwards left
        # the documented fallback passing None into chain.wait and losing the context entirely.
        ctx = ctx if ctx is not None else OpContext(self.request_id, self.c.steps, layer)

        # Graph A. The expert ids come OUT of it -- they are not handed to the driver. This is the
        # line that makes this an engine and not a replay of someone else's route.
        # EDGE 1: graph A reads h and pre_mix, which graph B of the PREVIOUS layer wrote. Both
        # stages write the same single `h` buffer, so this is also why A(L+1) cannot simply be run
        # early: it would clobber what B(L) still needs.
        self.chain.wait("h", layer - 1, ctx=ctx)
        # EDGE 5: eg_rows[L] comes off a second async NVMe stream and is read by graph A. The
        # source signals per layer; the driver only waits.
        self.chain.wait("engram", layer, ctx=ctx)

        e = self.obs.enabled
        if e:
            self.obs.safe_emit(Event(now_ns(), "layer_a_start", ctx=ctx))
        self.obs.safe_gpu_begin("layer_a", ctx)
        route = self._compute(lambda: self.leaves.layer_a(layer))
        self.obs.safe_gpu_end("layer_a", ctx)
        uniq = route.uniq
        if e:
            self.obs.safe_emit(Event(now_ns(), "layer_a_end", ctx=ctx))
            self.obs.safe_emit(Event(now_ns(), "route_ready", ctx=ctx, value=len(uniq), aux=uniq))
        # EDGE 2 and 6: A wrote y / route_idx / route_w, and the KV buffers this layer's
        # bookkeeping will clone.
        self.chain.set("y", layer, ctx=ctx)
        self.chain.set("route", layer, ctx=ctx)
        self.chain.set("kv", layer, ctx=ctx)

        # What did speculation actually buy, and what did it cost? Settled BEFORE reserve(), while
        # the previous prediction for this layer is still distinguishable from a real residency.
        self._settle_speculation(layer, uniq, ctx)

        # Apply anything the loader asked to un-map (discarded speculation, torn reads) HERE, on
        # the driver thread, so the cache is only ever mutated from one place.
        self.loader.drain_forgets()
        slot_of, to_load, to_wait = self.slots.reserve(layer, uniq, prefill=False)
        if e:
            for k, sl, g in to_load:
                self.obs.safe_emit(Event(now_ns(), "cache_miss", ctx=ctx, key=k, slot=sl, gen=g))
            # A resident key whose write is still in flight is a MAPPING hit, not a ready one --
            # reserve() puts it in to_wait and the consumer blocks. Reporting them together
            # overstates the cache.
            self.obs.safe_emit(Event(now_ns(), "cache_ready_hit", ctx=ctx,
                                     value=len(uniq) - len(to_load) - len(to_wait)))
            if to_wait:
                self.obs.safe_emit(Event(now_ns(), "cache_pending_hit", ctx=ctx, value=len(to_wait)))
        self.c.fetches += len(to_load)
        self.loader.submit(to_load, ctx=ctx)
        # A prefetch that has not landed yet is waited on exactly like a demand read. It is not
        # counted as a fetch -- it was already counted when it was issued.
        to_load = to_load + to_wait
        # Host bookkeeping, so it belongs BEFORE the wait: the real provider copies an index tensor
        # to the device here, which does not need the expert bytes to have arrived.
        self.leaves.bind_slots(route, slot_of)

        # Speculation is queued AFTER this layer's demand reads, at lower priority, so a demand
        # miss never sits behind a prefetch for a layer we have not reached.
        self._issue_speculation(layer, uniq, ctx)

        if self.policy.resolve_blocks:
            # v1: resolve() ends in list(pool.map(...)). Nothing issues work except a call that
            # immediately blocks on it, which is why the device cannot be kept busy at any thread
            # count. Everything after this point runs with the loader already drained.
            self._wait(to_load, ctx)

        # The shared expert is expert-INdependent, so a provider that captured it outside graph B
        # can run it here, while the reads fly. One that did not has it inside layer_b, where it
        # cannot overlap anything -- see leaves.Leaves.shared_first.
        if self.leaves.shared_first:
            self._compute(lambda: self.leaves.shared(layer))

        if not self.policy.resolve_blocks:
            self._wait(to_load, ctx)

        # Graph B. `route` carries the FUNCTIONAL input -- the provider's route-aligned slot tensor,
        # built by bind_slots above. `reads` is safety bookkeeping only: which arena slots this
        # compute reads, so the loader can order a later write against it. The two are not the same
        # thing and collapsing them (passing sorted(set(...)) as the input) loses the per-expert
        # ordering and multiplicity that moe_fn needs.
        reads = sorted(set(slot_of.values()))
        if e:
            self.obs.safe_emit(Event(now_ns(), "layer_b_start", ctx=ctx))
        self.obs.safe_gpu_begin("layer_b", ctx)
        # EDGE 3 and 4 are already enforced above: the ids came out of A, and _wait covers both the
        # `slots` mapping and the expert bytes being resident.
        self.chain.wait("y", layer, ctx=ctx)
        self._compute(lambda: self.leaves.layer_b(layer, route), slots=reads)
        self.obs.safe_gpu_end("layer_b", ctx)
        if e:
            self.obs.safe_emit(Event(now_ns(), "layer_b_end", ctx=ctx))
        self.chain.set("h", layer, ctx=ctx)            # B wrote h and pre_mix for the next layer

    # ------------------------------------------------------------------ prefetch bookkeeping
    def _settle_speculation(self, layer: int, uniq, ctx=None) -> None:
        """Score the predictions that were made for THIS layer, then drop the ones that missed."""
        want = {(layer, e) for e in uniq}
        named = self._pred.pop(layer, None)
        if named:
            hit = set(named) & want
            self.pf.pred_hit += len(hit)
            self.pf.pred_miss += len(named) - len(hit)
            if self.obs.enabled:
                # A correct prediction that was ALREADY RESIDENT never became I/O, so it appears in
                # no load chain. Emitting it keeps predictor accounting complete rather than
                # I/O-only, and it carries its own batch id.
                for k in hit:
                    if k not in self._spec:
                        self.obs.safe_emit(Event(now_ns(), "prediction_resident_hit",
                                                 ctx=ctx or NO_CTX, key=k, cause_id=named[k]))
        wrong = []
        for key in [k for k in self._spec if k[0] == layer]:
            slot, gen, cause = self._spec.pop(key)
            if key in want:
                self.pf.used += 1
                # READY or LATE? A prefetch whose read is still in flight is a mapping hit that the
                # consumer still blocks on; counting it with the ready ones would report a win the
                # engine never got.
                ready_ts = self.loader.ready.ready_ts(slot, gen)
                if ready_ts is not None:
                    self.pf.ready_hit += 1
                    lead = now_ns() - ready_ts
                    self.pf.lead_ns += lead
                    if self.obs.enabled:
                        self.obs.safe_emit(Event(now_ns(), "prefetch_ready_hit", ctx=ctx or NO_CTX,
                                                 key=key, slot=slot, gen=gen, cause_id=cause,
                                                 value=lead))
                else:
                    self.pf.late_hit += 1
                    if self.obs.enabled:
                        self.obs.safe_emit(Event(now_ns(), "prefetch_late_hit", ctx=ctx or NO_CTX,
                                                 key=key, slot=slot, gen=gen, cause_id=cause))
            else:
                self.pf.wasted += 1
                wrong.append((key, slot, gen))
                if self.obs.enabled:
                    self.obs.safe_emit(Event(now_ns(), "prefetch_wasted", ctx=ctx or NO_CTX,
                                             key=key, slot=slot, gen=gen, cause_id=cause))
        if wrong and self.discard_wrong_asap:
            # A wrong prefetch holds a slot AND a pending write, so it blocks eviction as well as
            # occupying capacity. Cancelling recovers the slot for reads that are already known to
            # be needed. Reads already in flight are not interrupted -- that cost is real and stays.
            self.wrong_keys_for_test.extend(wrong)
            q, r, f = self.loader.cancel(wrong)
            self.pf.cancelled_queued += q
            self.pf.discarded_running += r
            self.pf.discarded_finished += f

    def _issue_speculation(self, layer: int, uniq, ctx=None) -> None:
        self.prefetch.observe(layer, uniq, self.c.steps)
        keys = self.prefetch.predict(layer, uniq, self.c.steps)
        if not keys:
            return
        keys = [k for k in keys if k not in self._spec]
        # One id per prediction batch, carried by every event that batch causes, so a prediction's
        # whole life joins on a key instead of on timestamps. The commit text claimed this existed;
        # the schema did not have it.
        pred_id = next_span()
        for k in keys:
            self._pred.setdefault(k[0], {})[k] = pred_id
        to_load, refused = self.slots.reserve_speculative(keys)
        self.pf.refused += refused
        if self.obs.enabled:
            self.obs.safe_emit(Event(now_ns(), "prediction", ctx=ctx or NO_CTX, cause_id=pred_id,
                                     value=len(keys), aux=self.prefetch.name))
            if refused:
                self.obs.safe_emit(Event(now_ns(), "refused", ctx=ctx or NO_CTX, cause_id=pred_id,
                                         value=refused))
        if not to_load:
            return
        for key, slot, gen in to_load:
            self._spec[key] = (slot, gen, pred_id)
        self.pf.issued += len(to_load)
        if self.obs.enabled:
            for k, sl, g in to_load:
                # source_layer / horizon travel with the event, so a prediction's whole life is
                # readable without inferring it from timestamps.
                self.obs.safe_emit(Event(now_ns(), "prefetch_issued", ctx=ctx or NO_CTX, key=k,
                                         slot=sl, gen=g, cause_id=pred_id,
                                         aux=(layer, self.prefetch.horizon, self.prefetch.name)))
        self.loader.submit(to_load, speculative=True, ctx=ctx or NO_CTX, cause_id=pred_id)

    def decode(self, steps: int) -> Counters:
        """Run `steps` decode steps. Where the expert ids come from is the provider's business."""
        t0 = time.perf_counter()
        for step in range(steps):
            # EDGE 8: this step's inputs depend on the PREVIOUS step's logits, through draft and
            # verify. Waited on before anything else runs, and the step-level event is keyed by step
            # number so the reset below cannot delete it before it has been consumed -- which is
            # what made this edge decorative until now.
            if step:
                self.chain.wait("logits", step - 1)
            self.chain.reset(keep=("logits",))
            self.engram.issue(range(N_LAYERS), step, self.chain)
            ctx = OpContext(self.request_id, step, -1)
            for layer in range(N_LAYERS):
                self.decode_layer(layer, ctx.at(layer))
            self.chain.wait("h", N_LAYERS - 1)      # EDGE 7: gF replays after the last layer
            # Per-step GPU work outside the layer loop (head, draft). Zero unless the provider was
            # told that part of the unattributed 78.5 % lives here rather than in a layer.
            if self.obs.enabled:
                self.obs.safe_emit(Event(now_ns(), "final_start", ctx=OpContext(self.request_id, step, -1)))
            self._compute(self.leaves.step_other)
            if self.obs.enabled:
                self.obs.safe_emit(Event(now_ns(), "final_end", ctx=OpContext(self.request_id, step, -1)))
            self.chain.set("logits", step)
            self.c.steps += 1
        self.c.wall_s = time.perf_counter() - t0
        bw = self.loader.bw
        if bw is not None:                      # a real provider models no device; leave at 0.0
            self.c.mean_inflight = bw.mean_inflight
            self.c.achieved_gbs = bw.achieved_gbs
            self.c.device_busy_s = bw.busy_s
        return self.c

    def warm(self, calls, upto: int) -> None:
        """Bring the cache to the state the scored window starts in, with no I/O and no timing.

        AND NO EVENTS. Warm-up is not in the measured window; emitting from it flooded a trace with
        ~13,000 slot_ready events for 172 real loads and would have made any counter meaningless.
        """
        obs, self.obs = self.obs, NullObserver()
        self.loader.obs = self.loader.ready.obs = self.loader.stage.obs = self.obs
        try:
            for layer, uniq in calls[:upto]:
                _, to_load, _ = self.slots.reserve(layer, uniq, prefill=False)
                for key, slot, gen in to_load:
                    self.arena.content[slot] = key
                    self.loader.ready.set(slot, gen)
        finally:
            self.obs = obs
            self.loader.obs = self.loader.ready.obs = self.loader.stage.obs = obs
            self.chain.obs = obs

    # ------------------------------------------------------------------ prefill
    def prefill_chunked(self, layer: int, chunks, ctx: OpContext | None = None) -> None:
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
        ctx = ctx if ctx is not None else OpContext(self.request_id, self.c.steps, layer)
        pending = []
        per_chunk = []
        for uniq in chunks:
            self.loader.drain_forgets()
            slot_of, to_load, to_wait = self.slots.reserve(layer, uniq, prefill=True)
            self.c.fetches += len(to_load)
            # SUBMIT ONLY THE NEW READS. `to_wait` is already in flight from an earlier chunk;
            # extending before the submit queued those a second time, so one expert could be read
            # twice into the same slot. Wait on the union, submit only the difference.
            self.loader.submit(to_load, ctx=ctx)
            waits = to_load + to_wait
            pending.extend(waits)
            per_chunk.append((waits, sorted(set(slot_of.values()))))
            self._compute(lambda: self.leaves.prefill_attn(layer))

        for to_load, reads in per_chunk:
            # D3 ON: this chunk's FFN waits for EVERY chunk's reads. OFF: only its own.
            self._wait(pending if self.policy.global_barrier else to_load, ctx)
            self._compute(lambda r=reads: self.leaves.prefill_moe(layer, r), slots=reads)
