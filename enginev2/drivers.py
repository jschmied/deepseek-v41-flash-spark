"""drivers.py -- request input -> prefill (chunked) and decode (stepped), over the replayed router.

The decode driver is the one the profile measured, so it is the one the dependency table is built
from. The prefill driver exists because D3 (`global_barrier`) is STRUCTURALLY UNREACHABLE at decode
and only a chunked driver can exercise it -- see the note on `decode_step`.
"""

from __future__ import annotations

import dataclasses
import os
import time

from .chain import Chain, EngramSource
from .leaves import Bandwidth, ModelLeaves
from .observe import NO_CTX, Event, NullObserver, OpContext, next_span, now_ns
from .prefetch import PrefetchStats, Prefetcher, SpecAttempt
from .sched import ComputeStream, LoaderService, Policy
from .store import ExpertSlots, SlotArena
from .trace import N_LAYERS


@dataclasses.dataclass
class HostPhases:
    """Wall time per driver phase, on the host, accumulated across steps.

    This measures WALL, not device time, and that is deliberate: the term being hunted is time in
    which neither the GPU nor the NVMe device is doing anything, so a device-time instrument cannot
    see it by construction. A phase that ends in a synchronous D2H therefore shows up as expensive
    HERE and cheap in a kernel trace -- which is exactly the signature being tested for.

    Off unless DSV41_HOST_PROFILE=1. Enabled it costs ~200 perf_counter calls per step (~10 us
    against a 203 ms step); disabled, `__call__` returns a shared no-op context manager.
    """

    class _Span:
        """Reusable only because the driver is single-threaded and these never nest with the same
        name; a fresh object per phase per layer would be 200 allocations per step."""

        __slots__ = ("p", "name", "t0")

        def __init__(self, parent, name):
            self.p, self.name, self.t0 = parent, name, 0.0

        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *a):
            self.p.t[self.name] += time.perf_counter() - self.t0
            self.p.n[self.name] += 1
            return False

    class _Off:
        def __enter__(self): return None
        def __exit__(self, *a): return False

    _OFF = _Off()

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.t: dict = {}
        self.n: dict = {}
        self._open: dict = {}
        self._spans: dict = {}

    def __call__(self, name: str):
        if not self.enabled:
            return self._OFF
        sp = self._spans.get(name)
        if sp is None:
            sp = self._spans[name] = self._Span(self, name)
            self.t.setdefault(name, 0.0)
            self.n.setdefault(name, 0)
        return sp

    def enter(self, name: str) -> None:
        """For a span that does not nest cleanly in a `with` -- see the resolve block."""
        if self.enabled:
            self._open[name] = time.perf_counter()
            self.t.setdefault(name, 0.0)
            self.n.setdefault(name, 0)

    def exit(self, name: str) -> None:
        if self.enabled and name in self._open:
            self.t[name] += time.perf_counter() - self._open.pop(name)
            self.n[name] += 1

    def reset(self) -> None:
        """Drop everything accumulated so far. The caller must do this after a warm-up, or the
        warm-up's phases are divided by the timed step count and every row is scaled by
        (warm + timed) / timed."""
        self.t.clear()
        self.n.clear()
        self._open.clear()
        self._spans.clear()

    def report(self, steps: int) -> str:
        if not self.enabled or not steps:
            return ""
        rows = sorted(self.t.items(), key=lambda kv: -kv[1])
        tot = sum(self.t.values())
        out = [f"  host phases over {steps} steps (wall, per step):"]
        for k, v in rows:
            out.append(f"    {k:<14} {v / steps * 1e3:7.2f} ms  x{self.n[k] / steps:6.1f}"
                       f"  {v / tot * 100:5.1f}% of instrumented")
        out.append(f"    {'INSTRUMENTED':<14} {tot / steps * 1e3:7.2f} ms")
        return "\n".join(out)


class Counters:
    steps: int = 0
    wall_s: float = 0.0
    blocked_s: float = 0.0        # driver time inside a wait for expert data
    compute_s: float = 0.0        # driver time inside a modelled compute leaf
    fetches: int = 0
    mean_inflight: float = 0.0
    achieved_gbs: float = 0.0
    device_busy_s: float = 0.0
    copies_awaited: int = 0       # H2Ds turned into a GPU dependency instead of a host block

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
        # FAST PATH, decided once and BEFORE anything that depends on it. The provider says whether
        # the DEVICE orders a slot's reuse against its previous reader; when it does, the host-side
        # ComputeStream, its per-slot wait, the synthetic SlotArena's write bookkeeping and all but
        # the engram edge of the Chain are verification machinery rather than synchronisation.
        # CHECK_INVARIANTS=1 forces every one of them back on.
        self._dev_orders = (getattr(self.leaves, "device_orders_slot_reuse", False)
                            and os.environ.get("CHECK_INVARIANTS") != "1")
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
        self.chain = Chain(observer=self.obs, fast=self._dev_orders)
        self.engram = engram if engram is not None else EngramSource()
        self._spec: dict[tuple, SpecAttempt] = {}      # key -> the attempt in flight for it
        # Attempts that DIED before their target and were retired so the key could be predicted
        # again closer to demand. They keep their TERMINAL REASON: relabelling an I/O failure as an
        # eviction made the failure-mode breakdown say the wrong thing about why prediction fails.
        self._expired: dict[int, list] = {}
        # Scored-ness is per attempt, not a global cutoff -- see SpecAttempt. `_scoring` is the
        # flag new attempts inherit; settlement clears it and restores it, so a later decode on the
        # same Engine is scored again.
        self._scoring = True
        # Every key the predictor NAMED, including ones already resident. Needed because precision
        # over predictions and precision over fetches are different numbers (see PrefetchStats):
        # a correct prediction that was already cached never becomes a fetch, so scoring only the
        # fetches would credit the predictor with none of its cheap hits.
        self._pred: dict[int, dict] = {}       # target layer -> {key: cause_id}
        # Slots the CURRENT layer has bound and graph B has not yet read. Speculation may not evict
        # these -- see ExpertSlots.reserve_speculative's protected_slots.
        self._layer_slots: frozenset = frozenset()
        self.c = Counters()
        self.hostprof = HostPhases(os.environ.get("DSV41_HOST_PROFILE") == "1")

    def close(self):
        self.loader.shutdown()

    # ------------------------------------------------------------------ compute helpers
    def _compute(self, fn, slots=()):
        """Run one compute leaf. `slots` are the arena slots it READS.

        ComputeStream is host-side bookkeeping that lets the loader order a slot's next writer
        against its previous reader. For a MODELLED leaf that is the only mechanism there is. For a
        real one it is both redundant and WRONG in the unsafe direction: layer_b() queues a CUDA
        graph and returns, so run() would declare the reader finished while the GPU is still
        reading. RealLeaves records an event after the graph and h2d waits on it per slot, which is
        the actual ordering -- see Leaves.device_orders_slot_reuse.

        There are ~84 _compute() calls per step (A and B over 40 layers plus the step's own), each
        a pair of condition-lock sections with Counter updates and notify_all.
        """
        t0 = time.perf_counter()
        if self._dev_orders:
            r = fn()
        else:
            with self.compute.run(slots):
                r = fn()
        self.c.compute_s += time.perf_counter() - t0
        return r

    def _wait(self, to_load, ctx=NO_CTX, scored: bool = True):
        """The consumer's own context goes in, not the producer's: for a speculative read these
        differ, and the wait belongs to whoever is blocked.

        `scored` is a PARAMETER, not a read of self._scoring, so the cohort of a wait is decided by
        the caller that owns it and cannot drift with engine state.
        """
        t0 = time.perf_counter()
        try:
            if self.policy.global_barrier:
                self.loader.wait_all(ctx=ctx, scored=scored)
            self.loader.wait_slots(to_load, ctx=ctx, scored=scored)
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
        with self.hostprof("wait_h"):
            self.chain.wait("h", layer - 1, ctx=ctx, scored=self._scoring)
        # EDGE 5: eg_rows[L] comes off a second async NVMe stream and is read by graph A. The
        # source signals per layer; the driver only waits.
        #
        # INSTRUMENTED DELIBERATELY: 288 engram rows per step arrive as 575 buffered preads on a
        # separate pool, and the open question is whether the driver ever BLOCKS on them. Two
        # engram layers (1 and 14) means two waits per step, so a large number here would be a
        # small number of very long waits -- which is what an IOPS queue behind the expert reads
        # would look like.
        with self.hostprof("wait_engram"):
            self.chain.wait("engram", layer, ctx=ctx, scored=self._scoring)

        e = self.obs.enabled
        hp = self.hostprof
        if e:
            self.obs.safe_emit(Event(now_ns(), "layer_a_start", ctx=ctx, scored=self._scoring))
        self.obs.safe_gpu_begin("layer_a", ctx)
        with hp("layer_a"):     # graph A replay PLUS the synchronous D2H of route_idx at its end
            route = self._compute(lambda: self.leaves.layer_a(layer))
        self.obs.safe_gpu_end("layer_a", ctx)
        uniq = route.uniq
        if e:
            self.obs.safe_emit(Event(now_ns(), "layer_a_end", ctx=ctx, scored=self._scoring))
            self.obs.safe_emit(Event(now_ns(), "route_ready", ctx=ctx, value=len(uniq), aux=uniq, scored=self._scoring))
        # EDGE 2 and 6: A wrote y / route_idx / route_w, and the KV buffers this layer's
        # bookkeeping will clone.
        self.chain.set("y", layer, ctx=ctx, scored=self._scoring)
        self.chain.set("route", layer, ctx=ctx, scored=self._scoring)
        self.chain.set("kv", layer, ctx=ctx, scored=self._scoring)

        # What did speculation actually buy, and what did it cost? Settled BEFORE reserve(), while
        # the previous prediction for this layer is still distinguishable from a real residency.
        hp.enter("resolve")
        self._settle_speculation(layer, uniq, ctx)

        # Apply anything the loader asked to un-map (discarded speculation, torn reads) HERE, on
        # the driver thread, so the cache is only ever mutated from one place.
        self.loader.drain_forgets()
        slot_of, to_load, to_wait = self.slots.reserve(layer, uniq, prefill=False)
        if e:
            for k, sl, g in to_load:
                self.obs.safe_emit(Event(now_ns(), "cache_miss", ctx=ctx, key=k, slot=sl, gen=g, scored=self._scoring))
            # A resident key whose write is still in flight is a MAPPING hit, not a ready one --
            # reserve() puts it in to_wait and the consumer blocks. Reporting them together
            # overstates the cache.
            self.obs.safe_emit(Event(now_ns(), "cache_ready_hit", ctx=ctx,
                                     value=len(uniq) - len(to_load) - len(to_wait), scored=self._scoring))
            if to_wait:
                self.obs.safe_emit(Event(now_ns(), "cache_pending_hit", ctx=ctx, value=len(to_wait),
                                         scored=self._scoring))
        self.c.fetches += len(to_load)
        # DEMAND work carries the cohort too. Settlement generates real demand misses, and submit
        # and _wait defaulted to scored=True -- so a settlement load_queued / nvme / staging /
        # EXPERT_DATA chain was labelled measured. Test 28 cannot see it: it compares the aggregate
        # against the trace on Event.scored, and if BOTH receive a wrongly-scored event they agree.
        self.loader.submit(to_load, ctx=ctx, scored=self._scoring)
        # A prefetch that has not landed yet is waited on exactly like a demand read. It is not
        # counted as a fetch -- it was already counted when it was issued.
        to_load = to_load + to_wait
        # Host bookkeeping, so it belongs BEFORE the wait: the real provider copies an index tensor
        # to the device here, which does not need the expert bytes to have arrived.
        self.leaves.bind_slots(route, slot_of)
        # From here until graph B has consumed them, these slots are live: they are baked into the
        # provider's slot tensor. Speculation issued below must not be able to evict one.
        self._layer_slots = frozenset(slot_of.values())

        # Speculation is queued AFTER this layer's demand reads, at lower priority, so a demand
        # miss never sits behind a prefetch for a layer we have not reached.
        self._issue_speculation(layer, uniq, ctx)
        hp.exit("resolve")      # everything from the route ids to the slot tensor, all host work

        # The shared expert is expert-INdependent, so a provider that captured it outside graph B
        # can enqueue it HERE, before any wait, and the device runs it while the reads fly. One
        # that did not has it inside layer_b, where it cannot overlap anything -- see
        # leaves.Leaves.shared_first.
        #
        # THIS USED TO SIT BETWEEN THE TWO WAIT BRANCHES, which made it dead code under the policy
        # that actually runs. bench_tokens constructs Policy(), i.e. V1, where resolve_blocks is
        # True -- so the wait above fired first and the shared expert was enqueued AFTER the reads
        # had already landed. Jobs 395, 400 and 405 all measured that arrangement and all returned
        # nulls, which is the correct answer to the question they were really asking and not the
        # question they were labelled with.
        if self.leaves.shared_first:
            with hp("shared"):
                self._compute(lambda: self.leaves.shared(layer))

        # v1 (resolve_blocks): resolve() ends in list(pool.map(...)). Nothing issues work except a
        # call that immediately blocks on it, which is why the device cannot be kept busy at any
        # thread count. Everything after this point runs with the loader already drained.
        #
        # The branch that used to be here chose only WHERE this wait sat relative to the shared
        # expert above; with that moved ahead of both, the two arms were the same statement twice.
        # resolve_blocks still shapes the run through _issue_speculation and the barriers -- it is
        # this ordering that stopped depending on it.
        with hp("wait_reads"):
            self._wait(to_load, ctx, scored=self._scoring)

        # Graph B. `route` carries the FUNCTIONAL input -- the provider's route-aligned slot tensor,
        # built by bind_slots above. `reads` is safety bookkeeping only: which arena slots this
        # compute reads, so the loader can order a later write against it. The two are not the same
        # thing and collapsing them (passing sorted(set(...)) as the input) loses the per-expert
        # ordering and multiplicity that moe_fn needs.
        reads = sorted(set(slot_of.values()))
        # The loader reported these ready as soon as their copies were ENQUEUED, so the bytes may
        # still be in flight. Put those copies on the compute stream before the graph reads them;
        # from here the device orders it. A provider without this seam never gets early readiness.
        with hp("await_copies"):
            self.c.copies_awaited += self.leaves.await_copies(reads)
        if e:
            self.obs.safe_emit(Event(now_ns(), "layer_b_start", ctx=ctx, scored=self._scoring))
        self.obs.safe_gpu_begin("layer_b", ctx)
        # EDGE 3 and 4 are already enforced above: the ids came out of A, and _wait covers both the
        # `slots` mapping and the expert bytes being resident.
        self.chain.wait("y", layer, ctx=ctx, scored=self._scoring)
        with hp("layer_b"):     # ENQUEUE only; the device work lands in the next sync
            self._compute(lambda: self.leaves.layer_b(layer, route), slots=reads)
        self.obs.safe_gpu_end("layer_b", ctx)
        if e:
            self.obs.safe_emit(Event(now_ns(), "layer_b_end", ctx=ctx, scored=self._scoring))
        self.chain.set("h", layer, ctx=ctx, scored=self._scoring)            # B wrote h and pre_mix for the next layer
        self._layer_slots = frozenset()        # B has consumed them; they are ordinary victims again

    # ------------------------------------------------------------------ prefetch bookkeeping
    def _replay_step(self) -> int:
        """Which decode step the REPLAY is on. layer_a has already consumed the current call, so
        the current index is cursor-1. Falls back to the timed counter for a provider with no
        cursor -- a real one computes its own routes and does not need this at all."""
        cur = getattr(self.leaves, "i", None)
        base = getattr(self.leaves, "start", None)
        if cur is None or base is None:
            return self.c.steps
        return max(0, (cur - 1 - base) // N_LAYERS)

    def _settle_speculation(self, layer: int, uniq, ctx=None) -> None:
        """Score the predictions that were made for THIS layer, then drop the ones that missed."""
        want = {(layer, e) for e in uniq}
        scored_wrong: dict = {}
        named = self._pred.pop(layer, None)
        if named:
            scored = {k for k, (_c, sc) in named.items() if sc}
            hit = scored & want
            self.pf.pred_hit += len(hit)
            self.pf.pred_miss += len(scored) - len(hit)
            if self.obs.enabled:
                # A correct prediction that was ALREADY RESIDENT never became I/O, so it appears in
                # no load chain. Emitting it keeps predictor accounting complete rather than
                # I/O-only, and it carries its own batch id.
                for k in hit:
                    if k not in self._spec:
                        self.obs.safe_emit(Event(now_ns(), "prediction_resident_hit",
                                                 ctx=ctx or NO_CTX, key=k, cause_id=named[k][0],
                                                 scored=named[k][1]))
        wrong = []
        # Attempts retired earlier for this layer: the read happened, so it counts, but it can
        # only ever have failed to serve demand.
        for att in self._expired.pop(layer, []):
            if not att.scored:
                continue                          # nothing physical left to do: already retired
            # BOTH dimensions. evicted_before_use means "right, but too early"; labelling a retired
            # attempt that way without checking whether the layer wanted it turned a plain false
            # positive into a timing failure and made the failure-mode breakdown say the wrong
            # thing about WHY prediction fails. The fetch-precision denominator is unaffected --
            # that is measured at the read leaf -- but the diagnosis is not.
            if att.key not in want:
                self.pf.wasted += 1               # simply wrong; its read is in started_cohort
                continue
            if att.terminal == "failed":
                self.pf.failed_before_use += 1
            else:
                self.pf.evicted_before_use += 1
            self.pf.wasted += 1

        for key in [k for k in self._spec if k[0] == layer]:
            att = self._spec.pop(key)
            slot, gen, cause = att.slot, att.gen, att.cause_id
            # `scored` GATES ACCOUNTING ONLY. Returning early here also skipped the physical
            # discard, so a wrong prediction issued during settlement stayed RESIDENT -- settlement
            # silently switched the discard policy it was supposed to be holding constant, and the
            # retained expert then evicted something a later scored prediction needed.
            if key in want:
                # RESIDENT NOW, not "was ready once". _spec is independent of residency and
                # SlotReady keeps (slot, gen) readiness for the life of the process, so a prefetch
                # that completed and was then evicted before its target layer still looked ready --
                # and was counted as a timely, useful hit while the engine re-read it as a demand
                # miss. Classify against the CURRENT mapping and generation.
                st = self.loader.ready.state(slot, gen)
                if st == "error":
                    # The prediction may have been perfect; the I/O did not land. The driver's
                    # deferred un-map has not run yet, so the mapping still looks valid here --
                    # which is how a FAILED read was being counted as a timely hit.
                    if att.scored:
                        self.pf.failed_before_use += 1
                        self.pf.wasted += 1
                    continue
                resident = (self.slots.lru.get(key) == slot
                            and self.slots.gen.get(slot) == gen)
                if not resident:
                    if att.scored:
                        self.pf.evicted_before_use += 1
                        self.pf.wasted += 1
                    if self.obs.enabled:
                        self.obs.safe_emit(Event(now_ns(), "prefetch_evicted_before_use",
                                                 ctx=ctx or NO_CTX, key=key, slot=slot, gen=gen,
                                                 cause_id=cause, scored=att.scored))
                    continue
                if att.scored:
                    self.pf.used += 1
                # READY or LATE? A prefetch whose read is still in flight is a mapping hit that the
                # consumer still blocks on; counting it with the ready ones would report a win the
                # engine never got.
                ready_ts = self.loader.ready.ready_ts(slot, gen) if st == "ready" else None
                if ready_ts is not None:
                    lead = now_ns() - ready_ts
                    if att.scored:
                        self.pf.ready_hit += 1
                        self.pf.lead_ns += lead
                    if self.obs.enabled:
                        self.obs.safe_emit(Event(now_ns(), "prefetch_ready_hit", ctx=ctx or NO_CTX,
                                                 key=key, slot=slot, gen=gen, cause_id=cause,
                                                 value=lead, scored=att.scored))
                else:
                    if att.scored:
                        self.pf.late_hit += 1
                    if self.obs.enabled:
                        self.obs.safe_emit(Event(now_ns(), "prefetch_late_hit", ctx=ctx or NO_CTX,
                                                 key=key, slot=slot, gen=gen, cause_id=cause, scored=att.scored))
            else:
                if att.scored:
                    self.pf.wasted += 1
                wrong.append((key, slot, gen))
                scored_wrong[(slot, gen)] = att.scored
                if self.obs.enabled:
                    self.obs.safe_emit(Event(now_ns(), "prefetch_wasted", ctx=ctx or NO_CTX,
                                             key=key, slot=slot, gen=gen, cause_id=cause, scored=att.scored))
        if wrong and self.discard_wrong_asap:
            # A wrong prefetch holds a slot AND a pending write, so it blocks eviction as well as
            # occupying capacity. Cancelling recovers the slot for reads that are already known to
            # be needed. Reads already in flight are not interrupted -- that cost is real and stays.
            for _k, _sl, _g, state in self.loader.cancel(wrong):
                if not scored_wrong.get((_sl, _g), True):
                    self.pf.discarded_unscored += 1   # discarded physically, not accounted
                    continue
                if state == "queued":
                    self.pf.cancelled_queued += 1
                elif state == "running":
                    self.pf.discarded_running += 1
                else:
                    self.pf.discarded_finished += 1

    def _issue_speculation(self, layer: int, uniq, ctx=None) -> None:
        # THE REPLAY POSITION, not the timed step counter. c.steps is deliberately frozen during
        # settlement, so once settlement crossed layer 39 -> 0 the predictor was handed the
        # PREVIOUS step and read its future from the wrong place in the trace. Derive it from the
        # provider's own cursor, which is the only thing that actually tracks where the replay is.
        step = self._replay_step()
        self.prefetch.observe(layer, uniq, step)
        keys = self.prefetch.predict(layer, uniq, step)
        if not keys:
            return
        live = []
        for k in keys:
            old = self._spec.get(k)
            if old is None:
                live.append(k)
                continue
            still = (self.slots.lru.get(k) == old.slot and self.slots.gen.get(old.slot) == old.gen)
            if still:
                continue                      # genuinely outstanding or resident: no duplicate
            # DEAD. Suppressing the retry made the scheduler "earliest prediction wins forever":
            # a prediction that loaded at L0 and was evicted at L7 blocked every closer prediction
            # for the same key until L16, where it could only ever classify as evicted_before_use.
            # Retire it so it still accounts, and let the nearer prediction through.
            st = self.loader.ready.state(old.slot, old.gen)
            old.terminal = "failed" if st == "error" else "evicted"
            self._expired.setdefault(k[0], []).append(old)
            del self._spec[k]
            if old.scored:
                self.pf.reissued += 1
            live.append(k)
        keys = live
        # One id per prediction batch, carried by every event that batch causes, so a prediction's
        # whole life joins on a key instead of on timestamps. The commit text claimed this existed;
        # the schema did not have it.
        pred_id = next_span()
        for k in keys:
            # The scored bit travels with the PREDICTION too: a resident-correct prediction never
            # becomes a fetch, so a fetch cutoff could never have gated it.
            #
            # MERGE, NEVER OVERWRITE. One record per (target, key), and an unscored settlement
            # prediction must not downgrade a scored one. The oracle re-predicts the same target
            # every layer inside its horizon, and a RESIDENT correct prediction has no _spec entry
            # to suppress the duplicate -- so a prediction genuinely issued in the timed window
            # vanished from pred_hit/pred_miss simply because settlement named the same target
            # again. Scoring is a property of "was this ever predicted while measuring", so it is
            # a logical OR and the cause of record stays with the scored emission.
            tgt = self._pred.setdefault(k[0], {})
            prev = tgt.get(k)
            if prev is None:
                tgt[k] = (pred_id, self._scoring)
            elif self._scoring and not prev[1]:
                tgt[k] = (pred_id, True)          # upgrade; never the reverse
        # The current layer's slots are OFF LIMITS to speculation: bind_slots has already written
        # them into the provider's route-aligned tensor and graph B has not consumed them yet.
        to_load, refused = self.slots.reserve_speculative(
            keys, protected_slots=self._layer_slots)
        if self._scoring:
            self.pf.refused += refused
        if self.obs.enabled:
            self.obs.safe_emit(Event(now_ns(), "prediction", ctx=ctx or NO_CTX, cause_id=pred_id,
                                     value=len(keys), aux=self.prefetch.name, scored=self._scoring))
            if refused:
                self.obs.safe_emit(Event(now_ns(), "refused", ctx=ctx or NO_CTX, cause_id=pred_id,
                                         value=refused, scored=self._scoring))
        if not to_load:
            return
        for key, slot, gen in to_load:
            self._spec[key] = SpecAttempt(key, slot, gen, pred_id, scored=self._scoring)
        if self._scoring:
            self.pf.issued += len(to_load)
        if self.obs.enabled:
            for k, sl, g in to_load:
                # source_layer / horizon travel with the event, so a prediction's whole life is
                # readable without inferring it from timestamps.
                self.obs.safe_emit(Event(now_ns(), "prefetch_issued", ctx=ctx or NO_CTX, key=k,
                                         slot=sl, gen=g, cause_id=pred_id,
                                         aux=(layer, self.prefetch.horizon, self.prefetch.name),
                                         scored=self._scoring))
        self.loader.submit(to_load, speculative=True, ctx=ctx or NO_CTX, cause_id=pred_id,
                           scored=self._scoring)

    def decode(self, steps: int) -> Counters:
        """Run `steps` decode steps. Where the expert ids come from is the provider's business."""
        t0 = time.perf_counter()
        for step in range(steps):
            # EDGE 8: this step's inputs depend on the PREVIOUS step's logits, through draft and
            # verify. Waited on before anything else runs, and the step-level event is keyed by step
            # number so the reset below cannot delete it before it has been consumed -- which is
            # what made this edge decorative until now.
            # THIS step's context, built before the first wait that is attributed to it. It used
            # to be constructed after, so the wait for the previous step's logits was filed under
            # the previous iteration's ctx -- typically step-1/layer 39 -- and on the first step
            # under a name that did not exist yet. Execution is unchanged; the trace was wrong.
            ctx = OpContext(self.request_id, step, -1)
            if step:
                self.chain.wait("logits", step - 1, ctx=ctx, scored=self._scoring)
            self.chain.reset(keep=("logits",))
            # THE BLOCK FIRST, then the engram reads that hash it. v1 builds the verify block,
            # hashes it and submits both tables' reads before the step runs; issuing engram first
            # would read rows for the PREVIOUS step's tokens.
            self._compute(lambda: self.leaves.select_block(step))
            self.engram.issue(range(N_LAYERS), step, self.chain, ctx=ctx, scored=self._scoring)
            # The step's own prologue, before any layer: see Leaves.begin_step.
            with self.hostprof("begin_step"):
                self._compute(lambda: self.leaves.begin_step(step))
            for layer in range(N_LAYERS):
                self.decode_layer(layer, ctx.at(layer))
            self.chain.wait("h", N_LAYERS - 1, ctx=ctx, scored=self._scoring)      # EDGE 7: gF replays after the last layer
            # Per-step GPU work outside the layer loop (head, draft). Zero unless the provider was
            # told that part of the unattributed 78.5 % lives here rather than in a layer.
            if self.obs.enabled:
                self.obs.safe_emit(Event(now_ns(), "final_start",
                                         ctx=OpContext(self.request_id, step, -1),
                                         scored=self._scoring))
            with self.hostprof("step_other"):
                self._compute(self.leaves.step_other)
            with self.hostprof("end_step"):   # draft, verify, rollback -- all host, all synchronous
                self._compute(lambda: self.leaves.end_step(step))
            if self.obs.enabled:
                self.obs.safe_emit(Event(now_ns(), "final_end",
                                         ctx=OpContext(self.request_id, step, -1),
                                         scored=self._scoring))
            self.chain.set("logits", step, ctx=ctx, scored=self._scoring)
            self.c.steps += 1
        self.c.wall_s = time.perf_counter() - t0
        # The window boundary, not the total: speculation issued near the last layer has not begun
        # reading yet. finalize_stats() settles it.
        self.pf.started_at_window_end = self.loader.started_spec
        self.pf.started = self.loader.started_spec
        bw = self.loader.bw
        if bw is not None:                      # a real provider models no device; leave at 0.0
            self.c.mean_inflight = bw.mean_inflight
            self.c.achieved_gbs = bw.achieved_gbs
            self.c.device_busy_s = bw.busy_s
        return self.c

    def settle(self, layers: int | None = None) -> int:
        """Run extra layers UNTIMED with prediction OFF, so predictions already issued for targets
        past the timed window get CLASSIFIED instead of censored.

        Without this the tail is right-censored and the censoring is horizon-dependent: a horizon-4
        predictor has four layers' worth of useful predictions that the window stopped before
        consuming, while horizon 1 has one. `started` grows, ready_hit/late_hit/wasted cannot, and
        the longer horizon is penalised for nothing but where the benchmark ended. That is a bias
        in favour of short horizons in exactly the sweep meant to choose a horizon.

        THE PREDICTOR STAYS ON. Disabling it removes the contention the tail would actually meet:
        in steady-state decode, layer L+1 issues speculation that competes for NVMe admission and
        cache slots with a prediction still targeting L+3. Settling with prediction off gives those
        tail predictions an artificially clear path to becoming ready and surviving until use --
        the opposite bias to censoring, and again horizon-dependent. So the cohort is delimited by
        SEQUENCE instead: predictions issued during settlement run, contend, and are simply not
        scored. The timed counters are restored afterwards; only classification of the cohort moves.
        """
        if layers is None:
            layers = getattr(self.prefetch, "horizon", 0)
        if layers <= 0:
            return 0
        # MODEL/REPLAY ONLY. This drives decode_layer() directly, without a step's chain.reset(),
        # engram.issue() or the final/draft/verify transition -- fine when the next route comes off
        # a recorded trace, wrong as a runtime API. Refused on a provider that is not replaying.
        if not hasattr(self.leaves, "calls"):
            raise RuntimeError("settle() is a replay mechanism; it has no meaning for a provider "
                               "that computes its own routes")
        # Validate the replay HAS the layers before starting, rather than discovering it halfway.
        avail = len(self.leaves.calls) - self.leaves.i
        if avail < layers:
            raise RuntimeError(f"settle({layers}) needs {layers} more calls, trace has {avail}")
        import copy
        saved_c = copy.copy(self.c)
        self._scoring = False                        # run and contend, but do not count
        ran = 0
        try:
            for _ in range(layers):
                # CONTINUE the sequence, do not restart it. `i % N_LAYERS` happened to be right
                # only when settle() ran immediately after a whole number of steps; a second
                # settle, or one after a partial step, desynced against the replay.
                layer = self.leaves.calls[self.leaves.i][0]
                self.decode_layer(layer, OpContext(self.request_id, self.c.steps, layer))
                ran += 1
        finally:
            self._scoring = True                     # restored: a later decode is scored again
            wall, steps = self.c.wall_s, self.c.steps
            self.c = saved_c                         # timed counters are the window's, not this
            self.c.wall_s, self.c.steps = wall, steps
        # STRICT. Swallowing exceptions here would let the anti-censoring mechanism silently
        # reintroduce the censoring it exists to remove -- a read failure, an invariant trip or
        # slot exhaustion would leave predictions unclassified and finalize_stats() would still
        # return apparently valid numbers. The 99.8/99.4/99.0 -> 100 % measurement shows a handful
        # of unclassified entries moves the metric, so partial settlement must be loud.
        if ran != layers:
            raise RuntimeError(f"settle ran {ran} of {layers} layers; stats are censored")
        return ran

    def finalize_stats(self, cancel_outstanding: bool = False, timeout: float = 60.0,
                       settle_layers: int | None = None) -> None:
        """Settle the asynchronous counters. Call before reading pf for an experiment.

        Speculation outstanding when the timed window closed still causes real I/O, and reading
        precision before it resolves flatters the predictor. Two honest choices, both offered:
        let it finish (the I/O the window caused), or cancel what has not started (the I/O a
        predictor would cause if the request ended here). Default is the former, because it is the
        one that answers "what did this prediction window cost".
        """
        if settle_layers is not None or cancel_outstanding is False:
            # settle first: classify what the window issued but did not reach
            self.settle(settle_layers)
        if cancel_outstanding:
            scored_of = {(a.slot, a.gen): a.scored for a in self._spec.values()}
            pending = [(a.key, a.slot, a.gen) for a in self._spec.values()]
            if pending:
                for _k, _sl, _g, state in self.loader.cancel(pending):
                    if not scored_of.get((_sl, _g), True):
                        continue                   # cancelled physically, not accounted
                    if state == "queued":
                        self.pf.cancelled_queued += 1
                    elif state == "running":
                        self.pf.discarded_running += 1
                    else:
                        self.pf.discarded_finished += 1
            # POP them: leaving cancelled entries in _spec made a second finalize_stats() account
            # the same tail twice, and a queued-cancelled item is no longer in _queued, so the
            # second pass would misclassify it as running.
            for k, _sl, _g in pending:
                self._spec.pop(k, None)
        self.loader.quiesce(timeout)
        self.loader.drain_forgets()
        self.pf.started = self.loader.started_spec
        self.pf.started_cohort = self.loader.started_spec_scored

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
            self.loader.submit(to_load, ctx=ctx, scored=self._scoring)
            waits = to_load + to_wait
            pending.extend(waits)
            per_chunk.append((waits, sorted(set(slot_of.values()))))
            self._compute(lambda: self.leaves.prefill_attn(layer))

        for to_load, reads in per_chunk:
            # D3 ON: this chunk's FFN waits for EVERY chunk's reads. OFF: only its own.
            self._wait(pending if self.policy.global_barrier else to_load, ctx,
                       scored=self._scoring)
            self._compute(lambda r=reads: self.leaves.prefill_moe(layer, r), slots=reads)
