"""prefetch.py -- the preload seam, and the perfect-oracle that bounds every predictor.

WHY AN ORACLE FIRST. Build the ideal component before any predictor: it bounds every possible
implementation at once, so a training budget is only ever spent against a measured ceiling. The
same discipline killed the expert prediction head in the PROTECTION role -- a perfect oracle there
was worth 0.4 pp, so no head could be worth more.

THREE THINGS THIS MUST MODEL, each of which was a real error in this project before:

  * SPECULATIVE READS CONTEND WITH DEMAND READS. A prefetch for layer L+1 and a demand miss on
    layer L want the same device. They go down the same queue and take the same expert-read admission permits;
    demand reads are only PRIORITISED, never exempted. An oracle that prefetched for free would
    report a ceiling the device cannot deliver.

  * A WRONG PREFETCH OCCUPIES A SLOT. It evicted something to get there and it holds a pending
    write, so it also costs capacity and blocks eviction. `discard_wrong_asap` models the mitigation
    -- when the layer's real ids arrive, still-pending speculation that missed is cancelled and its
    slots released -- and the counters below separate useful from wasted so neither can hide.

  * THE BASELINE ARM MUST BE PHYSICALLY REALIZABLE. The worst error of 2026-09-15 was an oracle
    whose no-prediction arm was handed compute that ran before the misses were knowable; the
    conclusion inverted once that was fixed. Here both arms run the SAME driver, and the only
    difference is whether `predict()` returns anything. Nothing is granted to either arm.

WHAT AN ORACLE CANNOT TELL YOU. It bounds the win from perfect knowledge, and says nothing about
whether a predictor can reach any given recall. That is what the transition table measures.
"""

from __future__ import annotations

import dataclasses

from .trace import N_LAYERS


@dataclasses.dataclass(slots=True)
class SpecAttempt:
    """One speculative attempt, from prediction to terminal state.

    This exists because the same information was an anonymous tuple that changed shape three times
    -- (slot, gen), then +cause_id, then +seq -- and the last change broke
    finalize_stats(cancel_outstanding=True) at runtime, because one unpack site was missed. A named
    object makes that class of regression a NameError at edit time instead of a ValueError in a
    benchmark.

    `scored` is per attempt rather than a global sequence cutoff: settlement runs the predictor and
    its physical work, and only the STATISTICS are withheld. A global cutoff also leaked -- it
    stayed set after settle(), so any later decode on the same Engine was permanently unscored.
    """

    key: tuple
    slot: int
    gen: int
    cause_id: int
    scored: bool = True
    # Why a RETIRED attempt ended, and only that: "evicted" or "failed". Attempts that reach their
    # target are classified in place and never carry one, so this is not a general lifecycle field.
    terminal: str | None = None


@dataclasses.dataclass
class PrefetchStats:
    """TWO PRECISIONS, and the gap between them is a finding, not bookkeeping noise.

    A predictor's precision over PREDICTIONS is not its precision over FETCHES, and the second is
    the one that costs anything. Correct predictions are mostly already resident -- the cache runs
    at ~92 % hit rate, so naming an expert that will be used usually asks for no I/O at all. Wrong
    predictions are essentially never resident, so every one of them buys a slot, an eviction and a
    device admission. The issued mix is therefore dominated by the errors.

    Measured here: a synthetic predictor asked for precision 0.50 at horizon 1 realises 0.161 over
    fetches, and 0.25 realises 0.092. That is not the knob misbehaving; it is the amplification
    above. The consequence for "is a trained prediction head worth it" is direct -- a predictor has
    to be far more precise than intuition suggests, because its hits are largely free and its
    misses are always paid in full.
    """

    pred_hit: int = 0        # keys named that the layer really wanted (resident or fetched)
    pred_miss: int = 0       # keys named that it did not
    issued: int = 0          # of all named keys, the ones that actually needed a read
    used: int = 0            # ready_hit + late_hit
    # A HIT THAT ARRIVED TOO LATE IS NOT A WIN, and precision alone cannot tell them apart. A
    # predictor at 80 % precision whose hits are mostly `late` has not solved decode: the consumer
    # still blocks, it has merely blocked on a read that was started earlier. This is the number a
    # decision about training a head has to be made on.
    ready_hit: int = 0       # wanted, resident AND ready when the layer arrived
    late_hit: int = 0        # wanted and still mapped, but the read was in flight -- blocked anyway
    # Correct prediction, read completed, and then EVICTED before its target layer arrived. The
    # prefetch saved nothing and the expert is read again as a demand miss. Distinct from a late
    # hit (arrived too slowly) and from a wrong prediction (never wanted): this one was right and
    # too EARLY. Counting it as a ready hit -- which the classifier did until it checked residency
    # rather than historical readiness -- overstates used, ready_hit, timeliness, fetch precision
    # and mean lead all at once.
    evicted_before_use: int = 0
    # Speculative read that FAILED. Distinct from every other state: the prediction may have been
    # perfect and the I/O simply did not land.
    failed_before_use: int = 0
    # A key predicted again, closer to its target, after an earlier attempt died. The earlier read
    # still counts in the denominator -- it happened -- but suppressing the retry made the
    # scheduler "earliest prediction wins forever", which is the wrong policy when early death is
    # common.
    reissued: int = 0
    # Wrong attempts that were DISCARDED PHYSICALLY but not accounted, because they were issued
    # while scoring was off. Its only job is to prove the physical path still ran for them: the
    # scored flag must gate counting and never behaviour.
    discarded_unscored: int = 0
    lead_ns: int = 0         # summed (demand_ts - ready_ts) over ready hits
    wasted: int = 0          # speculative reads that did not serve demand, for any reason
    # THREE STATES, kept apart because they mean different things experimentally: queued cost
    # nothing, running cost a read that was wasted, finished means the predictor was wrong AND
    # early enough to have consumed residency. Folding finished into `cancelled` contradicted its
    # own definition ("dropped before it started").
    cancelled_queued: int = 0
    discarded_running: int = 0
    discarded_finished: int = 0
    # TWO BOUNDARIES, because speculation is asynchronous. `started_at_window_end` is what had
    # begun when the timed window closed; `started` is the honest total after outstanding
    # speculation is resolved, which is the I/O the prediction window actually caused. Reading
    # precision off the first flatters the predictor: predictions issued near the last layer start
    # their reads after the window closed.
    started_at_window_end: int = 0
    started: int = 0
    refused: int = 0         # predictions the store had no free slot for

    # Reads that started AND are scored. MEASURED at the read leaf by the loader, not reconstructed
    # from outcomes: an injected failure raises BEFORE started_spec increments, so summing outcome
    # buckets counted a read that provably never reached the leaf. Set from
    # LoaderService.started_spec_scored by finalize_stats().
    started_cohort: int = 0

    @property
    def precision(self) -> float:
        """Over FETCHES THAT HAPPENED, within the scored cohort.

        `issued` counts SUBMISSIONS, and a queued cancellation means the submission never became a
        read -- which is the entire point of cancelling it. Dividing by `issued` therefore charged
        the predictor for I/O it did not do and understated fetch precision, on exactly the surface
        a training decision is read from.

        `started` is counted where the read leaf is entered, and only settles once outstanding
        speculation is resolved -- call Engine.finalize_stats(). Before that it holds the
        window-end value, which is an UNDERCOUNT of the I/O caused.
        """
        # No fallback to `issued`: before finalize_stats() runs, reverting to submissions would
        # report the very number the started counter was introduced to replace.
        n = self.started_cohort
        return self.used / n if n else 0.0

    @property
    def discarded_total(self) -> int:
        """All wrong speculation, in any state. Named so it cannot be read as `reads_avoided`."""
        return self.cancelled_queued + self.discarded_running + self.discarded_finished

    @property
    def reads_avoided(self) -> int:
        """Wrong speculation cancelled BEFORE it read anything -- the only state that saves I/O."""
        return self.cancelled_queued

    @property
    def timeliness(self) -> float:
        """Of the hits, how many actually arrived in time to save the consumer a wait."""
        return self.ready_hit / self.used if self.used else 0.0

    @property
    def mean_lead_ms(self) -> float:
        return (self.lead_ns / self.ready_hit / 1e6) if self.ready_hit else 0.0

    @property
    def precision_predicted(self) -> float:
        """Over PREDICTIONS: what a predictor's own offline eval would report."""
        n = self.pred_hit + self.pred_miss
        return self.pred_hit / n if n else 0.0


class Prefetcher:
    """Return the expert ids to preload for a FUTURE layer, or () for none.

    `horizon` is how many layers ahead this predictor speaks for; the driver uses it to decide when
    a prediction can still be cancelled. A predictor that cannot see past the current layer returns
    () and costs nothing -- that is the null arm, and it is the honest baseline.
    """

    name = "null"
    horizon = 0

    def predict(self, layer: int, uniq: tuple, step: int) -> tuple:
        """-> ((layer, expert), ...) to preload. Called AFTER this layer's demand reads are queued,
        because a demand miss must never queue behind speculation for a layer not yet reached."""
        return ()

    def observe(self, layer: int, uniq: tuple, step: int) -> None:
        """What layer `layer` actually wanted. A learned predictor updates here; the oracle ignores
        it; the null arm does nothing. Called before predict() for the same layer."""


class OraclePrefetcher(Prefetcher):
    """Perfect knowledge, bounded by nothing but the device and the slots.

    It reads the trace `horizon` layers ahead. That is legitimate for a CEILING -- the question it
    answers is "what is the most any predictor could be worth", not "can a predictor do this".
    """

    name = "oracle"

    def __init__(self, calls, horizon: int = 1, start: int = 0):
        self.calls = calls
        self.horizon = horizon
        self.start = start

    def predict(self, layer: int, uniq: tuple, step: int) -> tuple:
        out = []
        i = self.start + step * N_LAYERS + layer
        for k in range(1, self.horizon + 1):
            j = i + k
            if j >= len(self.calls):
                break
            fl, fu = self.calls[j]
            out.extend((fl, e) for e in fu)
        return tuple(out)


class RecallOraclePrefetcher(OraclePrefetcher):
    """An oracle degraded to a fixed activation recall, to trace the recall->win curve.

    The corrected ceiling says that curve is near-linear, so recall 0.30 -- which our untrained
    transition table already reaches -- should capture 28-41 % of the oracle win. This arm is how
    that claim gets checked against a scheduler instead of against arithmetic. Selection is
    deterministic in the key so the same expert is predicted consistently, not resampled per call.
    """

    name = "recall_oracle"

    def __init__(self, calls, horizon: int = 1, recall: float = 0.30,
                 precision: float = 1.0, n_experts: int = 384, start: int = 0):
        super().__init__(calls, horizon=horizon, start=start)
        self.recall = recall
        # PRECISION IS A SEPARATE AXIS, and without it the discard policy is untestable. A
        # recall-degraded oracle still only names experts that ARE used, so nothing is ever wrong
        # and `discard_wrong_asap` never fires -- which would let a study conclude that wrong
        # prefetches are free. A real predictor misses AND misfires; precision < 1 injects the
        # misfires, each of which takes a slot, an eviction and a share of the device.
        self.precision = precision
        self.n_experts = n_experts

    def predict(self, layer: int, uniq: tuple, step: int) -> tuple:
        truth = super().predict(layer, uniq, step)
        # EXCLUDE THE WHOLE TRUTH, not the recall-retained part of it. Drawing "wrong" picks against
        # the truncated set lets an expert that recall DROPPED come back as a supposed false
        # positive -- and it is used when the layer arrives, so the arm's realised precision quietly
        # exceeds the requested one. That biases exactly the recall/precision surface a decision
        # about training a prediction head would be read off. Caught in review 2026-09-15.
        all_true = set(truth)
        full = truth
        if self.recall < 1.0:
            n = int(len(full) * self.recall)
            # deterministic, key-stable subset: rank by a cheap hash of the key
            full = tuple(sorted(full, key=lambda k: (k[0] * 2654435761 + k[1] * 40503) & 0xFFFF)[:n])
        if self.precision >= 1.0 or not full:
            return tuple(full)
        # keep |full| correct picks and pad with wrong ones until the ratio holds, so the arm's
        # ISSUE COUNT reflects what a predictor of this precision would really put on the device.
        n_wrong = int(len(full) * (1.0 - self.precision) / self.precision)
        drawn = set(all_true)
        wrong, e, guard = [], 0, 0
        while len(wrong) < n_wrong and guard < 10 * self.n_experts:
            guard += 1
            fl = full[e % len(full)][0]
            cand = (fl, (step * 7919 + layer * 104729 + guard * 31) % self.n_experts)
            if cand not in drawn:
                wrong.append(cand)
                drawn.add(cand)
            e += 1
        return tuple(full) + tuple(wrong)
