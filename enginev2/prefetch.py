"""prefetch.py -- the preload seam, and the perfect-oracle that bounds every predictor.

WHY AN ORACLE FIRST. Build the ideal component before any predictor: it bounds every possible
implementation at once, so a training budget is only ever spent against a measured ceiling. The
same discipline killed the expert prediction head in the PROTECTION role -- a perfect oracle there
was worth 0.4 pp, so no head could be worth more.

THREE THINGS THIS MUST MODEL, each of which was a real error in this project before:

  * SPECULATIVE READS CONTEND WITH DEMAND READS. A prefetch for layer L+1 and a demand miss on
    layer L want the same device. They go down the same queue and take the same `nvme_qd` permits;
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


@dataclasses.dataclass
class PrefetchStats:
    issued: int = 0          # speculative reads submitted
    used: int = 0            # prefetched keys that a later demand found resident
    wasted: int = 0          # prefetched keys evicted or cancelled without ever being demanded
    cancelled: int = 0       # still-pending speculation dropped by discard_wrong_asap
    refused: int = 0         # predictions the store had no free slot for

    @property
    def precision(self) -> float:
        return self.used / self.issued if self.issued else 0.0


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
        full = super().predict(layer, uniq, step)
        if self.recall < 1.0:
            n = int(len(full) * self.recall)
            # deterministic, key-stable subset: rank by a cheap hash of the key
            full = tuple(sorted(full, key=lambda k: (k[0] * 2654435761 + k[1] * 40503) & 0xFFFF)[:n])
        if self.precision >= 1.0 or not full:
            return tuple(full)
        # keep |full| correct picks and pad with wrong ones until the ratio holds, so the arm's
        # ISSUE COUNT reflects what a predictor of this precision would really put on the device.
        n_wrong = int(len(full) * (1.0 - self.precision) / self.precision)
        true_set = set(full)
        wrong, e, guard = [], 0, 0
        while len(wrong) < n_wrong and guard < 10 * self.n_experts:
            guard += 1
            fl = full[e % len(full)][0]
            cand = (fl, (step * 7919 + layer * 104729 + guard * 31) % self.n_experts)
            if cand not in true_set:
                wrong.append(cand)
                true_set.add(cand)
            e += 1
        return tuple(full) + tuple(wrong)
