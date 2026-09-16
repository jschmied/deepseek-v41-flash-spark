"""Offline: can anything predict the NEXT layer's CACHE MISSES well enough to be worth a read?

The bar is not "reproduce layer L+1's route". Job 295 measured horizon-1 oracle at +37.8 % with
ready_hit 96 against late_hit 2292 -- almost nothing arrived finished. What the oracle bought was
STARTING correct reads before the layer boundary, so the queue never drains. With ~2 misses per
layer, finding ONE is enough to bridge the gap: at per-miss recall r the chance of starting at
least one is 1-(1-r)^2, so r=0.30 already covers 51 % of layers.

So this scores predictors on the two numbers that decide the economics, not on route accuracy:

  future-miss recall   of the experts that WOULD have missed at L+1, how many did we name
  fetch precision      of the reads we actually caused, how many were needed

The second is the one that killed the earlier transition-prefetch work: a predictor can be 70 %
right about route IDs and still terrible about reads, because correct predictions are usually
ALREADY RESIDENT and cost nothing while wrong ones are nonresident and always cost a read.
Candidates are therefore filtered to nonresident BEFORE ranking, and only the top-K survive.

RESIDENCY IS THE ENGINE'S OWN. This replays through enginev2.store.ExpertSlots rather than a
private LRU, so the ground truth is the code the engine actually runs.

TRAIN/TEST SPLIT is real: the transition table is built on the first 60 % of steps and evaluated on
the last 40 %. Fitting and scoring on one trace is how a table that recalls 30 % looks like 60 %.

Run:  python enginev2/eval_miss_predictors.py [trace.jsonl]
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enginev2.store import ExpertSlots              # noqa: E402

TRACE = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/ds41-queue/logs/route-decode.jsonl")
N_EXPERT, N_LAYER = 384, 40
# THE CACHE MUST BE THE ONE THE TRACE WAS TAKEN WITH. Sized at 2759 (today's 40 GB arena) this
# replay produced ~3.9 misses per layer while the trace itself records 1.71 -- a denominator wrong
# by 2.3x, which silently deflates every recall number. phase 1 reproduces this trace exactly at
# lru 5328 / transient 400 (64.6 fetches per step = 1.615 per layer), so that is the size, and the
# harness now CHECKS itself against the trace's own recorded miss count instead of assuming.
LRU_SLOTS = int(os.environ.get("LRU_SLOTS", 5328))
TRANSIENT = int(os.environ.get("TRANSIENT", 400))
EPS = 1e-6


def load(path):
    """`i` is a per-CALL counter, not a step: every row is one (call, layer) with its own i, and L
    cycles 0..39. So a step is a run of rows whose L increases; it ends when L wraps."""
    steps, cur, last = [], [], -1
    traced_miss = traced_rows = 0
    for line in open(path):
        r = json.loads(line)
        if r.get("pf"):
            continue                                 # decode only; prefill has a different shape
        L = r["L"]
        if L <= last:                                # wrapped: the previous step is complete
            if len(cur) == N_LAYER:
                steps.append(cur)
            cur = []
        cur.append(r["uniq"])
        traced_miss += len(r.get("miss") or [])
        traced_rows += 1
        last = L
    if len(cur) == N_LAYER:
        steps.append(cur)
    return steps, traced_miss / max(1, traced_rows)


class Popularity:
    """The baseline that must be beaten. Static per-layer frequency, nothing conditional."""
    name = "popularity"

    def __init__(self, train):
        self.c = np.zeros((N_LAYER, N_EXPERT), dtype=np.float64)
        for st in train:
            for L, u in enumerate(st):
                self.c[L, u] += 1
        self.c /= max(1, len(train))

    def score(self, layer, active, target):
        return self.c[target]


class Transition:
    """P(b at L+1 | a at L) / P(b at L+1), summed in log space over the active set.

    Counts, not a learned model -- this is the existing table, rebuilt here only so the split is
    honest and so it is scored on MISSES rather than on route membership.
    """
    name = "transition"

    def __init__(self, train, horizon=1):
        self.h = horizon
        self.joint = np.zeros((N_LAYER, N_EXPERT, N_EXPERT), dtype=np.float32)
        self.src = np.zeros((N_LAYER, N_EXPERT), dtype=np.float64)
        self.tgt = np.zeros((N_LAYER, N_EXPERT), dtype=np.float64)
        n = 0
        for st in train:
            for L in range(N_LAYER - horizon):
                a, b = st[L], st[L + horizon]
                self.src[L, a] += 1
                self.tgt[L, b] += 1
                self.joint[L][np.ix_(a, b)] += 1
                n += 1
        self.n = max(1, n)
        self.tgt_p = self.tgt / self.n * N_LAYER
        with np.errstate(divide="ignore", invalid="ignore"):
            self.cond = self.joint / np.maximum(self.src[:, :, None], 1.0)

    def score(self, layer, active, target):
        L = layer
        if L >= N_LAYER - self.h:
            return None
        prior = self.tgt_p[L] + EPS
        s = np.log(self.cond[L][active] + EPS) - np.log(prior)[None, :]
        return s.sum(axis=0)


class PrevStepMiss:
    """Did this expert miss at the SAME layer in the previous step?

    Not a co-occurrence model at all -- it asks whether the residual misses are temporally sticky.
    If they are, the cheapest possible predictor works and nothing needs training. If they are not,
    that says the misses are the cold tail, which is what LRU leaves behind by construction, and it
    reframes what any learned model would have to do.
    """
    name = "prev-miss"

    def __init__(self, train):
        self.last = {}                               # layer -> set of experts that missed there

    def observe(self, layer, missed):
        self.last[layer] = set(missed)

    def score(self, layer, active, target):
        sc = np.zeros(N_EXPERT, dtype=np.float64)
        prev = self.last.get(target)
        if prev:
            sc[list(prev)] = 10.0
        return sc


def evaluate(pred, test, topk, thresh, warm_steps=40):
    sl = ExpertSlots(lru_slots=LRU_SLOTS, transient_slots=TRANSIENT, policy="lru")

    def touch(layer, uniq):
        slot_of, to_load, to_wait = sl.reserve(layer, tuple(uniq), prefill=False)
        for _k, s, g in to_load:
            sl.clear_pending(s, g)                   # the read completes; this is an offline replay
        return {k for k, _s, _g in to_load}

    for st in test[:warm_steps]:                     # warm the cache; not scored
        for L, u in enumerate(st):
            m = touch(L, u)
            if hasattr(pred, "observe"):
                pred.observe(L, [k[1] for k in m])

    hit = issued = need = 0
    wrong_reads = 0
    pending = {}                                     # target layer -> [(key, was_nonresident)]
    for st in test[warm_steps:]:
        for L in range(N_LAYER):
            uniq = st[L]
            # score the predictions made for THIS layer, before the demand touch makes them moot
            for key, _ in pending.pop(L, []):
                if key[1] in uniq:
                    hit += 1
                else:
                    wrong_reads += 1
            missed_here = touch(L, uniq)
            if hasattr(pred, "observe"):
                pred.observe(L, [k[1] for k in missed_here])

            if L + 1 >= N_LAYER:
                continue
            nxt = st[L + 1]
            resident = set(sl.lru) | set(sl.transient_map)
            # the DENOMINATOR of recall: experts L+1 will want that are not here yet
            future_miss = [e for e in nxt if (L + 1, e) not in resident]
            need += len(future_miss)
            if not future_miss:
                continue

            sc = pred.score(L, np.asarray(uniq, dtype=np.int64), L + 1)
            if sc is None:
                continue
            # FILTER TO NONRESIDENT BEFORE RANKING. Ranking first and filtering after is what
            # spends the whole budget re-fetching things already in cache.
            cand = np.array([e for e in range(N_EXPERT) if (L + 1, e) not in resident],
                            dtype=np.int64)
            if not len(cand):
                continue
            cs = sc[cand]
            order = np.argsort(-cs)[:topk]
            chosen = [(int(cand[i]), float(cs[i])) for i in order if cs[i] >= thresh]
            if not chosen:
                continue
            issued += len(chosen)
            pending[L + 1] = [((L + 1, e), True) for e, _ in chosen]
            for e, _ in chosen:
                sp, _ref = sl.reserve_speculative([(L + 1, e)])
                for _k, s, g in sp:
                    sl.clear_pending(s, g)
    scored_layers = max(1, (len(test) - warm_steps) * N_LAYER)
    return dict(recall=hit / max(1, need), precision=hit / max(1, issued),
                issued=issued, hit=hit, need=need,
                miss_per_layer=need / scored_layers,
                wrong_per_layer=wrong_reads / scored_layers)


DSPARK = "--dspark" in sys.argv
if not DSPARK:
    steps, traced_miss_per_layer = load(TRACE)
    cut = int(len(steps) * 0.6)
    train, test = steps[:cut], steps[cut:]
    print(f"  {os.path.basename(TRACE)}: {len(steps)} decode steps, "
          f"train {len(train)} / test {len(test)}, ~{np.mean([len(u) for u in steps[0]]):.1f} active/layer")

    first = True
    for P in (Popularity, Transition, PrevStepMiss):
        p = P(train)
        for topk in (1, 2):
            for thresh in (-1e18, 4.0, 8.0, 16.0, 24.0):
                r = evaluate(p, test, topk, thresh)
                t = "none" if thresh < -1e17 else f"{thresh:.0f}"
                if first:
                    first = False
                    # SELF-CHECK: the replay's own miss rate against the trace's recorded one. If these
                    # disagree the residency model is wrong and every recall below is against the wrong
                    # denominator -- which is exactly what happened at 2759 slots.
                    d = abs(r["miss_per_layer"] - traced_miss_per_layer) / max(1e-9, traced_miss_per_layer)
                    flag = "OK" if d < 0.15 else "MISMATCH -- recall below is not trustworthy"
                    print(f"  replay {r['miss_per_layer']:.2f} misses/layer vs trace {traced_miss_per_layer:.2f}"
                          f"  ({d * 100:.0f}% apart) {flag}")
                print(f"  {p.name:11s} top-{topk} thr {t:>4s}  "
                      f"miss-recall {r['recall'] * 100:5.1f}%  fetch-prec {r['precision'] * 100:5.1f}%  "
                      f"issued {r['issued']:6d}  wrong/layer {r['wrong_per_layer']:.2f}")


# --------------------------------------------------------------------------- DSpark
def eval_dspark(path):
    """Score the DSpark drafter's routing against the backbone's per-layer MISS set.

    Two questions, in order, because the second is only interesting if the first says yes:

      1. MUTUAL INFORMATION, not a model. For each backbone layer, does knowing which drafter
         experts fired change the distribution over which backbone experts miss? Measured as the
         lift of a count-based conditional over the per-layer miss prior -- the same estimator the
         transition table used, so the numbers are comparable to 8419ade's.
      2. Only if there is lift: build a predictor and measure fetch precision.

    Trained on the first 60 % of steps and scored on the last 40 %, as before.

    The drafter's vocabulary is its own 3 x 128, disjoint from the backbone's 384, so this learns
    the cross-vocabulary mapping from counts rather than assuming one exists.
    """
    rows = [json.loads(l) for l in open(path)]
    rows = rows[50:]                                  # drop the cold cache at the head of the run
    cut = int(len(rows) * 0.6)
    train, test = rows[:cut], rows[cut:]
    D = 3 * 128

    def dfeat(r):
        out = set()
        for k, layers in enumerate(r["d_idx"]):
            for posn in layers:
                for e in posn:
                    out.add(k * 128 + int(e))
        return sorted(out)

    joint = np.zeros((N_LAYER, D, N_EXPERT), dtype=np.float32)
    src = np.zeros((N_LAYER, D), dtype=np.float64)
    prior = np.zeros((N_LAYER, N_EXPERT), dtype=np.float64)
    nstep = np.zeros(N_LAYER, dtype=np.float64)
    for r in train:
        f = dfeat(r)
        for lay in r["layers"]:
            L, miss = lay["L"], lay.get("miss_ids")
            if miss is None:
                continue                              # older traces recorded only the miss COUNT
            joint[L][np.ix_(f, miss)] += 1
            src[L, f] += 1
            prior[L, miss] += 1
            nstep[L] += 1
    if nstep.sum() == 0:
        print("  the trace records miss COUNTS but not miss IDS; re-capture with miss_ids to score "
              "DSpark. Nothing else in this function can run.")
        return
    cond = joint / np.maximum(src[:, :, None], 1.0)
    pri = prior / np.maximum(nstep[:, None], 1.0)

    hit = issued = need = 0
    for r in test:
        f = np.asarray(dfeat(r), dtype=np.int64)
        for lay in r["layers"]:
            L, miss = lay["L"], set(lay.get("miss_ids") or [])
            need += len(miss)
            if not miss or not len(f):
                continue
            s = (np.log(cond[L][f] + EPS) - np.log(pri[L] + EPS)[None, :]).sum(axis=0)
            top = int(np.argmax(s))
            issued += 1
            if top in miss:
                hit += 1
    print(f"  dspark->miss  top-1  miss-recall {hit / max(1, need) * 100:5.1f}%  "
          f"fetch-prec {hit / max(1, issued) * 100:5.1f}%  issued {issued}  "
          f"(baseline: popularity was 6.2 % precision on the route trace)")


if DSPARK:
    eval_dspark(sys.argv[sys.argv.index("--dspark") + 1])
