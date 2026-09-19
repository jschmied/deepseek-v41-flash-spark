# IDEA 2, offline: does CONTINUOUS stripe residency beat whole-expert LRU at the SAME byte budget?
#
# Exact decomposition, no quantization or router argument:
#   y = sum_j W2[:,J_j] ( silu(W1[J_j,:] x) * W3[J_j,:] x )
# so stripe j needs W1[J_j,:], W3[J_j,:] AND W2[:,J_j].
#
# THE LAYOUT IS THE WHOLE QUESTION, and the idea as posed assumes it away:
#   L1 today      W1/W3 are [INTER, DIM] so a stripe is contiguous; W2 is [DIM, INTER] so a stripe is
#                 a column slice over 5120 rows of 576 B -- at 4096 B pages every page is touched, so
#                 W2 cannot be partially fetched at all. Stripe j costs (1/S)(w1+w3) + ALL of w2.
#   L2 repacked   W2 stored INTER-major. Fully linear: stripe j costs (1/S)(w1+w3+w2). Needs a
#                 transposed down-kernel, so this is the "if we do the work" bound.
#   L3 w2-pinned  every expert's W2 permanently resident, only W1/W3 stripes stream.
#
# Baseline is whole-expert LRU at the same budget. Assignment policies:
#   lru        stripe-level LRU, fully causal
#   causal     stripe count per expert from frequency over the FIRST 20 % of the trace, evaluated on
#              the remaining 80 % -- realizable
#   oracle     stripe count from frequency over the WHOLE trace -- an upper bound, labelled as one
import json, sys, collections, heapq

TRACE = sys.argv[1] if len(sys.argv) > 1 else "/home/jschmied/ds41-queue/payloads/routes-940.json"
W13, W2 = 9_165_312, 4_608_000          # bytes, from the shipped manifest
REC = W13 + W2
SLOT = 14_454_784                        # device bytes per whole expert (unpacked scales)
S = 8                                    # stripes
N_SLOTS = 5421                           # job 1005's measured auto-sized FP4 arena
BUDGET = N_SLOTS * SLOT

raw = json.load(open(TRACE))
ev = []
for k, experts in raw.items():
    # KEY IS "step:layer". field0 spans 0..99 (100 steps), field1 spans 0..39 (40 layers) --
    # getting this backwards keyed the cache by (step, expert) and destroyed all cross-step reuse,
    # which is how the first two runs of this simulation produced 35,184 distinct experts.
    st, L = k.split(":")
    ev.append((int(st), int(L), [int(e) for e in experts]))
ev.sort()
print(f"  trace {TRACE.split('/')[-1]}: {len(ev)} (layer,step) events, "
      f"{sum(len(e[2]) for e in ev)} accesses, budget {BUDGET/1e9:.1f} GB = {N_SLOTS} whole slots")


def fetch_cost(layout, n_new_stripes, w2_missing):
    """Disk bytes to make n_new_stripes computable."""
    if layout == "L1":
        return n_new_stripes * (W13 / S) + (W2 if w2_missing else 0)
    if layout == "L2":
        return n_new_stripes * (REC / S)
    if layout == "L3":
        # W2 is pinned only for experts the budget could afford; every OTHER expert must still fetch
        # its W2, and in the L3 layout that is the un-striped whole plane. Getting this wrong is what
        # made the first run of this simulation report -35.8 % for L3.
        return n_new_stripes * (W13 / S) + (W2 if w2_missing else 0)
    raise ValueError(layout)


def dev_cost(layout, stripes, w2_res):
    if layout == "L3":
        return stripes * (SLOT * (W13 / REC) / S) + SLOT * (W2 / REC)
    return stripes * (SLOT / S) + (SLOT * (W2 / REC) if (layout == "L1" and w2_res) else 0)


def run_whole_lru():
    """Baseline: whole experts, LRU, N_SLOTS of them."""
    cache, clock, fetched = collections.OrderedDict(), 0, 0
    for st, L, experts in ev:
        for e in experts:
            key = (L, e)
            if key in cache:
                cache.move_to_end(key)
            else:
                fetched += REC
                cache[key] = clock
                if len(cache) > N_SLOTS:
                    cache.popitem(last=False)
            clock += 1
    return fetched


def run_stripe(layout, mode, split=0.2):
    """Stripe residency. `mode` in {lru, causal, oracle}."""
    if mode in ("causal", "oracle"):
        cut = int(len(ev) * split) if mode == "causal" else len(ev)
        freq = collections.Counter()
        for st, L, experts in ev[:cut]:
            for e in experts:
                freq[(L, e)] += 1
        # give the hottest experts all S stripes, then descend; fill to the budget
        order = [k for k, _ in freq.most_common()]
        want, spent = {}, 0
        for k in order:
            step = dev_cost(layout, S, True) if layout != "L3" else dev_cost(layout, S, True)
            if spent + step <= BUDGET:
                want[k] = S; spent += step
            else:
                break
        # remaining budget in quarter-experts for the next tier
        for k in order[len(want):]:
            step = dev_cost(layout, S // 4, True)
            if spent + step <= BUDGET:
                want[k] = S // 4; spent += step
            else:
                break
        resident = {k: v for k, v in want.items()}
        start = cut if mode == "causal" else 0
        fetched = 0
        for st, L, experts in ev[start:]:
            for e in experts:
                have = resident.get((L, e), 0)
                if have < S:
                    # w2 is already there only if this expert got residency at all (L1/L2), or if L3
                    # pinned it -- which L3 can only do for experts inside the budget.
                    fetched += fetch_cost(layout, S - have, w2_missing=((L, e) not in resident))
        return fetched, len(want), start
    # stripe-level LRU: every access needs all S stripes, evict whole experts by LRU but count
    # partial credit is meaningless when every access needs the full expert -- so this degenerates to
    # whole-expert LRU. Report it to make that explicit rather than hiding it.
    return run_whole_lru(), N_SLOTS, 0


EXPERTS_TOTAL = len({(L, e) for _, L, es in ev for e in es})
w2_dev = SLOT * (W2 / REC)
print(f"  distinct (layer,expert) in trace: {EXPERTS_TOTAL}; pinning W2 for ALL of them costs "
      f"{EXPERTS_TOTAL * w2_dev / 1e9:.1f} GB of the {BUDGET/1e9:.1f} GB budget "
      f"({EXPERTS_TOTAL * w2_dev / BUDGET:.0%})")

base = run_whole_lru()
print(f"\n  BASELINE whole-expert LRU, {N_SLOTS} slots: {base/1e9:.1f} GB fetched")
print(f"  (the engine measured 259.0 GB on five prompts; this trace is a different length, so use"
      f" RATIOS below, never the absolute)\n")
for layout in ("L1", "L2", "L3"):
    for mode in ("causal", "oracle"):
        f, nres, start = run_stripe(layout, mode)
        # causal evaluates only the tail, so rescale the baseline to the same tail
        if start:
            sub = ev[start:]
            cache, clock, b2 = collections.OrderedDict(), 0, 0
            for st, L, experts in sub:
                for e in experts:
                    key = (L, e)
                    if key in cache: cache.move_to_end(key)
                    else:
                        b2 += REC; cache[key] = clock
                        if len(cache) > N_SLOTS: cache.popitem(last=False)
                    clock += 1
            ref = b2
        else:
            ref = base
        print(f"  {layout} {mode:7s}: {f/1e9:7.1f} GB vs LRU {ref/1e9:7.1f} GB  "
              f"-> {f/ref-1:+7.1%}   ({nres} experts given residency)")
print("\n  NOTE: stripe-LRU is omitted because every ACCESS needs all S stripes -- partial residency")
print("  only helps if some stripes are never needed, which exact FFN evaluation never permits.")
