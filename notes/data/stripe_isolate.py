# IDEA 2, ISOLATED: at ONE fixed byte budget, is it better to hold N experts at 100 % or kN at 1/k?
#
# The previous run did not test this at all: its budget was exhausted by fully-resident experts before
# any partial tier was reached, so it compared frequency-static caching against LRU and said nothing
# about stripes. This compares only the residency FRACTION, holding the budget and the ranking fixed.
#
# The structural point the sweep has to respect: exact FFN evaluation needs EVERY stripe of a routed
# expert, so a partially resident expert pays (1-f) of its bytes on EVERY access -- it never gets a
# free hit. Striping therefore trades "some experts always free" for "more experts always cheaper",
# and which wins is decided entirely by how skewed the access distribution is.
import json, collections, sys

W13, W2 = 9_165_312, 4_608_000
REC = W13 + W2
SLOT = 14_454_784
BUDGET = 5421 * SLOT

raw = json.load(open("/home/jschmied/ds41-queue/payloads/routes-940.json"))
ev = []
for k, experts in raw.items():
    st, L = k.split(":")                      # step:layer
    ev.append((int(st), int(L), [int(e) for e in experts]))
ev.sort()
acc = collections.Counter()
for _, L, es in ev:
    for e in es:
        acc[(L, e)] += 1
total_acc = sum(acc.values())
print(f"  {len(ev)} events, {total_acc} accesses, {len(acc)} distinct experts, "
      f"budget {BUDGET/1e9:.1f} GB\n")
sk = sorted(acc.values(), reverse=True)
top = sum(sk[:len(sk)//10]) / total_acc
print(f"  skew: the hottest 10 % of experts take {top:.1%} of accesses\n")


def lru(n_slots, events):
    cache, fetched = collections.OrderedDict(), 0
    for _, L, es in events:
        for e in es:
            k = (L, e)
            if k in cache:
                cache.move_to_end(k)
            else:
                fetched += REC
                cache[k] = 1
                if len(cache) > n_slots:
                    cache.popitem(last=False)
    return fetched


def uniform_frac(f, rank, events, layout):
    """Give the top experts a residency fraction f, as many as the budget allows.
    layout L1: W2 cannot be partially fetched, so a partial expert still pays all of W2 each access.
    layout L2: W2 repacked, everything linear."""
    per = SLOT * f
    n = int(BUDGET // per)
    res = set(rank[:n])
    fetched = 0
    for _, L, es in events:
        for e in es:
            k = (L, e)
            if k in res:
                if f >= 1.0:
                    continue
                fetched += (1 - f) * (W13 if layout == "L1" else REC)
                if layout == "L1":
                    fetched += W2          # column stripes touch every page of W2
            else:
                fetched += REC
    return fetched, n


base = lru(5421, ev)
print(f"  BASELINE whole-expert LRU @5421 slots: {base/1e9:.1f} GB "
      f"(engine measured 259.0 GB -- calibrated)\n")
# ORACLE ranking (upper bound, labelled) and CAUSAL ranking (first 20 %, realizable)
cut = int(len(ev) * 0.2)
warm = collections.Counter()
for _, L, es in ev[:cut]:
    for e in es:
        warm[(L, e)] += 1
ranks = {"oracle": [k for k, _ in acc.most_common()],
         "causal": [k for k, _ in warm.most_common()]}
for layout in ("L1", "L2"):
    print(f"  --- {layout} ({'today, W2 unstripeable' if layout=='L1' else 'W2 repacked, linear'}) ---")
    for rname, rank in ranks.items():
        evs = ev if rname == "oracle" else ev[cut:]
        ref = base if rname == "oracle" else lru(5421, evs)
        row = []
        for f in (1.0, 0.5, 0.25, 0.125):
            got, n = uniform_frac(f, rank, evs, layout)
            row.append(f"f={f:<5} n={n:<6} {got/1e9:7.1f} GB ({got/ref-1:+6.1%})")
        print(f"   {rname:7s} vs LRU {ref/1e9:7.1f} GB")
        for r in row:
            print(f"     {r}")
