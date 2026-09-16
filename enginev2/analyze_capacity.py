"""Are the misses COMPULSORY (never seen before) or CAPACITY (seen, evicted, wanted again)?

If the working set converges but misses persist, the cache is smaller than the working set and the
misses are re-references. That is a capacity problem, and prediction is the wrong tool for it --
you cannot predict your way out of a cache 4x too small. Measured, not assumed.
"""
import json, sys

for path in sys.argv[1:]:
    seen = set()
    comp = cap = 0
    per_layer_ws = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("pf"):
            continue
        L = r["L"]
        mi = r.get("miss")
        miss = mi if isinstance(mi, list) else []
        for e in miss:
            (cap if (L, e) in seen else comp).__class__     # no-op, keeps the branch explicit
            if (L, e) in seen:
                cap += 1
            else:
                comp += 1
        for e in r["uniq"]:
            seen.add((L, e))
            per_layer_ws.setdefault(L, set()).add(e)
    tot = comp + cap
    if not tot:
        print(f"{path.split('/')[-1]}: no miss ids recorded, skipped")
        continue
    ws = sum(len(v) for v in per_layer_ws.values())
    print(f"\n{path.split('/')[-1]}")
    print(f"  misses: {cap / tot * 100:.1f}% CAPACITY (seen before, evicted), "
          f"{comp / tot * 100:.1f}% compulsory (first touch)")
    print(f"  working set {ws} (layer,expert) pairs; arena at 40 GB holds 2767 "
          f"= {2767 / ws * 100:.0f}% of it")
    pl = sorted(len(v) for v in per_layer_ws.values())
    print(f"  per layer: median {pl[len(pl)//2]} distinct experts of 384, "
          f"min {pl[0]}, max {pl[-1]}")
