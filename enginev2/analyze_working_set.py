"""Is the expert working set converging, or does it keep growing?

If a same-topic run saturates the arena with its own experts, the miss rate decays toward zero and
long agentic work is a different regime from the short prompts measured so far. If the set of
distinct (layer, expert) pairs is still climbing at the end of the trace, it does not, and the
short-prompt finding is not an artifact of length.
"""
import json, sys, collections

for path in sys.argv[1:]:
    seen = set()
    step = -1
    per_step_new, per_step_miss = [], []
    new_this, miss_this, layers_this = 0, 0, 0
    last_L = -1
    for line in open(path):
        r = json.loads(line)
        if r.get("pf"):
            continue
        L = r["L"]
        if L <= last_L:
            if layers_this:
                per_step_new.append(new_this)
                per_step_miss.append(miss_this)
            new_this = miss_this = layers_this = 0
        last_L = L
        layers_this += 1
        mi = r.get("miss")
        miss_this += len(mi) if isinstance(mi, list) else (mi or 0)
        for e in r["uniq"]:
            k = (L, e)
            if k not in seen:
                seen.add(k)
                new_this += 1
    n = len(per_step_new)
    if n < 40:
        print(f"{path}: only {n} decode steps, skipped")
        continue
    def win(a, lo, hi):
        s = a[lo:hi]
        return sum(s) / max(1, len(s))
    q = n // 4
    print(f"\n{path.split('/')[-1]}  {n} decode steps, {len(seen)} distinct (layer,expert) pairs "
          f"of 15360 ({len(seen)/15360*100:.1f}%)")
    print(f"  {'quarter':<10}{'new pairs/step':>16}{'misses/step':>14}")
    for i in range(4):
        lo, hi = i * q, (i + 1) * q
        print(f"  Q{i+1:<9}{win(per_step_new, lo, hi):16.1f}{win(per_step_miss, lo, hi):14.1f}")
    print(f"  last 50 steps: {win(per_step_new, n-50, n):.1f} new pairs/step, "
          f"{win(per_step_miss, n-50, n):.1f} misses/step")
