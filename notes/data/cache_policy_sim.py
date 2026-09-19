# Layer-aware cache policies and a two-hit doorkeeper, at ONE equal byte budget.
#
# Why these two together. The host-mapped cold path came out performance-neutral because it moves the
# SAME bytes as the baseline: both do NVMe -> pinned and pinned -> device for every miss. It relocated
# a copy rather than removing one. So the next question is not how to schedule those bytes but which
# bytes to move at all -- which is a cache-policy question, and it is free to answer offline.
#
# Every arm is charged both movements explicitly:
#   nvme  bytes read from the SSD
#   h2d   bytes copied from pinned memory into the device arena (an ordinary load's H2D, or a
#         promotion -- they are the same 13.77 MB either way)
#
# Today's engine and today's cold path are BOTH "every miss pays nvme + h2d". A doorkeeper breaks that
# link: an expert used once executes from the mapped pool and is never promoted, so it pays nvme only.
import collections, json, sys

REC = 13_774_848
SLOT = 14_454_784                 # device bytes per resident expert (unpacked scales)
BUDGET_SLOTS = 5421               # job 1005's measured auto-sized arena
TRACE = sys.argv[1] if len(sys.argv) > 1 else "/home/jschmied/ds41-queue/payloads/routes-940.json"

raw = json.load(open(TRACE))
ev = []
for k, experts in raw.items():
    st, L = k.split(":")          # step:layer
    ev.append((int(st), int(L), [int(e) for e in experts]))
ev.sort()
LAYERS = sorted({L for _, L, _ in ev})
NL = len(LAYERS)
acc_total = sum(len(e) for _, _, e in ev)
print(f"  {TRACE.split('/')[-1]}: {len(ev)} (step,layer) events, {acc_total:,} accesses, "
      f"{NL} layers, {len({(L,e) for _,L,es in ev for e in es}):,} distinct experts")
print(f"  budget {BUDGET_SLOTS} slots = {BUDGET_SLOTS*SLOT/2**30:.1f} GiB\n")


def lru_misses(seq, cap):
    """seq: list of lists of expert ids (one per access group). Returns miss count."""
    cache, miss = collections.OrderedDict(), 0
    for group in seq:
        for e in group:
            if e in cache:
                cache.move_to_end(e)
            else:
                miss += 1
                cache[e] = 1
                if len(cache) > cap:
                    cache.popitem(last=False)
    return miss


# ---------------------------------------------------------------- A: global LRU
def arm_global(cap):
    cache, nvme, h2d, hits = collections.OrderedDict(), 0, 0, 0
    for _, L, es in ev:
        for e in es:
            k = (L, e)
            if k in cache:
                cache.move_to_end(k); hits += 1
            else:
                nvme += REC; h2d += REC
                cache[k] = 1
                if len(cache) > cap:
                    cache.popitem(last=False)
    return dict(nvme=nvme, h2d=h2d, hits=hits)


# ---------------------------------------------------------------- B/C: per-layer LRU
def arm_per_layer(quota):
    caches = {L: collections.OrderedDict() for L in LAYERS}
    nvme = h2d = hits = 0
    for _, L, es in ev:
        c, cap = caches[L], quota[L]
        for e in es:
            if e in c:
                c.move_to_end(e); hits += 1
            else:
                nvme += REC; h2d += REC
                if cap > 0:
                    c[e] = 1
                    if len(c) > cap:
                        c.popitem(last=False)
    return dict(nvme=nvme, h2d=h2d, hits=hits)


# ---------------------------------------------------------------- D: two-hit doorkeeper + victim pool
def arm_doorkeeper(hot_cap, pool_cap, promote_on=2):
    """Mapped pool holds recently-read records; an expert is promoted to the device arena only after
    `promote_on` uses. A single-use expert therefore pays nvme and NEVER pays h2d."""
    hot, pool = collections.OrderedDict(), collections.OrderedDict()
    seen = collections.Counter()
    nvme = h2d = hot_hits = pool_hits = mapped_exec = 0
    for _, L, es in ev:
        for e in es:
            k = (L, e)
            if k in hot:
                hot.move_to_end(k); hot_hits += 1
                continue
            if k in pool:
                pool.move_to_end(k); pool_hits += 1
            else:
                nvme += REC
                pool[k] = 1
                if len(pool) > pool_cap:
                    pool.popitem(last=False)
            mapped_exec += 1
            seen[k] += 1
            if seen[k] >= promote_on:
                h2d += REC                      # the promotion, paid once, only for reused experts
                hot[k] = 1
                pool.pop(k, None)
                if len(hot) > hot_cap:
                    hot.popitem(last=False)
    return dict(nvme=nvme, h2d=h2d, hits=hot_hits, pool_hits=pool_hits, mapped=mapped_exec)


# ---------------------------------------------------------------- E: phase-aware protection
def arm_phase(cap, protect):
    """Global LRU, but a victim whose layer is within `protect` layers AHEAD of the current one is
    skipped: it will be wanted again almost immediately, while a layer just behind has a full
    traversal to wait."""
    cache, nvme, h2d, hits = collections.OrderedDict(), 0, 0, 0
    for _, L, es in ev:
        for e in es:
            k = (L, e)
            if k in cache:
                cache.move_to_end(k); hits += 1
                continue
            nvme += REC; h2d += REC
            cache[k] = 1
            while len(cache) > cap:
                for vk in list(cache.keys()):
                    if (vk[0] - L) % NL > protect:
                        cache.pop(vk); break
                else:
                    cache.popitem(last=False)
    return dict(nvme=nvme, h2d=h2d, hits=hits)


def show(name, r, base=None):
    tot = r["nvme"] + r["h2d"]
    extra = ""
    if base:
        b = base["nvme"] + base["h2d"]
        extra = (f"   nvme {r['nvme']/base['nvme']-1:+6.1%}  h2d {r['h2d']/base['h2d']-1:+6.1%}  "
                 f"TOTAL {tot/b-1:+6.1%}")
    hr = r["hits"] / acc_total
    ph = f" pool_hits {r['pool_hits']:,}" if "pool_hits" in r else ""
    print(f"  {name:46s} nvme {r['nvme']/1e9:7.1f} GB  h2d {r['h2d']/1e9:7.1f} GB  "
          f"hot_hit {hr:5.1%}{ph}{extra}")
    return r


base = show("A global LRU (today)", arm_global(BUDGET_SLOTS))
print()
eq = {L: BUDGET_SLOTS // NL for L in LAYERS}
show(f"B per-layer LRU, equal quota ({eq[LAYERS[0]]}/layer)", arm_per_layer(eq), base)

# C: marginal-gain allocation from each layer's own miss curve
per_layer_seq = collections.defaultdict(list)
for _, L, es in ev:
    per_layer_seq[L].append(es)
maxd = {L: len({e for g in per_layer_seq[L] for e in g}) for L in LAYERS}
curve = {L: [lru_misses(per_layer_seq[L], c) for c in range(0, maxd[L] + 1)] for L in LAYERS}
import heapq
quota = {L: 0 for L in LAYERS}
heap = [(-(curve[L][0] - curve[L][1]), L) for L in LAYERS if maxd[L] >= 1]
heapq.heapify(heap)
for _ in range(BUDGET_SLOTS):
    if not heap:
        break
    g, L = heapq.heappop(heap)
    quota[L] += 1
    if quota[L] < maxd[L]:
        heapq.heappush(heap, (-(curve[L][quota[L]] - curve[L][quota[L] + 1]), L))
show(f"C per-layer LRU, marginal-gain quota ({min(quota.values())}-{max(quota.values())}/layer)",
     arm_per_layer(quota), base)
for p in (1, 2, 4, 8):
    show(f"E global LRU, protect next {p} layer(s)", arm_phase(BUDGET_SLOTS, p), base)
print()
for pool in (64, 128, 256):
    hot = BUDGET_SLOTS - int(pool * REC / SLOT)      # the mapped pool costs budget too
    for thr in (2, 3):
        show(f"D doorkeeper pool {pool} promote-on-{thr} (hot {hot})",
             arm_doorkeeper(hot, pool, thr), base)
