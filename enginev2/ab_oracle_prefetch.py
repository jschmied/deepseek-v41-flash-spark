"""The oracle prefetch ceiling, measured in the REAL engine instead of the model.

The model said a perfect oracle is worth +81-84 %; the read-depth trace said filling the 44.3 % of
the span with no read in flight is worth +83 %. Both are inferences. This runs it: pass 1 records
the true routes, pass 2 replays the identical step sequence with a prefetcher that knows layer
L+horizon's experts while layer L is computing.

That is a CEILING, not a proposal -- no predictor has this information. What it bounds is every
predictor at once, in the real engine, with real reads, real slots and real evictions, including
the ones the model cannot charge for: a prefetch still takes a slot, still evicts something, and
still competes for the same device.

HORIZON=n layers ahead (default 1). ARM=null runs the same harness with prediction off, which is
the baseline the ceiling is measured against -- and it is physically realizable, unlike an arm that
is handed compute before the misses are knowable.

Run:  ARM=oracle HORIZON=1 DSV41_CB3_CACHE=... python enginev2/ab_oracle_prefetch.py [steps]
"""
import os, sys, time, collections, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = os.path.expanduser("~/git/deepseek-v41-flash-spark")
sys.path[:0] = [V1, os.path.join(V1, "tools")]
os.chdir(V1)

from engine.v41_engine import V41Engine              # noqa: E402
from enginev2.observe import TraceObserver           # noqa: E402
from enginev2.prefetch import Prefetcher             # noqa: E402
from enginev2.real import RealEngramSource, RealLeaves   # noqa: E402
from enginev2.sched import Policy                    # noqa: E402

# ONE PASS PER PROCESS. The in-process rewind (c.rollback(0) + begin_prompt) restores decode
# state exactly at 8 steps and NOT at 30: with STEPS=6 pass 2 reproduced 240/240 recorded layer
# routes, with STEPS=30 it differed at the very first layer, (0,0). Measured 2026-09-16; the cause
# is not established and does not need to be, because the fix removes the rewind entirely. PASS=rec
# records the true routes to ROUTES and exits; PASS=run loads them into a FRESH engine. Each arm
# therefore starts from its own cold engine and its own prefill, which is also what made the
# stage-6 A/B trustworthy.
PASS = os.environ.get("PASS", "run")
ROUTES = os.environ.get("ROUTES", "/tmp/ds41_routes.json")
ARM = os.environ.get("ARM", "oracle")
HORIZON = int(os.environ.get("HORIZON", 1))
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
RECORD = 13_774_848
N_L = None


class TraceOracle(Prefetcher):
    """Reads the RECORDED route `horizon` layers ahead, in the same step, then the next step."""
    name = "oracle"

    def __init__(self, routes, horizon):
        self.routes = routes          # {(step, layer): (expert ids,)}
        self.horizon = horizon
        # THE WARM-UP OFFSET. The driver derives `step` from its own counter, which the harness
        # RESETS after the warm-up so the measurement starts at zero -- while the timed window is
        # recorded steps WARM..WARM+STEPS-1. Without this the oracle reads the warm-up's own steps,
        # names experts the warm-up just loaded, finds them all resident and issues NOTHING. Job
        # 335 shipped four arms like that: issued=0, used=0, pred_miss=7487 for a predictor that
        # cannot miss. Set by the harness once the base is known.
        self.base = 0

    def predict(self, layer, uniq, step):
        step = step + self.base
        tgt = layer + self.horizon
        s, L = (step, tgt) if tgt < N_L else (step + 1, tgt - N_L)
        ids = self.routes.get((s, L))
        return tuple((L, int(e)) for e in ids) if ids else ()


if "ENGRAM" not in os.environ:
    raise RuntimeError(
        "set ENGRAM=1 (the production model) or ENGRAM=0 (ablated, what every arm before 50bfdf2 "
        "measured) explicitly. There is no default: the ablation is a DIFFERENT MODEL -- priced at "
        "7.717 in the logits -- and it went unnoticed for a whole day precisely because it was the "
        "quiet fallback.")
ENGRAM = os.environ["ENGRAM"] == "1"
# EVICTION POLICY IS A CHOICE, NOT A DEFAULT. Every v2 measurement so far hardcoded "lru", and the
# box has already measured age/(1+count) ahead of it: decode 1.88/2.13/2.14 tok/s against LRU's
# 1.74/1.85/1.86 (job 140), and offline 94.07 % hit / 52.3 fetches per step against 92.67 % / 64.6
# (phase 1). Running the A/B on the worse policy is a fair comparison at the wrong operating point.
EVICT = os.environ.get("EVICT", "lru")


def build(eng, obs=None, prefetch=None):
    from enginev2 import drivers as v2drivers
    v2drivers.N_LAYERS = eng.args.n_layers
    rl = RealLeaves(os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin"), eng.arena)
    # ENGRAM=0 ZEROES eg_rows, which is engram_ablate -- a different model, not a neutral default
    # (mutation-checked at 7.717 in the logits). It is also how every arm was measured before
    # 50bfdf2, so it stays available for a like-for-like comparison and is never the silent state.
    src = RealEngramSource(eng, rl) if ENGRAM else None
    rl.engram = src
    e2 = v2drivers.Engine(Policy(), evict=EVICT, lru_slots=eng.store.n_slots - 8,
                          transient_slots=8, n_workers=8, staging=8, expert_read_qd=8,
                          h2d_inflight=2, leaves=rl, observer=obs, prefetch=prefetch,
                          engram=src)
    for k, slot in eng.store.lru.items():
        e2.slots.lru[k] = slot
        e2.slots.slot_key[slot] = k
        e2.slots.gen[slot] = e2.slots.gen.get(slot, 0)
        e2.slots.evict.on_insert(k, slot, 0)
    e2.slots.free_lru = [s for s in e2.slots.free_lru if s not in e2.slots.slot_key]
    return e2, rl


def warm_policy_BROKEN(e2, rl, routes, steps, n_layers):
    """Give the eviction policy real HISTORY before the timed window.

    Seeding calls on_insert(key, slot, 0) for every resident key, so all of them carry count 1 and
    the same age. age/(1+count) then picks the count-1 bucket's LRU head -- which is LRU. Job 310
    measured LRU twice and reported byte-identical fetch counts for both policies because of this,
    not because the policies tie.

    So replay recorded routes through the slot table first, untimed and with no reads: hits bump
    use counts through on_hit exactly as a real run would, and the frequency signal the policy
    exists to exploit is present when the measurement starts.
    """
    # WITHDRAWN 2026-09-16, DO NOT USE. reserve() only rewrites METADATA: the slot is remapped to
    # the new key while the arena still holds the previous expert's bytes, and clear_pending() then
    # declares the write finished. Every "warmed" key was a lie -- resident by bookkeeping, wrong
    # by content -- so the timed decode both under-read (971 against 1895) and computed with the
    # wrong weights. Job 320's own route check printed "31 layers match, 1249 differ" and nothing
    # acted on it. Kept only so the defect is legible next to its replacement.
    #
    # WARM FROM THE TAIL, NOT THE HEAD. The timed window always replays steps 0..STEPS-1 from a
    # fresh prefill, so warming on those same steps pre-loads exactly the experts the measurement
    # is about to ask for. Job 315 did that: every arm, oracle included, collapsed to 658 reads
    # against 1895 and the oracle's demand fetches went 51 -> 666, because there was nothing left
    # to prefetch. Warming on LATER recorded steps gives the policy real use counts over related
    # content without handing it the answer.
    seen = 0
    for st in range(STEPS, STEPS + steps):
        for L in range(n_layers):
            u = routes.get((st, L))
            if not u:
                continue
            slot_of, to_load, _w = e2.slots.reserve(L, tuple(u), prefill=False)
            for _k, sl, g in to_load:
                e2.slots.clear_pending(sl, g)      # offline replay: the read "completes" at once
            seen += 1
    return seen


eng = V41Engine(os.path.expanduser("~/dsv41-lean"), max_seq=8192,
                arena_gb=float(os.environ.get("ARENA_GB", 40)),
                spec=True, expert_format="cb3")
N_L = eng.args.n_layers
ids = eng.tokenizer.encode(
    "Explain how an NVMe SSD controller schedules writes, and why the flash translation "
    "layer matters for tail latency under a mixed read/write workload.", add_special_tokens=False)
m, fd = eng.model, eng.fast


def prefill():
    m.begin_prompt()
    lg, mh = m.forward(torch.tensor(ids, dtype=torch.long, device="cuda"), 0,
                       prefill=True, need_logits=True)
    if eng.spec:
        m.dspark_seed(mh, 0)
    return lg


lg = prefill()
T = fd.ids.numel()
nb = lambda l, s: torch.full((T,), int(l[-1].argmax()), dtype=torch.long, device="cuda")

if PASS == "rec":
    # ---- record the true routes and exit. Untimed; its only output is the route table.
    routes = {}
    e2, rl = build(eng)
    rl.attach(eng, nb(lg, 0), next_block=nb)
    _la = rl.layer_a
    def spy(L, _f=_la):
        r = _f(L)
        routes[(e2.c.steps, L)] = list(r.uniq)
        return r
    rl.layer_a = spy
    e2.decode(STEPS)
    e2.close(); rl.close()
    import json
    json.dump({f"{s_}:{l_}": v for (s_, l_), v in routes.items()}, open(ROUTES, "w"))
    print(f"  recorded {len(routes)} routes over {STEPS} steps -> {ROUTES}")
    sys.exit(0)

import json                                          # noqa: E402
routes = {tuple(int(x) for x in k.split(":")): tuple(v)
          for k, v in json.load(open(ROUTES)).items()}
print(f"  loaded {len(routes)} recorded routes from {ROUTES}")

# ---- the timed arm, in its own process, from its own cold engine and prefill
obs = TraceObserver(capacity=1 << 18)
pf = TraceOracle(routes, HORIZON) if ARM == "oracle" else None
e2, rl = build(eng, obs=obs, prefetch=pf)
rl.attach(eng, nb(lg, 0), next_block=nb)
# DOES THE ORACLE ACTUALLY KNOW THE FUTURE? Pass 2 must walk the same route sequence pass 1
# recorded, or the "oracle" is a random predictor wearing the name and every number below is about
# something else. Checked, not assumed: the first run of this harness reported pred_miss 11145
# against pred_hit 2837, which for a trace-reading oracle is impossible and was the tell.
agree = [0, 0]
# Base 0 during the warm-up (which really is recorded steps 0.._warm-1); rebased to _warm once the
# timed window starts. Defined HERE because the check runs inside the warm decode too.
ROUTE_BASE = 0
_la2 = rl.layer_a
def check(L, _f=_la2):
    r = _f(L)
    want = routes.get((ROUTE_BASE + e2.c.steps, L))
    ok = want == r.uniq
    agree[0 if ok else 1] += 1
    if not ok and agree[1] == 1:
        print(f"  first mismatch at key {(e2.c.steps, L)}: "
              f"recorded {None if want is None else want[:6]} got {r.uniq[:6]}  "
              f"(recorded keys start {sorted(routes)[:2]})")
    return r
rl.layer_a = check
# Policy history before timing. WARM=0 reproduces job 310's flat-seeded behaviour for comparison.
_warm = int(os.environ.get("WARM", 0))
if _warm:
    # REAL DECODE, chronologically ahead of the timed window. Nothing else warms a cache: the
    # arena only holds an expert once a read has actually put its bytes there. So steps
    # 0.._warm-1 run untimed with prediction off, and the timed window is steps _warm.._warm+STEPS-1
    # of the SAME continuous generation -- no re-prefill, no rewind, no replay of the test set.
    have = max((k[0] for k in routes), default=-1) + 1
    if have < STEPS + _warm:
        raise RuntimeError(
            f"WARM={_warm} + STEPS={STEPS} needs {STEPS + _warm} recorded steps, have {have}")
    saved_pf, e2.prefetch = e2.prefetch, Prefetcher()
    e2._scoring = False
    e2.decode(_warm)
    e2.prefetch, e2._scoring = saved_pf, True
    e2.c = type(e2.c)()                      # the warm-up's counters are not the measurement's
    print(f"  warmed with {_warm} REAL decode steps ({e2.loader.started_demand} reads issued), "
          f"routes reproduced {agree[0]}/{agree[0] + agree[1]}")
    if agree[1]:
        raise SystemExit(f"ABORT: {agree[1]} route divergences during the WARM-UP itself")
    ROUTE_BASE = _warm
    agree[0] = agree[1] = 0                  # the timed window is scored on its own
    if pf is not None:
        pf.base = _warm                      # the oracle must read the TIMED window's routes
# The observer's drain() is a SNAPSHOT, not a drain, so the warm-up's events are still in the ring
# and would be counted in both the wall-window read count and the depth histogram -- the first run
# of the repaired warm-up reported 2369 reads and a 9.61 s depth span against a 5.55 s wall.
# Everything is therefore filtered to events at or after this mark.
t0_ns = time.perf_counter_ns()
# NVTX marks the TIMED window so a profile can be restricted to it. Without this a trace of this
# script covers the 65 s engine load, the warm start and the warm-up decode as well, and any
# "GPU busy %" computed over that is about the wrong thing entirely.
torch.cuda.nvtx.range_push("timed_decode")
t0 = time.perf_counter()
c = e2.decode(STEPS)
torch.cuda.nvtx.range_pop()
wall = time.perf_counter() - t0
# TWO DIFFERENT QUESTIONS, two drains.
#
#   depth-over-time is a WALL-WINDOW statistic: what was in flight between t0 and t0+wall. It has
#   to be drained here, before settlement issues reads of its own.
#
#   bytes is a CAUSAL question: what I/O did this window cause. A scored speculative read issued
#   inside the window can start or finish just after it, and counting brackets inside the window
#   silently drops it -- which is how "2393 against 2392 reads" could look like parity that the
#   accounting had produced rather than found. Event.scored already follows the cohort across the
#   async boundary, so the causal count is every scored nvme_start after settlement and quiesce.
ev = obs.drain()
# SETTLEMENT, the real-engine way. drivers.settle() is a replay mechanism and refuses a provider
# that computes its own routes -- correctly, it drives decode_layer() without a step's chain.reset()
# or engram.issue(). The equivalent here is real decode steps with prediction OFF and scoring off:
# predictions issued near the end of the window get CLASSIFIED instead of right-censored, and they
# contend for the device exactly as they really would.
e2.prefetch = Prefetcher()
e2._scoring = False
e2.decode(2)
e2.finalize_stats(settle_layers=0)
e2.loader.quiesce(60.0)              # nothing still in flight when the causal count is taken
ev_causal = obs.drain()
e2.close(); rl.close()

# Every scored nvme_start, wherever it physically landed. Settlement runs with scoring off, so its
# own demand reads are excluded by the cohort bit rather than by a timestamp comparison.
#
# ev_causal ALONE, not ev + ev_causal: TraceObserver.drain() is a non-destructive SNAPSHOT of the
# rings, not a drain, so the later call already contains the earlier one and adding them counted
# every read twice. That produced a causal count of exactly 2x the window count, identical across
# arms -- arithmetic, not measurement. Caught 2026-09-16 by the count being suspiciously round.
causal_reads = sum(1 for e in ev_causal
                   if e.kind == "nvme_start" and e.scored and e.ts_ns >= t0_ns)
# The rings are finite. If they wrapped, the earliest events are gone and the causal count is a
# silent undercount, which is worse than no number at all -- so say so.
ring_full = len(ev_causal) >= obs.effective_capacity
if ring_full:
    print(f"  WARNING: observer ring wrapped ({len(ev_causal)} >= {obs.effective_capacity}); "
          f"the causal count is truncated and must not be compared across arms")

marks = sorted((e.ts_ns, +1 if e.kind == "nvme_start" else -1)
               for e in ev if e.kind in ("nvme_start", "nvme_end") and e.ts_ns >= t0_ns)
depth, dur, prev = 0, collections.Counter(), (marks[0][0] if marks else 0)
for t, d in marks:
    dur[depth] += t - prev
    prev = t
    depth += d
span = sum(dur.values()) / 1e9
idle = dur[0] / 1e9
busy = span - idle
reads = len(marks) // 2
print(f"  evict {EVICT}  lru_slots {eng.store.n_slots - 8}  "
      f"v1 staging {eng.store.io_threads} buffers")
print(f"  engram {'LIVE' if ENGRAM else 'ABLATED'}"
      + (f", rows {sum(t.stats['rows'] for t in eng.tables.values())}" if ENGRAM else
         f", zeroed-layer fills {rl.engram_ablated}"))
print(f"  route sequence reproduced: {agree[0]} layers match, {agree[1]} differ")
# FAIL, do not merely report. Job 320 printed "31 match, 1249 differ" for four arms and every
# number from them was reported as a result. A run whose routes diverge inside the timed window is
# computing with weights that are not the model's, and nothing it measures is about this engine.
_timed = STEPS * eng.args.n_layers
if agree[0] < _timed:
    raise SystemExit(f"ABORT: {agree[1] - (agree[0] + agree[1] - _timed)} route divergences inside "
                     f"the timed window ({agree[0]} of {_timed} matched). The cache is serving "
                     f"experts whose bytes were never loaded, or the state was not reproduced.")
print(f"ARM {ARM} horizon {HORIZON}  {STEPS} steps  wall {wall:.2f}s  {STEPS / wall:.3f} steps/s")
print(f"  demand fetches {c.fetches}  reads in the wall window {reads}  "
      f"causal scored reads {causal_reads}  {causal_reads * RECORD / 1e9:.2f} GB  "
      f"achieved {reads * RECORD / wall / 1e9:.2f} GB/s")
print(f"  depth 0 {idle:.2f}s = {idle / span * 100:.1f}% of span   mean depth while reading "
      f"{sum(k * v for k, v in dur.items() if k) / 1e9 / busy if busy else 0:.2f}   "
      f"in-flight {reads * RECORD / busy / 1e9 if busy else 0:.2f} GB/s")
print(f"  prefetch: {e2.pf}")
if ARM == "oracle":
    # An oracle reads a recorded trace: pred_miss must be 0 and it must issue something. Anything
    # else means it is not reading the window being measured, which is how job 335's oracle arms
    # were reported with issued=0.
    if e2.pf.pred_miss or e2.pf.issued == 0:
        raise SystemExit(f"ABORT: not an oracle -- pred_miss={e2.pf.pred_miss} "
                         f"issued={e2.pf.issued}; it is not reading the timed window's routes")
