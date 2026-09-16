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
from enginev2.real import RealLeaves                 # noqa: E402
from enginev2.sched import Policy                    # noqa: E402

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

    def predict(self, layer, uniq, step):
        tgt = layer + self.horizon
        s, L = (step, tgt) if tgt < N_L else (step + 1, tgt - N_L)
        ids = self.routes.get((s, L))
        return tuple((L, int(e)) for e in ids) if ids else ()


def build(eng, obs=None, prefetch=None):
    from enginev2 import drivers as v2drivers
    v2drivers.N_LAYERS = eng.args.n_layers
    rl = RealLeaves(os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin"), eng.arena)
    e2 = v2drivers.Engine(Policy(), evict="lru", lru_slots=eng.store.n_slots - 8,
                          transient_slots=8, n_workers=8, staging=8, expert_read_qd=8,
                          h2d_inflight=2, leaves=rl, observer=obs, prefetch=prefetch)
    for k, slot in eng.store.lru.items():
        e2.slots.lru[k] = slot
        e2.slots.slot_key[slot] = k
        e2.slots.gen[slot] = e2.slots.gen.get(slot, 0)
        e2.slots.evict.on_insert(k, slot, 0)
    e2.slots.free_lru = [s for s in e2.slots.free_lru if s not in e2.slots.slot_key]
    return e2, rl


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

# ---- pass 1: record the true routes. Untimed; its only output is the route table.
routes = {}
e2, rl = build(eng)
rl.attach(eng, nb(lg, 0), next_block=nb)
_la = rl.layer_a
def spy(L, _f=_la):
    r = _f(L)
    routes[(e2.c.steps, L)] = r.uniq
    return r
rl.layer_a = spy
e2.decode(STEPS)
e2.close(); rl.close()
print(f"  recorded {len(routes)} routes over {STEPS} steps")

# ---- pass 2: the timed arm, from the same prefilled state and the same residency
m.c.rollback(m.c.len - m.c.len)
lg = prefill()
obs = TraceObserver(capacity=1 << 18)
pf = TraceOracle(routes, HORIZON) if ARM == "oracle" else None
e2, rl = build(eng, obs=obs, prefetch=pf)
rl.attach(eng, nb(lg, 0), next_block=nb)
t0 = time.perf_counter()
c = e2.decode(STEPS)
wall = time.perf_counter() - t0
st = e2.finalize_stats()
ev = obs.drain()
e2.close(); rl.close()

marks = sorted((e.ts_ns, +1 if e.kind == "nvme_start" else -1)
               for e in ev if e.kind in ("nvme_start", "nvme_end"))
depth, dur, prev = 0, collections.Counter(), (marks[0][0] if marks else 0)
for t, d in marks:
    dur[depth] += t - prev
    prev = t
    depth += d
span = sum(dur.values()) / 1e9
idle = dur[0] / 1e9
busy = span - idle
reads = len(marks) // 2
print(f"ARM {ARM} horizon {HORIZON}  {STEPS} steps  wall {wall:.2f}s  {STEPS / wall:.3f} steps/s")
print(f"  demand fetches {c.fetches}  reads issued {reads}  "
      f"{reads * RECORD / 1e9:.2f} GB  achieved {reads * RECORD / wall / 1e9:.2f} GB/s")
print(f"  depth 0 {idle:.2f}s = {idle / span * 100:.1f}% of span   mean depth while reading "
      f"{sum(k * v for k, v in dur.items() if k) / 1e9 / busy if busy else 0:.2f}   "
      f"in-flight {reads * RECORD / busy / 1e9 if busy else 0:.2f} GB/s")
print(f"  prefetch: {st}")
