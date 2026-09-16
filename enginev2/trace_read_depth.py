"""How many expert reads are actually in flight during decode, over time?

The aggregate numbers cannot separate two very different worlds that both average 2.24 reads in
flight: reads CONTINUOUSLY in flight at depth ~2, or reads in flight at depth ~5 for 45 % of the
wall and nothing for the rest. The first says the device is being asked for 2 concurrent reads and
is answering at its 2-concurrent rate; the second says the pipe empties between layers. They point
at completely different fixes, so this reads the loader's own nvme_start/nvme_end events and builds
the depth-over-time histogram directly.

Run:  DSV41_CB3_CACHE=~/dsv41-cb3/experts-cb3-s3.bin DSV41_DENSE_FP4=attn,wo_a DSV41_HEAD_FMT=fp8 \
      python enginev2/trace_read_depth.py [steps]
"""
import os, sys, time, collections, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = os.path.expanduser("~/git/deepseek-v41-flash-spark")
sys.path[:0] = [V1, os.path.join(V1, "tools")]
os.chdir(V1)

from engine.v41_engine import V41Engine              # noqa: E402
from enginev2.observe import TraceObserver           # noqa: E402
from enginev2.real import RealLeaves                 # noqa: E402
from enginev2.sched import Policy                    # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
RECORD = 13_774_848

eng = V41Engine(os.path.expanduser("~/dsv41-lean"), max_seq=8192,
                arena_gb=float(os.environ.get("ARENA_GB", 40)),
                spec=True, expert_format="cb3")
ids = eng.tokenizer.encode(
    "Explain how an NVMe SSD controller schedules writes, and why the flash translation "
    "layer matters for tail latency under a mixed read/write workload.", add_special_tokens=False)
m, fd = eng.model, eng.fast
m.begin_prompt()
lg, mh = m.forward(torch.tensor(ids, dtype=torch.long, device="cuda"), 0,
                   prefill=True, need_logits=True)
if eng.spec:
    m.dspark_seed(mh, 0)
T = fd.ids.numel()
nb = lambda l, s: torch.full((T,), int(l[-1].argmax()), dtype=torch.long, device="cuda")

from enginev2 import drivers as v2drivers            # noqa: E402
v2drivers.N_LAYERS = eng.args.n_layers
obs = TraceObserver(capacity=1 << 18)
rl = RealLeaves(os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin"), eng.arena)
rl.attach(eng, nb(lg, 0), next_block=nb)
e2 = v2drivers.Engine(Policy(), evict="lru", lru_slots=eng.store.n_slots - 8, transient_slots=8,
                      n_workers=8, staging=8, expert_read_qd=8, h2d_inflight=2,
                      leaves=rl, observer=obs)
for k, slot in eng.store.lru.items():               # same warm residency as v1
    e2.slots.lru[k] = slot
    e2.slots.slot_key[slot] = k
    e2.slots.gen[slot] = e2.slots.gen.get(slot, 0)
    e2.slots.evict.on_insert(k, slot, 0)
e2.slots.free_lru = [s for s in e2.slots.free_lru if s not in e2.slots.slot_key]

t0 = time.perf_counter()
c = e2.decode(STEPS)
wall = time.perf_counter() - t0
ev = obs.drain()
e2.close(); rl.close()

# depth over time from the loader's OWN nvme brackets
marks = []
for e in ev:
    if e.kind == "nvme_start":
        marks.append((e.ts_ns, +1))
    elif e.kind == "nvme_end":
        marks.append((e.ts_ns, -1))
marks.sort()
depth = 0
dur = collections.Counter()
prev = marks[0][0] if marks else 0
for t, d in marks:
    dur[depth] += t - prev
    prev = t
    depth += d
span = sum(dur.values()) / 1e9
print(f"  {STEPS} steps  wall {wall:.2f}s  {STEPS / wall:.3f} steps/s  fetches {c.fetches}  "
      f"achieved {c.fetches * RECORD / wall / 1e9:.2f} GB/s")
print(f"  read brackets {len(marks) // 2}  traced span {span:.2f}s of {wall:.2f}s wall")
idle = dur[0] / 1e9
busy = span - idle
mean_depth = sum(k * v for k, v in dur.items()) / 1e9 / span if span else 0
print(f"  depth 0 (NO read in flight): {idle:.2f}s = {idle / span * 100:.1f}% of the traced span")
print(f"  mean depth over span {mean_depth:.2f}; mean depth while >=1 read in flight "
      f"{sum(k * v for k, v in dur.items() if k) / 1e9 / busy if busy else 0:.2f}")
for k in sorted(dur):
    if dur[k]:
        print(f"    depth {k:2d}  {dur[k] / 1e9:6.2f}s  {dur[k] / 1e9 / span * 100:5.1f}%")
print(f"  in-flight bandwidth (bytes / time with >=1 read) "
      f"{c.fetches * RECORD / busy / 1e9 if busy else 0:.2f} GB/s")
