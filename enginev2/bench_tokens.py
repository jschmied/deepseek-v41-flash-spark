"""Real tokens/second from the v2 engine: DSpark draft, verify, rollback -- the served loop.

Every throughput number this branch has produced was a proxy. `next_block` fed the block in and
`end_step` advanced the cache by all 6 tokens, i.e. 100 % acceptance, and nothing was ever
committed -- so "8.1 steps/s" converted to a tokens/s only through an assumption. With spec mode
the drafter's graph replays, the verify decides how many tokens survive, and the cache rolls back
to what was actually committed, so tokens/s is counted rather than inferred.

Reports what the heartbeat protocol asks for: tokens via the committed count (never a chunk
count), bytes and reads PER COMMITTED TOKEN, and the measured accept length.

    ENGRAM=1 EVICT=lru ARENA_GB=79 python enginev2/bench_tokens.py [steps] [warm_steps]
"""
import os, sys, time, statistics as st, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = os.path.expanduser("~/git/deepseek-v41-flash-spark")
sys.path[:0] = [V1, os.path.join(V1, "tools")]
os.chdir(V1)

from engine.v41_engine import V41Engine                    # noqa: E402
from enginev2.real import RealEngramSource, RealLeaves     # noqa: E402
from enginev2.sched import Policy                          # noqa: E402

if "ENGRAM" not in os.environ:
    raise RuntimeError("set ENGRAM=1 (production) or ENGRAM=0 (ablated) explicitly")
ENGRAM = os.environ["ENGRAM"] == "1"
EVICT = os.environ.get("EVICT", "lru")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
WARM = int(sys.argv[2]) if len(sys.argv) > 2 else 60
RECORD = 13_774_848

eng = V41Engine(os.path.expanduser("~/dsv41-lean"), max_seq=8192,
                arena_gb=float(os.environ.get("ARENA_GB", 79)), spec=True, expert_format="cb3")
m, fd = eng.model, eng.fast
prompt = ("Write a detailed technical explanation of how an NVMe SSD controller schedules writes, "
          "why the flash translation layer matters for tail latency under a mixed read/write "
          "workload, and how a host can measure the effect without vendor telemetry.")
ids = eng.tokenizer.encode(prompt, add_special_tokens=False)
m.c.rollback(0)
m.begin_prompt()
lg, mh = m.forward(torch.tensor(ids, dtype=torch.long, device="cuda"), 0, prefill=True,
                   need_logits=True)
m.dspark_seed(mh, 0)
first = int(lg[-1].argmax())

from enginev2 import drivers as v2drivers                  # noqa: E402
v2drivers.N_LAYERS = eng.args.n_layers
rl = RealLeaves(os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin"), eng.arena)
rl.attach(eng, None, spec=True, temperature=0.0, first_token=first)
src = RealEngramSource(eng, rl) if ENGRAM else None
rl.engram = src
e2 = v2drivers.Engine(Policy(), evict=EVICT, lru_slots=eng.store.n_slots - 8, transient_slots=8,
                      n_workers=8, staging=8, expert_read_qd=8, h2d_inflight=2,
                      leaves=rl, engram=src)
for k, slot in eng.store.lru.items():
    e2.slots.lru[k] = slot
    e2.slots.slot_key[slot] = k
    e2.slots.gen[slot] = e2.slots.gen.get(slot, 0)
    e2.slots.evict.on_insert(k, slot, 0)
e2.slots.free_lru = [s for s in e2.slots.free_lru if s not in e2.slots.slot_key]

print(f"  engram {'LIVE' if ENGRAM else 'ABLATED'}  evict {EVICT}  "
      f"arena slots {eng.store.n_slots}  block {fd.ids.numel()}", flush=True)
if WARM:
    e2.decode(WARM)
    print(f"  warm {WARM} steps: {rl.tokens_out} tokens committed, "
          f"{e2.loader.started_demand} reads", flush=True)
t_out0, reads0, acc0 = rl.tokens_out, rl.read_bytes, len(rl.accepted)

torch.cuda.synchronize()
t0 = time.perf_counter()
c = e2.decode(STEPS)
torch.cuda.synchronize()
wall = time.perf_counter() - t0
e2.close(); rl.close()

toks = rl.tokens_out - t_out0
gb = (rl.read_bytes - reads0) / 1e9
acc = rl.accepted[acc0:]
print(f"  {STEPS} steps, wall {wall:.2f}s")
print(f"  TOKENS {toks}  ->  {toks / wall:.2f} tok/s          <- the number")
print(f"  steps/s {STEPS / wall:.3f}   accept_len_mean {st.mean(acc) + 1:.2f}  "
      f"(tokens per step {toks / STEPS:.2f})")
print(f"  nvme {gb:.2f} GB  {gb * 1000 / max(1, toks):.1f} MB per committed token  "
      f"{(rl.read_bytes - reads0) / RECORD / max(1, toks):.1f} reads per token")
