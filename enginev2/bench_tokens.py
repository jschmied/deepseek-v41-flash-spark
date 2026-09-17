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

# Resolve `engine` and `tools` from THIS checkout only. This file used to prepend
# ~/git/deepseek-v41-flash-spark to sys.path and chdir into it, which silently ran a different
# checkout of the same repo -- on a different branch. That is how this branch produced real-graph
# resident-first numbers while its own engine/fastdecode.py had no split in it at all. The
# provenance line below exists so the substitution can never be invisible again.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
os.chdir(ROOT)

from engine.v41_engine import V41Engine                    # noqa: E402
from enginev2.real import RealEngramSource, RealLeaves     # noqa: E402
from enginev2.sched import Policy                          # noqa: E402
import engine.fastdecode as _fd_mod                        # noqa: E402

print(f"engine.fastdecode -> {_fd_mod.__file__}")

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
# D3 (global_barrier) is the difference between "wait for every demand copy to LAND" and "wait for
# the slots this layer needs to be PUBLISHED". With it on, the driver blocks until _demand hits 0,
# which _complete_h2d decrements only after handle.synchronize() -- so the whole device-ordered
# fast path sits downstream of a full physical barrier and cannot show a benefit. Off, the per-slot
# published event and await_copies are what order the work, and graph B can be queued while bytes
# are still in flight. Default keeps Policy(), i.e. V1, so every earlier number stays comparable.
POL = Policy(global_barrier=os.environ.get("DSV41_GLOBAL_BARRIER", "1") == "1")
print(f"  policy: global_barrier={POL.global_barrier} resolve_blocks={POL.resolve_blocks} "
      f"compute_barrier_global={POL.compute_barrier_global} "
      f"lease_until_completion={POL.lease_until_completion}")
e2 = v2drivers.Engine(POL, evict=EVICT, lru_slots=eng.store.n_slots - 8, transient_slots=8,
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
# ENGRAM IS AN IOPS QUESTION, NOT A BYTES ONE. 144 rows per step at 264 B is ~0.003 % of decode
# bytes, which is why it was written off -- but engram.py does TWO preads per row (weight then
# scale, line 60), so that is ~288 operations per step against ~67-80 expert reads. On a device
# whose binding constraint is operations in flight rather than GB/s, the small stream can cost more
# than its bytes. These counters are the engine's own.
eg0 = {L: dict(t.stats) for L, t in eng.tables.items()}

torch.cuda.synchronize()
# The host profile must cover the TIMED window only. Without this the warm-up's phases are divided
# by STEPS and every row is scaled by (WARM + STEPS) / STEPS.
e2.hostprof.reset()
# SAME REASON, SAME WINDOW: Counters accumulate across decode() calls, so the residency figures
# below used to cover warm + measured while tok/s, NVMe GB, acceptance and the host profile all
# covered the measured window alone. Residency evolves precisely DURING cache warming, so pricing
# the resident-first split on one window and its throughput on another compares two different
# cache states. Snapshot here and report deltas.
_res0 = (c0 := e2.c).pairs_total, c0.pairs_resident, c0.uniq_total, c0.uniq_resident, c0.steps
t0 = time.perf_counter()
c = e2.decode(STEPS)
torch.cuda.synchronize()
wall = time.perf_counter() - t0
e2.close(); rl.close()

toks = rl.tokens_out - t_out0
gb = (rl.read_bytes - reads0) / 1e9
acc = rl.accepted[acc0:]
print(f"  {STEPS} steps, wall {wall:.2f}s")
# DSV41_HOST_PROFILE=1 only. The interesting quantity is not any single row but
# (wall/step - INSTRUMENTED): whatever is left is time the driver spends outside every phase, and
# if that is near zero the ~69 ms/step of section 14 is inside one of these rows.
_pt = c.pairs_total - _res0[0]; _pr = c.pairs_resident - _res0[1]
_ut = c.uniq_total - _res0[2]; _ur = c.uniq_resident - _res0[3]
_cs = c.steps - _res0[4]
if _pt:
    # UNIQUE is the weight that prices the split, not PAIR. The MoE streams an expert once per
    # launch however many pairs it serves, so what phase 1 can compute without the reads is set by
    # the distinct experts already in the arena -- weight BYTES -- not by the arithmetic. PAIR is
    # reported next to it because it is what an earlier estimate wrongly used, and keeping both
    # visible is what stops that estimate being made a fifth time.
    print(f"  UNIQUE residency {_ur}/{_ut} = {_ur / max(1, _ut) * 100:.1f} %  "
          f"<- the WEIGHT-BYTES weight; THIS is what prices a resident-first MoE split")
    print(f"  unique/layer-step {_ut / max(1, _cs * 40):.1f}")
    print(f"  PAIR residency {_pr}/{_pt} = {_pr / _pt * 100:.1f} %  "
          f"<- arithmetic, not bytes; not the weight for the split")
    print(f"  pairs/layer-step {_pt / max(1, _cs * 40):.1f}, non-resident pairs {_pt - _pr}")
    print(f"  (residency over the {_cs} TIMED steps only; warm-up excluded)")
_rd = os.environ.get("DSV41_ROUTE_DUMP")
if _rd and e2._route_dump:
    import pickle
    with open(_rd, "wb") as fh:
        pickle.dump(e2._route_dump[-40 * 200:], fh)     # last ~200 steps is plenty
    print(f"  routes dumped: {len(e2._route_dump)} layer-visits -> {_rd}")
gt = rl.gt_report()
if gt:
    print(gt)
hp = e2.hostprof.report(STEPS)
if hp:
    print(hp)
    print(f"    {'wall/step':<14} {wall / STEPS * 1e3:7.2f} ms  "
          f"({(wall / STEPS - sum(e2.hostprof.t.values()) / STEPS) * 1e3:+.2f} ms unattributed)")
print(f"  TOKENS {toks}  ->  {toks / wall:.2f} tok/s          <- the number")
print(f"  steps/s {STEPS / wall:.3f}   accept_len_mean {st.mean(acc) + 1:.2f}  "
      f"(tokens per step {toks / STEPS:.2f})")
print(f"  nvme {gb:.2f} GB  {gb * 1000 / max(1, toks):.1f} MB per committed token  "
      f"{(rl.read_bytes - reads0) / RECORD / max(1, toks):.1f} reads per token")
if ENGRAM:
    rows = sum(t.stats["rows"] - eg0[L]["rows"] for L, t in eng.tables.items())
    calls = sum(t.stats["calls"] - eg0[L]["calls"] for L, t in eng.tables.items())
    rd = sum(t.stats.get("read_s", 0.0) - eg0[L].get("read_s", 0.0) for L, t in eng.tables.items())
    sec = sum(t.stats["seconds"] - eg0[L]["seconds"] for L, t in eng.tables.items())
    print(f"  engram rows {rows} ({rows / STEPS:.0f}/step, {rows * 2} preads = "
          f"{rows * 2 / STEPS:.0f}/step)  bytes {rows * 264 / 1e6:.1f} MB")
    print(f"  engram read_s {rd:.2f}s ({rd / wall * 100:.1f} % of wall)  "
          f"total_s {sec:.2f}s ({sec / wall * 100:.1f} % of wall)  "
          f"{rd / max(1, rows) * 1e6:.0f} us per row")
