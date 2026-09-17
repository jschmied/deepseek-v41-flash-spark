"""Gate for the real engram source: v2 must match v1's REAL engram path, exactly.

The stage-5 logits gate passed `{}` as v1's engram_rows and zeroed v2's, so BOTH arms ran the model
engram-ablated and the gate could not see this at all. Every v2 number before this ran on a
different model from the server's.

Here v1 is driven the way v41_engine drives it -- hash the block, D2H the ids, submit both tables'
read_raw on the engram pool, and hand `lambda: futs` to fast.step -- and v2 runs the same step
through the Engine with RealEngramSource on the chain's engram edge.

The floor is 0.000e+00: both replay the same graphs over the same weights and the same rows.

MUT=1 ablates v2's engram (no source) while v1 keeps its rows. That is the mutation check: it must
break the equality, or the gate is not testing that the rows arrived.

Run:  DSV41_CB3_CACHE=~/dsv41-cb3/experts-cb3-s3.bin DSV41_DENSE_FP4=attn,wo_a DSV41_HEAD_FMT=fp8 \
      ARENA_GB=8 python enginev2/falsify_engram.py
"""
import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# THIS checkout -- see RealLeaves.ROOT in enginev2/real.py. This used to be a hardcoded
# ~/git/deepseek-v41-flash-spark: a DIFFERENT working tree of the SAME repo, on a different
# branch, whose engine/ silently shadowed this one. Every real-graph number this branch
# produced came from code that was not committed here.
V1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [V1, os.path.join(V1, "tools")]
os.chdir(V1)

from engine.v41_engine import V41Engine                      # noqa: E402
from enginev2.real import RealEngramSource, RealLeaves       # noqa: E402
from enginev2.sched import Policy                            # noqa: E402

MUT = os.environ.get("MUT") == "1"

eng = V41Engine(os.path.expanduser("~/dsv41-lean"), max_seq=8192,
                arena_gb=float(os.environ.get("ARENA_GB", 8)),
                spec=True, expert_format="cb3")
ids = eng.tokenizer.encode("Explain how an NVMe SSD controller schedules writes.",
                           add_special_tokens=False)
m, fd = eng.model, eng.fast
layer_ids = tuple(eng.args.engram_layer_ids)
print(f"  engram layers {layer_ids}, tables {sorted(eng.tables)}")


def prefill():
    m.begin_prompt()
    lg, mh = m.forward(torch.tensor(ids, dtype=torch.long, device="cuda"), 0,
                       prefill=True, need_logits=True)
    if eng.spec:
        m.dspark_seed(mh, 0)
    return lg


lg = prefill()
T = fd.ids.numel()
block = torch.full((T,), int(lg[-1].argmax()), dtype=torch.long, device="cuda")

# ---- v1, with its real engram path, exactly as v41_engine.generate() drives it
S0 = m.c.len
hashes = m.hash_state(block[None], S0)[0]
h_np = hashes.cpu().numpy()
futs = {L: (eng.eg_pool.submit(eng.tables[L].read_raw, h_np[:, li, :]), eng.tables[L].to_device)
        for li, L in enumerate(layer_ids)}
ref_logits, _ = fd.step(block, S0, lambda: futs)
ref = ref_logits.clone()
rows_v1 = sum(t.stats["rows"] for t in eng.tables.values())

# ---- v2, same step, through the Engine, with the real source on the engram edge
m.c.rollback(m.c.len - T)
m.begin_prompt()
from enginev2 import drivers as v2drivers                    # noqa: E402
v2drivers.N_LAYERS = eng.args.n_layers
rl = RealLeaves(os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin"), eng.arena)
rl.attach(eng, block)
src = None if MUT else RealEngramSource(eng, rl)
if src is not None:
    rl.engram = src
e2 = v2drivers.Engine(Policy(), evict="lru", lru_slots=eng.store.n_slots - 8, transient_slots=8,
                      n_workers=8, staging=8, expert_read_qd=8, h2d_inflight=2,
                      leaves=rl, engram=src)
for k, slot in eng.store.lru.items():
    e2.slots.lru[k] = slot
    e2.slots.slot_key[slot] = k
    e2.slots.gen[slot] = e2.slots.gen.get(slot, 0)
    e2.slots.evict.on_insert(k, slot, 0)
e2.slots.free_lru = [s for s in e2.slots.free_lru if s not in e2.slots.slot_key]
e2.decode(1)
got = fd.logits.clone()
rows_v2 = sum(t.stats["rows"] for t in eng.tables.values()) - rows_v1
e2.close(); rl.close()

same = bool(torch.equal(ref, got))
d = (ref.float() - got.float()).abs().max().item()
print(f"  engram rows read: v1 {rows_v1}, v2 {rows_v2}   v2 ablated layers {rl.engram_ablated}")
print(f"  logits: exact match {same}, max |delta| {d:.3e}   (MUT={int(MUT)})")
sys.exit(0 if same != MUT else 1)
