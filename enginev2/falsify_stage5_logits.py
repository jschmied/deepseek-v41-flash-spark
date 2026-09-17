"""Stage 5 gate: the v2 Engine, driving the real leaves, must produce v1's logits EXACTLY.

Unlike stage 4 this runs the whole v2 machine -- ExpertSlots, LoaderService with real worker
threads, the pinned staging pool, real O_DIRECT reads and real H2D into the arena -- and v2 owns
the arena mapping. v1's warm start is irrelevant to it: v2 starts with an empty slot table and
faults every expert it needs in through its own loader.

The gate is EXACT equality, not a tolerance. Both arms replay the same captured graphs over the
same weights; the only thing that differs is which arena slot each expert sits in and who put it
there. A slot mix-up, a torn read, an H2D that had not completed, or a route-misaligned
bind_slots all change the logits, and none of them change them by "a little".

Needs the box. Run:
    DSV41_CB3_CACHE=~/dsv41-cb3/experts-cb3-s3.bin DSV41_DENSE_FP4=attn,wo_a \
    DSV41_HEAD_FMT=fp8 python enginev2/falsify_stage5_logits.py [n_layers]

An optional argument limits how many layers v2 drives, so the one-layer case can be checked before
the forty-layer one; with fewer than all layers the comparison is against v1 stopped at the same
layer, using h rather than logits.

MUT=1 rotates v2's slot mapping by one expert. That is the mutation check: it must break the
equality, or the gate is not testing that v2 put the right bytes in the right slot.
"""
import os, sys, time, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# THIS checkout -- see RealLeaves.ROOT in enginev2/real.py. This used to be a hardcoded
# ~/git/deepseek-v41-flash-spark: a DIFFERENT working tree of the SAME repo, on a different
# branch, whose engine/ silently shadowed this one. Every real-graph number this branch
# produced came from code that was not committed here.
V1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V1)
sys.path.insert(0, os.path.join(V1, "tools"))
os.chdir(V1)

from engine.v41_engine import V41Engine              # noqa: E402
from enginev2.real import RealLeaves                 # noqa: E402
from enginev2.sched import Policy                    # noqa: E402

MUT = os.environ.get("MUT") == "1"
NL = int(sys.argv[1]) if len(sys.argv) > 1 else 0

eng = V41Engine(os.path.expanduser("~/dsv41-lean"), max_seq=8192,
                arena_gb=float(os.environ.get("ARENA_GB", 40)),
                spec=True, expert_format="cb3")
n_layers = eng.args.n_layers
if NL == 0:
    NL = n_layers
ids = eng.tokenizer.encode("Explain how an NVMe SSD controller schedules writes.",
                           add_special_tokens=False)
m = eng.model
m.begin_prompt()
logits, mh = m.forward(torch.tensor(ids, dtype=torch.long, device="cuda"), 0,
                       prefill=True, need_logits=True)
if eng.spec:
    m.dspark_seed(mh, 0)

fd = eng.fast
T = fd.ids.numel()
block = torch.full((T,), int(logits[-1].argmax()), dtype=torch.long, device="cuda")

# ---- v1 reference. Stopped early if NL < n_layers, in which case h is the observable.
S0 = m.c.len
if NL == n_layers:
    fd.step(block, S0, {})
    ref = fd.logits.clone()
    what = "logits"
else:
    # drive the same NL layers through v1's OWN resolve, so the reference is v1's slot mapping
    rl0 = RealLeaves(os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin"), eng.arena).attach(eng, block)
    rl0.begin_step(0)
    for L in range(NL):
        r0 = rl0.layer_a(L)
        fd.slots.copy_(eng.store.resolve(L, r0.opaque, False))
        rl0.layer_b(L, r0)
    ref = fd.h.clone()
    what = "h"
    rl0.close()

# ---- v2. Its own slot table, its own loader, its own reads.
if NL == n_layers:
    m.c.rollback(m.c.len - T)          # the reference step advanced the cache; put it back
m.begin_prompt()
from enginev2 import drivers as v2drivers            # noqa: E402
v2drivers.N_LAYERS = NL                              # drivers imported the name, so rebind it THERE
V2Engine = v2drivers.Engine

arena_slots = eng.store.n_slots
transient = 8                                        # ExpertSlots' floor; v2 does not use the ring here
rl = RealLeaves(os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin"), eng.arena).attach(eng, block)
if MUT:
    _bind = rl.bind_slots
    rl.bind_slots = lambda route, slot_of: _bind(
        route, {k: v for k, v in zip(sorted(slot_of), [slot_of[k] for k in sorted(slot_of)][1:] +
                                     [slot_of[sorted(slot_of)[0]]])})
e2 = V2Engine(Policy(), evict="lru", lru_slots=arena_slots - transient, transient_slots=transient,
              n_workers=int(os.environ.get("V2_WORKERS", 8)),
              staging=int(os.environ.get("V2_STAGING", 8)),
              expert_read_qd=int(os.environ.get("V2_QD", 8)),
              h2d_inflight=int(os.environ.get("V2_H2D", 2)), leaves=rl)
t0 = time.perf_counter()
c = e2.decode(1)
dt = time.perf_counter() - t0
got = (fd.logits if what == "logits" else fd.h).clone()
e2.close()
rl.close()

same = bool(torch.equal(ref, got))
d = (ref.float() - got.float()).abs().max().item()
print(f"  layers {NL}  arena slots {arena_slots}  fetches {c.fetches}  "
      f"staging peak {e2.loader.stage.peak_in_use}  {dt:.2f}s")
print(f"  {what}: exact match {same}, max |delta| {d:.3e}")
sys.exit(0 if same != MUT else 1)
