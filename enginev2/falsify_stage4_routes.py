"""Stage 4 falsifier: RealLeaves' graph half must reproduce v1's routing exactly.

Drives one decode step twice on the same prefilled state -- once through v1's
FastDecoder.step(), once through RealLeaves.begin_step/layer_a/layer_b -- and
compares route_idx layer by layer. Routes are chained: layer L's route depends
on every preceding layer's MoE output, so 40/40 equality is a statement about
the whole A->resolve->B chain, not just the router.

Needs the box (a full engine load, ~90 s). Run:
    DSV41_CB3_CACHE=~/dsv41-cb3/experts-cb3-s3.bin DSV41_DENSE_FP4=attn,wo_a \
    DSV41_HEAD_FMT=fp8 python enginev2/falsify_stage4_routes.py

MUT=1 skips graph B from layer 1 on. That is the mutation check: it must turn
the result red (measured 2026-09-16: 40/40 -> 2/40), or the comparison above is
vacuous and proves nothing.
"""
import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# THIS checkout -- see RealLeaves.ROOT in enginev2/real.py. This used to be a hardcoded
# ~/git/deepseek-v41-flash-spark: a DIFFERENT working tree of the SAME repo, on a different
# branch, whose engine/ silently shadowed this one. Every real-graph number this branch
# produced came from code that was not committed here.
V1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V1)
sys.path.insert(0, os.path.join(V1, "tools"))
os.chdir(V1)

from engine.v41_engine import V41Engine          # noqa: E402
from enginev2.real import RealLeaves             # noqa: E402

MUT = os.environ.get("MUT") == "1"

eng = V41Engine(os.path.expanduser("~/dsv41-lean"), max_seq=8192,
                arena_gb=float(os.environ.get("ARENA_GB", 40)),
                spec=True, expert_format="cb3")
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

# v1: one real step, recording route_idx as each layer resolves.
v1_routes = []
_orig = fd._resolve
fd._resolve = lambda L: (v1_routes.append(fd.route_idx.clone()), _orig(L))[1]
fd.step(block, m.c.len, {})
fd._resolve = _orig

# v2: rewind to the same state and drive the same step through RealLeaves.
m.c.rollback(m.c.len - T)
m.begin_prompt()
rl = RealLeaves(os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin"), eng.arena)
rl.attach(eng, block)
rl.begin_step(0)
same = diff = 0
for L in range(eng.args.n_layers):
    r = rl.layer_a(L)
    if torch.equal(r.opaque, v1_routes[L]):
        same += 1
    else:
        diff += 1
        if diff <= 2:
            print(f"    layer {L}: route differs")
    # graph B needs the slots v1 would have bound, so the next layer's input is right
    fd.slots.copy_(eng.store.resolve(L, r.opaque, False))
    if not (MUT and L >= 1):
        rl.layer_b(L, r)
rl.close()
print(f"  route ids identical to v1: {same}/{eng.args.n_layers} layers (differing {diff})")
sys.exit(0 if (same == eng.args.n_layers) != MUT else 1)
