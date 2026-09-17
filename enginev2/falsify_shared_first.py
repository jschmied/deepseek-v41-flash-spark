"""Is the shared-expert split bitwise identical to the unsplit graph B?

The existing gates (falsify_stage5_logits, falsify_engram) compare v2 against v1 INSIDE ONE
PROCESS. They cannot gate this change, because DSV41_SHARED_FIRST is read at import time and
therefore applies to both arms at once: two arms that are wrong in the same way would agree at
0.000e+00 and the gate would pass a broken build.

So this dumps the state that a step produces and the caller runs it TWICE, once with the flag and
once without, comparing the two files. That makes the reference arm the engine as it ships today,
which is the only baseline that is physically realizable here.

    DSV41_SHARED_FIRST=0 OUT=/tmp/off.pt python enginev2/falsify_shared_first.py
    DSV41_SHARED_FIRST=1 OUT=/tmp/on.pt  python enginev2/falsify_shared_first.py
    python enginev2/falsify_shared_first.py --compare /tmp/off.pt /tmp/on.pt

Logits alone are too forgiving: argmax survives small perturbations and the token stream would look
identical while the residual stream drifted. So `h` and `pre_mix` -- the state carried from layer to
layer -- are compared too, and they are where a wrong sh_out shows up first.
"""
import os, sys, torch

if sys.argv[1:2] == ["--compare"]:
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    bad = 0
    # sh_out is expected to differ: it is untouched (zeros) with the flag off and holds the last
    # layer's shared output with it on. The gate is about what the MODEL produces.
    for k in sorted(k for k in a if k != "sh_out"):
        d = (a[k].float() - b[k].float()).abs().max().item()
        exact = torch.equal(a[k], b[k])
        print(f"  {k:<10} exact match {exact}   max |delta| {d:.3e}")
        bad += 0 if exact else 1
    print("  SHARED-FIRST GATE", "PASS" if not bad else f"FAIL ({bad} tensors differ)")
    sys.exit(1 if bad else 0)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# THIS checkout -- see RealLeaves.ROOT in enginev2/real.py. This used to be a hardcoded
# ~/git/deepseek-v41-flash-spark: a DIFFERENT working tree of the SAME repo, on a different
# branch, whose engine/ silently shadowed this one. Every real-graph number this branch
# produced came from code that was not committed here.
V1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [V1, os.path.join(V1, "tools")]
os.chdir(V1)
from engine.v41_engine import V41Engine                    # noqa: E402
from engine import fastdecode as FD                        # noqa: E402
from enginev2.real import RealEngramSource, RealLeaves     # noqa: E402
from enginev2.sched import Policy                          # noqa: E402

STEPS = int(os.environ.get("STEPS", 8))
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

# THE POINT OF THIS: drive the REAL fastdecode path. The first version of this script called
# m.forward in a loop, which does NOT touch fastdecode -- eng.fast exists but only the spec loop
# replays its graphs. Both arms came back with fd.logits all zeros and the gate "passed" by
# comparing two no-ops. The v2 driver replays gA/gB (and gS under the flag), so it exercises
# exactly the code this change touches.
from enginev2 import drivers as v2drivers                  # noqa: E402
v2drivers.N_LAYERS = eng.args.n_layers
rl = RealLeaves(os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin"), eng.arena)
rl.attach(eng, None, spec=True, temperature=0.0, first_token=first)
src = RealEngramSource(eng, rl) if os.environ.get("ENGRAM", "1") == "1" else None
rl.engram = src
e2 = v2drivers.Engine(Policy(), evict=os.environ.get("EVICT", "age_over_freq"),
                      lru_slots=eng.store.n_slots - 8, transient_slots=8, n_workers=8, staging=8,
                      expert_read_qd=8, h2d_inflight=2, leaves=rl, engram=src)
print(f"  SHARED_FIRST={FD.SHARED_FIRST}  leaves.shared_first={rl.shared_first}  steps={STEPS}")
e2.decode(STEPS)
torch.cuda.synchronize()
out = {"logits": fd.logits.detach().clone(), "h": fd.h.detach().clone(),
       "pre_mix": fd.pre_mix.detach().clone(), "sh_out": fd.sh_out.detach().clone(),
       "y": fd.y.detach().clone()}
nz = {k: int((v != 0).sum()) for k, v in out.items()}
# A GATE THAT CANNOT BE VACUOUS. If the tensors being compared are all zero, the arms agree because
# nothing ran, not because the change is sound -- which is how the first version of this script
# passed. Refuse to write such a file.
dead = [k for k in ("logits", "h", "y") if nz[k] == 0]
if dead:
    raise RuntimeError(f"refusing to write a vacuous reference: {dead} are all zero, so the "
                       f"fastdecode path did not run. nonzero counts: {nz}")
if FD.SHARED_FIRST and nz["sh_out"] == 0:
    raise RuntimeError("SHARED_FIRST is on but sh_out is all zero -- the shared graph never ran, "
                       "so this arm is not testing the split at all")
torch.save(out, os.environ["OUT"])
print(f"  wrote {os.environ['OUT']}  nonzero {nz}")
