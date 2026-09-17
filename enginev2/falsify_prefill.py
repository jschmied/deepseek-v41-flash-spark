"""Is v2's prefill bitwise identical to v1's shipped layer-major pass?

`RealLeaves.prefill_attn` / `prefill_moe` are a transcription of
`engine/model.py::encoder_prefill_layer_major`, split where the driver needs to sit: attention and
the router before the reads, the MoE and the HC residual after them. A transcription is only worth
anything if it reproduces the original exactly, and the original is the path the server takes --
DSV41_LAYER_MAJOR and DSV41_SWA_REPLAY both default to "1" and neither is set in .env.

ARM=v1 runs `encoder_prefill_layer_major` directly. ARM=v2 drives `Driver.prefill_chunked` over the
same layers. Both then take the same decoder replay, and both dump the state the prompt produced.

TWO PROCESSES, one arm each -- not one process running both. The in-process rewind
(`c.rollback(0)` + `begin_prompt`) does not restore the compressor's `pending` or the transient
ring, and `ab_oracle_prefetch.py` records what that cost: three jobs' worth of results that were
harness artefacts. A prefill is exactly the state a rewind fails to restore.

    ARM=v1 OUT=/tmp/pf_v1.pt python enginev2/falsify_prefill.py
    ARM=v2 OUT=/tmp/pf_v2.pt python enginev2/falsify_prefill.py
    python enginev2/falsify_prefill.py --compare /tmp/pf_v1.pt /tmp/pf_v2.pt

Logits alone are too forgiving -- argmax survives a drifting residual stream -- so the replay
buffer's `h` and `pre_mix`, which carry layer to layer, are compared too, and `c.len` and the
prompt-cache checkpoint keys are checked because dropping them is silent until the NEXT turn.
"""
import os, sys, torch

if sys.argv[1:2] == ["--compare"]:
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    bad = 0
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            print(f"  {k:<12} MISSING from {'v1' if k not in a else 'v2'}"); bad += 1; continue
        x, y = a[k], b[k]
        if not torch.is_tensor(x):
            ok = x == y
            print(f"  {k:<12} equal {ok}   v1={x} v2={y}")
            bad += 0 if ok else 1
            continue
        d = (x.float() - y.float()).abs().max().item()
        exact = torch.equal(x, y)
        print(f"  {k:<12} exact match {exact}   max |delta| {d:.3e}")
        bad += 0 if exact else 1
    print("  PREFILL GATE", "PASS" if not bad else f"FAIL ({bad} differ)")
    sys.exit(1 if bad else 0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
os.chdir(ROOT)
from engine.v41_engine import V41Engine                    # noqa: E402
from enginev2.real import RealLeaves                       # noqa: E402
from enginev2 import drivers as v2drivers                  # noqa: E402
from enginev2.sched import Policy                          # noqa: E402

ARM = os.environ["ARM"]
eng = V41Engine(os.path.expanduser("~/dsv41-lean"), max_seq=8192,
                arena_gb=float(os.environ.get("ARENA_GB", 79)), spec=True, expert_format="cb3")
m = eng.model
prompt = ("Write a detailed technical explanation of how an NVMe SSD controller schedules writes, "
          "why the flash translation layer matters for tail latency under a mixed read/write "
          "workload, and how a host can measure the effect without vendor telemetry. ") * 12
ids = torch.tensor(eng.tokenizer.encode(prompt, add_special_tokens=False)[:2048],
                   dtype=torch.long, device="cuda")
m.c.rollback(0)
m.begin_prompt()

if ARM == "v1":
    m.c.checkpoint(0)
    m.encoder_prefill_layer_major(ids, 0, eng.store, eng.store.arena, m.args.n_routed_experts)
else:
    v2drivers.N_LAYERS = eng.args.n_layers
    rl = RealLeaves(os.environ["DSV41_CB3_CACHE"], eng.arena)
    rl.attach(eng, None, spec=True, temperature=0.0, first_token=0)
    # A REAL transient ring: prefill lives in it. bench_tokens runs transient_slots=8 because decode
    # never touches the ring; a prefill with 8 raises "transient ring exhausted" on chunk 0 (v1 job
    # 230 recorded exactly that), so the split here mirrors the shipped engine's.
    e2 = v2drivers.Engine(Policy(), evict="lru",
                          lru_slots=eng.store.n_slots - eng.store.transient_slots,
                          transient_slots=eng.store.transient_slots,
                          n_workers=8, staging=8, expert_read_qd=8, h2d_inflight=2, leaves=rl)
    # Start from the SAME resident set v1 would see, or the two arms differ in how many reads they
    # must issue and the comparison is not of the prefill but of the cache it inherited.
    for k, slot in eng.store.lru.items():
        e2.slots.lru[k] = slot
        e2.slots.slot_key[slot] = k
        e2.slots.gen[slot] = e2.slots.gen.get(slot, 0)
        e2.slots.evict.on_insert(k, slot, 0)
    e2.slots.free_lru = [s for s in e2.slots.free_lru if s not in e2.slots.slot_key]
    m.c.checkpoint(0)
    n_chunks = rl.begin_prefill(ids, 0)
    for L in range(m.args.candidate_source_layer + 1):
        e2.prefill_chunked(L, n_chunks)
    rl.finish_prefill()
    e2.close()

logits, mh, s_rep = m.decoder_replay(need_logits=True)
out = {
    "logits": logits.detach().clone(),
    "mh": mh.detach().clone() if torch.is_tensor(mh) else torch.zeros(1),
    "rep_h": torch.cat(m._rep["h"]).detach().clone(),
    "rep_pre_mix": torch.cat(m._rep["pre_mix"]).detach().clone(),
    "c_len": int(m.c.len),
    "ckpt_keys": str(sorted(m.c._ckpt.keys())),
    "s_rep": int(s_rep),
}
torch.save(out, os.environ["OUT"])
print(f"  ARM={ARM} wrote {os.environ['OUT']}  c.len={out['c_len']} "
      f"ckpt={out['ckpt_keys']} logits{tuple(logits.shape)}")
