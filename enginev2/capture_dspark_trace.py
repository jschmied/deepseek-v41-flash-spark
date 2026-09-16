"""Capture the DSpark drafter's routing alongside the backbone's per-layer MISSES.

The question this exists to answer: the drafter runs BEFORE the verify step it feeds, so its
routing is the earliest signal the engine has about what the backbone is about to do. Does it say
anything about which BACKBONE experts will MISS?

That matters because the offline study (8419ade) closed co-occurrence and recurrence over expert
identity -- the residual misses are the cold tail LRU leaves behind. DSpark's claim is not better
accuracy, it is LEAD TIME: a prediction for layer 20 can exist before layer 0 starts, and job 295
showed the oracle's value was head start, not readiness (ready_hit 96 against late_hit 2292).

NOTE THE VOCABULARY GAP, which is why this is a capture and not a lookup: the drafter routes its
OWN 3 x 128 experts, resident in a separate arena. There is no identity mapping to the backbone's
384. Any predictor has to learn one.

Runs the REAL loop -- dspark_draft, verify, rollback -- so the token sequence, the accept length
and therefore the route sequence are the served ones, not this branch's full-acceptance stand-in.

    DSV41_DRAFT_TRACE=1 DSV41_GRAPHS=1 DSV41_CB3_CACHE=... python enginev2/capture_dspark_trace.py [steps] [out.jsonl]
"""
import json
import os
import sys

import torch

V1 = os.path.expanduser("~/git/deepseek-v41-flash-spark")
sys.path[:0] = [V1, os.path.join(V1, "tools")]
os.chdir(V1)
from engine.v41_engine import V41Engine              # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/ds41-queue/logs/dspark-trace.jsonl")
if os.environ.get("DSV41_DRAFT_TRACE") != "1":
    raise RuntimeError("set DSV41_DRAFT_TRACE=1; without it the drafter's routing is not captured")

eng = V41Engine(os.path.expanduser("~/dsv41-lean"), max_seq=8192,
                arena_gb=float(os.environ.get("ARENA_GB", 40)),
                spec=True, expert_format="cb3")
m, fd = eng.model, eng.fast
assert fd.draft_trace is not None, "draft trace buffers were not allocated"
prompt = ("Write a short technical explanation of how an NVMe SSD controller schedules writes, "
          "why the flash translation layer matters for tail latency, and how a host can measure it.")
ids = eng.tokenizer.encode(prompt, add_special_tokens=False)
m.begin_prompt()
lg, mh = m.forward(torch.tensor(ids, dtype=torch.long, device="cuda"), 0, prefill=True,
                   need_logits=True)
m.dspark_seed(mh, 0)
tok = int(lg[-1].argmax())
layer_ids = tuple(eng.args.engram_layer_ids)
T = fd.ids.numel()

# the backbone's per-layer route and miss set, recorded as each layer resolves
per_layer = []
_orig = fd._resolve
def spy(L):
    uniq = sorted(set(fd.route_idx.flatten().tolist()))
    # WHICH experts miss, not how many. A count cannot be scored against a predictor that names
    # experts, and the first version of this capture recorded only the count.
    res = eng.store.lru
    miss_ids = [e for e in uniq if (L, e) not in res]
    r = _orig(L)
    per_layer.append({"L": L, "uniq": uniq, "miss": len(miss_ids), "miss_ids": miss_ids})
    return r
fd._resolve = spy

out = open(OUT, "w")
n = 0
with torch.inference_mode():
    for step in range(STEPS):
        pos = m.c.len
        # fd.draft, NOT m.dspark_draft: the latter is the model's own eager path (Model.block over
        # the MTP weights) and never touches fastdecode, so the capture buffers stayed zero. The
        # production loop calls self.fast.draft(); so does this.
        drafts, _q = fd.draft(tok, pos - 1, 0.0)
        # the drafter has now run: its routing is what we are testing as a predictor
        d_idx = fd.draft_trace["idx"].tolist()
        if step == 0 and not any(any(any(p) for p in k) for k in d_idx):
            raise RuntimeError("drafter routing is all zeros -- the capture buffers are not being "
                               "written; check DSV41_DRAFT_TRACE was set BEFORE FastDecoder was "
                               "built, since the writes are decided at graph-capture time")
        d_top = fd.draft_trace["score"].topk(8, dim=-1)
        block = torch.cat([torch.tensor([tok], device="cuda"), drafts])
        h_np = m.hash_state(block[None], pos)[0].cpu().numpy()
        futs = {L: (eng.eg_pool.submit(eng.tables[L].read_raw, h_np[:, li, :]),
                    eng.tables[L].to_device) for li, L in enumerate(layer_ids)}
        per_layer.clear()
        logits, _mh = fd.step(block, pos, lambda: futs)
        am = logits.argmax(-1)
        acc = am[:T - 1].eq(drafts).to(torch.int32).cumprod(0).sum().item()
        cand = am.tolist()
        m.c.rollback(pos + acc + 1)                   # the real cache advance: not the whole block
        rec = {
            "step": step, "pos": pos, "accept": acc, "tok": tok,
            "miss_total": sum(x["miss"] for x in per_layer),
            "drafts": drafts.tolist(),
            "d_idx": d_idx,                            # [3][T_DRAFT][3] drafter expert ids
            "d_top_idx": d_top.indices.tolist(),       # [3][T_DRAFT][8] strongest drafter experts
            "d_top_val": [[[round(v, 4) for v in p] for p in k] for k in d_top.values.tolist()],
            "layers": [{"L": r["L"], "uniq": r["uniq"], "miss": r["miss"],
                        "miss_ids": r["miss_ids"]} for r in per_layer],
        }
        out.write(json.dumps(rec) + "\n")
        n += 1
        tok = cand[acc]
        if n % 25 == 0:
            out.flush()
            print(f"  {n} steps, accept mean so far pending, c.len {m.c.len}", flush=True)
out.close()
fd._resolve = _orig
print(f"  wrote {n} steps -> {OUT}")
