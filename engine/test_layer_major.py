"""Layer-major encoder prefill must not change what the model writes.

The transpose visits layers outermost so each layer's experts are read once for the whole prompt
instead of once per chunk. It is the same arithmetic on the same weights in a different visiting
order, so greedy decoding must produce the tokens chunk-major produces.

It is NOT expected to be bit-identical: the MoE of a layer is now routed for every chunk before any
of it is applied, and the shared/routed GEMMs see the same shapes but a different allocation order.
What must hold is that the tokens agree -- and that the prompt is long enough to cross several
chunk boundaries, or the two paths are trivially the same.

Run: DSV41_CB3_CACHE=... python engine/test_layer_major.py --model-dir ... [--max-tokens 60]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from engine.v41_engine import V41Engine, log  # noqa: E402

FILLER = "The following is background material that should be summarised at the end. " * 420
PROMPT = ("Here is a document.\n\n" + FILLER +
          "\n\nName one topic it covers, in one short sentence.")


def run(eng, ids, max_tokens):
    out = []
    for t in eng.generate(ids, max_tokens=max_tokens, temperature=0.0):
        out += t
    return out, dict(eng.last_stats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.expanduser("~/dsv41-lean"))
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--engine-kwargs", default="{}")
    a = ap.parse_args()
    kw = json.loads(a.engine_kwargs)
    kw.setdefault("trace_stats", "results/trace-unmasked-20260913/stats/coverage.json")
    if os.environ.get("DSV41_CB3_CACHE"):
        kw.setdefault("expert_format", "cb3")
    kw.setdefault("transient_slots", 400)
    kw.setdefault("keep_free_gb", 6.0)

    import engine.v41_engine as E
    eng = V41Engine(a.model_dir, max_seq=16384, **kw)
    sys.path.insert(0, os.path.join(eng.model_dir, "encoding"))
    from encoding import encode_messages  # noqa: E402
    pr = encode_messages([{"role": "user", "content": PROMPT}], thinking_mode="chat")
    ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
    n_chunks = (len(ids) + E.MAX_CHUNK - 1) // E.MAX_CHUNK
    log(f"prompt {len(ids)} tokens = {n_chunks} chunks of {E.MAX_CHUNK}")
    if n_chunks < 3:
        log("FAIL: fewer than 3 chunks -- the transpose has almost nothing to do, test is vacuous")
        return 2

    E.LAYER_MAJOR = False
    ref, st_ref = run(eng, ids, a.max_tokens)
    log(f"chunk-major: prefill {st_ref['prefill_s']}s, nvme {st_ref['nvme_gb']} GB, "
        f"prefill misses {st_ref['prefill_expert_misses']}")

    E.LAYER_MAJOR = True
    got, st_lm = run(eng, ids, a.max_tokens)
    log(f"layer-major: prefill {st_lm['prefill_s']}s, nvme {st_lm['nvme_gb']} GB, "
        f"prefill misses {st_lm['prefill_expert_misses']}")

    failed = 0
    n = min(len(ref), len(got))
    first = next((j for j in range(n) if ref[j] != got[j]), None)
    if first is None and len(ref) == len(got):
        log(f"PASS: {len(ref)} tokens identical chunk-major and layer-major")
    else:
        failed = 1
        log(f"FAIL: first divergence at token {first} of {n} (chunk {len(ref)}, layer {len(got)})")
        lo = max(0, (first or n) - 12)
        log("  chunk-major: " + repr(eng.tokenizer.decode(ref[lo:(first or n) + 24])))
        log("  layer-major: " + repr(eng.tokenizer.decode(got[lo:(first or n) + 24])))
    # the whole point: the second arm must have read fewer experts, not merely the same
    if st_lm["prefill_expert_misses"] >= st_ref["prefill_expert_misses"]:
        log(f"FAIL: layer-major did not reduce prefill loads "
            f"({st_lm['prefill_expert_misses']} vs {st_ref['prefill_expert_misses']}) -- "
            f"either the transpose did not run or the ring is refilling per chunk")
        failed = 1
    else:
        r = st_ref["prefill_expert_misses"] / max(st_lm["prefill_expert_misses"], 1)
        log(f"prefill expert loads {st_ref['prefill_expert_misses']} -> "
            f"{st_lm['prefill_expert_misses']} = {r:.2f}x fewer; "
            f"NVMe {st_ref['nvme_gb']} -> {st_lm['nvme_gb']} GB")
    print("== VOID ==" if failed else "== ALL DONE ==")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
