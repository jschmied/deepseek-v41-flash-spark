"""The extend-only prompt cache must not change what the model writes.

Turn 2 of a conversation is turn 1's prompt + turn 1's reply + new user text, so with
DSV41_PROMPT_CACHE=1 the engine resumes from a prefill-chunk checkpoint instead of re-prefilling
the whole thing. Greedy turn 2 must then produce what greedy turn 2 produces on a cold engine.

It is NOT expected to be bit-identical arithmetic: the reused positions were computed under a
different chunk alignment, so their GEMM shapes differed. What must hold is that the tokens agree
-- and on a long enough turn, that the second turn actually prefilled far fewer tokens than it
was given, otherwise the test passes by never having taken the fast path at all.

Run: python engine/test_prompt_cache.py --model-dir ... [--max-tokens 80]
The engine is built once; the cold arm is produced by clearing the cache between turns.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from engine.v41_engine import V41Engine, log  # noqa: E402

# Long enough that turn 2 crosses several 2048-token chunk boundaries, or the resume point is 0
# and the test proves nothing. FILLER is inert prose the model does not need to reason about.
FILLER = ("The following is background material that should be summarised at the end. " * 240)
TURN1 = "Here is a document.\n\n" + FILLER + "\n\nName one topic it covers, in one short sentence."
TURN2 = "Now name a second topic, in one short sentence."


def encode(eng, messages):
    sys.path.insert(0, os.path.join(eng.model_dir, "encoding"))
    from encoding import encode_messages  # noqa: E402
    pr = encode_messages(messages, thinking_mode="chat")
    return eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)


def run(eng, ids, max_tokens):
    out = []
    for t in eng.generate(ids, max_tokens=max_tokens, temperature=0.0):
        out += t
    return out, dict(eng.last_stats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--engine-kwargs", default="{}")
    a = ap.parse_args()
    if os.environ.get("DSV41_PROMPT_CACHE") != "1":
        log("DSV41_PROMPT_CACHE is not 1 -- this test would compare two identical cold paths")
        return 2
    kw = json.loads(a.engine_kwargs)
    kw.setdefault("trace_stats", "results/trace-full-20260910/stats/coverage.json")
    # Match what the server runs. The engine defaults to expert_format="fp4"; with
    # DSV41_CB3_CACHE set that hands CB3 planes to an FP4 arena and dies in load_slot.
    if os.environ.get("DSV41_CB3_CACHE"):
        kw.setdefault("expert_format", "cb3")
    kw.setdefault("transient_slots", 400)
    kw.setdefault("keep_free_gb", 6.0)
    eng = V41Engine(a.model_dir, max_seq=16384, **kw)

    msgs1 = [{"role": "user", "content": TURN1}]
    ids1 = encode(eng, msgs1)
    reply1, st1 = run(eng, ids1, a.max_tokens)
    text1 = eng.tokenizer.decode(reply1)
    log(f"turn 1: {st1['prompt_tokens']} prompt tokens, reused {st1.get('prompt_cache_reused')}, "
        f"prefill {st1['prefill_s']}s")

    msgs2 = msgs1 + [{"role": "assistant", "content": text1}, {"role": "user", "content": TURN2}]
    ids2 = encode(eng, msgs2)

    warm, st_warm = run(eng, ids2, a.max_tokens)          # resumes from turn 1
    eng._reset()                                          # throw the cache away
    cold, st_cold = run(eng, ids2, a.max_tokens)          # same prompt, full prefill
    log(f"turn 2 warm: reused {st_warm.get('prompt_cache_reused')} of {st_warm['prompt_tokens']}, "
        f"prefilled {st_warm.get('prompt_tokens_prefilled')}, prefill {st_warm['prefill_s']}s, "
        f"nvme {st_warm['nvme_gb']} GB")
    log(f"turn 2 cold: reused {st_cold.get('prompt_cache_reused')} of {st_cold['prompt_tokens']}, "
        f"prefilled {st_cold.get('prompt_tokens_prefilled')}, prefill {st_cold['prefill_s']}s, "
        f"nvme {st_cold['nvme_gb']} GB")

    failed = 0
    if not st_warm.get("prompt_cache_reused"):
        log("FAIL: turn 2 did not resume -- the comparison below is vacuous")
        failed += 1
    n = min(len(cold), len(warm))
    first = next((j for j in range(n) if cold[j] != warm[j]), None)
    if first is None and len(cold) == len(warm):
        log(f"PASS: {len(cold)} tokens identical warm and cold")
    else:
        failed += 1
        log(f"FAIL: first divergence at token {first} of {n} (cold {len(cold)}, warm {len(warm)})")
        lo = max(0, (first or n) - 12)
        log("  cold: " + repr(eng.tokenizer.decode(cold[lo:(first or n) + 24])))
        log("  warm: " + repr(eng.tokenizer.decode(warm[lo:(first or n) + 24])))
    if st_warm["prefill_s"] >= st_cold["prefill_s"]:
        log(f"WARN: warm prefill was not faster ({st_warm['prefill_s']}s vs {st_cold['prefill_s']}s)")
    print("== VOID ==" if failed else "== ALL DONE ==")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
