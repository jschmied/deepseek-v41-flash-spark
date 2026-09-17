# enginev2 gate results

Recorded here because the queue logs are not in the repository, and a commit message saying a gate
is green is not evidence. Each entry names the job, the commit it ran against, and the numbers.

## Job 520 — real graphs, provenance, and the resident-first split (2026-09-17)

Ran `enginev2/bench_tokens.py` against `f30a47b` with `DSV41_SHARED_FIRST=1 DSV41_RESIDENT_FIRST=1`.

    engine.fastdecode -> /home/jschmied/git/dsv41-enginev2/engine/fastdecode.py
    UNIQUE residency 8083/10105 = 80.0 %
    PAIR residency 14781/17280 = 85.5 %
    TOKENS 27 -> 2.88 tok/s
    PROVENANCE OK: bound from the v2 worktree

The provenance line is the point: before `f30a47b` this branch resolved `engine` from a different
working tree of the same repo on a different branch, so every real-graph number it had produced
described code it did not contain.

Bitwise, `tools/test_moe_split_bitwise.py`: `SPLIT IS BITWISE`, `max |delta| 0.000e+00` at T=6, T=1
and T=5, both the block form and the phase API with shared h/parts. Neither variant masks the
original `slots` tensor -- both mask `block_slot`. Commit 66aa464 exists *because* masking
`slots` was tried: its -1 sentinel group exceeded BM and overflowed into the next block's pair
list, and it passed on T=1 because the second phase was empty. The earlier "masked-slots form"
label here invited someone to try it again.

## Job 536 — v2 prefill is bitwise identical to v1's layer-major pass (2026-09-17)

Against `74cb18c`. Two processes per arm, because the in-process rewind does not restore the
compressor's `pending` or the transient ring and a prefill is exactly that state.

| compared | logits | mh | rep_h | rep_pre_mix | c.len | s_rep | ckpt keys |
|---|---|---|---|---|---|---|---|
| v1 vs v2 (`global_barrier=1`) | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 541 = 541 | 413 = 413 | [0] = [0] |
| v1 vs v2 (`global_barrier=0`) | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 541 = 541 | 413 = 413 | [0] = [0] |

Both arms PASS. The second is the one v1 cannot express at all -- `join_pending` is a barrier over
every chunk's reads, so "chunk k's FFN waits only for chunk k's experts" has no v1 equivalent. That
it is bitwise means **D3 is a free toggle at prefill**, which is what makes the v2 schedule usable
rather than merely different.

SCOPE: a 2,048-token prompt through `candidate_source_layer`. It proves the transcription; it says
nothing about long prompts or the decoder half.

## Job 555 — the arena repoint changed nothing (2026-09-17)

Against `86c80fb`, which moved the arena from `m.store.arena` to `m.arena`. Needed a gate rather
than an argument because the decode graphs CAPTURE pointers into those tensors: "it is the same
object" is a claim about source, not about what was baked into 40 layers of graphs.

Gate 1 (real graphs + provenance) passed. Gate 2 re-ran job 536's comparison in full:

    v1 vs v2   : logits / mh / rep_h / rep_pre_mix all 0.000e+00, c.len 541, s_rep 413 -- PASS
    v1 vs v2f  : logits / mh / rep_h / rep_pre_mix all 0.000e+00, c.len 541, s_rep 413 -- PASS

## Not yet gated

`V2Engine` has never served a request. Its prefill half is covered above; the API half -- bursts,
stop handling, `max_tokens`, the grammar gate against a real gate, penalties, two consecutive
requests, client close -- has only CPU fakes behind it.
