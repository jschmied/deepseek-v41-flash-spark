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

## Job 560 — V2Engine serves real requests (2026-09-17)

First time `V2Engine.generate()` ran on weights. Six cases in one process, so case 6 is a genuine
second request. Against `be125be`, arena 40 GB.

    1 greedy       24 tokens  hit 0.698  nvme 45.47 GB  accept 1.92  ttft 5.539 s
    2 sampled      seed 4242 twice -> IDENTICAL FALSE                      <- FAILS, see below
    3 stop id      stopped at 8 tokens, the stop id is in the output
    4 max_tokens   asked 7, got 7
    5 close        closed mid-stream, the epilogue ran
    6 after close  12 tokens  nvme 16.25 GB  steps 6  accept 2.17  decode 2.41 tok/s

Case 6 is the one the review asked for: a later request reports its OWN nonzero steps and
acceptance, and `nvme_gb` 16.25 is per-request rather than the 45.47 + 6.82 that a cumulative
counter would have shown. v2's warm start also ran through v2's own loader: 2,367 experts resident
of 2,367 ranked, 32.6 GB in 7 s.

**BOTH RUNS DECODED ENGRAM-ABLATED, found 2026-09-17 after the fact.** `attach()` nulled
`self.engram`, and V2Engine installs the source at construction and then attaches -- so it was
wired, wiped, and wiped again per request. `layer_a`'s `if self.engram is not None: deliver(...)`
never ran; the driver still issued the engram reads and waited on them, and `begin_step` zeroed the
rows. That is `engram_ablate`, a DIFFERENT MODEL rather than a slower one. **The token streams
above, and job 565's divergence, were produced by it.** The pass/fail of cases 3-6 still holds --
they are about control flow -- but nothing about token identity or throughput here is comparable to
v1. Fixed by making engram provider-lifetime state; re-run before quoting any of these tokens.

**Case 2 is open, and the cause is not yet known.** Either the seed path is wrong, or the MoE is not
bit-reproducible and `multinomial` amplifies what `argmax` absorbs at temperature 0 -- which would
make the expectation itself wrong, not the code. Job 565 separates them with a greedy double-run
that involves no RNG at all. Do not quote case 2 as a defect until it reports.

## Job 575 — the x2 divergence is the TARGET, not the accept path (2026-09-17)

Warm graphs (a throwaway generation captures both parities first, so first-use capture is out) and
a per-step verify trace. Engram live and delivering: `engram_ablated 0`. Greedy, temperature 0, so
no RNG anywhere. Step 0 of two runs:

| field | run 1 | run 2 |
|---|---|---|
| `tok`, `drafts`, `a_n`, `c_len` | identical | identical |
| `argmax` | 270, **10869**, 18505, 1505, 270, 1205 | 270, **5085**, 7855, 305, 30698, 270 |
| `margin` | 0.25, 0.125, 0.125, 0.625, 0.0, 0.125 | 0.25, 0.625, 0.125, 0.0, 0.625, 0.25 |

Same block in, same drafts, the same number accepted and the same cache length out — and a
different target argmax from row 1 onward. **The target forward is not reproducible**, at margins of
0 to a few ulp on a bf16 logit scale.

That clears two things I had suspected and stated: the accept/commit path (`a_n` and `c_len` match)
and the drafter (`drafts` match). It does NOT yet say whether the engine is nondeterministic on its
own or whether v2's scheduling causes it — the MoE reduce is shared, but v2 changes when experts
land, and an order-dependent reduce would make the target depend on the schedule. Job 570 uses v1
as the baseline to separate those.

## Not yet gated

`V2Engine` has served real requests (job 560) but has NOT been through the HTTP layer: `--engine v2`
is wired and unexercised, and the grammar gate has only been run against a CPU fake, never against
`server/tool_grammar.py`.

**And the v2 decode path is not reproducible run to run** (jobs 565, 575), where v1 is. Until that
is understood, no v2 throughput or quality number should be compared against v1's.

**Note on what "v2" means in these jobs:** `V2Engine` defaults to `Policy()`, which is all-True --
that is V1 semantics. So every v2 figure here is the V2 cache, loader, provider and driver under V1
toggles: the V2 EXECUTION PATH, not the V2 SCHEDULE. The all-false `sched.V2` policy has not been
measured end to end at all.
