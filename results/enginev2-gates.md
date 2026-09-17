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

## Jobs 570 + 580 — it is NOT a race; it is deterministic state carryover (2026-09-17)

**570**, engram live on both sides: v1 x2 **identical** (13 tokens); v2 x2 **not**; v1 != v2 from
token 1. v2's very first request in a process differs from v1 immediately.

**580**, the early-readiness discriminator — and the result is not the one the arms were for:

    EARLY_READY=1  run1 [52480, 270, 10869, 294, 18505, 9335, 305, 30123]
                   run2 [52480, 270,  5085, 18505, 9335, 305, 270, 7629]
    EARLY_READY=0  run1 [52480, 270, 10869, 294, 18505, 9335, 305, 30123]   <- identical
                   run2 [52480, 270,  5085, 18505, 9335, 305, 270, 7629]   <- identical

Run 1 is reproducible, run 2 is reproducible, and they differ from each other **the same way in
both arms**. So publishing readiness only after `handle.synchronize()` changes nothing, which
retires the slot-lifecycle hypothesis — and, more importantly, the divergence is **not
nondeterminism at all**. It is state that survives a request, which v1 resets and v2 does not.

That also retires, in order: the accept/commit path (575), the drafter (575), an order-dependent
expert reduce (reasoned out), first-use graph capture (575, warm), RNG and seeding (565), and now
early H2D visibility (580). Job 585 establishes the shape — A B A B, A B B B or A B C D — before
anything else is guessed at. Graph parity was checked and ruled out before queueing: `parity = S % 2`
is derived from the cache length, and every request prefills to the same S.

## Job 585 — the shape is A B B' B'' (2026-09-17)

Four 8-token generations in one process, after a warm-up, greedy:

    gen1  [52480, 270, 10869, 294, 18505, 9335, 305, 30123]   steps=4
    gen2  [52480, 270,  5085, 18505, 9335, 305, 270,  7629]   steps=3
    gen3  [52480, 270,  5085, 18505, 9335, 305, 270,  1167]   steps=4
    gen4  [52480, 270,  5085, 18505, 9335, 305, 270, 60944]   steps=4

**gen2-4 share the first seven tokens and differ only in the eighth**; gen1 differs from token 2.
At 7 tokens, gen1 = gen2 = gen3 exactly — that same seven-token prefix — and only gen4 breaks.

So it is a **one-time transition after the first generation**, plus a smaller effect that reaches
the tail occasionally. Not a two-state toggle (A B A B) and not steady accumulation (A B C D).
`S0=9, parity=1` on every generation, confirming parity is constant as predicted.

The one-time transition is the tractable half. **Correction to what this file said first: v1 does
NOT drop its expert cache between requests.** `_reset()` (engine/v41_engine.py:646) clears the model
caches and the stats and never touches `ExpertStore` or the arena. So persistence itself is not the
defect — v1 persists and stays reproducible. What differs is v2's cache-management SEMANTICS, and
the concrete one is that on a decode hit in the transient ring v1 calls `_promote_transient()`
(experts.py:636, fired at :733), moving the hot expert into the LRU and swapping a donor back into
the ring, while v2's `reserve()` takes the `transient_map` hit and leaves it there. Job 590 resets one half of the
state at a time — model caches, or the expert map — and logs a digest of the mapping per request so
an A → B transition can be lined up against a mapping change rather than inferred.

Note the CB3 kernel writes each (token, top-k) pair to a fixed pair position and reduces in fixed
K,T order, so slot assignment should NOT affect arithmetic. If restoring the mapping changes the
answer, the defect is stale or wrong expert bytes, or a mapping-generation hole — not reduction
order.

## Job 590 — the model-cache reset is not it; the expert arm was unsound (2026-09-17)

    control  req1 map=21fd9cc19a05 hits=0     [52480, 260, 9162, 294, 5085, 89673, 4061, 305]
             req2 map=be873e6c8986 hits=2715  [52480, 270, 5085, 18505, 9335, 305, 270, 1167]
             req3 map=303f04c4cbf2 hits=5578  [52480, 270, 5085, 18505, 9335, 305, 270, 18967]
    meta     IDENTICAL to control, token for token, all three requests
    expert   CRASHED -- "slot collision in reserve()"

**The cache-metadata reset changes nothing.** `rollback(0)` + `begin_prompt()` already cover what v1
clears explicitly, so that hypothesis is retired.

**The expert arm is VOID, and its crash is the proof.** Restoring `lru`/`slot_key` to their old
values without reloading the corresponding experts points slots at bytes request 1 had overwritten;
a complete restore needs the physical bytes, the LRU *order* (not just the pairs), `free_lru`,
`transient_map`, `transient_pos`, `slot_key`, the per-slot generations, no `_pending`/`_displaced`
leftovers, and the eviction policy's own age/count state. A digest of `sorted((layer, expert, slot))`
cannot prove equivalent cache state — that arm was uninterpretable before it ran, and the engine's
assert caught it rather than letting it produce a number.

What the digests do show: the mapping churns every request (`21fd → be87 → 303f`) while `resident`
stays 2367, which lines the A → B transition up against a mapping change instead of leaving it
inferred.

**One caveat on the B′/B″ tail, which should not yet be treated as a second defect.** `V2Engine`
truncates the EMITTED burst to `max_tokens` after `RealLeaves` has already committed the whole
speculative burst, so a request can execute hidden extra tokens on its final step and perturb
residency by an amount that depends on acceptance length — 585 had `steps=3` for gen2 and 4 for the
others. Job 595 asks for 16 tokens and compares the first 8, putting the truncation far past the
compared prefix.

## Job 600 — `--engine v2` serves HTTP (2026-09-17)

The whole surface, first time through `server/app.py`:

    /health OK
    /v1/models  deepseek-v4.1-flash, max_model_len 8192
    request 1   completion_tokens=2 prompt_tokens=11  'Hello!'
    request 2   completion_tokens=2 prompt_tokens=11  'Hello!'
    streaming   7 SSE data lines, usage frame present
    x_engine_stats  engine=v2 tokens=8 ttft=3.111s decode=1.62 tok/s steps=3
                    accept_len_mean=3.0 expert_hit_rate=0.6348 misses=769 prefill_misses=187

Chat completions, SSE, usage accounting, `/health` and `/v1/models` all work, and the stats are
per-request deltas rather than lifetime counters. Note both requests returned the same two tokens:
they hit EOS before reaching the divergence point, so this does NOT contradict the reproducibility
defect — it bounds it to longer generations.

## Job 595 — the transient ring is not the carrier either (2026-09-17)

    control   req1 ring=0   [52480, 260, 9162, 294, 5085, 89673, 4061, 305]
              req2 ring=400 map=95a595dd5819 hits=5889 miss=5008
                            [52480, 270, 3615, 294, 270, 30123, 18505, 343]
    ringcold  req1 ring=0   [52480, 260, 9162, 294, 5085, 89673, 4061, 305]
              req2 ring=0   map=95a595dd5819 hits=5889 miss=5008
                            [52480, 270, 5085, 18505, 9335, 305, 270, 5085]

Cooling the ring does not restore request 1 — it produces a third answer. And request 1 ALSO ran
with `ring=0`, so an empty ring is not what distinguishes it.

**The sharper datum is the pair the arms accidentally produced.** Control req2 and ringcold req2
have the **same LRU pairs digest, the same hits and the same misses** at the start of the request,
and return different tokens. So the `(layer, expert, slot)` pairing is not the carrier — which is
also what job 590's digest could not have told us on its own.

Remaining inside `ExpertSlots`: the **LRU order** and the **per-slot generations**, both functional
and both invisible to a sorted-pairs digest. Job 605 records all three separately.

## Jobs 610, 615, 620 (2026-09-17)

**610 — the V2 policy is a no-op at decode.** `Policy()` (all-true) and `sched.V2` (all-false)
returned identical hit rate, NVMe GB, steps and accept length to the decimal across three reps
(1.70/2.08/3.63 vs 1.69/2.06/3.63 tok/s). Consistent with `decode_layer`'s own note that global and
per-slot waits are structurally the same at decode, since only the current layer has reads
outstanding. **The V2 schedule has still never shown a benefit anywhere.**

**615 — the arena never lies.** 24 live mappings per checkpoint, each expert re-read from the CB3
file into a scratch arena and compared plane by plane against the slot the map points at: **zero
mismatches** after the warm start and after each of three requests. The cache bookkeeping churns
constantly and its contents are always correct, so the cache is exonerated as the carrier.

**620 — it is decode-local.** Three requests, one process, same prompt:

| compared to req1 | logits | mh | rep_h | rep_pre_mix | s_rep |
|---|---|---|---|---|---|
| req2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | equal |
| req3 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | equal |

and the decoded tokens still differ at every request. Prefill and the drafter seed inputs are
identical; the carrier is inside decode. Job 625 traces the first decode step layer by layer.

## Job 625 — the carrier is the PHYSICAL SLOT ASSIGNMENT (2026-09-18)

First decode step, traced layer by layer, request 1 vs request 2, after 620 showed the prefill is
bitwise identical:

    layer 4:  h_in  IDENTICAL       route  IDENTICAL
              slots req1 [[2253, 2497, 2252, 2488, 2502, 2246], ...]
                    req2 [[2253, 2550, 2252, 2541, 2555, 2246], ...]
              h_out req1 ( 4.13925838470459, 10178.1162109375)
                    req2 (-0.3770885467529297, 10174.91015625)

Same input, same route, **the same experts** — 615 verified the mappings byte-for-byte — at
**different physical slots**, and a different output.

**The mechanism is in `tools/fp4_moe.py:389`.** `build_routing_small` does
`order = torch.argsort(flat, stable=True)` **on the slot numbers**, and block ids follow that sorted
order. So the order in which a token's K contributions are accumulated is a function of physical
slot addresses. Move an expert and the same six numbers are summed in a different order: last-ulp
differences, which is exactly the 0-to-few-ulp margins job 575 measured, and argmax flips wherever a
margin is tiny.

This reconciles every earlier result without any of them being wrong. v1 is reproducible because its
slot assignment repeats across requests; v2's churns. The cache contents were always correct (615).
The policy toggles were always irrelevant (610). The accept path, drafter, RNG, parity, ring, early
readiness and metadata were all correctly exonerated — the carrier was never in any of them.

It also promotes the transient-promotion gap
(`test_a_decode_hit_in_the_transient_ring_is_promoted`, currently xfail) from a parity nicety to a
plausible reason v2's assignment never settles.

**Job 630 tests the kernel alone** — same experts, two slot assignments, no engine, no cache, no
scheduler — because the engine-level evidence is circumstantial until the kernel itself is shown to
be slot-order sensitive.

## Not yet gated

`V2Engine` has served real requests (job 560) but has NOT been through the HTTP layer: `--engine v2`
is wired and unexercised, and the grammar gate has only been run against a CPU fake, never against
`server/tool_grammar.py`.

**v2 generation is REQUEST-HISTORY DEPENDENT** (jobs 565, 575, 585, 595) where v1 is reproducible
(570). Job 620 localized it: three requests of one prompt in one process gave **bitwise identical
prefill** — logits, mh, rep_h, rep_pre_mix all `0.000e+00`, s_rep equal — and still decoded to
different tokens. So it is decode-local, and prefill is exonerated along with the expert-cache
contents (615: 24 sampled mappings, zero mismatches). Until it is understood, no v2 throughput or
quality number should be compared against v1's.

**Note on what "v2" means in these jobs:** `V2Engine` defaults to `Policy()`, which is all-True --
that is V1 semantics. So every v2 figure here is the V2 cache, loader, provider and driver under V1
toggles: the V2 EXECUTION PATH, not the V2 SCHEDULE. The all-false `sched.V2` policy has not been
measured end to end at all.
