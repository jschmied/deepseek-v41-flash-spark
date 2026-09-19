# The DSpark drafter should be CB3, and the comment saying otherwise is wrong on both clauses

`engine/v41_engine.py:387` dispatches the MoE kernel on the arena's type, and the comment above it
has carried the standing decision since the CB3 arena shipped:

> The DSpark draft arena stays FP4 whatever the main arena is -- it is 384 experts (7.2 GB), it is
> read five times per step, and a 3-bit drafter would cost acceptance for nothing.

Both clauses are now measured, and both are wrong.

## Clause 1: "would cost acceptance for nothing" -- it costs 1.05 pp, and buys 1.554 GiB

Third-party evidence, not ours: `coolbho3k/DeepSeek-V4.1-Flash-DSpark-EXL3-3bpw` quantizes exactly
these 384 draft experts to 3 bits (EXL3 MUL1) and publishes the serving comparison -- 160 requests,
20 cases, temperatures 0/0.6/1/1.2, two seeds, code + math + prose + multilingual + images + tool
calls:

| | native draft | 3-bit draft |
| --- | --- | --- |
| draft-token acceptance | 54.22 % | **53.17 %** (-1.05 pp) |
| next-token agreement with native | -- | 94.05 % (1,764 held-out predictions) |
| label accuracy | 47.62 % | 47.22 % |
| NLL | 3.57307 | 3.57705 |

Their own caveats are worth carrying: these are campaign-specific checks rather than a standard
benchmark, 107 of 384 experts had no naturally routed held-out example, and the decode figure
(30.53 vs 30.75 tok/s) is on **two** DGX Sparks, so it is not our number. The acceptance delta is
the part that transfers, and it is small.

## Clause 2: "read five times per step" -- that is why CB3 WINS

This was never measured. Job 980 measures it with no pack, no engine and no prompt: two 384-slot
arenas and the two shipped kernels at the drafter's own shapes (T_DRAFT rows, **top-3**, three mtp
calls per pass -- not the main arena's top-6).

| | fp4 | cb3 | ratio | fp4 re-run |
| --- | --- | --- | --- | --- |
| T=5, per draft pass | 4.2757 ms | **3.5507 ms** | **0.830** | 4.2833 ms |
| T=1, per replay pass | 1.4111 ms | **1.2845 ms** | **0.910** | 1.4123 ms |

The interleaved re-run agrees with the first block to 0.2 %, so the ratio is not thermal drift.

CB3 is **17 % faster** at the draft-block shape, and the mechanism is the obvious one rather than a
guess: the drafter arena is resident, so this is bandwidth-bound, and CB3 reads 14,454,784 against
FP4's 18,800,640 bytes per slot = **0.769** of the bytes. Measured 0.830 sits just above that floor,
which is what a bandwidth-bound kernel plus per-group decode overhead should look like. Being read
five times per step is precisely what makes the smaller record pay.

## What the port actually costs

Almost no code. `FixedStore` is ten lines of index arithmetic (`slot = (L-40)*128 + e`) with no
layout dependency, and `moe_fn` already dispatches on `isinstance(arena, cb3_cls)` -- handing the
drafter a `CB3ArenaV2` routes it to the CB3 kernel with **no kernel work at all**. The change at
`v41_engine.py:425` is `arena_cls(384, device)` -> `make_expert_arena(384)`.

The cost is data. `~/dsv41-cb3/experts-cb3-s3.bin` holds exactly 15,360 = 40 x 384 records and its
manifest says `"layers": [0, 39]`, so the drafter's 3 x 128 experts are **not packed**. They need 384
more records, 5.3 GB. The packer applies unchanged: the load call at `v41_engine.py:542` already uses
the same `EX.W13_SHAPE` / `EX.W2_SHAPE` as the main arena, so the drafter experts are the same shape
and the record layout is identical.

## What is NOT claimed

The 17 % is the **MoE kernel alone**. A draft pass is also attention, the shared FFN and two heads,
so the end-to-end tok/s gain is smaller and is not measured here. The 1.554 GiB is resident memory
returned to the main arena -- at 14,454,784 B/slot that is 115 more expert slots -- and its effect on
hit rate is a separate measurement. And the -1.05 pp acceptance figure is somebody else's, on their
quantizer; our CB3 is a different codec at a different effective bit rate and would have to be
confirmed on our own harness before the port is called quality-neutral.

## Why we are not using their weights

EXL3 saves more: 2,562,494,976 B/rank packed against 3,609,722,880 native = 0.975 GiB/rank, so 2.09 GB
at our TP1 against CB3's 1.67 GB. That extra 0.42 GB (about 29 slots) costs a trellis kernel port, a
TP2 -> TP1 re-layout, and pulls in AGPL-3.0-only serving code. We already ship the kernel that gets
94 % of the benefit. The download's value was the evaluation, not the tensors.


## The pack turned out to be unnecessary, and would not have worked anyway (job 985)

Two things came out of trying to build it.

**The scale codec is not lossless outside layers 0-39.** `ue8m0-3bit-rowbase-v1` stores one u8 row
base plus 3 bits per group, and its exactness rests on a survey of 149,422,080 rows in which the
intra-row exponent range never exceeds 7. That survey is the **routed** experts only. mtp.0 and mtp.1
packed clean, 128 experts each in 22.1 s and 18.7 s, and then mtp.2 raised on a row of range **8**.
So the correct statement is that the codec is lossless on the routed experts, not on the checkpoint,
and any new tensor family must be surveyed before it is packed. `tools/scale_codec.py` now says so.

**And the drafter needs no pack at all.** `CB3ArenaV2.load_slot` takes exactly the same six FP4
tensors as `ExpertArena.load_slot` and runs `fp4_to_cb3_v2` itself, so the existing load loop at
`v41_engine.py:542` is already correct for either arena -- the whole port is the one line above it.
The draft arena is resident and never reads a record off disk, so the packed-scale disk format it was
going to need buys it nothing. Job 980's 1.554 GiB and 17 % were measured on the unpacked-scale arena
(14,454,784 B/slot) in the first place.

One interaction had to be blocked rather than inherited: with `DSV41_PACKED_SCALES=1` a CB3 draft
arena would call `pack_torch` on mtp.2 and **raise at startup**. `DSV41_DRAFT_CB3` therefore builds
the arena directly instead of through `make_expert_arena()`, keeping scales unpacked whatever the
main arena does.

The packer changes are kept -- `--prefix-fmt` and `--index` are what made the codec limit visible,
and they leave the main 0-39 build byte-identical (verified against the shipped manifest: total,
n_layers, n_experts, and idx at (0,0), (17,200), (39,383)).

## The EXL3 weights verified, and their filenames are not their hashes

All three shards match `release-manifest.json`'s sha256 at 1,704,542,280 B each. Worth recording that
the CDN served them under 64-hex-character filenames that are **not** the content hashes -- treating
the filename as the checksum would have looked like verification and been none.

## End to end (job 990): the output check failed, which voids this job's speed number

Four arms, DSV41_DRAFT_CB3 0/1 interleaved, two rounds, `arena_gb=86.0`.

| arm | draft arena | main slots | tok/s | accept_len_mean | steps | misses | tokens_sha |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cb3=0 r1 | ExpertArena | 5,949 | 5.422 | 3.0952 | 157 | 13,849 | 47c29400 |
| cb3=1 r1 | CB3ArenaV2 | 5,949 | 5.481 | 3.1681 | 155 | 13,342 | 37cf84d3 |
| cb3=0 r2 | ExpertArena | 5,949 | 5.447 | 3.0952 | 157 | 13,849 | 47c29400 |
| cb3=1 r2 | CB3ArenaV2 | 5,949 | 5.649 | 3.1681 | 155 | 13,342 | 37cf84d3 |

**Check 1 (output) failed, and it was pre-registered as outranking the rest.** The arms emit different
greedy tokens. This is not nondeterminism: each arm reproduces its own hash, accept, steps, nvme and
miss count *exactly* across both rounds, so it is deterministic and drafter-dependent.

That voids checks 3 and 4 as stated. The arms do not decode the same tokens, so 5.422 -> 5.481/5.649
is not a like-for-like speed comparison and 3.0952 -> 3.1681 is not an acceptance comparison -- the
CB3 arm took a different trajectory with 507 fewer misses and 7.1 GB less NVMe, and some unknown part
of its advantage is that trajectory rather than the kernel. The clean kernel number stays job 980's.

**Check 2 (slots) failed for a reason in the harness, not the engine.** Main slots are 5,949 in every
arm because the job passes `arena_gb=86.0`, which pins the arena and overrides the auto-sizer. So the
1.554 GiB the CB3 draft arena frees was simply left unused here, and the memory half of the case is
still unmeasured. That also means whatever speed difference is real is the kernel alone.

**What the divergence is not.** The verification rule is exact greedy and is drafter-independent in
exact arithmetic: `engine/v41_engine.py` computes `am = logits.argmax(-1)` over the six rows, accepts
the leading run where `am[i] == drafts[i]`, and then emits `cand[:a]` -- which is `am[:a]`, the
TARGET's argmax, never the draft's value -- with `bonus = am[a]` conditioned only on accepted tokens.
So a worse drafter cannot change what is emitted by that rule.

**What it might be, untested.** A verify block is six tokens processed together, and its routed-expert
set is the union over all six. Different drafts give a different expert set and so a different
grouped-GEMM reduction order, which perturbs the target's own logits at the ulp level and can flip an
argmax on a near-tie. That would make the target's logits not invariant to the draft tokens -- a
numerical property of batched verification over a routed MoE, not a broken rule. It is consistent with
what we already know (a 1-ulp kernel change moves acceptance by ten points) but it is a hypothesis
with no counterfactual yet, so it is written here as one.


## Job 995: prompt 1 alone does not diverge

Both arms emitted 99 byte-identical ids for "Explain how an NVMe controller schedules writes." run on
its own. That is the third pre-registered branch, so the divergence job 990 saw is not a property of
prompt 1 in isolation.

Which leaves two candidates, and job 1000 separates them by reproducing 990's condition exactly --
five prompts, in order, one engine -- while recording each prompt's ids separately:

  (a) it is one of prompts 2-5; or
  (b) it is a CARRY-OVER. 990 ran all five in sequence, so arena occupancy and the prompt cache at
      prompt N depend on every prompt before it. If prompt 1 diverges in 1000 having been identical
      alone in 995, the drafter is not perturbing the target's logits at all -- it is changing what
      the arena holds, and the divergence is downstream of cache state. That would reframe this as an
      arena-occupancy effect rather than anything in the verify path.

Worth noting against my own earlier wording: 990 folded all five prompts into ONE sha, so "the arms
emit different greedy tokens" was correct but told us nothing about where. Per-prompt hashes are what
1000 adds.
