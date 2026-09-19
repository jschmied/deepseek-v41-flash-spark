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
