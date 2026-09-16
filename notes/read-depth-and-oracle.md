# Where the decode step actually goes (2026-09-16)

All numbers from the engines' own counters and the loader's own `nvme_start`/`nvme_end` brackets.
Never `/proc/diskstats`.

## 1. The step is read time, not compute

`GPU_BUSY_S = 3.0` over a 104 s span at 1.73 steps/s is **16.7 ms of GPU kernel time per step**,
against a measured step of ~437 ms. So perfectly overlapping *all* compute with *all* I/O is worth
**3.8 %**, and forking the shared expert alone (`C_IND` x 40 = 0.18 ms/step) is worth **0.04 %**.

**That bound covers the COMPUTE-overlap levers only, and an earlier revision of this note
overstated it as covering "all four dependency-graph projects".** It does not. `lease_until_completion`
overlaps NVMe with H2D -- two I/O operations -- and nothing about GPU kernel busy time bounds that.
The device-event experiment (`5d40062`) suggests the lever is small in practice, measured as a null,
but "small when measured once" is not "bounded by 3.8 % as a matter of arithmetic". Corrected after
review, 2026-09-16.

That bounds the dependency-graph work. The coarse barriers are real -- `stream.wait_stream(compute)`
is issued inside the sink *after* the read, so it waits on whatever was queued during it -- but the
room behind them is nearly empty at this operating point.

## 2. 44.3 % of the span has no read in flight

| | |
|---|---|
| depth 0 (nothing in flight) | 5.35 s = **44.3 %** of the span |
| mean depth while reading | 4.40 |
| in-flight bandwidth | 4.89 GB/s |
| achieved over the wall | 2.67 GB/s |

One layer supplies only **2.0 misses** (79.7 reads / 40 layers) and the next layer's ids do not
exist yet, so the pipe empties between layers. Reproduced independently by job 275's null arm
(2392 reads, 44.0-45.4 % depth 0) from a harness sharing no code path.

## 3. The oracle: +37 % at h=1, at identical bytes

| arm | steps/s | depth 0 | reads | GB |
|---|---|---|---|---|
| null | 2.343 / 2.413 | 45.4 % | 2392 | 32.95 |
| oracle h=1 | 3.271 / 3.267 | 25.2 / 26.0 % | 2393 | 32.96 |
| oracle h=2 | 3.362 / 3.387 | 23.4 / 22.9 % | 2395 | 32.99 |

`pred_miss` 0, `wasted` 0, 2388/2388 used -- a real oracle. The same bytes are read; the whole win
is depth. Horizon 2 adds 4 points over horizon 1, because one layer of lead is ~417 us of compute
against a ~12 ms read: `ready_hit` 96 against `late_hit` 2292. The win is from *starting* the read
earlier, not from having it finished.

**Scope**: greedy `next_block`, not DSpark draft+verify; warm residency seeded from v1; `POLICY=v1`;
30 steps; arena 40 GB. Ranks arms; not the server's steps/s.

## 4. The "27 % engine read penalty" does not survive its own test

Built to blame the engine's footprint. At `ARENA_GB=40` the **bare** stage -- before CUDA, before
any arena, before any pinned memory -- already read 4.67/4.87 GB/s, and adding the CUDA context,
a 40 GB device arena, 862 MiB of pinned host memory and 96 threads did not lower it (4.97/5.06).
Two later blocks, differing only in a variable that cannot touch the bare stage, read 6.76 GB/s and
stayed there.

Arena size explains nothing.

## 5. It is bimodal, and order does not explain it either (job 285)

The identical block, five times, nothing varying but when it ran:

| run | stage 0 (bare) | stage 4 |
|---|---|---|
| 1 back to back | 4.71 / 4.90 | 5.02 / 5.03 |
| 2 back to back | **6.79 / 6.79** | 6.75 / 6.74 |
| 3 back to back | 4.65 / 4.81 | 5.04 / 5.15 |
| 4 after 120 s idle | 4.80 / 4.85 | 5.05 / 5.13 |
| 5 after 120 s idle | 4.62 / 4.78 | 5.05 / 5.09 |

So the "first block is slow" reading from job 280 was wrong too: four of five runs are ~4.8 and one
is 6.8, and it is not position in the sequence, not time-healed, and flat across stages within each
run. The box delivers 4.8 or 6.8 for the same work under nominally identical conditions.

**A confound is identified and not yet excluded**: the watchdog restored a DS4.1 server at 11:28:56,
three minutes before this job, and a restored server warm-starts by reading 40 GB. Whether a server
was up, and what it was doing, was not recorded alongside these numbers. The re-run must record it.

Until then, treat the ~4.9 GB/s in-flight ceiling in section 3 as **unverified**, and note that if
the true rate is 6.8, the remaining gap in job 275 is made of something other than what section 3
implies.

## 6. Every number above was measured on an engram-ablated model

`RealLeaves.begin_step` zeroed `eg_rows` whenever no source supplied them, and the stage-5 gate
passed `{}` to v1 as well -- so both arms were ablated, agreed to 0.000e+00, and the gate was blind
to it. The real source landed in `50bfdf2`; its mutation check prices the ablation at **7.717** in
the logits.

This does not invalidate the A/B comparisons in sections 2-4 -- both arms of each were ablated
identically -- but it does mean the absolute steps/s are not the server's, and the engram stream's
~144 rows per step of small reads were absent from the device contention those sections measure.

## 7. Every v2 steps/s on this page assumes full acceptance

`RealLeaves.end_step` advances the cache by the whole block (`fd.c.len = S + T`, T = 6). The real
engine drafts with DSpark, verifies, and rolls back to `pos + a + 1`; measured `accept_len` on this
box is **2.66-3.72**, so a real step yields ~3.4 tokens where these harnesses behave as if all 6
were accepted.

Consequences, stated separately because they are not equally serious:

* **The A/B comparisons are unaffected.** Reads per step are set by the route, not by acceptance,
  and every arm shares the same block sequence. Depth-zero fractions and the oracle's margin stand.
* **The absolute steps/s does NOT convert to tokens/s.** Nothing on this page may be multiplied by
  6, or by 3.4, to get a serving rate: the KV also grows ~1.8x faster than it really would, so the
  route sequence past the first few steps is not the sequence a served request would walk.

Fixing this means real draft + verify + rollback in v2, not just the MTP head -- a head without
verification still advances the cache by the full block.

## 8. The misses are CAPACITY misses, and the router is near-uniform

Prompted by a question worth recording: if 20k tokens of agentic work stay on one topic, shouldn't
early expert activations predict later ones -- and were the short prompts simply too small to show
signal? Measured on the two decode traces (548 and 443 steps):

| | route-decode | route-decode-code |
|---|---|---|
| new (layer,expert) pairs/step, Q1 -> Q4 | 63.2 -> 3.1 | 83.2 -> 3.7 |
| misses/step, Q1 -> Q4 | 80.9 -> 66.0 | 123.9 -> 127.2 |
| working set (pairs) | 11,087 | 11,879 |
| distinct experts at the median layer | **276 of 384** | **296 of 384** |
| misses that are CAPACITY (seen, evicted, wanted again) | **76.4 %** | **83.0 %** |

So activations do repeat and the working set does converge -- but it converges to 72-77 % of the
whole model, not to a topic-sized set, and the arena holds 23-25 % of it. Misses persist because
the cache is ~4x too small, not because the model keeps finding new experts.

This is NOT a short-prompt artifact: the predictor evaluation in section 7's companion (8419ade)
scored only the LAST 40 % of the trace, already in the converged regime. And longer context makes
it worse, not better -- at 10x the tokens the per-layer distinct count approaches 384.

**CORRECTION (same day).** An earlier revision of this section said routing was "near-uniform",
quoting job 102's entropy of 8.52 of 8.58 bits. That is the wrong reading and the wrong
distribution. Measured directly on the realized ACCESSES:

| | route-decode | route-decode-code |
|---|---|---|
| entropy of accesses | **6.89 bits** (uniform over 384 = 8.58) | 7.12 |
| experts holding half the accesses | **30 of 384 (8 %)** | 29 |
| top-139 (36 %, 0xBakeer's keep) | **93.3 %** of accesses | 89.6 % |
| top-169 (44 %, our keep) | 96.3 % | 93.7 % |

The distribution is heavily skewed, which is exactly why published pruning profiles at keep ~0.36
work. Entropy is also a poor skew detector over a 384-alphabet: a top-36 % holding 60 % of the mass
still scores ~8.42, so 8.52 never established uniformity in the first place.

What actually survives: **LRU already captures the skew.** The production arena at ~82 GB holds
5,679 pairs, about 37 % of all pairs -- essentially the hot set the pruning profiles identify. The
hot experts are therefore resident by construction, and what is left to predict is the warm TAIL
beyond them, where the distribution is flat. That is the honest reason co-occurrence scores 6 % and
DSpark 0.5 %: not that routing is uniform, but that the predictable part is already in cache.

## 9. DSpark carries no signal about backbone misses either (job 300)

600 steps of the real draft/verify/rollback loop, the drafter's own 3 x 128 routing scored against
the backbone's per-layer miss ids:

  dspark -> miss, top-1:  miss-recall 0.4 %, fetch precision 0.5 %  (popularity baseline 6.2 %)

Below the baseline. The lead-time argument -- that a weak predictor 40 layers early beats a better
one a layer early -- does not get to apply, because there is no predictor.

## 10. Three scope errors in everything above

Every v2 experiment on this page ran at `ARENA_GB=40`. Production auto-sizes to ~82 GB, where the
measured hit rate is 0.894 against the 0.829 these runs saw, and decode is 6.26 tok/s against the
3.71 measured at 56.75 GB. The A/B comparisons hold -- both arms share the arena -- but 44 %
depth-zero and +37.8 % are at an operating point with roughly twice the miss rate of the real one.

The arena sweep is also the largest measured lever on this whole page:

  56.75 GB  3.71 tok/s  hit 0.796      79 GB  6.00 tok/s  hit 0.890
  68 GB     4.89 tok/s  hit 0.869      auto   6.26 tok/s  hit 0.894

**Second: every v2 run used `evict="lru"`, hardcoded.** The box has already measured
age/(1+count) ahead of it -- decode 1.88/2.13/2.14 tok/s against LRU's 1.74/1.85/1.86 (job 140),
and offline 94.07 % hit with 52.3 fetches per step against 92.67 % and 64.6 (phase 1). A fair A/B
at the worse operating point. The policy is now an env choice and is printed.

**Third: v1 and v2 were never compared at the memory each can actually afford.** Both arms ran a
FIXED 40 GB arena, so v2's smaller staging footprint -- 8 buffers against v1's 48 -- was never
converted into slots. Worse, in the current harness it is not even realized: v2 rides on a
V41Engine, whose ExpertStore allocates its own 48 pinned buffers regardless, so v2 today pins MORE
than v1, not less. Realizing it needs `DSV41_IO_THREADS=8` on the v2 arm, which is the knob that
sizes that staging array. Until that is run, "105 MiB against 862 MiB" is a property of v2 as a
standalone engine, not of anything measured.

## 11. The oracle margin survives the arena scope fix (job 305)

Re-run at 79 GB, engram live, against 40 GB as the tie-back:

| arena | arm | steps/s | reads | GB | depth 0 |
|---|---|---|---|---|---|
| 79 GB | null | 2.671 / 2.698 | 1895 | 26.10 | 51.6 / 48.5 % |
| 79 GB | oracle h=1 | **3.687 / 3.606** | 1896 | 26.12 | 33.9 / 34.7 % |
| 40 GB | null | 2.308 / 2.323 | 2392 | 32.95 | 45.9 / 45.6 % |
| 40 GB | oracle h=1 | 3.228 | 2393 | 32.96 | 27.7 % |

Three things, in order of how much they change the picture:

* **The margin holds: +36.0 % at 79 GB against +39.6 % at 40 GB.** The headline was measured at
  half the production arena, and it did not depend on that.
* **depth-0 RISES with the bigger arena, 45.9 % -> 51.6 %.** Predicted before the run and confirmed:
  a better cache means fewer misses per layer, which means LESS work available to keep the device
  busy. The pipe gets emptier as the engine gets faster.
* Capacity does what the sweep said: reads 2392 -> 1895 (-21 %), bytes 32.95 -> 26.10 GB, null
  2.32 -> 2.68 steps/s (+16 %).

CAVEAT, from the job's own stamp. It recorded `HEAD efb184f 0 dirty`, but the 40 GB arms print an
`evict`/`slots`/`staging` line the 79 GB arms do not -- so `0907c87` landed mid-run. That commit
only added an EVICT env defaulting to the previously hardcoded "lru" plus print statements, so both
halves ran the same eviction behaviour and the comparison stands. Recorded because the stamp is
only worth having if its warnings are acted on rather than explained away.

## 12. WITHDRAWN -- job 320's warmed arms computed with the wrong weights

Found by review, confirmed in job 320's own output. `warm_policy` called `reserve()` then
`clear_pending()` and never LOADED anything: `reserve()` only rewrites metadata, so each slot was
remapped to a new key while the arena still held the previous expert's bytes, and `clear_pending()`
then declared the write finished. Every warmed key was resident by bookkeeping and wrong by content.

The harness printed the evidence for all four warmed arms and nothing acted on it:

    route sequence reproduced: 31 layers match, 1249 differ

Thirty-one of 1,280. The WARM=0 arms show 1280/0, so the defect is exactly the warm-up.

So BOTH conclusions below are withdrawn: "the policies tie" and "the oracle is a net loss on a warm
cache, -13 %". The 971-against-1895 read count that looked like a warm cache was the cache serving
experts it had never read.

Two fixes, not one. The warm-up is now REAL decode -- steps 0..WARM-1 run untimed with prediction
off and the timed window is steps WARM..WARM+STEPS-1 of the same continuous generation, no
re-prefill and no replay. And the route check now RAISES on any divergence inside the timed window
instead of printing it, because a run whose routes diverge is not measuring this engine at all.

The original section is kept below, struck, so the numbers are not quoted from memory later.

## 12-OLD (WITHDRAWN, DO NOT QUOTE) age/(1+count) ties LRU, oracle hurts a warm cache

Warmed on recorded steps 30-59, timed on 0-29, at 79 GB with engram live:

| warm | policy | arm | steps/s | demand | reads | GB |
|---|---|---|---|---|---|---|
| 30 | age_over_freq | null | 4.211 | 980 | 971 | 13.38 |
| 30 | lru | null | 4.198 | 980 | 971 | 13.38 |
| 30 | age_over_freq | oracle | **3.677** | 637 | **1911** | 26.32 |
| 30 | lru | oracle | **3.661** | 637 | **1913** | 26.35 |
| 0 | age_over_freq | null | 2.767 | 1942 | 1895 | 26.10 |
| 0 | lru | null | 2.677 | 1942 | 1895 | 26.10 |

**The policies tie.** Warmed, they finally differ at all -- 1911 against 1913 reads -- but that is
2 reads in 1911. At this cache size (5,457 slots, ~37 % of all pairs) the victim choice does not
matter, which is consistent with protection also being near-null: when the hot set fits, recency
and frequency pick equally well. The server-level win (job 140: 2.13 against 1.85 tok/s) is not
contradicted -- that was a full generation building its own history at a different arena size.

**The oracle is a NET LOSS on a warm cache**, and this qualifies every oracle number above. With
the cache warmed, the null arm needs 971 reads; the oracle issues **1911** and runs **3.68 against
4.21 steps/s, -13 %**. It nearly doubles I/O to prefetch a layer ahead, because a prefetch takes a
slot and `protected_slots` only shields the CURRENT layer -- so it evicts residents that later
layers still want, and those come back as new misses.

So the +36-38 % headline is conditional on a miss-rich cache. The mechanism that made it a win
(an empty pipe with work to fill it) is the same one that makes it a loss when the pipe has little
to do. Depth-zero rising with arena size (section 11) already pointed here; this is the same effect
crossing zero.

WHAT THAT DOES NOT SAY: that prefetch is worthless. It says prefetching the FULL next-layer set,
unprotected past one layer, is worthless once the cache is warm. A prefetcher that only issues when
read depth is low, and that cannot evict anything a near-future layer wants, was never measured.

## What this closes and what it leaves

- Closed here: the engine-footprint explanation for the read penalty (refuted by its own bare stage).
- Bounded here: every compute/IO overlap project, at 3.8 % combined. NOT every scheduler toggle --
  see the correction in section 1; an NVMe/H2D overlap lever is outside that arithmetic.
- Closed by sections 8-9: prediction of expert IDENTITY, by any of co-occurrence, recurrence, or
  the DSpark drafter. Routing entropy 8.52/8.58 says there is almost nothing to infer.
- Live, and non-predictive: CONCURRENCY. More requests in flight means more misses per layer to
  issue, which raises read depth without knowing anything about the future -- the one lever left
  that the entropy result does not touch.
- Live, and already the biggest measured effect: arena capacity.
