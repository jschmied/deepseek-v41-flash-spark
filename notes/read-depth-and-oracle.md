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

## 13. The warm cache, measured properly at last (job 345)

Three prior attempts were harness bugs, each producing plausible numbers: a policy with no history
(310), a warm-up that replayed the timed window (315), a warm-up that loaded nothing at all and ran
31/1280 routes (320), and an oracle reading the warm-up's own steps with issued=0 (335). The
harness now aborts on each of those conditions.

79 GB, engram live, warm-up = 30 REAL decode steps, timed on steps 30-59, three reps:

| policy | arm | steps/s | reads | GB |
|---|---|---|---|---|
| lru | null | 5.461 / 5.455 / 5.468 | 474 | 6.53 |
| lru | **oracle** | **6.596 / 6.553 / 6.734** | 474 | 6.53 |
| age_over_freq | null | 5.491 / 5.388 / 5.384 | 474 | 6.53 |
| age_over_freq | **oracle** | **6.485 / 6.493 / 6.588** | 474 | 6.53 |

**The oracle is worth +20.4 % warm, at identical bytes.** Not the -13 % job 320 claimed (that run
computed with wrong weights), and not the "approximately nothing" I reported from a single arm
whose 5.27 happened to land inside the null spread -- with reps the arms separate cleanly, null at
+-0.2 % and oracle at +-2.7 %.

**The policies still tie**, now on a warm-up that genuinely builds history: 5.46 against 5.42 mean.
At 5,457 slots the victim choice does not matter, consistent with protection measuring near-null.

WHY IT STILL HELPS WHEN THE DEVICE IS IDLE. Warm issues 15.8 reads per step at ~12 ms latency,
which is ~190 ms against a measured step of 183 ms. The step time is essentially the SERIALIZED
miss latency. The device is idle 84 % of the span and the step is still waiting on reads -- because
they are taken one layer at a time, not because the device is busy. The oracle removes most of that
wait by starting them a layer early.

So "the pipe is empty" means two different things cold and warm. Cold: the device is starved of
work it could be doing. Warm: the device has little to do, and what little there is sits directly
on the critical path. Prefetch helps in both, for different reasons.

## 14. The decode GPU budget, measured at last (job 355)

Node-traced (`--cuda-graph-trace=node`) inside an NVTX range around the timed window only, so the
65 s engine load, the warm start and the warm-up are excluded. With node tracing the graph nodes
appear as ordinary kernel rows -- 247k of them -- which is why GRAPH_TRACE reads 0 here and why the
union equals the kernel table. That is the correct behaviour, and the opposite of job 175's
capture, where the nodes were invisible.

| | span | GPU busy | GPU idle | per step |
|---|---|---|---|---|
| warm | 6.09 s | **3.03 s = 49.8 %** | 50.2 % | 101 ms |
| cold | 11.94 s | **3.09 s = 25.9 %** | 74.1 % | 103 ms |

**GPU busy per step is ~102 ms in BOTH regimes** -- the same work, as it must be. What changes is
the wait around it:

    warm step 203 ms = 102 GPU + ~101 wait
    cold step 398 ms = 102 GPU + ~296 wait
    warm + oracle 152 ms = 102 GPU + ~50 wait

So the hypothesis that a warm cache leaves the GPU running without pause is **false**: it is idle
half the time even when the NVMe pipe is nearly empty. And the earlier constant was wrong by more
than the 8x already corrected -- job 175's 16.7 ms/step against a measured **102 ms/step**.

THE CEILING THIS SETS. If every wait were removed, the step would be 102 ms = **9.8 steps/s**,
against 5.46 warm and 6.6 warm-with-oracle. So perfect overlap is worth +80 % from the warm
baseline, and the oracle already captures about half of the available wait.

Caveat: under nsys the warm null arm ran 6.09 s against 5.46 s unprofiled, so the percentages carry
profiler overhead and should be read as approximate.

## 15. v1 and v2 are the same engine at steady state (job 350)

300 steps, engram live on both sides, 79 GB, two reps. Every earlier v1/v2 comparison ran 30 steps,
which sits entirely inside the transient.

| arm | steps/s | reads | GB | h2d_s | staging |
|---|---|---|---|---|---|
| v1 | 8.097 / 8.130 | 1900 | 26.17 | 9.50 | 48 buffers |
| v2 | 8.174 / 8.107 | 1952 | 26.89 | 3.37 / 4.10 | 8 buffers |

**A tie: 8.11 against 8.14.** v2's earlier +5.8 % and +9.6 % were transient-window artifacts. Hit
rate at steady state is 98.24 %, against the 82-89 % the short runs saw, and throughput is 8.1
steps/s against ~2.3 -- so the 30-step numbers were measuring a cache filling up, not an engine
serving.

What v2 does keep: a third of the H2D time and 8 pinned staging buffers against 48. It reads
slightly MORE (1952 against 1900) from different eviction timing.

## 16. Real tokens/s at last, and the eviction policy finally separates (job 360)

First measurement from the served loop -- DSpark draft, verify, rollback, tokens COUNTED rather
than inferred (`2906b20`). 600 steps after a 150-step warm-up, 79 GB, engram live, two reps:

| policy | tok/s | reads/token | MB/token | accept_len |
|---|---|---|---|---|
| lru | 5.80 / 5.94 | 27.1 | 373.4 | 2.95 |
| **age_over_freq** | **6.32 / 6.35** | **22.8** | **314.6** | 2.95 |

**age/(1+count) is +8.0 % on real tokens**, with 16 % fewer reads and 16 % fewer bytes per token.
Reproducible across reps (+-1.2 % and +-0.5 %).

THIS OVERTURNS "THE POLICIES TIE", WHICH THIS FILE SAID TWICE. Jobs 310/315/320/345 all reported a
tie, and every one of them was blind to the difference for a different reason: a flat-seeded policy
with no history, a warm-up that replayed the timed window, a warm-up that loaded nothing, and --
running through all of them -- a greedy `next_block` that collapsed to a repeating token, so the
same experts were touched every step and no eviction decision mattered. The policy only separates
when the token stream is real and acceptance is real.

It also agrees in direction with the server-level job 140 (2.13 against 1.85 tok/s), which had been
the one measurement saying the policy mattered and was repeatedly explained away here.

SANITY: 5.80-6.35 tok/s brackets production's 6.26, so this harness is finally measuring the engine
the server runs rather than a proxy.

STILL OPEN: accept_len 2.95 against production's 3.65. The verify matches v41_engine line for line,
so it is prompt or drafter-state, not logic -- but it is 24 % of tokens per step and therefore sits
directly in the tok/s.

SHIPPABLE TODAY: `DSV41_EVICT_POLICY=age_over_freq` is an env var production does not set.

## 17. The scheduler review, items 1-10: what was worth doing and what the measurement killed

The review proposed ten simplifications to the loader. Job 370 measured the premise behind most of
them and came back a NULL: removing ~1080 host-side lock acquisitions per step moved real tokens by
0.08 %, against within-arm spreads of 2.7 % and 1.9 %. Python synchronisation is not material in a
~0.46 s step. That result is load-bearing for everything below -- it is the reason several items are
now declined ON EVIDENCE rather than deferred.

The distinction that survives it: a LOCK is overhead, and overhead is 0.08 %. A BLOCKED OBSERVER is
a scheduling bubble, and a bubble holds a physical resource closed. Item 7 was the only item of the
ten on the second side of that line.

### Item 7 -- the completer retired copies in submission order (done, 470c161)

One thread called `handle.synchronize()` on each completion in the order workers queued it. The
copies run on per-worker streams and finish OUT of order, so a finished copy could sit behind a
running one -- and `_complete_h2d` is what releases the pinned staging buffer and the h2d permit.
With `h2d_inflight=2` that halves the effective copy depth in the worst case.

The loop now drains everything the queue holds before deciding anything, retires whatever `query()`
says has landed, and polls at 200 us when copies are outstanding but none is ready. It never blocks
on one specific event.

My first attempt reproduced the same defect one step along: it took ONE item, found it unfinished,
and parked in `synchronize()` on it -- without looking at the queue, where a finished copy was
already waiting. The new test caught that on the first run. It is worth saying why the existing 38
tests could not have: every fake provider in the suite returns `None` from `h2d()`, so the whole
suite exercises the no-event path and passes identically against the old in-order loop.

Measured by job 375, with `DSV41_COMPLETER_INORDER=1` restoring the old loop in the SAME binary --
and with a test asserting that arm really does block on the older event, so the A/B cannot return a
null by construction. Four earlier scheduler verdicts went wrong exactly that way.

**RESULT: a null.** 600 steps after a 150-step warm-up, 79 GB, engram live, real committed tokens:

| arm | rep 1 | rep 2 | mean |
|---|---|---|---|
| out of order (new) | 6.33 | 6.46 | 6.395 |
| in order (old) | 6.36 | 6.38 | 6.370 |

+0.4 %, against a within-arm spread of 2.1 % in the new arm. That is the pre-registered `< 1 %`
branch: **stop looking for wins in the completer.**

SCOPE, because the mechanism was real and the effect is not. Head-of-line blocking can only cost
what it actually blocks, and here it blocked almost nothing: `staging peak 8` against
`h2d_inflight=2`, and an expert READ takes ~13.6 ms while its H2D is a small fraction of that. The
completer was therefore idle most of the time and the in-order wait usually had nothing queued
behind it. The defect was genuine -- the test proves the old arm does block -- it simply sat on a
resource that was never scarce. Keep the new code: it is simpler, it has tests, and it removes a
failure mode that WOULD bite at higher h2d depth. Do not claim a speed-up for it.

TAKEN TOGETHER WITH JOB 370, this closes the loader-scheduler family by elimination. 370 removed
~1080 host locks per step: 0.08 %. 375 removed the one scheduling bubble in the same machinery:
0.4 %. Whatever the ~69 ms/step of section 14 is made of, it is not in the loader.

### Item 6 -- the loader lock, split three ways (two done, one declined)

- DONE: an immutable `LoadTicket`. The queue payload was `(ctx, cause_id, speculative, scored) +
  tuple(item)` spliced into a 4-tuple, read positionally in `_worker` as `entry[2]`/`entry[3]`, with
  the demand flag carried twice (`entry[3]` was `prio == 0`, which is `not speculative`, which is
  already inside `entry[2]`).
- DONE: two deques replacing the PriorityQueue. Demand ahead of speculation, FIFO within each, is
  all the PriorityQueue was doing -- and the `_seq` counter existed ONLY to break priority ties
  before Python compared the payloads, which meant taking the service lock to generate it. The
  shutdown sentinel also stops being a magic priority of -1 and simply goes to the front.
- DECLINED: per-thread statistics counters. This would remove ~270 lock acquisitions per step. Job
  370 priced ~1080 at 0.08 %, so this is worth about 0.02 %, and it costs a `threading.local`, a
  registry and two summing properties. That is complexity bought with a measured non-effect. (The
  same change IS in `real.py`, where the counters sit in the read path and the contention is real --
  the difference is the measurement, not the pattern.)
- DECLINED: driver-owned cancellation. `_queued`/`_cancelled`/`_discard` is the genuinely intricate
  part of this file, but rewriting it changes WHEN a speculative read stops being cancellable. That
  is a semantic change to the one feature v2 exists for, for no measured gain.

### What is left, and it is not in the scheduler

Items 1-5 and 8-10 are done or subsumed. The remaining term is the one job 370 pointed at on its way
past: keeping the NVMe pipe occupied. Decode runs 44 % cold and 84 % warm at depth 0, misses are
capacity misses (76-83 %), and expert IDENTITY is unpredictable (sections 8-9). The levers that
survive all of that are arena capacity, concurrency, and computing the resident experts while the
misses load -- review item 3, the only proposal that attacks the term that sets step time.

## 18. Arena capacity converts to tokens, and it is the largest effect here (job 380)

600 steps after a 150-step warm-up, engram live, `age_over_freq`, real committed tokens:

| arena | tok/s | nvme | reads / token | MemAvailable after load |
|---|---|---|---|---|
| 79 GB | 6.35 | 556.26 GB | 22.8 | 17 GiB |
| 86 GB | **6.96** | 462.38 GB | 19.0 | 12 GiB |

**+9.6 % tok/s for +9 % arena, at -17 % bytes.** Job 381 replicates it (79 GB -> 6.41) and extends
to 89 and 92 GB. A one-line env change on the server; no code.

THIS CORRECTS THE FRAMING I QUEUED IT UNDER. I argued that fewer bytes need not mean more tokens,
because 6.2 tok/s sits far below the 27.6-34 tok/s BYTE ceiling and sections 2 and 14 say the step
is mostly waiting rather than transferring. It converted anyway, and close to proportionally. The
resolution is that a miss costs more than its bytes: it costs a DEPENDENCY -- graph B for that layer
cannot run until the bytes land, and at ~2 misses per layer there is nothing else for that layer to
do. Removing a miss removes a serialisation point, not just 14.45 MB. That is why capacity beats
bandwidth here and why it does not contradict the idle-pipe finding.

SCOPE. Single stream, `age_over_freq`, this corpus, 5465 -> 5949 slots. It does not reach the
working set: ~15360 (layer,expert) pairs would need 222 GB and the box has 121 GiB total, so this is
a marginal-return curve, not a fix.

**JOB 381 REPLICATES IT AND FINDS THE CEILING.** Two reps each:

| arena | rep 1 | rep 2 | mean |
|---|---|---|---|
| 79 GB | 6.41 | 6.31 | 6.36 |
| 86 GB | 7.07 | 6.93 | **7.00** |
| 89 GB | refused | refused | -- |
| 92 GB | refused | refused | -- |

**+10.1 %, and 86 GB is the maximum the engine accepts.** 89 and 92 never started: `V41Engine`
refuses them in its own pre-flight at `engine/v41_engine.py:507`. So the ceiling is set by the
engine's host-memory reservation (~20.5 GiB) and not by my job's guard, which never had to fire.

That reservation is inconsistent with the auto-sizer's ~9 GB -- a known open item -- and raising it
is exactly the change that once left the box at 4 GiB under sustained load with no server. Not
touched here. `ARENA_GB=86` is shippable as it stands: one env var, no code, +10.1 %.

SAFETY, learned the expensive way. Job 380 was queued with a 92 GB arm behind a 20 GiB pre-flight
margin. That margin gates the START and says nothing about the STEADY STATE, which is exactly where
job 150's 4 GiB left the box without a server. 380 was killed at that arm; 381 carries a real guard
instead -- 200 s in, the arm is killed if MemAvailable falls under 7 GiB, printing ABORTED and
continuing the sweep. A missing cell, not a dead box.

## 19. The step, attributed at last (job 385)

Wall time per driver phase, on the host, `DSV41_HOST_PROFILE=1`. 600 steps after a 150-step warm-up,
engram live, `age_over_freq`. **The table below is job 390**, the clean re-run after the divisor bug
described at the end of this section was fixed; job 385's columns are kept underneath it because the
79 GB arm has not been re-measured.

| phase | 86 GB rep 1 | rep 2 | what it is |
|---|---|---|---|
| `wait_reads` | 272.0 ms | 266.0 ms | the driver blocked on this layer's expert misses |
| `layer_a` | 122.5 ms | 123.1 ms | graph A replay + the synchronous D2H of `route_idx` |
| `resolve` | 24.3 ms | 24.2 ms | settle, drain, reserve, submit, bind_slots, speculate |
| `end_step` | 6.2 ms | 6.2 ms | draft, verify, rollback |
| `await_copies` | 2.4 ms | 2.2 ms | putting copy events on the compute stream |
| `layer_b` | 2.1 ms | 1.9 ms | ENQUEUE only; the device work lands in the next sync |
| `wait_engram` | 2.0 ms | 1.0 ms | |
| `begin_step` | 1.0 ms | 1.6 ms | |
| INSTRUMENTED | 432.6 ms | 426.1 ms | |
| wall/step | 445.6 ms | 439.1 ms | **residual +13.1 / +12.9 ms, 2.9 %** |

Counts are right in this run -- `layer_a x40.0` for 40 layers, `begin_step x1.0` per step -- and the
residual is positive and small, which is what says the instrument is now telling the truth.

### Job 385's uncorrected run, and where my correction was wrong

385's raw rows were scaled by the warm-up bug. I reported them divided by 1.25 and called that
"exact, not an estimate". **That was wrong, and job 390 shows by how much.** Dividing by 1.25 is
exact only for a phase whose per-call cost is the same in the warm-up as in the timed window:

| phase | 385 / 1.25 | 390 measured | error |
|---|---|---|---|
| `layer_a` | 122.1 | 122.8 | +0.6 % |
| `end_step` | 6.2 | 6.2 | 0 % |
| `wait_reads` | 292.8 | 269.0 | **-8.1 %** |
| `begin_step` | 1.65 | 1.26 | **-24 %** |

The warm-up runs colder, so its misses cost more: the phases that BLOCK were inflated by more than
1.25x and the phases that do constant work were inflated by exactly 1.25x. The correction had to be
per-phase and I applied it uniformly. Nothing qualitative changes -- `wait_reads` still dominates at
62 %, `layer_a` is still 29 % -- but the read-wait figure to quote is **269 ms, not 293 ms**.

Job 385, raw rows divided by 1.25 (79 GB was measured only here), for the arena comparison only:

| phase | 79 GB | 86 GB |
|---|---|---|
| `wait_reads` | 328.9 ms | 292.8 ms |
| `layer_a` | 122.4 ms | 122.1 ms |
| `resolve` | 26.0 ms | 22.3 ms |

**THE DRIVER'S TIME IS FULLY ACCOUNTED FOR.** The residual is 2.9 %, so there is no hidden term: the
step is read wait, graph A, and host resolve, in that order, and nothing else is material.

**1. `wait_reads` is ~65 % of the step, and it is where the arena win lands.** 79 -> 86 GB moves
`wait_reads` by -36.1 ms while `layer_a` moves by -0.3 ms. That was the pre-registered test of note
18's claim that a miss costs a DEPENDENCY rather than its bytes, and it passes: extra capacity buys
time in exactly one phase, the one that blocks.

**2. The engram stream does not block the driver.** 1.9 ms per step, 0.4 %, against 288 rows
arriving as 575 buffered preads. The IOPS concern is answered: the engram reads cost device time
(7.3 % of wall in `read_s`) but the driver is essentially never waiting on them.

**3. NOTHING OVERLAPS, and the instrument shows it by omission.** There is no `shared` row in the
table at all, because `RealLeaves.shared_first = False` -- `engine/fastdecode.py` captures the shared
expert inside graph B, so the driver's shared-expert fork is inert on the real provider.

**4. THE PRIZE -- AND THE +39 % FIGURE IS WITHDRAWN.** `layer_a` carries essentially all the GPU
work (~102 ms/step by job 355's node trace, plus launch and sync latency) and it is strictly serial
against the read wait. I computed the prize as

    max(269, 123) + 24 + 12 + 13 = 318 ms   against 442 ms today   = +39 %

**That arm is not physically realizable, which is exactly the check the heartbeat protocol demands
and I did not make.** The chain is A(L) -> reads(L) -> B(L) -> A(L+1), and graph A must FINISH
before this layer's expert ids exist -- so graph A can never overlap the reads it itself causes.
Only graph B's work can move into the read window. The bound is therefore graph B's cost, not total
GPU time, and job 400 measures it with CUDA events around each graph replay.

Job 395 already pins one component: graph S, the shared expert alone, is **2.7 ms/step**, measured
as the drop in `layer_a` when it left graph B (122.2/122.4 -> 119.6/119.6, against a 0.23 ms
within-arm spread). If graph B is mostly its routed MoE and the MoE at T=6 is small, the whole
overlap family is worth single-digit percent and this section's headline needs rewriting down.

That supersedes the "every compute/IO overlap project, bounded at 3.8 % combined" line in the
closing section, which came from the loader-overlap family on a modelled provider. The 3.8 % bound
was real for what it measured and is not the bound on THIS.

What can actually be overlapped is limited by what is knowable: layer L+1's expert ids do not exist
until A(L+1) runs, which needs B(L). So the only compute available to hide under layer L's read wait
is layer L's OWN work -- the shared expert, and the routed MoE over the experts that are already
resident (~83-85 % of the 6 per token). Realising it needs graph B split, per-expert output buffers,
and a fixed-order reduction to keep the result bitwise identical.

### The instrument had a divisor bug, and printed the evidence itself

`HostPhases` accumulated across `decode(WARM)` and `decode(STEPS)` while `report()` divided by
`STEPS` alone, so every row was scaled by (150 + 600) / 600 = 1.25. It was caught immediately
because the report prints the residual and `INSTRUMENTED` came out 125 ms ABOVE `wall/step` -- a
negative residual, which is impossible. Two rows name the same bug independently: `begin_step x1.2`
where it must be x1.0, and `layer_a x50.0` where there are 40 layers.

Fixed by `HostPhases.reset()`, called after the warm-up. The corrected figures above are the raw
rows divided by 1.25, which is exact, not an estimate -- and the corrected residual falls to under
1 %, which is the check that the correction is the right one.

### Scope

Single stream, this corpus, `age_over_freq`, block 6, accept_len 2.95. The probe costs 1.3 %
(86 GB: 6.58 profiled against 6.67 unprofiled), so the absolute ms are ~1 % high and the proportions
are unaffected. Cross-JOB absolutes are not comparable: 385's 86 GB arms ran at 6.58-6.67 against
381's 7.00 for the same configuration, and the counters say why -- engram cost 112 us per row here
against 62 us in job 380, i.e. the device was in the slow mode of section 5. Within-job, the arena
effect replicates: **+9.3 % in 385 against +10.1 % in 381.**

## 20. The shared expert can overlap the read wait, and it is worth 0.2 % (job 395)

`DSV41_SHARED_FIRST=1` captures the shared expert as its own graph so the driver can replay it
after the route is resolved and the reads are submitted, but BEFORE it blocks on them. Gated
bitwise against the unsplit engine in a two-process comparison (`falsify_shared_first.py`):
logits, h, pre_mix and y all at 0.000e+00.

86 GB, 600 steps after 150 warm, `age_over_freq`. **One of the four arms ran with the device in its
fast mode** -- engram 61 us per row against 114/115/114 in the other three -- so that arm is excluded
and the comparison uses matched modes only:

| arm | wall/step | tok/s | `layer_a` | `shared` | engram us/row |
|---|---|---|---|---|---|
| sf=0 rep 1 | 442.65 | 6.66 | 122.21 | -- | 114 |
| sf=0 rep 2 | 446.89 | 6.59 | 122.44 | -- | 115 |
| sf=1 rep 2 | 443.64 | 6.64 | **119.55** | 1.64 | 114 |
| sf=1 rep 1 (EXCLUDED, fast device) | 437.86 | 6.73 | 119.61 | 1.76 | 61 |

**THE MECHANISM WORKS. THE QUANTITY DOES NOT MATTER.** `layer_a` falls by 2.7 ms in both sf=1 arms
against a 0.23 ms within-arm spread, so the shared expert's GPU work definitively left the critical
path -- that is the overlap, and it is not ambiguous. But it costs 1.6-1.8 ms of launch in the new
`shared` row, so the net is ~1 ms/step, about 0.2 %, which is inside the 4.2 ms spread of the
baseline arm. Keep the flag -- it is free, off by default, and it is the only working demonstration
that anything CAN be hidden under the read wait -- but claim no speed-up for it.

### The one-off fast arm, and a wrong turn I took on it

Three of the four arms in order gave 114, 61, 115 us per engram row, which correlates with the ARM
(sf=1 fast) and not with time. I wrote that down as possibly causal and queued an order-reversed
control. The fourth arm came back at 114 us with sf=1, which kills that reading: one fast run in
four is the device bimodality of section 5 (one fast in five, same signature), and the control was
withdrawn before it ran. **Three points that fit a story are not the story.**

### What it says about the routed MoE

Moving 2.7 ms of GPU out of the critical path cost 1.7 ms of launch -- a 63 % tax, because a graph
replay costs about the same whatever it contains. The routed MoE would pay that tax ONCE for a much
larger body of work, so the ratio there is far better. Whether the work itself is large is what job
400 measures, and it is the question that decides the whole overlap family.

## 21. The decode GPU budget, per graph (job 400)

CUDA events around each graph replay, drained once per step. 86 GB, 300 steps after 100 warm. The
two arms are internally consistent, which is what says the instrument works: graph B is 79.36 ms
with the shared expert inside it and 69.42 ms with it split out, a difference of 9.94 ms against
the 10.44 ms that graph S measures directly.

| graph | contents | ms / step | share |
|---|---|---|---|
| A | attention + HC + router | 46.1 | 37 % |
| B | routed MoE + HC residual (sf=1) | **69.4** | 55 % |
| S | shared expert alone | 10.4 | 8 % |
| total | | 125.8 | |

Against job 355's node-traced ~102 ms/step for the whole step -- a different instrument on a
profiled run -- this is the right order, and it is the first per-graph split we have had.

**GRAPH A IS 37 % OF THE DEVICE TIME AND CANNOT EVER OVERLAP.** A(L) must finish before layer L's
expert ids exist, so it cannot hide behind the reads it itself causes. That is the structural reason
the +39 % of section 19 was never available, independent of any implementation.

**B_r = 69.4 ms IS THE CEILING ON EVERY OVERLAP PROJECT LEFT.** At the ~83 % hit rate, about 58 ms
of device time is in pairs whose experts are already resident and could therefore compute while the
misses load.

### How much of moved device time becomes wall time: WITHDRAWN, see section 22

The shared expert is the calibration, because it is the one case measured both ways. 10.44 ms of
device time moved out of graph B produced a **7.07 ms** drop in wall (434.59 -> 427.52). Where it
shows up is worth noting: `layer_a` fell only 2.30 ms while `wait_reads` fell 6.84 ms. Work pushed
into the read window SHORTENS THE MEASURED WAIT rather than the compute phase, which is exactly what
overlap looks like from the driver's side, and it is why reading `layer_a` alone understated it.

So the routed split is worth roughly 0.83 x 69.4 x 2/3 = **~38 ms of 434, or 5-12 %** once the
second kernel for the missing pairs and the un-movable HC residual are paid for. Worth building; not
the 13.5 % the device time alone suggests.

### The shared-expert gain, restated

This job gives +1.6 % for shared_first (6.49 vs 6.39 tok/s, one rep each, 300 steps) against +0.2 %
from job 395's matched arms (600 steps). Those disagree and neither has the reps to settle it. Job
405 replicates both arms twice with the arm order alternated. Until then: shared_first is worth
somewhere between nothing and 1.6 %, and the reason to keep it is the mechanism, not the number.

## 22. Three nulls that were all measuring the wrong thing (jobs 395, 400, 405)

Job 405 replicates the shared-expert A/B properly -- four arms, 600 steps, arm order alternated per
rep -- and it is a **null**: sf=1 6.66/6.64 against sf=0 6.65/6.64, **+0.08 %**.

| | sf=0 rep1 | sf=0 rep2 | sf=1 rep1 | sf=1 rep2 |
|---|---|---|---|---|
| `layer_a` | 123.00 | 123.07 | 119.46 | 119.96 |
| `wait_reads` | 268.32 | 270.26 | 268.17 | 268.59 |
| `shared` | -- | -- | 4.11 | 3.88 |
| wall/step | 443.30 | 443.62 | 442.28 | 443.76 |

**FIRST: THIS WITHDRAWS THE "TWO THIRDS REALIZATION" CALIBRATION OF SECTION 21.** That came from job
400's single pair, where `wait_reads` fell 267.2 -> 260.4. With four arms `wait_reads` is flat to
within 2 ms and the 6.8 ms drop was noise. What actually happens is `layer_a` falls 3.3 ms, the
`shared` row costs 4.0 ms of host launch, and the wall does not move.

**SECOND, AND IT INVALIDATES WHAT ALL THREE JOBS WERE LABELLED AS.** The driver's `shared()` call
sat BETWEEN the two `wait_reads` branches:

    if policy.resolve_blocks:   wait(reads)      <-- fires under Policy(), which is V1
    if leaves.shared_first:     shared(L)
    if not policy.resolve_blocks: wait(reads)

`bench_tokens` constructs `Policy()`, i.e. V1, where `resolve_blocks` is True. So the wait fired
first and the shared expert was enqueued AFTER this layer's reads had already landed. **It never
overlapped anything.** The comment directly above it -- "can run it here, while the reads fly" --
described the other branch. Three jobs, eight arms and a bitwise gate all measured a reordering
inside the post-wait region, which is worth nothing, and correctly reported nothing.

The seam now sits ahead of both branches. With that, the two branches became the same statement
twice and collapsed to one; `resolve_blocks` still shapes the run through `_issue_speculation` and
the barriers, it is only this ordering that stopped depending on it.

### What survives from those jobs

The graph budget, which did not depend on the placement and replicates across both:

| graph | job 400 | job 405 rep1 / rep2 |
|---|---|---|
| A | 46.20 / 46.08 | 46.17 / 46.09 |
| B with shared | 79.36 | 79.68 / 79.74 |
| B without | 69.42 | 69.67 / 69.87 |
| S alone | 10.44 | 10.55 |

**B_r = 69.4-69.9 ms/step** and **graph A = 46.1 ms/step that can never overlap** both stand. So does
the fact that graph S is 10.0-10.5 ms by two independent instruments.

### The lesson, which is the same one as the four earlier scheduler verdicts

A gate that proves the OUTPUT is bitwise identical says nothing about whether the code under test
ran in the position you think it did. `falsify_shared_first.py` was right that the split is exact --
and exactness was never the question the A/B was asking. Job 410 re-runs it against the fixed seam,
with the outcomes pre-registered.

## 23. With the seam fixed, the overlap works -- and prices the routed split (job 410)

The same A/B as jobs 395/405, against the corrected placement (shared expert enqueued BEFORE the
read wait rather than between the two wait branches). 86 GB, 600 steps after 150 warm, arm order
alternated:

| | sf=0 rep1 / rep2 | sf=1 rep1 / rep2 | delta |
|---|---|---|---|
| `layer_a` | 122.77 / 122.38 | 116.34 / 116.91 | **-5.95 ms** |
| `shared` | -- | 2.33 / 2.08 | +2.20 ms |
| `wait_reads` | 263.78 / 270.07 | 266.48 / 269.15 | +0.9 ms (flat) |
| wall/step | 438.87 / 445.12 | 438.03 / 439.53 | -3.22 ms |
| tok/s | 6.71 / 6.62 | 6.73 / 6.70 | **+0.75 %** |

**THE MECHANISM IS ESTABLISHED.** `layer_a` falls 5.95 ms against within-arm spreads of 0.39 and
0.57, and the wall follows: -5.95 saved, +2.20 paid in launch, -3.75 predicted against -3.22
measured. Under the broken seam the same change moved `layer_a` only 3.3 ms and the wall not at all,
so the placement was the difference.

**THE MAGNITUDE, over 12 arms (jobs 410 + 415).** Four more reps each, order alternated:

| | n | mean tok/s | `layer_a` | range |
|---|---|---|---|---|
| sf=0 | 6 | 6.638 | 122.48 | 6.60-6.71 |
| sf=1 | 6 | **6.703** | **116.62** | 6.63-6.79 |

**+0.98 %**, t ~ 2.25 on a two-sample test, p ~ 0.05. Marginal on throughput; unambiguous on the
mechanism, where `layer_a` separates by 5.86 ms with non-overlapping ranges across all 12 arms. The
accounting closes: -5.86 saved, +2.4 paid in launch, -3.5 predicted against -4.7 measured.

RECOMMENDATION: leave the default OFF. One per cent at p ~ 0.05 does not justify changing what
production captures (40 extra graphs) on its own, and the value of this work is the conversion rate
below, not the shared expert. `DSV41_SHARED_FIRST=1` is there for anyone who wants it, and it is
bitwise identical.

### The conversion rate, which is the number that matters

Graph S is 10.0-10.5 ms of device time (two instruments, jobs 400/405), and moving it bought 5.86 ms
of critical path over 12 arms -- **about 57 %**. The rest is presumably device time that was already overlapped
with something, or that falls outside the window.

That rate is the multiplier on the routed-MoE split. At a ~83 % hit rate, 0.83 x 69.7 = 57.9 ms of
graph B is in pairs whose experts are already resident:

    57.9 ms movable  x  0.60 conversion  -  ~2 ms launch  =  ~33 ms of 442  =  ~7.5 %

This supersedes the 5-12 % of section 21, which rested on a "two thirds" figure taken from a single
pair where `wait_reads` moved 6.8 ms -- noise, as four arms later showed. The estimate here rests on
a measurement in which the seam demonstrably works.

### What would have to be true for it to fail

The split needs two kernel launches over disjoint rows of the existing `parts` buffer -- resident
pairs first, missing pairs after the wait -- then the unchanged `parts.view(K,T,DIM).sum(dim=0)`.
`block_m` must be pinned so both launches tile pairs exactly as the single launch does, or the
per-pair accumulation order changes and the result is no longer bitwise identical. The prerequisite
the dependency review named -- per-expert output buffers and a fixed-order reduction -- already
exists in `tools/cb3_moe.py::moe_forward`.

## 24. The early-H2D path, finally exercised -- and it is worth ~1 % (job 420)

Every measurement before this one ran with `Policy()`, i.e. V1, where `global_barrier` is True. That
makes `_wait()` call `wait_all()`, which blocks on `_demand == 0`, and `_demand` is decremented in
`_complete_h2d` AFTER `handle.synchronize()`. So the driver already waited for every demand copy to
LAND before it reached `wait_slots` -- and the early `ready.set()` at enqueue, the published event
and `await_copies` all sat downstream of a full physical barrier. Found in review, confirmed in the
code, and it means the device-ordered fast path had never been exercised at all.

Three reps each, order alternated, token equality held throughout (1768 tokens, accept_len 2.95):

| arm | rep 1 | rep 2 | rep 3 | mean |
|---|---|---|---|---|
| `global_barrier=True` | 6.54 | 6.66 | 6.58 | 6.593 |
| `global_barrier=False` | 6.73 | 6.61 | 6.63 | **6.657** |

**+0.97 %.** That is the pre-registered 0-2 % branch: keep the architecture for its correctness
properties, stop treating it as a speed project.

**AND IT CORRECTS TWO OF MY OWN REPORTS.** The first pair alone read +2.9 % and I said so; with all
six arms it is +0.97 %. A single pair on this box is not a result -- the sf=0 arms here span 6.54 to
6.66, which is 1.8 % on its own.

**What it does NOT rescue is job 370's conclusion.** I wrote there that "Python synchronisation is
not material in the ~0.46 s step". That claim was unsupported: 370 removed host-side machinery that
sat behind a barrier which dominated it, so the experiment could not have returned anything else.
The honest combined statement from 370 (locks, 0.08 %) and 420 (the barrier itself, 0.97 %) is that
**host-side scheduling in this engine is worth about one per cent -- not because locks are cheap,
but because the step is 269 ms of NVMe read wait out of 443.**

## What this closes and what it leaves

- Closed here: the engine-footprint explanation for the read penalty (refuted by its own bare stage).
- Bounded here: every compute/IO overlap project, at 3.8 % combined. NOT every scheduler toggle --
  see the correction in section 1; an NVMe/H2D overlap lever is outside that arithmetic.
  **SUPERSEDED for the real engine by section 19**: that 3.8 % came from the loader-overlap family on
  a modelled provider. Measured phases on the real one put it higher, but see the withdrawal in section 19: the realisable bound is graph B's cost, not total GPU time.
- Closed by sections 8-9: prediction of expert IDENTITY, by any of co-occurrence, recurrence, or
  the DSpark drafter. Routing entropy 8.52/8.58 says there is almost nothing to infer.
- Live, and non-predictive: CONCURRENCY. More requests in flight means more misses per layer to
  issue, which raises read depth without knowing anything about the future -- the one lever left
  that the entropy result does not touch.
- Live, and already the biggest measured effect: arena capacity.
