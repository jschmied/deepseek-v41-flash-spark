# Field survey: MoE expert offloading, September 2026

Where DS4.1-Flash-on-one-Spark sits against published systems, and which of their ideas are worth
taking. Done because the field has moved and several of our open questions are ones other people
have now answered — or reframed.

**Our position**: 40 layers x 384 routed experts, top-6, streamed from NVMe through a CB3 arena.
At 86 GB / 6,243 slots that is **41 % of the 15,360 (layer, expert) pairs resident**, hit rate
**0.93**, **6.90 tok/s** single-stream on GB10. Every throughput number we have is the **null** arm:
no prefetch, no prediction.

---

## 1. The reframing that matters: prediction *as* routing, not as prefetch

**Edge0 / "The Other Half of the Memory Wall"** ([arXiv:2609.18063](https://arxiv.org/abs/2609.18063),
2026-09-16) serves a 35B MoE from SSD at ~20 tok/s in ~3 GiB of peak active memory on a single 24 GB
machine, within a few benchmark points of its fp16 teacher.

Their statement of the problem is exactly ours:

> naive offloading to SSD does not help on its own, because layer N+1's experts must be chosen
> before layer N's output exists, so the reads cannot start early enough to hide behind compute

Their answer is not ours. A trained per-layer *prerouter* predicts the next layer's routing one token
ahead, and **the prediction is consumed as the routing itself** — the staged expert set *equals* the
routed set, so nothing is ever missed. Recall stops being a variable: it is 1.0 by construction, and
the cost moves entirely into model quality, which they buy back with an unmerged recovery LoRA.

**This is a third role for prediction**, and our standing list does not have it:

| role | status here |
|---|---|
| PROTECTION (predict to avoid eviction) | CLOSED — perfect oracle worth 0.4 pp |
| PREFETCH (predict to start the read early) | the live question; job 895 is measuring it |
| **ROUTING SUBSTITUTION (the prediction *is* the route)** | **not considered** |

It is a genuine departure for us: this project's discipline is bitwise identity against v1, and
substitution deliberately changes what the model computes. But it is the only published approach that
makes the recall question disappear rather than answering it, and it deserves a decision rather than
silence. **Not proposing it — recording that the option exists and what it costs.**

## 2. Where our numbers actually sit

Cross-system hit rates are close to incomparable (different models, residency fractions, and backing
stores), so this is orientation, not a scoreboard:

| system | backing | residency | hit rate |
|---|---|---|---|
| MoE-Infinity ([arXiv:2401.14361](https://arxiv.org/pdf/2401.14361)) | CPU RAM, LFU | 10 % | ~17 % |
| MoE-Beyond ([arXiv:2508.17137](https://arxiv.org/pdf/2508.17137)) | learned activation prediction | 10 % | >70 % |
| SP-MoE ([arXiv:2510.10302](https://arxiv.org/pdf/2510.10302)), DeepSeek | speculative decode + prefetch | — | 40.1 % |
| tinyserve / vLLM RFC #38256 | **CPU pinned RAM**, LFRU + temporal prediction | 238 slots, GPT-OSS-20B | **97-100 %** |
| **ours** | **NVMe**, age/(1+count) | **41 %** | **93 %** |

The tinyserve figure is the one that looks alarming and is the least comparable: it backs from **CPU
RAM over PCIe**, not from SSD. On GB10 the arena *is* host memory, so their entire CPU-offload tier
is something we get for free — our bottleneck is one tier further down. A 97 % hit against RAM and a
93 % hit against NVMe are not the same achievement in either direction.

## 3. vLLM RFC #38256 — the closest thing to our architecture, and where we differ

[RFC #38256](https://github.com/vllm-project/vllm/issues/38256) (open since 2026-03-26, PR #37190):
expert weights in CPU pinned memory, fixed-size GPU cache of the hottest experts, LFRU eviction,
cross-layer prediction. Its design principle — *"the cache is a weight provider, not a special forward
path"* — is the same seam discipline as our `Leaves`.

Three things in it are worth naming:

* **"Expert loads per layer (batched prefill) O(num_experts) vs O(seq_len x top_k)."** That is our
  layer-major prefill, arrived at independently. Confirmation, not news.
* **Their eviction is LFRU; ours is age/(1+count)**, measured at +8.0 % over LRU on two independent
  harnesses (jobs 140 and 360). Same family, and we have the A/B they do not.
* **Cold start**: they report 48-56 % hit for the first 80-160 tokens, and a commenter proposes
  seeding the cache from GGUF `imatrix` activation counts to remove it. **We already do this** —
  `v41_engine.py:616` ranks the warm start with `rank_from_trace(trace_stats, profile=hot_profile)`
  from a coverage file. Our equivalent ramp (job 505: 0.78 -> 0.93 over ~200 steps) is what remains
  *after* seeding. Worth saying on the RFC; it is a real data point they are asking for.

## 4. The one finding that should change what we do next

**SP-MoE**: *"speculative prediction is effective due to residual connections in Transformer blocks,
but its effectiveness decreases as prefetch distance increases."*

Job 895 is running `HORIZON=2`. If the distance-decay result holds for us, horizon is not a free
parameter to leave at whatever the harness defaulted to — it trades lead time against accuracy, and
the optimum may be 1. That is a cheap arm to add once 895 says whether prefetch converts at all, and
it is the kind of parameter this project has previously left implicit and then been bitten by.

## 5. What we are NOT taking

* **In-place expert compression / SVD merging** (TurboQuant, D2-MoE — raised and answered in the RFC
  thread): solves a different problem, reducing the footprint of a model that fits but is tight. Ours
  does not fit at all. The RFC author's reply is right and applies to us unchanged.
* **llama.cpp `--n-cpu-moe`**: CPU-RAM expert placement with CPU compute. On GB10 there is no PCIe
  hop to avoid, so the mechanism has no purchase here.
* **KTransformers**: no GPU expert cache at all; not comparable.

## 6. Honest gaps this survey exposes

1. **We have never measured a predictor end to end.** Every number is the null arm. Job 895 is the
   first real attempt, and it only became possible after the `seq_step` continuation fix.
2. **Our hit rate is high because our residency is high** (41 % vs the 10 % these papers target). The
   systems reporting 70-100 % at 10 % residency are solving a harder version of the problem. We
   should stop quoting 0.93 as if it were comparable.
3. **We have no quality-vs-speed axis at all.** Edge0 trades a few benchmark points for 20 tok/s.
   We hold bitwise identity and take 6.90. Both are defensible; only one of them has been *chosen*
   here, and it was chosen by default.

## Sources

- [The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction (arXiv:2609.18063)](https://arxiv.org/abs/2609.18063)
- [vLLM RFC #38256: Incremental MoE Expert Offloading](https://github.com/vllm-project/vllm/issues/38256)
- [MoE-Infinity (arXiv:2401.14361)](https://arxiv.org/pdf/2401.14361)
- [MoE-Beyond: Learning-Based Expert Activation Prediction on Edge Devices (arXiv:2508.17137)](https://arxiv.org/pdf/2508.17137)
- [SP-MoE: Speculative Decoding and Prefetching (arXiv:2510.10302)](https://arxiv.org/pdf/2510.10302)
- [A Survey on Inference Optimization Techniques for MoE Models (arXiv:2412.14219)](https://arxiv.org/pdf/2412.14219)
- [FineMoE / fine-grained expert offloading (arXiv:2502.05370)](https://arxiv.org/pdf/2502.05370)

---

## Both borrowed ideas measured (jobs 900, 905) — one holds, one does not

### Prefetch converts, but less than the analysis said (job 900)

All arms replay the identical route sequence (2480/2480 reproduced), so tokens are equal by
construction and the window span *is* the throughput comparison. Measured directly, not reconstructed:

| arm | span | steps/s | vs null | demand fetches | ready / late |
|---|---|---|---|---|---|
| null | 17.844 s | 3.474 | — | 19,280 | — |
| recall 0.305, h2 | 16.551 | 3.746 | +7.8 % | 13,560 | 5,722 / 36 |
| recall 0.60, h2 | 13.540 | 4.579 | +31.8 % | 7,942 | 9,863 / 1,543 |
| **oracle, h2** | 11.124 | 5.573 | **+60.4 %** | 610 | 1,170 / **17,610** |
| recall 0.60, **h1** | 13.433 | 4.616 | +32.9 % | | |
| recall 0.60, **h4** | 12.421 | **4.992** | **+43.7 %** | | |

**Two standing claims are now wrong.** The corrected ceiling put a perfect oracle at **+81-84 %**;
measured it is **+60.4 %**. And recall 0.30 was to capture **28-41 %** of that win; it captures
**12.9 %**. The recall→win curve is markedly *convex*, not near-linear — which matters, because the
whole "our untrained transition table is already at 30.5 % recall" argument rests on the linear
reading. At 12.9 % of a 60 % ceiling, an untrained table is worth ~8 % and is not obviously worth
building on.

The mechanism is visible in the last column: the oracle issues 18,780 prefetches and **17,610 arrive
late**. Beyond some recall the binding constraint stops being knowledge and becomes device bandwidth
(1.02 GB/s vs null's 0.81). Demand-fetch reduction stays ~1:1 with recall; throughput does not follow
it, because a late prefetch still blocks -- just on a read already in flight.

**Horizon is not a free parameter, and longer wins.** At fixed recall 0.60: h1 +32.9 %, h2 +31.8 %,
h4 **+43.7 %**. SP-MoE's distance-decay result does **not** apply at our scale -- lead time still
dominates. `HORIZON=2` was the harness default that nobody chose, and it was costing ~9 %.

### The union law does NOT transfer (job 905)

RFC #38256's sizing law: *"the expert side's value function is a cliff at the live union, not a
curve"* -- on OLMoE at batch 8, 24 → 48 slots gave 2.59x decode. Our live union is 916 slots
(22.9 mean experts/layer/step x 40 layers), measured from job 895's recorded routes.

| slots | x union | % of 100-step working set | tok/s | hit |
|---|---|---|---|---|
| 622 | 0.68x | 7 % | 0.68 | 0.015 |
| 899 | 0.98x | 9 % | 0.81 | 0.202 |
| 1,383 | 1.51x | 15 % | 1.13 | 0.467 |
| 2,767 | 3.02x | 29 % | 2.13 | 0.746 |
| 6,243 (job 875) | 6.8x | 66 % | 6.90 | 0.935 |

**There is no cliff at the union and no plateau above it.** Gain per slot-doubling *accelerates*:
x1.45 slots -> x1.19 tok/s, then x1.54 -> x1.40, x2.00 -> x1.88, x2.26 -> **x3.24**. The largest
returns are far ABOVE the union, not at it.

Why it differs: their per-layer union is 35.3 of 64 experts -- **55 % of the pool**, so covering it
is the whole problem. Ours is 22.9 of 384 -- **6 % of the pool**, while 237 distinct experts per
layer are touched across 100 steps. Our value function is governed by the **working set (9,492
slots)**, not the single-step union. The law is real; it is a law about a regime we are not in.

This is the cross-model check the RFC asked for, and the answer is a negative worth telling them.

**Scope**: P0 prompt, 3 reps, temperature 0, single stream. The 622-1,383 slot rungs are pathological
(hit 0.015-0.467), not serving configurations -- they exist to locate a cliff, and there isn't one.

### Horizon has an optimum, and the ceiling I reported was an artifact of missing it (job 910)

Same session, same null, so these are comparable to each other and **not** to job 900's:

| arm | span | steps/s | vs null |
|---|---|---|---|
| null | 19.090 s | 3.248 | — |
| recall 0.60, h4 | 12.821 | **4.836** | **+48.9 %** |
| recall 0.60, h8 | 12.964 | 4.782 | +47.2 % |
| recall 0.60, h16 | 13.149 | 4.715 | +45.2 % |
| oracle, h4 | 10.947 | 5.663 | **+74.4 %** |
| oracle, h8 | 10.942 | 5.666 | +74.4 % |

**Horizon peaks at 4**: h1 +32.9, h2 +31.8, h4 +48.9, h8 +47.2, h16 +45.2. SP-MoE's distance decay
is real after all -- it bites beyond 4, not at 2, which is why job 900's h1/h2/h4 slice looked like
"longer always wins".

**The +60.4 % ceiling was an h2 artifact.** At h4 the oracle reaches **+74.4 %**, close to the
predicted +81-84 % and far from what I reported an hour ago. The tell was already in job 900: 17,610
of the oracle's 18,780 prefetches arrived LATE. That was a lead-time failure being read as an engine
limit. h8 ties h4 exactly, so the oracle is saturated at h4.

**A methodological correction that invalidates some of my own arithmetic.** The null arm measured
3.474 steps/s in job 900 and **3.248** in job 910 -- a **6.5 % run-to-run swing on the arm that is
supposed to be the fixed reference**. Every cross-job percentage I computed tonight inherits that
error. Only same-session comparisons are safe, and the recall→win fractions quoted above from job 900
(12.9 % for recall 0.305) should be re-measured at h4 in one session before anything rests on them.

What still stands: recall 0.60 at h4 captures 48.9/74.4 = **66 %** of the oracle win for 60 % recall
-- roughly linear, not the convex shape h2 suggested. The convexity claim was also an h2 artifact.

### The clean recall curve, and the oracle is not the ceiling (job 915)

One null, one session, h4 throughout -- both confounds from the earlier runs removed:

| recall | span | steps/s | vs null | % of oracle win |
|---|---|---|---|---|
| null | 19.290 s | 3.214 | — | — |
| 0.15 | 17.960 | 3.452 | +7.4 % | 10.3 % |
| **0.305** | 16.540 | 3.748 | **+16.6 %** | **23.2 %** |
| 0.45 | 14.522 | 4.269 | +32.8 % | 45.8 % |
| 0.60 | 12.711 | 4.878 | +51.8 % | 72.3 % |
| **0.80** | 10.741 | **5.772** | **+79.6 %** | **111 %** |
| oracle (1.0) | 11.243 | 5.514 | +71.6 % | 100 % |

**Recall 0.80 beats perfect knowledge.** The oracle issues every prefetch it can -- 18,780, of which
17,610 arrived late in job 900 -- and saturates the device; recall 0.80 issues fewer and lands more
of them in time. A "perfect oracle" arm is therefore NOT an upper bound on this engine, and calling
it the ceiling was wrong. The real optimum is a rate, not a recall: somewhere around 0.8 the benefit
of knowing collides with the cost of asking.

That reframes the design question. It is not "how much recall does a predictor need" but "how many
prefetches per step can the device absorb, and which ones" -- a throttling and priority problem that
a predictor feeds, rather than a prediction-accuracy problem.

**Our untrained transition table (30.5 % recall) is worth +16.6 %**, capturing 23.2 % of the oracle
win -- below the pre-registered 28-41 % band, but the shape is near-linear in recall up to 0.8, so
training it toward 0.6 would roughly triple the payoff (+51.8 %). That is the number the "before any
training compute is spent" decision actually needed, and it took four jobs and two confound fixes to
get it right.

**Scope**: P0 prompt, 60 timed steps after a 30-step warm-up, 40 GB arena, single stream, recorded
routes replayed identically in every arm (2480/2480). The recall arms are a recall-degraded ORACLE,
not a real predictor -- they model what a predictor of that recall would fetch, with precision 1.0.
A real predictor also misfires, and precision < 1 costs slots and bandwidth that these arms never pay.

### The precision sweep is void, and it found something else (job 925)

| arm | steps/s | vs null | pred_miss | wasted |
|---|---|---|---|---|
| null | 3.349 | — | 0 | 0 |
| recall 0.60, p1.00 | 4.774 | +42.6 % | 0 | 0 |
| recall 0.60, p0.80 | 5.529 | +65.1 % | 8,095 | 7,176 |
| recall 0.60, p0.60 | 5.570 | +66.3 % | 22,410 | 19,936 |
| recall 0.60, **p0.40** | **5.769** | **+72.3 %** | 50,065 | 44,644 |
| oracle | 5.477 | +63.5 % | 0 | 0 |

**Lower precision measured faster, monotonically, and p0.40 beat the oracle.** That is not a finding
about precision; it is a broken instrument, and the flaw is in code I wrote.

`_RecallFromTrace` draws misfires from outside the truth **for that call**. But a layer touches 237
of its 384 experts across 100 steps, so an expert that is wrong for *this* step is very likely right
for a later one. The `wasted` counter means "not used at the predicted (layer, step)" — it does not
mean "not used". So the precision axis did not inject misfires; it injected **extra prefetch
breadth**, and on a cache holding 29 % of the working set more breadth is simply better.

Two consequences, and they point in opposite directions:

1. **The precision question is unanswered.** A real predictor's misfires are experts the trace never
   wants at that layer at all. Modelling them needs the draw taken from the complement of the layer's
   *whole-trace* expert set, not of one call's route. Until that is fixed, the recall curve's status
   as an upper bound is still unverified — which was the entire point of the job.
2. **Prefetch breadth looks like a real lever, discovered by accident.** Issuing more experts per
   layer than the predicted set was worth +30 pp over p1.00 here. That is a *width* knob, orthogonal
   to recall and horizon, and nothing in this project has ever tested it deliberately. It is also
   consistent with job 905: we sit at 29 % of the working set with an accelerating capacity curve, so
   anything that pulls more of the working set in early pays.

Recorded as an accident rather than a result. The next job fixes the misfire draw and tests width as
its own axis, so the two are not confounded again.

### Queue depth refuted — and it exposes something worse (job 935)

The hypothesis: job 930 showed *genuinely useless* reads making things faster (p0.40 +82.1 % vs
p1.00 +52.8 %, with 7x fewer useful prefetches landing in time), and the null arm leaves the device
idle 32.5 % of the window. So maybe "prefetch" was partly measuring queue occupancy, not knowledge.

**Refuted.** Giving the null arm more depth makes it dramatically worse:

| arm | steps/s | device idle | in-flight |
|---|---|---|---|
| null 8/8/8/2 | 3.119 | 32.3 % | 4.45 GB/s |
| null 16/16/8/4 | 1.301 | 41.2 % | 4.53 |
| null **48/48/24/8** | **0.619** | 47.3 % | 4.90 |
| recall 0.60, 8/8/8/2 | 4.498 | 9.5 % | 4.47 |
| recall 0.60, 48/48/24/8 | 0.747 | 37.3 % | 5.00 |

So prefetch's win is real and not a queue artifact. But the shape of the refutation is alarming:
**48/48/24/8 is production's own loader configuration** (`io 48/96` is shipped), and in this harness
it is **5x slower** than 8/8/8/2 — with idle time *rising* as depth rises, while in-flight bandwidth
improves. Depth goes up, the device gets busier per read, and wall time explodes.

The harness's own comment at that call site anticipated the opposite:

> These were fixed at 8/8/8/2 while production runs io 48/96, and job 545 then saturated at
> 2.8 GB/s with the queue full ... against a device that does ~6.3 from depth 2. A ceiling measured
> at one eighth of the shipped concurrency is not a device ceiling until it has been swept.

It has now been swept, and the answer is the reverse of what that note expected.

**What this means for every prefetch number above.** They were all measured at 8/8/8/2 — one sixth of
shipped concurrency — on a loader that *degrades* when corrected toward production. The relative
ordering (more recall is better, horizon peaks at 4, precision barely matters) held across many arms
and is probably safe. The **magnitudes** are not: +52.8 % or +82.1 % against a baseline that is itself
a configuration production does not use.

Two candidate explanations, neither tested: contention (48 worker threads serialising on the arena or
the slot table, which would make depth actively harmful), or the mechanism already on the standing
priorities list for job 170 — `wait_stream` records its event where CALLED, so more queued work means
longer waits for the very reads it meant to overlap. The second would predict exactly this: depth up,
idle up, wall up.

**Next, and it is a loader question rather than a prediction one**: find out why depth hurts. Until
that is answered, the prefetch magnitudes should be quoted as "at 8/8/8/2" or not quoted at all.

### Which knob, and the retraction that follows (job 940)

One knob at a time from 8/8/8/2, because job 935 moved four together:

| arm | steps/s | vs baseline |
|---|---|---|
| baseline 8/8/8/2 | 3.290 | — |
| **workers 48** | **0.743** | **-77 %** |
| staging 48 | 3.354 | +1.9 % (noise) |
| read_qd 24 | 3.173 | -3.6 % |
| h2d 8 | 2.721 | -17.3 % |
| all four 48/48/24/8 | 0.622 | -81 % (job 935: 0.619 — session pinned) |

`N_WORKERS` alone reproduces nearly the whole degradation. That is the *contention* branch of the
pre-registered discriminator, not job 170's — `read_qd` and `h2d` were its signature and are minor.

**And then the mechanism turns out not to be contention either.** `enginev2/leaves.py::Bandwidth`
says what it is:

> The NVMe device, shared by every read in flight. EXACT, not sliced. total(n) = 5.58 GB/s at n=1 and
> 6.82 GB/s at n>=2, **flat above 2 because the device saturates near 2 concurrent reads. Each read
> gets total(n)/n.**

**The v2 harness models its reads; it does not perform them.** With 48 workers each read receives
6.82/48 ≈ 0.14 GB/s and takes ~24x longer than at n=2, so the demand read being waited on finishes
much later. The 5x is precisely what this model predicts. It is not a loader defect, and
"the v2 loader does not scale with worker count" is **withdrawn**.

It may still be real physics: for a device that saturates at 2 concurrent reads, 48-way issue
genuinely does stretch every individual read, and aggregate bandwidth being flat is exactly why.
That would make high loader concurrency harmful to *latency-critical* demand reads while leaving
throughput counters unchanged — consistent with `in-flight` staying 4.6-5.0 GB/s across every arm
above while wall time moved 5x.

### The caveat this puts on the whole night's prefetch sequence

Jobs 895-940 run **real model compute with modelled I/O**: `RealLeaves` drives the actual engine for
compute, while expert reads go through `Bandwidth`. So the prefetch results are statements about a
device model whose parameters were fitted, not about the NVMe in this box.

What that does and does not undermine:

* **Safe**: the *ordering* and *shape* results, which held across many arms and two confound fixes —
  more recall is better; horizon peaks at 4; precision barely matters; a perfect oracle is not an
  upper bound because it over-issues.
* **Not safe**: every magnitude. `+52.8 %`, `+82.1 %`, `+74.4 %` are all against a modelled device at
  8/8/8/2, and the model's own saturation point (n=2) is what makes worker count so punishing.
* **Untested**: whether the real NVMe saturates at 2 concurrent reads at all. Production ships
  `io 48/96` and works, which is weak evidence that it does not.

The next measurement worth making is therefore not another prefetch arm. It is a real-device
concurrency curve — achieved bandwidth and per-read latency against concurrent read count on this
NVMe — to find out whether `Bandwidth`'s n>=2 saturation is true here. Every number above is
downstream of that constant.

### The device saturates at ONE concurrent read (job 945)

Real NVMe, O_DIRECT, one 13,774,848 B record per read — the engine's own shape — random offsets over
a 211.6 GB file so neither page cache nor readahead helps. No engine, no arena, no model.

| conc | aggregate | per-read mean | p99 | model says per read |
|---|---|---|---|---|
| 1 | 4.83 GB/s | 2.8 ms | 3.1 | 5.58 GB/s |
| 2 | 5.15 | 5.3 | 6.1 | 3.41 |
| 4 | 4.90 | 11.2 | 12.9 | 1.71 |
| 8 | 4.79 | 22.6 | 32.8 | 0.85 |
| 16 | 4.77 | 44.7 | 113.7 | 0.43 |
| 24 | 4.88 | 64.7 | 178.7 | 0.28 |
| 32 | 4.88 | 84.2 | 173.0 | 0.21 |
| 48 | 4.91 | 121.0 | 257.8 | 0.14 |
| 96 | 4.86 | 209.8 | 426.6 | 0.07 |

**`total(n)/n` is the right sharing model.** Per-read latency is 2.8 ms x n to within a few percent
across two decades of concurrency. `Bandwidth`'s structure is correct and job 940's 5x is real
physics, not a simulator artifact — 48-way issue stretches every read 43x.

**But the device saturates at n=1, not n=2, and at a lower rate than modelled**: real aggregate is
flat at **~4.87 GB/s** everywhere, where the model uses 5.58 at n=1 and 6.82 at n>=2. The model is
optimistic by 15 % at n=1 and **39 %** above it. So modelled reads finish sooner than real ones, and
a late prefetch costs *more* in reality than jobs 895-940 charged it. Their ordering stands; their
magnitudes are measured against a device kinder than this one.

### The consequence is a policy split, and it is shippable

There is **no aggregate benefit to read concurrency at all** — 1 read saturates the device. So
concurrency is purely a latency tax, and the two phases want opposite settings:

* **Prefill** issues reads it will all consume. Aggregate is what matters, aggregate is flat, so
  concurrency neither helps nor hurts total time. 48 workers is harmless here.
* **Decode** blocks on *particular* reads. Latency is what matters, and 48-way issue makes the read
  you are waiting for 43x slower. Every concurrent read beyond the one you need is a tax.

Production ships `io 48/96` globally. On this evidence decode should run a *shallow* reader and only
prefill a deep one — which is the opposite of one setting for both, and costs nothing to try.

This also reframes the prefetch results one more time. A prefetch is only worth issuing if it will
complete before its layer arrives; at n concurrent reads that takes 2.8 ms x n. The oracle's failure
mode in job 900 (17,610 of 18,780 arriving late) was never about knowing *what* to fetch — it is that
issuing 18,780 reads into a device that serves one at a time cannot possibly land them in time. The
lever is admission control, and it is bounded by 4.87 GB/s no matter what predicts.

### The prediction holds on the real engine: io 48 -> 2 is +22.5 % decode (job 950)

`DSV41_IO_THREADS` swept on the real v1 engine at decode, 40 GB arena, with a repeat 48 arm to pin
the session:

| io_threads | tok/s | load_wait_s | hit | GB/tok |
|---|---|---|---|---|
| **48 (shipped)** | 2.12 | 42.02 | 0.7456 | 1.382 |
| 8 | 2.17 | 40.13 | 0.7456 | 1.382 |
| 4 | 2.48 | 34.27 | 0.7456 | 1.382 |
| **2** | **2.61** | **32.60** | 0.7456 | 1.382 |
| 48 again | 2.13 | 41.81 | 0.7456 | 1.382 |

**+22.5 %** for one environment variable, and the mechanism is confirmed rather than inferred:

* `load_wait_s` drops 42.0 -> 32.6, **-22.4 %**, matching the throughput gain almost exactly. The win
  is the wait, which is what job 945 predicted a latency tax would look like.
* **hit rate and GB/tok are identical to the digit in every arm.** Same reads, same bytes, same cache
  decisions — nothing about the workload changed, only how long the engine waited for reads it was
  always going to make. That is as clean a confirmation as this harness can produce.
* The two 48 arms bracket at 2.12 / 2.13, so the session is pinned and the difference is not drift.

**Scope, and it matters before shipping.** This is a 40 GB arena at hit 0.73-0.75, i.e. miss-heavy.
At 86 GB the hit rate is 0.93, so there are ~3.7x fewer misses to wait on and the win should shrink —
possibly a lot. It is also decode-only: prefill issues reads it will all consume, and job 945 says
aggregate bandwidth is flat, so prefill *should* be indifferent to depth, but that is an inference,
not a measurement, and the CB3 unpack path may interact.

So the shippable claim is not yet "set io_threads=2". It is "read concurrency is a pure latency tax
at decode, worth up to +22.5 % at high miss rates, and the shipped global 48 is the wrong setting for
at least one phase." The next job measures it at 86 GB and checks prefill is unharmed.

### At 86 GB the win moves to the cold phase (job 955)

Same sweep at the real operating point, 26,400-token prefill first in every arm:

| io_threads | prefill | rep1 (hit .897) | rep2 (.915) | rep3 (.920) | wait rep3 |
|---|---|---|---|---|---|
| 48 (shipped) | 87.1 s | 4.21 | 4.76 | 4.89 | 13.62 |
| 8 | 75.3 s | 4.15 | 4.90 | 5.00 | 13.32 |
| **2** | 70.3 s | **4.94** | 4.96 | 5.01 | 13.22 |
| 48 again | 74.8 s | 4.17 | 4.81 | 4.91 | 13.61 |

**Steady state: +2.2 %.** The +22.5 % from job 950 was a 40 GB effect at hit 0.73-0.75, and it
shrank almost exactly as predicted when the hit rate rose to 0.92 — ~3.7x fewer misses to wait on.

**First request: +17 %** (4.94 against 4.21 and 4.17). That is the same mechanism seen where it still
has purchase: the latency tax is proportional to the number of misses waited on, so it is largest when
the cache is cold and vanishes as it fills. io 2 also reaches 4.94 on rep1 — a number the io 48 arms
do not reach until rep3.

**Prefill: no evidence of harm, and no usable measurement either.** 70.3 s at io 2 is the fastest of
the four, but the two *identical* io-48 arms came in at 87.1 s and 74.8 s — a 16 % spread between
arms that differ in nothing. Prefill wall time in this harness needs repetition before it can support
a claim in either direction; one number per arm is not enough. Recorded as untested rather than
favourable.

### What is actually shippable

`DSV41_IO_THREADS=2` for decode: **+17 % on the first request, +2 % once warm, no measured prefill
cost.** Real serving starts cold and sees varied prompts, so the cold-phase number is not a corner
case — job 785 measured a 3.57-13.02 tok/s spread across five prompts, and every new prompt re-enters
the warming regime for the experts it needs.

The general statement is the one worth keeping: **read concurrency buys no bandwidth on this device
(flat 4.87 GB/s from n=1 to n=96) and costs latency linearly (2.8 ms x n), so it should be set by
which phase is latency-critical — not globally.** The shipped `io 48/96` was chosen for prefill
throughput and silently taxes every decode miss.

### io 2 wins all five prompts, and prefill is faster too (job 960)

Five prompts, each scored on its FIRST request at 86 GB, with io 48 run twice to pin the session:

| prompt | hit | io 48 | io 48 again | **io 2** | gain | load_wait |
|---|---|---|---|---|---|---|
| P0 | .846 | 3.09 | 3.19 | **3.52** | +10.3 % | 27.7 -> 22.9 |
| P1 | .925 | 11.82 | 11.63 | **12.01** | +1.6 % | 7.3 -> 6.4 |
| P2 | .949 | 13.42 | 13.24 | **14.50** | +8.0 % | 6.3 -> 5.0 |
| P3 | .927 | 7.44 | 7.93 | **8.05** | +1.5 % | 10.4 -> 8.8 |
| P4 | .906 | 3.82 | 3.94 | **4.32** | +9.6 % | 19.8 -> 16.1 |

**5 of 5, and `load_wait_s` falls 13-20 % in every arm.** The mechanism is not prompt-specific: it is
the latency tax being removed wherever there are misses to wait on. Gains track miss rate loosely
(P0 at hit .846 gains most) but the *wait* reduction is uniform, which is the cleaner signal.

**Prefill is faster, not harmed**: 71.8 s cold at io 2 against 76.0 and 79.5; 8.3 s warm against 9.5
and 9.4. So there is no phase split to make — shallow is better for both, and job 945 explains why:
read concurrency buys no bandwidth at all on this device, so depth is pure cost everywhere.

**A correction to job 955's write-up.** I reported a "16 % spread between identical io-48 arms" and
called prefill timing too noisy to use. This job's prefill x3 shows 76.0 / 9.5 / 9.1 s — the second
and third repetitions hit the PROMPT CACHE. That is cold-vs-warm, not variance, and 955's spread was
almost certainly the same artifact. My x3 design did not measure what I said it measured; repeating a
prefill on the same prompt cannot.

### Shipped

`DSV41_IO_THREADS=2` added to `.env` with the evidence in a comment. +1.5 to +10.3 % on first-request
decode across five prompts, 13-20 % less load wait in every arm, prefill 6-12 % faster, hit rate and
byte counts unchanged. The shipped 48 was chosen for prefill throughput on the assumption that read
concurrency buys bandwidth; job 945 measured that it does not.

### The read-piece sweep refutes my extrapolation, and sharpens the rule (job 965)

Same five prompts, io pinned at the newly shipped 2, sweeping the SECOND pool -- the one that splits
each 13,774,848 B expert into aligned pieces:

| config | pieces/expert | P0 | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|---|
| **96 x 4 MB (shipped)** | 3.3 | 3.73 | 12.79 | 15.79 | 8.79 | 4.66 |
| 96 x 4 MB again | 3.3 | 3.83 | 12.85 | 13.53 | 7.82 | 4.48 |
| 96 x 16 MB | 0.8 | 3.64 | 12.67 | 14.30 | 8.32 | 4.44 |
| 8 x 4 MB | 3.3 | 3.67 | 13.25 | 14.96 | 8.38 | 4.28 |
| 8 x 16 MB | 0.8 | 3.39 | 11.41 | 14.67 | 8.07 | 4.14 |

**Nothing beats the shipped configuration**, and `8 x 16 MB` -- the fewest concurrent pieces, the n=1
the device supposedly wants -- is lowest on four of five prompts. My extrapolation from job 945 was
wrong.

**Why, and this is the useful part.** Job 945's tax applies to concurrency *across independent reads*,
where only one of them is on the critical path and the rest stretch it. Pieces of a *single* expert are
not independent: all of them must land before that expert is usable, so splitting one 13.77 MB read
into 3.4 concurrent pieces costs nothing — aggregate is flat, so 3.4 pieces at total/3.4 finish in the
same 2.8 ms as one read at total. The tax needs a victim, and within one expert there is none.

So the rule is narrower than I wrote it: **read concurrency is a latency tax only when it is across
requests you do not all need yet.** `io_threads` fetches different experts, most of which the current
step does not need — taxable. `read_threads` fetches pieces of one expert, all needed — neutral.

**And a noise-floor correction that touches job 960.** The two identical 96 x 4 arms here differ by
17 % on P2 and 12 % on P3. First-request measurements on those two prompts are not reliable to better
than ~15 %, which means **job 960's +1.6 % (P1) and +1.5 % (P3) were noise**, not small wins. What
survives there is P0 +10.3 %, P2 +8.0 % and P4 +9.6 %, where the repeat arms agreed to 1.4-4 % — three
prompts, not five. `DSV41_IO_THREADS=2` still stands on those three plus the uniform 13-20 % drop in
`load_wait_s`, which is the counter that does not have this variance problem.

### io_threads=2 verified properly: 3 of 3 paired wins (job 970)

Summed over five prompts, arms interleaved 48/2/48/2/48/2 so drift cannot pass for an effect:

| round | io 48 | io 2 | gain | load_wait |
|---|---|---|---|---|
| 1 | 4.849 tok/s | 5.179 | +6.8 % | 19.1 -> 16.4 s |
| 2 | 4.817 | 5.244 | +8.9 % | 18.4 -> 16.9 s |
| 3 | 4.847 | 5.811 | +19.9 % | 18.7 -> 14.3 s |

**The summed metric fixes the noise problem that job 965 exposed.** The three io-48 arms land at
4.849 / 4.817 / 4.847 — a **0.7 %** spread, against 12-17 % on individual first-request prompts. Summing
five prompts averages out exactly the variance that made job 960's small numbers meaningless.

io 2 wins all three pairs. Its own spread is wider (5.18-5.81, round 3 high), so the honest headline is
the median: **+9 %**, not the +19.9 % of the best round. `load_wait_s` falls in every pair.

`DSV41_IO_THREADS=2` stays shipped, now on a paired design rather than single first-request numbers.
The pre-registered revert branch does not fire.
