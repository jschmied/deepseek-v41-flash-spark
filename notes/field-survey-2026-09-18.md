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
