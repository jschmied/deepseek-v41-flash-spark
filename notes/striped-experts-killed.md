# Sub-expert stripe residency: killed offline, with the condition that would revive it

The proposal: the routed FFN decomposes exactly as
`y = sum_j W2[:,J_j] ( silu(W1[J_j,:] x) * W3[J_j,:] x )`, so an expert need not be an atomic cache
object -- replace `resident | not resident` with a continuous residency fraction and let an optimizer
spend the arena on stripes of hotter experts. No router change, no quantization-quality argument.

Evaluated on a recorded route trace (`routes-940.json`: 100 steps x 40 layers, 91,636 accesses, 9,492
distinct (layer, expert)) at the arena size job 1005 actually measured (5,421 slots = 78.4 GB).

**The simulation is calibrated**: whole-expert LRU at 5,421 slots fetches 252.7 GB, against the
engine's measured 259.0 GB on the five-prompt suite.

## Result: full residency wins everywhere, and partial residency loses badly

| layout | ranking | f=1.0 | f=0.5 | f=0.25 | f=0.125 |
| --- | --- | --- | --- | --- | --- |
| L1 (today's layout) | oracle | **-52.1 %** | +233 % | +316 % | +358 % |
| L1 | causal | +54.7 % | +276 % | +338 % | +369 % |
| L2 (W2 repacked) | oracle | **-52.1 %** | +150 % | +275 % | +337 % |
| L2 | causal | +54.7 % | +214 % | +307 % | +354 % |

## Why -- structural, not a tuning failure

Exact FFN evaluation needs **every** stripe of a routed expert. So a partially resident expert never
gets a free hit; it pays `(1-f)` of its bytes on every single access. Striping therefore trades "some
experts always free" for "more experts always cheaper", and the trade is decided by how much of the
working set the budget can already hold outright.

Here the budget holds 5,421 of 9,492 distinct experts, 57 %, and the hottest 10 % of experts take
39.6 % of accesses. Full residency turns most accesses into free hits (measured miss rate 0.20), while
f=0.5 holds *all* experts at half and pays 0.43 of every access forever. 0.43 > 0.20, so it loses, and
every finer fraction loses harder.

**The crossover condition, worth keeping for when it changes:**

    striping wins only when   miss_rate(whole) > 1 - slots / distinct_experts

Today 0.20 vs 0.43. It would take a working set far larger relative to the arena -- many concurrent
requests, or a regime where distinct experts touched approaches the 15,360 cap while slots stay near
5,400 -- to flip it. At that point this is worth re-running, and the simulator is kept for it.

## Two physical facts the proposal assumed away

**W2 cannot be partially fetched in today's layout.** W1/W3 are `[INTER, DIM]`, so a stripe of INTER
rows is contiguous. W2 is `[DIM, INTER]`, so a stripe is a *column* slice: 144 B out of each of 5,120
rows of 576 B. At 4,096 B page granularity that touches every page of W2, so a column stripe of W2
saves **zero** bytes. W2 is 4,608,000 of the 13,773,312 B record, so today's layout can strip at most
**66.5 %** of an expert; the rest needs a repack to INTER-major and a transposed down-kernel. The L1
rows above price that honestly, and L2 is the "if we do that work" bound. It does not help: L2 loses
too, for the structural reason above.

**"Pin every W2, stream only W1/W3" is not free either.** Pinning W2 for all 9,492 distinct experts in
this trace costs 45.9 GB of the 78.4 GB budget (59 %); at the full 15,360 it is 74.3 GB of 78.4 (95 %),
leaving 2.8 % of the arena for W1/W3. Once W2 fetches are charged to the experts the budget cannot
cover, that configuration collapses onto L2 exactly.

## Two bugs in my own first two runs, recorded because they nearly became results

1. I parsed the trace key as `layer:step`; it is **`step:layer`** (field 0 spans 0..99, field 1 spans
   0..39). That keyed the cache by `(step, expert)`, destroyed all cross-step reuse, and reported
   35,184 distinct experts -- impossible, since 40 x 384 = 15,360. The impossible count is what caught
   it; the baseline then moved from 1,123 GB to 252.7 GB and matched the engine.
2. The first "oracle" arm never made any expert partially resident: the budget was exhausted by fully
   resident experts before the partial tier ran, so its -52.1 % was measuring **frequency-static
   caching against LRU**, not striping at all. That number survives above, correctly relabelled -- it
   is the one genuinely interesting by-product here (see below) and has nothing to do with stripes.

## The by-product worth keeping

`f=1.0 oracle` is -52.1 % against LRU at the same budget: a perfect-frequency **whole-expert** static
cache halves NVMe traffic. That is an oracle and not realizable -- the causal version of the same
ranking is +54.7 %, i.e. much worse than LRU, because a static assignment cannot adapt while LRU can.
But it bounds what any frequency/recency policy could reach and it is a whole-expert result, so it
belongs with the eviction-policy work rather than here.
