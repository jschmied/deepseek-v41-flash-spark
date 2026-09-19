# Layer-aware caching and a two-hit doorkeeper: four ideas, four negatives

Offline on `routes-940.json` (4,000 events, 91,636 accesses, 40 layers, 9,492 distinct experts) at the
arena job 1005 measured, 5,421 slots. No GPU time. Simulator: `notes/data/cache_policy_sim.py`.

Every arm is charged **both** movements, because the host-mapped result turned on exactly this: today's
engine and today's cold path both pay NVMe -> pinned *and* pinned -> device for every miss. That is why
relocating the copy bought nothing, and why "which bytes move" is the only question left.

| arm | reads | copies | bytes | time | vs A (bytes) | vs A (time) |
| --- | --- | --- | --- | --- | --- | --- |
| **A global LRU (today)** | 18,352 | 18,352 | 505.6 GB | 47.6 s | — | — |
| B per-layer LRU, equal quota | 19,064 | 19,064 | 525.2 | 49.5 | +3.9 % | +3.9 % |
| C per-layer LRU, marginal-gain quota | 43,899 | 43,899 | 1,209.4 | 113.9 | +139 % | +139 % |
| E global LRU, protect next 8 layers | 18,287 | 18,287 | 503.8 | 47.4 | -0.4 % | -0.4 % |
| D doorkeeper, pool 64, promote-on-2 | 22,076 | 12,813 | 480.6 | 55.1 | **-4.9 %** | **+15.7 %** |
| D doorkeeper, pool 256, promote-on-3 | 25,133 | 9,082 | 471.3 | 61.4 | **-6.8 %** | **+29.0 %** |

Time weights each movement by what it actually costs on this box: an O_DIRECT record read is 2.357 ms
(job 1045) and a pinned -> device copy is 0.237 ms (job 1070, synchronous, an upper bound). **NVMe is
9.9x the copy.**

## The doorkeeper is a byte win and a time loss

It does what it was meant to: a single-use expert executes from the mapped pool and is never promoted,
so promotion bytes fall 27-50 %. But it pays for that by re-reading evicted-from-pool experts, and
NVMe reads rise 16-41 %. At ~10:1 cost the trade is firmly the wrong way round: total bytes fall 5-7 %
while total time rises 16-29 %.

This is the same trap as the stripe simulation, in a new place: **total bytes is not the objective**.
The two movements differ by an order of magnitude in cost, and any policy that trades one for the other
has to be scored in time.

The doorkeeper would only pay where a promotion is expensive relative to a read -- a much slower
interconnect than a shared DRAM pool, or a much faster device. Neither describes GB10.

## Layer partitioning loses, and phase-awareness is nil

Global LRU beats both partitioned arms. Equal quotas cost +3.9 %: a layer's working set varies, and a
fixed share wastes slots on layers that do not need them while starving ones that do. Marginal-gain
quotas are far worse (+139 %) because greedy allocation on an LRU miss curve starves whole layers --
some received zero slots, which makes every access to them a miss. *Caveat on that arm:* LRU miss
curves are not guaranteed convex, so greedy is not optimal here and the true optimum lies somewhere
between B and A; what the arm establishes is that naive marginal allocation is dangerous, not that no
quota scheme could work.

Phase-aware protection -- refusing to evict an expert whose layer is within *d* ahead -- is worth
-0.0 % to -0.4 % at d = 1, 2, 4, 8. The intuition is sound (a layer 1 ahead is reused within
milliseconds, one just behind waits a full traversal) but at 80 % hit rate and 5,421 slots there is
simply not enough eviction pressure for the choice of victim to matter much.

## What this says about where the remaining headroom is

The perfect-frequency oracle is still -52 % against LRU at the same budget (notes/striped-experts-killed.md),
so the gap is real. But it is not reachable by reordering evictions, partitioning by layer, or changing
admission: three independent attempts now land within a few percent of plain LRU, and the one that
moves bytes moves time the wrong way. Whatever closes that gap has to know something about the future
that recency does not -- which points back at prediction, and at the transition table that already sits
at 30.5 % recall untrained.
