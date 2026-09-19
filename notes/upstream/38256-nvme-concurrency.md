DRAFT — needs the user's go. GitHub issue comment, vllm-project/vllm RFC #38256 (2026-09-19).

Three measurements from a single-GPU MoE offload engine on a DGX Spark (GB10, sm_121), backed by
**NVMe** rather than pinned host RAM. Two of them cost us a shipped default, so they may be worth the
thread's time. All figures are single-stream at temperature 0; the box is GB10 with unified memory, so
there is no PCIe hop and our bottleneck sits one tier below this RFC's.

**1. Read concurrency buys no bandwidth on this device, and taxes latency linearly.**

Measured directly, not inferred: O_DIRECT, one 13,774,848 B expert record per read (our engine's own
read shape), random offsets over a 211.6 GB file so neither page cache nor readahead helps.

| concurrent reads | aggregate | per-read mean | p99 |
|---|---|---|---|
| 1 | 4.83 GB/s | 2.8 ms | 3.1 ms |
| 2 | 5.15 | 5.3 | 6.1 |
| 8 | 4.79 | 22.6 | 32.8 |
| 48 | 4.91 | 121.0 | 257.8 |
| 96 | 4.86 | 209.8 | 426.6 |

Aggregate is flat from one concurrent read to ninety-six; per-read latency is 2.8 ms x n to within a
few percent across two decades. The device saturates at **n=1**.

**2. So our shipped reader depth was wrong, by ~9 %.** We ran 48 I/O threads, chosen for prefill
throughput on the assumption that concurrency buys bandwidth. Dropping to 2, summed over five prompts
with arms interleaved 48/2/48/2/48/2:

| round | 48 threads | 2 threads | cumulative load_wait |
|---|---|---|---|
| 1 | 4.849 tok/s | 5.179 | 19.1 -> 16.4 s |
| 2 | 4.817 | 5.244 | 18.4 -> 16.9 s |
| 3 | 4.847 | 5.811 | 18.7 -> 14.3 s |

Three of three paired wins, median **+9 %**, and hit rate and bytes read are unchanged — the same
reads, the same bytes, less waiting. Attributed with the store's own per-phase
counters, identical `loads` and bytes in every arm: **read time per load falls 11.5 -> 4.1-4.5 ms
(-61 %, 31 s of 46 s saved)**, and H2D also falls (4.87 -> 0.94-2.03 ms, 12-17 s), with lease/sync/
submit negligible. So read latency is the dominant term but not the only one -- `io_threads` also sets
the copy streams and H2D concurrency, and both move.

One calibration worth stating: the curve above predicts 121 ms/read at n=48, and we measure 11.5 ms,
so the 48-thread pool never actually holds 48 reads in flight -- effective concurrency is ~4, falling
to ~1.5 at two threads. The tax is real; the pool was simply never saturated, which is why this is
~9 % and not the 5x the curve would allow.

**3. A cross-model check on the union law, since the thread asked for one.** The sizing law here — "the
expert side's value function is a cliff at the live union, not a curve" — does not transfer to our
model. Live union measured from recorded routes: 22.9 experts per layer per step of 384, so ~916 slots
across 40 layers.

| slots | x union | % of 100-step working set | tok/s | hit |
|---|---|---|---|---|
| 622 | 0.68x | 7 % | 0.68 | 0.015 |
| 899 | 0.98x | 9 % | 0.81 | 0.202 |
| 1,383 | 1.51x | 15 % | 1.13 | 0.467 |
| 2,767 | 3.02x | 29 % | 2.13 | 0.746 |

**No cliff at the live union, and performance keeps improving strongly through 3x union.** (We have a
6,243-slot point at 6.90 tok/s, but it comes from a different job with a different scale-plane layout,
so we are leaving it out rather than assert parity we cannot check.) We think the
difference is pool fraction: your per-layer union is 35.3 of 64 experts, **55 % of the pool**, so
covering it is the whole problem. Ours is 22.9 of 384, **6 %**, while 237 distinct experts per layer are
touched across 100 steps — so our value function is governed by the working set, not the single-step
union. The law looks right for its regime; we are not in it.

Two smaller notes. Your cold-start item — 48-56 % hit for the first 80-160 tokens, with imatrix
seeding proposed — we already do the equivalent, ranking the warm start from a coverage profile; our
residual ramp after seeding is 0.78 -> 0.93 over ~200 steps, if a data point helps. And our eviction is
age/(1+use count) rather than LFRU, measured at +8.0 % over LRU in two independent harnesses, which
sits alongside the Belady 1.068x bound cited above rather than against it.

Happy to run specific cells on the GB10 if any of this is worth pinning down.

AI assistance was used.
