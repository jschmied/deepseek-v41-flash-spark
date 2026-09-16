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

## What this closes and what it leaves

- Closed here: the engine-footprint explanation for the read penalty (refuted by its own bare stage).
- Bounded here: every compute/IO overlap project, at 3.8 % combined. NOT every scheduler toggle --
  see the correction in section 1; an NVMe/H2D overlap lever is outside that arithmetic.
- Live: read depth. The oracle needs information no predictor has, and the transition table is
  negative on this instrument (job 265: every arm below its oracle, fetch precision 2.8-3.7 %).
