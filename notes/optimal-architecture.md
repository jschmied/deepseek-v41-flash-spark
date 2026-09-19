# The architecture the measurements point to

Everything below is labelled **measured** (we ran it), **projected** (arithmetic over measured terms —
and today taught us twice that these overpromise), or **open** (not established).

The single most important number: we run at ~6.2 tok/s against a raw byte ceiling of ~27.6 tok/s under
LRU. **The gap is not bytes. It is the dependency chain.** Any architecture that only moves bytes around
more cleverly is optimising the wrong term — which is exactly what the host-mapped cold path proved by
coming out performance-neutral.

## 1. Format and storage — settled

**CB3, 3-bit codebook, PACKED scale planes, record-major, 4096-aligned.**

* packed scales: **+25 % capacity, bitwise identical** (measured, shipped). They are also what makes
  everything below possible: the packed slot is byte-identical to the on-disk record (13,773,312 B,
  padded to 13,774,848 = 3363 x 4096), so an expert needs **no transform between disk and kernel**.
* record-major: **~0 % cost on device** (measured, job 1040 arm D), and it is what lets one read land in
  one place.
* EXL3 for the main weights is **dead**: 2.9 bpw is 196 GiB for this model, only 3.8 % smaller than our
  CB3, still needs TP2 (measured against the published quant).
* sub-expert stripes are **dead**: exact FFN needs every stripe of a routed expert, so partial residency
  converts free hits into perpetual partial misses — 150-370 % worse at equal budget (measured). Revive
  only if `miss_rate(whole) > 1 - slots/distinct_experts`; today 0.20 vs 0.43.

## 2. Memory — one arena, allocated from CUDA VMM host-NUMA

**This is the change with the clearest case and it has not been built yet.**

    NVMe --O_DIRECT--> arena slot --> kernels read it in place, permanently

* a `cuMemCreate(HOST_NUMA)` + `cuMemMap` arena runs the **real CB3 kernel bitwise at every shape**, at
  **+4.5 % at the T=6 verify shape** and −7.6 % at T=1 (measured). The pinned pool costs +9.6 % at the
  same shape, so VMM is less than half the penalty.
* it accepts `preadv` as an O_DIRECT destination directly (measured), so the arena **is** the read
  destination. No staging buffer, no H2D, no promotion, no hot/cold split, no two-phase MoE, no
  promotion lifecycle.
* granularity is 2 MiB, so it is large-page backed — which retires the GPU TLB-reach question over a
  ~78 GB arena **by construction** rather than by a sweep (the 4 KiB sweep hard-reset the box).
* **projected:** the penalty is per ACCESS, the saved H2D (0.936-2.034 ms/expert, measured) is per MISS,
  so break-even is a **6-14 % miss rate**. Measured miss rates are 9.7 % at 86 GB and 30 % at 40 GB, so
  it wins in the regime we actually operate in and is break-even only at the largest arena.
* why this is not the cold path again: the cold path paid NVMe->pinned **and** pinned->device, merely
  reordered. Here the second movement **stops existing**. That is the distinction that made the
  difference between neutral and not.

What this deletes: the cold pool, the promotion state machine, `moe_forward_cold_split`, the staging
buffers and copy streams. Roughly a day of this week's work becomes dead code, and that is the right
outcome.

## 3. I/O — settled

**O_DIRECT, one record per read, `io_threads=2`.** The device saturates at **one** concurrent read
(aggregate flat ~4.87 GB/s from n=1 to n=96, per-read latency 2.8 ms x n — measured), so depth buys
nothing and costs linearly. 2 is measured best (**+9 % median, 3/3 paired**); the win is mostly read
latency but H2D concurrency moves too, so it is not a pure latency story.

## 4. Cache policy — settled, and the headroom is not here

**Global LRU with `age/(1 + use count)`** (+8.0 % over plain LRU, measured in two harnesses).

Measured negative, do not revisit: per-layer LRU (+3.9 %), per-layer marginal-gain quotas (+139 %),
phase-aware eviction protection (−0.0 to −0.4 %), two-hit doorkeeper admission (−5-7 % bytes but
**+16-29 % time**, because an O_DIRECT read costs 9.9x the copy it avoids — total bytes is not the
objective).

A perfect-frequency oracle is still **−52 %** against LRU at the same budget (measured). So real headroom
exists, but four independent attempts now land within a few percent of plain LRU. It is not reachable by
recency, admission, or partitioning.

## 5. Scheduling — where the remaining gain actually is

**Open, and the largest item.** Today **nothing overlaps**: `fastdecode::_layer_b` computes the routed
MoE *before* the shared expert, and graph B does not launch until the host resolve has finished. So a
layer's compute and its reads are strictly serial.

1. **Reorder `_layer_b`**: shared expert first — it needs no reads — then the routed MoE. This creates
   the overlap window at all. Nothing else in this list works without it.
2. **Resident-first split**: `moe_forward_v3_split` already exists and is **bitwise** (measured,
   `test_moe_split_bitwise.py`), so resident pairs can compute while this layer's misses are still
   arriving. Pair residency is 94-95 %, so most of the work does not depend on the reads it currently
   waits behind.
3. **Cross-layer transition prediction in the PREFETCH role** — the one open lever with a large ceiling.
   The corrected ceiling puts prediction at 1.2-50x what an async loader adds; recall 0.30 captures
   28-41 % of the oracle win; our existing transition table is at **30.5 % recall with nothing trained**.
   Run it through the corrected scheduler before spending any training compute.

Prediction in the *protection* role is **closed** (perfect oracle 0.4 pp).

## 6. Drafter

**CB3, not FP4** — 17 % faster kernel at the draft shape, 1.554 GiB freed, −5.7 % NVMe traffic, output
byte-identical (measured). No wall-clock gain was demonstrated, so it is `DSV41_DRAFT_CB3`, off. With a
VMM arena its memory is no longer a separate budget question, which removes the only reason to hesitate.

## Summary of the ordering

1. record-major main arena behind a flag, bitwise-gated (verified in isolation already)
2. back it with `engine/vmm_alloc.py`; delete staging, H2D, cold pool, promotion
3. equality gate on five prompts, then paired A/B/A/B/A/B — **no number before both**
4. reorder `_layer_b`, then resident-first split
5. transition-table prefetch through the corrected scheduler

Steps 1-3 remove a movement. Steps 4-5 attack the serialisation, which is where the 6.2-vs-27.6 gap
lives. Nothing in 1-3 helps if 4-5 never happen, and 4-5 are cheaper to try than they look because the
split is already written and already bitwise.
