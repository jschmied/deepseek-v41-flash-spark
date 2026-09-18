# 2026-09-18: arena capacity, prefill memory, and the fusion verdict

Every number here is from the queue logs named in each line. Withdrawn claims are kept, marked,
because three of them were mine and survived several hours before being caught.

## 1. Arena capacity is the dominant decode lever — but 92 GB is not shippable

| arena | slots | decode (P0, steady) | 26.4k prefill floor |
|---|---|---|---|
| 68.8 GB (auto) | 4,760 | 3.78 | — |
| 78 GB | 5,396 | 4.54 | — |
| 82 GB | 5,672 | 4.94 | — |
| **86 GB** | **5,949** | **5.61–5.88** | **9.9 GiB SAFE** |
| 92 GB | 6,364 | 6.45 (9-token prompt) | **4.0 GiB FAILS** |

Job 715 gives the local slope: 82→86 is +4.88 % slots for **+16.2 %** decode, superlinear because the
hit rate climbs with it. Job 725 reported 92 GB / 6.45 tok/s as "the max" — that was a **nine-token
prompt**. Job 755 gated each rung on a 26,400-token prefill first and 92 GB took the box to 4.0 GiB,
under the watchdog's 8 GiB floor and exactly job 150's failure mode. **86 GB is the max that survives
a real request.** The limit is physical, not the pre-flight: 92 GB passed the refusal check and died.

## 2. The arena is 4.70 % larger than the file it loads

Nine of twelve planes are byte-identical between the CB3 record and the arena slot. The entire
679,936 B/slot difference is the three scale planes: the file stores `ue8m0-3bit-rowbase-v1`
(documented exactly lossless on this checkpoint), the arena expands them to one byte per group.
Packing them would give **+4.70 % slots** — the step job 715 prices at **+16 %** — and make the slot
byte-identical to the record, collapsing a miss from nine plane copies plus three device-side
`_unpack_scales` into one contiguous 13.77 MB copy. Details: `arena-layout-scale-packing.md`.

## 3. The two memory knobs, one real and one accounting

`DSV41_CB3_SCRATCH_SLOTS` defaults to **384 in `tools/cb3_moe.py:439` and to 0 in the sizer and the
pre-flight** (`v41_engine.py:433,501`), and the var is not in `.env`. So the engine sizes the arena as
if the CB3 unpack scratch were free and then allocates 7.22 GB of it lazily on the first prefill.
Setting it to 32 frees **6.62 GB of resident memory with the transient peak unchanged** (job 750:
8.38 vs 8.37 GB). That is real arena.

`DSV41_PREFILL_CHUNK` is the opposite: lowering it shrinks the sizer's `MAX_CHUNK * 5e6` term without
shrinking real usage, which is how 725 "reached" 92 GB.

## 4. Fusion: quality-neutral, a prefill win, decode-neutral

Job 780, teacher-forced so both configs score the same tokens:

| arm | mean NLL | top-1 |
|---|---|---|
| eagerA | 0.315920 | 618/672 |
| eagerB | 0.315920 | 618/672 |
| fused+gather | 0.318577 | 618/672 |

Yardstick eagerA→eagerB: dNLL +0.000000, 672/672 identical positions. Verdict eagerA→fused:
**+0.002657 ± 0.016215** (95 % paired, straddles zero), **top-1 unchanged at every position**. Scope:
672 paired positions, which is what the engine returns logits for. Prefill wall −32 % (76.6→52.3 s
and 69.5→47.1 s, measured twice); transient −27 % (7.10→5.18 GB). **Closes the paired-NLL item.**

## 5. Three withdrawn claims, and the instrument that caused two of them

- ~~"The prefill reserve is 4.7× too big / flat in chunk size."~~ Withdrawn (job 750). I measured host
  `MemAvailable`, which is blind here: torch's caching allocator does not return freed blocks to the
  OS, so the host high-water reports the allocator's **lifetime** maximum, not the per-chunk
  transient. `max_memory_allocated() - memory_allocated()` shows 3.08 GB at chunk 1024 and 8.38 GB at
  4096 — linear, right shape, over-reserved by 2.4× not 4.7×.
- ~~"The ~10.6 GB fixed cost is `Caches` sized by `max_seq`."~~ Withdrawn: computed from the real
  config it is **0.29 GB**, off by 36×.
- ~~"Gather fusion costs 13 % of decode."~~ Withdrawn (job 785). Five prompts: misses split 3 worse /
  2 better and ms/step favours fusion on **4 of 5**. The 13 % was P0 alone — the one prompt 760, 765
  and 770 all used.

## 6. Scoping note that applies to every throughput figure above

Jobs 705–770 measured **P0 only** ("Explain how an NVMe controller schedules writes."). Job 785 shows
P0 is at the pessimistic end: across five prompts the engine runs **3.57–13.02 tok/s** with accept
1.9–5.28, and P0 sits near the bottom at 2.33. Arena *comparisons* should survive; the absolute
numbers are P0-scoped and are labelled so from here.

## 7. Process failures, both fixed structurally

- **Two runners ran concurrently** (my second start), so arms in 730/735/740 measured memory against
  the other runner's resident engine — 50 GB MemAvailable instead of ~110, and the sizer silently
  **capped their arenas** below the arm's label. An `flock` in a wrapper did not help: `watchdog.sh`
  execs `bash runner.sh` directly and walked past it. The lock now re-execs inside `runner.sh` itself,
  with an absolute path (`flock` uses `execvp`, so a relative `$0` silently never armed). Verified.
- **I edited `runner.sh` while a runner was executing it**, and bash's incremental parsing killed the
  running copy at a shifted offset. Same hazard as the job-file rule, never extended to the runner.
- Arms now assert `n_slots` matches the arm's label and **skip** rather than run at whatever memory
  happens to be free. That assert is the only reason the capped arms above are not in the record.
