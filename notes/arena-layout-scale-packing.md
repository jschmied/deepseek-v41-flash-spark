# The arena is 4.70 % bigger than the file it loads

2026-09-18. Arithmetic against `/home/jschmied/dsv41-cb3/experts-cb3-s3.bin.json` and
`tools/cb3_moe.py:50` — no measurement, so nothing here is contingent on a run.

| plane | on disk | in arena |
|---|---|---|
| `w1_lo` `w1_hi` `w1_cb` | 2,949,120 / 1,474,560 / 18,432 | identical |
| `w3_lo` `w3_hi` `w3_cb` | 2,949,120 / 1,474,560 / 18,432 | identical |
| `w2_lo` `w2_hi` `w2_cb` | 2,949,120 / 1,474,560 / 40,960 | identical |
| **`s1`** | **140,544** | **368,640** |
| **`s3`** | **140,544** | **368,640** |
| **`s2`** | **143,360** | **368,640** |
| | 13,773,312 + 1,536 pad = **13,774,848** | **14,454,784** |

Nine of twelve planes are byte-identical. **The entire 679,936 B/slot difference is the three scale
planes**, and it is not padding or alignment: the arena holds UE8M0 group scales one byte per group,
while the file holds them in `ue8m0-3bit-rowbase-v1` — one u8 row base plus 3 bits per group.

`tools/scale_codec.py` documents that codec as **exactly lossless on this checkpoint**: the
intra-row exponent range never exceeds 7 over all 149,422,080 rows of all 40 layers
(`scale-survey-20260913.txt`), and records that 2 bits would *not* be, failing on 0.2 % of layer 39.
So this is not a precision trade-off that was made and could be revisited. It is 4.70 % of every
slot spent expanding a format the engine already reads, writes and trusts on the way in.

## What packing the arena's scale planes would buy

1. **4.70 % more slots at the same GB.** At `ARENA_GB=86`: 5,949 → 6,228.
2. **The arena slot becomes byte-identical to the record.** `CB3Cache.load_slot` is today nine
   plane `copy_` calls plus three device-side `_unpack_scales`; it would become one contiguous
   13.77 MB pinned→device copy. That is per miss, and job 705 ran at hit 0.89–0.92, so misses are
   not rare.

## What it costs

The Triton kernels decode the scale inline instead of loading a byte. The granularity already
matches: `_cb3v2_block_dot` does `_split8(tl.load(s_ptr))`, eight scales at a time, and the codec
packs exactly eight groups per 24-bit little-endian word (value *i* at bit 3*i*). So the change is
`load 8 bytes → split` becoming `load base + 3 bytes → assemble → (w >> 3i) & 7 → + base`, against
a kernel that already carries the `_cb3_pack_quad` gather-and-`prmt` dance per quad.

## Gate before building

Job **715** prices it directly rather than by extrapolation: 82 → 86 GB is **+4.88 % slots**, which
is the step packing buys. Six reps per arm, verdict on reps 5–6 only — job 705's rep 3 was still
climbing (hit 0.8933 → 0.9192 → 0.9244), so a three-rep arm compares transients.

If the capacity step turns out to be worth little, the *load path* half still stands on its own and
gets re-gated on a miss-cost measurement instead. The two halves are separable.

## A second, smaller item in the same family

`w1_cb`/`w3_cb`/`w2_cb` store the 8-entry codebook as **8 bytes, one nibble per byte**; `_cbword`
(`cb3_moe.py:129`) loads `[BN, 8]` and packs it into a single int32, i.e. 4 bits per entry, 4 bytes.
The file has the same 2× expansion, so shrinking it is a change on both sides. Worth 38,912 B/slot
= **0.27 %** — noted, not led with.

## Related

- Arena capacity is the dominant decode lever right now: job 705 measured 68.8 → 86 GB (4,760 →
  5,949 slots, +25 %) as 3.78 → 5.17 tok/s (**+37 %**), superlinear because the hit rate climbs.
- [ring-lending-port-and-prefill-memory.md](ring-lending-port-and-prefill-memory.md) — the other two
  ways to get slots back, both of which compose with this one.
