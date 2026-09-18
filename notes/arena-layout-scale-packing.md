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

---

## Both gates passed (2026-09-18)

**Losslessness on the arena's own data — job 790.** 87 real resident scale planes packed and
unpacked with `scale_codec`: **0 bad, EXACT**. This is stronger than the manifest's guarantee: the
file round-trips by construction because it is built with that codec, whereas these are the planes as
they sit in the arena after `load_slot`.

**Kernel cost — job 810**, the same arithmetic as a real Triton kernel rather than a torch stand-in:

```
rows=2304 groups=160 (lanes padded to 256)   intra-row range 7
plain 160 B/row   packed 61 B/row   (61.9 % smaller)
BITWISE IDENTICAL   max|d| 0.000e+00
IN TRITON: plain 6.2 us   packed 6.2 us   +0.0 us (+0.3 %)
```

**The unpack is free.** Job 790's torch stand-in had said +281 %; that was an artifact of torch
running reshape/shift/gather as separate kernels with materialised intermediates, where Triton fuses
the extract into registers around a load the kernel already performs. The denominator that 790 lacked
turns out not to matter — the added cost is +0.0 us absolute.

Three attempts failed before this one, all my errors and none of them measuring anything: synthetic
scales with intra-row range 63 (the codec needs <= 7); a `@jit` kernel in a heredoc (Triton reads the
defining source file); `tl.arange(0, 160)` (must be a power of two). Recorded because each cost a
queue slot.

### Verdict: build it

### Corrections from review, 2026-09-18

**The capacity arithmetic was wrong, in our favour.** 679,936 / 14,454,784 = 4.704 % is the *saving
as a fraction of the old slot*; the *capacity gain* is `old/new - 1` = 14,454,784 / 13,774,848 - 1 =
**4.936 %**. At 86 GB that is **5,949 -> 6,243 slots**, not 6,228.

**"The slot becomes byte-identical to the record, so a miss is one contiguous copy" is not true of
this change.** The record is contiguous; the arena is *twelve independent plane-major tensors*
(`CB3Arena.__init__`), and `CB3Cache.load_slot` copies into each. Packing the scale planes changes a
miss from *nine plane copies + three scale-unpack kernels* to *twelve plane copies and no unpack
kernels* — it does not produce one H2D. That needs record-major backing storage and per-slot strides
in every kernel. **Two separate changes, and they must not be one patch:**

| | change | gain | blast radius |
|---|---|---|---|
| **A** | packed scales in the existing plane-major arena | +4.94 % capacity, scale unpack removed | the three scale load sites |
| **B** | record-major arena | one H2D per miss | every kernel's slot stride |

A stands on capacity alone. B is a later, separate optimisation with its own benchmark.

**"The unpack is free" overstates job 810.** 6.2 us vs 6.2 us gates the *extraction arithmetic in
isolation*. It does not capture register pressure or occupancy inside `_cb3v3_up_kernel` /
`_cb3v3_down_kernel`, which are already large — inlining a base + 3-bit extract can cross a register
threshold even when the operation is free standing alone. The correct claim is: **scale extraction
has no measurable standalone Triton cost; the integrated MoE cost is not yet gated.** Build it, then
benchmark real routes (T=6 at realistic and high distinct-expert counts, BM 16/32/64 as actually
selected, up and down) and record compiled register counts where available. Do not book the gain
first.

Expected after A: **+4.94 % slots** (5,949 -> 6,243 at 86 GB), worth ~+16 % decode on job 715's
measured slope *if* the integration gate passes.

---

## `DSV41_CB3_SCRATCH_SLOTS=32` is NOT a free 6.62 GB — withdrawn

Job 750 measured that scratch 32 vs 384 frees 6.62 GB of resident memory with the transient peak
unchanged (8.38 vs 8.37 GB). I reported that as 6.62 GB the arena can have. **That does not follow,
and the code says why.**

`moe_forward_prefill` (`cb3_moe.py:1019`) takes the layer-scoped unpack cache only when
`scratch.slots >= n`, where `n` is the chunk's distinct expert count. A layer has ~362 distinct
experts and every chunk touches almost all of them, so at 32 slots that test is **false** and it
falls back to re-unpacking in batches of 32 (`:1050`). The comment at `cb3_moe.py:418` is explicit
about what that costs, and it is the reason the cache exists:

> 1512 `_unpack_into` calls x <=32 experts = up to 48,384 expert-unpacks where 21 x 362 = 7,600 are
> needed, a **6.4x redundancy**. At 33.25 MB per unpack that is **1.61 TB**, and
> `_cb3_unpack_kernel` was the largest single GPU consumer in the profile at **7.44 s of a 40.3 s
> GPU-busy prefill (18.5 %)**.

So scratch=32 reinstates precisely the problem `SCRATCH_SLOTS` was built to solve. Job 750 measured
**memory** and never measured **prefill time**, so it cannot price the trade either way.

**This affects runs already quoted.** Jobs 755, 760, 820 and 825 all ran `SCRATCH_SLOTS=32`, so
their absolute prefill walls (755: 76.6 s; 760: 52.3 s at 26,400 tokens) are pessimistic. Comparisons
*within* those jobs hold — both arms carried the same setting — but the absolute numbers should not
be quoted as this engine's prefill speed. The 86 GB arena result also depends on the 6.62 GB, so
whether 86 GB survives at a larger scratch is an open question, not a settled one.

Job 830 sweeps 32/128/256/384 at fixed arena and fixed prompt, reporting prefill wall, unpack GPU
time and unpacked-expert count. A middle value may keep most of the cache win and most of the memory.

---

## Register pressure: the integration gate the review asked for (2026-09-18)

**Job 850 — the live decode kernels as they stand.** The path is `fastdecode.py:450` ->
`moe_v3_phase` -> `_cb3v3_up/down_kernel` -> `_cb3v3_block_dot`, whose scales come through
`_split16`/`_split8` (16 and 8 groups = exactly whole 24-bit packed words, so no sub-word
addressing is needed there; `_cb3_quad_dot`'s `[BN, 4]` would need it and is not live).

```
_cb3v3_up_kernel  : n_regs=250  n_spills=0  shared=25088  warps=4
_cb3v3_down_kernel: n_regs=128  n_spills=4  shared=12800  warps=4
moe_forward_v3 T=6: 1908.5 us     T=24: 5555.3 us
```

**Five registers of headroom on the up kernel, and the down kernel already spills.** The review was
right to refuse "the unpack is free" on the strength of job 810's isolated 6.2 us. It also supplies
the denominator job 790 lacked: 810's delta was +0.0 us against a ~1,900 us MoE forward.

**Job 860 — does packed decode cost registers in this consumption shape?** Same `_split8` the live
kernel uses, two candidate shapes:

| variant | n_regs | n_spills | shared | time |
|---|---|---|---|---|
| unpacked | 40 | 0 | 2048 | 7.0 us |
| packed, base re-loaded per block | 40 | 0 | 256 | 8.3 us |
| packed, base hoisted per row | 40 | 0 | 256 | 8.2 us |

**Zero register cost, and shared memory falls 2048 -> 256** because the tile is smaller. Hoisting
the row base makes no difference, so the simpler per-block form is fine.

**Scope.** This probe is a 40-register kernel and the real one is 250; register allocation is not
linear, so this removes the *shape* objection without predicting the real count. The decisive number
is `_cb3v3_up_kernel`'s own `n_regs` after the change, measured the same way.

### Four Triton authoring failures preceded these two results

795 used test data the codec cannot encode (intra-row range 63 against a 3-bit format); 800 defined
`@jit` inside a heredoc, which Triton rejects because it reads the defining source file; 805 used
`tl.arange(0, 160)`, which must be a power of two; 855 referenced a module global from inside `@jit`.
Each cost a queue slot and none measured anything. 860 was **compile-checked locally before being
queued**, which caught a fifth (`s[:, 0]` scalar indexing is unsupported -- the real code uses
`_split8`, which the probe now imports rather than reimplements). That check is the rule from here.
