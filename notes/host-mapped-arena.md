# Executing cold experts from host memory, and what that does to the miss path

**State: the compute path is validated and the split is measured. What remains is deferred promotion,
without which this silently destroys the hit rate.**

On GB10 the CPU and GPU share one DRAM pool, so a "device" allocation and mapped page-locked host
memory are the same physical memory at the same bandwidth. The H2D copy on the miss path is therefore
not a transfer between tiers; it is a DRAM-to-DRAM copy of bytes the kernel could already read.

Device attributes on this box: `PageableMemoryAccessUsesHostPageTables = 1` (ATS: the GPU uses the
CPU-side page tables), `PageableMemoryAccess = 1`, `CanUseHostPointerForRegisteredMem = 1`,
`HostRegisterSupported = 1`, `ConcurrentManagedAccess = 1`, `DirectManagedMemAccessFromHost = 0`. So
this is mapped page-locked access, not demand-paged managed migration. Nothing here uses
`cudaMallocManaged`, deliberately.

## What is measured

**Raw bandwidth, L2-free** (job 1020; 38-record working set, each touched once):

| arm | ms/record | GB/s |
| --- | --- | --- |
| device (`cudaMalloc`) | 0.061 | 227.3 |
| host-resident execution | 0.063 | 219.4 |
| H2D + device execution | 0.250 | 55.1 |

**The real CB3 kernel** (jobs 1025/1035/1040), 384-slot arena, bitwise identical to the device arena
at every shape. This is the 250-register kernel whose occupancy is register-limited, so it has few
spare warps to hide latency — the case where mapped memory was most likely to fail:

| shape | device | mapped host | ratio |
| --- | --- | --- | --- |
| T=1 | 0.711 ms | 0.655 | 0.921 |
| T=6 | 2.764 | 3.030 | 1.096 |
| T=6 high-distinct | 2.834 | 3.128 | 1.104 |
| T=24 | 90.975 | 92.215 | 1.014 |

**Record-major layout costs nothing on device** (job 1040, arm D): +0.0 / +0.7 / +0.5 / −1.1 % against
plane-major. So the mapped penalty above is the memory, not the layout.

**O_DIRECT straight into the final mapped slot** (job 1045), real 211.6 GB pack, random records,
O_DIRECT so nothing is page-cached:

| arm | ms/record | GB/s |
| --- | --- | --- |
| staging buffer + twelve H2D copies (today) | 3.249 | 4.24 |
| **O_DIRECT into the mapped slot** | **2.357** | **5.84** |

**−27.5 %, 0.892 ms per record**, same bytes to the kernel either way. This works only because packed
scales make `CB3_BYTES_PER_SLOT_PACKED` = 13,773,312 equal the pack's `payload_bytes` exactly, with the
record padded to 13,774,848 = 3363 × 4096 — so file offsets, transfer lengths and slot offsets are all
O_DIRECT-aligned and an expert needs **no transform at all** between disk and kernel. With unpacked
scales the slot is 14,454,784 and this is impossible. The packed-scale work turns out to be the
enabler for this, not merely a capacity win.

**The mixed hot/cold layer** (job 1065), T=6 top-6, ~34 distinct experts of which 3 cold (~9 %):

| arm | ms | |
| --- | --- | --- |
| A1 device plane-major, one call | 2.562 | today |
| A2 device record-major, one call | 2.588 | +1.0 % vs A1 |
| **B hot device phase + cold mapped phase + one reduce** | **2.696** | **+4.2 % vs A2** |

`B == A2` and `A2 == A1` **bitwise**, so the split reproduces the unsplit arithmetic exactly. The split
costs **+0.108 ms/layer** for 3 cold experts, i.e. +0.036 ms per cold expert.

### The whole accounting, per cold expert

| term | ms |
| --- | --- |
| fill saving, O_DIRECT into the mapped slot (job 1045) | **−0.892** |
| split overhead (job 1065) | +0.036 |
| promotion exposed after hiding (job 1070) | +0.102 |
| **net** | **−0.754** |

So a cold expert costs about 0.75 ms less than it does today, **including the measured cost of
restoring hot residency**. It is not a claim that cache semantics are preserved: no lifetime or
ownership logic exists yet, and that is the next thing to build.

**And the three terms come from three separate experiments, so this is a component model rather than a
measured miss path.** The interaction none of them contains is bandwidth contention with the NVMe reads
themselves. In production, four things share GB10's one memory fabric at the same time: NVMe DMA
writing into DRAM, the GPU reading mapped cold records, the GPU reading hot records, and the
DRAM-to-DRAM promotion. Job 1070 already shows the promotion cannot be fully hidden because it contends
with the kernels; the same argument applies with several GB/s of SSD traffic landing in the same pool.
The per-miss figure is not a tok/s claim, and this project has twice had a wall-clock result dissolve
under drift and token-count artefacts.

## Record-major addressing works (and the bug that made it look otherwise)

`_cb3v3_up_kernel` and `_cb3v3_down_kernel` take `RSTRIDE`: 0 keeps each plane's own slot stride, which
is the shipped plane-major arithmetic unchanged; `RSTRIDE = record bytes` gives all planes one shared
slot stride with each pointer pre-offset into the record. Verified bitwise equal to plane-major at
SLOTS = 1, 8, 64, and jobs 1040/1045/1065 all depend on it.

It did not work at first, and the cause was mine: two text replacements with `count=1` landed in
*different kernels* — the signature and constexpr block in `_cb3v3_up_kernel`, the body in v1's
`_cb3_up_kernel`, whose body carries the `offs_q`/`offs_c` suffixes the pattern included. So v3 took
the parameter, computed the strides and never used them. Fixed in `7201e236`; v1 reverted, v3 wired.

## What this does NOT license, and the piece that is missing

**Promotion is not optional.** Executing a miss from the mapped cold pool leaves that expert *only* in
the cold pool. Today's miss path also installs it in the device-resident cache. Skip that and the
expert is cold again on its next use, so the 0.9029 hit rate the arithmetic above multiplies by is
destroyed by the very change that arithmetic justifies. The design has to be:

    NVMe --O_DIRECT--> mapped cold record
                        |-- GPU computes the miss directly from it
                        `-- later, contiguous copy --> record-major device hot slot

It is **one contiguous ~13.77 MB copy** rather than twelve plane scatters, because both sides are
record-major. **Measured** (job 1070, an 8-layer chain with 3 cold experts per layer):

| arm | ms/layer | vs A | |
| --- | --- | --- | --- |
| A no promotion | 2.660 | — | lower bound, wrong semantics |
| B promotion synchronous after the cold phase | 3.371 | +26.7 % | upper bound |
| **C promotion on a side stream after the cold phase** | **2.966** | **+11.5 %** | |
| D promotion on a side stream *before* the cold phase | 2.907 | +9.3 % | |

**57 % hidden, not all of it** — a side stream exposes +0.102 ms per promoted record against +0.237 ms
synchronous. That is consistent with the copy being DRAM-to-DRAM and therefore competing for bandwidth
with kernels that are themselves bandwidth-bound, so there is nothing to hide *behind* in the usual
sense; a full 40-layer traversal of wall time does not help if the bytes still have to move through the
same pool.

**A prediction of mine failed here.** I pre-registered that issuing the promotion *after* the cold phase
would beat issuing it before, on contention grounds. D (before) measured 2.907 against C's 2.966 — 2 %
the other way. The margin is small and it is one measurement, so it does not invert the design, but the
ordering argument is unsupported and should not be repeated as if it were established.

The design nevertheless takes **after**, on ownership grounds rather than speed: with promotion issued
after the cold phase, the cold slot's compute lifetime is already finished when the copy starts, so the
source slot has one reader at a time and its state machine is simpler. Issuing before gives the cold
record two concurrent GPU readers — the cold CB3 kernel and the promotion copy — and recycling that slot
when either one finishes would corrupt the other. 2 % is not worth that. Revisit only if the end-to-end
gain makes it matter.

**The "we have 40 layers to hide the copy" model is dead.** Temporal room is not free bandwidth. The
copy moves bytes through the same pool as the kernels, so overlap buys only what spare bandwidth
exists — 57 %, here.

Job 975's 0.9–2.0 ms H2D still looks dominated by today's staging and scatter rather than by the
physical copy: one contiguous record copy costs 0.237 ms synchronous.

**A wholly host-mapped arena is not the design.** At the T=6 verify shape — the decode shape, so the
one that decides it — mapped execution costs +9.6 % on every access while the fill saving only reaches
the ~10 % that miss. Resident experts stay in device memory.

**The large-arena TLB question does not arise.** At T=6 top-6 there are at most 36 distinct routed
experts, so a cold pool of 64 slots is 882 MiB. Job 1050 already measured 5 GiB / 1.31 M pages at a
ratio of 1.078 — an order of magnitude more footprint than this design needs. Closed. (Huge pages would
only matter for the all-host arena, which the T=6 number rules out anyway.)

Superseded, kept so the record is not silently rewritten: an earlier version of this note argued from
the 3.5 % streaming figure that the whole arena could be host-allocated; the real kernel costs +9.6 %
at T=6 and that is withdrawn. And a "−0.063 ms/access" estimate for the split multiplied a whole-call
difference by a miss rate as if it were a per-expert cost; job 1065 measures the real thing at
+0.108 ms/layer and supersedes it.

## Operational note

Job 1055 hard-reset this box: a 60 GiB pinned host arena followed by a 60 GiB device arena on a
121.6 GiB machine. Pinned pages are unreclaimable and the box resets when MemAvailable goes negative.
The defect was the guard, not the sizing — an in-process loop waited for the previous arena to be
released and then continued anyway on timeout, which looks like a check and is only a delay. Nothing
was lost. Any sweep over large allocations must be one arena per **process**, with a pre-check that
aborts. That experiment is retired rather than fixed, since the cold pool is under 1 GiB.

## Engine wiring, and what the boring gate caught (job 1075)

The cold path is wired behind `DSV41_COLD_POOL=1`, default off: `ExpertStore.attach_cold_pool`,
a decode miss in `resolve()` fetched by O_DIRECT into a mapped slot instead of loaded into its hot
slot, `Model.moe()` running `moe_forward_cold_split` when any expert of the layer is cold, and
promotion issued on a side stream afterwards with `cold_reap()` retiring the landed ones at the next
layer boundary. The first gate is equality, on the graph-free decode path (`DSV41_FAST=0`) so a
failure cannot be confused with CUDA-graph capture.

**Everything semantic matched on the first run** — emitted token ids per prompt, `expert_misses`
(6,423 and 3,839), `expert_hit_rate`, `accept_len_mean`, `steps`, and the **resident key set: 2,504
keys, identical sha**. That last one is the P0 hazard, and it is clean: promotion does restore
residency, so the hit rate the design depends on survives the design.

**The gate now PASSES in full (job 1080).** Every field identical: token ids per prompt, misses
(6,423 / 3,839), hit rate (0.7019 / 0.7508), **nvme_gb (103.52 / 65.39)**, accept_len_mean, steps, and
the resident key set (2,504 keys, same sha) — with 10,262 experts read by O_DIRECT into mapped slots,
computed there and promoted back. `cold_reuses` 0, `cold_full` 0, `failed` 0. The cold path is
semantically indistinguishable from the baseline, so timing may now be measured.

It took two fixes to get there, both found by the gate rather than by reading the code.

**First, a metric bug that would have manufactured a fake win.** `nvme_gb` fell from
168.91 GB to 27.55 GB. The gap is 141.36 GB; 10,262 cold fetches at 13.775 MB is 141.36 GB, residual
0.00. The cold path reads exactly the same records and simply never added them to
`store.stats["bytes_read"]`. Had the first measurement been a timing run instead of an equality run,
that would have read as a 6x reduction in NVMe traffic and been completely false. Fixed: the cold
fetch now accounts its bytes and its read time into the store's own counters.

**Second, the 0.01 GB that was left turned out to be two problems.** 1,536 B per fetch x 10,262 =
0.016 GB, because the ordinary path counts the PADDED record (13,774,848) and the cold path counted the
payload (13,773,312). Chasing that surfaced the more serious half: the payload is 512-aligned but **not
4096-aligned**, so the O_DIRECT read worked only because this device's logical block size is 512 — on a
4 KiB logical-block device it would have failed outright. Reading the whole padded record fixes both;
the pad lands in the slot's own padding, which no kernel reads. Construction now asserts the slot
stride equals the pack's record stride.

Two implementation notes, one a correction of my own design note:

**Residency is shadowed, not withheld.** The note above says the hot mapping must not be published
until the copy lands. In practice `_lru_slot_for` publishes `lru[key] = hot_slot` eagerly, as the
engine already does for every ordinary miss, and pass 1 of `resolve()` consults the in-flight table
*before* `lru` — so the cold slot shadows the hot mapping until promotion lands. Equivalent given the
ordering and simpler than withholding, but it is shadowing and the note should say so.

**Promotion is twelve scatters here, not one copy.** The main arena is still plane-major, so
`ColdPool.promote` falls back to per-plane copies. That is deliberate: it lets the cold path be gated
on without converting the main arena first, and it means any promotion cost measured in this
configuration is an upper bound. The one-contiguous-copy form needs the hot arena record-major too.

Counters worth keeping from the run: `cold_reuses: 0` over 10,262 fetches, so no expert was wanted
again while its promotion was still in flight; and `cold_full: 0` with a 64-slot pool, so the pool
never ran dry even though a layer can want 36 distinct experts.

## The engine divergence: what it is not, and the one-position shift (jobs 1085-1125)

With `DSV41_COLD_POOL=1` the five-prompt gate diverges at p3, token 5, deterministically. Eliminated,
each by measurement rather than argument:

| hypothesis | test | result |
| --- | --- | --- |
| async ordering | `DSV41_COLD_SYNC=1` (1095 D) | diverges identically |
| cold-slot recycling | 256-slot pool (1095 C) | diverges identically |
| split kernel arithmetic | per-layer reference (1100/1105) | 2,746 layers, zero differences |
| uninitialised h/parts | `DSV41_COLD_ZERO=1` (1120) | diverges identically |
| eviction protection bypass | `_blocked_slots()` (in 1120) | diverges identically |

**Localisation (1115/1125).** First difference is trace 2178, layer 21: routes IDENTICAL, slots
different, and the ROUTED OUTPUT different. The layer after it then routes differently, which is the
downstream consequence. Decisively, **that layer has `cold 0` in both arms** -- no cold experts at
all, so it ran the ordinary `moe_fn` path in both. Identical code, identical route, different value
means the ARENA CONTENTS differ, not the split.

**The shape of the slot difference is a one-position shift.** Same 37 experts; 23 of them sit in the
slot the *previous* expert held in the baseline:

    expert   7: OFF 2631  ON  868      <- a slot new to this layer
    expert  13: OFF  831  ON 2631      <- expert 7's baseline slot
    expert  41: OFF  348  ON  831      <- expert 13's baseline slot
    expert  57: OFF   49  ON 2749      <- expert 22's baseline slot

That is an allocation sequence offset by one, not a corrupted mapping -- and it fits ON having made
one fewer allocation earlier (its miss count is 23,046 against the baseline's 23,302).

**What is still unexplained, and it is the important part.** A shifted-but-self-consistent assignment
should not change any value: every expert still occupies a slot of its own. The value DID change, at a
layer with no cold experts. So somewhere an expert's slot does not hold that expert's bytes, and the
shift is a symptom of the same cause rather than the cause itself.

**RESOLVED: the promotion never called `invalidate_scratch(hot_slot)`.** Both `CB3Cache.load_slot`
and `CB3ArenaV2.load_slot` do it first -- every write to a slot must invalidate that slot's cached
unpacked-FP4 copy -- and the promotion path did not. So a promoted slot left a stale FP4 scratch entry
behind, and the five-prompt equality gate now PASSES on all five prompts with every field identical:
token ids, misses (6,423 / 3,839 / 1,800 / 4,571 / 6,669), hit rates, nvme_gb, accept_len_mean, steps,
and the resident key set.

Two things about how this was found are worth keeping, because both were my errors.

**I dismissed this hypothesis on reasoning that was too narrow.** The FP4 scratch serves
`moe_forward_prefill`, and decode at P=36 never takes that path -- so I argued it could not explain a
decode-time divergence. But *prefill between prompts* does take it, and p3 is the fourth prompt: a
promotion during p2's decode left a stale scratch entry that p3's prefill then read. The review that
raised it called it "a real correctness bug regardless, not proven", which was the right calibration;
mine was worse.

**And the shift was a symptom, exactly as suspected, not the cause.** The one-position slot shift came
from ON taking a different trajectory once a value went wrong; with the fix the trace-line counts match
exactly (2,996 in both arms, where ON had been 2,916).

Two by-products of the hunt, both kept:

* `engine/test_promote_plane_major.py` gates the twelve-scatter promotion into a plane-major arena
  against `CB3Cache.load_slot` over 20 random pack records -- all twelve planes byte-identical. That
  branch had no gate at all, since the real-record test promotes record-major to record-major.
* `DSV41_COLD_RESIDCHECK` validates live slot contents against the pack. It reports `cold-inflight`
  mappings as failures, which is a false positive in the checker rather than a fault: an expert whose
  promotion is in flight has its bytes in the COLD slot by design. What matters is that it found **zero
  failures among settled `lru` and `transient` mappings**.

**A correction to an earlier claim in this note.** Job 1105's "bitwise over 2,746 in-engine layers"
is weaker than stated: its reference recompute ran immediately after the split and would have drawn
the same allocator blocks, so it could not have detected a garbage-dependent difference, and more
fundamentally it compares split against reference WITHIN one arm -- it can never see the two arms
being fed different inputs, which is exactly the situation that obtains here. The microbenchmark
bitwise results (1040, 1065, and the real-record test) do not depend on that and stand.
