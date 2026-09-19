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
costs **+0.108 ms/layer** for 3 cold experts, against 3 × 0.892 = 2.676 ms of fill saving: **−2.568 ms
per layer on the critical path**, or about 0.856 ms saved per cold expert.

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

Two things make the promotion cheap. It is **one contiguous ~13.77 MB copy** rather than twelve plane
scatters, because both sides are record-major. And it need not be on the critical path: the expert
cannot be needed at the same layer again until the next token or verify step, roughly a full 40-layer
traversal later, so it should be issued **after** the cold phase rather than overlapped with it — both
would contend for the same DRAM bandwidth. Job 975's 0.9–2.0 ms H2D looks dominated by today's
staging/scatter/stream machinery rather than by the physical copy, which the streaming numbers put at a
few tenths of a millisecond; that is the next thing to measure.

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
