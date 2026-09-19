# On GB10 the H2D copy is pure waste, and my first probe said the opposite

## The result

Streaming-sum over CB3-record-sized buffers, working set 38 records (0.49 GiB) so nothing is
L2-resident, each record touched once per pass:

| arm | ms/record | GB/s |
| --- | --- | --- |
| device (`cudaMalloc`), L2-free | 0.061 | 227.3 |
| **host-resident execution, no H2D** | **0.063** | **219.4** |
| H2D + device execution (today's miss path) | 0.250 | 55.1 |

Host-resident execution runs at **96.5 %** of device speed and beats today's miss path by **-74.9 %**,
saving 0.187 ms per record. Two capability facts that make it possible: `cudaHostGetDevicePointer`
returns rc=0 with the *same* address, so host memory is directly device-addressable, and Triton accepts
a pinned CPU tensor as a kernel argument with no launcher complaint.

The physical reading is the stronger statement. 227 vs 219 GB/s is one number twice: on GB10 the CPU
and GPU share one DRAM pool, so "device" and "host" memory are the same memory at the same bandwidth.
The H2D is therefore not a transfer between tiers at all -- it is a copy from DRAM to DRAM whose only
effect is to move bytes the kernel could already read. 0.250 ms to copy-then-read against 0.063 ms to
read in place.

## My first probe (job 1015) concluded the exact opposite, twice over

It reported host streaming at **14.7 %** of device and printed "GATE C: kill". Both halves were wrong.

**Confounded baseline.** GB10's L2 is 24.0 MiB and one CB3 record is 13.1 MiB, so a single record read
30 times sits entirely in L2. The device arm was measuring cache at 784 GB/s while the host arm
measured DRAM. Against the real DRAM figure of 227 GB/s the host path is 96.5 %, not 14.7 %.

**Mis-specified gate.** It compared host-kernel speed against device-kernel speed. That is not the
decision variable: on the miss path the alternatives are *host execution with no copy* versus *H2D plus
device execution*, and 1015's own numbers already put the kernel penalty at +0.101 ms against a
0.9-2.0 ms copy (job 975). The arithmetic that mattered was visible in the output and the gate pointed
away from it.

Both errors pushed the same direction, which is why the wrong answer looked clean.

## What this does and does not license

It does **not** yet license changing the arena. This is a streaming sum, not the CB3 kernel: the real
one gathers across slots with a different access pattern, decodes 3-bit groups, and has 250 registers
at BM=16 with zero spills (job 850). Its behaviour on mapped memory has to be measured, not assumed.

~~It also reframes the hot/cold split the review proposed. At a 3.5 % penalty there is no strong
reason to keep resident experts in device memory either -- the arena itself could be host-allocated.~~
**Withdrawn, falsified by jobs 1040 + 1045 below.** The 3.5 % came from a streaming sum; the real
kernel costs +8.9 % at the T=6 verify shape, and that penalty is paid on EVERY access while the read
saving only reaches the ~10 % that miss. The review's original hot/cold split was right and my
extrapolation was wrong.

~~One constraint: `change_current_allocator` must precede any CUDA allocation, so device and host
arenas cannot coexist in one process.~~ **Withdrawn.** Triton takes pinned CPU tensors directly, so a
device plane-major arena and a pinned host arena run interleaved in ONE process -- jobs 1025/1035/1040
all do. The allocator only matters if you want PyTorch to treat host storage as an ordinary CUDA
tensor, which none of this needs.

## The real CB3 kernel on mapped memory (jobs 1025, 1035)

| shape | device arena | mapped host arena | t_mapped / t_device |
| --- | --- | --- | --- |
| T=1, top-6 | 0.707 ms | 0.664 ms | **0.939** |
| T=6, top-6 | 2.758 ms | 3.041 ms | **1.103** |
| miss path (H2D of the slots used, then device exec) | 2.576 / 13.832 ms | -- | -- |

Bitwise identical to the device arena at both shapes (job 1035), and the host arm beats the miss path
by 74-78 %. So the streaming result does transfer to the gather: this is the 250-register kernel, whose
occupancy is register-limited and which therefore has few spare warps to hide added latency. It passes
a 10 % bar at T=1 and sits on it at T=6.

`cudaDeviceGetAttribute` on this box: `PageableMemoryAccessUsesHostPageTables = 1` (ATS: the GPU uses
the CPU-side page tables), `PageableMemoryAccess = 1`, `CanUseHostPointerForRegisteredMem = 1`,
`HostRegisterSupported = 1`, `ConcurrentManagedAccess = 1`, `DirectManagedMemAccessFromHost = 0`. So
this is mapped page-locked access, not demand-paged managed migration -- and nothing here uses
`cudaMallocManaged`. TLB reach over an 80 GB mapped arena at 4 KiB pages remains an open risk that
these 384-slot (5.17 GiB) measurements do not probe; huge-page-backed registered memory is the lever
if it shows up.

## Record-major: my addressing is wrong, and it is mine, not the memory's

Record-major is the stronger design, because host-allocating today's arena removes the H2D but keeps
the scatter: `load_slot` explodes one contiguous record into twelve plane-major tensors. The kernels
now take `RSTRIDE` (0 = shipped plane-major strides, unchanged; record bytes = one shared slot stride
with each pointer pre-offset into the record).

It does not work yet, and the failure is isolated:

- base offsets are right -- with SLOTS=1 (slot 0 only, so `slot * stride` is 0) record-major is
  **bitwise equal** to plane-major;
- the slot stride is wrong -- with SLOTS=8 it produces NaN, in BOTH kernels independently;
- the record fill is right -- `v[s, OFF:OFF+n] == plane[s]` for every slot and every plane;
- the address model is right -- `as_strided((S, rows, row), (RECORD, row, 1), OFF)` equals each plane
  byte for byte, for all twelve;
- every plane offset is 16-byte aligned, and RECORD equals `CB3_BYTES_PER_SLOT`;
- it is NOT a Triton specialization-cache artifact -- compiling the record-major arm FIRST in a fresh
  process still fails, so `RSTRIDE` is being honoured;
- and it is NOT a regression in the shipped path -- with `RSTRIDE` defaulted, routing to slot 0 and to
  slot 7 still produce different outputs, so the plane-major arithmetic is intact and job 1035's
  bitwise equality was against a correct reference.

I intended to hold the kernel commit local until `RSTRIDE != 0` was verified, and then pushed a notes
commit on top of it -- which pushes the ancestor too. So it IS pushed (d0cc981). That is safe in
substance rather than by plan: the `RSTRIDE = 0` default is verified (slot resolution intact, bitwise
equality in job 1035) and **no call site passes `RSTRIDE`**, so the broken branch is unreachable from
anything that runs. But the sequencing was wrong, and the lesson is that "commit locally, push later"
does not survive committing anything else on top of it: stage the unverified change last, or keep it
on its own branch.


## Step 2: O_DIRECT straight into the final mapped slot (job 1045)

Possible only because of packed scales: `CB3_BYTES_PER_SLOT_PACKED` = 13,773,312 equals the pack's
`payload_bytes` exactly, and the on-disk record is padded to 13,774,848 = 3363 x 4096. So file offsets,
transfer lengths and arena slot offsets are all 4096-aligned and an expert needs **no transform at all**
between disk and kernel. With unpacked scales the slot is 14,454,784 and this is impossible.

Real 211.6 GB pack, randomly chosen records, O_DIRECT so nothing is page-cached:

| arm | ms/record | GB/s |
| --- | --- | --- |
| A staging buffer + twelve H2D copies (today) | 3.249 | 4.24 |
| **B O_DIRECT into the mapped slot** | **2.357** | **5.84** |

**-27.5 %, 0.892 ms saved per record**, and the same record fetched both ways drives the kernel to
bitwise identical output.

## Putting steps 1 and 2 together: the split is required, a wholly host-mapped arena LOSES

The kernel penalty is per ACCESS; the read saving is per MISS. At hit rate 0.9029 (job 1010):

| | T=6 verify | T=1 draft |
| --- | --- | --- |
| host kernel penalty per access | +0.246 ms | -0.058 ms |
| whole arena host-mapped | **+0.159 ms/access — LOSS** | -0.145 ms/access |
| hot/cold split | **-0.063 ms/access — win** | -0.092 ms/access |

Per miss the whole path goes 6.013 -> 5.367 ms, **-10.7 %**.

So resident experts must stay in device memory and only cold ones execute in place from the read
destination -- exactly what the review proposed and the opposite of the extrapolation struck out above.
T=6 is the decode verify block, so it is the shape that decides this, and it is the one where a wholly
mapped arena loses.

The engine already has the machinery: `moe_forward_v3_split` runs the MoE in phases over one routing
for the resident-first split, which is the shape a device-resident/host-cold split needs.


## TLB reach: flat to 30 GiB, but the largest point is still 2.6x short of the real arena (job 1050)

Record-major host vs record-major device, T=6 verify shape, one arena at a time, slots drawn over the
whole arena so a bigger arena means a wider translation footprint:

| arena | slots | 4 KiB pages | host | device | ratio |
| --- | --- | --- | --- | --- | --- |
| 5 GiB | 371 | 1.31 M | 2.982 ms | 2.765 ms | 1.078 |
| 15 GiB | 1,114 | 3.93 M | 3.079 | 2.796 | 1.101 |
| 30 GiB | 2,228 | 7.86 M | 3.081 | 2.817 | 1.094 |

**Ratio 1.078 -> 1.094 over a 6x size increase, +1.4 %.** TLB reach is not biting at this access
pattern. The device arm also rises slightly (2.765 -> 2.817, +1.9 %), so both layouts pay a small
locality cost with size and the ratio stays the decision variable -- the third pre-registered branch.

**What this does not cover.** The 50 GiB arm was skipped: the guard requires MemAvailable >= size + 20
GiB and the previous arena had only released back to 60 GiB. So the largest point measured is 7.86 M
pages against roughly 21 M for a 78 GB arena -- 2.6x short. The trend over the range measured is flat
and slightly *falling* at the top, which is evidence against a translation wall rather than for one,
but it is extrapolation and the sweep should be finished with tighter memory hygiene (free and settle
before the availability check) before the split is built on it.


## Job 1055 hard-reset the box, and the sweep is capped because of it

The 60 GiB arm crashed the machine. It allocated a 60 GiB **pinned** host arena and then a 60 GiB
device arena on a 121.6 GiB box; pinned pages are unreclaimable and this box resets when MemAvailable
goes negative. The defect was mine and specifically in the guard: an in-process loop waited up to 40 s
for the previous arena to be released and then **continued anyway** on timeout, which looks like a
check and is only a delay. Nothing was lost -- both repos were clean and pushed, the 211.6 GB pack and
the drafter download intact.

The sweep is now one arena per PROCESS (release is the OS's job at exit), with a pre-check that aborts
rather than proceeds, and a cap at 40 % of MemTotal. That cap stops the sweep at **45 GiB = 11.8 M
pages**, against roughly 21 M for a 78 GB arena. So the largest point is still 1.8x short and the
conclusion rests on the trend, not on reaching the real size: it cannot be closed safely on this box by
this method. If the trend matters more than that, the way to settle it is huge-page-backed registered
memory, which reduces the page count instead of raising the footprint.
