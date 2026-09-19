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

It also reframes the hot/cold split the review proposed. At a 3.5 % penalty there is no strong reason
to keep *resident* experts in device memory either -- which, if the CB3 kernel agrees, means the arena
itself could be host-allocated and the entire H2D path, its staging buffers, copy streams and copy
events deleted rather than merely bypassed for cold experts. That is a much larger change than the
review suggested and the measurement points at it.

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

The kernel commit is therefore held LOCAL, unpushed: its default branch is verified, its `RSTRIDE != 0`
branch is not.
