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

One constraint on how any of this gets tested: `torch.cuda.change_current_allocator` must be called
before any CUDA allocation, so a device arena and a host-allocated arena cannot coexist in one process.
Every A/B here needs separate processes, interleaved.
