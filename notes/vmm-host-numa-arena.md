# The O_DIRECT destination can BE the arena: CUDA VMM host-NUMA on GB10

**Result: a `cuMemCreate(HOST_NUMA)` arena runs within 2.4 % of `cudaMalloc`, is 7 % faster than the
pinned pool we already ship, and accepts an O_DIRECT read directly into it.**

```
cuMemGetAllocationGranularity(HOST_NUMA)   2,097,152 (2 MiB)
mapped, 4096-aligned True, CPU access granted True
O_DIRECT preadv into the range: 13,774,848 of 13,774,848 -> OK

cudaMalloc (device)                0.056 ms/record   245.6 GB/s
cudaMallocHost (today's pool)      0.062 ms/record   223.1 GB/s
cuMemCreate HOST_NUMA + cuMemMap   0.057 ms/record   240.0 GB/s
                                   VMM/device 1.024   VMM/pinned 0.930
```

## Why this matters when the cold path did not

The host-mapped cold path came out performance-neutral because it **relocated** a copy instead of
removing one: both it and the baseline move NVMe -> pinned *and* pinned -> device for every miss.

    today   NVMe -> pinned staging -> device arena      two movements
    VMM     NVMe -> arena, computed in place            one

With a VMM host-NUMA arena there is no second residency to establish, so the `h2d` term does not move
elsewhere -- it stops existing. In the engine that term is 0.9-2.0 ms per expert (job 975) over 23,302
misses on the five-prompt suite.

## What VMM does and does not do

It separates physical allocation from virtual mapping: `cuMemCreate` makes a physical handle,
`cuMemAddressReserve` + `cuMemMap` place it, `cuMemSetAccess` grants CPU and GPU access to the same
pages. It does **not** re-tag host pages as device pages -- the location is a property of the physical
allocation, not of the mapping -- so this is not a way to convert an existing pinned buffer. The value
is that the arena can be created host-located from the outset and still be read by the GPU at close to
device speed.

## A risk this retires

Granularity is **2 MiB**, so the allocation is large-page backed. GPU TLB reach over a ~78 GB arena was
the open concern that the 4 KiB-paged pinned pool could not be tested for safely -- job 1055 hard-reset
the box attempting it -- and 2 MiB pages give 512x the reach per entry. The concern is addressed by
construction rather than by a sweep.

## What is NOT established

This is a streaming sum, not the CB3 gather kernel, and that distinction has already mattered once: the
pinned pool measured 96.5 % of device on a stream and 0.92-1.10x on the real 250-register kernel. The
next gate is the same one job 1025/1040 applied -- the real kernel, real shapes, bitwise -- on a
VMM-backed arena. Nothing should be rebuilt before that passes.

And the engine change is larger than the cold pool's was: the arena is allocated once at startup (decode
graphs bake base pointers), so a VMM arena replaces `CB3ArenaV2`'s storage wholesale rather than sitting
beside it. `torch.frombuffer` over the mapped range gives tensors the kernels accept -- the probe drives
the kernel through exactly that -- but every arena helper that assumes `torch.empty(device=...)` has to
be routed through the new allocation.
