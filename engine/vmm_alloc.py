"""CUDA VMM host-NUMA allocations: memory the GPU reads at near-device speed that O_DIRECT can fill.

Why this exists. Every expert miss today pays two movements -- NVMe into pinned memory, then pinned
into the device arena -- and the host-mapped cold path was performance-neutral because it relocated the
second one instead of removing it. A host-NUMA allocation made through CUDA's VMM can be BOTH the
O_DIRECT destination and the arena the kernels read, so the second movement does not exist. Measured on
GB10: 1.024x cudaMalloc latency, 0.930x the pinned pool, preadv() accepted, 2 MiB granularity (so
large-page backed, which is also what makes TLB reach over a large arena a non-issue).

What VMM does NOT do: re-tag host pages as device pages. The location is a property of the physical
allocation, not of the mapping, so this cannot convert an existing pinned buffer -- the allocation has
to be created this way from the start.
"""
from __future__ import annotations

import ctypes

import torch

CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_LOCATION_TYPE_HOST_NUMA = 3
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0


class _Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _Flags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class _Prop(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int),
                ("location", _Loc), ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", _Flags)]


class _Access(ctypes.Structure):
    _fields_ = [("location", _Loc), ("flags", ctypes.c_int)]


_cu = None


def _lib():
    global _cu
    if _cu is None:
        torch.cuda.init()                     # a context must exist before any VMM call
        _cu = ctypes.CDLL("libcuda.so")
    return _cu


def _chk(rc: int, what: str) -> None:
    if rc != 0:
        s = ctypes.c_char_p()
        _lib().cuGetErrorString(rc, ctypes.byref(s))
        raise RuntimeError(f"{what}: rc={rc} {s.value.decode() if s.value else ''}")


class HostNumaAlloc:
    """One host-NUMA physical allocation, mapped for both the GPU and the CPU.

    `tensor` is a uint8 view the Triton kernels accept directly (they take its data_ptr; it reports
    is_cuda=False, which is a torch bookkeeping fact rather than a statement about who can read it).
    `memoryview()` is the same bytes for preadv().
    """

    def __init__(self, nbytes: int, device_ordinal: int = 0, numa_node: int = 0):
        cu = _lib()
        prop = _Prop()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA
        prop.location.id = numa_node
        gran = ctypes.c_size_t()
        _chk(cu.cuMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop),
                                             CU_MEM_ALLOC_GRANULARITY_MINIMUM),
             "cuMemGetAllocationGranularity")
        self.granularity = gran.value
        size = (nbytes + self.granularity - 1) // self.granularity * self.granularity
        self.size = size
        self.nbytes = nbytes
        h = ctypes.c_ulonglong()
        _chk(cu.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(size), ctypes.byref(prop), 0),
             "cuMemCreate")
        self._handle = h
        p = ctypes.c_void_p()
        _chk(cu.cuMemAddressReserve(ctypes.byref(p), ctypes.c_size_t(size),
                                    ctypes.c_size_t(self.granularity), ctypes.c_void_p(0), 0),
             "cuMemAddressReserve")
        self._ptr = p
        _chk(cu.cuMemMap(p, ctypes.c_size_t(size), ctypes.c_size_t(0), h, 0), "cuMemMap")
        d = (_Access * 2)()
        d[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE
        d[0].location.id = device_ordinal
        d[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        d[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA
        d[1].location.id = numa_node
        d[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _chk(cu.cuMemSetAccess(p, ctypes.c_size_t(size), d, ctypes.c_size_t(2)), "cuMemSetAccess")
        self._arr = (ctypes.c_uint8 * nbytes).from_address(p.value)
        self.tensor = torch.frombuffer(memoryview(self._arr), dtype=torch.uint8)
        if self.tensor.data_ptr() != p.value:
            raise RuntimeError("the torch view does not alias the mapped range")
        if p.value % 4096:
            raise RuntimeError(f"mapped at 0x{p.value:x}, not 4096-aligned; O_DIRECT needs it")

    def memoryview(self):
        return memoryview(self._arr)

    def free(self) -> None:
        cu = _lib()
        self.tensor = None
        self._arr = None
        cu.cuMemUnmap(self._ptr, ctypes.c_size_t(self.size))
        cu.cuMemAddressFree(self._ptr, ctypes.c_size_t(self.size))
        cu.cuMemRelease(self._handle)
