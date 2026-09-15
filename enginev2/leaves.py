"""leaves.py -- the fake leaves, every one derived from a MEASURED quantity.

The rule this file exists to enforce: no leaf duration is ever tuned to make a total come out
right. Each is bytes/bandwidth, or a measured per-call cost, or a measured share of measured GPU
time. When the modelled total disagrees with the box, that disagreement is REPORTED as
`unmodelled`, never absorbed into a leaf. (The engine's arena sizer carried an 0.82 fudge that was
fitted to make one number look right and it cost three jobs on 2026-09-15; that is the failure mode
this file is written against.)

MEASURED CONSTANTS -- provenance for every one:

  EXPERT_BYTES  13,774,848 B   one native CB3 disk record; what a miss actually reads.
  BW_ONE        5.58 GB/s      one O_DIRECT expert read alone.
  BW_TWO        6.82 GB/s      two in flight; the device SATURATES near 2 concurrent reads, so
                               total bandwidth is flat at 6.82 for n >= 2 (it does not keep rising).
  H2D_S         942 us         cudaMemcpyAsync per expert H2D: 19.90 s over 21,132 calls.
  GPU_BUSY_S    3.0 s          GPU busy in the profiled decode span.
  SPAN_S        104.0 s        that span.
  STEPS_PER_S   1.73           measured decode steps/s -> 179.9 steps in the span.
  F_IND         0.0105         C_ind / C_layer (job 175, attribution by kernel identity).
  SHARE_MOE     0.185          routed MoE as a share of decode GPU time.
  SHARE_ATTN_HC 0.020          attention + HC.
                               (the shared expert's own share is 1.0 %, which rounds to F_IND
                               above; F_IND is quoted to more digits so C_ind is derived from it.)

WHAT THE PHASE SPLIT DOES NOT COVER, stated because it is large. The three attributed phases are
0.0105 + 0.185 + 0.020 = 21.5 % of decode GPU busy. The other 78.5 % is real measured GPU time that
job 175's attribution did not assign to these three kernels. It is modelled as `C_other`, a
per-layer phase with no dependency on any expert, placed AFTER the MoE. That placement is a CHOICE,
not a measurement: putting it before the router would hand the model an overlap window 78x the
shared expert's, which nothing measured justifies. Placing it last is the conservative option --
it can overlap nothing.

HOST TIMER RESOLUTION, and the bias it introduces. C_ind is 4.4 us and C_dep is 77 us per layer.
`time.sleep` on this box does not resolve either; a 4.4 us sleep takes ~60 us and would inflate the
shared expert's overlap window ~14x, which is precisely the quantity the v2 case is argued from. So
compute leaves below SPIN_FLOOR are busy-waited on perf_counter instead, which is accurate but
holds the GIL. Compute runs on the driver thread, so under the v2 arm that spin throttles the
loader threads -- i.e. the distortion is biased AGAINST v2, which is the safe direction. Reads and
H2D are above the floor and use time.sleep, which releases the GIL.
"""

from __future__ import annotations

import threading
import time

EXPERT_BYTES = 13_774_848
BW_ONE = 5.58e9
BW_TWO = 6.82e9
H2D_S = 942e-6

GPU_BUSY_S = 3.0
SPAN_S = 104.0
STEPS_PER_S = 1.73
N_LAYERS = 40

F_IND = 0.0105
SHARE_MOE = 0.185
SHARE_ATTN_HC = 0.020
SHARE_OTHER = 1.0 - F_IND - SHARE_MOE - SHARE_ATTN_HC

# GPU busy per decode step, then per layer. 3.0 s / (104 s * 1.73 steps/s) / 40 layers.
_STEPS_IN_SPAN = SPAN_S * STEPS_PER_S
C_LAYER = GPU_BUSY_S / _STEPS_IN_SPAN / N_LAYERS        # 417 us
C_IND = F_IND * C_LAYER                                 # 4.4 us   shared expert, expert-INdependent
C_PRE = SHARE_ATTN_HC * C_LAYER                         # 8.3 us   attention + HC, pre-router
C_DEP = SHARE_MOE * C_LAYER                             # 77 us    routed MoE, needs its experts
C_OTHER = SHARE_OTHER * C_LAYER                         # 327 us   unattributed, dependency-free

SPIN_FLOOR = 500e-6


def delay(seconds: float) -> None:
    """One modelled leaf. Spin below the host's sleep resolution, sleep above it -- see the header."""
    if seconds <= 0:
        return
    if seconds < SPIN_FLOOR:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            pass
        return
    time.sleep(seconds)


class Bandwidth:
    """The NVMe device, shared by every read in flight. EXACT, not sliced.

    total(n) = 5.58 GB/s at n=1 and 6.82 GB/s at n>=2, flat above 2 because the device saturates
    near 2 concurrent reads. Each read gets total(n)/n.

    Integration is event-driven rather than time-sliced: whenever a read joins or leaves, every
    active read is SETTLED (its remaining bytes reduced by the rate it has actually been running at
    since the last settle) and the rates are recomputed. A read then waits for its own predicted
    finish, and is woken early if the population changes. No slicing error, and no fitted constant.
    """

    def __init__(self, bw_one: float = BW_ONE, bw_two: float = BW_TWO, scale: float = 1.0):
        self.bw_one = bw_one / scale
        self.bw_two = bw_two / scale
        self._cv = threading.Condition()
        self._active: dict[int, float] = {}          # token -> remaining bytes
        self._last = 0.0
        self._next = 0
        self.bytes_read = 0
        self.busy_s = 0.0                            # wall time with >= 1 read in flight
        # concurrency-integral: sum of n*dt while n >= 1. Divided by busy_s it gives the mean
        # number of reads in flight WHILE THE DEVICE IS WORKING, which is the quantity the design
        # note's "the device saturates near 2 concurrent reads" is about. v1 achieves 2.78 of a
        # measured 6.82 GB/s whatever is tuned; if that is because fork-join can never have more
        # than one layer's misses in flight, this number says so directly.
        self.conc_integral = 0.0

    def _total(self, n: int) -> float:
        return 0.0 if n == 0 else (self.bw_one if n == 1 else self.bw_two)

    def _settle(self, now: float) -> None:
        n = len(self._active)
        if n:
            share = self._total(n) / n
            dt = now - self._last
            self.busy_s += dt
            self.conc_integral += n * dt
            done = share * dt
            for k in list(self._active):
                self._active[k] -= done
        self._last = now

    def read(self, nbytes: float = EXPERT_BYTES) -> float:
        t0 = time.perf_counter()
        with self._cv:
            self._settle(t0)
            tok = self._next
            self._next += 1
            self._active[tok] = nbytes
            self._cv.notify_all()
            while True:
                now = time.perf_counter()
                self._settle(now)
                rem = self._active.get(tok, 0.0)
                if rem <= 0:
                    break
                share = self._total(len(self._active)) / len(self._active)
                self._cv.wait(rem / share)
            del self._active[tok]
            self._settle(time.perf_counter())
            self._cv.notify_all()
            self.bytes_read += nbytes
        return time.perf_counter() - t0

    @property
    def mean_inflight(self) -> float:
        return self.conc_integral / self.busy_s if self.busy_s else 0.0

    @property
    def achieved_gbs(self) -> float:
        """Bytes divided by DEVICE-BUSY time -- what the device delivered while it had work."""
        return self.bytes_read / self.busy_s / 1e9 if self.busy_s else 0.0

    def peak_note(self) -> str:
        return (f"read model: {self.bw_one / 1e9:.2f} GB/s alone, {self.bw_two / 1e9:.2f} GB/s at "
                f">=2 in flight, {EXPERT_BYTES / 2 ** 20:.2f} MiB per expert")
