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
        # `scale` compresses the clock: every leaf must get SHORTER by that factor. Compute and H2D
        # divide their durations, so bandwidth has to be MULTIPLIED -- dividing it made reads 20x
        # slower while compute ran 20x faster, a 400x distortion of exactly the read-vs-compute
        # ratio the ordering tests exercise. Caught in review 2026-09-15. (phase2 runs at scale 1.0
        # and was never affected; the invariant tests were.)
        self.bw_one = bw_one * scale
        self.bw_two = bw_two * scale
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


# ---------------------------------------------------------------------------
# The leaf provider -- the seam a real component is swapped in at.
# ---------------------------------------------------------------------------
#
# GRANULARITY IS SET BY THE REAL ENGINE, NOT BY THE MODEL. engine/fastdecode.py captures TWO CUDA
# graphs per backbone layer: A = attention + HC + router (ends with the expert ids), then the host
# resolve, then B = routed MoE + shared expert + HC residual. So a layer has exactly two compute
# leaves that anything real can implement, and `C_IND`/`C_DEP`/`C_OTHER` are all INSIDE B.
#
# That is why the shared-expert ordering (what the skeleton called D4, `moe_before_shared`) is a
# property of THIS OBJECT and not of Policy. Reordering the shared expert against the routed MoE
# means editing `_layer_b` and recapturing the graph -- it is a different kind of change from a
# scheduling toggle, and modelling it as free would over-credit v2. A provider declares which
# ordering it has; it cannot be flipped per call.
#
# Segmented capture (DSV41_GRAPH_SEGMENTS, several layers in one graph with no host resolve
# between them) is RESIDENT-MODE ONLY. We stream, so the per-layer A/B seam is the real one.


class Leaves:
    """What a decode layer is made of. Implement all four to run the engine on real components.

    layer_a  -> the expert ids this layer wants. In the model this reads them from the captured
                route trace; on the real path it replays graph A and reads the router's output.
                THIS is what makes the driver an engine rather than a replay: the ids come out of
                compute, they are not handed to it.
    layer_b  -> the routed MoE + shared expert + HC residual, over slots already resident.
    read     -> one expert from NVMe into a leased staging buffer. One leaf from the caller's view;
                v1's `_read_leased` chunks it internally onto its own `read_pool`, and a real
                provider MUST keep that inner pool or it re-acquires v1's two-pool deadlock.
    h2d      -> staging buffer into the arena slot.
    """

    # True = this provider has the shared expert SPLIT OUT of graph B, so the driver can run it
    # while the reads fly. A real provider can only claim this if graph B was captured in two
    # pieces (B1 = shared expert, B2 = routed MoE + the rest); it is a capture-time cost, which is
    # exactly why it is declared here and not toggled per call.
    shared_first = False

    def layer_a(self, layer: int) -> tuple:
        raise NotImplementedError

    def shared(self, layer: int) -> None:
        """The shared expert alone. Only called when `shared_first`; otherwise it is inside layer_b."""
        raise NotImplementedError

    def layer_b(self, layer: int, slots) -> None:
        raise NotImplementedError

    def read(self, key: tuple, staging_id: int) -> object:
        raise NotImplementedError

    def h2d(self, slot: int, key: tuple, payload: object) -> None:
        raise NotImplementedError

    # --- prefill. Its real seam is NOT designed yet: v1's prefill runs a different MoE kernel
    # (tools/cb3_moe.py moe_forward_prefill) over a chunk of many tokens, not the decode path.
    # These exist so the chunked driver -- the only shape that can reach D3 -- stays runnable on
    # the modelled provider. A real provider must define them before prefill means anything.
    def prefill_attn(self, layer: int) -> None:
        raise NotImplementedError

    def prefill_moe(self, layer: int, slots, chunks: int = 1) -> None:
        raise NotImplementedError


class ModelLeaves(Leaves):
    """The modelled provider: calibrated sleeps, ids replayed from the captured route trace.

    Every duration here comes from the MEASURED CONSTANTS at the top of this file. None is tuned to
    make a total come out right -- when the model disagrees with the box, phase2 reports the gap as
    `unmodelled`.
    """

    def __init__(self, calls, bw: "Bandwidth", scale: float = 1.0, shared_first: bool = False,
                 start: int = 0):
        self.calls = calls
        self.bw = bw
        self.scale = scale
        self.shared_first = shared_first
        self.i = start

    def layer_a(self, layer: int) -> tuple:
        delay(C_PRE / self.scale)
        L, uniq = self.calls[self.i]
        if L != layer:
            raise AssertionError(f"trace desync: driver at layer {layer}, trace at {L}")
        self.i += 1
        return uniq

    def shared(self, layer: int) -> None:
        delay(C_IND / self.scale)

    def layer_b(self, layer: int, slots) -> None:
        # One leaf, because graph B is ONE replay. The modelled shares are summed rather than run
        # as separate sleeps: splitting them would invent overlap windows the real engine does not
        # have, and each sub-500us sleep would be spun on the GIL for nothing. The shared expert is
        # in here unless this provider split it out (`shared_first`).
        d = C_DEP + C_OTHER + (0.0 if self.shared_first else C_IND)
        delay(d / self.scale)

    def prefill_attn(self, layer: int) -> None:
        delay(C_PRE / self.scale)

    def prefill_moe(self, layer: int, slots, chunks: int = 1) -> None:
        delay(C_DEP * chunks / self.scale)

    def read(self, key: tuple, staging_id: int) -> object:
        self.bw.read(EXPERT_BYTES)
        return None

    def h2d(self, slot: int, key: tuple, payload: object) -> None:
        delay(H2D_S / self.scale)
