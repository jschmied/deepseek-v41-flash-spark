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
job 175's attribution did not assign to these three kernels, and it is modelled as `C_other`.

ITS PLACEMENT IS UNKNOWN, NOT SETTLED. An earlier revision argued that because fastdecode captures
two graphs per layer, C_other must sit inside graph B. That does not follow and the claim is
withdrawn. Job 175 sums CUPTI_ACTIVITY_KIND_KERNEL over the WHOLE decode span and buckets by kernel
NAME; it never correlates a kernel to a graph replay range. So C_other is "everything in a decode
step that matched none of three name patterns", which includes the final head graph and the
draft/MTP graphs -- not only per-layer work inside B.

Because 78.5 % dwarfs every attributed component, where it sits is not a detail: charging all of it
to a per-layer leaf inflates the compute a prefetch can hide behind, which flatters precisely the
prefetch arms. So it is a DECLARED PARAMETER (`c_other_in_layer`, default 1.0 = the old behaviour),
not a silent choice, and both ends of the range should be run until it is measured.

WHAT WOULD CLOSE IT: correlate kernels to graph executions in the sqlite export job 175 already
produces -- CUPTI records a graph node / graphExec id per launch, so a join against the per-layer
graph B exec gives the split directly. That is a query, not a new study.

HOST TIMER RESOLUTION, and the bias it introduces. C_ind is 4.4 us and C_dep is 77 us per layer.
`time.sleep` on this box does not resolve either; a 4.4 us sleep takes ~60 us and would inflate the
shared expert's overlap window ~14x, which is precisely the quantity the v2 case is argued from. So
compute leaves below SPIN_FLOOR are busy-waited on perf_counter instead, which is accurate but
holds the GIL. Compute runs on the driver thread, so under the v2 arm that spin throttles the
loader threads -- i.e. the distortion is biased AGAINST v2, which is the safe direction. Reads and
H2D are above the floor and use time.sleep, which releases the GIL.
"""

from __future__ import annotations

import dataclasses
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


@dataclasses.dataclass
class RouteResult:
    """What graph A produced. `uniq` is all the cache needs; `opaque` is the provider's own state.

    The driver must never unpack `opaque`. A real provider keeps route_idx and route_w in it --
    fastdecode's `_layer_b` consumes a slot tensor SHAPED LIKE route_idx (`moe_fn(y, self.slots,
    self.route_w, ...)`), so the per-expert ordering and multiplicity are functional inputs, not
    bookkeeping. Handing layer_b `sorted(set(slot_of.values()))` throws exactly that away and no
    real provider could rebuild `self.slots` from it.
    """

    uniq: tuple
    opaque: object = None


class StagedExpert:
    """One expert's bytes in a pinned staging buffer, OWNING the lease on that buffer.

    This exists because the lease and the data cannot be separated. v1's `_read_leased` hands the
    sink views that ALIAS the pinned buffer and requires the sink to finish before it returns; the
    real CB3 path is lease -> read_into(pinned) -> load_slot(pinned) -> synchronize -> release. If
    the payload does not carry the lease, a provider has only bad options: clone 13.8 MB per expert
    (wrong performance model), release before the copy completes (corruption), or hide the H2D
    inside read() (destroys the seam). So `read()` returns this, and the loader releases it only
    after `h2d` has completed.
    """

    __slots__ = ("sid", "payload", "_pool", "_released")

    def __init__(self, sid: int, payload: object, pool):
        self.sid = sid
        self.payload = payload
        self._pool = pool
        self._released = False

    def release(self) -> None:
        if self._released:
            return                      # idempotent: the loader releases in finally
        self._released = True
        self.payload = None             # use-after-release is a bug, so make it one
        self._pool.release(self.sid)

    @property
    def released(self) -> bool:
        return self._released


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

    # --- the per-STEP bracket. Stage 4 of the real bring-up found the contract had only an
    # epilogue (`step_other`, the final head graph) and nowhere to put a prologue -- but a decode
    # step really begins with work no layer owns: token ids and positions into their static
    # buffers, engram rows, the embedding gather, pre_mix, selecting the graphs for this step's
    # PARITY (the ratio-2 compressor grouping depends on S % 2), and capturing them the first time
    # a parity is seen. It ends with per-layer KV `pending` clones and advancing the cache length.
    # A provider that cannot express those is not an engine, so they are part of the contract.
    def make_staging(self, n: int, observer=None):
        """The staging pool is the PROVIDER's, because its buffers are the provider's medium: the
        modelled one hands out plain memoryviews, a real one hands out page-locked, ALIGN-aligned
        host memory that O_DIRECT can land in. The loader owns how many there are and when each is
        released; it does not own what they are made of."""
        from .store import StagingPool
        return StagingPool(n, observer=observer)

    def select_block(self, step: int) -> None:
        """Decide this step's input block, BEFORE anything else in the step runs.

        Split out of begin_step because the engram source needs it. v1's order is: build the block
        (draft + verify), hash it, D2H the hash ids, submit both tables' NVMe reads, and only then
        run the step. A block chosen inside begin_step is chosen AFTER EngramSource.issue() has
        already been called, so the second NVMe stream would be reading rows for the previous
        step's tokens -- silently, and only visibly as a quality loss.

        Default: nothing, for a provider whose block does not depend on the previous step.
        """

    def begin_step(self, step: int) -> None:
        """Everything a step does before its first layer. Default: nothing."""

    def end_step(self, step: int) -> None:
        """Everything a step does after its final head. Default: nothing."""

    def layer_a(self, layer: int) -> "RouteResult":
        raise NotImplementedError

    def bind_slots(self, route: "RouteResult", slot_of: dict) -> None:
        """Give graph B its route-aligned slot mapping. `slot_of` is expert id -> arena slot; a real
        provider builds the tensor shaped like route_idx here and copies it to the device, which is
        what `_resolve` does today (`self.slots.copy_(slots)`). Host work, so it runs before the
        wait, not after."""

    def shared(self, layer: int) -> None:
        """The shared expert alone. Only called when `shared_first`; otherwise it is inside layer_b."""
        raise NotImplementedError

    def layer_b(self, layer: int, route: "RouteResult") -> None:
        raise NotImplementedError

    def step_other(self) -> None:
        """Per-step GPU work outside the layer loop. Default: none."""

    def read(self, key: tuple, pool, ctx=None, scored: bool = True) -> "StagedExpert":
        """Acquire a staging buffer from `pool`, read the expert into it, and return a handle that
        OWNS that lease. The loader releases it after h2d completes, never before.

        `ctx` and `scored` are passed EXPLICITLY rather than through thread-locals so a provider
        with its own pool or thread layout still attributes a staging wait to the right request AND
        the right measurement cohort. Cohort identity has to travel with the work: a global "are we
        measuring" switch gets asynchronous ownership wrong in both directions -- it drops the end
        of a scored wait that outlives the window, and counts an unscored read that outlives
        settlement.
        """
        raise NotImplementedError

    def h2d(self, slot: int, key: tuple, staged: "StagedExpert") -> None:
        """Copy staged bytes into the arena slot. Must not return until the copy has completed --
        the buffer is released immediately afterwards."""
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
                 start: int = 0, c_other_in_layer: float = 1.0):
        self.calls = calls
        self.bw = bw
        self.scale = scale
        self.shared_first = shared_first
        self.start = start
        self.i = start
        # Fraction of the unattributed 78.5 % charged to the per-layer leaf; the remainder becomes
        # per-step work outside the layer loop. 1.0 reproduces the old model. See the header: this
        # is unmeasured, so anything sensitive to it must be reported at both ends.
        if not 0.0 <= c_other_in_layer <= 1.0:
            raise ValueError("c_other_in_layer must be in [0, 1]")
        self.c_other_in_layer = c_other_in_layer

    def layer_a(self, layer: int) -> RouteResult:
        delay(C_PRE / self.scale)
        L, uniq = self.calls[self.i]
        if L != layer:
            raise AssertionError(f"trace desync: driver at layer {layer}, trace at {L}")
        self.i += 1
        return RouteResult(uniq=uniq)

    def bind_slots(self, route: RouteResult, slot_of: dict) -> None:
        # The model has no tensors, but it keeps the mapping so the shape of the contract is
        # exercised: a provider that ignored this would fail against the real one, not here.
        route.opaque = tuple(slot_of[e] for e in route.uniq)

    def shared(self, layer: int) -> None:
        delay(C_IND / self.scale)

    def layer_b(self, layer: int, route: RouteResult) -> None:
        # One leaf, because graph B is ONE replay. The modelled shares are summed rather than run
        # as separate sleeps: splitting them would invent overlap windows the real engine does not
        # have, and each sub-500us sleep would be spun on the GIL for nothing. The shared expert is
        # in here unless this provider split it out (`shared_first`).
        d = C_DEP + C_OTHER * self.c_other_in_layer + (0.0 if self.shared_first else C_IND)
        delay(d / self.scale)

    def step_other(self) -> None:
        """Per-STEP work outside the layer loop (head, draft graphs) -- the part of C_other not
        charged to a layer. Zero under the default, which is why the default changes nothing."""
        d = C_OTHER * N_LAYERS * (1.0 - self.c_other_in_layer)
        if d > 0:
            delay(d / self.scale)

    def prefill_attn(self, layer: int) -> None:
        delay(C_PRE / self.scale)

    def prefill_moe(self, layer: int, slots, chunks: int = 1) -> None:
        delay(C_DEP * chunks / self.scale)

    def read(self, key: tuple, pool, ctx=None, scored: bool = True) -> StagedExpert:
        sid = pool.acquire(ctx, scored) if ctx is not None else pool.acquire(scored=scored)
        try:
            self.bw.read(EXPERT_BYTES)
            # A VIEW of the leased buffer, never a copy. This is the whole zero-copy contract: a
            # real reader preads into this memory and hands the H2D views that alias it. Returning
            # a clone here would be the 13.8 MB-per-expert mistake, and the tests check identity.
            view = pool.buffer(sid)
        except BaseException:
            pool.release(sid)
            raise
        return StagedExpert(sid, view, pool)

    def h2d(self, slot: int, key: tuple, staged: StagedExpert) -> None:
        delay(H2D_S / self.scale)
