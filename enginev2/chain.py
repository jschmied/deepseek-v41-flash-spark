"""chain.py -- the happens-before edges of a decode step, stated explicitly.

WHY THIS FILE EXISTS. The skeleton's job is to be wait/event COMPLETE so real building blocks can be
plugged in, not to predict timings. An edge that is merely implied by the order of statements in a
for-loop is invisible to anything plugged in: a lookahead component would run ahead and silently
violate it. So every real edge gets a named event that must be waited on, even where sequential
execution would have provided it anyway.

THE EDGES, each read out of engine/fastdecode.py rather than assumed:

  graph A(L) reads   h, pre_mix, pos, eg_rows[L], c.win[L]
             writes  h, y, route_idx, route_w, kvl_buf[L], sc_buf[L]
  graph B(L) reads   y, slots, route_w, h, arena[slots]
             writes  h, pre_mix

  1. B(L-1) -> A(L)        via h and pre_mix. Both stages write `h`; A(L) reads it.
  2. A(L)   -> B(L)        via y, route_idx, route_w.
  3. A(L)   -> resolve(L)  the expert ids do not exist until A has run.
  4. resolve(L) -> B(L)    via `slots`, and via the expert bytes being resident.
  5. engram(L) -> A(L)     eg_rows[L] is read by A; the rows arrive from NVMe on a SEPARATE async
                           stream whose future is joined immediately before gA[L].replay().
  6. A(L) -> kv_bookkeeping(L)   c.pending[L] clones kvl_buf[L]/sc_buf[L] that A wrote.
  7. B(n-1) -> final       gF replays after the last layer.
  8. final -> next step    the next block's ids depend on this step's logits, via draft/verify.

THE BUFFERS ARE SHARED AND SINGLE, WHICH IS THE WHOLE LOOKAHEAD STORY. h, pre_mix, y, route_idx,
route_w and slots are ONE static tensor each, reused by every layer (`self.h.copy_(h)` in both A and
B, `self.route_idx.copy_(idx)`, `self.slots.copy_(slots)`). So layer L+1's A cannot be run early
even if its inputs were ready -- it would overwrite buffers layer L's B still needs. Running ahead
requires MORE BUFFERS, exactly as reading ahead requires more slots and more staging. That is why
the shortfall is a prediction problem and not a scheduling one: the ids for L+1 are not merely
unavailable, they are unproducible without a second set of buffers.

Edges 6, 7 and 8 are declared and waited on but have no leaf behind them yet; they are here so that
a plugged-in component cannot forget them, which is the point of the file.
"""

from __future__ import annotations

import threading

from .observe import NO_CTX, Event, NullObserver, WaitReason, next_span, now_ns


class Chain:
    """Named happens-before edges, keyed by (name, index). One-shot, monotonic within a step.

    `index` is the layer for per-layer edges and the STEP NUMBER for step-level ones. Step-level
    events are keyed by step rather than reset, because `reset()` deleting the previous step's
    "logits" before anything waited on it made edge 8 exist only in comments. Keying by step also
    survives final/draft/verify overlapping a following step, which is the shape this is for.

    A wait on a per-layer index below zero is satisfied trivially -- layer 0 has no predecessor,
    which is not a special case worth branching on at the call site.
    """

    def __init__(self, observer=None):
        self._lk = threading.Lock()
        self._cv = threading.Condition(self._lk)
        self._done: set = set()
        # The edge abstraction owns its own instrumentation: every caller of wait() is covered by
        # this one site, so a blocked edge is never also counted by whoever asked for it.
        self.obs = observer if observer is not None else NullObserver()
        # Two different numbers, and the difference is the finding. `checks` counts every edge the
        # driver actually reached; `blocks` counts the ones that were not yet satisfied. Today
        # blocks is ZERO on a healthy run: the driver is strictly sequential, so every edge is
        # already met when it is reached. That is not the chain being useless -- it is the precise
        # statement that nothing currently runs ahead. The moment a lookahead component exists,
        # these are exactly the edges it will block on.
        self.checks = 0
        self.blocks = 0

    def reset(self, keep: tuple = ()) -> None:
        """Clear per-layer edges for a new step. Step-level names in `keep` survive, because a
        later step may still have to wait on them. Counters are cumulative on purpose."""
        with self._lk:
            self._done = {(n, i) for (n, i) in self._done if n in keep}
            self._cv.notify_all()

    def set(self, name: str, index: int = -1, ctx=NO_CTX) -> None:
        with self._lk:
            self._done.add((name, index))
            self._cv.notify_all()
        if self.obs.enabled:
            self.obs.safe_emit(Event(now_ns(), "edge_set", ctx=ctx, aux=(name, index)))

    def wait(self, name: str, index: int = -1, timeout: float = 60.0, ctx=NO_CTX) -> None:
        if index < 0 and name in _PER_LAYER:
            return                     # no predecessor: layer 0
        with self._lk:
            self.checks += 1
            if self.obs.enabled:
                self.obs.safe_emit(Event(now_ns(), "edge_check", ctx=ctx, aux=(name, index)))
            if (name, index) in self._done:
                return
            self.blocks += 1
            sp = next_span()
            if self.obs.enabled:
                self.obs.safe_emit(Event(now_ns(), "wait_start", ctx=ctx, span=sp,
                                         aux=WaitReason.ENGRAM if name == "engram"
                                         else WaitReason.CHAIN))
            try:
                if not self._cv.wait_for(lambda: (name, index) in self._done, timeout):
                    raise TimeoutError(f"edge {name}@{index} never satisfied")
            finally:
                if self.obs.enabled:
                    self.obs.safe_emit(Event(now_ns(), "wait_end", ctx=ctx, span=sp,
                                             aux=WaitReason.ENGRAM if name == "engram"
                                             else WaitReason.CHAIN))

    def is_set(self, name: str, index: int = -1) -> bool:
        with self._lk:
            return (name, index) in self._done


_PER_LAYER = frozenset({"h", "y", "route", "engram", "kv"})


class EngramSource:
    """The SECOND async NVMe stream, whose rows graph A reads.

    Not a timing detail -- a second consumer with its own wait in the middle of the layer chain.
    engine/engram.py reads 264 B rows (256 B weights + 8 B scale) with buffered preadv on a thread
    pool, ~144 lookups per layer per step before dedup against a 200k-row process cache. That is
    ~0.2 % of the expert byte traffic and a large number of tiny syscalls, so it competes for host
    CPU and IOPS rather than for bandwidth.

    The default source completes immediately: a skeleton must not invent a delay it has not
    measured. What it must do is make the EDGE exist, so a component plugged in here is waited on.
    """

    name = "null"

    def issue(self, layers, step: int, chain: "Chain") -> None:
        """Start this step's reads and SIGNAL each layer as its rows land.

        Event-based on purpose: a blocking `join(layer)` would hide the edge inside a call, and the
        driver could not tell a satisfied edge from one nobody ever produced. A real source hands
        work to its own pool here and each worker calls `chain.set("engram", L)` on completion; the
        driver only ever waits. The null source has no reads, so every layer is ready at once --
        the edge still EXISTS, which is the point.
        """
        for L in layers:
            chain.set("engram", L)
