"""Review 2026-09-16, item 2: the real arena reader lifetime must be a DEVICE edge, not a host one.

ComputeStream.run(slots) models a slot as read for the lifetime of the Python call. For RealLeaves
that is false: layer_b() replays a CUDA graph and returns while the GPU is still reading the arena,
so the host-side wait_slot_free() sees readers[] at zero and D1-off would let a copy overwrite a
slot the current graph is mid-read of.

These tests do not race the GPU -- a race that happens to pass proves nothing. They assert the
INVARIANT: after layer_b, every slot that layer bound carries the event recorded after its replay,
and h2d waits on exactly that event before touching the slot.

Needs torch; does not need the model, the box's NVMe, or an engine.
"""
import inspect
import os
import re
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from enginev2.leaves import RouteResult                    # noqa: E402
from enginev2.real import RealLeaves                       # noqa: E402


class _Graph:
    """Stands in for a captured CUDA graph. Only replay() is ever called on one."""

    def __init__(self, log, layer):
        self._log, self._layer = log, layer

    def replay(self):
        self._log.append(self._layer)


class _Bare(RealLeaves):
    """RealLeaves without CB3Cache or an engine: only the stream/event bookkeeping under test.

    It INHERITS layer_b and h2d rather than reimplementing them. An earlier version of this file
    defined its own layer_b, so mutating the real one left these tests green -- a test that
    reimplements the code under test verifies the copy, not the code.
    """

    def __init__(self):
        import threading
        self._ctr = threading.Lock()
        self._tls = threading.local()
        self.engram = None
        self.engram_ablated = 0
        self._arena_slots = 64
        self._last_reader = [None] * self._arena_slots
        self._stats_tls = threading.local()
        self._stats_all = []
        self._stats_lk = threading.Lock()
        # Mirrors RealLeaves.__init__. This fixture reimplements that constructor, so it drifts
        # whenever state is added -- it has now broken twice that way. Anything new there belongs
        # here too, or the test stops covering the code it names.
        # Per-request state RealLeaves.__init__ declares. Third drift; the guard test below is
        # what caught it each time.
        self.eng = None
        self.fd = None
        self.accepted = []
        self.last_burst = []
        self.hist = []
        self.grammar = None
        self.penalties = None
        self.tok = None
        self.tokens_out = 0
        self.engram_ablated = 0
        self.trace_verify = None
        self._tv_tok = None
        self._copy_event = {}
        self.copies_awaited = 0
        self._reader_lk = threading.Lock()
        self._bound_slots = frozenset()
        self.reader_waits = 0
        self.record = 4096
        self._replays = []
        self._gB = [_Graph(self._replays, L) for L in range(4)]


def test_every_bound_slot_carries_the_event_recorded_after_its_replay():
    rl = _Bare()
    rl._bound_slots = frozenset({3, 7, 11})
    rl.layer_b(0, RouteResult(uniq=(1,)))
    # A LIST indexed by slot, not a dict: one writer (layer_b) and many readers (h2d), so under
    # the GIL a plain list needs no lock. Slots nobody bound stay None.
    assert [i for i, e in enumerate(rl._last_reader) if e is not None] == [3, 7, 11]
    ev0 = rl._last_reader[3]
    assert rl._last_reader[7] is ev0 and rl._last_reader[11] is ev0, \
        "one event per layer is the design; per-slot events are thousands of objects for nothing"

    # a second layer over overlapping slots must REPLACE the event for the slots it re-reads and
    # leave the others on their own layer's event -- otherwise a slot is freed by a graph that
    # never read it.
    rl._bound_slots = frozenset({7, 20})
    rl.layer_b(1, RouteResult(uniq=(1,)))
    assert rl._last_reader[3] is ev0
    assert rl._last_reader[7] is not ev0
    assert rl._last_reader[20] is rl._last_reader[7]


def test_h2d_waits_on_the_last_reader_of_that_slot_and_no_other():
    """The copy stream must wait for ONE event -- the last graph that read THIS slot -- not for the
    compute stream as a whole. wait_stream(compute) is v1's edge and it takes everything queued up
    to the moment it is called, which under early submission includes work queued during the read
    it was meant to overlap."""
    rl = _Bare()
    rl._bound_slots = frozenset({5})
    rl.layer_b(0, RouteResult(uniq=(1,)))
    want = rl._last_reader[5]

    waited = []
    rl._tls.stream = torch.cuda.Stream()
    # Patch the METHOD, not the object: torch.cuda.stream() requires a real Stream, so a stand-in
    # object fails before the code under test runs.
    orig_wait = torch.cuda.Stream.wait_event

    def spy(self, ev, _o=orig_wait):
        waited.append(ev)
        return _o(self, ev)

    torch.cuda.Stream.wait_event = spy

    class _Staged:
        payload = None

    class _Cache:
        def load_slot(self, *a, **k):
            pass

    rl.cache = _Cache()
    rl.arena = None
    try:
        # slot 5 has a reader: it must be waited on
        rl.h2d(5, (0, 0), _Staged())
        assert waited == [want], waited
        assert rl.reader_waits == 1

        # slot 9 was never read by any graph: there is nothing to wait for, and inventing a wait
        # would serialise copies against unrelated compute
        waited.clear()
        rl.h2d(9, (0, 1), _Staged())
        assert waited == [], waited
        assert rl.reader_waits == 1
    finally:
        torch.cuda.Stream.wait_event = orig_wait


def test_the_bare_fixture_has_not_drifted_from_RealLeaves():
    """_Bare reimplements RealLeaves.__init__, and that has broken three separate times -- once
    for the reader table, once for the copy-event table, once for the arena slot count. Each time
    the symptom was an AttributeError inside an unrelated test, which says nothing about what to
    add.

    This compares the two directly, so drift fails HERE and names the missing attribute. Fields
    that need a CB3 file or a live arena are excluded: avoiding those is why the fixture exists.
    """
    src = inspect.getsource(RealLeaves.__init__)
    assigned = {m.strip().removeprefix("self.").split(":")[0].split("=")[0].strip()
                for m in re.findall(r"^\s+(self\.\w+)\s*(?::[^=]+)?=", src, re.M)}
    provider_owned = {"cache", "arena", "record", "_arena_slots"}
    bare = _Bare()
    missing = sorted(a for a in assigned - provider_owned if not hasattr(bare, a))
    assert not missing, (
        f"_Bare is missing state RealLeaves.__init__ sets: {missing}. Add it to the fixture, or "
        f"these tests silently stop covering the code paths that use it.")


# ----------------------------------------------------------------------------------------------
# The completer retires copies in the order they FINISH.
#
# Every fake provider in the suite returns None from h2d(), so the whole suite exercises the
# completer's trivial path (no event -> retire at once) and would pass just as well with the old
# in-order synchronize(). This drives the loop directly with stub events instead.
# ----------------------------------------------------------------------------------------------

class _StubEvent:
    """A device event whose completion we control. synchronize() is a *failure* here: the point of
    the change is that a ready copy is retired without blocking on an unready older one."""

    def __init__(self, name, done=False):
        self.name, self.done, self.synchronized = name, done, False
        self.queries = 0

    def query(self):
        self.queries += 1
        return self.done

    def synchronize(self):
        self.synchronized = True
        self.done = True


class _StubLoader:
    """Only what LoaderService._completer touches. The poll interval is TAKEN from the real class
    rather than copied, so the timing assertions below stay honest if it is ever retuned."""

    from enginev2.sched import LoaderService as _LS
    _COMPLETER_POLL_S = _LS._COMPLETER_POLL_S
    _completer_in_order = False
    del _LS

    def __init__(self):
        import queue as _q
        self._completions = _q.Queue()
        self.retired = []

    def _complete_h2d(self, handle, *rest):
        self.retired.append(handle.name if handle is not None else None)


def _run_completer(loader):
    import threading
    from enginev2.sched import LoaderService
    t = threading.Thread(target=LoaderService._completer, args=(loader,), daemon=True)
    t.start()
    return t


def test_a_finished_copy_is_not_held_behind_a_running_one():
    slow, fast = _StubEvent("slow", done=False), _StubEvent("fast", done=True)
    ld = _StubLoader()
    ld._completions.put((slow,))
    ld._completions.put((fast,))
    t = _run_completer(ld)
    # The fast copy must come out while the slow one is still running. Give the thread a moment;
    # it has to observe an empty queue before it polls, which is one non-blocking get.
    for _ in range(200):
        if ld.retired:
            break
        time.sleep(0.005)
    assert ld.retired == ["fast"], (
        f"expected the finished copy first, got {ld.retired} -- the completer is still retiring "
        f"in submission order, so a done copy waits on a running one")
    assert not slow.synchronized, "the completer must never block on one specific event"
    slow.done = True
    ld._completions.put(None)
    t.join(timeout=5)
    assert not t.is_alive()
    assert ld.retired == ["fast", "slow"]


def test_the_completer_drains_pending_work_before_it_exits():
    """Shutdown must not drop a copy that is still in `pending`: its staging buffer and its H2D
    permit are released by _complete_h2d, and a caller may be waiting on the queue to join."""
    ev = _StubEvent("late", done=False)
    ld = _StubLoader()
    ld._completions.put((ev,))
    t = _run_completer(ld)
    time.sleep(0.05)
    ld._completions.put(None)
    ev.done = True
    t.join(timeout=5)
    assert not t.is_alive()
    assert ld.retired == ["late"]
    ld._completions.join()          # every get() was matched by exactly one task_done()


def test_the_completer_does_not_spin_when_nothing_is_ready():
    """With one unfinished copy outstanding the loop must poll at a bounded rate, not as fast as
    the CPU allows -- otherwise every copy in flight burns a core for its whole duration."""
    ev = _StubEvent("only", done=False)
    ld = _StubLoader()
    ld._completions.put((ev,))
    t = _run_completer(ld)
    time.sleep(0.2)
    assert ld.retired == []
    assert not ev.synchronized, "the completer must never block on one specific event"
    # 0.2 s at the 200 us poll is ~1000 passes. A busy loop would be six orders of magnitude more.
    assert ev.queries < 20000, f"{ev.queries} query() calls in 0.2 s -- the completer is spinning"
    ev.done = True
    ld._completions.put(None)
    t.join(timeout=5)
    assert ld.retired == ["only"]


def test_the_in_order_measurement_arm_still_reproduces_the_old_behaviour():
    """The A/B arm has to be the DEFECT, not a second copy of the fix -- otherwise the measurement
    compares the change against itself and returns a null by construction."""
    slow, fast = _StubEvent("slow", done=False), _StubEvent("fast", done=True)
    ld = _StubLoader()
    ld._completer_in_order = True
    ld._completions.put((slow,))
    ld._completions.put((fast,))
    t = _run_completer(ld)
    for _ in range(200):
        if len(ld.retired) == 2:
            break
        time.sleep(0.005)
    assert ld.retired == ["slow", "fast"], ld.retired
    assert slow.synchronized, "the in-order arm must block on the older event -- that is the defect"
    ld._completions.put(None)
    t.join(timeout=5)
