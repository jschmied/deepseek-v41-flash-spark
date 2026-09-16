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
import os
import sys

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
        self._last_reader = {}
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
    assert set(rl._last_reader) == {3, 7, 11}
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
