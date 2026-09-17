"""The server's generator contract, tested without a GPU.

`server/engine_api.py` is specific about how `generate` must behave, and every clause is a way a
server breaks quietly rather than loudly:

  * yield BURSTS (spec decode commits several tokens per step), not one token at a time
  * stop after emitting a stop id, and never exceed max_tokens
  * the server DRAINS after a stop id (up to 4 more bursts, ignored) so the epilogue runs, but
    CLOSES the generator on a stop STRING or a disconnect -- GeneratorExit at the pending yield, so
    cleanup must be in a `finally` and the engine usable afterwards
  * the grammar gate is observed once per settled token, in order

None of that involves weights, so it is tested here against fake leaves and a fake driver. The GPU
half (prefill and the decode step itself) is gated separately: job 536 for prefill, bitwise.
"""
import sys, types, pytest, torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from enginev2.v2_engine import V2Engine


class _FakeLeaves:
    def __init__(self, bursts):
        self._bursts, self.i = bursts, 0
        self.last_burst, self.accepted, self.grammar = [], [], None
        self.attached = None

    def attach(self, *a, **kw):
        self.attached = kw
        return self

    def step(self):
        self.last_burst = self._bursts[self.i] if self.i < len(self._bursts) else []
        self.i += 1
        self.accepted.append(max(0, len(self.last_burst) - 1))


class _FakeDriver:
    def __init__(self, leaves):
        self.leaves, self.closed = leaves, False
        self.c = types.SimpleNamespace(fetches=0, blocked_s=0.0, compute_s=0.0)
        self.policy = types.SimpleNamespace(global_barrier=True, resolve_blocks=True)

    def decode(self, n):
        for _ in range(n):
            self.leaves.step()


class _Gate:
    def __init__(self):
        self.seen = []

    def observe(self, ids):
        self.seen.extend(ids)

    def mask_rows(self, logits, block_ids):
        return 0


def _engine(bursts, first=7):
    e = V2Engine.__new__(V2Engine)
    e.device = "cpu"
    e.leaves = _FakeLeaves(bursts)
    e.driver = _FakeDriver(e.leaves)
    e.engram_src = None
    e._stats = {}
    e.v1 = types.SimpleNamespace(
        model=types.SimpleNamespace(dspark_seed=lambda *a: None),
        store=types.SimpleNamespace(stats={"misses": 0, "prefill_misses": 0, "bytes_read": 0,
                                           "load_s": 0.0}, hit_rate=lambda: 1.0),
        tables={})
    lg = torch.zeros(1, 16); lg[0, first] = 10.0
    e._prefill = lambda ids: (lg, None, 0)
    return e


def test_bursts_and_stop_id():
    e = _engine([[1, 2], [3, 9, 4]])
    out = list(e.generate([5, 5], max_tokens=100, temperature=0.0, top_p=1.0,
                          stop_token_ids={9}, seed=None))
    assert out == [[7], [1, 2], [3, 9, 4]], out
    assert e.stats()["tokens"] == 6


def test_max_tokens_is_never_exceeded():
    """A burst that would overshoot is TRUNCATED, not yielded whole -- the server bills these."""
    e = _engine([[1, 2, 3], [4, 5, 6]])
    out = list(e.generate([5], max_tokens=4, temperature=0.0, top_p=1.0,
                          stop_token_ids=set(), seed=None))
    assert sum(len(b) for b in out) == 4, out
    assert e.stats()["tokens"] == 4


def test_first_token_alone_can_be_the_stop():
    e = _engine([[1, 2]], first=3)
    out = list(e.generate([5], max_tokens=50, temperature=0.0, top_p=1.0,
                          stop_token_ids={3}, seed=None))
    assert out == [[3]], out


def test_close_mid_stream_runs_the_epilogue():
    """A stop STRING or a disconnect CLOSES the generator: cleanup must still happen."""
    e = _engine([[1, 2], [3, 4], [5, 6]])
    g = e.generate([5], max_tokens=100, temperature=0.0, top_p=1.0,
                   stop_token_ids=set(), seed=None)
    assert next(g) == [7]
    assert next(g) == [1, 2]
    g.close()                                   # GeneratorExit at the pending yield
    assert e.leaves.grammar is None, "the gate must be detached even when the client vanishes"
    assert e.stats()["tokens"] == 3, e.stats()

def test_a_step_that_commits_nothing_ends_the_generation():
    """Otherwise the loop spins to max_tokens emitting empty bursts."""
    e = _engine([[1], []])
    out = list(e.generate([5], max_tokens=100, temperature=0.0, top_p=1.0,
                          stop_token_ids=set(), seed=None))
    assert out == [[7], [1]], out


def test_grammar_is_observed_once_per_settled_token_in_order():
    e = _engine([[1, 2], [3]])
    gate = _Gate()
    list(e.generate([5], max_tokens=100, temperature=0.0, top_p=1.0,
                    stop_token_ids=set(), seed=None, grammar=gate))
    assert gate.seen == [7, 1, 2, 3], gate.seen
