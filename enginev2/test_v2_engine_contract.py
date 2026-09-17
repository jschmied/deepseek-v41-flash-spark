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
        self.read_bytes, self.read_s = 0, 0.0

    def attach(self, *a, **kw):
        self.attached = kw
        return self

    def step(self):
        self.last_burst = self._bursts[self.i] if self.i < len(self._bursts) else []
        self.i += 1
        self.accepted.append(max(0, len(self.last_burst) - 1))
        # counters that only ever go up, like the real ones
        self.read_bytes += 1_000_000_000
        self.read_s += 0.5


class _FakeDriver:
    def __init__(self, leaves):
        self.leaves, self.closed = leaves, False
        self.c = types.SimpleNamespace(fetches=0, blocked_s=0.0, compute_s=0.0)
        self.policy = types.SimpleNamespace(global_barrier=True, resolve_blocks=True)
        # v2 owns the cache now, so stats() reads ExpertSlots' own accounting rather than v1's
        # store. The fake carries the same surface, and it only ever counts UP -- which is the
        # whole point: absolute reads would fold every earlier request into this one.
        self.slots = types.SimpleNamespace(hits=0, misses=0, prefill_misses=0)

    def decode(self, n):
        for _ in range(n):
            self.leaves.step()
            self.slots.hits += 3
            self.slots.misses += 1
            self.c.fetches += 1


class _Gate:
    """Records what it was shown, and can actually forbid a token."""

    def __init__(self, ban=None):
        self.seen, self.ban, self.mask_calls = [], ban, []

    def observe(self, ids):
        self.seen.extend(ids)

    def mask_rows(self, logits, block_ids):
        self.mask_calls.append(block_ids)
        if self.ban is not None:
            logits[..., self.ban] = float("-inf")
        return 1


def _engine(bursts, first=7):
    e = V2Engine.__new__(V2Engine)
    e.device = "cpu"
    e.leaves = _FakeLeaves(bursts)
    e.driver = _FakeDriver(e.leaves)
    e.engram_src = None
    e._stats = {}
    e.v1 = types.SimpleNamespace(
        model=types.SimpleNamespace(dspark_seed=lambda *a: None), tables={})
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


def test_grammar_constrains_the_FIRST_token_not_just_reports_it():
    """The gate must mask the prefill row BEFORE the first token is sampled.

    The first version sampled from `logits[-1]` and attached the grammar afterwards, so `observe()`
    saw the first token while `mask_rows()` never had a chance to constrain it: an illegal token
    could be SELECTED and then merely reported. That matters more now that a request WITHOUT tools
    also carries a plain-text gate, whose entire job is to keep one token out of a completion that
    has no legal use for it.
    """
    e = _engine([[1]], first=7)          # logits favour 7; the gate forbids it
    gate = _Gate(ban=7)
    out = list(e.generate([5], max_tokens=2, temperature=0.0, top_p=1.0,
                          stop_token_ids=set(), seed=None, grammar=gate))
    assert out[0] != [7], "the banned token was selected as the first token"
    assert gate.mask_calls and gate.mask_calls[0] is None, (
        "the prefill row must be masked as a single row (block_ids=None)")


def test_same_seed_gives_the_same_first_token():
    """Seeding used to happen inside attach(), which runs AFTER the first token is drawn."""
    firsts = []
    for _ in range(2):
        e = _engine([[1]], first=7)
        g = e.generate([5], max_tokens=1, temperature=1.0, top_p=1.0,
                       stop_token_ids=set(), seed=1234)
        firsts.append(next(g))
        g.close()
    assert firsts[0] == firsts[1], f"same prompt and seed gave {firsts}"


def test_stats_are_per_request_not_cumulative():
    """Counters live for the life of the process and the warm start moves them before request 1."""
    e = _engine([[1, 2], [3, 4]])
    list(e.generate([5], max_tokens=100, temperature=0.0, top_p=1.0,
                    stop_token_ids=set(), seed=None))
    first = e.stats()
    e.leaves._bursts, e.leaves.i = [[8]], 0       # a second, shorter request
    list(e.generate([5], max_tokens=100, temperature=0.0, top_p=1.0,
                    stop_token_ids=set(), seed=None))
    second = e.stats()
    # Request 1 runs three steps: two bursts plus the one that returns empty and ends it.
    assert first["nvme_gb"] == 3.0 and first["steps"] == 3, first
    # Request 2 runs two. Cumulative reads would report 5.0 here.
    assert second["nvme_gb"] == 2.0, (
        f"request 2 reported {second['nvme_gb']} GB -- it is carrying request 1's reads")
    assert second["steps"] == 2, second
    assert second["expert_misses"] == 2, second
