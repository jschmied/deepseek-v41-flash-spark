"""`V2Engine` -- the v2 driver behind the server's `Engine` ABC.

THE ENDPOINT LIST IS NOT THE WORK. `server/app.py` implements /v1/chat/completions, SSE streaming,
/v1/completions, chat-template rendering, thinking/reasoning, tools and DSML parsing, xgrammar
constraints, stop strings, usage accounting, /health and /v1/models against a THREE-METHOD abstract
class in `server/engine_api.py`, and `MockEngine` runs that whole half with no GPU. So everything
those endpoints need from v2 is this file.

What it reuses from v1, deliberately: the weights, tokenizer, CB3 arena and expert store
(`V41Engine` as a resource holder), and the leaf math `RealLeaves` already calls. What it does NOT
reuse is v1's control: `_generate`, `_decode_loop`, its prefill branch and its stats assembly are
replaced by v2's driver, which is the point of v2 existing.

Contract notes that are easy to get wrong and are handled here:

  * The server DRAINS the generator after a stop id or max_tokens (up to 4 more bursts, ignored) so
    the engine's epilogue runs. On a stop STRING or a client disconnect it CLOSES the generator
    instead, raising GeneratorExit at the pending yield -- so cleanup is in a `finally` and the
    engine must be usable for the next request afterwards.
  * `grammar` is passed to ANY engine with `supports_grammar`, not only when the request carries
    tools: a request without tools gets a plain-text gate that keeps the DSML bar out of a
    completion with no legal use for it. Calling it unconditionally is correct.
  * `context_margin` is the DSpark draft block the engine needs BEYOND prompt + max_tokens.
"""
from __future__ import annotations

import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.v41_engine import V41Engine, sample_probs          # noqa: E402
from enginev2 import drivers as v2drivers                      # noqa: E402
from enginev2.real import RealEngramSource, RealLeaves         # noqa: E402
from enginev2.sched import Policy                              # noqa: E402

try:                                                            # the ABC lives in the server tree
    from server.engine_api import Engine as _ServerEngine
except Exception:                                               # noqa: BLE001
    _ServerEngine = object


class V2Engine(_ServerEngine):
    """Single-sequence generation on the v2 driver."""

    eos_token_id = 1
    supports_grammar = True
    #: presence/frequency penalties, the cycle breaker and no_repeat_ngram. The server
    #: only builds and passes Penalties for an engine that says yes, so leaving this
    #: False meant the API validated those fields and then silently dropped them.
    supports_penalties = True
    #: DSpark drafts 6 tokens and the verify block is one wider.
    context_margin = 8

    def __init__(self, model_dir: str, *, max_seq: int = 32768, arena_gb=None,
                 cb3_path: str | None = None, keep_free_gb: float = 20.0,
                 transient_slots: int = 400, evict: str = "lru",
                 engram: bool = True, policy: Policy | None = None,
                 n_workers: int = 48, staging: int = 48,
                 trace_stats=None, hot_profile=None):
        # warm_start=False: v1 allocates the arena and the weights and leaves residency ALONE.
        # v2 owns the expert cache -- its own ExpertSlots, its own loader, its own warm start.
        self.v1 = V41Engine(model_dir, max_seq=max_seq, arena_gb=arena_gb, spec=True,
                            expert_format="cb3", keep_free_gb=keep_free_gb,
                            transient_slots=transient_slots, warm_start=False,
                            trace_stats=trace_stats, hot_profile=hot_profile)
        self.max_context = max_seq
        self.tokenizer = self.v1.tokenizer
        v2drivers.N_LAYERS = self.v1.args.n_layers
        cb3 = cb3_path or os.environ["DSV41_CB3_CACHE"]
        self.leaves = RealLeaves(cb3, self.v1.arena)
        self.engram_src = RealEngramSource(self.v1, self.leaves) if engram else None
        self.leaves.engram = self.engram_src
        self.driver = v2drivers.Engine(
            policy if policy is not None else Policy(), evict=evict,
            # The ARENA's capacity, read from the arena -- not from v1's store. v2 must not
            # reach into bookkeeping it does not own, even to ask how big the memory is.
            lru_slots=self.v1.arena.slots - transient_slots,
            transient_slots=transient_slots, n_workers=n_workers, staging=staging,
            leaves=self.leaves, engram=self.engram_src)
        # THE RANKING IS THE RESIDENCY. Without `trace_stats`, V41Engine falls back to
        # [(L, e) for e in range(384) for L in range(40)] -- expert-ID order -- and warming the
        # first arena-sized slice of THAT fills the cache with experts 0..k of every layer instead
        # of the measured hot set. It would load, serve, and quietly give up the entire residency
        # advantage. Fail loudly instead of warming the wrong thing.
        rank = self.v1.warm_rank
        if trace_stats is not None:
            naive = [(L, e) for e in range(384) for L in range(40)]
            if list(rank[:64]) == naive[:64]:
                raise RuntimeError(
                    "trace_stats was supplied but warm_rank is the unranked expert-ID fallback: "
                    "the hot-expert ranking was lost on the way in, and the arena would be warmed "
                    "with experts 0..k of every layer instead of the measured working set.")
        self.warm_start(rank)
        # BIND ONCE, HERE. _prefill() drives the leaves before a request can know its first token,
        # and attach(spec=True) requires that token -- so the spec attach cannot come first.
        # A non-spec attach binds eng/fd and runs the capture-provenance check; generate() then
        # re-attaches per request with the token and the sampling parameters.
        self.leaves.attach(self.v1, None)
        self._stats: dict = {}
        # Injectable so the generator CONTRACT (bursts, stop ids, max_tokens, the
        # GeneratorExit path) can be tested without a GPU -- that logic is where a
        # server breaks silently, and it has nothing to do with the device.
        self.device = "cuda"

    def warm_start(self, ranked_keys, log=print) -> int:
        """Fill v2's OWN cache, through v2's own reserve/submit path.

        This used to be `_adopt_resident`: v1 warm-started its ExpertSlots and v2 copied the map
        into its own. That is two populated maps over one arena, and every question about residency
        then had two answers. Here the reads go through the v2 loader, so the slots, the generation
        counters and the eviction policy are v2's from the first byte -- and the warm start becomes
        a measurement of v2's own I/O path rather than of v1's.
        """
        keys = list(ranked_keys)[: self.driver.slots.lru_slots]
        t0 = time.perf_counter()
        by_layer: dict[int, list[int]] = {}
        for L, e in keys:
            by_layer.setdefault(int(L), []).append(int(e))
        n = 0
        for L, experts in sorted(by_layer.items()):
            _slot_of, to_load, _to_wait = self.driver.slots.reserve(L, experts, prefill=False)
            self.driver.loader.submit(to_load)
            self.driver.loader.quiesce(timeout=1800)
            self.driver.loader.drain_forgets()
            n += len(to_load)
        gb = self.leaves.read_bytes / 1e9
        log(f"v2 warm start: {n} experts resident of {len(keys)} ranked, {gb:.1f} GB read "
            f"in {time.perf_counter() - t0:.0f}s")
        return n

    # ------------------------------------------------------------------ prefill
    def _prefill(self, ids: torch.Tensor):
        """Run the prompt and return (logits, mh, s_rep). Bitwise-gated against v1 (job 536)."""
        m = self.v1.model
        m.c.rollback(0)
        m.begin_prompt()
        m.c.checkpoint(0)
        n_chunks = self.leaves.begin_prefill(ids, 0)
        for L in range(m.args.candidate_source_layer + 1):
            self.driver.prefill_chunked(L, n_chunks)
        self.leaves.finish_prefill()
        return m.decoder_replay(need_logits=True)

    # ------------------------------------------------------------------ the ABC
    def generate(self, prompt_ids, *, max_tokens: int = 4096, temperature: float = 1.0,
                 top_p: float = 0.95, stop_token_ids=None, seed=None, grammar=None,
                 penalties=None, ignore_eos: bool = False, **_ignored):
        stop = set(stop_token_ids or ())
        # SEED BEFORE ANYTHING IS SAMPLED. This used to sit inside attach(), which runs AFTER the
        # first token has been drawn -- so the first token was not reproducible from (prompt, seed),
        # and the reseed then restarted the RNG stream mid-generation, which is not v1's semantics
        # either. v1 seeds before prefill; so does this now.
        if seed is not None:
            torch.manual_seed(int(seed))
        ids = torch.tensor(list(prompt_ids), dtype=torch.long, device=self.device)
        t0 = time.perf_counter()
        n_out = 0
        # PER-REQUEST WINDOW. Every counter below lives for the life of the process, and warm_start()
        # drives reserve()/submit() before request 1 even arrives -- so absolute reads would put the
        # warm start in request 1 and requests 1..N-1 in request N, and nvme_gb_per_token would be
        # nonsense. Snapshot here, report deltas. This is the same windowing mistake that cost the
        # host-phase profile and the residency figures earlier today; making it structural is the
        # only fix that holds.
        base = self._counters()
        try:
            logits, mh, s_rep = self._prefill(ids)
            t_prefill = time.perf_counter() - t0
            # THE GATE CONSTRAINS THE FIRST TOKEN TOO. Sampling first and attaching the grammar
            # afterwards let observe() see the token while mask_rows() never had a chance to
            # constrain it -- so an illegal token could be SELECTED and then merely reported. That
            # matters more now that a request without tools also carries a plain-text gate whose
            # whole job is to keep the DSML bar out of a completion that has no legal use for it.
            row = logits[-1:].float()
            hist: list = []
            pen = penalties if (penalties is not None and penalties.active) else None
            if pen is not None:
                pen.apply(row, hist)
            if grammar is not None:
                grammar.mask_rows(row, None)
            p = sample_probs(row[0], temperature, top_p)
            first = int(torch.multinomial(p, 1)) if temperature > 0 else int(p.argmax())
            self.v1.model.dspark_seed(mh, s_rep)
            self.leaves.attach(self.v1, None, spec=True, temperature=temperature, top_p=top_p,
                               first_token=first, stop_ids=stop)
            self.leaves.grammar = grammar
            self.leaves.penalties = pen
            self.leaves.hist = hist
            n_out = 1
            hist.append(first)
            if pen is not None:
                pen.observe([first])
            if grammar is not None:
                grammar.observe([first])
            yield [first]
            if first in stop and not ignore_eos:
                return
            t_dec0 = time.perf_counter()
            while n_out < max_tokens:
                self.driver.decode(1)
                burst = self.leaves.last_burst
                if not burst:
                    break                      # a step that committed nothing cannot make progress
                if n_out + len(burst) > max_tokens:
                    # The step already COMMITTED the whole burst; hand the extra positions back so
                    # the cache agrees with what the caller received. See discard_tail: it does not
                    # undo the expert reads, only the cache.
                    keep = max_tokens - n_out
                    self.leaves.discard_tail(len(burst) - keep)
                    burst = burst[:keep]
                n_out += len(burst)
                hist.extend(burst)
                if pen is not None:
                    pen.observe(burst)
                if grammar is not None:
                    grammar.observe(burst)
                yield burst
                if not ignore_eos and any(t in stop for t in burst):
                    return
        finally:
            # CLEANUP BELONGS HERE, not after the loop: a stop STRING or a disconnect closes the
            # generator and raises GeneratorExit at the pending yield, so the lines after the loop
            # never run. The engine has to be usable for the next request either way.
            self.leaves.grammar = None
            self.leaves.penalties = None
            self.leaves.hist = []
            wall = time.perf_counter() - t0
            self._stats = self._collect(n_out, wall, locals().get("t_prefill"),
                                        locals().get("t_dec0"), base)

    def _counters(self) -> dict:
        """Everything cumulative, sampled at one instant. See the note in generate()."""
        sl, c = self.driver.slots, self.driver.c
        return {"hits": sl.hits, "misses": sl.misses, "prefill_misses": sl.prefill_misses,
                "read_bytes": self.leaves.read_bytes, "read_s": self.leaves.read_s,
                "fetches": c.fetches, "blocked_s": c.blocked_s, "compute_s": c.compute_s,
                "engram_rows": (sum(t.stats["rows"] for t in self.v1.tables.values())
                                if self.engram_src is not None else 0)}

    def _collect(self, n_out: int, wall: float, t_prefill, t_dec0, base: dict) -> dict:
        now = self._counters()
        d = {k: now[k] - base[k] for k in now}
        # `accepted` is RESET BY attach() on every request, so it is already request-local. Slicing
        # it against a cumulative base made request 2 report steps=0 and accept_len_mean=None: the
        # base was request 1's length, and the list had been emptied underneath it.
        acc = self.leaves.accepted
        c, sl = self.driver.c, self.driver.slots
        out = {
            "engine": "v2",
            "tokens": n_out,
            "wall_s": round(wall, 3),
            "ttft_s": round(t_prefill, 3) if t_prefill else None,
            # n_out - 1: the first token comes out of PREFILL and is charged to TTFT. Counting it
            # as decode overstates the rate by 1/n, which is ~11 % on a 10-token test generation.
            "decode_tok_s": (round(max(n_out - 1, 0) / (time.perf_counter() - t_dec0), 2)
                             if t_dec0 and n_out > 1 else None),
            "steps": len(acc),
            # +1: a step commits the accepted drafts AND the bonus token.
            "accept_len_mean": round(sum(acc) / len(acc) + 1, 2) if acc else None,
            # ALL DELTAS over this request. v2's own counters, not v1's store -- with
            # warm_start=False that map is empty and every figure would be a zero reported as a fact.
            "expert_hit_rate": (round(d["hits"] / max(1, d["hits"] + d["misses"] + d["prefill_misses"]), 4)),
            "expert_misses": d["misses"],
            "prefill_expert_misses": d["prefill_misses"],
            "nvme_gb": round(d["read_bytes"] / 1e9, 2),
            "nvme_gb_per_token": round(d["read_bytes"] / 1e9 / max(n_out, 1), 3),
            "nvme_read_s": round(d["read_s"], 2),
            "v2_fetches": d["fetches"],
            "v2_blocked_s": round(d["blocked_s"], 2),
            "v2_compute_s": round(d["compute_s"], 2),
            "policy": {"global_barrier": self.driver.policy.global_barrier,
                       "resolve_blocks": self.driver.policy.resolve_blocks},
        }
        if self.engram_src is not None:
            out["engram_rows"] = d["engram_rows"]
        return out

    def stats(self) -> dict:
        return dict(self._stats)

    def close(self) -> None:
        try:
            self.driver.close()
        finally:
            self.leaves.close()
            self.v1.close()
