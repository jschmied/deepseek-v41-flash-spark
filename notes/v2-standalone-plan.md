# A standalone v2: what it needs, in the order it needs it

**Written 2026-09-17** after the user asked for a v2 that does not depend on v1, reusing what fits.
Status: plan + the first item in progress. Nothing here is a claim about performance.

## The finding that sets the scope

**The HTTP layer does not need porting.** `server/app.py` (1,153 lines) talks to an engine through
`server/engine_api.py`, and that file is a **three-method abstract class**:

```python
class Engine(ABC):
    def generate(self, prompt_ids, *, max_tokens, temperature, top_p,
                 stop_token_ids, seed, grammar=None, ...) -> Iterator[list[int]]
    def stats(self) -> dict
    def close(self) -> None
```

plus `supports_grammar` and a two-call gate contract (`gate.observe(ids)`,
`gate.mask_rows(logits, block_ids)`).

Every item on the list — `/v1/chat/completions`, SSE streaming, `/v1/completions`, chat-template
rendering, thinking/reasoning, tools and DSML parsing, xgrammar constraints, stop strings and EOS,
usage accounting, `/health`, `/v1/models` — is **already implemented in `app.py` against that ABC**,
and `MockEngine` proves the HTTP half runs with no GPU at all. So the work is not eleven endpoints.
It is **one adapter class**, `V2Engine(Engine)`, and the engine capabilities its `generate` needs.

That reframing is the whole point of writing this down: the endpoint list looks like the work and
is not.

## What `generate()` actually requires, and what v2 has

| requirement | v2 today |
|---|---|
| prefill a real prompt | **implemented 2026-09-17** (`RealLeaves.begin_prefill/prefill_attn/prefill_moe/finish_prefill`); gate = job 536, unpushed until green |
| greedy decode with DSpark draft/verify/rollback | present (`select_block`, `end_step`) |
| **temperature / top-p sampling** | **missing** — `RealLeaves.temperature` is stored and never used; the first token and every bonus token are `argmax` |
| **non-greedy verification** | **missing** — `end_step` accepts on `am.eq(self.drafts)`, which is greedy-only. `fd.draft()` already returns the draft distribution `q` and v2 throws it away (`self.drafts, _q = ...`) |
| stop ids | partial (`stop_ids` stored, not enforced in the loop) |
| `max_tokens`, burst yielding, generator close/drain | missing — there is no `generate()` at all, only `decode(steps)` |
| grammar gate | missing |
| `stats()` | counters exist (`Counters`, `HostPhases`, provider stats); no dict in the server's shape |

## Order of work, and why this order

1. **Non-greedy verification + rollback.** Deepest item and everything above it depends on the
   sampling contract. v1's reference is `engine/v41_engine.py:1028-1051`: rejection sampling against
   the draft distribution with a residual `resid / resid.sum()` for the bonus token, not a greedy
   comparison. v2 must match it or sampled output is silently a different distribution — and that
   is the kind of defect no throughput test can see.
2. **Temperature + top-p.** `sample_probs(logits, temperature, top_p)` (v41_engine.py:279) is the
   reference and is 6 lines; the work is threading it through `select_block`, the bonus token and
   the first token, not writing it.
3. **Finish the prefill driver.** Job 536 must be green first. Then: the engram source is still
   wired for decode only, and `prefill_chunked` is driven layer-by-layer by the caller rather than
   by a prompt-level entry point.
4. **`V2Engine(Engine)`** — the adapter. Wraps 1-3 in the ABC's shape: burst yielding, stop ids,
   `max_tokens`, the `try/finally` the contract requires for cache/arena cleanup on `GeneratorExit`,
   `stats()` in the server's dict shape.
5. **Grammar gate.** `supports_grammar = True` plus the two calls, which is what unlocks tools and
   xgrammar for free — `tool_grammar.py` (974 lines) is engine-agnostic.
6. **Run `app.py --engine v2`** against the existing server tests (`server/test_server.py`,
   `test_tool_grammar.py`, `test_tool_repair.py` — 1,339 lines already written).

## What "not depending on v1" does and does not mean

`RealLeaves` calls `block_attn`, `block_ffn_in`, `moe_route`, `moe_apply`, `hc_post`, `engram_forward`
and the CB3 arena. **Those are the model, not v1's scheduler**, and reimplementing them would produce
a different model rather than an independent engine. The dependency worth removing is on v1's
*control*: `V41Engine._decode_loop`, `_generate`, its prefill branch, its stats assembly. That is
what items 1-4 replace. Keeping the leaf math shared is the reuse the request asks for, and the
bitwise gates (jobs 520, 536) are what keep it honest.

## Branch convergence, and the standing decision on upstream

**Decided 2026-09-17.** All four of our branches sit on `main`;
`feat/enginev2-realleaves` already CONTAINS `one-spark-full-router` (merged in for the P0 fix), and
`merge/upstream-server` touches only `server/`, which the serving line never modified. So ours
converge cleanly. The order, each step with the gate that has to be met first:

1. **`merge/upstream-server` -> `one-spark-full-router`.** Independent of the engine; green on CPU
   (16 + 29 + repair, MockEngine). Gate: a decision on `DSV41_TOOL_GRAMMAR` 0 -> 1. That is a
   serving default we held pending our own end-to-end run, and it is not a merge question.
2. **`feat/enginev2-realleaves` -> `one-spark-full-router`.** Gate: job 555 green, because this
   branch carries changes to v1 FILES -- `warm_start=False` in `v41_engine.py`, the arena repoint in
   `model.py` and `fastdecode.py` -- into the engine that serves. Plus `V2Engine` having actually
   served a request: today its prefill half is bitwise-gated and its API half has only CPU fakes.
3. `one-spark-full-router` stays the single line. `main` stays the stale fork point and is never
   pushed (memory `push-own-repos-standing-go`).

### Upstream is NEVER merged. Cherry-pick only.

`0xBakeer/deepseek-v41-flash-spark` is 101 commits ahead of our fork point. Five were the
endpoint/template fixes, taken wholesale into `merge/upstream-server` because `server/` is
engine-agnostic and upstream was a strict superset there. **The remaining ~96 are engine commits and
must not arrive as a merge.**

The reason is a hazard we have already written down (memory `merge-hazard-bypassed-kernel`):
**upstream fusing an op into a kernel we bypass merges CLEANLY and is silently wrong.** v2 bypasses
v1's `_resolve`, v1's prefill loop and v1's expert store -- precisely the surfaces upstream is most
likely to have rewritten. A merge would report success, change nothing we can see, and invalidate
measurements that took a day each to produce.

Procedure for taking something from upstream:

* Survey by AREA, not by reading 96 commits: `git log upstream/main --oneline -- <path>`.
* **Diff the file, never trust the commit list.** `git log base..head -- <path>` reports 0 commits
  while `git diff` shows the change, because of history simplification. That has cost us once.
* Check whether upstream is a SUPERSET of ours for that path before taking it wholesale --
  `git diff ours upstream -- <path>` and read the `-` lines, which are the ones of ours that would
  be lost. For `server/` those turned out to be older versions of the same code; for anything in
  `engine/` assume they are not.
* `git cherry-pick -x <sha>` so the origin is recorded in the message.
* Gate by area: `server/` -> the three CPU suites; `engine/` -> the bitwise gates (jobs 520, 536,
  555). An engine cherry-pick with no bitwise gate does not land.
