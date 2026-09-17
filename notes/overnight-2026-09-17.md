# Overnight 2026-09-17 — where this stands and what is queued

## The one blocking defect

**v2 decode is not reproducible across requests; v1 is.** Same prompt, greedy, temperature 0, no
RNG anywhere. Request 1 gives one answer, request 2 another, and BOTH are individually reproducible
— so it is deterministic state carried between requests, not a race.

Retired, each with the job that did it:

| hypothesis | retired by |
|---|---|
| RNG / seeding | 565 (seeded 1-token identical; greedy also diverges) |
| the accept/commit path | 575 (`a_n` and `c_len` identical across runs) |
| the drafter | 575 (`drafts` identical) |
| first-use graph capture | 575 (warm graphs, still diverges) |
| an order-dependent expert reduce | reasoned: only the current layer has reads outstanding and `await_copies` puts every event on the compute stream before graph B |
| early H2D visibility | 580 (`DSV41_V2_EARLY_READY=0` changes nothing) |
| graph parity | `parity = S % 2`, and every request prefills to the same S |
| the model-cache reset | 590 (metadata arm token-identical to control) |

Shape (585): **A B B′ B″** — a one-time transition after the first generation, plus a smaller effect
that reaches only the last token. At 7 tokens three consecutive generations are identical.

**Live suspect: v2's expert-cache SEMANTICS.** Not persistence — v1 persists its store too
(`_reset()` never touches it) and stays reproducible. The concrete divergence is that on a decode
hit in the transient ring v1 calls `_promote_transient()` (experts.py:636, fired at :733), moving
the expert into the LRU and swapping a donor back; v2's `reserve()` takes the hit and leaves it.

## What is queued, in order

1. **595 transient-ring-cold** — drop only the ring's bookkeeping between requests (safe: forces
   re-reads, creates no stale mappings), LRU and every byte untouched. Compares a prefix well short
   of `max_tokens`, because `V2Engine` truncates the emitted burst *after* `RealLeaves` has committed
   the whole speculative burst, so a request can execute hidden tokens and perturb residency by an
   amount that depends on acceptance length — the likely source of the B′/B″ tail, and a separate
   defect worth fixing on its own.
2. **600 http-smoke** — `--engine v2` through `server/app.py`. Wired in review round 2, never once
   exercised. Independent of the determinism thread.
3. **605 lru-order-and-generations** — if the ring is not the carrier: LRU *order* and per-slot
   *generations* are both functional and both invisible to 590's `sorted((layer, expert, slot))`
   digest. Records all three separately per request. Skip if 595 lands.
4. **610 policy-v1-vs-v2** — every "v2" figure so far ran `Policy()`, which is all-True, i.e. **V1
   semantics**. The all-false `sched.V2` policy that v2 exists to enable has never been measured end
   to end.

## Rules that earned their place today

- **Do not edit a tree a running job imports.** Cost two void runs. It is not only `jobs/*.sh` — a
  multi-file source edit has a window even when each file is individually valid.
- **Check `state/<job>.done` before assuming a requeue will run.** A stale `.done` silently skipped
  a requeue and nearly had me read an old log as fresh.
- **An anchored string edit can land in the wrong scope.** `self.accepted = []` matched the line in
  `attach()`, so a constructor "fix" changed nothing. Verify the edit landed where intended.
- **A restore that does not restore the bytes is not an experiment.** 590's expert arm crashed on
  the engine's own slot-collision assert; it was uninterpretable before it ran.
