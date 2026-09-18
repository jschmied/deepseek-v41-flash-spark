# Ring lending in v2, and where the prefill memory actually goes

Written 2026-09-18 while job 705 holds both trees. **No code edited yet** — the rule is that the
source tree a running job imports is off limits, so this is the plan and job 710 is the measurement
that decides half of it.

## 1. Porting `DSV41_RING_TO_LRU` to v2

v1: `engine/experts.py:467` `lend_ring_to_lru()` / `reclaim_ring()`, called from
`engine/v41_engine.py:895` right after `_snap_prefill()`. Worth +12 % decode and −16.6 %/−14.4 %
NVMe (commit `f7ee09f`), off by default.

`enginev2/store.py:559` `ExpertSlots` carries the same six structures under the same names — `lru`,
`slot_key`, `free_lru`, `transient_ring`, `transient_pos`, `transient_map` — so the body of both
methods transfers almost verbatim. Three things are v2-only and a verbatim copy gets them wrong:

1. **`_pending` is a dict, not a set.** v1 skips `self._pending_slots`; v2's equivalent is
   `pending_slots()`, which takes `_pending_lk`. Lending must not hand out a slot whose write is in
   flight, and it must read that set under the lock.
2. **`_displaced` must be purged for every lent slot.** A speculative reservation that displaced a
   resident can be rolled back at zero cost because the old tenant's bytes are still there
   (`rollback_speculative`). If the slot has since been lent to the LRU and rewritten by decode,
   that rollback restores a key onto bytes that are now some other expert. v1 has no speculation
   and therefore no such record, which is exactly why the copy is not safe.
3. **`gen` handles reuse, `transient_map` does not.** v1's lend deletes the stale `transient_map`
   entries so a decode `reserve()` cannot count one as a hit on a slot the LRU is about to reuse.
   v2 needs the same deletion; the generation counter protects a *consumer* from a stale tenant, it
   does not stop the hit being counted.

Call site: **after** `finish_decoder_replay`, not before. Since the root-cause fix the replay runs
layers `src+1..39` through v2's own arena at `prefill=False`, so its reservations come from the LRU
region — lending earlier would hand out slots the replay is still using.

Then `reclaim_ring()` at the top of the next `_prefill`, and a v2 copy of
`engine/test_ring_lending.py` (it already tests the double-lend and the take-back).

## 2. Bigger prefills without spending memory

Three levers, cheapest first. Only the first costs nothing.

### (a) A bigger ring is free once lending exists

`TRANSIENT_SLOTS=400` (`.env:11`) against **384 experts per layer** — `experts.py:171` notes a
prefill chunk can touch all 384. That is a 16-slot margin. Those 400 slots are dead weight through
the whole decode phase, which is what lending fixes; but the same fact means the ring can be *much*
bigger at zero decode cost, because lending gives every one of them back. A 1200-slot ring costs
the arena 11.6 GB during prefill and nothing at all during decode.

This is the honest answer to "bigger prefill without spending memory": it does not buy memory, it
re-labels bytes that were already ours and were idle half the time. It is also the general form of
the earlier "small arena during prefill, big during decode" question — the arena can never be
*resized*, because the twelve slot-major base pointers are baked into the decode CUDA graphs.
Partition, don't realloc.

### (b) The 20.5 GB reserve is not expert bytes — it is one materialised tensor

`v41_engine.py:491` reserves `MAX_CHUNK * 5e6`, and that is why job 700 got a 68.8 GB arena instead
of 86. `model.py:36-45` says what the number stands for: "The ceiling is activation memory: at
T=2048 the gathered window+compressed KV of one layer is ~2.7 GB." And `model.py:105-114`, on
`DSV41_ATTN_GATHER_FUSED`:

> the gathered kv_all is 1.34 GB at T=2048 and ~2.7 GB at its construction peak, **which is the
> ceiling MAX_CHUNK is written against** … the ring and the compressed cache are 4 MB tables every
> query shares, where kv_all is 1.34 GB streamed once.

So the flag that unblocks bigger chunks already exists and is off. It also removes 2.15 s of
`CatArrayBatchedCopy*` and 1.31 s of `vectorized_gather_kernel` from a 40 s prefill, and the kernel
itself is 5.65 ms against 27.23 ms.

Two costs, both bounded:

- **The ring must hold `window_size + MAX_CHUNK` positions** (`model.py:114`): `RING >= 8320` for a
  8192 chunk. `RING` is `[RING, 5120] bf16 × 40 layers` plus 3 MTP rings ⇒ 4096 costs 1.68 GB and
  8448 costs 3.47 GB. **+1.8 GB to double the chunk**, against a reserve term of 20.5 GB and an
  NVMe saving of half the prefill's expert traffic (`model.py:36`: traffic ≈ chunks × layers × 384,
  because even a 512-token chunk already touches ~370 of 384).
- **`ATTN_GATHER_FUSED` requires `ATTN_FUSED_PREFILL`, which is not bit-identical to eager** and
  still owes the paired-NLL verdict. Gather fusion *is* bit-identical to fused — it only
  re-addresses rows — so the memory lever is free once the fusion itself is accepted, and worthless
  if it is not. Job 710 arm C gates the narrow bitwise claim; the NLL verdict is a separate job.

### (c) The reserve is an inherited constant we have never checked

`v41_engine.py:487` is explicit that `5e6`/token is 0xBakeer's measurement (`122350f`, their box),
adopted **without** their companion auto-sizer change. If our own peak is materially below 5 MB the
arena grows with no fusion, no ring change and no risk at all. That is the cheapest outcome on the
table and it is one measurement. Job 710 arms A and E measure it.

## Order

1. job 710 (queued) — measure (b) and (c); no code change, env flags only.
2. port lending to v2 (§1) once 705 releases the tree, with the three v2-only fixes and a test.
3. ring-size sweep (a) — only meaningful after 2, since it is lending that makes it free.

---

## Corrections from review, 2026-09-18

**"A bigger ring is free" is withdrawn.** Two reasons, both right:

1. **It frees no memory.** The arena tensors stay fully allocated; lending repartitions ownership
   inside memory we already hold. So it cannot enable a larger prefill activation allocation, which
   is how I presented it in the "bigger prefills" section. Removed from there.
2. **400 already exceeds the per-layer expert universe (384).** For the current layer-at-a-time
   prefill, more than 400 simultaneously live transient experts is not something the algorithm can
   use. A bigger ring only helps if some future multi-layer prefill or prefetch needs it.

There is also a **cross-request cost I missed entirely**: during decode, lent ring slots fill with
hot LRU entries; `reclaim_ring()` at the next request forcibly drops every one of them. A 1200-slot
ring means dropping up to 1200 decode residents at *every* new prefill. Steady-state decode inside
one request can look excellent while first-token and early-decode cache state degrade each request —
and every ring measurement so far reads steady-state decode only. Any ring sizing test must run
**multiple consecutive real requests** and report prefill NVMe, the first ~10 decode steps, later
steady decode, and residents dropped by `reclaim_ring()`, separately.

**`_displaced` must be handled at RECLAIM, not at lend.** I had the lifecycle point wrong. The
danger is not the initial lend: during decode a lent ring slot becomes an LRU slot and can acquire a
speculative `_displaced` record, and at the next request reclaim wants that physical slot back.
Deleting the record is not enough if the speculative operation is still outstanding — a later
cancellation can still call `rollback_speculative()` or `forget()` against a slot whose role has
changed underneath it. The contract before `reclaim_ring()` must be: settle or cancel speculative
attempts targeting ring slots, quiesce the loader, `drain_forgets`, assert no pending and no
speculatively-owned ring slots, reclaim, then purge dead `_displaced` records. Prefetch is a no-op in
today's `V2Engine`, so this cannot bite serving yet — but `ExpertSlots` is built for speculation and
should not acquire a lifecycle bug just because the current server never exercises it.
