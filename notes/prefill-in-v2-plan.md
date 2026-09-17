# Implementing prefill in engine v2 — what it is and is not

**Status: plan only. Nothing implemented.** Written 2026-09-17 so the survey is not redone.

## What already exists

- `drivers.py::prefill_chunked(layer, chunks, ctx)` — a real chunked driver. It routes and submits
  **every chunk up front**, then waits per chunk, so `global_barrier` is an isolated toggle. (An
  earlier revision submitted lazily in one arm and eagerly in the other, so D3 was confounded with
  *when reads were issued*; fixed in review 2026-09-15.)
- `ExpertSlots.reserve(layer, uniq, prefill=True)` — a real transient ring beside the LRU region,
  `transient_slots` configurable. The slot layer needs nothing.
- `ModelLeaves.prefill_attn` / `prefill_moe` — calibrated sleeps.

## What is missing

**`RealLeaves` has no prefill methods at all.** `grep -n prefill enginev2/real.py` returns nothing.
That is the whole gap: the driver and the slot layer are ready, the provider is not.

## The thing that makes prefill a DIFFERENT problem, not a bigger decode

From `engine/model.py`, on `MAX_CHUNK`:

> a prefill chunk streams nearly every expert of every layer through the transient ring whatever its
> length (a 512-token chunk already touches **~370 of 384**), so the NVMe traffic of a prompt is
> ~chunks x layers x 384 experts

Contrast with what we measured at decode (jobs 460/465/475):

| | decode | prefill |
|---|---|---|
| unique experts per layer | 21.7 | ~370 of 384 |
| residency | 93-95 % | ~0 — it streams |
| what the arena buys | +10 % (79 -> 86 GB) | nothing; it does not fit |
| bytes | 262 MB / token | ~chunks x 40 x 370 x 13.77 MB |

A 24k-token prompt at chunk 4096 is 6 x 40 x ~370 x 13.77 MB ~ **1.2 TB** of reads. That is why
prefill is tens of seconds and why every decode conclusion transfers badly:

- **Eviction policy is irrelevant.** `age_over_freq` won +8 % at decode by keeping hot experts. In
  prefill every expert is touched once per chunk per layer; there is no hot set.
- **Capacity is irrelevant.** The lever that dominates decode does nothing here.
- **The resident-first MoE split is irrelevant.** It moves work that depends on already-resident
  experts; at prefill essentially nothing is resident.
- **Prediction is trivially perfect and worthless.** The next chunk needs ~all 384 experts of the
  next layer. You do not need to predict that, and knowing it buys nothing, because the constraint
  is bytes, not knowledge.

## Where prefill time actually goes, per the engine's own notes

> Two of the three largest prefill costs scale with the **CHUNK COUNT**, not the token count — the
> per-chunk kernel launches, and the **CB3 unpack**, which re-unpacks nearly the same ~362 experts
> for every chunk of a layer.

So the levers are:

1. **Fewer, larger chunks.** Already exploited: 2048 -> 4096 measured -4.7 s on a 12,624-token
   prefill, byte-identical. 8192 is better on paper and the pre-flight refuses it, because the
   reserve is `MAX_CHUNK * 5e6` = 41 GB at 8192. **This is the same reserve job 485 is measuring.**
   If the reserve is larger than the true high-water mark, chunk 8192 may become affordable — which
   would be a prefill win obtained from the decode-side memory work, and is the one real coupling
   between the two.
2. **The CB3 unpack.** `moe_forward_prefill` unpacks CB3 -> an FP4 scratch arena per call
   (`arena.fp4_scratch(batch)`, `DSV41_CB3_SCRATCH_SLOTS`). Layer-major ordering already exists to
   amortise it; `notes/cb3-chunk-invariance.md` and the `layer-major-*` jobs cover it.
3. **Read depth.** Prefill is the shape where the pipe SHOULD be full — unlike decode, where 44-84 %
   of the span has zero reads in flight because only ~2 misses per layer are knowable at a time. At
   prefill every expert of the layer is knowable at once. **This is the one place v2's loader could
   plausibly beat v1**, and it is the reason to implement prefill in v2 at all.

## What `RealLeaves` would have to provide

```
prefill_attn(layer)              -> model.py attention for this chunk, on the real tensors
prefill_moe(layer, slots, chunks) -> C3.moe_forward_prefill over the transient-ring slots
```

Both need chunk state the decode path does not carry: the chunk's `x`, its position range, the
compressor/window KV checkpoints (`Caches.rollback` uses per-chunk inputs), and the engram rows for
layers 1 and 14. `RealLeaves.attach()` currently binds decode buffers only.

Note `moe_forward_prefill` takes `slots` as arena slot ids and does its own `torch.unique` + batched
unpack into the FP4 scratch, so the provider hands it the transient-ring slots and does not
replicate the unpack.

## Order of work, and the honest reason for it

1. **Wait for job 485.** If the prefill reserve is reclaimable, chunk 8192 may be reachable, and that
   changes the shape of everything below. It is also the only prefill lever with a measured payoff
   path today.
2. **Then `RealLeaves.prefill_attn` / `prefill_moe`**, gated the way the shared-expert split was:
   bitwise against v1 on the same prompt before any timing is quoted.
3. **Then the only question worth the build:** does v2's loader keep the NVMe pipe fuller than v1's
   `join_pending()` barrier during prefill? Decode said host scheduling is worth ~1 % because the
   pipe is idle for structural reasons. Prefill has no such excuse, so this is where the async
   loader should show its value if it has any.

**Do not start at step 2.** The decode work repeatedly produced numbers that had to be withdrawn
because the arm was not physically realizable or the weight was wrong; prefill has a measured cost
model already written down above, and the first job should test it rather than assume it.
