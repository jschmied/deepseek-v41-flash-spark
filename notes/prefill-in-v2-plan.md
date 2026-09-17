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

**That comment is about a chunk, and this note read it as if it were about a prompt. Measured, it
is wrong -- but so was the correction that first replaced it. Job 510, 2026-09-17, v1 engine,
13,200 tokens of NON-REPETITIVE text in 4 chunks, arena warmed by 150 decode steps:**

| model | predicted misses | predicted bytes | |
|---|---|---|---|
| per-CHUNK (`chunks x layers x 384`) | 59,200 | 815 GB | refuted, 3.3x too high |
| per-LAYER (`layers x 384`, dedup in the ring) | <= 15,360 | <= 212 GB | **also exceeded** |
| **measured** | **17,926** (+ 9,089 LRU hits) | **246.9 GB** | 448.1 misses per layer, of 384 |

So the 1.2 TB figure this note was built on is gone -- job 500's own timing had already made it
impossible, since 815 GB at the device's ~6.3 GB/s is 129 s of pure I/O and that prefill finished in
92.5 s -- but the clean per-layer bound does not hold either. **448 misses per layer against 384
experts means some experts are read more than once within a layer**, so the transient ring is not
retaining a layer's union the way the argument assumed.

The mechanism behind that argument is real and visible in the code: `reserve(layer, uniq,
prefill=True)` checks `self.lru` and then `self.transient_map` before allocating
(`enginev2/store.py:717-731`), `prefill_chunked` reserves every chunk of a layer up front and
submits only `to_load` (`enginev2/drivers.py:857-867`), and the ring defaults to 400 slots against
384 experts. **But job 510 measured v1's `engine/model.py` prefill, not v2's `prefill_chunked`** --
v2's cannot be measured at all yet, because `RealLeaves.prefill_attn` / `prefill_moe` do not exist.
Whether the bound holds for v2's driver is therefore still open; what is settled is that it does
not hold for the engine we serve with. Dedup is partial: 27,015 expert requests against the 59,200
a no-dedup path would issue, so v1 already avoids ~55 % of them and re-reads the rest.

**And capacity is not irrelevant: 9,089 of those requests were LRU hits** -- 9,089 reads, ~125 GB,
that the prompt did not have to issue because decode had left them resident. That is a third of the
total request stream. The "capacity is irrelevant / residency ~0 / resident-first is irrelevant"
paragraph below was asserted, never measured, and is withdrawn.

Contrast with what we measured at decode (jobs 460/465/475):

| | decode | prefill |
|---|---|---|
| unique experts per layer | 21.7 | ~370 of 384 per CHUNK; 448 MISSES per layer measured |
| residency | 93-95 % | 9,089 LRU hits of 27,015 requests = 34 % (job 510) |
| what the arena buys | +10 % (79 -> 86 GB) | ~125 GB of reads not issued (job 510) |
| bytes | 262 MB / token | 246.9 GB for a 13.2k prompt (job 510, v1, real text) |

What survives, and what does not:

- **Eviction policy is probably still irrelevant** -- within a layer nothing needs to be evicted
  (384 fits in 400), and across layers the previous layer's entries are genuinely dead. This one
  the original argument gets right, for a better reason than it gave.
- **Capacity is NOT irrelevant -- measured.** 9,089 of 27,015 expert requests were LRU hits, ~125 GB
  of reads the prompt did not issue because decode had left them resident.
- **The resident-first split is NOT obviously irrelevant** for the same reason: "essentially
  nothing is resident" was asserted, and a third of the request stream is resident.
- **Prediction is still worthless.** The next chunk needs ~all 384 experts of the next layer. This
  one does not depend on the byte model.

**The open question is now narrower and sharper: where do the re-reads come from?** 448 misses per
layer against 384 experts is not a rounding error, and the ring is big enough on paper. Per-chunk
and per-layer instrumentation -- new `to_load`, transient hits, LRU hits, union of distinct experts
per layer -- is what would localise it, and it has to be taken on v1's prefill, since v2's does not
exist. That is the measurement this plan should start from, not a design.

The part of this note that survives unchanged: prefill is still a different problem from decode,
and it should allow a much greater read depth.

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
