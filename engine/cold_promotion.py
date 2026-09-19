"""Lifecycle for the mapped cold pool and its promotion into the device hot arena.

WHY THIS IS A SEPARATE, CUDA-FREE MODULE. The measured case for executing a miss out of mapped host
memory is strong (notes/host-mapped-arena.md: -0.892 ms fill, +0.036 split, +0.102 exposed promotion,
net -0.754 ms per cold expert, all bitwise-exact). The risk is not performance, it is that a miss now
has a four-state lifetime instead of two, and every one of the transitions has a way to corrupt data
that timing usually hides. So the state machine lives here, with no torch import, and is tested on its
own -- test_cold_promotion.py.

THE LIFECYCLE

    ABSENT
      | O_DIRECT read into a reserved cold slot
    COLD_READY          the cold CB3 phase may read it; the hot slot is NOT resident
      | cold phase kernel consumes it            -> compute_done
    PROMOTING           contiguous cold -> hot copy issued on the promotion stream
      | copy event lands                          -> promo_done
    HOT_READY           only now may (layer, expert) map to the hot slot

FOUR RULES, each of which is a bug if dropped

 1. PUBLISHED IS NOT LANDED. `lru[key] = hot_slot` must not happen when the promotion is merely
    enqueued. enginev2/store.py::SlotReady already learned this the hard way; the same distinction
    applies one tier further out. Until the copy lands the expert is readable ONLY from the cold slot.

 2. THE HOT DESTINATION IS PROTECTED WHILE THE COPY IS OUTSTANDING. It holds no key yet, so nothing
    in the eviction path would otherwise keep its hands off it -- and it is exactly the slot a
    same-layer resolve would find attractive, being unmapped. `pending_hot()` is what
    ExpertStore._protect_pending must be widened with.

 3. THE COLD SLOT IS RELEASED ONLY ON compute_done AND promo_done -- a two-party handshake, whichever
    party arrives second doing the release. Relying on stream order instead is what makes the
    "promote before the cold phase" variant unsafe: there the cold record has two concurrent GPU
    readers and releasing on either one's completion corrupts the other. We issue promotion AFTER the
    cold phase for that reason (2 % slower in job 1070, and worth it), but the handshake is kept
    because it is what makes the ordering a performance choice rather than a correctness dependency.

 4. EVERY WAIT AND RELEASE IS GENERATION-TAGGED. A cold pool of 32-64 slots recycles within a few
    layers, so a completion from a previous tenant must not satisfy anything belonging to the
    current one. Readiness is recorded against the exact (slot, generation) pair -- never ">= gen",
    which SlotReady's docstring records as having been wrong in both directions.
"""
from __future__ import annotations


class ColdSlotBusy(RuntimeError):
    """No cold slot can be reserved: every one is still owed a compute or a promotion."""


class PromotionPool:
    """Bookkeeping only -- reads, kernels and copies are the caller's; this decides what is legal.

    The caller drives it in this order per miss:
        i, gen = reserve(key, hot_slot)     # after it has reserved hot_slot itself
        ... O_DIRECT into cold slot i ...
        cold_ready(i, gen)
        ... cold phase kernel reads slot i ...
        compute_done(i, gen)
        ... issue the contiguous copy, record its event ...
        promo_done(i, gen)                  # called when the EVENT LANDS, not when enqueued
    and asks `resident_key(key)` / `cold_of(key)` to decide where an expert may be read from.
    """

    def __init__(self, n_slots: int):
        if n_slots <= 0:
            raise ValueError(f"n_slots must be positive, got {n_slots}")
        self.n = n_slots
        self._gen = [0] * n_slots
        self._free = list(range(n_slots))
        # per (slot, gen): which of the two parties has arrived
        self._compute: set[tuple[int, int]] = set()
        self._promo: set[tuple[int, int]] = set()
        self._ready: set[tuple[int, int]] = set()      # bytes are in the cold slot
        # Whether THIS (slot, gen) promotion failed. It must live on the pair, not on the call that
        # happens to trigger the release: fail() usually arrives before compute_done, so the release
        # runs inside compute_done and would otherwise use its default and count a failed promotion
        # as promoted. The lifecycle tests caught exactly that.
        self._failed: set[tuple[int, int]] = set()
        # key -> (cold_slot, gen, hot_slot); present from reserve() until promotion lands
        self._inflight: dict[tuple, tuple[int, int, int]] = {}
        self._owner: dict[int, tuple] = {}             # cold slot -> key, while in flight
        self.stats = {"reserved": 0, "promoted": 0, "released": 0, "failed": 0,
                      "cold_hits": 0, "busy": 0}

    # ---------------------------------------------------------------- reservation
    def reserve(self, key: tuple, hot_slot: int) -> tuple[int, int]:
        """Take a cold slot for `key`, whose promotion destination is `hot_slot`."""
        if key in self._inflight:
            raise RuntimeError(f"{key} is already in flight; resolve() must not re-reserve it")
        if not self._free:
            self.stats["busy"] += 1
            raise ColdSlotBusy(f"all {self.n} cold slots are awaiting compute or promotion")
        i = self._free.pop()
        self._gen[i] += 1
        gen = self._gen[i]
        self._inflight[key] = (i, gen, hot_slot)
        self._owner[i] = key
        self.stats["reserved"] += 1
        return i, gen

    # ---------------------------------------------------------------- transitions
    def cold_ready(self, slot: int, gen: int) -> None:
        """The O_DIRECT read has completed: the cold phase may now read this slot."""
        self._check(slot, gen, "cold_ready")
        self._ready.add((slot, gen))

    def is_cold_ready(self, slot: int, gen: int) -> bool:
        return (slot, gen) in self._ready

    def compute_done(self, slot: int, gen: int) -> None:
        self._check(slot, gen, "compute_done")
        if (slot, gen) not in self._ready:
            raise RuntimeError(f"compute_done before cold_ready for slot {slot} gen {gen}")
        self._compute.add((slot, gen))
        self._maybe_release(slot, gen)

    def promo_done(self, slot: int, gen: int) -> None:
        """The promotion COPY HAS LANDED. Only here does the hot slot become resident."""
        self._check(slot, gen, "promo_done")
        self._promo.add((slot, gen))
        self._maybe_release(slot, gen)

    def fail(self, slot: int, gen: int) -> tuple | None:
        """Abandon this promotion. Returns the key so the caller can undo its own bookkeeping.

        The cold slot goes back only once BOTH parties are accounted for, exactly as on the success
        path: a failed promotion does not tell us the cold kernel has finished reading the slot.
        """
        self._check(slot, gen, "fail")
        key = self._owner.get(slot)
        self.stats["failed"] += 1
        self._failed.add((slot, gen))
        self._promo.add((slot, gen))
        self._maybe_release(slot, gen)
        return key

    # ---------------------------------------------------------------- queries
    def cold_of(self, key: tuple) -> tuple[int, int] | None:
        """(cold_slot, gen) if this expert must be read from the cold pool this layer."""
        e = self._inflight.get(key)
        if e is None:
            return None
        self.stats["cold_hits"] += 1
        return e[0], e[1]

    def hot_pending(self, key: tuple) -> int | None:
        """The hot slot a promotion is heading for, or None. NOT resident yet."""
        e = self._inflight.get(key)
        return None if e is None else e[2]

    def pending_hot(self) -> set[int]:
        """Hot slots with an outstanding promotion. ExpertStore._protect_pending must include these
        or a same-layer eviction can hand this destination to another expert mid-copy."""
        return {hot for (_, _, hot) in self._inflight.values()}

    def pending_cold(self) -> set[int]:
        return {i for (i, _, _) in self._inflight.values()}

    def free_slots(self) -> int:
        return len(self._free)

    # ---------------------------------------------------------------- internals
    def _check(self, slot: int, gen: int, who: str) -> None:
        if not 0 <= slot < self.n:
            raise IndexError(f"{who}: cold slot {slot} out of range 0..{self.n-1}")
        if gen != self._gen[slot]:
            # A stale completion. Dropping it is correct and must never release the slot: the
            # current tenant's own parties have not arrived.
            raise RuntimeError(f"{who}: stale generation {gen} for cold slot {slot} "
                               f"(current {self._gen[slot]})")

    def _maybe_release(self, slot: int, gen: int) -> None:
        if (slot, gen) not in self._compute or (slot, gen) not in self._promo:
            return                                    # the other party has not arrived
        key = self._owner.pop(slot, None)
        if key is not None:
            self._inflight.pop(key, None)
        promoted = (slot, gen) not in self._failed
        self._compute.discard((slot, gen))
        self._promo.discard((slot, gen))
        self._ready.discard((slot, gen))
        self._failed.discard((slot, gen))
        self._free.append(slot)
        self.stats["released"] += 1
        if promoted:
            self.stats["promoted"] += 1
