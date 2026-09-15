"""phase1.py -- REPRODUCE v1 before evaluating anything.

(a) CACHE BEHAVIOUR, exactly. A hard gate. The skeleton's own ported policy code -- not a separate
    replay written for the occasion -- must produce, on the 25 % warm-up convention:

        lru            92.67 % hit   64.6 fetches/step   849 MiB/step
        age_over_freq  94.07 % hit   52.3 fetches/step   687 MiB/step

    If the miss stream is wrong then every timing number downstream is wrong, so this runs first
    and the rest of the package refuses to draw conclusions if it misses.

(b) TIMING SHAPE is NOT a gate, by instruction and for a good reason: 26.2 s of v1's measured 78.1 s
    block has no tracked device work and is unexplained, so a model made of calibrated sleeps can
    only hit 1.73 steps/s by carrying a fudge fitted to that target. Instead phase2.py runs the
    v1-shaped arm and REPORTS the gap as `unmodelled`.

Run:  python -m enginev2.phase1
"""

from __future__ import annotations

import sys

from .leaves import EXPERT_BYTES
from .store import ExpertSlots
from .trace import N_LAYERS, load_decode, warmup_cut

TARGETS = {
    "lru": (92.67, 64.6, 849),
    "age_over_freq": (94.07, 52.3, 687),
}


def replay(calls, policy: str, cut: int, lru_slots: int = 5328, transient_slots: int = 400):
    """Drive ExpertSlots.reserve over the decode trace; count only the scored window.

    Nothing is faked: this is the same reserve() the timing arms call, with the same victim search.
    """
    sl = ExpertSlots(lru_slots=lru_slots, transient_slots=transient_slots, policy=policy)
    for i, (L, uniq) in enumerate(calls):
        if i == cut:
            sl.hits = sl.misses = sl.prefill_misses = 0      # warm, then score
        sl.reserve(L, uniq, prefill=False)
    steps = (len(calls) - cut) / N_LAYERS
    acc = sl.hits + sl.misses
    return (100.0 * sl.hits / acc, sl.misses / steps, sl.misses * EXPERT_BYTES / steps / 2 ** 20)


def main(argv=()) -> int:
    calls = load_decode(argv[1]) if len(argv) > 1 else load_decode()
    cut = warmup_cut(calls)
    print(f"phase 1(a) -- cache behaviour, {len(calls):,} decode calls, "
          f"warm [0,{cut:,}) score the rest ({(len(calls) - cut) / N_LAYERS:.0f} steps), "
          f"5,328 LRU slots\n")
    print(f"  {'policy':16s} {'hit %':>16s} {'fetches/step':>22s} {'MiB/step':>18s}   verdict")
    ok = True
    for pol, (t_hit, t_f, t_mib) in TARGETS.items():
        hit, f, mib = replay(calls, pol, cut)
        # tolerance: the published figures are quoted to 2 / 1 / 0 decimals, so compare at the
        # precision they were published at. Anything looser would let a real regression through.
        good = (round(hit, 2) == t_hit and round(f, 1) == t_f and round(mib) == t_mib)
        ok &= good
        print(f"  {pol:16s} {hit:8.2f} (t {t_hit:5.2f}) {f:12.1f} (t {t_f:5.1f}) "
              f"{mib:10.0f} (t {t_mib:4.0f})   {'MATCH' if good else 'MISS'}")
    print()
    if ok:
        print("  phase 1(a) MATCH -- the miss stream the timing arms will be driven by is v1's.")
    else:
        print("  phase 1(a) MISS -- STOP. A skeleton that does not reproduce v1's cache cannot "
              "measure an improvement over v1.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
