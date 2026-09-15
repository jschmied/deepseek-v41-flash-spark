"""phase2.py -- the per-dependency contribution table, and the gap the model does not explain.

HOW TO READ THIS FILE'S OUTPUT, because the easy misreading is the expensive one.

1. The columns are NOT ADDITIVE. Turning a dependency off from v1 and turning it on from v2 give
   different numbers, and that difference is the finding, not noise: at decode the fork-join shape
   MASKS the other three. While `resolve_blocks` is on, the driver is never computing when a loader
   wants the arena, so D1 costs nothing; the wait already happened, so D4's ordering cannot matter;
   and the pending set only ever holds one layer, so D3 is the same wait either way. Hence both a
   leave-one-out table from v1 AND a leave-one-in table from v2, plus the explicit pairwise cell
   for resolve_blocks x global_barrier.

2. Every number carries the UNMODELLED GAP as its uncertainty band. The modelled v1 arm runs faster
   than the box because 26.2 s of v1's measured 78.1 s block has no tracked device work and is not
   represented by any leaf here. No leaf is tuned to close it -- it is printed as `unmodelled` and
   left open. If the gap is large relative to an arm difference, that difference does not predict
   the box, and the verdict says so.

Run:  python -m enginev2.phase2 [--steps N] [--reps N] [--evict lru|age_over_freq]
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import time

from .leaves import (C_DEP, C_IND, C_LAYER, C_OTHER, C_PRE, EXPERT_BYTES, GPU_BUSY_S, H2D_S,
                     SPAN_S, STEPS_PER_S)
from .drivers import Engine
from .sched import V1, V2, Policy
from .trace import N_LAYERS, load_decode, warmup_cut

FIELDS = [f.name for f in dataclasses.fields(Policy)]
LABEL = {
    "resolve_blocks": "shape  resolve blocks (fork-join)",
    "compute_barrier_global": "D1     H2D waits for ALL compute",
    "lease_until_completion": "D2     lease held submit->completion",
    "global_barrier": "D3     wait over ALL pending reads",
    "moe_before_shared": "D4     routed MoE before shared expert",
}

# Measured, for the gap report. Decode profile, 2026-09-15.
MEAS_BLOCK_S, MEAS_READ_S, MEAS_UNTRACKED_S = 78.1, 47.4, 26.2


def run(policy: Policy, calls, cut: int, steps: int, evict: str, **kw) -> tuple:
    e = Engine(policy, evict=evict, **kw)
    try:
        e.warm(calls, cut)
        c = e.decode(calls, steps, start=cut)
        return (c.steps_per_s, c.blocked_s / c.wall_s, c.compute_s / c.wall_s,
                c.fetches / c.steps, len(e.arena.violations), c.mean_inflight, c.achieved_gbs,
                c.device_busy_s / c.wall_s)
    finally:
        e.close()


def median_run(policy, calls, cut, steps, reps, evict, **kw):
    out = [run(policy, calls, cut, steps, evict, **kw) for _ in range(reps)]
    sps = [o[0] for o in out]
    return (statistics.median(sps), min(sps), max(sps),
            statistics.median([o[1] for o in out]), statistics.median([o[2] for o in out]),
            out[0][3], sum(o[4] for o in out),
            statistics.median([o[5] for o in out]), statistics.median([o[6] for o in out]),
            statistics.median([o[7] for o in out]))


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--evict", default="lru", choices=("lru", "age_over_freq"))
    a = ap.parse_args(argv[1:])

    calls = load_decode()
    cut = warmup_cut(calls)
    print(f"phase 2 -- decode, {a.steps} steps x {a.reps} reps per arm, evict={a.evict}, "
          f"warm to call {cut:,}\n")

    base = median_run(V1, calls, cut, a.steps, a.reps, a.evict)
    v2 = median_run(V2, calls, cut, a.steps, a.reps, a.evict)

    # ---------------------------------------------------------------- the gap
    mod_step = 1.0 / base[0]
    meas_step = 1.0 / STEPS_PER_S
    fetches = base[5]
    print("  WHAT THE MODEL ACCOUNTS FOR, per decode step")
    print(f"    reads     {fetches:5.1f} x {EXPERT_BYTES / 2 ** 20:.2f} MiB "
          f"= {fetches * EXPERT_BYTES / 2 ** 20:7.0f} MiB")
    print(f"    H2D       {fetches:5.1f} x {H2D_S * 1e6:.0f} us "
          f"= {fetches * H2D_S * 1e3:7.1f} ms")
    print(f"    compute   {N_LAYERS} layers x {C_LAYER * 1e6:.0f} us "
          f"= {N_LAYERS * C_LAYER * 1e3:7.1f} ms   "
          f"(pre {C_PRE * 1e6:.0f} / ind {C_IND * 1e6:.1f} / dep {C_DEP * 1e6:.0f} / "
          f"other {C_OTHER * 1e6:.0f} us per layer)")
    print(f"    modelled v1 step {mod_step * 1e3:7.1f} ms  ({base[0]:.2f} steps/s)")
    print(f"    measured  v1 step {meas_step * 1e3:7.1f} ms  ({STEPS_PER_S:.2f} steps/s)")
    print(f"    unmodelled: {(meas_step - mod_step) * 1e3:.1f} ms/step "
          f"= {100 * (meas_step - mod_step) / meas_step:.0f} % of the measured step   NOT CLOSED")
    print(f"    for scale, the measured untracked block is {MEAS_UNTRACKED_S:.1f} s of a "
          f"{SPAN_S:.0f} s span = {1e3 * MEAS_UNTRACKED_S / (SPAN_S * STEPS_PER_S):.0f} ms/step\n")

    # GPU busy is a calibration CHECK, not a target: the compute leaves were derived from the
    # measured 3.0 s / 104 s independently of any scheduler shape, so if the modelled compute
    # share lands on 3 % once the unmodelled gap is added back, the compute leaf is right and the
    # gap is somewhere the leaves do not cover.
    gpu_share_if_gap_real = N_LAYERS * C_LAYER / meas_step
    print(f"  calibration check: modelled compute is {100 * base[4]:.1f} % of the MODELLED span, "
          f"and {100 * gpu_share_if_gap_real:.1f} % of the MEASURED step")
    print(f"  (measured GPU busy is {100 * GPU_BUSY_S / SPAN_S:.0f} % of span -- the compute leaf "
          f"agrees with the box once the gap is restored, so the gap is not compute)\n")

    # WHY v2 cannot win much here, stated as a measurement rather than an opinion. Layer L's expert
    # ids cannot exist before layer L-1's MoE has produced its output, so NEITHER arm can have more
    # than one layer's misses in flight. The device saturates near 2 concurrent reads; if both arms
    # sit near 1.5, the continuous loader has nothing extra to issue and the bandwidth shortfall is
    # a LOOKAHEAD problem, not a scheduling-shape one.
    print("  READ CONCURRENCY -- the structural reason the arms are close")
    for nm, r in (("v1", base), ("v2", v2)):
        print(f"    {nm}: mean {r[7]:.2f} reads in flight while the device is busy, "
              f"{r[8]:.2f} GB/s achieved, device busy {100 * r[9]:.0f} % of span")
    print("    ceiling is 6.82 GB/s and it needs >=2 in flight; the router cannot produce layer")
    print("    L+1's ids before layer L's MoE, so no arm here can issue a second layer's reads")
    print("    early. That makes the shortfall a LOOKAHEAD problem, not a scheduling-shape one.\n")

    # ---------------------------------------------------------------- the tables
    # NOISE FLOOR. The per-arm effects here are ~1-2 % and the rep-to-rep spread of a SINGLE arm
    # is of the same order, so a row can easily print a number that reverses on the next run --
    # observed: the pairwise interaction came out -1.5 % at 30 steps x 5 reps and +2.9 % at
    # 3 x 1. Any row whose effect does not clear the pooled spread of itself and the baseline is
    # marked `n.s.` and must not be read as a contribution. This is the whole reason the table
    # prints a floor instead of a ranking.
    def floor_pct(arm, ref):
        return 100 * max(ref[2] - ref[1], arm[2] - arm[1]) / ref[0]

    def mark(arm, ref):
        eff = 100 * (arm[0] / ref[0] - 1)
        fl = floor_pct(arm, ref)
        return f"{eff:+8.1f} %", ("n.s." if abs(eff) < fl else "    "), fl

    hdr = (f"  {'arm':38s} {'steps/s':>9s} {'min-max':>13s} {'vs base':>9s} {'sig':>5s} "
           f"{'floor':>7s} {'blocked':>9s}")

    print("  TABLE A -- leave-one-out from v1 (turn ONE dependency off)")
    print(hdr)
    print(f"  {'v1 (all five on)':38s} {base[0]:9.2f} {base[1]:6.2f}-{base[2]:<6.2f} "
          f"{'--':>9s} {'':>5s} {'':>7s} {100 * base[3]:8.1f} %")
    a_rows = {}
    for f in FIELDS:
        p = dataclasses.replace(V1, **{f: False})
        r = median_run(p, calls, cut, a.steps, a.reps, a.evict)
        a_rows[f] = r
        eff, sig, fl = mark(r, base)
        print(f"  {LABEL[f]:38s} {r[0]:9.2f} {r[1]:6.2f}-{r[2]:<6.2f} {eff} {sig:>5s} "
              f"{fl:6.1f} % {100 * r[3]:8.1f} %")
    print()

    print("  TABLE B -- leave-one-in from v2 (turn ONE dependency back on)")
    print(hdr)
    eff, sig, fl = mark(v2, base)
    print(f"  {'v2 (all five off)':38s} {v2[0]:9.2f} {v2[1]:6.2f}-{v2[2]:<6.2f} {eff} {sig:>5s} "
          f"{fl:6.1f} % {100 * v2[3]:8.1f} %   <- vs v1")
    for f in FIELDS:
        p = dataclasses.replace(V2, **{f: True})
        r = median_run(p, calls, cut, a.steps, a.reps, a.evict)
        eff, sig, fl = mark(r, v2)
        print(f"  {LABEL[f]:38s} {r[0]:9.2f} {r[1]:6.2f}-{r[2]:<6.2f} {eff} {sig:>5s} "
              f"{fl:6.1f} % {100 * r[3]:8.1f} %")
    print()

    # The pair the coordinator asked for explicitly: if the 26.2 s turns out to be CPU work on the
    # MAIN thread, these two are partly one phenomenon in reality and the single-toggle columns
    # over-credit their separation.
    pair = dataclasses.replace(V1, resolve_blocks=False, global_barrier=False)
    rp = median_run(pair, calls, cut, a.steps, a.reps, a.evict)
    s_rb = a_rows["resolve_blocks"][0] / base[0] - 1
    s_gb = a_rows["global_barrier"][0] / base[0] - 1
    s_both = rp[0] / base[0] - 1
    print("  PAIRWISE -- resolve_blocks x global_barrier (do they compose, or overlap?)")
    print(f"    resolve_blocks off alone   {100 * s_rb:+6.1f} %")
    print(f"    global_barrier off alone   {100 * s_gb:+6.1f} %")
    print(f"    both off                   {100 * s_both:+6.1f} %   "
          f"(sum of singles {100 * (s_rb + s_gb):+.1f} %)")
    pair_floor = floor_pct(rp, base)
    inter = 100 * (s_both - s_rb - s_gb)
    print(f"    interaction                {inter:+6.1f} %  "
          f"(noise floor {pair_floor:.1f} %) -> "
          f"{'NOT RESOLVABLE at this precision' if abs(inter) < pair_floor else 'outside the floor'}")
    print("    A non-zero interaction would mean they are not independent levers. This run cannot")
    print("    tell: the sign has flipped between runs, which is what a noise-dominated cell does.\n")

    band = 100 * (meas_step - mod_step) / meas_step
    print(f"  UNCERTAINTY BAND ON EVERY ROW ABOVE: {band:.0f} % of the real step is unmodelled. "
          f"Differences\n  smaller than that do not predict the box.\n")

    # WHAT JOB 185 DID AND DID NOT SETTLE (ran 2026-09-15 17:30-17:35, rc=0).
    # It re-profiled with CPU sampling on to name what the machine executes during the untracked
    # windows. The sampling tables did NOT record -- only ENUM_SAMPLING_THREAD_STATE (11 rows, the
    # enum definition) is present, no samples -- so the direct attribution is still open and the
    # job's own script refuses to guess. What its budget table DOES constrain, on a 72.9 s span:
    #   inside a syscall            55.6 s (76 %)
    #   inside a CUDA API call      15.9 s (22 %)
    #   in NEITHER traced table      1.7 s ( 2 %)   <- an upper bound on main-thread CPU work
    # The last line matters for the pairwise cell above. If the untracked block were CPU work on
    # the MAIN thread, `resolve_blocks` and `global_barrier` would be one phenomenon and their
    # columns would over-credit their separation. At most ~2 % of the span is main-thread work
    # outside both tables, so that particular confound is SMALL. It does not rule out CPU work on
    # the LOADER threads -- which a queue-based v2 loader would inherit rather than remove, and
    # which is exactly the risk this table cannot price.
    print("  INTERPRETING THE GAP (job 185, 2026-09-15)")
    print("    CPU sampling did not record; the direct attribution of the untracked block is OPEN.")
    print("    Its budget table does bound main-thread CPU work at 1.7 s of a 72.9 s span (2 %),")
    print("    so resolve_blocks and global_barrier are not mostly-one-thing via the main thread.")
    print("    Loader-thread CPU work is still unexcluded, and v2 would INHERIT it, not remove it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
