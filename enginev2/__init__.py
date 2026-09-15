"""enginev2 -- a function-complete skeleton of the event-driven expert loader.

Real structure, fake leaves. Every queue, pool, buffer, semaphore, cache and event in here is real;
`preadv` and the kernels are `time.sleep`/spin calibrated to measured numbers. A real component is
swapped in later by replacing a LEAF, not by rewriting the skeleton.

  leaves.py   the calibrated fakes and the shared-bandwidth NVMe model. No leaf is ever tuned to
              make a total come out right -- that is the arena sizer's 0.82 mistake.
  store.py    slot reservation, LRU + transient ring, the two eviction policies, staging leases,
              per-slot readiness. Eviction is PORTED from engine/experts.py because it is
              measured-good.
  trace.py    the captured route trace, which is the fake leaf standing in for the model's router.
  sched.py    the continuous loader service and the five independently togglable dependencies.
  drivers.py  request input -> chunked prefill and stepped decode.
  phase1.py   REPRODUCE v1's cache exactly before evaluating anything. A hard gate.
  phase2.py   the per-dependency contribution table and the unmodelled gap.
  test_v2_invariants.py
              the semantics engine/test_io_path.py pins for v1, re-asserted here.

Order of use:  python -m enginev2.phase1  ->  python -m enginev2.test_v2_invariants
               ->  python -m enginev2.phase2
"""
