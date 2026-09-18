# The experiment pipeline is the bottleneck, not the box

2026-09-18. Review's verdict, and it is right: turnaround is limited by how experiments are
submitted, not by DGX throughput. Today four queue slots went to jobs whose errors were static or
compile-time, and several more to hidden pipeline failures. The scripts live in `~/ds41-queue/`
(never committed -- runners carry credentials); this records the design and why each piece exists.

## `submit.sh` -- the only way into the queue

**Preflight is mandatory.** Submission is impossible when it fails.

**The job runs from an immutable worktree at a captured SHA.** `git worktree add --detach` into
`~/ds41-worktrees/<job>/{repo,v2}`, pinned to the SHA both branches had at submission, `.env` copied
in (it is gitignored and carries `DSV41_CB3_CACHE`, without which the store falls back to
safetensors shards that were deleted to reclaim disk), payloads snapshotted, a manifest printed into
the job log. Tracked files are 5.8 MB, so a snapshot is free -- the repo's 2.6 GB is untracked
results. Last 8 kept, older pruned.

This removes a whole class of failure seen today and lifts the rule that caused others:

- `runner.sh` was edited while bash was still parsing it and died at a shifted byte offset.
- Two runners once ran the same job concurrently, each measuring memory against the other's resident
  engine -- arms saw 50 GB MemAvailable instead of ~110 and the sizer silently capped their arenas.
- The standing rule "the source tree a running job imports is off limits" blocked useful work for
  the length of every experiment. **It no longer applies**: the main checkout is free to edit while a
  job runs, because the job is not reading it.

Jobs address repos as `$JOB_REPO` and `$JOB_V2`; preflight **rejects any job that names a live
checkout**, so this cannot regress by habit.

`runner.sh` is deliberately untouched -- the queued file is a generated wrapper. Editing the thing
that runs everything, while it runs, is the failure this change exists to prevent.

## `preflight.sh` -- everything that can fail without a GPU fails here

| gate | catches |
|---|---|
| `bash -n` | shell syntax |
| no live-checkout paths | jobs reading a mutable tree |
| `py_compile` every named payload | Python syntax |
| **@triton.jit payloads must declare a compile gate** | job 855: a module global inside `@jit` is *valid Python*, so `py_compile` passes it; only the Triton frontend rejects it |
| heredoc python compiled; `@triton.jit` in a heredoc rejected outright | job 800 -- Triton reads the defining source file |
| `MODEL_DIR` / `DSV41_CB3_CACHE` exist | a start that dies 80 s in on a missing input |
| `# PREFLIGHT: <cmd>` lines must exit 0 | anything job-specific, e.g. compiling the exact target kernel with its real constexprs |

`# PREFLIGHT:` lines are exempt from the live-checkout scan: they run at submit time, before the
worktree exists.

Self-test: preflight rejects job **855** as submitted, naming both its live-checkout reference and
its missing Triton gate. That job cost a real queue slot and measured nothing.

## Still to build, from the same review

- ~~A permanent decode-kernel fixture~~ **BUILT**: `payloads/kernel_fixture.py`. One invocation,
  fail-fast in increasing cost -- compile metadata (regs/spills/shared), then bitwise, then latency,
  aborting at the first failure so a kernel that spills is never benchmarked and one that changes
  the answer is never timed. Shapes are the engine's: T=1 (MTP single token), T=6 (~12 distinct, the
  common decode case), T=6 high-distinct (worst cache case), T=24. `--packed` compares a
  packed-scale arena against an unpacked one; the default self-comparison proves the fixture itself
  is sound. Scale data is generated representably (base + 0-7), since random bytes would make it a
  test of the generator -- which is how job 795 was wasted.
- **A prefill-memory stress fixture** -- `payloads/prefill_stress.py` written, **not yet
  correlated**. Runs one full chunk of the worst shape through the real forward for a few layers and
  reports `torch peak - post-load baseline`, which is the transient the arena cannot have. It
  deliberately does NOT decide on host MemAvailable: the caching allocator does not return freed
  blocks to the OS, so that number is the process lifetime maximum -- the error that made jobs
  720/735/745 conclude prefill cost was flat in chunk size when job 750 showed it is not. It is a
  PREDICTOR: it must be correlated once against a real 26.4k run before any decision rests on it,
  and a configuration it accepts still gets one real confirmation. What it buys is rejecting unsafe
  settings in seconds instead of ~75 s, so the arena/scratch/chunk search can be wide.
- **State-based stopping** instead of fixed repetitions: warm until hit rate and NVMe/token move less
  than a threshold over two consecutive windows, then measure two paired windows. Job 715 needed six
  reps because it was still climbing; that does not make six reps right everywhere. A +16 % effect
  against ~1-2 % same-config noise can stop early.
- **Gate pyramid, no level skipped**: static/domain -> unit/invariant -> exact Triton compile -> GPU
  microbenchmark -> short real-engine correctness -> representative performance -> long-prefill and
  memory safety. A 26.4k prompt is the LAST thing to run, only for claims that depend on it. Kernel
  register pressure, codec validity, bitwise arithmetic and most decode latency never need it.
- **One bundle per question, fail-fast**: compile metadata, bitwise, T=6 and T=24 timing belong in
  one script that aborts at the first failed gate, not four queue jobs.
