"""Stage 6: the decode A/B. One arm per process, both from the same warm-started engine.

WHY ONE PROCESS PER ARM. Both arms share one physical arena. Whichever runs second inherits the
other's residency, so running them back to back in one process is not an A/B -- it is a warm arm
against a cold one. Each arm therefore starts from its own engine load, warm-started identically.

WHAT IS MEASURED, all from the engines' OWN counters (never /proc/diskstats):
  steps/s          wall time over the timed decode loop
  experts read     v1: store.stats['loads']; v2: leaves' own read counter
  GB read          reads x 13,774,848 B, the CB3 record
  per-read GB/s    bytes / time INSIDE the read leaf. That time is summed over threads, so this
                   is the rate one read sees, NOT the device aggregate. Both are reported: the
                   aggregate is bytes / wall, which is the number a bandwidth trade needs.
  hit rate         resident / (resident + miss) over UNIQUE experts per layer per step -- counted
                   the same way in both arms, because v1's store counts uniques and the route
                   tensor has 36 positions for ~11.7 uniques, so the two are not interchangeable

WHAT IS NOT MEASURED. v2 has no draft/verify. The block is TEACHER-FORCED from a fixed corpus --
identical in both arms -- so the route sequence is realistic and reproducible, but acceptance is
not modelled and these steps/s are not v1's serving steps/s. The A/B is
between two DRIVERS over one fixed route sequence, which is the only thing it can honestly claim.

    ARM=v1 python enginev2/ab_stage6_decode.py [steps]
    ARM=v2 python enginev2/ab_stage6_decode.py [steps]
"""
import os, sys, time, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# THIS checkout -- see RealLeaves.ROOT in enginev2/real.py. This used to be a hardcoded
# ~/git/deepseek-v41-flash-spark: a DIFFERENT working tree of the SAME repo, on a different
# branch, whose engine/ silently shadowed this one. Every real-graph number this branch
# produced came from code that was not committed here.
V1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V1)
sys.path.insert(0, os.path.join(V1, "tools"))
os.chdir(V1)

from engine.v41_engine import V41Engine              # noqa: E402
from enginev2.real import RealEngramSource, RealLeaves   # noqa: E402
from enginev2.sched import Policy                    # noqa: E402

if "ENGRAM" not in os.environ:
    raise RuntimeError(
        "set ENGRAM=1 (the production model) or ENGRAM=0 (ablated) explicitly -- this benchmark "
        "measured the ablated model for a whole day because zeroed eg_rows were the quiet default")
ENGRAM = os.environ["ENGRAM"] == "1"
# EVICTION POLICY IS A CHOICE, NOT A DEFAULT. Every v2 measurement so far hardcoded "lru", and the
# box has already measured age/(1+count) ahead of it: decode 1.88/2.13/2.14 tok/s against LRU's
# 1.74/1.85/1.86 (job 140), and offline 94.07 % hit / 52.3 fetches per step against 92.67 % / 64.6
# (phase 1). Running the A/B on the worse policy is a fair comparison at the wrong operating point.
EVICT = os.environ.get("EVICT", "lru")
ARM = os.environ.get("ARM", "v2")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
RECORD = 13_774_848

eng = V41Engine(os.path.expanduser("~/dsv41-lean"), max_seq=8192,
                arena_gb=float(os.environ.get("ARENA_GB", 40)),
                spec=True, expert_format="cb3")
ids = eng.tokenizer.encode(
    "Explain how an NVMe SSD controller schedules writes, and why the flash translation "
    "layer matters for tail latency under a mixed read/write workload.", add_special_tokens=False)
m, fd = eng.model, eng.fast
m.begin_prompt()
logits, mh = m.forward(torch.tensor(ids, dtype=torch.long, device="cuda"), 0,
                       prefill=True, need_logits=True)
if eng.spec:
    m.dspark_seed(mh, 0)
T = fd.ids.numel()


# TEACHER-FORCED ON REAL TEXT, not greedy argmax.
#
# The greedy rule collapsed: measured token sequences went [0, 5, 223, 5180, 201, 201, 201, 201]
# and stayed there. A repeating token re-routes to the same experts every step, so job 350 saw a
# 98.24 % hit rate and 6.5 reads per step against production's 0.894 and ~64 -- a tenfold
# difference. It was measuring a fixed point, not a model generating text.
#
# Feeding real tokens keeps the route sequence realistic AND perfectly reproducible, which the
# oracle's recorded trace requires. The model still computes everything; only the choice of the
# next block changes, and it changes identically in every arm.
_CORPUS = None


def _corpus(tokenizer, need):
    global _CORPUS
    if _CORPUS is None:
        text = (
            "The flash translation layer maps logical block addresses to physical pages, and its "
            "garbage collector decides when to relocate live data. Under a mixed read/write "
            "workload the collector competes with host reads for the same channels, which is why "
            "tail latency rises sharply once the over-provisioned region is exhausted.\n\n"
            "def merge_intervals(intervals):\n"
            "    intervals.sort(key=lambda p: p[0])\n"
            "    out = []\n"
            "    for start, end in intervals:\n"
            "        if out and start <= out[-1][1]:\n"
            "            out[-1][1] = max(out[-1][1], end)\n"
            "        else:\n"
            "            out.append([start, end])\n"
            "    return out\n\n"
            "In a mixture-of-experts transformer each token is routed to a small subset of the "
            "feed-forward experts, so the memory traffic of a decode step depends on the routing "
            "distribution rather than on the parameter count alone.\n\n"
            "Die Wettervorhersage fuer die kommende Woche zeigt einen deutlichen Temperatur"
            "rueckgang, begleitet von anhaltenden Niederschlaegen im Alpenvorland.\n\n"
        ) * 60
        _CORPUS = tokenizer.encode(text, add_special_tokens=False)
    assert len(_CORPUS) >= need, f"corpus has {len(_CORPUS)} tokens, need {need}"
    return _CORPUS


def next_block(lg, step):
    """The SAME rule in both arms: the next T real tokens. Deterministic, and not a fixed point."""
    c = _corpus(eng.tokenizer, (step + 2) * T)
    return torch.tensor(c[step * T:(step + 1) * T], dtype=torch.long, device="cuda")


block0 = next_block(logits, 0)
st0 = dict(eng.store.stats)

if ARM == "v1":
    blk = block0
    layer_ids = tuple(eng.args.engram_layer_ids)

    def eg_futs(blk_, pos_):
        """v1's own engram path: hash, D2H the ids before any graph is queued, submit both tables."""
        h_np = m.hash_state(blk_[None], pos_)[0].cpu().numpy()
        return {L: (eng.eg_pool.submit(eng.tables[L].read_raw, h_np[:, li, :]),
                    eng.tables[L].to_device)
                for li, L in enumerate(layer_ids)}

    t0 = time.perf_counter()
    for step in range(STEPS):
        pos = m.c.len
        rows = (lambda _f=eg_futs(blk, pos): _f) if ENGRAM else {}
        lg, _ = fd.step(blk, pos, rows)
        blk = next_block(lg, step + 1)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    st = eng.store.stats
    reads = st["loads"] - st0["loads"]
    read_s = st["read_s"] - st0["read_s"]
    h2d_s = st["h2d_s"] - st0["h2d_s"]
    hits = st["hits"] - st0["hits"]
    misses = st["misses"] - st0["misses"]
    staging = eng.store.pool._max_workers if hasattr(eng.store, "pool") else -1
else:
    from enginev2 import drivers as v2drivers        # noqa: E402
    v2drivers.N_LAYERS = eng.args.n_layers
    rl = RealLeaves(os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin"), eng.arena)
    rl.attach(eng, block0, next_block=next_block)
    eg_src = RealEngramSource(eng, rl) if ENGRAM else None
    rl.engram = eg_src
    # Count lookups the way v1's store does -- unique experts per layer -- or the hit rates are
    # two different quantities printed under one name.
    uniq_lookups = [0]
    _la = rl.layer_a
    def layer_a(L, _f=_la, _n=uniq_lookups):
        r = _f(L)
        _n[0] += len(r.uniq)
        return r
    rl.layer_a = layer_a
    staging = int(os.environ.get("V2_STAGING", 8))
    # Policy() is V1: all four barriers ON. That arm isolates the driver rewrite from the policy
    # change and is the control, NOT the v2 arm. POLICY=v2 releases them.
    pol = Policy() if os.environ.get("POLICY", "v1") == "v1" else Policy(False, False, False, False)
    e2 = v2drivers.Engine(pol, evict=EVICT,
                          lru_slots=eng.store.n_slots - 8, transient_slots=8,
                          n_workers=int(os.environ.get("V2_WORKERS", 8)), staging=staging,
                          expert_read_qd=int(os.environ.get("V2_QD", 8)),
                          h2d_inflight=int(os.environ.get("V2_H2D", 2)), leaves=rl,
                          engram=eg_src)
    # SEED v2's slot table from v1's, so both arms start from the same physical residency.
    # Without this v2 faults in 2,367 experts v1 already has and the A/B measures the warm start.
    for k, slot in eng.store.lru.items():
        e2.slots.lru[k] = slot
        e2.slots.slot_key[slot] = k
        e2.slots.gen[slot] = e2.slots.gen.get(slot, 0)
        e2.slots.evict.on_insert(k, slot, 0)
    e2.slots.free_lru = [s for s in e2.slots.free_lru if s not in e2.slots.slot_key]
    seeded = len(e2.slots.lru)
    t0 = time.perf_counter()
    c = e2.decode(STEPS)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    # h2d_s from the LOADER, not the provider: the provider only enqueues the copy now, so its
    # own clock would measure the enqueue and report H2D as free.
    reads, read_s, h2d_s = rl.read_bytes // RECORD, rl.read_s, e2.loader.h2d_s
    misses = c.fetches
    hits = uniq_lookups[0] - misses
    staging = e2.loader.stage.peak_in_use
    print(f"  seeded {seeded} resident keys from v1's store")
    e2.close(); rl.close()

gb = reads * RECORD / 1e9
if ARM != "v1":
    print(f"  policy {pol.name}: resolve_blocks={pol.resolve_blocks} "
          f"compute_barrier_global={pol.compute_barrier_global} "
          f"lease_until_completion={pol.lease_until_completion} "
          f"global_barrier={pol.global_barrier}")
print(f"  engram {'LIVE' if ENGRAM else 'ABLATED'}  evict {EVICT}  "
      f"arena slots {eng.store.n_slots}  v1 staging {eng.store.io_threads} buffers")
print(f"ARM {ARM}  steps {STEPS}  wall {wall:.2f}s  {STEPS / wall:.3f} steps/s")
print(f"  experts read {reads}  {gb:.2f} GB  read_s {read_s:.2f}  "
      f"per-read {gb / read_s if read_s else 0:.2f} GB/s  "
      f"achieved {gb / wall:.2f} GB/s  h2d_s {h2d_s:.2f}")
print(f"  lookups: {hits} resident, {misses} miss  hit rate "
      f"{hits / (hits + misses) * 100 if hits + misses else 0:.2f}%  staging peak {staging}")
