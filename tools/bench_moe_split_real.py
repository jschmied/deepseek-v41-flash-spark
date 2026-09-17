"""How much of graph B can actually move ahead of the expert reads?

THE QUESTION. Graph B is 69.4 ms/step of device time (jobs 400/405). Pair residency is 94-95 % and
unique-expert residency ~83 % (job 475). Those two weights price different things:

    unique residency  ->  WEIGHT BYTES, and the CB3 kernel is bandwidth-bound at decode
    pair residency    ->  arithmetic per (token, expert) pair

`build_routing_small` emits one BM-block per DISTINCT slot, so an expert streams its 13.77 MB once
per launch whether it serves one pair or four. So unique should bind, and pair residency should
OVERSTATE the movable fraction -- resident experts are the popular ones and carry more pairs each
than the tail misses. This measures the only quantity that settles it: the kernel's own time on the
exact resident/missing partition.

    all       the baseline: one launch over every block
    resident  what would run BEFORE the wait
    missing   what would run AFTER it
    split     resident + missing, i.e. what the two-phase path actually costs

REAL ROUTES, REAL WEIGHTS. The previous attempt used randperm-per-token, which gave 36 distinct
experts for 36 pairs where decode has ~11.6 -- the exact property the answer turns on. This replays
routes captured from real decode steps (DSV41_ROUTE_DUMP) against records read out of the real CB3
cache file, so both the routing skew and the weight bytes are the engine's.

    python tools/bench_moe_split_real.py <routes.pkl> [n_layer_visits]
"""
import os, pickle, statistics as st, sys, time, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cb3_moe as C3                                            # noqa: E402
from fp4_moe import DIM, _pick_bm, build_routing_small          # noqa: E402
from engine.cb3_cache import CB3Cache                           # noqa: E402

ROUTES = sys.argv[1]
# Default kept small: every DISTINCT (layer, expert) across the sampled visits needs an arena slot
# at 14.45 MB, so 40 visits is ~6 GB while 200 would be ~33 GB.
N_VISITS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
visits = pickle.load(open(ROUTES, "rb"))
print(f"  {len(visits)} layer-visits captured; using {min(N_VISITS, len(visits))}")

cache = CB3Cache(os.path.expanduser(os.environ.get(
    "DSV41_CB3_CACHE", "~/dsv41-cb3/experts-cb3-s3.bin")))
# One arena slot per distinct expert we will touch, filled from the real cache file.
need = sorted({(L, e) for L, flat, _m in visits[:N_VISITS] for e in set(flat)})
print(f"  {len(need)} distinct (layer, expert) records to stage")
arena = C3.CB3ArenaV2(len(need), "cuda")
# read_into() is O_DIRECT, so the destination must be 4096-ALIGNED. A plain pinned tensor is not
# guaranteed to be; over-allocate and take the aligned slice. The same slice is handed to
# load_slot(), which wants a torch uint8 CPU tensor.
_raw = torch.empty(cache.record + 4096, dtype=torch.uint8, device="cpu", pin_memory=True)
_off = (-_raw.data_ptr()) % 4096
buf = _raw[_off:_off + cache.record]
assert buf.data_ptr() % 4096 == 0, "staging buffer is not page-aligned; O_DIRECT will EINVAL"
mv = memoryview(buf.numpy())
slot_of = {}
for i, (L, e) in enumerate(need):
    cache.read_into(mv, L, e)
    cache.load_slot(arena, i, buf)
    slot_of[(L, e)] = i
torch.cuda.synchronize()
print("  arena staged from the real cache file")

def timeit(fn, n=30):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return st.median(ts)

tot = {"all": 0.0, "resident": 0.0, "missing": 0.0, "split": 0.0}
pairs_res = pairs_tot = uniq_res = uniq_tot = 0
for L, flat, miss in visits[:N_VISITS]:
    K = 6
    T = len(flat) // K
    if T == 0:
        continue
    slots = torch.tensor([[slot_of[(L, e)] for e in flat[t * K:(t + 1) * K]] for t in range(T)],
                         dtype=torch.int32, device="cuda")
    x = (torch.randn(T, DIM, device="cuda") * 0.5).to(torch.bfloat16)
    wgt = torch.rand(T, K, device="cuda")
    BM = _pick_bm(T * K)
    bslot, bpair, NB = build_routing_small(slots, BM)
    missing = {slot_of[(L, e)] for e in miss}
    blk_mis = torch.zeros_like(bslot, dtype=torch.bool)
    for i, sv in enumerate(bslot.tolist()):
        if sv >= 0 and sv in missing:
            blk_mis[i] = True
    bs_res = torch.where(blk_mis, torch.full_like(bslot, -1), bslot)
    bs_mis = torch.where(blk_mis, bslot, torch.full_like(bslot, -1))

    f_all = lambda: C3.moe_forward_v3_split(x, slots, wgt, arena, block_slot=bslot,   # noqa: E731
                                            block_pair=bpair, NB=NB, BM=BM, phase_masks=(bslot,))
    f_res = lambda: C3.moe_forward_v3_split(x, slots, wgt, arena, block_slot=bslot,   # noqa: E731
                                            block_pair=bpair, NB=NB, BM=BM, phase_masks=(bs_res,))
    f_mis = lambda: C3.moe_forward_v3_split(x, slots, wgt, arena, block_slot=bslot,   # noqa: E731
                                            block_pair=bpair, NB=NB, BM=BM, phase_masks=(bs_mis,))
    f_spl = lambda: C3.moe_forward_v3_split(x, slots, wgt, arena, block_slot=bslot,   # noqa: E731
                                            block_pair=bpair, NB=NB, BM=BM,
                                            phase_masks=(bs_res, bs_mis))
    tot["all"] += timeit(f_all); tot["resident"] += timeit(f_res)
    tot["missing"] += timeit(f_mis); tot["split"] += timeit(f_spl)
    u = set(flat); pairs_tot += len(flat); uniq_tot += len(u)
    pairs_res += sum(1 for e in flat if e not in set(miss)); uniq_res += len(u - set(miss))

n = min(N_VISITS, len(visits))
print(f"\n  per layer-visit, median of 30, over {n} real visits:")
for k in ("all", "resident", "missing", "split"):
    print(f"    {k:<9} {tot[k] / n:7.3f} ms")
print(f"\n  PAIR residency   {pairs_res / pairs_tot * 100:.1f} %   <- arithmetic weight")
print(f"  UNIQUE residency {uniq_res / uniq_tot * 100:.1f} %   <- weight-bytes weight")
print(f"  MEASURED movable {tot['resident'] / tot['all'] * 100:.1f} %   <- the only one that counts")
print(f"  split overhead   {(tot['split'] / tot['all'] - 1) * 100:+.1f} % vs one launch")
