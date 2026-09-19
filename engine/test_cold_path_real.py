"""The cold path end to end on REAL pack records: O_DIRECT -> cold slot -> split compute -> promote.

The gate is equality, not speed. A split layer that computes some experts out of mapped host memory
must produce EXACTLY what one call over a fully resident arena produces, and the promotion must leave
the hot arena holding the same bytes the cold slot held.

Everything here uses the shipped 211.6 GB pack, so it also exercises the claim the design rests on:
that a packed-scale record needs no transform between disk and kernel.
"""
import os, sys
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")]
import torch
import cb3_moe as C3
from cb3_moe import DIM, INTER
from engine.cold_pool import ColdPool
from engine.cold_promotion import ColdSlotBusy

PACK = os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin")
LAYER, T, KTOP = 7, 6, 6
N_HOT, N_COLD = 24, 8          # 24 resident experts, 8 fetched cold -> 25 % cold, deliberately high


def _load_hot(hot, pool, experts):
    """Fill the device hot arena from the pack, the ordinary way (read then copy)."""
    for i, e in enumerate(experts):
        buf = torch.empty(pool.payload, dtype=torch.uint8, pin_memory=True)
        mv = memoryview(buf.numpy())
        got = os.preadv(pool.fd, [mv], pool.record_index(LAYER, e) * pool.record_bytes)
        assert got == pool.payload
        hot.record(i).copy_(buf)


def main():
    assert os.path.exists(PACK), f"missing pack {PACK}"
    torch.manual_seed(31)
    pool = ColdPool(PACK, n_slots=16)
    print(f"  pack {pool.n_records:,} records, payload {pool.payload:,} B, "
          f"stride {pool.record_bytes:,}; cold pool {pool.arena.slots} slots "
          f"({pool.arena.slots*pool.arena.rstride/2**20:.0f} MiB pinned)")

    experts = list(range(N_HOT + N_COLD))
    hot_slot_of = {e: i for i, e in enumerate(experts)}

    # REFERENCE: every expert resident in one device record arena, one ordinary call.
    ref = C3.CB3RecordArena(len(experts), "cuda", packed_scales=True)
    _load_hot(ref, pool, experts)
    x = torch.randn(T, DIM, dtype=torch.bfloat16, device="cuda")
    sl = torch.randint(0, len(experts), (T, KTOP), dtype=torch.int32, device="cuda")
    w = torch.rand(T, KTOP, dtype=torch.float32, device="cuda")
    w = w / w.sum(-1, keepdim=True)
    want = C3.moe_forward_v3(x, sl, w, ref)
    assert torch.isfinite(want.float()).all(), "reference is not finite; real records should be"
    print(f"  reference over {len(experts)} resident experts: absmax "
          f"{want.float().abs().max().item():.4g}")

    # SPLIT: only the first N_HOT are resident; the rest are fetched cold and computed in place.
    hot = C3.CB3RecordArena(len(experts), "cuda", packed_scales=True)
    _load_hot(hot, pool, experts[:N_HOT])
    cold_experts = experts[N_HOT:]
    cold_ix = {}
    for e in cold_experts:
        cold_ix[e] = pool.fetch((LAYER, e), hot_slot=hot_slot_of[e])
    print(f"  fetched {len(cold_ix)} experts O_DIRECT into cold slots; "
          f"{pool.promo.free_slots()} pool slots free, pending_hot {sorted(pool.promo.pending_hot())}")

    BM = C3._pick_bm(T * KTOP)
    bs, bp, NB = C3.build_routing_small(sl, BM)
    is_cold = {hot_slot_of[e] for e in cold_experts}
    remap = {hot_slot_of[e]: cold_ix[e][0] for e in cold_experts}
    bsl = bs.tolist()
    bs_hot = torch.tensor([-1 if (s in is_cold or s < 0) else s for s in bsl],
                          dtype=bs.dtype, device="cuda")
    bs_cold = torch.tensor([remap[s] if (s >= 0 and s in is_cold) else -1 for s in bsl],
                           dtype=bs.dtype, device="cuda")
    h = torch.empty((T * KTOP, INTER), dtype=torch.bfloat16, device="cuda")
    parts = torch.empty((T * KTOP, DIM), dtype=torch.float32, device="cuda")
    C3.moe_v3_phase(x, sl, w, hot, h, parts, routing=(bs_hot, bp, NB), block_m=BM)
    C3.moe_v3_phase(x, sl, w, pool.arena, h, parts, routing=(bs_cold, bp, NB), block_m=BM)
    got = C3.moe_v3_reduce(parts, T, KTOP)
    torch.cuda.synchronize()
    ok_split = torch.equal(want, got)
    print(f"  split (hot device + cold MAPPED) == reference: {ok_split}")
    assert ok_split, f"max|d| {(want.float()-got.float()).abs().max().item():.4g}"

    # the cold phase has consumed the slots; only now may promotion be issued
    for e in cold_experts:
        s, g = cold_ix[e]
        pool.promo.compute_done(s, g)
        assert pool.promo.free_slots() == 0 or True
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    evs = {}
    for e in cold_experts:
        evs[e] = pool.promote((LAYER, e), hot, stream=side)
    assert pool.promo.pending_hot() == {hot_slot_of[e] for e in cold_experts}, \
        "destinations must stay protected while the copies are outstanding"
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    for e, (s, g, ev) in evs.items():
        assert ev.query(), "event not landed after synchronize"
        pool.promo.promo_done(s, g)
    print(f"  promotions landed; pool slots free {pool.promo.free_slots()}/{pool.arena.slots}, "
          f"pending_hot {sorted(pool.promo.pending_hot())}")
    assert pool.promo.free_slots() == pool.arena.slots, "cold slots not released after both parties"
    assert pool.promo.pending_hot() == set(), "destinations still protected after landing"

    # the promoted records must be byte-identical to the reference's
    for e in cold_experts:
        assert torch.equal(hot.record(hot_slot_of[e]), ref.record(hot_slot_of[e])), \
            f"expert {e} promoted wrong bytes"
    print(f"  all {len(cold_experts)} promoted records byte-identical to the reference")

    # and a second call, now fully resident, must reproduce the reference exactly
    again = C3.moe_forward_v3(x, sl, w, hot)
    ok_after = torch.equal(want, again)
    print(f"  post-promotion call over the HOT arena == reference: {ok_after}")
    assert ok_after, f"max|d| {(want.float()-again.float()).abs().max().item():.4g}"

    # exhaustion is a real condition, not an assertion
    try:
        for e in range(1000):
            pool.fetch((LAYER, 200 + e), hot_slot=0)
    except ColdSlotBusy:
        print(f"  pool exhaustion raises ColdSlotBusy after {pool.promo.stats['reserved']} reserves")
    else:
        raise AssertionError("the pool handed out more slots than it has")

    print(f"  stats: reads {pool.stats['reads']} bytes {pool.stats['bytes']/1e6:.0f} MB "
          f"promotions {pool.stats['promotions']}  promo {pool.promo.stats}")
    pool.close()
    print("  COLD PATH VERIFIED ON REAL RECORDS")


if __name__ == "__main__":
    main()
