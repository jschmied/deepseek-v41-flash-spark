"""Does the cached path fill an arena slot bit-identically to the FP4 path?"""
import os, sys, torch
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "tools"))
import cb3_moe as C3
from engine.codebook_sim import CodebookSim
from engine.cb3_cache import CB3Cache
from safetensors import safe_open

CACHE = sys.argv[1]
SHARD = "/opt/llm/models/dsv41-shards/model-00023-of-00048.safetensors"
LAYER, N = 20, 8
dev = torch.device("cuda")
sim = CodebookSim(3, dev)

ref = C3.CB3ArenaV2(N, dev); ref.sim = sim
got = C3.CB3ArenaV2(N, dev); got.sim = sim

with safe_open(SHARD, framework="pt") as f:
    for e in range(N):
        p = f"layers.{LAYER}.ffn.experts.{e}."
        ref.load_slot(e, f.get_tensor(p+"w1.weight"), f.get_tensor(p+"w1.scale"),
                         f.get_tensor(p+"w2.weight"), f.get_tensor(p+"w2.scale"),
                         f.get_tensor(p+"w3.weight"), f.get_tensor(p+"w3.scale"))

c = CB3Cache(CACHE, dev)
buf = torch.empty(c.record + 8 * 4096, dtype=torch.uint8, pin_memory=True)
addr = buf.data_ptr(); off = (-addr) % 4096
mv = memoryview(buf.numpy())
for e in range(N):
    c.read_into(mv[off:off + c.record], LAYER, e)
    c.load_slot(got, e, buf[off:off + c.record])
torch.cuda.synchronize()

bad = 0
for name in ("w1_lo","w1_hi","w1_cb","s1","w3_lo","w3_hi","w3_cb","s3","w2_lo","w2_hi","w2_cb","s2"):
    a, b = getattr(ref, name)[:N], getattr(got, name)[:N]
    eq = bool((a == b).all())
    if not eq:
        d = (a != b); bad += 1
        print(f"  {name:<7} MISMATCH  {int(d.sum())} of {a.numel()} bytes")
    else:
        print(f"  {name:<7} identical  ({a.numel():,} bytes x {N} slots)")

x = torch.randn(4, C3.DIM, dtype=torch.bfloat16, device=dev)
slots = torch.randint(0, N, (4, 6), dtype=torch.int32, device=dev)
w = torch.full((4, 6), 1/6, dtype=torch.float32, device=dev)
ya, yb = C3.moe_forward_v3(x, slots, w, ref), C3.moe_forward_v3(x, slots, w, got)
same = bool((ya == yb).all())
print(f"\n  moe_forward_v3 output bit-identical: {same}  (max |delta| {float((ya-yb).abs().max()):.1e})")
print("== ALL DONE ==" if bad == 0 and same else "== FAILED ==")
