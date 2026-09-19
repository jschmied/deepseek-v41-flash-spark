#!/usr/bin/env python3
"""Build a native on-disk CB3 expert cache for DeepSeek-V4.1-Flash.

Today an LRU miss reads 18,800,640 B of packed FP4 from the checkpoint, ships it to the GPU, and
repacks it to CB3 to fill a 14,454,784 B slot. This writes the CB3 form once, with 3-bit scales
(lossless -- notes/data/scale-survey-20260913.txt), so a miss becomes one aligned 13,774,848 B read
with no repack: 26.7% fewer bytes and less work.

Layout: record i = layer*384 + expert at offset i * 13,774,848, no index. Inside a record the planes
are in the order the arena wants them, so the loader is six slices and one scale expansion.

Layer shards are streamed from the backup server one at a time and deleted behind, so the 510 GB
checkpoint is never resident: peak local footprint is the cache plus two shards.
"""
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", default="/opt/llm/models/dsv41-cb3/experts-cb3-s3.bin")
ap.add_argument("--shard-dir", default="/opt/llm/models/dsv41-shards")
ap.add_argument("--layers", default="0-39")
ap.add_argument("--experts", type=int, default=384)
ap.add_argument("--verify-every", type=int, default=32, help="read back and compare 1 in N records")
ap.add_argument("--src", default="root@10.0.0.70:/mnt/bulk/hf/deepseek-ai--DeepSeek-V4.1-Flash")
ap.add_argument("--key", default="/home/jschmied/.ssh/id_ed25519")
ap.add_argument("--keep-shards", action="store_true")
# THE DRAFTER PACK. The DSpark draft experts live under a different prefix (mtp.{k}.ffn.experts.{e}.),
# there are 128 of them per layer rather than 384, and their shards are already local -- so the
# arithmetic shard_name(L) guess and the rsync prefetch must both be bypassed. Every default below
# reproduces the main 0-39 build byte for byte; see notes/drafter-cb3-gate.md.
ap.add_argument("--prefix-fmt", default="layers.{L}.ffn.experts.{e}.")
ap.add_argument("--index", default="", help="resolve each layer's shard from this index.json "
                                           "instead of the model-{L+3:05d} arithmetic")
a = ap.parse_args()

sys.path.insert(0, a.repo)
sys.path.insert(0, os.path.join(a.repo, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cb3_moe as C3
from engine.codebook_sim import CodebookSim
from scale_codec import pack, unpack, packed_row_bytes

DIM, INTER, SG1, SG2 = C3.DIM, C3.INTER, C3.SG1, C3.SG2
W13, W2 = (INTER, DIM), (DIM, INTER)
# plane sizes, in the order the arena wants them
PLANES = [
    ("w1_lo", INTER * (DIM // 4)), ("w1_hi", INTER * (DIM // 8)), ("w1_cb", INTER * 8),
    ("s1", INTER * packed_row_bytes(SG1)),
    ("w3_lo", INTER * (DIM // 4)), ("w3_hi", INTER * (DIM // 8)), ("w3_cb", INTER * 8),
    ("s3", INTER * packed_row_bytes(SG1)),
    ("w2_lo", DIM * (INTER // 4)), ("w2_hi", DIM * (INTER // 8)), ("w2_cb", DIM * 8),
    ("s2", DIM * packed_row_bytes(SG2)),
]
PAYLOAD = sum(n for _, n in PLANES)
ALIGN = 4096
RECORD = (PAYLOAD + ALIGN - 1) // ALIGN * ALIGN
N_LAYERS, N_EXPERTS = 40, 384
OFF = {}
o = 0
for nm, n in PLANES:
    OFF[nm] = (o, o + n); o += n

lo_l, hi_l = (int(x) for x in a.layers.split("-"))
dev = torch.device("cuda")
sim = CodebookSim(3, dev)
SSH = f"ssh -o BatchMode=yes -o StrictHostKeyChecking=no -i {a.key}"


_WMAP = json.load(open(a.index))["weight_map"] if a.index else None
FETCHED = set()


def shard_name(L):
    if _WMAP is not None:
        return _WMAP[a.prefix_fmt.format(L=L, e=0) + "w1.weight"]
    return f"model-{L+3:05d}-of-00048.safetensors"


def ensure_shard(L):
    p = os.path.join(a.shard_dir, shard_name(L))
    if os.path.exists(p):
        return p
    FETCHED.add(p)
    subprocess.run(["flock", f"/tmp/dsv41-fetch-{shard_name(L)}.lock", "-c",
                    f"[ -f {p} ] || (rsync -a --partial -e '{SSH}' {a.src}/{shard_name(L)} {p}.part "
                    f"&& mv {p}.part {p})"], check=True)
    return p


def build_layer(L, fd, f_read):
    from safetensors import safe_open
    p = ensure_shard(L)
    t0 = time.time()
    nver = 0
    with safe_open(p, framework="pt") as f:
        for e in range(a.experts):
            pre = f"layers.{L}.ffn.experts.{e}."
            rec = bytearray(RECORD)
            for tag, wname, shape, sg in (("w1", "w1", W13, SG1), ("w3", "w3", W13, SG1),
                                          ("w2", "w2", W2, SG2)):
                w = f.get_tensor(pre + wname + ".weight").view(torch.uint8).to(dev)
                s = f.get_tensor(pre + wname + ".scale").view(torch.uint8).to(dev)
                lo, hi, cb = C3.fp4_to_cb3_v2(w, s, sim)
                sp = pack(s.cpu().numpy().reshape(shape[0], sg))
                for nm, arr in ((f"{tag}_lo", lo), (f"{tag}_hi", hi), (f"{tag}_cb", cb)):
                    b0, b1 = OFF[nm]
                    v = arr.contiguous().view(torch.uint8).cpu().numpy().reshape(-1)
                    assert v.nbytes == b1 - b0, f"{nm} {v.nbytes} != {b1-b0}"
                    rec[b0:b1] = v.tobytes()
                b0, b1 = OFF[{"w1": "s1", "w3": "s3", "w2": "s2"}[tag]]
                assert sp.nbytes == b1 - b0
                rec[b0:b1] = sp.tobytes()
                # the codec must be exactly lossless, every expert, no sampling
                assert (unpack(sp, sg) == s.cpu().numpy().reshape(shape[0], sg)).all(), \
                    f"scale roundtrip failed at L{L} e{e} {tag}"
            idx = L * a.experts + e
            os.pwrite(fd, bytes(rec), idx * RECORD)
            if a.verify_every and e % a.verify_every == 0:
                back = os.pread(f_read, RECORD, idx * RECORD)
                assert back == bytes(rec), f"read-back mismatch at L{L} e{e}"
                nver += 1
    os.fsync(fd)
    if not a.keep_shards and (p in FETCHED or not a.index):
        # The main 0-39 build MUST keep deleting: 40 shards at ~7 GB against 194 GB free, so
        # reclamation is load-bearing there and its behaviour is unchanged. An --index build reads
        # shards that were already local and are not ours to delete.
        os.remove(p)
    return time.time() - t0, nver


if __name__ == "__main__":
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    total = (hi_l + 1) * a.experts * RECORD
    print(f"  payload {PAYLOAD:,} B  record {RECORD:,} B (pad {RECORD-PAYLOAD})  "
          f"full cache {total/1e9:.1f} GB")
    print(f"  planes: " + " ".join(f"{n}@{OFF[n][0]}" for n, _ in PLANES))
    fd = os.open(a.out, os.O_RDWR | os.O_CREAT)
    os.ftruncate(fd, total)
    f_read = os.open(a.out, os.O_RDONLY)
    for L in range(lo_l, hi_l + 1):
        if not a.index and L + 1 <= 39:
            subprocess.Popen(["flock", f"/tmp/dsv41-fetch-{shard_name(L+1)}.lock", "-c",
                              f"[ -f {a.shard_dir}/{shard_name(L+1)} ] || "
                              f"(rsync -a --partial -e '{SSH}' {a.src}/{shard_name(L+1)} "
                              f"{a.shard_dir}/{shard_name(L+1)}.part && "
                              f"mv {a.shard_dir}/{shard_name(L+1)}.part {a.shard_dir}/{shard_name(L+1)})"])
        dt, nver = build_layer(L, fd, f_read)
        free = os.statvfs("/").f_bavail * os.statvfs("/").f_frsize / 1e9
        print(f"  layer {L:>2}: {a.experts} experts in {dt:6.1f}s  ({dt/a.experts*1e3:5.1f} ms/expert)"
              f"  {nver} read-back verified   disk free {free:.0f} GB", flush=True)
    os.close(fd); os.close(f_read)
    man = dict(format_version=1, codec=__import__("scale_codec").CODEC, record_bytes=RECORD, payload_bytes=PAYLOAD, align=ALIGN, n_layers=hi_l - lo_l + 1,
               n_experts=a.experts, prefix_fmt=a.prefix_fmt, planes={n: OFF[n] for n, _ in PLANES},
               scale_bits=3, scale_groups={"s1": SG1, "s3": SG1, "s2": SG2},
               codebook="CodebookSim(3)", source="deepseek-ai/DeepSeek-V4.1-Flash",
               built=time.strftime("%Y-%m-%dT%H:%M:%S"), layers=[lo_l, hi_l])
    json.dump(man, open(a.out + ".json", "w"), indent=2)
    print(f"  manifest -> {a.out}.json")
    print("== ALL DONE ==")
