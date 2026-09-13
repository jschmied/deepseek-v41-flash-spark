#!/usr/bin/env python3
"""Extract the non-expert tensors of every layer into a small pack, so the engine can run without
the 296 GB of layer shards.

With a native CB3 expert cache (tools/cb3_cache_build.py) the engine never reads an expert from the
checkpoint -- but `engine/model.py::Weights` still loads attention, shared-expert, gate and
hyper-connection weights by name through the safetensors index, and those live inside the same 7.4 GB
layer shards. They are only 181 MB per layer, and measured on the real files they form exactly TWO
contiguous runs, so they can be pulled by byte range instead of streaming the whole shard: 7.2 GB and
about a minute of LAN, against 296 GB and 48 minutes.

Writes one safetensors file per layer, byte-for-byte the original payloads with the original dtype
strings -- the file is assembled directly rather than round-tripped through torch, so no dtype
mapping can go wrong -- plus a rewritten `model.safetensors.index.json` for a lean model directory.
"""
import argparse, json, os, struct, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/dsv41-lean"))
ap.add_argument("--layers", default="0-39")
ap.add_argument("--host", default="root@10.0.0.70")
ap.add_argument("--key", default=os.path.expanduser("~/.ssh/id_ed25519"))
ap.add_argument("--dir", default="/mnt/bulk/hf/deepseek-ai--DeepSeek-V4.1-Flash")
ap.add_argument("--local-shard-dir", default="/opt/llm/models/dsv41-shards")
a = ap.parse_args()
lo, hi = (int(x) for x in a.layers.split("-"))
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-i", a.key, a.host]


def rd(path, off, n, local=None):
    if local and os.path.exists(local):
        with open(local, "rb") as f:
            f.seek(off)
            b = f.read(n)
        if len(b) != n:
            raise IOError(f"short local read {len(b)}/{n}")
        return b
    p = subprocess.run(SSH + ["dd", f"if={path}", "bs=4M", f"skip={off}", f"count={n}",
                              "iflag=skip_bytes,count_bytes", "status=none"],
                       stdout=subprocess.PIPE, check=True)
    if len(p.stdout) != n:
        raise IOError(f"short remote read {len(p.stdout)}/{n} at {off}")
    return p.stdout


def header(path, local):
    n = struct.unpack("<Q", rd(path, 0, 8, local))[0]
    return json.loads(rd(path, 8, n, local)), 8 + n


def write_safetensors(path, entries, payloads):
    """entries: [(name, dtype, shape, nbytes)] in order; payloads: matching bytes."""
    hdr, off = {}, 0
    for (name, dt, shape, nb) in entries:
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
        off += nb
    blob = json.dumps(hdr, separators=(",", ":")).encode()
    pad = (-(8 + len(blob))) % 8                       # safetensors wants the payload 8-aligned
    blob += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for p in payloads:
            f.write(p)
    return 8 + len(blob) + off


if __name__ == "__main__":
    os.makedirs(a.out, exist_ok=True)
    wm = {}
    total = 0
    for L in range(lo, hi + 1):
        sh = f"model-{L+3:05d}-of-00048.safetensors"
        remote = f"{a.dir}/{sh}"
        local = os.path.join(a.local_shard_dir, sh)
        hdr, base = header(remote, local)
        dense = sorted((base + v["data_offsets"][0], base + v["data_offsets"][1], k, v)
                       for k, v in hdr.items()
                       if k != "__metadata__" and ".ffn.experts." not in k)
        runs = []
        cur = [dense[0][0], dense[0][1]]
        for s, t, _, _ in dense[1:]:
            if s == cur[1]:
                cur[1] = t
            else:
                runs.append(tuple(cur)); cur = [s, t]
        runs.append(tuple(cur))
        buf = {}
        for s, t in runs:
            buf[s] = rd(remote, s, t - s, local)
        def slice_of(s, t):
            for rs, rt in runs:
                if rs <= s and t <= rt:
                    return buf[rs][s - rs:t - rs]
            raise KeyError((s, t))
        entries = [(k, v["dtype"], v["shape"], t - s) for s, t, k, v in dense]
        payloads = [slice_of(s, t) for s, t, _, _ in dense]
        name = f"dense-{L:02d}.safetensors"
        n = write_safetensors(os.path.join(a.out, name), entries, payloads)
        for k, _, _, _ in entries:
            wm[k] = name
        total += n
        print(f"  layer {L:>2}: {len(entries):>3} tensors, {len(runs)} runs, "
              f"{n/1e6:7.1f} MB -> {name}", flush=True)
    json.dump({"metadata": {"note": "dense pack; expert tensors served from the CB3 cache"},
               "weight_map": wm}, open(os.path.join(a.out, "dense.index.json"), "w"), indent=1)
    print(f"\n  {total/1e9:.2f} GB total -> {a.out}")
    print("== ALL DONE ==")
