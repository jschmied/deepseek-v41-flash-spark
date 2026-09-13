#!/usr/bin/env python3
"""Assemble a model directory the engine can serve without the 296 GB of layer shards.

Contents: the per-layer dense packs (tools/dense_pack_build.py), symlinks to the shards that are
still needed whole (embeddings, head, the Engram tables and their aux), the tokenizer/config/
inference tree, and a merged `model.safetensors.index.json`.

Expert tensors keep their ORIGINAL shard filenames in the index, and those files are deliberately
NOT present: with `DSV41_CB3_CACHE` set the engine never opens them, so an absent file is a loud
failure if that ever stops being true, rather than a silent wrong answer.
"""
import argparse, json, os, shutil, sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/dsv41-lean"))
ap.add_argument("--src", default="/opt/llm/models/dsv41-shards")
ap.add_argument("--keep-shards", default="1,2,43,44,45,46,47,48",
                help="shard numbers to link whole (embeddings, head, engram, aux)")
a = ap.parse_args()

idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
wm = dict(idx["weight_map"])
dense = json.load(open(os.path.join(a.out, "dense.index.json")))["weight_map"]
for k, v in dense.items():
    wm[k] = v

keep = {f"model-{int(n):05d}-of-00048.safetensors" for n in a.keep_shards.split(",")}
linked, missing = [], []
for f in sorted(keep):
    s, d = os.path.join(a.src, f), os.path.join(a.out, f)
    if not os.path.exists(s):
        missing.append(f); continue
    if not os.path.exists(d):
        os.symlink(s, d)
    linked.append(f)

for item in ("config.json", "tokenizer.json", "tokenizer_config.json", "inference", "encoding",
             "generation_config.json"):
    s, d = os.path.join(a.src, item), os.path.join(a.out, item)
    if os.path.exists(s) and not os.path.exists(d):
        os.symlink(s, d)

idx["weight_map"] = wm
json.dump(idx, open(os.path.join(a.out, "model.safetensors.index.json"), "w"))

files = sorted({v for v in wm.values()})
present = [f for f in files if os.path.exists(os.path.join(a.out, f))]
absent = [f for f in files if not os.path.exists(os.path.join(a.out, f))]
n_exp = sum(1 for k in wm if ".ffn.experts." in k)
print(f"  {len(wm):,} tensors, {n_exp:,} of them experts")
print(f"  index names {len(files)} files: {len(present)} present, {len(absent)} absent")
print(f"  absent are the layer shards whose experts come from the CB3 cache "
      f"({len([f for f in absent if 'model-' in f])} shards)")
if missing:
    print(f"  STILL MISSING from --src, cannot link: {missing}")
nd = len([f for f in present if f.startswith('dense-')])
print(f"  present: {nd} dense packs + {len(present)-nd} whole shards")
print("== ALL DONE ==" if not missing else "== INCOMPLETE ==")
