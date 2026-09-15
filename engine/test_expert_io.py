"""
test_expert_io.py -- the O_DIRECT expert reader must return exactly the checkpoint's bytes.

`ExpertStore` reads an expert as two contiguous file runs cut into `DSV41_READ_CHUNK_MB` aligned
pieces that several threads issue in parallel (the arithmetic around alignment, the run tail that
may reach past EOF, and the offsets of the six tensors inside the staging buffer are all easy to
get subtly wrong, and a wrong expert shows up only as slightly worse text). This compares the
reader's bytes against `safetensors.safe_open` for a few experts at several chunk sizes.

Run on the box (it needs the real checkpoint; ~250 MB of pinned buffers, safe next to the server):

    python -m engine.test_expert_io

The routed-expert shards this test needs (raw safetensors, not the CB3 cache) were deleted for
layers 0-39 when the box moved to the CB3 cache (see .env); only layer 9 (model-00012) is still
on disk, at /opt/llm/models/dsv41-shards. That is what this runs against by default -- MODEL_DIR
itself (dsv41-lean) no longer has a routed-expert shard to compare.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from engine import experts as EX  # noqa: E402
from engine import _testenv as ET  # noqa: E402


class _StubArena:
    """`read_expert` never touches the arena; only `.slots` is read at construction."""

    slots = 1024


RAW_SHARDS_DIR = "/opt/llm/models/dsv41-shards"  # last routed-expert shard left on this box (layer 9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None,
                     help="default: MODEL_DIR (env/.env) if it holds the shard the test cases need, "
                          "else " + RAW_SHARDS_DIR)
    ap.add_argument("--chunks-mb", default="0,1,4,20", help="DSV41_READ_CHUNK_MB values to test (0 = no split)")
    a = ap.parse_args()
    # Layer 9 (model-00012) is the one routed-expert shard still on disk anywhere but the archive
    # host; the other 39 (layers 0-39) were converted into the CB3 cache and deleted. So every
    # case below reuses that one layer -- it still exercises the reader's chunk/offset arithmetic
    # across different experts (first, last, and two in between), just not across layers.
    cases = [(9, 0), (9, 383), (9, 123), (9, 1)]
    model_dir = a.model_dir or ET.env("MODEL_DIR") or RAW_SHARDS_DIR
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    probe = index["weight_map"].get(f"layers.{cases[0][0]}.ffn.experts.{cases[0][1]}.w1.weight")
    if not probe or not os.path.exists(os.path.join(model_dir, probe)):
        print(f"{model_dir} does not have the layer {cases[0][0]} expert shard -- "
              f"falling back to {RAW_SHARDS_DIR}")
        model_dir = RAW_SHARDS_DIR
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    a.model_dir = model_dir
    bad = 0
    for chunk in [float(c) for c in a.chunks_mb.split(",")]:
        store = EX.ExpertStore(a.model_dir, index, _StubArena(), 40, read_chunk_mb=chunk)
        t0 = time.perf_counter()
        for layer, e in cases:
            got = store.read_expert(layer, e)
            p = f"layers.{layer}.ffn.experts.{e}."
            f = safe_open(os.path.join(a.model_dir, index["weight_map"][p + "w1.weight"]), "pt", device="cpu")
            for i, name in enumerate(EX.NAMES):
                want = f.get_tensor(p + name).view(torch.uint8).reshape(-1)
                if not torch.equal(got[i].reshape(-1), want):
                    print(f"MISMATCH chunk={chunk} layer={layer} expert={e} {name}")
                    bad += 1
        dt = time.perf_counter() - t0
        gb = store.stats["bytes_read"] / 1e9
        print(f"chunk={chunk:>5} MB  {len(cases)} experts byte-exact  "
              f"({gb:.2f} GB in {dt:.2f}s = {gb / max(dt, 1e-9):.2f} GB/s, one expert at a time)")
        store.pool.shutdown(wait=True)
        store.read_pool.shutdown(wait=True)
    print("FAILED" if bad else "ok")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
