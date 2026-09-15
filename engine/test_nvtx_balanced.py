"""Standalone check that nvtx_range never desyncs push/pop, and is a true no-op when off.

Doesn't touch the GPU or load a model: it monkeypatches torch.cuda.nvtx.range_push/range_pop with
counters and calls engine.model.nvtx_range directly, reloading the module under both states of the
gate (DSV41_NVTX=0 and =1) since NVTX is read once at import time. tools/cb3_moe.py carries an
independent copy of the same six lines (no engine/ import from tools/) and is not re-tested here.

Run: /home/jschmied/vllm-venv-main-dflash2/bin/python engine/test_nvtx_balanced.py
"""
from __future__ import annotations

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import torch  # noqa: E402


def _load(nvtx_env: str):
    os.environ["DSV41_NVTX"] = nvtx_env
    import engine.model as M  # noqa: PLC0415
    importlib.reload(M)  # NVTX is a module-level constant read at import time
    return M


def _check(nvtx_env: str, expect_on: bool):
    calls = {"push": 0, "pop": 0}
    orig_push, orig_pop = torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop
    torch.cuda.nvtx.range_push = lambda name: calls.__setitem__("push", calls["push"] + 1)
    torch.cuda.nvtx.range_pop = lambda: calls.__setitem__("pop", calls["pop"] + 1)
    try:
        M = _load(nvtx_env)
        assert M.NVTX is expect_on, f"DSV41_NVTX={nvtx_env!r} -> M.NVTX={M.NVTX}, expected {expect_on}"

        with M.nvtx_range("test.plain"):
            pass
        assert calls["push"] == calls["pop"], (nvtx_env, "plain", calls)

        for _ in range(5):  # repetition would surface any leaked/imbalanced state fast
            with M.nvtx_range("test.outer"):
                with M.nvtx_range("test.inner"):
                    pass
        assert calls["push"] == calls["pop"], (nvtx_env, "nested x5", calls)

        try:
            with M.nvtx_range("test.exc"):
                raise ValueError("boom")
        except ValueError:
            pass
        assert calls["push"] == calls["pop"], (nvtx_env, "exception", calls)

        if expect_on:
            assert calls["push"] > 0, "DSV41_NVTX=1 emitted no range_push calls"
        else:
            assert calls["push"] == 0, f"DSV41_NVTX=0 still emitted {calls['push']} range_push calls"
    finally:
        torch.cuda.nvtx.range_push = orig_push
        torch.cuda.nvtx.range_pop = orig_pop
    print(f"DSV41_NVTX={nvtx_env}: pushes==pops=={calls['push']}, "
          f"{'emitted to nvtx' if expect_on else 'no nvtx calls'} -- OK")


if __name__ == "__main__":
    _check("0", expect_on=False)
    _check("1", expect_on=True)
    print("ALL OK")
