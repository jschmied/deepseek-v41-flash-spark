"""Shared .env fallback for the engine/test_*.py smoke tests.

run.sh/start.sh source ./.env at the shell level ("environment beats .env", see run.sh's own
comment) before invoking anything. A bare `python -m engine.test_x` skips that step, so
MODEL_DIR/DSV41_CB3_CACHE/etc. are silently absent and a test either falls back to a stale
hardcoded path (~/models/DeepSeek-V4.1-Flash, which stopped existing when the box moved to
dsv41-lean) or a FileNotFoundError. This gives the tests the same fallback without requiring the
caller to `set -a; . ./.env; set +a` first -- an explicitly exported variable still wins, exactly
as it does for run.sh.
"""
from __future__ import annotations

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_PATH = os.path.join(_ROOT, ".env")
_loaded = False


def load_dotenv_defaults() -> None:
    """Populate os.environ from ./.env for keys that are not already set. Idempotent."""
    global _loaded
    if _loaded or not os.path.exists(_ENV_PATH):
        return
    _loaded = True
    for line in open(_ENV_PATH):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def env(key: str, default: str | None = None) -> str | None:
    """os.environ.get(key, default), after ensuring ./.env has had a chance to fill gaps."""
    load_dotenv_defaults()
    return os.environ.get(key, default)


def mem_available_gb() -> float:
    """/proc/meminfo MemAvailable, in GB. The GB10 is Grace-Blackwell unified memory -- GPU
    allocations by another process show up here too, there is no separate host-vs-device split to
    check (see MEMORY.md: GB10 OOM protection)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("inf")  # unknown platform (not Linux): don't block on a number we can't read


def require_memory_or_skip(min_gb: float, what: str) -> None:
    """Exit 0 with a SKIP message instead of loading `what` if MemAvailable is under min_gb.

    Some of these suites load a large chunk of the model (tens of GB, sometimes twice). Attempting
    that while another job already holds most of the box's unified memory does not fail cleanly --
    it either OOMs this process or starves the other job, which is exactly what we must not do.
    Re-running once memory frees up is enough; there is nothing else gating these tests.
    """
    avail = mem_available_gb()
    if avail < min_gb:
        print(f"SKIP: {what} -- needs ~{min_gb:.0f} GB free, this box has {avail:.1f} GB "
              f"MemAvailable right now (another job is holding the rest of the unified-memory "
              f"pool). Re-run once it frees up.")
        raise SystemExit(0)
