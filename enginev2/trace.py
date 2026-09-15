"""trace.py -- the captured route trace, which is the fake leaf standing in for the model's router.

`~/ds41-queue/logs/route-decode.jsonl` is a real capture written by engine/experts.py's ROUTE_LOG:
one JSON line per resolve() call in forward order, fields L (layer), pf (1 = prefill), uniq (the
expert ids that call wanted) and miss (the ids that were not resident AT CAPTURE TIME).

`miss` is deliberately NOT used to drive the skeleton. It is v1's own miss stream under whatever
cache state that run happened to be in; replaying it would bake one run's history in and make every
policy arm identical. The skeleton replays `uniq` through its OWN cache and derives misses from it,
which is what makes the eviction policy a real, swappable component.

Shape, verified: 40 prefill lines (one per layer) then 21,956 decode lines, strictly cyclic
0..39, so decode call index // 40 is the token and % 40 is the layer -- 548.9 decode steps.
"""

from __future__ import annotations

import json
import os

N_LAYERS = 40
DEFAULT_TRACE = os.path.expanduser("~/ds41-queue/logs/route-decode.jsonl")


def load_decode(path: str = DEFAULT_TRACE):
    """-> [(layer, (expert ids, ...)), ...] for decode calls only, in forward order."""
    calls = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("pf"):
                continue
            calls.append((o["L"], tuple(o["uniq"])))
    return calls


def warmup_cut(calls, prefix: float = 0.25) -> int:
    """The call index where scoring starts, rounded DOWN to a whole-token boundary.

    Every policy is warmed on calls [0, cut) and scored on the rest. This is not a free parameter:
    it is the offline replay's convention (eviction_oracle.py --prefix 0.25), and the published
    numbers the skeleton must reproduce -- LRU 92.67 % / 64.6 / 849 MiB -- are defined on that
    window. Scoring the whole trace instead gives 91.71 % / 72.7 / 955, because the first 2,433
    compulsory misses land in the score.
    """
    return int(len(calls) * prefix) // N_LAYERS * N_LAYERS
