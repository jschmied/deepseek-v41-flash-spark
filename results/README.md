# These files are NOT regenerable on this box

`cb3_vs_fp4_paired_nll.json`, `cb2_vs_fp4_paired_nll.json` and `cb2_vs_cb3_paired_nll.json` were
measured 2026-09-14, while the FP4 routed-expert shards were still on local disk. They are the only
quality reference we have against the native FP4 checkpoint, and **they cannot be reproduced here
any more.**

The arithmetic, checked 2026-09-16:

| | |
|---|---|
| FP4 routed experts alone | **289 GB** (15,360 x 18,800,640 B) |
| full checkpoint, on the backup server | 428 GB |
| local free disk | **134 GB** |
| CB3 cache the engine runs on | **212 GB** |

Deleting the CB3 cache frees 346 GB, which holds the FP4 experts -- but then the engine has no cache
and rebuilding it from the FP4 source takes hours. So the box can hold FP4 **or** CB3, never both.
Re-running a teacher-forced comparison against FP4 costs the cache the engine runs on.

What they contain, arm A = FP4, arm B = the pack under test:

| vs FP4 | top-1 agreement, coding | top-1 agreement, general |
|---|---|---|
| CB3 (3 bit) | 96.70 % | 92.38 % |
| CB2 (2 bit) | 94.34 % | 89.17 % |

For external comparison, the published EXL3 SAGE packs report (different corpus, so this ranks and
does not control): 1.59 bpw at 97.9 % coding / 82.0 % general, 3.30 bpw at 94.0 % overall. Their
sensitivity-shaped allocation beats our CB3 on code at half the bits and loses badly on general
text.

**Do not clean this directory.** If disk is needed, take the nsys reports in `~/ds41-queue/logs`
(several GB each and regenerable) instead.
