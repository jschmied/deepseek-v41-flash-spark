"""Where do 2 GB/s of read bandwidth go between a bare process and the engine?

Measured today, same CB3Cache.read_into, same pinned aligned buffers, same 8 reader threads:

    bare process                6.81 GB/s
    engine loaded, GPU idle     4.99 GB/s      <- 27 % gone before any scheduling
    during a real decode        4.80-4.98 GB/s

The engine-loaded number was taken with the whole engine up, which changes a dozen things at once.
This adds them ONE AT A TIME to a bare process and re-runs the identical burst after each, so the
cost lands on a specific cause instead of on "the engine". No model weights are loaded, so this is
cheap and needs ~41 GB rather than ~106 GB.

  0  bare                          CB3Cache + 8 pinned aligned buffers
  1  + CUDA context                torch.cuda.init, nothing allocated
  2  + 40 GB device arena          what the engine pins; on GB10 device and host share LPDDR
  3  + 862 MiB pinned host         v1's 48 staging buffers of EXPERT_BYTES + 8*ALIGN
  4  + torch intra-op threads      the engine's thread population, without its work

The null hypothesis is that none of these matter and the gap is something else; stage 4 printing
6.8 GB/s would say exactly that, and would be worth knowing before any fix is attempted.

Run:  python enginev2/isolate_read_penalty.py
"""
import os, sys, time, random, threading, torch

V1 = os.path.expanduser("~/git/deepseek-v41-flash-spark")
sys.path[:0] = [V1, os.path.join(V1, "tools")]
os.chdir(V1)
from engine.cb3_cache import CB3Cache               # noqa: E402

ALIGN = 4096
EXPERT_BYTES = 3 * (2304 * 2560 + 2304 * 160)       # v1's staging size, the FP4 record
THREADS = int(os.environ.get("THREADS", 8))
NEACH = int(os.environ.get("NEACH", 32))
ARENA_GB = float(os.environ.get("ARENA_GB", 40))

cache = CB3Cache(os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin"), "cuda")
REC = cache.record
rng = random.Random(1234)
keys = [(rng.randrange(40), rng.randrange(384)) for _ in range(8192)]
cur = [0]
lk = threading.Lock()

bufs = []
for _ in range(THREADS):
    b = torch.empty(REC + ALIGN, dtype=torch.uint8, pin_memory=True)
    off = (-b.data_ptr()) % ALIGN
    bufs.append((memoryview(b[off:off + REC].numpy()), b))


def burst():
    def work(t):
        mv = bufs[t][0]
        for _ in range(NEACH):
            with lk:
                k = keys[cur[0] % len(keys)]
                cur[0] += 1
            cache.read_into(mv, k[0], k[1])
    ths = [threading.Thread(target=work, args=(t,)) for t in range(THREADS)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    dt = time.perf_counter() - t0
    n = THREADS * NEACH
    return n * REC / dt / 1e9


def report(stage):
    a = burst()
    b = burst()                                     # twice: the first can pay a one-off
    print(f"  {stage:34s} {a:5.2f} / {b:5.2f} GB/s")


report("0 bare")
torch.cuda.init()
torch.zeros(1, device="cuda")
report("1 + CUDA context")
hold = torch.empty(int(ARENA_GB * 1e9), dtype=torch.uint8, device="cuda")
report(f"2 + {ARENA_GB:.0f} GB device arena")
pinned = [torch.empty(EXPERT_BYTES + 8 * ALIGN, dtype=torch.uint8, pin_memory=True)
          for _ in range(48)]
report(f"3 + {48 * (EXPERT_BYTES + 8 * ALIGN) / 2**20:.0f} MiB pinned host")
torch.set_num_threads(max(1, os.cpu_count() or 8))
idle = [threading.Thread(target=lambda: time.sleep(20), daemon=True) for _ in range(96)]
for t in idle:
    t.start()
report("4 + 96 idle threads")
del hold, pinned
cache.close()
