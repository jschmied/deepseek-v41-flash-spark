"""
model.py -- DeepSeek-V4.1-Flash text model for one-box serving: chunked prefill, multi-token
decode blocks (DSpark verification) and cache rollback, batch size 1.

Ported from the reference `inference/model.py`; the tilelang kernels are replaced by torch ops
and the routed experts by `engine.experts.ExpertStore` (+ the Triton FP4 grouped-MoE kernel in
`tools/fp4_moe.py`). Deviations from the reference, all towards MORE precision:
  * activations are not fake-quantized to fp8 (optional flag),
  * window KV and compressed KV caches are kept in bf16 instead of fp8 / FP4-E4M3,
  * the indexer's Q/K are not FP4-quantized.
Position semantics are the reference's: a chunk of T tokens at absolute start position S.
Any (S, T) with T <= 512 works, which is what chunked prefill and 6-token verify blocks need.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import v41_ref as R  # noqa: E402

try:
    from decode_attn import prefill_attention, prefill_attention_gather  # tools/decode_attn.py
except Exception:  # noqa: BLE001
    prefill_attention = prefill_attention_gather = None

# Window ring slots. Must exceed window_size + the longest chunk a single forward sees, because
# `attention` gathers a query's window out of the ring AFTER writing the whole chunk into it
# (128 + 2048 here). 4096 slots x 512 dims x bf16 x 40 layers = 167 MB.
RING = int(os.environ.get("DSV41_RING", 4096))
# Longest prefill chunk. Bigger chunks are strictly cheaper on this recipe: a prefill chunk streams
# nearly every expert of every layer through the transient ring whatever its length (a 512-token
# chunk already touches ~370 of 384), so the NVMe traffic of a prompt is ~chunks x layers x 384
# experts and quadrupling the chunk quarters it. The ceiling is activation memory: at T=2048 the
# gathered window+compressed KV of one layer is ~2.7 GB.
# 4096, not 2048: measured -4.7 s on a 12,624-token prefill (57.9 -> 53.2 s warm), byte-identical
# output. Two of the three largest prefill costs scale with the CHUNK COUNT, not the token count --
# the per-chunk kernel launches, and the CB3 unpack, which re-unpacks nearly the same ~362 experts
# for every chunk of a layer. Halving the chunk count halves both. 8192 is better still on paper but
# the pre-flight refuses it: the reserve below is MAX_CHUNK * 5e6, which is 41 GB at 8192.
MAX_CHUNK = int(os.environ.get("DSV41_PREFILL_CHUNK", 4096))
# Per-layer phase timing for the layer-major pass. Every scheduling idea on the table is a claim
# about where time goes INSIDE a layer, and route_s/load_s/moe_s are per-request totals that
# cannot see it. Costs a device sync per layer, so it is a diagnostic, never a serving setting.
LM_PHASES = os.environ.get("DSV41_LM_PHASES", "0") == "1"
# Submit expert reads before the whole layer has routed. Modes, so the +3.7 s that all-chunk
# submission put back into attn+route can be ATTRIBUTED rather than assumed:
#   off       today's behaviour: one resolve after every chunk has routed
#   all       resolve(defer=True) after every chunk  (7 blocking route-id D2H per layer)
#   chunk0    resolve(defer=True) after chunk 0 only (1 D2H; chunk 0 already names 85.4 % of the
#             layer's expert set, so the union resolve at the end loads only the ~15 % tail)
#   synconly  the per-chunk D2H barrier and nothing else -- the control that separates the cost of
#             synchronising from the cost of overlapping
# Remove the fp32 concatenation in _softmax_attn (bit-identical; see there).
ATTN_NOCAT = os.environ.get("DSV41_ATTN_NOCAT", "0") == "1"
EARLY_SUBMIT = os.environ.get("DSV41_EARLY_SUBMIT", "off").lower()
if EARLY_SUBMIT in ("1", "true", "yes"):
    EARLY_SUBMIT = "all"
elif EARLY_SUBMIT in ("0", "false", "no", ""):
    EARLY_SUBMIT = "off"
assert EARLY_SUBMIT in ("off", "all", "chunk0", "synconly"), EARLY_SUBMIT

# nsys attribution only: 44.7 % of 148,447 kernel launches on a real prefill are tiny
# elementwise_kernel ops, and the profile alone cannot say which Python region emits them.
# `nvtx_range` marks a phase for nsys to bucket launches by; `if not NVTX: yield; return` means the
# default (off) path never touches torch.cuda.nvtx, so it is free everywhere it wraps.
NVTX = os.environ.get("DSV41_NVTX", "0") == "1"


@contextlib.contextmanager
def nvtx_range(name: str):
    if not NVTX:
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


# One Triton kernel for the prefill softmax attention instead of the eager einsum -> masked_fill ->
# amax -> exp -> sum -> einsum chain. Profiled at the real shape (T=192, H=64, D=512, N=640) the
# eager path is 53 kernel launches per call and the kernel is 1; the eager count scales with
# ceil(T/ATTN_TILE), so a T=2048 chunk is ~560 launches per layer per call. See
# tools/decode_attn.prefill_attention for the trace numbers that motivate it.
# Default OFF: the PV product inside the kernel runs through tl.dot, so the probabilities are
# carried as a bf16 high/low pair rather than in fp32 and the result is close but not bit-identical
# -- the same reason DSV41_FUSED_ATTN is off on the decode path. Measured divergence at the real
# shape is max|d|/max|ref| 7.5e-4, ||d||/||ref|| 9.3e-5, 0.02 % of elements more than one bf16 step
# apart; engine/test_fused_prefill_attn.py holds it there. DSV41_ATTN_FUSED_PREFILL=1 turns it on.
ATTN_FUSED_PREFILL = os.environ.get("DSV41_ATTN_FUSED_PREFILL", "0") == "1"

# ... and one step further: let that kernel do the KV gather itself instead of being handed a
# materialised [T, 128+512, 512] bf16 `kv_all`. Requires DSV41_ATTN_FUSED_PREFILL (it is the same
# kernel, with a row-addressing mode per segment) and changes no arithmetic at all -- same softmax,
# same PV split, same sink -- so it is bit-identical to the fused path, unlike that path against
# eager. What it removes is memory and copies: the gathered kv_all is 1.34 GB at T=2048 and ~2.7 GB
# at its construction peak, which is the ceiling MAX_CHUNK is written against, plus 2.15 s of
# CatArrayBatchedCopy* and 1.31 s of vectorized_gather_kernel out of a 40 s GPU-busy prefill.
# Measured at T=2048, H=64, D=512, 128+512 keys: 5.65 ms against 27.23 ms for gather+cat+kernel
# (4.8x), and even the K-loop alone is faster (5.65 vs 7.02 ms) because the ring and the compressed
# cache are 4 MB tables every query shares, where kv_all is 1.34 GB streamed once.
# DSV41_ATTN_GATHER_FUSED=1 turns it on.
# NOTE the ring must still hold window_size + MAX_CHUNK positions (RING >= 8320 for a 8192 chunk);
# the kernel derives the same rows the host gather did, it does not relax that.
ATTN_GATHER_FUSED = os.environ.get("DSV41_ATTN_GATHER_FUSED", "0") == "1"

# Chunk invariance requires every GEMM to give the same row whatever the batch length M. cuBLAS
# picks split-K kernels for small M and, with this flag on, reduces the K-splits in bf16, so
# F.linear(x[:6], w) != F.linear(x, w)[:6] by ~2.4e-3 for the N=512 wkv projection -- which the
# attention softmax then amplifies ~2x per layer. fp32 reduction cuts that to ~9e-5.
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# ... and the same GEMM must be issued with the same M whatever the chunk length, or cuBLAS
# switches tiling/split-K and a row comes out a few ulps different. See v41_ref.mm().
# 16 and not something larger: for several of these shapes (wq_a, the expert w1/w3) cuBLAS gives a
# row a slightly different result depending on its OFFSET inside the tile, and a token's offset is
# chunk-relative. 8 and 16 are offset-invariant for every shape the model uses; 32/64/128 are not.
MM_TILE = 16
# The attention softmax is a *batched* GEMM (one independent problem per token), which is already
# offset-invariant, so it can use a bigger tile.
ATTN_TILE = 64
KEY_BLOCK = 512  # indexer score tile along the compressed-key axis (= index_topk)
R.MM_TILE = MM_TILE


# ----------------------------------------------------------------------------- weights
class IndexerWeights:
    def __init__(self, get, p: str, owns_k: bool, device: str):
        w, sc = get(p + "indexer.wq_b.weight").to(device), get(p + "indexer.wq_b.scale").to(device)
        self.wq_b = R.FP8Weight(w, sc) if (R.FP8Weight is not None and os.environ.get("DSV41_DENSE_FP8", "1") == "1") else R.dequant_fp8_block(w, sc)
        self.weights_proj = get(p + "indexer.weights_proj.weight").to(device).to(torch.bfloat16)
        self.owns_k = owns_k
        if owns_k:
            self.wk = get(p + "indexer.wk.weight").to(device).to(torch.bfloat16)
            self.k_norm = get(p + "indexer.k_norm.weight").to(device).to(torch.bfloat16)


class Weights:
    """All non-routed-expert weights on the GPU, bf16 (fp32 where the reference uses fp32)."""

    def __init__(self, model_dir: str, index: dict, args: R.Args, device: str, log=print, act_quant: bool = False,
                 n_layers: int | None = None, load_mtp: bool = True, engram_dir: str | None = None):
        self.args, self.device = args, device
        n_load = args.n_layers if n_layers is None else n_layers
        wm = index["weight_map"]
        handles = {}

        def get(name):
            f = wm[name]
            if f not in handles:
                path = os.path.join(model_dir, f)
                if not os.path.exists(path) and ".engram." in name:
                    # engram shard not (yet) on disk: the small non-table tensors fetched by tools/engram_rows.py
                    L = name.split(".")[1]
                    path = os.path.join(engram_dir or "engram_rows", f"layer{L}_weights.safetensors")
                handles[f] = safe_open(path, "pt", device="cpu")
            return handles[f].get_tensor(name)

        t0 = time.time()
        self.embed = get("embed.weight").to(device).to(torch.bfloat16)
        # bf16 (the stored dtype) unless DSV41_HEAD_FP32=1. The reference keeps the LM head in fp32
        # ("so the logits come out in fp32 directly"); the fast decode path already ran a bf16 copy
        # (fp32 accumulate, logits rounded to bf16), so with a bf16 head here the two paths use the
        # same weights and the 2.65 GB fp32 copy disappears (= ~140 more expert slots).
        # ... and, with DSV41_HEAD_FMT, in fp8 or fp4 instead: the head is read in full on every
        # decode step, so its stored format is worth as much as a dense projection group's.
        self.head = R.make_head(get("head.weight").to(device))
        self.norm = get("norm.weight").to(device).to(torch.bfloat16)
        self.layers = []
        self.indexers = {}
        self.engram = {}
        for L in range(n_load):
            self.layers.append(R.LayerWeights(get, L, args, device))
            if L in args.index_source_layers:
                self.indexers[L] = IndexerWeights(get, f"layers.{L}.attn.", L in args.kv_source_layers, device)
            if L in args.engram_layer_ids:
                self.engram[L] = R.EngramWeights(get, L, device)
            if L % 10 == 9:
                log(f"weights: layer {L} loaded ({time.time() - t0:.0f}s)")
            for f in list(handles):
                if f.endswith(f"{L + 3:05d}-of-00048.safetensors"):
                    del handles[f]
        # DSpark blocks
        self.mtp = []
        for k in range(3 if load_mtp else 0):
            self.mtp.append(MTPWeights(get, k, args, device))
        self.dspark_experts = None  # filled by the engine (arena of 3 x 128 experts)
        log(f"weights: all non-expert weights on GPU in {time.time() - t0:.0f}s")


class MTPWeights:
    """One DSpark block (`mtp.k.*`): a Block with a 128-expert MoE, plus stage-specific heads."""

    def __init__(self, get, k: int, args: R.Args, device: str):
        p = f"mtp.{k}."
        self.k = k

        class _A(R.Args):
            pass

        a = R.Args(**{f: getattr(args, f) for f in R.Args.__dataclass_fields__})
        a.n_routed_experts = 128
        a.n_activated_experts = 3
        self.args = a
        # LayerWeights expects "layers.{L}." prefix; build the same fields by hand
        dev = device

        def bf(name):
            return get(p + name).to(dev).to(torch.bfloat16)

        def f32(name):
            return get(p + name).to(dev).to(torch.float32)

        fp4_groups = R.dense_fp4_groups()

        def fp8lin(name):
            w, sc = get(p + name + ".weight").to(dev), get(p + name + ".scale").to(dev)
            if R.FP8Weight is not None and os.environ.get("DSV41_DENSE_FP8", "1") == "1":
                return R.maybe_fp4(R.FP8Weight(w, sc), name, fp4_groups)
            return R.dequant_fp8_block(w, sc)

        self.attn_norm = bf("attn_norm.weight"); self.ffn_norm = bf("ffn_norm.weight")
        self.attn_sink = f32("attn.attn_sink"); self.q_norm = bf("attn.q_norm.weight"); self.kv_norm = bf("attn.kv_norm.weight")
        self.wq_a = fp8lin("attn.wq_a"); self.wq_b = fp8lin("attn.wq_b"); self.wkv = fp8lin("attn.wkv")
        self.wo_a = R.make_wo_a(get(p + "attn.wo_a.weight").to(dev), get(p + "attn.wo_a.scale").to(dev), args, fp4_groups); self.wo_b = fp8lin("attn.wo_b")
        self.hc_attn_fn = f32("hc_attn_fn"); self.hc_ffn_fn = f32("hc_ffn_fn")
        self.hc_attn_base = f32("hc_attn_base"); self.hc_ffn_base = f32("hc_ffn_base")
        self.hc_attn_scale = f32("hc_attn_scale"); self.hc_ffn_scale = f32("hc_ffn_scale")
        self.gate_w = f32("ffn.gate.weight"); self.gate_bias = f32("ffn.gate.bias")
        self.sh_w1 = fp8lin("ffn.shared_experts.w1"); self.sh_w2 = fp8lin("ffn.shared_experts.w2"); self.sh_w3 = fp8lin("ffn.shared_experts.w3")
        self.ratio = 0
        self.is_kv_source = False
        self.layer = args.n_layers + k
        if k == 0:
            self.main_proj = fp8lin("main_proj")
            self.main_norm = bf("main_norm.weight")
        if k == 2:
            self.norm = bf("norm.weight")
            self.markov_embed = bf("markov_head.embed.weight")
            # fp32 once, for the same reason as the LM head: this one is applied once per drafted
            # token, i.e. five times per DSpark step.
            self.markov_head = get(p + "markov_head.head.weight").to(dev).to(torch.bfloat16)
            self.conf_proj = get(p + "confidence_head.proj.weight").to(dev).float()


# ----------------------------------------------------------------------------- caches
class Caches:
    def __init__(self, args: R.Args, max_seq: int, device: str):
        self.args, self.max_seq, self.device = args, max_seq, device
        d = args.head_dim
        self.win = [torch.zeros(RING, d, dtype=torch.bfloat16, device=device) for _ in range(args.n_layers)]
        self.mtp_win = [torch.zeros(RING, d, dtype=torch.bfloat16, device=device) for _ in range(3)]
        self.ckv = {}
        self.ik = {}
        self.pending = {}  # ratio-2 sources: (kv fp32 [512], score fp32 [512]) of an unpaired position, or None
        for L in args.kv_source_layers:
            r = args.compress_ratios[L]
            n = max_seq // r + 1
            self.ckv[L] = torch.zeros(n, d, dtype=torch.bfloat16, device=device)
            self.ik[L] = torch.zeros(n, args.index_head_dim, dtype=torch.bfloat16, device=device)
            self.pending[L] = None
        self.len = 0  # number of valid positions
        # per-chunk memory for rollback of the compressor state
        self._chunk_inputs = {}  # L -> (S, kv [T,512] fp32, score [T,512] fp32, pending_before)
        # Prefill-chunk-boundary checkpoints of the compressor state, for the prompt cache.
        # `rollback()` can only reach back into the LAST chunk, because that is all
        # `_chunk_inputs` keeps -- enough for speculative rejection, useless for resuming a
        # conversation. The only state that does not survive a jump backwards is `pending`, and it
        # is four layers x two [512] fp32 rows = 16 KB per boundary, so every boundary is kept.
        self._ckpt = {}  # position -> {L: pending_L or None}

    def rollback(self, n: int):
        """Discard everything at positions >= n.

        Only the compressor carries state across positions, so only `pending` has to be restored:
        the window ring and the compressed/index caches are append-only and every slot at or after
        n is rewritten by the next forward before anything can read it.
        """
        assert n <= self.len
        for L, (S, kv, sc, before) in self._chunk_inputs.items():
            r = self.args.compress_ratios[L]
            if r == 1:
                continue
            if n % r == 0:
                self.pending[L] = None  # n positions = n/r whole groups, nothing left over
                continue
            p = n - 1  # the position left unpaired at n
            if p >= S:
                self.pending[L] = (kv[p - S], sc[p - S])
            elif p == S - 1:
                self.pending[L] = before  # exactly back to the start of the last chunk
            else:
                raise ValueError(f"rollback({n}) reaches before the last chunk (start {S}); the "
                                 f"compressor input for position {p} is no longer kept")
        self.len = n

    # ------------------------------------------------------ prompt-cache checkpoints
    def checkpoint(self, n: int):
        """Remember the compressor state as of position `n` (a prefill chunk boundary)."""
        self._ckpt[n] = {L: (None if v is None else (v[0].clone(), v[1].clone()))
                         for L, v in self.pending.items()}

    def resume_at(self, n: int):
        """Jump back to checkpoint `n` and continue appending there.

        Unlike `rollback`, this reaches arbitrarily far back -- but only to a position that was
        checkpointed, because nothing else can restore `pending`. Everything else in the cache is
        either append-only (`ckv`, `ik`) or a ring whose entries at or after `n` are rewritten
        before anything reads them (`win`), so positions < n stay valid exactly as they are.
        """
        self.pending = {L: (None if v is None else (v[0].clone(), v[1].clone()))
                        for L, v in self._ckpt[n].items()}
        self._chunk_inputs.clear()   # rewritten by the first chunk of the resumed prefill
        # Checkpoints past the resume point describe positions this prefill is about to rewrite
        # with different tokens. Keeping them would let a LATER request resume onto a prefix that
        # was never written.
        for k in [k for k in self._ckpt if k > n]:
            del self._ckpt[k]
        self.len = n


class Shared:
    def __init__(self):
        self.ckv = None
        self.ik = None
        self.ratio = 0
        self.topk = None  # [T, k] absolute compressed positions or -1
        self.candidates = None  # [T, n_c] bool


# ----------------------------------------------------------------------------- model
class Model:
    def __init__(self, W: Weights, store, caches: Caches, moe_fn, act_quant: bool = False):
        self.W, self.store, self.c, self.moe_fn = W, store, caches, moe_fn
        # THE ARENA IS THE MODEL'S, not the expert store's. It is a buffer the LOADER writes
        # into and the MoE kernels read, and which loader that is has stopped being v1's:
        # enginev2 reimplemented the read, the H2D, the slot map and the scheduling, and
        # reached it as `m.store.arena` only because of where the attribute happened to sit.
        # Same object the store was handed; nothing owns it twice.
        self.arena = getattr(store, "arena", None)
        self.args = W.args
        self.dev = W.device
        a = self.args
        self.freqs_c = R.precompute_freqs_cis(a.rope_head_dim, caches.max_seq + 8, a.original_seq_len, a.compress_rope_theta,
                                              a.rope_factor, a.beta_fast, a.beta_slow, self.dev)
        self.freqs_w = R.precompute_freqs_cis(a.rope_head_dim, caches.max_seq + 8, 0, a.rope_theta, a.rope_factor,
                                              a.beta_fast, a.beta_slow, self.dev)
        self.tap = None  # optional diagnostic hook: callable(name, L, tensor)
        self.engram_rows = None  # callable (layer, hashes [T,24]) -> [T,24,256] float32
        self.hash_state = None  # reference NgramHashState
        if not act_quant:
            R.act_qdq_fp8 = lambda x, block=32: x.to(torch.bfloat16)
        self.stats = {"attn_s": 0.0, "moe_s": 0.0, "engram_s": 0.0, "tokens": 0}
        self.begin_prompt()

    def _tap(self, name, L, t):
        if self.tap is not None:
            self.tap(name, L, t)

    # ------------------------------------------------------------------ attention
    def _window_positions(self, pos: torch.Tensor):
        """[T, 128] absolute positions each query may see in its sliding window, -1 if none."""
        w = self.args.window_size
        p = pos[:, None] - torch.arange(w - 1, -1, -1, device=self.dev)[None, :]
        return torch.where(p >= 0, p, torch.full_like(p, -1))

    def attention(self, x: torch.Tensor, w, L: int, S: int, sh: Shared, ring: torch.Tensor,
                  freqs: torch.Tensor, mtp_extra=None, win_lo: int = 0):
        """x: [T, d] normed input. Returns [T, d]. `ring` is this layer's window KV ring.

        `win_lo` is the first position whose window KV this ring actually holds. It is 0 everywhere
        except in the decoder replay (SWA Bounded Replay, tech report 3.2.2), where the decoder
        layers have only seen the last 128 prompt tokens and a query near the start of the replay
        would otherwise gather whatever the ring happens to hold below it.
        """
        a = self.args
        T = x.size(0)
        pos = torch.arange(S, S + T, device=self.dev)
        rd = a.rope_head_dim
        fq = freqs[S:S + T]

        self._tap("attn_x", L, x)
        qr = R.rmsnorm(R.qlinear(x, w.wq_a), w.q_norm, a.norm_eps)
        q = R.qlinear(qr, w.wq_b).view(T, a.n_heads, a.head_dim)
        with nvtx_range("attn.rope"):
            q = torch.cat([q[..., :-rd], R.apply_rotary(q[..., -rd:], fq)], dim=-1)
        self._tap("q", L, q)

        kv = R.rmsnorm(R.qlinear(x, w.wkv), w.kv_norm, a.norm_eps)
        with nvtx_range("attn.rope"):
            kv = torch.cat([kv[:, :-rd], R.apply_rotary(kv[:, -rd:], fq)], dim=-1)
        self._tap("kv_new", L, kv)
        # the gather-fused arm: the attention kernel reads the ring and the compressed cache
        # itself, so `kv_all` is never built. It cannot serve the draft branch below, whose second
        # segment is the draft's own kv, not rows of a shared table.
        fuse = (mtp_extra is None and ATTN_GATHER_FUSED and ATTN_FUSED_PREFILL
                and prefill_attention_gather is not None and q.dtype == torch.bfloat16 and q.is_cuda)
        g_ckv = g_idx = kv_all = None
        if mtp_extra is None:
            # gather the window BEFORE writing (a chunk may overwrite slots older queries still need)
            wpos = self._window_positions(pos)  # [T, 128]
            ring[pos % RING] = kv
            wmask = wpos >= win_lo if win_lo else wpos >= 0
            if not fuse:
                wkv = ring[wpos.clamp_min(0) % RING]  # [T, 128, d]
                self._tap("win_kv", L, wkv)
            elif self.tap is not None:
                # the fused arm never builds these rows, but the taps are how this engine is
                # checked against tools/v41_ref, so a diagnostic run pays for them explicitly
                # rather than silently going dark. `self.tap is None` in serving.
                self._tap("win_kv", L, ring[wpos.clamp_min(0) % RING])
            self._tap("win_mask", L, wmask)
            mask = wmask
            if not fuse:
                kv_all = wkv
            if w.ratio:
                got, cmask = self._compressed(x, qr, w, L, S, T, pos, sh, return_rows=not fuse)
                self._tap("c_mask", L, cmask)
                mask = torch.cat([wmask, cmask], dim=1)
                if fuse:
                    g_ckv, g_idx = sh.ckv, got  # [n_c, d] table + [T, k] rows, not the rows
                    if self.tap is not None:
                        self._tap("ckv_rows", L, sh.ckv[got.clamp_min(0)])
                else:
                    self._tap("ckv_rows", L, got)
                    kv_all = torch.cat([wkv, got], dim=1)
        else:
            # DSpark draft attention: window from the main stream's ring (positions <= S-1) + all draft kvs
            main_last = mtp_extra  # position of the last main token in the ring
            wpos = main_last - torch.arange(a.window_size - 1, -1, -1, device=self.dev)
            wpos = torch.where(wpos >= 0, wpos, torch.full_like(wpos, -1))  # [128]
            wkv = ring[wpos.clamp_min(0) % RING][None].expand(T, -1, -1)
            kv_all = torch.cat([wkv, kv[None].expand(T, -1, -1)], dim=1)
            mask = torch.cat([(wpos >= 0)[None].expand(T, -1), torch.ones(T, T, dtype=torch.bool, device=self.dev)], dim=1)

        # The softmax runs on fixed-size token tiles for the same reason the GEMMs do: the batched
        # score/PV products are not invariant to the number of query rows in the call.
        # fp32 PV product, like tools/v41_ref: rounding the probabilities to bf16 first is a
        # cliff that turns 1e-7 fp32 GEMM jitter into 1e-4 output jitter, which is enough to flip
        # a borderline router top-k and make the MoE output depend on the chunk length.
        if fuse:
            o = self._gather_attn(q, ring, g_ckv, g_idx, S, mask, w.attn_sink)
        else:
            o = self._softmax_attn(q, kv_all, mask, w.attn_sink)
        with nvtx_range("attn.rope"):
            o = torch.cat([o[..., :-rd], R.apply_rotary(o[..., -rd:], fq, inverse=True)], dim=-1)
        o = o.reshape(T, a.o_groups, -1)
        # grouped output projection: "sgd,grd->sgr" is a GEMM with M = number of tokens, so it too
        # has to run on fixed-size token tiles (it differs most visibly at a 1-token chunk).
        o = R.wo_a_proj(o, w.wo_a, tiled=True)
        out = R.qlinear(o.flatten(1), w.wo_b)
        self._tap("attn_out", L, out)
        return out

    def _gather_attn(self, q, ring, ckv, idx, S: int, mask, sink):
        """`_softmax_attn` for the prefill window+compressed case, without the gathered KV.

        Same kernel, same math, same launch shape as the DSV41_ATTN_FUSED_PREFILL path -- the only
        difference is that the two key segments are ADDRESSED (ring row (pos - 127 + j) % RING,
        compressed row idx[t, j]) instead of having been copied into a [T, 640, 512] tensor first,
        so the result is bit-identical to `_softmax_attn` under that gate, not merely close.

        `S` is the absolute position of query row 0; prefill positions are contiguous, which is
        what lets the window rows be derived without any index tensor at all.
        """
        with nvtx_range("attn.softmax"):
            return prefill_attention_gather(q, ring, ckv, idx, S, mask, sink,
                                            self.args.head_dim ** -0.5,
                                            self.args.window_size, RING)

    def _softmax_attn(self, q, kv_all, mask, sink):
        """Sinked softmax attention over [T, n, d] KV, in fixed-size query tiles.

        Padding rows are all-masked: their scores are -inf, so m clamps to -1e30, p is 0 and the
        sink term makes the denominator +inf -- 0/inf = 0, no NaN.
        """
        with nvtx_range("attn.softmax"):
            T = q.size(0)
            scale = self.args.head_dim ** -0.5
            if (ATTN_FUSED_PREFILL and prefill_attention is not None
                    and q.dtype == torch.bfloat16 and kv_all.dtype == torch.bfloat16 and q.is_cuda):
                # No ATTN_TILE padding here: a kernel program reduces over the key axis of one token
                # in a fixed order, so a row is already independent of how many rows are in the call
                # -- which is the invariance the query tiling was buying.
                return prefill_attention(q, kv_all, mask, sink, scale)
            B = ATTN_TILE if ATTN_TILE > 0 else T

            def tile(qt, kvt, mt):
                scores = torch.einsum("thd,tnd->thn", qt, kvt) * scale
                scores = scores.masked_fill(~mt[:, None, :], float("-inf"))
                mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
                p = torch.exp(scores - mx)
                denom = p.sum(-1, keepdim=True) + torch.exp(sink[None, :, None] - mx)
                return torch.einsum("thn,tnd->thd", p / denom, kvt)

            # ATTN_NOCAT: write each tile straight into the final bf16 buffer instead of keeping every
            # fp32 tile alive and concatenating at the end. The old form materialises [T, H, D] fp32
            # (268 MB at T=2048), reads it back to concatenate, and reads it a third time to cast --
            # measured across all torch.cat sites: CatArrayBatchedCopy* is 2.15 s of a 40.3 s GPU-busy
            # prefill. Bit-identical by construction: the cast is elementwise, so converting a tile at a
            # time rounds every element exactly as converting the concatenation does, and no reduction
            # order changes. The gate exists only to A/B it; there is no numerical reason for two paths.
            out = None
            outs = [] if not ATTN_NOCAT else None
            for i in range(0, T, B):
                j = min(i + B, T)
                qt, kvt, mt = q[i:j].float(), kv_all[i:j].float(), mask[i:j]
                n = j - i
                if n < B:  # pad the last tile so every call sees exactly B query rows
                    qt = torch.cat([qt, qt.new_zeros(B - n, *qt.shape[1:])])
                    kvt = torch.cat([kvt, kvt.new_zeros(B - n, *kvt.shape[1:])])
                    mt = torch.cat([mt, mt.new_zeros(B - n, mt.size(1))])
                y = tile(qt, kvt, mt)[:n]
                if not ATTN_NOCAT:
                    outs.append(y)
                    continue
                if out is None:
                    out = torch.empty((T, *y.shape[1:]), dtype=torch.bfloat16, device=y.device)
                out[i:j].copy_(y)          # fp32 -> bf16 here, same rounding, no fp32 concatenation
            return out if ATTN_NOCAT else torch.cat(outs).to(torch.bfloat16)

    def _compressed(self, x, qr, w, L, S, T, pos, sh: Shared, return_rows: bool = True):
        """Produce/read the shared compressed KV for this chunk; run/reuse the indexer; return the
        gathered rows [T, k, d] and their mask [T, k].

        `return_rows=False` hands back the indexer's [T, k] absolute rows instead of the gathered
        KV, for the gather-fused attention kernel that follows the index itself: the gather is
        1.31 s of vectorized_gather_kernel per prefill and, at T=2048, 1.07 GB of the chunk's
        peak (512 rows x 512 dims x bf16 per query) before the concatenation doubles it."""
        with nvtx_range("attn.compressed"):
            a = self.args
            r = w.ratio
            c = self.c
            rd = a.rope_head_dim
            if w.is_kv_source:
                # latent for every position of the chunk (plus the pending unpaired one)
                if r > 1:
                    xf = x.float()
                    kvl, sc = R.mm(xf, w.comp_wkv), R.mm(xf, w.comp_wgate)
                    before = c.pending[L]
                    c._chunk_inputs[L] = (S, kvl, sc, before)
                    if before is not None:
                        kvl = torch.cat([before[0][None], kvl]); sc = torch.cat([before[1][None], sc])
                        first = S - 1
                    else:
                        first = S
                    n_tok = kvl.size(0)
                    cut = n_tok - n_tok % r
                    if n_tok % r:
                        c.pending[L] = (kvl[-1], sc[-1])
                    else:
                        c.pending[L] = None
                    if cut > 0:
                        g_kv = kvl[:cut].unflatten(0, (-1, r)); g_sc = sc[:cut].unflatten(0, (-1, r))
                        latent = (g_kv * g_sc.softmax(dim=1)).sum(dim=1)
                        latent = R.rmsnorm(latent.to(torch.bfloat16), w.comp_norm, a.norm_eps)
                        j0 = first // r
                    else:
                        latent, j0 = None, first // r
                else:
                    latent = R.rmsnorm(R.mm(x, w.comp_wkv), w.comp_norm, a.norm_eps)
                    j0 = S
                if latent is not None:
                    nj = latent.size(0)
                    jpos = (j0 + torch.arange(nj, device=self.dev)) * r
                    fj = self.freqs_c[jpos]
                    if L in self.W.indexers:  # index key from the pre-RoPE latent
                        iw = self.W.indexers[L]
                        k = R.rmsnorm(R.mm(latent, iw.wk), iw.k_norm, a.norm_eps)
                        k = torch.cat([k[:, :-rd], R.apply_rotary(k[:, -rd:], fj)], dim=-1)
                        c.ik[L][j0:j0 + nj] = k
                    lat = torch.cat([latent[:, :-rd], R.apply_rotary(latent[:, -rd:], fj)], dim=-1)
                    c.ckv[L][j0:j0 + nj] = lat
                    self._tap("latent", L, (j0, lat))
                sh.ckv, sh.ik, sh.ratio = c.ckv[L], c.ik[L], r
            assert sh.ratio == r, (L, sh.ratio, r)
            compress_lens = (pos + 1) // r  # visible compressed positions per query
            n_c = int((S + T) // r)
            if L in self.W.indexers:
                sh.topk = self._indexer(x, qr, L, pos, compress_lens, n_c, sh)
            idx = sh.topk  # [T, k] absolute compressed positions, -1 = none
            self._tap("topk", L, idx); self._tap("n_c", L, n_c)
            if not return_rows:
                return idx, idx >= 0
            rows = sh.ckv[idx.clamp_min(0)]
            return rows, idx >= 0

    def _indexer(self, x, qr, L, pos, compress_lens, n_c, sh: Shared):
        a = self.args
        iw = self.W.indexers[L]
        T = x.size(0)
        rd = a.rope_head_dim
        if n_c == 0:
            return self._pad_topk(torch.full((T, 0), -1, dtype=torch.int64, device=self.dev))
        q = R.qlinear(qr, iw.wq_b).view(T, a.index_n_heads, a.index_head_dim)
        q = torch.cat([q[..., :-rd], R.apply_rotary(q[..., -rd:], self.freqs_c[pos])], dim=-1)
        wts = (R.mm(x, iw.weights_proj).float() * (a.index_head_dim ** -0.5 * a.index_n_heads ** -0.5))  # [T, H]
        # fixed [MM_TILE queries x KEY_BLOCK keys] score tiles: n_c grows with the chunk, so a
        # single GEMM over sh.ik[:n_c] would be a different shape in every chunk. Columns past
        # n_c are masked out below, so the padding cannot be selected.
        NB = KEY_BLOCK
        n_pad = max(NB, (n_c + NB - 1) // NB * NB)
        k = sh.ik[:n_pad]
        if k.size(0) < n_pad:
            k = torch.cat([k, k.new_zeros(n_pad - k.size(0), k.size(1))])
        B = ATTN_TILE if ATTN_TILE > 0 else T
        score = torch.empty(T, n_pad, dtype=torch.float32, device=self.dev)
        for i in range(0, T, B):
            j = min(i + B, T)
            qt, wt = q[i:j], wts[i:j]
            if j - i < B:
                qt = torch.cat([qt, qt.new_zeros(B - (j - i), *qt.shape[1:])])
                wt = torch.cat([wt, wt.new_zeros(B - (j - i), wt.size(1))])
            for jb in range(0, n_pad, NB):
                sc = torch.einsum("thd,nd->thn", qt, k[jb:jb + NB])  # bf16
                sc = sc.float().relu_() * wt[:, :, None]
                score[i:j, jb:jb + NB] = sc.sum(dim=1)[:j - i]
        cpos = torch.arange(n_pad, device=self.dev)
        score.masked_fill_(cpos[None, :] >= compress_lens[:, None], float("-inf"))
        is_cand_src = L == a.candidate_source_layer
        if is_cand_src:
            sh.candidates = self._select_candidates(score, compress_lens, a.candidate_topk_blocks, a.candidate_block_size)
        elif 0 <= a.candidate_source_layer < L and sh.candidates is not None:
            score = score.masked_fill(~sh.candidates, float("-inf"))
        k_ = min(a.index_topk, n_c)
        idx = score.topk(k_, dim=-1, sorted=False).indices.sort(dim=-1).values
        idx = torch.where(idx < compress_lens[:, None], idx, torch.full_like(idx, -1))
        return self._pad_topk(idx)

    def _pad_topk(self, idx: torch.Tensor) -> torch.Tensor:
        """Always hand back exactly index_topk columns, the tail filled with -1 (= masked off).

        The number of compressed rows a chunk can reach, min(index_topk, (S+T)//ratio), grows with
        the chunk, so without this the concatenated KV of the attention softmax would be a
        different width in a short chunk than in a long one. The extra columns are fully masked
        and change nothing mathematically, but a different N makes cuBLAS pick a different kernel
        for the score GEMM, and the resulting ulp differences flip router decisions downstream.
        """
        pad = self.args.index_topk - idx.size(1)
        return idx if pad <= 0 else F.pad(idx, (0, pad), value=-1)

    @staticmethod
    def _select_candidates(logits, compress_lens, topk_blocks, block_size):
        width = logits.size(-1)
        scores = F.pad(logits, (0, -width % block_size), value=float("-inf"))
        scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
        num_blocks = scores.size(-1)
        last = ((compress_lens - 1) // block_size)[:, None]
        scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device)[None, :] == last, float("inf"))
        top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
        keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
        return keep.repeat_interleave(block_size, dim=-1)[..., :width]

    # ------------------------------------------------------------------ blocks
    def moe_route(self, y: torch.Tensor, w, L: int, n_experts: int):
        """Routing only: (indices [T, k], weights [T, k]). Split out of `moe` so a layer-major
        prefill can route every chunk, resolve the layer's whole expert set in ONE call, and then
        apply the kernel per chunk. Byte for byte the code `moe` used to run."""
        with nvtx_range("moe.route"):
            a = self.args
            self._tap("moe_in", L, y)
            scores = F.softplus(R.mm(y.float(), w.gate_w)).sqrt()
            k = 3 if n_experts == 128 else a.n_activated_experts
            logits = scores + w.gate_bias
            pm = getattr(self, "prune_mask", None)
            if pm is not None and n_experts != 128 and L in pm:
                logits = logits.masked_fill(~pm[L], float("-inf"))
            indices = logits.topk(k, dim=-1)[1]
            weights = scores.gather(1, indices)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * a.route_scale
            self._tap("route_idx", L, indices); self._tap("route_w", L, weights)
            return indices, weights

    def moe_apply(self, y: torch.Tensor, slots, weights, w, arena):
        """The kernel half: routed experts through their slots, plus the shared expert."""
        with nvtx_range("moe.apply"):
            a = self.args
            t0 = time.perf_counter()
            routed = self.moe_fn(y, slots, weights, arena, a.swiglu_limit).float()
            shared = R.expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, a.swiglu_limit).float()
            out = routed + shared
            self.stats["moe_s"] += time.perf_counter() - t0
            return out.to(y.dtype)

    def moe(self, y: torch.Tensor, w, L: int, prefill: bool, store, arena, n_experts: int):
        a = self.args
        self._tap("moe_in", L, y)
        scores = F.softplus(R.mm(y.float(), w.gate_w)).sqrt()
        k = 3 if n_experts == 128 else a.n_activated_experts
        logits = scores + w.gate_bias
        pm = getattr(self, "prune_mask", None)
        if pm is not None and n_experts != 128 and L in pm:
            # expert pruning experiment: the router may only pick surviving experts (REAP-style drop)
            logits = logits.masked_fill(~pm[L], float("-inf"))
        indices = logits.topk(k, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * a.route_scale
        self._tap("route_idx", L, indices); self._tap("route_w", L, weights)
        t0 = time.perf_counter()
        # All-resident configurations carry a device slot table (built by the engine from the
        # store's LRU). Using it here turns routing into one GPU gather instead of a host round-trip
        # plus a Python pass over every (layer, expert) pair in the chunk -- at a 2,048-token chunk
        # that pass was 77 % of prefill (NOTES 2026-09-12). The table is only valid while nothing is
        # evicted, which is exactly the pruned all-resident case; anything else takes the host path.
        lut = getattr(self, "slot_lut", None)
        if lut is not None and n_experts != 128:
            slots = lut[L][indices]
            self.stats["hits"] = self.stats.get("hits", 0) + indices.numel()
        else:
            slots = store.resolve(L, indices, prefill)
        # COLD PATH (DSV41_COLD_POOL). resolve() has put any expert whose bytes live in the mapped
        # pool into store.cold_this_call; those are computed from the pool in a second phase and the
        # promotion is issued afterwards. The lookup is inline rather than a helper so nothing new is
        # defined at module level here -- doing that once terminated the Model class body and moved
        # thirteen methods, including forward(), out of it, which ast.parse accepts happily.
        if os.environ.get("DSV41_COLD_TRACE", "0") == "1" and n_experts != 128:
            # Per (call, layer): a hash of the ROUTE and of the SLOTS it resolved to. COLD_VERIFY
            # compares the split against a reference inside one arm, so it cannot see the two arms
            # being fed different inputs -- which is exactly what is left after it found zero
            # differing layers while the tokens still differed. Diffing this between arms localises
            # the first divergence to routing, to slot assignment, or to neither.
            import hashlib as _h
            _i = getattr(self, "_ct_i", 0) + 1
            self._ct_i = _i
            _r = _h.sha256(indices.to("cpu").numpy().tobytes()).hexdigest()[:10]
            _s = _h.sha256(slots.to("cpu").numpy().tobytes()).hexdigest()[:10]
            self._ct_pend = (_i, L, _r, _s)
            if _i <= 200000:
                # the expert->slot MAP around the known first divergence, not just its hash: a hash
                # says the slots differ, the map says WHICH expert moved and where to.
                _extra = ""
                if 2170 <= _i <= 2186:
                    _e = indices.to("cpu").numpy().reshape(-1)
                    _sl = slots.to("cpu").numpy().reshape(-1)
                    _extra = " map " + ",".join(f"{a}:{b}" for a, b in
                                                sorted(set(zip(_e.tolist(), _sl.tolist()))))
                print(f"  CT {_i:04d} L{L:02d} route {_r} slots {_s}{_extra}", flush=True)
        _rc = os.environ.get("DSV41_COLD_RESIDCHECK", "")
        if _rc and n_experts != 128:
            # RESIDENCY VALIDATION. Every other axis is exhausted, and this tests the one claim left:
            # that each logical mapping points at byte-correct expert data. For each unique
            # (expert, slot) this layer resolved, load the expert's record from the pack into a scratch
            # slot with the SHIPPED loader and compare it against what the live arena slot holds. It
            # depends on no theory of how a slot might have gone wrong.
            _lo, _hi = (int(x) for x in _rc.split("-"))
            _i = getattr(self, "_ct_i", 0)
            if _lo <= _i <= _hi and getattr(store, "cb3_cache", None) is not None:
                import torch as _t, cb3_moe as _C3x
                _sc = getattr(self, "_rc_arena", None)
                if _sc is None:
                    _sc = self._rc_arena = _C3x.CB3ArenaV2(1, arena.device,
                                                           packed_scales=arena.packed_scales)
                    _sc.sim = getattr(arena, "sim", None)
                    self._rc_buf = _t.empty(store.cb3_cache.record, dtype=_t.uint8)
                _e = indices.to("cpu").numpy().reshape(-1)
                _sl = slots.to("cpu").numpy().reshape(-1)
                for _ex, _slt in sorted(set(zip(_e.tolist(), _sl.tolist()))):
                    store.cb3_cache.read_into(memoryview(self._rc_buf.numpy()), L, _ex)
                    store.cb3_cache.load_slot(_sc, 0, self._rc_buf, non_blocking=False)
                    _t.cuda.synchronize()
                    _badp = [nm for nm in _C3x.PLANE_ORDER
                             if not _t.equal(getattr(_sc, nm)[0], getattr(arena, nm)[_slt])]
                    if _badp:
                        _src = "cold-inflight" if (store.cold is not None and
                                                   (L, _ex) in store.cold.promo._inflight) else (
                               "transient" if (L, _ex) in getattr(store, "transient_map", {}) else
                               "lru" if (L, _ex) in store.lru else "UNMAPPED")
                        print(f"  RESID {_i:04d} L{L:02d} expert {_ex} slot {_slt} src {_src} "
                              f"WRONG planes {','.join(_badp)}", flush=True)
        cold_of = None
        if getattr(store, "cold", None) is not None and getattr(store, "cold_this_call", None):
            infl = store.cold.promo._inflight
            cold_of = {}
            for _e, _c in store.cold_this_call.items():
                _ent = infl.get((store._cold_layer, _e))
                if _ent is not None:
                    cold_of[_ent[2]] = _c            # hot slot -> cold slot
        if cold_of:
            import cb3_moe as _C3
            routed = _C3.moe_forward_cold_split(y, slots, weights, arena, store.cold.arena,
                                                cold_of, a.swiglu_limit).float()
            store.cold_finish_layer(arena, stream=getattr(store, "cold_stream", None))
            if os.environ.get("DSV41_COLD_VERIFY", "0") == "1":
                # With DSV41_COLD_SYNC=1 the promotions have landed by here, so the hot arena holds
                # every expert this layer used and an ordinary call over it is the reference. Compares
                # the split against it in situ, which is the only way to find WHICH layer differs --
                # the microbenchmark says the split is bitwise, so if the engine disagrees the
                # difference is in what the engine feeds it, not in the split itself.
                import torch as _t
                _ref = self.moe_fn(y, slots, weights, arena, a.swiglu_limit).float()
                if not _t.equal(_ref, routed):
                    _d = (_ref - routed).abs()
                    _n = int((_ref != routed).sum())
                    print(f"  COLD-VERIFY L{L}: {_n}/{_ref.numel()} differ, max|d| "
                          f"{_d.max().item():.6g}; cold_of {len(cold_of)} of "
                          f"{len(set(slots.flatten().tolist()))} slots", flush=True)
        else:
            routed = self.moe_fn(y, slots, weights, arena, a.swiglu_limit).float()
        shared = R.expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, a.swiglu_limit).float()
        if os.environ.get("DSV41_COLD_TRACE", "0") == "1" and getattr(self, "_ct_pend", None):
            import hashlib as _h2
            _i, _L, _r, _s = self._ct_pend
            self._ct_pend = None
            _o = _h2.sha256(routed.to("cpu").float().numpy().tobytes()).hexdigest()[:10]
            print(f"  CR {_i:04d} L{_L:02d} routed {_o} cold {len(cold_of) if cold_of else 0}",
                  flush=True)
        self._tap("moe_routed", L, routed); self._tap("moe_shared", L, shared)
        out = routed + shared
        self.stats["moe_s"] += time.perf_counter() - t0
        return out.to(y.dtype)

    def block_attn(self, h, pre_mix, w, L, S, sh, ring, freqs, mtp_extra=None, win_lo: int = 0):
        """The attention half of a block: everything up to and including its residual mix.
        Returns (h, attn_pre); `attn_pre` is what the FFN half needs as its own pre-mix."""
        with nvtx_range("attn"):
            a = self.args
            residual = h
            attn_pre, attn_post, attn_comb = R.hc_mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, a)
            y = R.hc_pre(h, pre_mix)
            y = R.rmsnorm(y, w.attn_norm, a.norm_eps)
            t0 = time.perf_counter()
            y = self.attention(y, w, L, S, sh, ring, freqs, mtp_extra, win_lo=win_lo)
            self.stats["attn_s"] += time.perf_counter() - t0
            return R.hc_post(y, residual, attn_post, attn_comb), attn_pre

    def block_ffn_in(self, h, attn_pre, w):
        """The FFN half's input and the mixes its output needs. Returns (y, ffn_pre, post, comb)."""
        with nvtx_range("ffn"):
            a = self.args
            ffn_pre, ffn_post, ffn_comb = R.hc_mixes(h, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, a)
            y = R.hc_pre(h, attn_pre)
            return R.rmsnorm(y, w.ffn_norm, a.norm_eps), ffn_pre, ffn_post, ffn_comb

    def block(self, h, pre_mix, w, L, S, sh, ring, freqs, prefill, store, arena, n_experts, mtp_extra=None,
              win_lo: int = 0):
        h, attn_pre = self.block_attn(h, pre_mix, w, L, S, sh, ring, freqs, mtp_extra, win_lo=win_lo)
        residual = h
        y, ffn_pre, ffn_post, ffn_comb = self.block_ffn_in(h, attn_pre, w)
        y = self.moe(y, w, L, prefill, store, arena, n_experts)
        h = R.hc_post(y, residual, ffn_post, ffn_comb)
        return h, ffn_pre

    # ------------------------------------------------------- layer-major encoder prefill
    @torch.inference_mode()
    def encoder_prefill_layer_major(self, ids: torch.Tensor, S0: int, store, arena, n_experts: int):
        """The CED encoder half, visiting LAYERS outermost and chunks innermost.

        Chunk-major prefill reads a layer's experts once per chunk. Measured on the real routes:
        74.2 % of prefill expert loads at an 11.3k prompt are a re-read, 89.3 % at 27.2k, and the
        engine moves 327.8 GB / 772.7 GB where the union of what it actually needs is 84.5 GB /
        82.3 GB (`notes/layer-major-prefill.md`). Transposing the loops reads each expert once per
        layer instead, and the floor is context-INDEPENDENT because it is bounded by the number of
        distinct (layer, expert) pairs, not by the chunk count.

        What stays exactly as it was, deliberately:
          * attention is still chunked at MAX_CHUNK, and the chunks of a layer still run in order --
            chunk k's attention reads the KV that chunks < k wrote. Enlarging the chunk is a
            different (and measured-worse) lever: 8192 reads 58 % less and is 16 % slower.
          * the MoE kernel still runs per chunk. Only the ROUTE and the RESOLVE are hoisted, so the
            transient ring holds a layer's expert set once instead of refilling it per chunk. That
            keeps the working set at one chunk's activations rather than the whole prompt's.

        Costs one `[T, hc_mult, dim]` bf16 buffer for the prompt: 0.46 GB at 11k, 1.34 GB at 32k.
        """
        a = self.args
        assert self.c.len == S0, (self.c.len, S0)
        P = ids.size(0)
        bounds = [(s, min(s + MAX_CHUNK, P)) for s in range(0, P, MAX_CHUNK)]
        last_L = a.candidate_source_layer

        t0 = time.perf_counter()
        hashes = [self.hash_state(ids[lo:hi][None], S0 + lo)[0] if self.hash_state is not None else None
                  for lo, hi in bounds]
        self.stats["engram_s"] += time.perf_counter() - t0

        H, PM = [], []
        for lo, hi in bounds:
            h = self.W.embed[ids[lo:hi]].unsqueeze(1).repeat(1, a.hc_mult, 1)
            pm = torch.zeros(hi - lo, a.hc_mult, device=self.dev)
            pm[:, 0] = 1.0
            H.append(h); PM.append(pm)
        # `Shared` is per-CHUNK but lives ACROSS layers: a kv-source layer fills ckv/ik/ratio and
        # the layers above it reuse that same compressed cache and candidate pool (layers 21..23
        # reuse layer 20's top-k, 24..39 search inside its candidate set). Chunk-major gets this for
        # free because one `Shared` is threaded down a chunk's whole layer stack. Transposing the
        # loops means holding one per chunk for the entire pass -- a fresh one per (layer, chunk)
        # trips `_compressed`'s own `sh.ratio == r` assert at the first layer that reuses.
        SH = [Shared() for _ in bounds]
        # Prompt-cache checkpoints. `Caches.checkpoint(n)` wants every layer's compressor state as
        # of position n, and chunk-major can take it in one go because the whole stack is at n at
        # the same moment. Layer-major never is -- layer L is at the end of the prompt while L+1 is
        # still at the start -- so the pieces are collected as each layer passes each boundary and
        # assembled at the end. Without this the pass takes ONE checkpoint, at the resume point, and
        # a later turn has nothing to resume from: the cache is silently dead.
        pend = {S0 + lo: {} for lo, _ in bounds}
        # the replay needs layer 20's top-k and candidate pool for the prompt TAIL only, so each
        # chunk keeps at most `window_size` rows of them rather than its whole [T, n_c] mask
        tails = []

        phase = [] if LM_PHASES else None
        for L in range(last_L + 1):
            w = self.W.layers[L]
            t_layer = time.perf_counter()
            post = []           # per chunk: (residual, ffn_post, ffn_comb)
            ys, idxs, wts = [], [], []
            lut = getattr(self, "slot_lut", None)
            lut = lut if (lut is not None and n_experts != 128) else None
            part = {}           # ci -> slots, when EARLY_SUBMIT resolved this chunk on its own
            for ci, (lo, hi) in enumerate(bounds):
                h = H[ci]
                if L in self.W.engram:
                    with nvtx_range("engram"):
                        te = time.perf_counter()
                        li = list(a.engram_layer_ids).index(L)
                        rows = self.engram_rows(L, hashes[ci][:, li, :])
                        h = R.engram_forward(h, rows, self.W.engram[L], a)
                        self.stats["engram_s"] += time.perf_counter() - te
                pv = self.c.pending.get(L)
                pend[S0 + lo][L] = None if pv is None else (pv[0].clone(), pv[1].clone())
                sh = SH[ci]
                h, attn_pre = self.block_attn(h, PM[ci], w, L, S0 + lo, sh, self.c.win[L],
                                              self.freqs_c if w.ratio else self.freqs_w)
                y, ffn_pre, ffn_post, ffn_comb = self.block_ffn_in(h, attn_pre, w)
                ys.append(y); post.append((h, ffn_post, ffn_comb)); PM[ci] = ffn_pre
                i_c, w_c = self.moe_route(y, w, L, n_experts)
                idxs.append(i_c); wts.append(w_c)
                if EARLY_SUBMIT == "synconly" and lut is None:
                    # the barrier alone: the same blocking device->host copy of the route ids that
                    # resolve() opens with, and none of the work that follows it
                    i_c.to("cpu", dtype=torch.int32, non_blocking=False)
                elif EARLY_SUBMIT != "off" and lut is None and (EARLY_SUBMIT == "all" or ci == 0):
                    # This chunk's expert ids are final the moment it has routed, so its reads can
                    # start while the remaining chunks are still doing attention. Measured on the
                    # recorded routes: chunk 0 alone names 85.4 % of the layer's ENTIRE expert set
                    # (worst layer 76.4 %), and there is ~0.74 s of remaining attention per layer
                    # against ~0.30 s of delivery -- 2.4x more lead time than the reads need.
                    # Slots are assigned synchronously inside resolve(); only the data is deferred.
                    part[ci] = store.resolve(L, i_c, True, defer=True)
                if L == last_L:
                    n = min(a.window_size, hi - lo)
                    tails.append((sh.topk[-n:],
                                  sh.candidates[-n:] if sh.candidates is not None else None))
            t_routed = time.perf_counter()
            # ONE resolve for the whole layer: every missing expert is read exactly once, and its
            # slot stays valid for every chunk below because nothing else claims the ring meanwhile.
            if lut is not None:
                all_slots = lut[L][torch.cat(idxs)]
                self.stats["hits"] = self.stats.get("hits", 0) + sum(i.numel() for i in idxs)
            elif part and len(part) == len(ys):
                # every chunk resolved itself as it routed; nothing left to assign, so this is the join
                store.join_pending()
                all_slots = torch.cat([part[ci] for ci in range(len(ys))])
            elif part:
                # chunk0 mode: chunk 0's reads are in flight and cover ~85 % of the layer. Join them,
                # then one union resolve picks up the ~15 % tail the later chunks discovered -- the
                # early experts are transient-ring hits by then, so each is still read once per layer.
                store.join_pending()
                all_slots = store.resolve(L, torch.cat(idxs), True)
            else:
                all_slots = store.resolve(L, torch.cat(idxs), True)
            t_resolved = time.perf_counter()
            off = 0
            for ci, y in enumerate(ys):
                n = y.size(0)
                out = self.moe_apply(y, all_slots[off:off + n], wts[ci], w, arena)
                off += n
                resid, ffn_post, ffn_comb = post[ci]
                H[ci] = R.hc_post(out, resid, ffn_post, ffn_comb)
            if phase is not None:
                torch.cuda.synchronize()
                phase.append((L, t_routed - t_layer, t_resolved - t_routed,
                              time.perf_counter() - t_resolved))

        # `c.len` is written once, at the end. Verified rather than assumed: nothing on the encoder
        # path reads it -- `_compressed` derives its indices from S, T and `pending[L]` alone, and
        # `pending` is per-layer, so walking chunks inside a layer advances exactly the state that
        # layer owns. The only readers are `forward`'s assert and `Caches.rollback`, neither of
        # which runs during this pass.
        if phase is not None:
            a_, b_, c_ = (sum(p[i] for p in phase) for i in (1, 2, 3))
            tot = a_ + b_ + c_
            print(f"[layer-major phases] {len(phase)} encoder layers, {tot:.1f}s: "
                  f"attn+route {a_:.1f}s ({100*a_/tot:.0f}%), resolve {b_:.1f}s "
                  f"({100*b_/tot:.0f}%), ffn {c_:.1f}s ({100*c_/tot:.0f}%)", flush=True)
            print("  per layer attn+route/resolve/ffn (s): "
                  + " ".join(f"L{p[0]}:{p[1]:.2f}/{p[2]:.2f}/{p[3]:.2f}" for p in phase[:8])
                  + (" ..." if len(phase) > 8 else ""), flush=True)
        for n, per_layer in pend.items():
            if len(per_layer) == last_L + 1:      # every encoder layer passed this boundary
                self.c._ckpt[n] = dict(per_layer)
        self.c.len = S0 + P
        self.stats["tokens"] += P
        for ci, (lo, hi) in enumerate(bounds):
            sh = Shared()
            sh.topk, sh.candidates = tails[ci]
            n = tails[ci][0].size(0)
            self._rep_keep(H[ci][-n:], PM[ci][-n:], sh, S0 + hi - n, n)
        return None, None

    # ------------------------------------------------------------------ SWA bounded replay
    def begin_prompt(self):
        """Drop whatever the previous prompt left in the replay buffer."""
        self._rep = {"h": [], "pre_mix": [], "topk": [], "cand": []}
        self._rep_end = 0

    def _rep_keep(self, h, pre_mix, sh: Shared, S: int, T: int):
        """Remember the last `window_size` encoder outputs of the prompt so far.

        Only the tail is ever needed, so each chunk contributes at most `window_size` rows and the
        buffer is trimmed as soon as it has more than that.
        """
        w = self.args.window_size
        n = min(w, T)
        sl = slice(T - n, T)
        r = self._rep
        r["h"].append(h[sl]); r["pre_mix"].append(pre_mix[sl])
        # layers 21..23 reuse layer 20's top-k and layers 24..39 search inside layer 20's candidate
        # pool, and both are computed per query -- so they belong to the queries, not to the caches,
        # and the replay has to carry them across from the encoder pass instead of recomputing them.
        r["topk"].append(sh.topk[sl])
        r["cand"].append(sh.candidates[sl] if sh.candidates is not None else None)
        self._rep_end = S + T
        while sum(t.size(0) for t in r["h"]) - r["h"][0].size(0) >= w:
            for k in r:
                r[k].pop(0)

    def _rep_tail(self):
        w = self.args.window_size
        r = self._rep
        h = torch.cat(r["h"])[-w:]
        pre_mix = torch.cat(r["pre_mix"])[-w:]
        topk = torch.cat(r["topk"])[-w:]
        cands = r["cand"]
        cand = None
        if cands and cands[0] is not None:
            width = max(c.size(1) for c in cands)
            # older chunks scored fewer compressed columns; a query can only ever see columns below
            # its own position, all of which are inside its own chunk's width, so padding the rest
            # with False changes nothing that is reachable.
            cand = torch.cat([c if c.size(1) == width else F.pad(c, (0, width - c.size(1)), value=False)
                              for c in cands])[-w:]
        return h, pre_mix, topk, cand, self._rep_end - h.size(0)

    @torch.inference_mode()
    def decoder_replay(self, need_logits: bool = True):
        """Decoder SWA Bounded Replay (tech report 2.2 / 3.2.2).

        Under CED the decoder's global KV is projected from the last encoder layer's hidden state,
        which the encoder pass has already written for every prompt position. The only thing the
        decoder layers still owe the first decode steps is their own sliding-window KV -- so they
        are run over the last `window_size` prompt tokens only, with SWA truncated to that segment,
        instead of over the whole prompt. The prompt's final logits come from this pass.
        """
        a = self.args
        h, pre_mix, topk, cand, S = self._rep_tail()
        T = h.size(0)
        sh = Shared()
        src = a.candidate_source_layer
        sh.ckv, sh.ik, sh.ratio = self.c.ckv[src], self.c.ik[src], self.args.compress_ratios[src]
        sh.topk, sh.candidates = topk, cand
        main_hiddens = []
        for L in range(src + 1, len(self.W.layers)):
            w = self.W.layers[L]
            if L in a.dspark_target_layer_ids:
                main_hiddens.append(h.float().mean(dim=1))
            freqs = self.freqs_c if w.ratio else self.freqs_w
            h, pre_mix = self.block(h, pre_mix, w, L, S, sh, self.c.win[L], freqs, True, self.store,
                                    self.arena, a.n_routed_experts, win_lo=S)
            self._tap("h", L, h); self._tap("pre_mix", L, pre_mix)
        self.last_h, self.last_pre_mix = h, pre_mix
        logits = None
        if need_logits:
            x = R.hc_pre(h, pre_mix)
            x = R.rmsnorm(x, self.W.norm, a.norm_eps)
            logits = R.head_logits(x, self.W.head)
        self.stats["replay_tokens"] = self.stats.get("replay_tokens", 0) + T
        return logits, (torch.cat(main_hiddens, dim=-1) if main_hiddens else None), S

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor, S: int, prefill: bool, need_logits: bool = True,
                encoder_only: bool = False):
        """ids: [T] token ids at positions S..S+T-1. Returns (logits [T, V] fp32 or None, main_hidden [T, 15360]).
        Caches must be valid for positions < S (self.c.len == S).

        ``encoder_only`` stops after the candidate-source layer (the last layer that writes global
        KV, layer 20 here): everything above it is replayed once over the prompt tail by
        `decoder_replay`. It returns (None, None) -- there are no logits and no DSpark hidden
        states below layer 37.
        """
        a = self.args
        assert self.c.len == S, (self.c.len, S)
        T = ids.size(0)
        assert T <= MAX_CHUNK, (T, MAX_CHUNK)
        t0 = time.perf_counter()
        hashes = self.hash_state(ids[None], S)[0] if self.hash_state is not None else None  # [T, 2, 24]
        self.stats["engram_s"] += time.perf_counter() - t0
        h = self.W.embed[ids].unsqueeze(1).repeat(1, a.hc_mult, 1)
        pre_mix = torch.zeros(T, a.hc_mult, device=self.dev)
        pre_mix[:, 0] = 1.0
        sh = Shared()
        main_hiddens = []
        n_layers = len(self.W.layers)
        last = a.candidate_source_layer if encoder_only else n_layers - 1
        for L in range(last + 1):
            w = self.W.layers[L]
            if L in self.W.engram:
                with nvtx_range("engram"):
                    t0 = time.perf_counter()
                    li = list(a.engram_layer_ids).index(L)
                    rows = self.engram_rows(L, hashes[:, li, :])
                    self._tap("engram_rows", L, rows)
                    h = R.engram_forward(h, rows, self.W.engram[L], a)
                    self._tap("engram_out", L, h)
                    self.stats["engram_s"] += time.perf_counter() - t0
            if L in a.dspark_target_layer_ids:
                main_hiddens.append(h.float().mean(dim=1))
            freqs = self.freqs_c if w.ratio else self.freqs_w
            h, pre_mix = self.block(h, pre_mix, w, L, S, sh, self.c.win[L], freqs, prefill, self.store,
                                    self.arena, a.n_routed_experts)
            self._tap("h", L, h); self._tap("pre_mix", L, pre_mix)
        self.c.len = S + T
        self.stats["tokens"] += T
        if encoder_only:
            self._rep_keep(h, pre_mix, sh, S, T)
            return None, None
        logits = None
        self.last_h, self.last_pre_mix = h, pre_mix
        if need_logits and n_layers == a.n_layers:
            x = R.hc_pre(h, pre_mix)
            x = R.rmsnorm(x, self.W.norm, a.norm_eps)
            logits = R.head_logits(x, self.W.head)
        return logits, (torch.cat(main_hiddens, dim=-1) if main_hiddens else None)

    # ------------------------------------------------------------------ DSpark
    @torch.inference_mode()
    def dspark_seed(self, main_hidden: torch.Tensor, S: int):
        """Write the drafter's window KV for main positions S..S+M-1 from their main hiddens [M, 15360]."""
        a = self.args
        m0 = self.W.mtp[0]
        main_x = R.rmsnorm(R.qlinear(main_hidden.to(torch.bfloat16), m0.main_proj), m0.main_norm, a.norm_eps)
        M = main_x.size(0)
        pos = torch.arange(S, S + M, device=self.dev)
        rd = a.rope_head_dim
        for k, w in enumerate(self.W.mtp):
            kv = R.rmsnorm(R.qlinear(main_x, w.wkv), w.kv_norm, a.norm_eps)
            kv = torch.cat([kv[:, :-rd], R.apply_rotary(kv[:, -rd:], self.freqs_w[S:S + M])], dim=-1)
            self.c.mtp_win[k][pos % RING] = kv

    @torch.inference_mode()
    def dspark_draft(self, tok: int, last_main_pos: int, temperature: float):
        """Draft block: returns (draft ids [B], draft probs [B, V] fp32 at the given temperature,
        confidence [B]) for B = the DSpark block size. Queries sit at last_main_pos+1 .. +B."""
        a = self.args
        from engine.fastdecode import T_DRAFT as B   # DSV41_BLOCK, default the checkpoint's 5
        ids = torch.full((B,), 128799, dtype=torch.long, device=self.dev)
        ids[0] = tok
        h = self.W.embed[ids].unsqueeze(1).repeat(1, a.hc_mult, 1)
        pre_mix = torch.zeros(B, a.hc_mult, device=self.dev); pre_mix[:, 0] = 1.0
        S = last_main_pos + 1
        sh = Shared()
        for k, w in enumerate(self.W.mtp):
            h, pre_mix = self.block(h, pre_mix, w, a.n_layers + k, S, sh, self.c.mtp_win[k], self.freqs_w, False,
                                    self.W.dspark_store, self.W.dspark_arena, 128, mtp_extra=last_main_pos)
        w = self.W.mtp[2]
        x = R.hc_pre(h, pre_mix)
        # The reference feeds the UN-normed hc_pre output to the confidence head and the normed one
        # to the LM head (inference/model.py::DSparkBlock.forward_head), so keep both.
        x_pre = x
        x = R.rmsnorm(x, w.norm, a.norm_eps)
        logits = R.head_logits(x, self.W.head)  # [B, V] fp32
        out = torch.empty(B + 1, dtype=torch.long, device=self.dev)
        out[0] = tok
        probs = []
        embeds = []
        for i in range(B):
            e = w.markov_embed[out[i]]  # [256]
            bias = F.linear(e.to(torch.bfloat16)[None], w.markov_head).float()[0]  # [V]
            lg = logits[i] + bias
            if temperature <= 0:
                p = torch.zeros_like(lg); p[lg.argmax()] = 1.0
                nxt = lg.argmax()
            else:
                p = torch.softmax(lg / temperature, dim=-1)
                nxt = torch.multinomial(p, 1)[0]
            out[i + 1] = nxt
            probs.append(p)
            embeds.append(e.float())
        # DSparkConfidenceHead returns the raw projection (no sigmoid); adaptive verification is off
        # in this engine, so it is reported, not acted on.
        conf = (torch.cat([x_pre.float(), torch.stack(embeds)], dim=-1) @ w.conf_proj.T).squeeze(-1)
        return out[1:], torch.stack(probs), conf
