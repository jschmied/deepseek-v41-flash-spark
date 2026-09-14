"""Paired per-token NLL between two expert_trace.py arms.

teacher_forced_check.json only stores means, so a 0.03-nat difference cannot be read
as significant or not.  Both arms save state/after_layer39.pt for the *same* corpus in
the *same* order, so the final norm+head can be re-run over each and the two NLL
vectors paired token by token; that removes the between-sequence variance which
dominates the unpaired standard error.

It also answers the obvious follow-up when the *lower-precision* arm has the lower
NLL: is that information, or only calibration?  Requantization noise softens the
output distribution, and a softer distribution scores better under NLL while being no
better at picking the mode.  So the baseline's logits are also swept over a single
temperature; if temperature alone reaches the other arm's NLL, the win was softening.

  python tools/paired_nll.py --model-dir ... --corpus ... --a <arm> --b <arm>
"""
import argparse, json, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v41_ref as R
from expert_trace import ShardGetter, load_corpus

TEMPS = [round(0.60 + 0.02 * k, 2) for k in range(96)]   # 0.60 .. 2.50


def run_arm(arm, seqs, norm_w, head, args, dev, sweep=False):
    """-> {seq_id: (nll[T-1], top1[T-1], entropy[T-1])}, and {T: total nll} if sweep."""
    st = torch.load(os.path.join(arm, "state", "after_layer39.pt"), map_location="cpu")
    assert st["layer"] == 39, arm
    out, tot = {}, {T: 0.0 for T in TEMPS}
    for s, rec in zip(seqs, st["states"]):
        h = R.hc_pre(rec["h"].to(dev), rec["pre_mix"].to(dev))
        h = R.rmsnorm(h, norm_w, args.norm_eps)
        logits = (h.float() @ head.T)[:-1]                       # [T-1, V]
        tgt = torch.tensor(s["ids"], device=dev)[1:]
        lp = torch.log_softmax(logits, dim=-1)
        out[s["id"]] = (-lp.gather(1, tgt[:, None]).squeeze(1).cpu(),
                        (logits.argmax(-1) == tgt).float().cpu(),
                        -(lp.exp() * lp).sum(-1).cpu())
        if sweep:
            for T in TEMPS:
                l = torch.log_softmax(logits / T, dim=-1)
                tot[T] += float(-l.gather(1, tgt[:, None]).sum())
        del logits, lp
    return out, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--a", required=True, help="baseline arm directory")
    ap.add_argument("--b", required=True, help="arm under test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-temp-sweep", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = a.device
    index = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))
    args = R.Args.from_json(os.path.join(a.model_dir, "inference", "config.json"))
    get = ShardGetter(a.model_dir, index)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    seqs = load_corpus(a.corpus, tok, a.max_len)

    norm_w = get("norm.weight").to(dev).to(torch.bfloat16)
    head = get("head.weight").to(dev).float()

    A, sweep = run_arm(a.a, seqs, norm_w, head, args, dev, sweep=not a.no_temp_sweep)
    B, _ = run_arm(a.b, seqs, norm_w, head, args, dev)

    res = {}
    for cat in sorted({s["category"] for s in seqs}):
        ids = [s["id"] for s in seqs if s["category"] == cat]
        na, nb = torch.cat([A[i][0] for i in ids]), torch.cat([B[i][0] for i in ids])
        ta, tb = torch.cat([A[i][1] for i in ids]), torch.cat([B[i][1] for i in ids])
        ha, hb = torch.cat([A[i][2] for i in ids]), torch.cat([B[i][2] for i in ids])
        d = nb - na
        n = d.numel()
        sem = float(d.std(unbiased=True) / n ** 0.5)
        res[cat] = {
            "n": n,
            "nll_a": float(na.mean()), "nll_b": float(nb.mean()),
            "delta_nll": float(d.mean()), "sem_paired": sem,
            "t": float(d.mean() / sem) if sem else 0.0,
            "top1_a": float(ta.mean()), "top1_b": float(tb.mean()),
            "top1_agree": float((ta == tb).float().mean()),
            "mcnemar_a_only": int(((ta > 0) & (tb == 0)).sum()),
            "mcnemar_b_only": int(((ta == 0) & (tb > 0)).sum()),
            "entropy_a": float(ha.mean()), "entropy_b": float(hb.mean()),
        }
    if not a.no_temp_sweep:
        # one temperature for the whole corpus; report against the pooled NLL
        n_all = sum(v["n"] for v in res.values())
        curve = {str(T): sweep[T] / n_all for T in TEMPS}
        best = min(curve, key=curve.get)
        nll_b_all = sum(v["nll_b"] * v["n"] for v in res.values()) / n_all
        res["_temp_sweep_on_a"] = {"best_T": float(best), "nll_a_at_best_T": curve[best],
                                   "nll_a_at_T1": curve["1.0"] if "1.0" in curve else float("nan"), "nll_b_at_T1": nll_b_all,
                                   "curve": curve}
    print(json.dumps({k: v for k, v in res.items() if k != "_temp_sweep_on_a"}, indent=1))
    if "_temp_sweep_on_a" in res:
        s = res["_temp_sweep_on_a"]
        print(f"temp sweep on A: best T={s['best_T']} -> {s['nll_a_at_best_T']:.4f} "
              f"(T=1: {s['nll_a_at_T1']:.4f}; B at T=1: {s['nll_b_at_T1']:.4f})")
    if a.out:
        json.dump({"a": a.a, "b": a.b, "cats": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
