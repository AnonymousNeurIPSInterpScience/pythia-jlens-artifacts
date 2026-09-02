#!/usr/bin/env python3
"""t13_bootstrap_controls.py — paired item bootstrap on the LAYER-SPECIFIC component.

E1 (`t13_transport_controls.py`) found that at 410M a layer-shuffled `J` beats the logit lens on
5 of 6 sets, and that `jlens > shuffled` on only 3 of 6. That decomposition is now the load-bearing
claim of the whole read result, and it was reported as six point estimates with no uncertainty —
the exact failure `t7_bootstrap_audit.py` was written to fix for T7.

This bootstraps the decomposition, paired over items:

    total     = AUC(jlens)    - AUC(logit)       what T7 reported
    generic   = AUC(shuffled) - AUC(logit)       what ANY fitted Jacobian buys
    specific  = AUC(jlens)    - AUC(shuffled)    the workspace claim

All three arms are scored on the SAME items, so one resample of item indices is applied to all
three — the CI reflects item sampling only, not arm-vs-arm variance.

The question that matters: **is `specific` distinguishable from zero, and on how many sets?**

EXPLORATORY. Cells already burned per `docs/experiments/preregs/superseded/PREREG_PYTHIA_T7_v2.md` section 0.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

import torch

torch.set_num_threads(min(8, os.cpu_count() or 8))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jlens.hf import Layout, from_hf  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402
from anchor_evals import (EVAL_SETS, load_eval, readout, readout_position,  # noqa: E402
                          rank_of, token_ids_of)
from t13_transport_controls import build_transports  # noqa: E402

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")
KS = (1, 2, 5, 10, 25, 100)
_LOGK = [math.log(k) for k in KS]
_SPAN = _LOGK[-1] - _LOGK[0]
_W = torch.tensor([_LOGK[i + 1] - _LOGK[i] for i in range(len(KS) - 1)])


@torch.no_grad()
def per_pair_hits(model, transport, eval_name: str, layers) -> torch.Tensor:
    """[n_pairs, len(KS)] binary hits, flat-pooled over (item, intermediate) — the PAPER metric."""
    rows = []
    for item in load_eval(eval_name):
        prompt = item["prompt"].rstrip()
        pos = readout_position(model.tokenizer, eval_name, prompt)
        logits = readout(model, transport, prompt, layers, pos)
        for word in item["intermediates"]:
            ids = token_ids_of(model.tokenizer, word)
            if not ids:
                continue
            r = min(rank_of(logits[l], ids) for l in layers)
            rows.append([1.0 if r <= k else 0.0 for k in KS])
    return torch.tensor(rows)


def auc_of(curves: torch.Tensor) -> torch.Tensor:
    """curves: [..., len(KS)] mean hit-rates -> normalized AUC over log k."""
    return ((curves[..., :-1] + curves[..., 1:]) / 2 * _W).sum(-1) / _SPAN


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    try:
        model = from_hf(hf, tok)
    except ValueError:
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    lens = JacobianLens.load(a.lens)
    layers = list(lens.source_layers)
    transports, perm = build_transports(lens, layers, a.seed)
    arms = ("logit", "shuffled", "jlens")

    g = torch.Generator().manual_seed(a.seed)
    per_set = {}
    for ev in EVAL_SETS:
        hits = {t: per_pair_hits(model, transports[t], ev, layers) for t in arms}
        n = hits["logit"].shape[0]
        assert all(hits[t].shape[0] == n for t in arms), "arms disagree on scored pairs"

        idx = torch.randint(0, n, (a.n_boot, n), generator=g)
        boot = {t: auc_of(hits[t][idx].mean(dim=1)) for t in arms}
        point = {t: float(auc_of(hits[t].mean(0))) for t in arms}

        comps = {"total": ("jlens", "logit"), "generic": ("shuffled", "logit"),
                 "specific": ("jlens", "shuffled")}
        rec = {"n_pairs": n, **{f"auc_{t}": round(point[t], 4) for t in arms}}
        for name, (x, y) in comps.items():
            d = boot[x] - boot[y]
            lo, hi = torch.quantile(d, torch.tensor([0.025, 0.975])).tolist()
            pt = point[x] - point[y]
            rec[name] = {
                "delta": round(pt, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "p_sign_flip": round(float((d <= 0).float().mean() if pt > 0
                                           else (d >= 0).float().mean()), 4)}
        per_set[ev] = rec
        s = rec["specific"]
        print(f"  {ev:13s} total {rec['total']['delta']:+.4f}  generic "
              f"{rec['generic']['delta']:+.4f}  SPECIFIC {s['delta']:+.4f} "
              f"CI[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}] "
              f"{'*' if s['ci_excludes_zero'] else ' '}", flush=True)

    n_spec = sum(1 for v in per_set.values() if v["specific"]["delta"] > 0)
    n_spec_ci = sum(1 for v in per_set.values()
                    if v["specific"]["delta"] > 0 and v["specific"]["ci_excludes_zero"])
    n_gen_ci = sum(1 for v in per_set.values()
                   if v["generic"]["delta"] > 0 and v["generic"]["ci_excludes_zero"])
    out = {
        "experiment": "T13b — paired item bootstrap on the generic / layer-specific decomposition",
        "status": "EXPLORATORY — burned cells per PREREG_PYTHIA_T7_v2 section 0",
        "model": a.model, "lens": a.lens, "layers": layers,
        "seed": a.seed, "n_boot": a.n_boot,
        "bootstrap": "paired over (item,intermediate) pairs; one index draw applied to all 3 arms",
        "metric": "normalized pass@k AUC over log k, PAPER form (flat pool), ks=(1,2,5,10,25,100)",
        "derangement_layer_map": perm,
        "n_sets_specific_positive": n_spec,
        "n_sets_specific_CI_excludes_zero": n_spec_ci,
        "n_sets_generic_CI_excludes_zero": n_gen_ci,
        "headline": (f"layer-specific advantage is positive on {n_spec}/6 sets and survives a 95% "
                     f"paired bootstrap on {n_spec_ci}/6; the generic basis-change component "
                     f"survives on {n_gen_ci}/6"),
        "per_set": per_set,
        "env": {"transformers": transformers.__version__, "torch": torch.__version__,
                "device": "cpu", "dtype": "float32"},
        "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n  {out['headline']}")
    print(f"wrote {a.out}  ({out['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
