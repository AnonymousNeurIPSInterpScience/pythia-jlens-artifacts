#!/usr/bin/env python3
"""t17_reaggregate.py — E18. Re-score every read cell under NON-EXISTENTIAL aggregations.

WHY. The anchor's §A.6 pass@k is existential: `recovered_k(i) = 1[EXISTS l : rank_l <= k]`.
`t15` showed that rewards DIVERSITY over ACCURACY -- a layer-shuffled J beats the real one under
min-over-layers while losing on 18 of 23 individual layers. `t15b` then showed the
generic/layer-specific partition INVERTS when the aggregation changes. So every read number in the
programme is currently reported under a metric known to be defective.

This re-scores the same forward passes under four aggregations, from ONE pass per prompt:

  min      min over layers        the anchor's metric (existential; kept for comparability)
  best1L   best single layer      which layer alone is most informative
  mean     mean over layers       average per-layer quality
  persist  persistence at K       rank <= k at >= K band layers, K a FRACTION of band depth
                                  (the fix; external review B3 confirms it has no canonical name)

DECISION RULE, fixed in advance (RESEARCH_NOTES sec13 E18):
  If the SIGN of the layer-specific component (jlens - shuffled) is stable across all
  aggregations on a set, that set's classification stands. Otherwise it is reported as
  AGGREGATION-DEPENDENT and excluded from any claim.

EXPLORATORY. Re-analysis of already-burned cells; no new model access beyond re-running the
readout, no refit, no GPU.
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


def auc(curve) -> float:
    a = sum((curve[i] + curve[i + 1]) / 2 * (_LOGK[i + 1] - _LOGK[i]) for i in range(len(KS) - 1))
    return a / _SPAN


def band_of(n_layers: int, layers: list[int]) -> list[int]:
    """The anchor's normalised workspace band, L38-L92 of a 0-100 scale, clipped to fitted layers."""
    # floor, not round: the declared rule is floor(0.38L)..floor(0.92L) (tests/test_band_rule.py).
    lo, hi = int(0.38 * n_layers), int(0.92 * n_layers)
    b = [l for l in layers if lo <= l <= hi]
    return b or layers


@torch.no_grad()
def collect_ranks(model, transport, eval_name, layers):
    """[n_pairs, n_layers] of min-over-synonym rank. One forward per prompt, all layers at once."""
    rows = []
    for item in load_eval(eval_name):
        prompt = item["prompt"].rstrip()
        pos = readout_position(model.tokenizer, eval_name, prompt)
        out = readout(model, transport, prompt, layers, pos)
        for word in item["intermediates"]:
            ids = token_ids_of(model.tokenizer, word)
            if not ids:
                continue
            rows.append([rank_of(out[l], ids) for l in layers])
    return torch.tensor(rows, dtype=torch.float)


def aggregate(R: torch.Tensor, layers: list[int], band: list[int], kfrac: float) -> dict:
    """R: [n_pairs, n_layers] ranks -> AUC under each aggregation."""
    idx = {l: i for i, l in enumerate(layers)}
    Rb = R[:, [idx[l] for l in band]]
    hits = lambda M, k: (M <= k).float()                                        # noqa: E731
    out = {}
    out["min"] = auc([float(hits(R.min(dim=1).values, k).mean()) for k in KS])
    per_layer = {str(l): auc([float(hits(R[:, idx[l]], k).mean()) for k in KS]) for l in layers}
    out["best1L"] = max(per_layer.values())
    out["best1L_layer"] = int(max(per_layer, key=per_layer.get))
    out["mean"] = sum(per_layer.values()) / len(per_layer)
    out["per_layer"] = {k: round(v, 4) for k, v in per_layer.items()}
    K = max(1, int(round(kfrac * len(band))))
    out["persist_K"] = K
    out["persist_band_size"] = len(band)
    out["persist"] = auc([float(((Rb <= k).sum(dim=1) >= K).float().mean()) for k in KS])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kfrac", type=float, default=0.5,
                    help="persistence threshold as a FRACTION of band depth, so K is comparable "
                         "across the ladder rather than an absolute layer count")
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda. The read path is pure forward passes plus a\n                          matmul against J, so it parallelises well on GPU.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.device)
    try:
        model = from_hf(hf, tok)
    except ValueError:
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    lens = JacobianLens.load(a.lens)
    # lens tensors must live where the model lives (see t20 note)
    for _l, _J in lens.jacobians.items():
        lens.jacobians[_l] = _J.to(a.device)
    layers = list(lens.source_layers)
    band = band_of(model.n_layers, layers)
    transports, perm = build_transports(lens, layers, a.seed)
    arms = ("logit", "jlens", "shuffled")
    AGGS = ("min", "best1L", "mean", "persist")

    results, verdicts = {}, {}
    for ev in EVAL_SETS:
        R = {t: collect_ranks(model, transports[t], ev, layers) for t in arms}
        agg = {t: aggregate(R[t], layers, band, a.kfrac) for t in arms}
        results[ev] = {t: {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in agg[t].items() if k != "per_layer"} for t in arms}
        results[ev]["per_layer_jlens"] = agg["jlens"]["per_layer"]
        results[ev]["per_layer_shuffled"] = agg["shuffled"]["per_layer"]
        spec = {g: agg["jlens"][g] - agg["shuffled"][g] for g in AGGS}
        gen = {g: agg["shuffled"][g] - agg["logit"][g] for g in AGGS}
        results[ev]["layer_specific"] = {g: round(spec[g], 4) for g in AGGS}
        results[ev]["generic"] = {g: round(gen[g], 4) for g in AGGS}
        signs = {1 if spec[g] > 0 else (-1 if spec[g] < 0 else 0) for g in AGGS}
        stable = len(signs) == 1
        verdicts[ev] = ("STABLE: layer-specific" if stable and spec["mean"] > 0 else
                        "STABLE: not layer-specific" if stable else
                        "AGGREGATION-DEPENDENT — excluded from any claim")
        results[ev]["verdict"] = verdicts[ev]
        print(f"  {ev:13s} specific " +
              "  ".join(f"{g}={spec[g]:+.4f}" for g in AGGS) + f"   {verdicts[ev]}", flush=True)

    n_stable = sum(1 for v in verdicts.values() if v.startswith("STABLE"))
    n_win = {g: sum(1 for ev in EVAL_SETS
                    if results[ev]["jlens"][g] > results[ev]["logit"][g]) for g in AGGS}
    out = {
        "experiment": "T17 / E18 — re-score reads under non-existential aggregations",
        "status": "EXPLORATORY — re-analysis of burned cells; no refit",
        "decision_rule": ("sign of (jlens - shuffled) stable across all four aggregations => the "
                          "set's classification stands; otherwise AGGREGATION-DEPENDENT and "
                          "excluded from any claim"),
        "model": a.model, "lens": a.lens, "layers": layers, "band": band,
        "persistence_kfrac": a.kfrac, "seed": a.seed,
        "aggregations": list(AGGS),
        "n_win_vs_logit_by_aggregation": n_win,
        "n_sets_stable": n_stable, "n_sets_aggregation_dependent": len(EVAL_SETS) - n_stable,
        "derangement_layer_map": perm,
        "results": results,
        "env": {"transformers": transformers.__version__, "torch": torch.__version__},
        "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n  n_win vs logit: " + "  ".join(f"{g}={n_win[g]}/6" for g in AGGS))
    print(f"  stable: {n_stable}/6   aggregation-dependent: {len(EVAL_SETS)-n_stable}/6")
    print(f"wrote {a.out} ({out['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
