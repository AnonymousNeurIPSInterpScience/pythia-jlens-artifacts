#!/usr/bin/env python3
"""t15_shuffle_diagnostic.py — WHY does a layer-shuffled J still beat the logit lens?

The E1 result (`t13`) is surprising enough that it must be explained mechanistically or
withdrawn. Two hypotheses, both testable, and they make opposite predictions.

  H-SIM  "The Jacobians are nearly identical across layers, so shuffling is close to a no-op."
         Predicts: high pairwise cos(J_i, J_j); shuffled ~ jlens EVERYWHERE, including
         per-layer.

  H-MIN  "pass@k takes a MIN OVER LAYERS, which rewards DIVERSITY among per-layer readouts.
         Shuffling re-pairs (h_l, J_l') without reducing how many readouts you get, and a more
         diverse set of readouts can lower the min by chance."
         Predicts: per-layer, jlens BEATS shuffled almost everywhere; the shuffled advantage
         exists only after the min is taken. **That would make it a metric artifact, not a
         property of the lens.**

This script measures both:

  A. pairwise cosine and relative distance between the fitted J_l, flattened
  B. PER-LAYER pass@k (no min) for jlens vs shuffled vs logit, from a single forward pass
     per prompt, then the same quantities WITH the min, so the gap is attributable

If H-MIN holds, `t13`'s headline needs restating: not "a broken J reads as well", but "the
anchor's own metric cannot distinguish them, because min-over-layers launders the difference".

EXPLORATORY.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

import torch

torch.set_num_threads(min(8, os.cpu_count() or 8))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jlens.hf import Layout, from_hf  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402
from anchor_evals import (load_eval, readout, readout_position,  # noqa: E402
                          rank_of, token_ids_of)
from t13_transport_controls import build_transports  # noqa: E402

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")
KS = (1, 2, 5, 10, 25, 100)
_LOGK = [math.log(k) for k in KS]
_SPAN = _LOGK[-1] - _LOGK[0]


def auc(curve: list[float]) -> float:
    a = sum((curve[i] + curve[i + 1]) / 2 * (_LOGK[i + 1] - _LOGK[i]) for i in range(len(KS) - 1))
    return a / _SPAN


def jacobian_similarity(lens: JacobianLens, layers: list[int]) -> dict:
    """A. How similar are the J_l to each other? If cos ~ 1, shuffling is a no-op."""
    V = torch.stack([lens.jacobians[l].float().flatten() for l in layers])
    Vn = V / V.norm(dim=1, keepdim=True)
    C = Vn @ Vn.T
    off = C[~torch.eye(len(layers), dtype=torch.bool)]
    # also: distance to the identity, to test the "J ~ I + M" story
    d = lens.d_model
    I = torch.eye(d).flatten()
    I = I / I.norm()
    cos_I = (Vn @ I).tolist()
    return {
        "pairwise_cos_mean": round(float(off.mean()), 4),
        "pairwise_cos_min": round(float(off.min()), 4),
        "pairwise_cos_max": round(float(off.max()), 4),
        "pairwise_cos_median": round(float(off.median()), 4),
        "cos_with_identity_per_layer": [round(c, 4) for c in cos_I],
        "cos_with_identity_mean": round(sum(cos_I) / len(cos_I), 4),
        "note": ("pairwise_cos near 1 would mean the Jacobians are interchangeable and shuffling "
                 "is a no-op (H-SIM). cos_with_identity near 1 would mean J ~ I, i.e. the J-lens "
                 "is nearly the logit lens."),
    }


@torch.no_grad()
def per_layer_auc(model, transports: dict, eval_name: str, layers: list[int]) -> dict:
    """B. AUC per layer (no min) AND with the min, for each transport, one forward per prompt."""
    items = load_eval(eval_name)
    # hits[t][layer] = list over (item,intermediate) of [len(KS)] binary
    hits = {t: {l: [] for l in layers} for t in transports}
    hits_min = {t: [] for t in transports}
    for item in items:
        prompt = item["prompt"].rstrip()
        pos = readout_position(model.tokenizer, eval_name, prompt)
        outs = {t: readout(model, T, prompt, layers, pos) for t, T in transports.items()}
        for word in item["intermediates"]:
            ids = token_ids_of(model.tokenizer, word)
            if not ids:
                continue
            for t in transports:
                ranks = {l: rank_of(outs[t][l], ids) for l in layers}
                for l in layers:
                    hits[t][l].append([1.0 if ranks[l] <= k else 0.0 for k in KS])
                m = min(ranks.values())
                hits_min[t].append([1.0 if m <= k else 0.0 for k in KS])
    out = {}
    for t in transports:
        per = {}
        for l in layers:
            H = torch.tensor(hits[t][l])
            per[str(l)] = round(auc(H.mean(0).tolist()), 4)
        M = torch.tensor(hits_min[t])
        out[t] = {"per_layer_auc": per,
                  "best_single_layer_auc": round(max(per.values()), 4),
                  "mean_over_layers_auc": round(sum(per.values()) / len(per), 4),
                  "min_over_layers_auc": round(auc(M.mean(0).tolist()), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evals", default="multihop,typo")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    try:
        model = from_hf(hf, tok)
    except ValueError:
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    lens = JacobianLens.load(a.lens)
    layers = list(lens.source_layers)
    transports, perm = build_transports(lens, layers, a.seed)
    keep = {k: transports[k] for k in ("logit", "jlens", "shuffled")}

    sim = jacobian_similarity(lens, layers)
    print("A. Jacobian similarity across layers")
    print(f"   pairwise cos(J_i,J_j):  mean {sim['pairwise_cos_mean']}  "
          f"median {sim['pairwise_cos_median']}  min {sim['pairwise_cos_min']}  "
          f"max {sim['pairwise_cos_max']}")
    print(f"   cos(J_l, I) mean:       {sim['cos_with_identity_mean']}")

    results = {}
    for ev in [s.strip() for s in a.evals.split(",")]:
        r = per_layer_auc(model, keep, ev, layers)
        results[ev] = r
        print(f"\nB. {ev} — AUC by aggregation")
        print(f"   {'transport':10s} {'best 1 layer':>13s} {'mean/layer':>11s} {'MIN over layers':>16s}")
        for t in ("logit", "jlens", "shuffled"):
            v = r[t]
            print(f"   {t:10s} {v['best_single_layer_auc']:13.4f} "
                  f"{v['mean_over_layers_auc']:11.4f} {v['min_over_layers_auc']:16.4f}")
        jl, sh = r["jlens"], r["shuffled"]
        print(f"   jlens - shuffled:  best1L {jl['best_single_layer_auc']-sh['best_single_layer_auc']:+.4f}"
              f"   meanL {jl['mean_over_layers_auc']-sh['mean_over_layers_auc']:+.4f}"
              f"   MIN {jl['min_over_layers_auc']-sh['min_over_layers_auc']:+.4f}")
        n_better = sum(1 for l in layers
                       if jl["per_layer_auc"][str(l)] > sh["per_layer_auc"][str(l)])
        results[ev]["n_layers_jlens_beats_shuffled"] = n_better
        results[ev]["n_layers"] = len(layers)
        print(f"   per-layer: jlens beats shuffled on {n_better}/{len(layers)} layers")

    out = {
        "experiment": "T15 — why does a layer-shuffled J still beat the logit lens?",
        "status": "EXPLORATORY — diagnostic on burned cells",
        "model": a.model, "lens": a.lens, "seed": a.seed, "layers": layers,
        "derangement_layer_map": perm,
        "hypotheses": {
            "H_SIM": "Jacobians are near-identical across layers, so shuffling is a no-op",
            "H_MIN": ("min-over-layers launders the difference; per-layer, jlens should beat "
                      "shuffled almost everywhere")},
        "jacobian_similarity": sim,
        "results": results,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out} ({out['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
