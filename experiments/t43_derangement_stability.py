#!/usr/bin/env python3
"""t43_derangement_stability.py — E49: is "J^P beats J^shuf on k/6 sets" STABLE, or one lucky draw?

WHY THIS EXISTS (operator, point blank)
  Every layer-derangement claim in this programme rests on a SINGLE random derangement:
    - t22_bootstrap_ci_persist_410m.json  stores one derangement_layer_map
    - e33_logit_baseline_410m_v2.json     one seed, reported as "control fires 5/6"
    - t20_ablation_kl_410m.json           "J^P beats the deranged control on 6 of 6 sets"
  A derangement is a random object. One draw is one sample. If n_win swings across draws, then
  "5/6" and "6/6" are draws from a distribution and were never results.

  This script does not assume the answer. It scores MANY independent derangements per (model,
  lens) and reports the DISTRIBUTION of n_win, not a point value.

DESIGN
  * models: 70m, 160m, 410m  (wikitext penultimate lenses -- corpus- and recipe-matched)
  * plus the five E28 corpus lenses at 410m, so corpus is crossed with derangement draw
  * n_seeds independent random derangements per cell (rejection-sampled: no layer keeps its own J)
  * reads only, both aggregations. Writes (t20's 6/6) are a separate, more expensive path and are
    NOT covered here -- that is stated rather than glossed.

WHAT WOULD INVALIDATE THE STANDING CLAIMS
  If the interquartile range of n_win spans 2 or more sets, then a single-draw "5/6" or "6/6"
  carries no information and every claim resting on it must be restated as a distribution.

  This script takes no position on which outcome is expected. It reports mean, sd, min, max and
  the full histogram of n_win, and the per-set win RATE across draws (which is the quantity a
  single draw was silently estimating with n=1).

    python t43_derangement_stability.py --seeds 20 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))

MODELS = {"70m": "EleutherAI/pythia-70m-deduped", "160m": "EleutherAI/pythia-160m-deduped",
          "410m": "EleutherAI/pythia-410m-deduped"}
K = [1, 2, 5, 10, 20, 50, 100]
CORPORA_410M = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]


def band_of(nl):
    b = [l for l in range(int(.35 * nl), int(.85 * nl) + 1) if l < nl - 2]
    return b if len(b) >= 2 else [max(0, nl - 4), nl - 3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e49_derangement_stability.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of, rank_of

    rec = {"experiment": "E49 — stability of the layer-derangement control across random draws",
           "why": "every standing derangement claim rests on ONE draw; this reports the "
                  "distribution", "n_seeds": a.seeds, "cells": {}}

    for short, mid in MODELS.items():
        tok = AutoTokenizer.from_pretrained(mid)
        hf = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(a.device).eval()
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
        nl = len(model.layers)
        BAND = band_of(nl)

        items = []
        with torch.no_grad():
            for name in EVAL_SETS:
                for it in load_eval(name):
                    tg = [t for t in (token_ids_of(tok, w)
                                      for w in it.get("intermediates", [])) if t]
                    if not tg:
                        continue
                    ids = model.encode(it["prompt"], max_length=256)
                    pos = readout_position(tok, name, it["prompt"])
                    with ActivationRecorder(model.layers, at=BAND) as r:
                        model.forward(ids)
                        acts = {l: r.activations[l][0][pos].detach().float() for l in BAND}
                    items.append((name, acts, tg))

        def score(T):
            per = {}
            with torch.no_grad():
                for name, acts, tg in items:
                    rows = [[min((rank_of(model.unembed(acts[l] @ T[l].T).float().cpu(), i)
                                  or 10**9) for i in t) for t in tg] for l in BAND]
                    per.setdefault(name, []).append(torch.tensor(rows, dtype=torch.float32))
            out = {}
            for name, Rs in per.items():
                R = torch.cat(Rs, dim=1); mn = R.min(dim=0).values
                out[name] = {
                    "min": sum((mn <= k).float().mean().item() for k in K) / len(K),
                    "persist": sum(((R <= k).float().sum(0) >= (len(BAND)//2)).float().mean().item()
                                   for k in K) / len(K)}
            return out

        lenses = {"wikitext": os.path.join(HERE, "..", "results", "lenses", "ladder", f"lens_{short}_n200_db128_pen.pt")}
        if short == "410m":
            for c in CORPORA_410M:
                lenses[c] = os.path.join(HERE, "..", "results", "lenses", "e28", f"e28_{c}_410m_n400_s0.pt")

        for lname, lpath in lenses.items():
            if not os.path.exists(lpath):
                continue
            blob = torch.load(lpath, map_location="cpu")["J"]
            JP = {l: blob[l].float() for l in BAND if l in blob}
            if len(JP) < len(BAND):
                print(f"  {short}/{lname}: lens lacks band layers, skipping"); continue
            base = score(JP)
            nwins = {"min": [], "persist": []}
            per_set_wins = {ag: {s: 0 for s in EVAL_SETS} for ag in ("min", "persist")}
            for sd in range(a.seeds):
                g = torch.Generator().manual_seed(1000 + sd)
                while True:
                    perm = [BAND[i] for i in torch.randperm(len(BAND), generator=g).tolist()]
                    if all(p != l for p, l in zip(perm, BAND)):
                        break
                sc = score({l: JP[q] for l, q in zip(BAND, perm)})
                for ag in ("min", "persist"):
                    w = [s for s in EVAL_SETS if base[s][ag] > sc[s][ag]]
                    nwins[ag].append(len(w))
                    for s in w:
                        per_set_wins[ag][s] += 1
            cell = {"lens": os.path.basename(lpath), "band": BAND, "baseline": base}
            for ag in ("min", "persist"):
                v = nwins[ag]
                cell[ag] = {"n_win_mean": statistics.mean(v), "n_win_sd": statistics.pstdev(v),
                            "n_win_min": min(v), "n_win_max": max(v),
                            "n_win_all": v,
                            "hist": {str(k): v.count(k) for k in range(7)},
                            "per_set_win_rate": {s: per_set_wins[ag][s] / a.seeds
                                                 for s in EVAL_SETS}}
                print(f"  {short:5s}/{lname:20s} {ag:8s} n_win = "
                      f"{statistics.mean(v):.2f} +/- {statistics.pstdev(v):.2f}  "
                      f"range [{min(v)},{max(v)}]  hist={cell[ag]['hist']}", flush=True)
            rec["cells"][f"{short}|{lname}"] = cell
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(rec, open(a.out, "w"), indent=1)
        del hf, model

    # verdict: does any cell have an n_win range >= 2?
    unstable = {k: (v["persist"]["n_win_max"] - v["persist"]["n_win_min"])
                for k, v in rec["cells"].items()}
    rec["max_n_win_range_persist"] = max(unstable.values()) if unstable else None
    rec["VERDICT"] = ("UNSTABLE — a single-draw n_win carries no information; every standing "
                      "derangement claim must be restated as a distribution"
                      if (rec["max_n_win_range_persist"] or 0) >= 2 else
                      "STABLE — n_win varies by at most 1 across draws; single-draw claims hold")
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"\nlargest n_win range across draws (persist): {rec['max_n_win_range_persist']}")
    print(f"VERDICT: {rec['VERDICT']}\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
