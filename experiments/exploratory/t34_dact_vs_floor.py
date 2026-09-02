#!/usr/bin/env python3
"""t34_dact_vs_floor.py — E34-a: does an activation-weighted operator distance predict the
corpus floor?  Reference = the identity, NOT a fitted J^eval.

AMENDMENT A1 to E34, written BEFORE any number, because the pre-registered design is infeasible
=================================================================================================
PLAN.tex section 6 pre-registers E34 as: fit J^eval on the anchor's own 541 eval prompts at
skip_first=16, then compute D_act(J^P, J^eval).  MEASURED, before running anything:

    set            n   min tok   median   usable at skip_first=16
    multihop      93         9       16      34
    multilingual 107         5       10       4
    order-ops     55         7        9       1
    poetry        98        25       27      98
    typo          96         6        8       0     <-- ZERO
    association  102        19       28     102
    TOTAL        551                        239  (43.4%)

jlens.fitting.valid_position_mask raises for seq_len <= skip_first+1.  So a J^eval fitted at
skip_first=16 would be built from 239 prompts of which 200 (84%) are poetry and association, and
NONE from typo -- the highest-AUC set.  That operator is not "the eval distribution"; it is "the
two longest eval sets".  Fitting it and calling it J^eval would be a proxy for the thing we care
about, which CLAUDE.md rule 0b forbids.

TWO OPTIONS, both declared rather than silently chosen:
  (a) refit J^eval at skip_first=0 -- uses all 551, but is ASYMMETRIC with every J^P we own
      (all fitted at skip_first=16), and skip_first is known to be immaterial to the READ (E17)
      while NOT immaterial to the ESTIMATOR (CLAUDE.md section 5).
  (b) drop J^eval and use a reference that needs no fit.

THIS SCRIPT RUNS (b), and labels itself E34-a so it is never confused with the pre-registered
E34.  The reference is the identity:

    D_act(J^P, I) = E_{x ~ eval} || (J^P - I) h(x) ||^2

which is well-posed, needs no fit, and has an outcome already measured for it: E33 scored the
logit lens (J=I) on this exact battery and band.  So "how far is this operator from the identity,
in the metric of the activations that actually occur" has a matched performance difference,
AUC(J^P) - AUC(I), to predict.  The pairwise matrix D_act(J^P, J^Q) is also computed, since it is
free once the activations are cached.

DECISION RULE -- inherited verbatim from PLAN.tex section 6 (E34), applied to E34-a's reference:
  * (b) the 25 within-set z-scored cells clear |r| >= 0.5 with LOO on every corpus AND every set
        => an activation-weighted operator distance predicts which concepts a corpus buys.
  * (a) the 5 pooled cells clear but (b) does not => it explains the level, not the interaction.
        Report as weak; do not headline.
  * neither => D_act joins the other 11 predictors. Report the null and retire the instrument.
CONTROL C1  D_act(J, J) = 0 exactly.
CONTROL C2  D_act(J^P, I) must exceed the within-corpus seed floor D_act(J^P_s0, J^P_s1), or the
            signal sits inside sampling noise and nothing below is interpretable.

    python t34_dact_vs_floor.py
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))

MODEL = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))
CORPORA = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
N_MATCH = 400


def pearson(x, y):
    n = len(x); mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx * sy else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e34a_dact_vs_floor_410m.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of
    from metrics_shift import m4_action_divergence

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)

    # ---- eval activations, per set, at the ladder's readout positions
    H = {s: {l: [] for l in BAND} for s in EVAL_SETS}
    with torch.no_grad():
        for name in EVAL_SETS:
            for it in load_eval(name):
                p = it["prompt"]
                tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
                if not tgt:
                    continue
                ids = model.encode(p, max_length=256)
                pos = readout_position(tok, name, p)
                with ActivationRecorder(model.layers, at=BAND) as rec:
                    model.forward(ids)
                    for l in BAND:
                        H[name][l].append(rec.activations[l][0][pos].detach().float())
    H = {s: {l: torch.stack(v) for l, v in d.items()} for s, d in H.items()}
    Hall = {l: torch.cat([H[s][l] for s in ADMITTED]) for l in BAND}
    print(f"activations cached: {[(s, H[s][BAND[0]].shape[0]) for s in EVAL_SETS]}", flush=True)

    def lens(c, s):
        return torch.load(os.path.join(HERE, "..", "results", "lenses", "e28", f"e28_{c}_410m_n{N_MATCH}_s{s}.pt"),
                          map_location="cpu")["J"]

    I = torch.eye(model.d_model)

    # ---- CONTROL C1: D_act(J, J) = 0 exactly
    J0 = lens(CORPORA[0], 0)
    c1 = max(m4_action_divergence(J0[l].float(), J0[l].float(), Hall[l])["d_act"] for l in BAND)
    print(f"C1  max D_act(J,J) = {c1:.3e}  {'FIRES' if c1 == 0.0 else 'DOES NOT FIRE'}", flush=True)

    # ---- CONTROL C2: seed floor, D_act between two seed blocks of the SAME corpus
    seedfloor = {}
    for c in CORPORA:
        A, B = lens(c, 0), lens(c, 1)
        seedfloor[c] = statistics.median(
            [m4_action_divergence(A[l].float(), B[l].float(), Hall[l])["d_act_normalised"]
             for l in BAND])

    # ---- D_act(J^P, I), pooled and per eval set
    dact_I, dact_I_set = {}, {}
    for c in CORPORA:
        per_seed = []
        for s in range(3):
            J = lens(c, s)
            per_seed.append(statistics.median(
                [m4_action_divergence(J[l].float(), I, Hall[l])["d_act_normalised"] for l in BAND]))
        dact_I[c] = statistics.mean(per_seed)
        dact_I_set[c] = {}
        J = lens(c, 0)
        for st in ADMITTED:
            dact_I_set[c][st] = statistics.median(
                [m4_action_divergence(J[l].float(), I, H[st][l])["d_act_normalised"] for l in BAND])
        print(f"  {c:20s} D_act(J^P,I)={dact_I[c]:8.4f}   seed floor={seedfloor[c]:8.4f}   "
              f"ratio={dact_I[c]/seedfloor[c]:6.1f}x", flush=True)

    c2_ok = all(dact_I[c] > seedfloor[c] for c in CORPORA)

    # ---- pairwise operator distance matrix, free once activations are cached
    pair = {}
    for i, p in enumerate(CORPORA):
        Jp = lens(p, 0)
        for q in CORPORA[i + 1:]:
            Jq = lens(q, 0)
            pair[f"{p}|{q}"] = statistics.median(
                [m4_action_divergence(Jp[l].float(), Jq[l].float(), Hall[l])["d_act_normalised"]
                 for l in BAND])

    # ---- outcomes: Y_inf,P and the per-set values, plus E33's logit constant
    cells = {}
    for f in glob.glob(os.path.join(HERE, "..", "results", "ladder410", "*.json")):
        d = json.load(open(f)); cells[(d["corpus"], d["seed"])] = d
    Y, Yset = {}, {}
    for c in CORPORA:
        Y[c] = statistics.mean([statistics.mean(
            [statistics.mean([cells[(c, s)]["by_N"][n][st]["persist"] for st in ADMITTED])
             for n in cells[(c, s)]["by_N"] if int(n) >= 75]) for s in range(3)])
        Yset[c] = {st: statistics.mean([statistics.mean(
            [cells[(c, s)]["by_N"][n][st]["persist"]
             for n in cells[(c, s)]["by_N"] if int(n) >= 75]) for s in range(3)])
            for st in ADMITTED}
    e33 = json.load(open(os.path.join(HERE, "..", "results", "e33_logit_baseline_410m_v2.json")))
    L = e33["persist_admitted_mean"]["logit_I"]
    Lset = {st: e33["by_transport"]["logit_I"][st]["persist"] for st in ADMITTED}

    # ---- PRIMARY (a): pooled, 5 corpora
    xs = [dact_I[c] for c in CORPORA]
    ya = [Y[c] - L for c in CORPORA]
    ra = pearson(xs, ya)
    loo_a = {c: pearson([v for j, v in enumerate(xs) if CORPORA[j] != c],
                        [v for j, v in enumerate(ya) if CORPORA[j] != c]) for c in CORPORA}

    # ---- PRIMARY (b): 25 within-set z-scored cells -- the interaction itself
    zx, zy, keys = [], [], []
    for st in ADMITTED:
        xv = [dact_I_set[c][st] for c in CORPORA]
        yv = [Yset[c][st] - Lset[st] for c in CORPORA]
        mx, sx = statistics.mean(xv), statistics.pstdev(xv)
        my, sy = statistics.mean(yv), statistics.pstdev(yv)
        for i, c in enumerate(CORPORA):
            zx.append((xv[i] - mx) / sx if sx else 0.0)
            zy.append((yv[i] - my) / sy if sy else 0.0)
            keys.append((c, st))
    rb = pearson(zx, zy)
    loo_b_corpus = {c: pearson([v for k, v in zip(keys, zx) if k[0] != c],
                               [v for k, v in zip(keys, zy) if k[0] != c]) for c in CORPORA}
    loo_b_set = {st: pearson([v for k, v in zip(keys, zx) if k[1] != st],
                             [v for k, v in zip(keys, zy) if k[1] != st]) for st in ADMITTED}

    b_clears = (abs(rb) >= 0.5 and all(abs(v) >= 0.5 for v in loo_b_corpus.values())
                and all(abs(v) >= 0.5 for v in loo_b_set.values()))
    a_clears = abs(ra) >= 0.878 and all(abs(v) >= 0.8 for v in loo_a.values())
    if b_clears:
        verdict = "D_act PREDICTS WHICH CONCEPTS A CORPUS BUYS — (b) clears with full LOO"
    elif a_clears:
        verdict = "WEAK — (a) clears, (b) does not: explains the level, not the interaction. Do not headline"
    else:
        verdict = ("NULL — D_act joins the other 11 predictors. Neither (a) nor (b) clears. "
                   "Report and retire the instrument as a floor predictor")

    rec = {
        "experiment": "E34-a — D_act(J^P, I) vs the corpus floor. Reference = identity, NOT J^eval",
        "amendment": "A1: the pre-registered J^eval fit is infeasible at skip_first=16 — "
                     "239/551 prompts survive, 84% poetry+association, ZERO typo. See docstring.",
        "prereg": "PLAN.tex section 6 (E34) decision rule, inherited verbatim",
        "status": "PRE-REGISTERED (rule) / AMENDED (reference operator)",
        "model": MODEL, "band": BAND, "n_match": N_MATCH, "admitted_sets": ADMITTED,
        "control_C1_max_dact_self": c1, "control_C1_fires": c1 == 0.0,
        "control_C2_seed_floor": seedfloor, "control_C2_fires": c2_ok,
        "dact_JP_vs_I": dact_I, "dact_JP_vs_I_per_set": dact_I_set,
        "dact_pairwise": pair,
        "Y_inf": Y, "Y_inf_per_set": Yset, "logit_constant": L, "logit_per_set": Lset,
        "primary_a": {"r": ra, "loo": loo_a, "clears": a_clears},
        "primary_b": {"r": rb, "n_cells": len(zx), "loo_by_corpus": loo_b_corpus,
                      "loo_by_set": loo_b_set, "clears": b_clears},
        "VERDICT": verdict,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)

    print(f"\nC2 D_act(J^P,I) > seed floor for all 5: {c2_ok}")
    print(f"\nPRIMARY (a) pooled, 5 corpora:  r = {ra:+.3f}   LOO "
          + " ".join(f"-{c[:6]}:{v:+.2f}" for c, v in loo_a.items()))
    print(f"PRIMARY (b) 25 within-set cells: r = {rb:+.3f}")
    print("   LOO by corpus: " + " ".join(f"-{c[:6]}:{v:+.2f}" for c, v in loo_b_corpus.items()))
    print("   LOO by set:    " + " ".join(f"-{s[:6]}:{v:+.2f}" for s, v in loo_b_set.items()))
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
