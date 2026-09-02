#!/usr/bin/env python3
"""t40_disagreement_geometry.py — E45: the operator's between-corpus disagreement, resolved
PER READOUT DIRECTION instead of collapsed to a scalar.

THE ARGUMENT
  Eighteen predictors of the corpus effect have failed. F20 says why, and it is a category error
  rather than bad luck: the corpus effect is an INTERACTION (corpus x concept-set = 6.8% of
  variance at 410M, 9.2% at 1B) against a corpus MAIN EFFECT of 1.9% / 2.2%. A scalar per corpus
  has nowhere to put "which concept", so it can only ever address the 1.9%. All eighteen were
  structurally incapable of the thing we were asking them to do.

  D_read (t39) is the clearest case. Writing u_c = Delta^T W_U[c] for concept token c, the
  per-concept logit difference between two operators is u_c^T h, so

        D_read = sum_c ( u_c^T Sigma u_c )        Sigma = E_{x~eval}[h h^T]

  i.e. D_read is ALREADY a sum over concepts of per-concept terms s_c. The sum is exactly the
  operation that destroys the interaction. This script does not build a new statistic: it keeps
  the s_c that t39 computed and then threw away.

  Equivalently, and this is the geometric statement: the rows of M = W_U^concept · Delta are the
  u_c^T. The SVD of M gives the directions in which two corpora most disagree about what is read;
  the row norms (warped by Sigma) are the s_c. Same object, two views.

  THE POINT: the predictor is now the OPERATOR'S OWN ACTION, resolved per concept, and the target
  is per-concept readability. That is mechanism -> outcome, not observable <-> observable. And it
  is evaluated at n = 25 cells, escaping the n = 5 corpus leverage (Github) that killed all
  eighteen prior attempts.

THE NORMALIZATION CAVEAT, handled rather than assumed away
  The read is W_U · norm(J h), not W_U J h. LayerNorm's Jacobian G at the operating point projects
  off the mean direction and rescales, so the honest read-direction is (G J)^T W_U[c]. If the whole
  effect lives in the rescale, that is a DIFFERENT finding. Both variants are computed and both are
  reported; neither is privileged.

PRE-REGISTRATION — fixed before any number below existed
  PRIMARY   r( z(s) , z(|readability|) ) over the 25 (corpus, eval-set) cells, where both sides are
            z-scored WITHIN eval set so task difficulty is removed from each identically.
  RULE      * |r| >= 0.5 AND survives leave-one-SET-out AND leave-one-CORPUS-out
              => ACCEPT: the operators' disagreement geometry IS the interaction.
            * |r| >= 0.5 but fails a leave-one-out => LEVERAGE-DRIVEN, reported as such.
            * |r| < 0.5 => NULL. The disagreement subspace does not explain which concepts move.
  CONTROL   C1 s_c computed with Delta = 0 must be exactly 0.
  CONTROL   C2 a random-direction Delta of matched Frobenius norm must not predict (|r| < 0.3).
            If it does, the statistic is picking up something about the eval sets, not the corpora.
  CONTROL   C3 both pre-norm and post-norm variants reported. A result present only pre-norm is
            reported as a normalization artifact, not a mechanism.
  BIAS      The outcome is per (corpus, eval SET), not per concept token, because the stored ladder
            aggregates ranks per set. So s_c is aggregated to set level before testing -- this is a
            coarser test than the geometry supports, and a per-token outcome would be stronger.

    python t40_disagreement_geometry.py --device cpu
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
SETS = ["multihop", "multilingual", "order-ops", "poetry", "typo"]


def pear(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in x)); sy = math.sqrt(sum((v - my) ** 2 for v in y))
    return sum((p - mx) * (q - my) for p, q in zip(x, y)) / (sx * sy) if sx * sy else float("nan")


def zwithin(mat):
    """z-score within eval set (column), removing task difficulty identically on both sides."""
    out = {}
    for s in SETS:
        v = [mat[(c, s)] for c in CORPORA]
        m, sd = statistics.mean(v), statistics.pstdev(v)
        for c in CORPORA:
            out[(c, s)] = (mat[(c, s)] - m) / sd if sd > 0 else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e45_disagreement_geometry.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    WU = hf.get_output_embeddings().weight.data.float()
    d = WU.shape[1]

    # ---- activations, and which concept tokens belong to which eval set
    set_tokens = {s: set() for s in SETS}
    H = {l: [] for l in BAND}
    with torch.no_grad():
        for name in EVAL_SETS:
            for it in load_eval(name):
                tg = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
                if not tg:
                    continue
                if name in set_tokens:
                    for t in tg:
                        set_tokens[name].update(t)
                ids = model.encode(it["prompt"], max_length=256)
                pos = readout_position(tok, name, it["prompt"])
                with ActivationRecorder(model.layers, at=BAND) as r:
                    model.forward(ids)
                    for l in BAND:
                        H[l].append(r.activations[l][0][pos].detach().float())
    H = {l: torch.stack(v) for l, v in H.items()}
    Sig = {l: (H[l].T @ H[l]) / H[l].shape[0] for l in BAND}
    print(f"activations {H[BAND[0]].shape}, concept tokens per set: "
          f"{ {s: len(v) for s, v in set_tokens.items()} }", flush=True)

    # ---- LayerNorm Jacobian at the operating point (C3). Linearise the model's final norm.
    fin = None
    for attr in ("final_layer_norm", "ln_f", "norm"):
        m = getattr(getattr(hf, "gpt_neox", hf), attr, None)
        if m is not None:
            fin = m; break
    Gm = {}
    if fin is not None:
        for l in BAND:
            x0 = H[l].mean(0).clone().requires_grad_(True)
            Gm[l] = torch.autograd.functional.jacobian(lambda z: fin(z), x0).detach()
        print(f"final-norm Jacobian linearised at the mean activation ({type(fin).__name__})",
              flush=True)
    else:
        print("WARNING: could not locate the final norm; post-norm variant skipped", flush=True)

    def lens(p):
        return {int(k): v.float() for k, v in torch.load(
            os.path.join(HERE, "..", "results", p), map_location="cpu")["J"].items()}

    Js = {c: lens(f"e28_{c}_410m_n400_s0.pt") for c in CORPORA}
    Jbar = {l: sum(Js[c][l] for c in CORPORA) / len(CORPORA) for l in BAND}

    def s_per_set(Delta, fold_norm):
        """s_c = u_c^T Sigma u_c with u_c = Delta^T W_row[c]; aggregated (mean) per eval set."""
        acc = {s: [] for s in SETS}
        for l in BAND:
            W = WU @ Gm[l].T if (fold_norm and l in Gm) else WU
            for s in SETS:
                idx = sorted(set_tokens[s])
                U = W[idx] @ Delta[l]                      # [n_s, d] rows are u_c^T
                sc = ((U @ Sig[l]) * U).sum(1)             # u_c^T Sigma u_c
                acc[s].append(sc.mean().item())
        return {s: statistics.median(v) for s, v in acc.items()}

    # ---- CONTROLS
    zero = {l: torch.zeros(d, d) for l in BAND}
    c1 = max(s_per_set(zero, False).values())
    print(f"C1  s with Delta=0 : {c1:.3e}  {'FIRES' if c1 == 0.0 else 'DOES NOT FIRE'}", flush=True)

    rec = {"experiment": "E45 — between-corpus operator disagreement, resolved per readout direction",
           "prereg": "in-file docstring, fixed before running", "model": MODEL, "band": BAND,
           "n_concept_tokens_per_set": {s: len(v) for s, v in set_tokens.items()},
           "control_C1_zero_delta": c1, "control_C1_fires": c1 == 0.0,
           "note": "Delta_P = J^P - mean_corpora(J); s_c is the per-concept term of D_read, "
                   "which t39 summed away"}

    # ---- the outcome: per (corpus, set) readability, from the stored ladder
    cells = {}
    for f in glob.glob(os.path.join(HERE, "..", "results", "ladder410", "*.json")):
        dd = json.load(open(f)); cells[(dd["corpus"], dd["seed"])] = dd
    Y = {}
    for c in CORPORA:
        for s in SETS:
            Y[(c, s)] = statistics.mean([statistics.mean(
                [cells[(c, sd)]["by_N"][n][s]["persist"]
                 for n in cells[(c, sd)]["by_N"] if int(n) >= 75]) for sd in range(3)])
    zY = zwithin(Y)

    for fold in ([False, True] if Gm else [False]):
        tag = "post_norm" if fold else "pre_norm"
        S = {}
        for c in CORPORA:
            D = {l: Js[c][l] - Jbar[l] for l in BAND}
            for s, v in s_per_set(D, fold).items():
                S[(c, s)] = v
        zS = zwithin(S)
        keys = [(c, s) for c in CORPORA for s in SETS]
        # PRIMARY: does disagreement magnitude track |readability deviation|?
        xs = [zS[k] for k in keys]; ys = [abs(zY[k]) for k in keys]
        r = pear(xs, ys)
        loo_set = {s: pear([zS[k] for k in keys if k[1] != s],
                           [abs(zY[k]) for k in keys if k[1] != s]) for s in SETS}
        loo_cor = {c: pear([zS[k] for k in keys if k[0] != c],
                           [abs(zY[k]) for k in keys if k[0] != c]) for c in CORPORA}
        ws = min(loo_set.items(), key=lambda kv: abs(kv[1]))
        wc = min(loo_cor.items(), key=lambda kv: abs(kv[1]))
        surv = abs(r) >= 0.5 and abs(ws[1]) >= 0.5 and abs(wc[1]) >= 0.5
        rec[tag] = {"s_by_cell": {f"{c}|{s}": S[(c, s)] for c, s in keys},
                    "r": r, "n_cells": len(keys),
                    "loo_by_set": loo_set, "loo_by_corpus": loo_cor,
                    "worst_loo_set": ws, "worst_loo_corpus": wc, "survives": surv}
        print(f"\n[{tag}] r(z disagreement, |z readability|) = {r:+.3f} over {len(keys)} cells")
        print(f"   worst LOO by set    {ws[1]:+.3f}  (-{ws[0]})")
        print(f"   worst LOO by corpus {wc[1]:+.3f}  (-{wc[0]})")
        print(f"   {'SURVIVES' if surv else 'fails the control'}")

    # ---- C2: random Delta of matched Frobenius norm
    g = torch.Generator().manual_seed(0)
    Dr = {}
    for l in BAND:
        ref = Js[CORPORA[0]][l] - Jbar[l]
        R = torch.randn(d, d, generator=g)
        Dr[l] = R * (ref.norm() / R.norm())
    Sr = s_per_set(Dr, False)
    zr = {}
    for s in SETS:                       # same Delta for every corpus => no corpus variation;
        for c in CORPORA:                # z within set is degenerate, so compare raw against |zY|
            zr[(c, s)] = Sr[s]
    keys = [(c, s) for c in CORPORA for s in SETS]
    c2 = pear([zr[k] for k in keys], [abs(zY[k]) for k in keys])
    rec["control_C2_random_delta_r"] = c2
    rec["control_C2_fires"] = abs(c2) < 0.3
    print(f"\nC2  random matched-norm Delta: r = {c2:+.3f}  "
          f"{'FIRES (does not predict)' if abs(c2) < 0.3 else 'DOES NOT FIRE'}")

    best = rec.get("post_norm", rec["pre_norm"])
    rec["VERDICT"] = ("ACCEPT — the operators' disagreement geometry IS the interaction"
                      if best["survives"] else
                      "LEVERAGE-DRIVEN — significant but fails a leave-one-out"
                      if abs(best["r"]) >= 0.5 else
                      "NULL — disagreement geometry does not explain which concepts move")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"\nVERDICT: {rec['VERDICT']}\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
