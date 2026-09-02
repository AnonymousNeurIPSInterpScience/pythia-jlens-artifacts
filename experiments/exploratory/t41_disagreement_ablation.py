#!/usr/bin/env python3
"""t41_disagreement_ablation.py — E46: INTERVENE on the disagreement subspace.

THE METHODOLOGICAL CORRECTION THIS SCRIPT EXISTS TO IMPLEMENT
  E44/E45 (and the seventeen before them) tried to PREDICT the per-(corpus, concept) readability
  delta. That was the wrong move: the readability delta is already MEASURED and sitting on disk as
  the F20 z-score grid. Correlating a derived quantity against a measured outcome produces a
  FINGERPRINT of a mechanism, not the mechanism. "Explain" in mechanistic interpretability means:
  intervene on the proposed mechanism and show the effect moves the way the mechanism says.

  The geometry claim is: the corpus changes the operator specifically along the read-directions of
  the concepts it buys. That makes an INTERVENTIONAL prediction.

THE INTERVENTION
  For a corpus pair (P, Q), let Delta_l = J^P_l - J^Q_l with SVD Delta = U S V^T. The top-k LEFT
  singular directions U_k span the output-space subspace where P and Q most disagree about what is
  read. Ablate that subspace from J^P:

        J^P_abl = (I - U_k U_k^T) J^P

  Two consequences follow algebraically and are worth stating because they are what makes this a
  test rather than a demonstration:
    (a) the readout changes by (U_k^T W_U[c])^T (U_k^T J^P h), so a concept is affected in
        proportion to how much its read-direction lies in the disagreement subspace;
    (b) (I - U_k U_k^T) Delta removes Delta's top-k terms, so J^P_abl and J^Q_abl converge.

  PREDICTION, and it is SELECTIVE, which is the whole point:
    J^P_abl should lose P's advantage on exactly the concept sets that FLIP between P and Q, and
    leave the NON-FLIPPING sets untouched. Flip magnitude is |z_P,S - z_Q,S| from the measured F20
    grid -- an independent quantity, not fitted here.

PRE-REGISTRATION — fixed before any number below existed
  PRIMARY   r( flip magnitude |z_P,S - z_Q,S| , readability movement induced by the ablation )
            across (pair, set) cells. Movement is measured toward J^Q:
              move(P,Q,S) = [read(J^P,S) - read(J^P_abl,S)] / [read(J^P,S) - read(J^Q,S)]
            positive means the ablation moved P's read toward Q's on that set.
  RULE      * r >= 0.5 for the TOP-k arm AND the RANDOM-k control shows |r| < 0.3
              => ACCEPT: the disagreement subspace is causally responsible for the interaction.
            * top-k and random-k behave alike => REJECT: any rank-k ablation degrades reads and
              the disagreement subspace is doing no special work.
            * top-k fails but random-k also fails => NULL.
  CONTROL C1 RANDOM-k: ablate k random orthonormal output directions, matched rank. This is the
             control that separates "the disagreement subspace matters" from "removing any k
             directions hurts". It is the load-bearing control of this experiment.
  CONTROL C2 BOTTOM-k: ablate the k SMALLEST singular directions of Delta. Should be inert.
  CONTROL C3 k dose-response: the effect should grow with k, not appear only at one value.
  BIAS      The outcome is per eval SET (the stored ladder aggregates ranks per set), so "concept"
            means "concept set" throughout -- 5 sets, not 1310 tokens. A per-token outcome would
            be a stronger test and is not available without rescoring per token.

    python t41_disagreement_ablation.py --pairs Github:USPTO_Backgrounds,Github:StackExchange
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
K = [1, 2, 5, 10, 20, 50, 100]
SETS = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
CORPORA = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]


def pear(x, y):
    n = len(x)
    if n < 3: return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in x)); sy = math.sqrt(sum((v - my) ** 2 for v in y))
    return sum((p - mx) * (q - my) for p, q in zip(x, y)) / (sx * sy) if sx * sy else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pairs", default="Github:USPTO_Backgrounds,Github:StackExchange,"
                                       "Wikipedia_en:USPTO_Backgrounds")
    ap.add_argument("--ranks", default="4,32")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e46_disagreement_ablation.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of, rank_of

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    d = model.d_model

    items = []
    with torch.no_grad():
        for name in EVAL_SETS:
            if name not in SETS: continue
            for it in load_eval(name):
                tg = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
                if not tg: continue
                ids = model.encode(it["prompt"], max_length=256)
                pos = readout_position(tok, name, it["prompt"])
                with ActivationRecorder(model.layers, at=BAND) as r:
                    model.forward(ids)
                    acts = {l: r.activations[l][0][pos].detach().float() for l in BAND}
                items.append((name, acts, tg))
    print(f"{len(items)} eval items over {len(SETS)} admitted sets", flush=True)

    def score(T):
        per = {}
        with torch.no_grad():
            for name, acts, tg in items:
                rows = [[min((rank_of(model.unembed(acts[l] @ T[l].T).float().cpu(), i) or 10**9)
                             for i in t) for t in tg] for l in BAND]
                per.setdefault(name, []).append(torch.tensor(rows, dtype=torch.float32))
        out = {}
        for name, Rs in per.items():
            R = torch.cat(Rs, dim=1)
            out[name] = sum(((R <= k).float().sum(0) >= (len(BAND)//2)).float().mean().item()
                            for k in K) / len(K)
        return out

    def lens(c):
        return {int(k): v.float() for k, v in torch.load(
            os.path.join(HERE, "..", "results", "lenses", "e28", f"e28_{c}_410m_n400_s0.pt"), map_location="cpu")["J"].items()}

    Js = {c: lens(c) for c in CORPORA}
    base = {c: score(Js[c]) for c in CORPORA}
    print("baselines scored", flush=True)

    # measured F20 grid -> flip magnitudes (independent of anything computed here)
    cells = {}
    for f in glob.glob(os.path.join(HERE, "..", "results", "ladder410", "*.json")):
        dd = json.load(open(f)); cells[(dd["corpus"], dd["seed"])] = dd
    Y = {(c, s): statistics.mean([statistics.mean(
            [cells[(c, sd)]["by_N"][n][s]["persist"]
             for n in cells[(c, sd)]["by_N"] if int(n) >= 75]) for sd in range(3)])
         for c in CORPORA for s in SETS}
    zY = {}
    for s in SETS:
        v = [Y[(c, s)] for c in CORPORA]; m, sd = statistics.mean(v), statistics.pstdev(v)
        for c in CORPORA: zY[(c, s)] = (Y[(c, s)] - m) / sd if sd else 0.0

    rec = {"experiment": "E46 — ablate the between-corpus disagreement subspace and test the "
                         "SELECTIVE prediction",
           "prereg": "in-file docstring, fixed before running", "model": MODEL, "band": BAND,
           "baseline_read": base, "flip_grid_z": {f"{c}|{s}": zY[(c, s)] for c in CORPORA for s in SETS},
           "arms": {}}

    g = torch.Generator().manual_seed(0)
    for pair in a.pairs.split(","):
        P, Q = pair.split(":")
        Delta = {l: Js[P][l] - Js[Q][l] for l in BAND}
        svd = {l: torch.linalg.svd(Delta[l], full_matrices=False) for l in BAND}
        for k in [int(x) for x in a.ranks.split(",")]:
            for arm in ["top", "random", "bottom"]:
                T = {}
                for l in BAND:
                    U = svd[l].U
                    if arm == "top":      B = U[:, :k]
                    elif arm == "bottom": B = U[:, -k:]
                    else:
                        M = torch.randn(d, k, generator=g); B, _ = torch.linalg.qr(M)
                    T[l] = Js[P][l] - B @ (B.T @ Js[P][l])
                sc = score(T)
                key = f"{P}->{Q}|k={k}|{arm}"
                mv = {}
                for s in SETS:
                    denom = base[P][s] - base[Q][s]
                    mv[s] = ((base[P][s] - sc[s]) / denom) if abs(denom) > 1e-9 else float("nan")
                rec["arms"][key] = {"pair": [P, Q], "k": k, "arm": arm, "read": sc, "movement": mv,
                                    "flip": {s: abs(zY[(P, s)] - zY[(Q, s)]) for s in SETS}}
                print(f"  {key:44s} " + " ".join(f"{s[:5]}:{mv[s]:+.2f}" for s in SETS), flush=True)
                json.dump(rec, open(a.out, "w"), indent=1)

    # ---- the test: does ablation movement track FLIP magnitude, and only for the top-k arm?
    for arm in ["top", "random", "bottom"]:
        xs, ys = [], []
        for key, v in rec["arms"].items():
            if v["arm"] != arm: continue
            for s in SETS:
                if not math.isnan(v["movement"][s]):
                    xs.append(v["flip"][s]); ys.append(v["movement"][s])
        if len(xs) >= 3:
            r = pear(xs, ys)
            rec.setdefault("correlations", {})[arm] = {"r": r, "n_cells": len(xs)}
            print(f"\n{arm:7s} r(flip magnitude, ablation movement) = {r:+.3f}  (n={len(xs)})")

    ct = rec.get("correlations", {}).get("top", {}).get("r", float("nan"))
    cr = rec.get("correlations", {}).get("random", {}).get("r", float("nan"))
    rec["control_C1_random_inert"] = abs(cr) < 0.3
    rec["VERDICT"] = (
        "ACCEPT — the disagreement subspace is causally responsible for the interaction"
        if (ct >= 0.5 and abs(cr) < 0.3) else
        "REJECT — top-k behaves like random-k; any rank-k ablation does this"
        if (ct >= 0.5 and abs(cr) >= 0.3) else
        "NULL — ablating the disagreement subspace does not selectively move the flipping sets")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"\nC1 random control inert: {rec['control_C1_random_inert']}")
    print(f"VERDICT: {rec['VERDICT']}\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
