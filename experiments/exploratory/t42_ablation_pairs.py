#!/usr/bin/env python3
"""t42_ablation_pairs.py — E47: is the disagreement-subspace knockout a LAW about the operator,
or a fact about Github?

THE CRITIQUE THIS ANSWERS
  E46 ran the knockout on three pairs, two of which had Github as an endpoint. Github has been the
  high-leverage point in six prior analyses. An intervention run mostly on the outlier is still one
  observation, and it is the observation most likely to be unrepresentative -- the genre changed
  from correlation to intervention, but the SAMPLING of interventions stayed Github-shaped.

  The mechanistic version is not "predict across more corpora". It is: run the knockout on several
  pairs spanning a range of operator distance, and show that SELECTIVITY IS INVARIANT while
  MAGNITUDE SCALES. That is what makes it a law about the operator rather than a fact about one
  corpus.

PAIRS, chosen before running, to span ||Delta||
  Github -> USPTO_Backgrounds   the extreme pair (the two ends of the measured read spread)
  Github -> StackExchange       second extreme, Github endpoint
  Wikipedia_en -> USPTO_Backg.  moderate, no Github
  Wikipedia_en -> StackExchange moderate English prose, no Github
  Pile-CC -> StackExchange      EXPECTED WEAK: the one pair whose read asymptotes do NOT separate
                                (t = -0.20, SPINE A2.4). If the mechanism is real this pair should
                                show little movement, because there is little disagreement to ablate.

THE METRIC, fixed after E46 showed the previous one was degenerate
  E46 normalised movement by the P-Q read gap. When P and Q agree about a concept set that gap is
  ~0, so the ratio explodes (multihop +9.00) or is undefined (poetry NaN). That is not a missing
  cell -- it is the mechanism's own NEGATIVE PREDICTION being thrown away: if Delta has no
  read-component for a concept, ablating Delta's top-k must leave that concept untouched.

  So movement is now ABSOLUTE:  move(S) = read(J^P, S) - read(J^P_ablated, S)
  and the per-set disagreement is  disagree(S) = read(J^P, S) - read(J^Q, S)
  Both are finite everywhere, so the agreed-on concepts become evidence rather than NaNs.

PRE-REGISTRATION — fixed before any number below existed
  PRIMARY   Per pair, r( |disagree(S)| , |move(S)| ) across the 5 admitted sets, for the top-k arm.
  RULE      * top-k r >= 0.5 in >= 4 of 5 pairs AND random-k r < 0.3 in >= 4 of 5
              => ACCEPT: selectivity is invariant across pairs. A law, not a Github fact.
            * top-k r >= 0.5 ONLY in pairs with Github as an endpoint
              => REJECT: the effect is a Github artifact and the honest paper is narrower.
            * top-k and random-k behave alike => REJECT: any rank-k ablation does this.
  SECONDARY Does magnitude scale? r( ||Delta||_F , max_S |move(S)| ) across the 5 pairs.
            Selectivity invariant + magnitude scaling = the law statement.
  CONTROL   random-k (matched rank, random orthonormal output directions) in every pair. This is
            load-bearing: it separates "this subspace" from "any k directions".
  CONTROL   bottom-k should be inert in every pair.
  BIAS      5 concept SETS per pair, not 1310 tokens -- the stored ladder aggregates ranks per set.
            Token granularity is the stronger test and is deferred, not dismissed.
  REUSE     Pairs already scored by E46 are read from its stored per-arm `read` values and
            re-analysed under the absolute metric; only new pairs are scored. No rescoring.

    python t42_ablation_pairs.py --device cpu
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
PAIRS = [("Github", "USPTO_Backgrounds"), ("Github", "StackExchange"),
         ("Wikipedia_en", "USPTO_Backgrounds"), ("Wikipedia_en", "StackExchange"),
         ("Pile-CC", "StackExchange")]
RANK = 4          # E46: at k=32 the random arm already bites, so the clean contrast is k=4


def pear(x, y):
    n = len(x)
    if n < 3: return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in x)); sy = math.sqrt(sum((v - my) ** 2 for v in y))
    return sum((p - mx) * (q - my) for p, q in zip(x, y)) / (sx * sy) if sx * sy else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reuse", default=os.path.join(HERE, "..", "results", "e46_disagreement_ablation.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e47_ablation_pairs.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of, rank_of

    prev = json.load(open(a.reuse)) if os.path.exists(a.reuse) else {"arms": {}, "baseline_read": {}}
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

    def score(T):
        per = {}
        with torch.no_grad():
            for name, acts, tg in items:
                rows = [[min((rank_of(model.unembed(acts[l] @ T[l].T).float().cpu(), i) or 10**9)
                             for i in t) for t in tg] for l in BAND]
                per.setdefault(name, []).append(torch.tensor(rows, dtype=torch.float32))
        return {n: sum(((torch.cat(R, 1) <= k).float().sum(0) >= (len(BAND)//2)).float().mean().item()
                       for k in K) / len(K) for n, R in per.items()}

    def lens(c):
        return {int(k): v.float() for k, v in torch.load(
            os.path.join(HERE, "..", "results", "lenses", "e28", f"e28_{c}_410m_n400_s0.pt"), map_location="cpu")["J"].items()}

    corpora = sorted({c for p in PAIRS for c in p})
    Js = {c: lens(c) for c in corpora}
    base = prev.get("baseline_read", {})
    for c in corpora:
        if c not in base:
            base[c] = score(Js[c]); print(f"  baseline {c} scored", flush=True)

    g = torch.Generator().manual_seed(0)
    rec = {"experiment": "E47 — is the disagreement knockout a law about the operator or a fact "
                         "about Github?", "prereg": "in-file docstring, fixed before running",
           "model": MODEL, "band": BAND, "rank": RANK, "baseline_read": base, "pairs": {}}

    for P, Q in PAIRS:
        Delta = {l: Js[P][l] - Js[Q][l] for l in BAND}
        dnorm = statistics.median([Delta[l].norm().item() /
                                   Js[P][l].norm().item() for l in BAND])
        svd = {l: torch.linalg.svd(Delta[l], full_matrices=False) for l in BAND}
        entry = {"delta_rel_fro": dnorm,
                 "disagree": {s: base[P][s] - base[Q][s] for s in SETS}, "arms": {}}
        for arm in ["top", "random", "bottom"]:
            key = f"{P}->{Q}|k={RANK}|{arm}"
            if key in prev.get("arms", {}):
                sc = prev["arms"][key]["read"]; src = "reused"
            else:
                T = {}
                for l in BAND:
                    U = svd[l].U
                    if arm == "top":      B = U[:, :RANK]
                    elif arm == "bottom": B = U[:, -RANK:]
                    else:
                        B, _ = torch.linalg.qr(torch.randn(d, RANK, generator=g))
                    T[l] = Js[P][l] - B @ (B.T @ Js[P][l])
                sc = score(T); src = "scored"
            move = {s: base[P][s] - sc[s] for s in SETS}
            r = pear([abs(entry["disagree"][s]) for s in SETS], [abs(move[s]) for s in SETS])
            entry["arms"][arm] = {"read": sc, "move": move, "selectivity_r": r, "source": src}
            print(f"  {P[:11]:11s}->{Q[:11]:11s} {arm:6s} r(|disagree|,|move|)={r:+.3f}  "
                  f"moves " + " ".join(f"{s[:5]}:{move[s]:+.4f}" for s in SETS), flush=True)
        rec["pairs"][f"{P}->{Q}"] = entry
        json.dump(rec, open(a.out, "w"), indent=1)

    tops = {p: v["arms"]["top"]["selectivity_r"] for p, v in rec["pairs"].items()}
    rnds = {p: v["arms"]["random"]["selectivity_r"] for p, v in rec["pairs"].items()}
    bots = {p: v["arms"]["bottom"]["selectivity_r"] for p, v in rec["pairs"].items()}
    ngh = [p for p in tops if "Github" not in p]
    n_top = sum(1 for v in tops.values() if v >= 0.5)
    n_rnd = sum(1 for v in rnds.values() if abs(v) < 0.3)
    n_top_nogh = sum(1 for p in ngh if tops[p] >= 0.5)

    mags = [(rec["pairs"][p]["delta_rel_fro"],
             max(abs(x) for x in rec["pairs"][p]["arms"]["top"]["move"].values())) for p in tops]
    r_mag = pear([m[0] for m in mags], [m[1] for m in mags])

    rec["summary"] = {"selectivity_top": tops, "selectivity_random": rnds,
                      "selectivity_bottom": bots,
                      "n_pairs_top_ge_0.5": n_top, "n_pairs_random_lt_0.3": n_rnd,
                      "n_nonGithub_pairs_top_ge_0.5": f"{n_top_nogh}/{len(ngh)}",
                      "r_magnitude_vs_delta": r_mag}
    rec["VERDICT"] = (
        "ACCEPT — selectivity invariant across pairs; a law about the operator, not a Github fact"
        if (n_top >= 4 and n_rnd >= 4) else
        "REJECT — selectivity only with a Github endpoint; the honest claim is narrower"
        if (n_top_nogh == 0 and n_top > 0) else
        "REJECT — top-k behaves like random-k; any rank-k ablation does this"
        if (n_top >= 4 and n_rnd < 4) else
        "NULL/MIXED — selectivity is not consistent across pairs; report per-pair and stop")
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"\nselectivity top    : " + "  ".join(f"{p.split('->')[0][:8]}->{p.split('->')[1][:8]}:{v:+.2f}" for p, v in tops.items()))
    print(f"selectivity random : " + "  ".join(f"{v:+.2f}" for v in rnds.values()))
    print(f"selectivity bottom : " + "  ".join(f"{v:+.2f}" for v in bots.values()))
    print(f"non-Github pairs with top r>=0.5: {n_top_nogh}/{len(ngh)}")
    print(f"magnitude scaling r(||Delta||, max|move|) = {r_mag:+.3f}")
    print(f"\nVERDICT: {rec['VERDICT']}\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
