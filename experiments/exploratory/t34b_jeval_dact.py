#!/usr/bin/env python3
"""t34b_jeval_dact.py — E34 as pre-registered: D_act(J^P, J^eval) vs the corpus floor.

This is the version PLAN.tex section 6 actually specifies. t34_dact_vs_floor.py (E34-a) used the
identity as the reference because the pre-registered J^eval fit is impossible at skip_first=16.
This script fits J^eval at skip_first=0 instead, which uses all 551 eval prompts, and DECLARES the
asymmetry that introduces -- every J^P we own was fitted at skip_first=16.

DECLARED BIAS, and the control that sizes it
============================================
skip_first is immaterial to the READ (E17: n_win spread = 1 across {0,4,16}) but NOT immaterial to
the ESTIMATOR (CLAUDE.md section 5). So D_act(J^P_16, J^eval_0) mixes two differences: the corpus
difference we want, and a skip_first difference we do not.

CONTROL C3 (this script's addition): refit one corpus at skip_first=0 and measure
D_act(J^P_16, J^P_0) -- the same corpus, differing ONLY in skip_first. If that is comparable to
D_act(J^P_16, J^eval_0), the eval-vs-corpus signal is confounded by the fitting convention and
NOTHING in the primary is interpretable. The control's number is reported either way.

DECISION RULE -- verbatim from PLAN.tex section 6 (E34), unchanged:
  * (b) 25 within-set z-scored cells clear |r| >= 0.5 with LOO on every corpus AND every set
  * (a) 5 pooled cells clear but (b) does not => explains the level, not the interaction
  * neither => NULL; D_act joins the other predictors
CONTROL C1  D_act(J,J) = 0 exactly.
CONTROL C2  D_act(J^P, J^eval) must exceed the within-corpus seed floor.
CONTROL C3  the skip_first confound above.

    python t34b_jeval_dact.py                 # ~1h CPU at 410M
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
import time

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
    ap.add_argument("--control-n", type=int, default=50,
                    help="corpus prompts for the C3 skip_first refit")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e34b_jeval_dact_410m.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from jlens.fitting import jacobian_for_prompt
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of
    from metrics_shift import m4_action_divergence

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)

    # ---- fit J^eval at skip_first=0 over ALL eval prompts (and per set)
    t0 = time.time()
    sums = {s: {l: torch.zeros(model.d_model, model.d_model) for l in BAND} for s in EVAL_SETS}
    counts = {s: 0 for s in EVAL_SETS}
    skipped = 0
    for name in EVAL_SETS:
        for it in load_eval(name):
            try:
                pj, _, _ = jacobian_for_prompt(model, it["prompt"], BAND, target_layer=-2,
                                               dim_batch=128, max_seq_len=128, skip_first=0)
            except ValueError as exc:
                skipped += 1
                if skipped <= 3:
                    print(f"    skip: {exc}", flush=True)
                continue
            for l in BAND:
                sums[name][l] += pj[l].float()
            counts[name] += 1
        print(f"  J^eval[{name}] {counts[name]} prompts  {time.time()-t0:.0f}s", flush=True)
    if sum(counts.values()) == 0:
        raise SystemExit("ABORT: zero eval prompts contributed")
    Jeval_set = {s: {l: sums[s][l] / counts[s] for l in BAND} for s in EVAL_SETS if counts[s]}
    tot = {l: sum(sums[s][l] for s in EVAL_SETS) / sum(counts.values()) for l in BAND}
    torch.save({"J": {l: tot[l].half() for l in BAND}, "n_prompts": sum(counts.values()),
                "source_layers": BAND, "d_model": model.d_model, "skip_first": 0,
                "per_set_counts": counts},
               os.path.join(HERE, "..", "results", "jeval_410m_skip0.pt"))
    print(f"J^eval fitted on {sum(counts.values())}/551 prompts ({skipped} skipped), "
          f"{time.time()-t0:.0f}s", flush=True)

    # ---- eval activations at the ladder's readout positions
    H = {s: {l: [] for l in BAND} for s in EVAL_SETS}
    with torch.no_grad():
        for name in EVAL_SETS:
            for it in load_eval(name):
                if not [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]:
                    continue
                ids = model.encode(it["prompt"], max_length=256)
                pos = readout_position(tok, name, it["prompt"])
                with ActivationRecorder(model.layers, at=BAND) as rec:
                    model.forward(ids)
                    for l in BAND:
                        H[name][l].append(rec.activations[l][0][pos].detach().float())
    H = {s: {l: torch.stack(v) for l, v in d.items()} for s, d in H.items()}
    Hall = {l: torch.cat([H[s][l] for s in ADMITTED]) for l in BAND}

    def lens(c, s):
        return torch.load(os.path.join(HERE, "..", "results", "lenses", "e28", f"e28_{c}_410m_n{N_MATCH}_s{s}.pt"),
                          map_location="cpu")["J"]

    def dact(A, B, Hd):
        return statistics.median([m4_action_divergence(A[l].float(), B[l].float(),
                                                       Hd[l])["d_act_normalised"] for l in BAND])

    # ---- controls
    J0 = lens(CORPORA[0], 0)
    c1 = max(m4_action_divergence(J0[l].float(), J0[l].float(), Hall[l])["d_act"] for l in BAND)
    seedfloor = {c: dact(lens(c, 0), lens(c, 1), Hall) for c in CORPORA}

    # C3: same corpus, skip_first=0 vs skip_first=16
    print(f"C3: refitting {CORPORA[0]} at skip_first=0, n={a.control_n}...", flush=True)
    texts = [json.loads(l)["text"]
             for l in open(os.path.join(HERE, "..", "corpora", f"{CORPORA[0]}.jsonl"))]
    b = len(texts) // 3
    pool = [t for t in texts[:b] if len(tok(t).input_ids) >= 128][:a.control_n]
    acc = {l: torch.zeros(model.d_model, model.d_model) for l in BAND}
    nc = 0
    for p in pool:
        try:
            pj, _, _ = jacobian_for_prompt(model, p, BAND, target_layer=-2, dim_batch=128,
                                           max_seq_len=128, skip_first=0)
        except ValueError:
            continue
        for l in BAND:
            acc[l] += pj[l].float()
        nc += 1
    Jp0 = {l: acc[l] / nc for l in BAND}
    c3 = dact(lens(CORPORA[0], 0), Jp0, Hall)
    print(f"C3 D_act(J^P_16, J^P_0) on {CORPORA[0]} (n={nc}) = {c3:.4f}", flush=True)

    # ---- primaries
    dJ = {c: statistics.mean([dact(lens(c, s), tot, Hall) for s in range(3)]) for c in CORPORA}
    dJ_set = {c: {st: dact(lens(c, 0), Jeval_set.get(st, tot), H[st]) for st in ADMITTED}
              for c in CORPORA}
    for c in CORPORA:
        print(f"  {c:20s} D_act(J^P,J^eval)={dJ[c]:8.4f}  seedfloor={seedfloor[c]:.4f}", flush=True)

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
             for n in cells[(c, s)]["by_N"] if int(n) >= 75]) for s in range(3)]) for st in ADMITTED}

    xs = [dJ[c] for c in CORPORA]; ya = [Y[c] for c in CORPORA]
    ra = pearson(xs, ya)
    loo_a = {c: pearson([v for j, v in enumerate(xs) if CORPORA[j] != c],
                        [v for j, v in enumerate(ya) if CORPORA[j] != c]) for c in CORPORA}
    zx, zy, keys = [], [], []
    for st in ADMITTED:
        xv = [dJ_set[c][st] for c in CORPORA]; yv = [Yset[c][st] for c in CORPORA]
        mx, sx = statistics.mean(xv), statistics.pstdev(xv)
        my, sy = statistics.mean(yv), statistics.pstdev(yv)
        for i, c in enumerate(CORPORA):
            zx.append((xv[i] - mx) / sx if sx else 0.0)
            zy.append((yv[i] - my) / sy if sy else 0.0); keys.append((c, st))
    rb = pearson(zx, zy)
    loo_b_c = {c: pearson([v for k, v in zip(keys, zx) if k[0] != c],
                          [v for k, v in zip(keys, zy) if k[0] != c]) for c in CORPORA}
    loo_b_s = {st: pearson([v for k, v in zip(keys, zx) if k[1] != st],
                           [v for k, v in zip(keys, zy) if k[1] != st]) for st in ADMITTED}

    b_clears = (abs(rb) >= 0.5 and all(abs(v) >= 0.5 for v in loo_b_c.values())
                and all(abs(v) >= 0.5 for v in loo_b_s.values()))
    a_clears = abs(ra) >= 0.878 and all(abs(v) >= 0.8 for v in loo_a.values())
    c3_confounded = c3 >= min(dJ.values())
    if c3_confounded:
        verdict = (f"CONFOUNDED — C3 skip_first-only distance ({c3:.4f}) is not smaller than the "
                   f"smallest corpus-vs-eval distance ({min(dJ.values()):.4f}). Primaries below "
                   f"are NOT interpretable.")
    elif b_clears:
        verdict = "D_act PREDICTS WHICH CONCEPTS A CORPUS BUYS — (b) clears with full LOO"
    elif a_clears:
        verdict = "WEAK — (a) clears, (b) does not: explains the level, not the interaction"
    else:
        verdict = "NULL — D_act does not predict the corpus floor, with J^eval as the reference"

    rec = {
        "experiment": "E34 — D_act(J^P, J^eval) vs the corpus floor (the pre-registered version)",
        "prereg": "PLAN.tex section 6 (E34), decision rule unchanged",
        "declared_bias": "J^eval fitted at skip_first=0 (all 551 prompts); every J^P at "
                         "skip_first=16. C3 sizes that asymmetry.",
        "model": MODEL, "band": BAND, "admitted_sets": ADMITTED,
        "jeval_n_prompts": sum(counts.values()), "jeval_per_set_counts": counts,
        "jeval_skipped": skipped,
        "control_C1_max_dact_self": c1, "control_C1_fires": c1 == 0.0,
        "control_C2_seed_floor": seedfloor,
        "control_C2_fires": all(dJ[c] > seedfloor[c] for c in CORPORA),
        "control_C3_skipfirst_only_dact": c3, "control_C3_n": nc,
        "control_C3_confounded": c3_confounded,
        "dact_JP_vs_Jeval": dJ, "dact_JP_vs_Jeval_per_set": dJ_set,
        "Y_inf": Y, "Y_inf_per_set": Yset,
        "primary_a": {"r": ra, "loo": loo_a, "clears": a_clears},
        "primary_b": {"r": rb, "n_cells": len(zx), "loo_by_corpus": loo_b_c,
                      "loo_by_set": loo_b_s, "clears": b_clears},
        "VERDICT": verdict,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"\nC1 fires: {c1 == 0.0}   C2 fires: {rec['control_C2_fires']}   "
          f"C3 confounded: {c3_confounded}")
    print(f"PRIMARY (a) r = {ra:+.3f}  LOO " + " ".join(f"-{c[:6]}:{v:+.2f}" for c, v in loo_a.items()))
    print(f"PRIMARY (b) r = {rb:+.3f}  LOO/corpus " + " ".join(f"-{c[:6]}:{v:+.2f}" for c, v in loo_b_c.items()))
    print(f"            LOO/set    " + " ".join(f"-{s[:6]}:{v:+.2f}" for s, v in loo_b_s.items()))
    print(f"\nVERDICT: {verdict}\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
