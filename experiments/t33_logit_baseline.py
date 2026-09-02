#!/usr/bin/env python3
"""t33_logit_baseline.py — E33: the missing J=I arm on the E28 read ladder.

WHY
  Every number in the E28 ladder is J-lens vs J-lens. The programme's thesis is a comparison
  against the logit lens (J=I), and the ladder never scored one. Until it does, "corpus identity
  sets a floor" cannot be written as "...and the unfitted baseline has no such floor", which is the
  sentence the paper needs. The logit lens is corpus-free, so it is ONE constant that all five
  corpus curves are measured against.

PRE-REGISTRATION — fixed before this ran; quoted from PLAN.tex section 6 (E33), not reinterpreted.
  PRIMARY  logit-lens read AUC (persist, 5 admitted eval sets: association is floored and excluded
           by E28 admission criterion 3), and the count of corpora whose Y_inf,P exceeds it.
  RULE     * I below all five corpora -> the fitted operator helps at 410M regardless of corpus.
           * I INSIDE the corpus spread -> the crossover already exists, on the corpus axis.
           * I above all five -> the J-lens does not help at 410M under persist; S1 moves up.
  CONTROL  J^shuf must score 0/6 vs J^P under persist, reproducing the known value.
  BIAS     Same battery, band and model, so this compares transports, not methods in general.

  Identical scoring path to trainval.py: same band, same readout positions, same pass@k over
  K=[1,2,5,10,20,50,100], same four aggregations. The ONLY thing that changes is the transport.

    python t33_logit_baseline.py --out results/e33_logit_baseline_410m.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))

MODELS = {"410m":"EleutherAI/pythia-410m-deduped","1b":"EleutherAI/pythia-1b-deduped",
          "1.4b":"EleutherAI/pythia-1.4b-deduped","2.8b":"EleutherAI/pythia-2.8b-deduped"}
BAND = list(range(9, 22))
K = [1, 2, 5, 10, 20, 50, 100]
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]   # association: floored
CORPORA = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="410m", choices=["410m","1b","1.4b","2.8b"])
    ap.add_argument("--band", default=None,
                    help="lo,hi where hi is EXCLUSIVE (this is range(lo,hi)); so the "
                         "canonical 410M band [9..21] is --band 9,22. Default = 0.35-0.85 "
                         "depth. The help formerly said 'inclusive', which was wrong and "
                         "cost the T1 sweep a flipped verdict; see "
                         "docs/reproducibility/T1_FINDING_E33_BAND.md")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ref-lens", default="e28_Github_410m_n400_s0.pt",
                    help="lens used for the J^shuf control and the rand norm match")
    ap.add_argument("--derangement", default="random", choices=["random", "cyclic"],
                    help="AMENDMENT A1 (2026-08-13): v1 of this script used `cyclic`, a shift by "
                         "one layer, so every layer received its IMMEDIATE NEIGHBOUR's Jacobian -- "
                         "the most similar pair available and thus the weakest derangement in the "
                         "family. The control did not fire (4/6, needed >=5). The 0/6 on record "
                         "used a RANDOM derangement (t22_bootstrap_ci_persist_410m.json). This is "
                         "a fix to a mis-specified control, NOT a change to the decision rule, "
                         "which is unchanged and was fixed before v1 ran.")
    ap.add_argument("--derangement-seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results",
                                                  "e33_logit_baseline_410m_v2.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of, rank_of

    MODEL = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    global BAND
    nl = len(model.layers)
    BAND = ([int(x) for x in range(*[int(v) for v in a.band.split(",")])] if a.band
            else [l for l in range(int(.35*nl), int(.85*nl)+1) if l < nl-2])
    print(f"model={a.model} n_layers={nl} band={BAND}", flush=True)

    # ---- cache eval activations ONCE, exactly as trainval.py does
    items = []
    with torch.no_grad():
        for name in EVAL_SETS:
            for it in load_eval(name):
                p = it["prompt"]
                ids = model.encode(p, max_length=256)
                pos = readout_position(tok, name, p)
                with ActivationRecorder(model.layers, at=BAND) as rec:
                    model.forward(ids)
                    acts = {l: rec.activations[l][0][pos].detach().float() for l in BAND}
                tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
                if tgt:
                    items.append((name, acts, tgt))
    print(f"cached {len(items)} eval items, band={BAND}", flush=True)

    def evaluate(T):
        """T: {layer: matrix} or None for the identity (the logit lens)."""
        per = {}
        with torch.no_grad():
            for name, acts, tgt in items:
                rows = []
                for l in BAND:
                    h = acts[l] if T is None else acts[l] @ T[l].T
                    lg = model.unembed(h).float().cpu()
                    rows.append([min((rank_of(lg, i) or 10 ** 9) for i in t) for t in tgt])
                per.setdefault(name, []).append(torch.tensor(rows, dtype=torch.float32))
        out = {}
        for name, Rs in per.items():
            R = torch.cat(Rs, dim=1); mn = R.min(dim=0).values
            out[name] = {
                "min":     sum((mn <= k).float().mean().item() for k in K) / len(K),
                "best1L":  max(sum((R[i] <= k).float().mean().item() for k in K) / len(K)
                               for i in range(R.shape[0])),
                "mean":    sum((R <= k).float().mean().item() for k in K) / len(K),
                "persist": sum(((R <= k).float().sum(0) >= (len(BAND) // 2)).float().mean().item()
                               for k in K) / len(K)}
        return out

    def admitted_mean(d):
        return statistics.mean([d[s]["persist"] for s in ADMITTED])

    print("scoring J=I (logit lens)...", flush=True)
    logit = evaluate(None)

    ref = torch.load(os.path.join(HERE, "..", "results", a.ref_lens), map_location=a.device)["J"]
    JP = {l: ref[l].float() for l in BAND}
    print("scoring J^P (reference corpus)...", flush=True)
    jp = evaluate(JP)

    # CONTROL: layer derangement -- every layer gets ANOTHER layer's Jacobian.
    # A RANDOM derangement, not a cyclic shift: adjacent-layer Jacobians are the most similar
    # pair in the band, so a shift-by-one is the weakest possible version of this control.
    if a.derangement == "cyclic":
        perm = BAND[1:] + BAND[:1]
    else:
        rg = torch.Generator().manual_seed(a.derangement_seed)
        while True:                                      # reject until no layer keeps its own J
            perm = [BAND[i] for i in torch.randperm(len(BAND), generator=rg).tolist()]
            if all(p != l for p, l in zip(perm, BAND)):
                break
    dmap = {int(l): int(q) for l, q in zip(BAND, perm)}
    print(f"scoring J^shuf ({a.derangement} derangement) {dmap}...", flush=True)
    jsh = evaluate({l: JP[q] for l, q in zip(BAND, perm)})

    # CONTROL: norm-matched random transport
    g = torch.Generator().manual_seed(0)
    rnd = {}
    for l in BAND:
        M = torch.randn(JP[l].shape, generator=g)
        rnd[l] = M * (JP[l].norm() / M.norm())
    print("scoring rand (norm-matched)...", flush=True)
    rd = evaluate(rnd)

    # ---- the E28 per-corpus asymptotes, recomputed here so the comparison is self-contained
    cells = {}
    for f in glob.glob(os.path.join(HERE, "..", "results", "ladder410", "*.json")):
        d = json.load(open(f)); cells[(d["corpus"], d["seed"])] = d
    Yinf = {}
    for c in CORPORA:
        per = [statistics.mean([statistics.mean([cells[(c, s)]["by_N"][n][st]["persist"]
                                                 for st in ADMITTED])
                                for n in cells[(c, s)]["by_N"] if int(n) >= 75]) for s in range(3)]
        Yinf[c] = {"mean": statistics.mean(per), "seed_sd": statistics.stdev(per)}

    L = admitted_mean(logit)
    above = [c for c in CORPORA if Yinf[c]["mean"] > L]
    below = [c for c in CORPORA if Yinf[c]["mean"] <= L]
    if not below:
        verdict = ("I BELOW ALL FIVE — the fitted operator helps at 410M regardless of corpus; "
                   "the corpus floor is a floor on a real advantage")
    elif not above:
        verdict = ("I ABOVE ALL FIVE — the J-lens does not help at 410M under persist; "
                   "S1's boundary moves up a scale")
    else:
        verdict = (f"I INSIDE THE CORPUS SPREAD — {len(below)} of 5 corpora fit an operator that is "
                   f"NO BETTER than not fitting at all. The crossover exists on the CORPUS axis.")

    nwin_shuf = sum(1 for s in EVAL_SETS if jp[s]["persist"] > jsh[s]["persist"])
    rec = {
        "experiment": "E33 — the missing J=I arm on the E28 read ladder",
        "prereg": "PLAN.tex section 6 (E33), fixed before this ran; quoted in this file's docstring",
        "status": "PRE-REGISTERED",
        "model": MODEL, "band": BAND, "device": a.device, "n_items": len(items),
        "admitted_sets": ADMITTED,
        "excluded_sets": {"association": "floored — 196/205 ladder cells exactly zero (E28 crit 3)"},
        "reference_lens": a.ref_lens,
        "derangement": a.derangement, "derangement_seed": a.derangement_seed,
        "derangement_layer_map": dmap,
        "supersedes": ("results/e33_logit_baseline_410m.json — v1, whose CONTROL DID NOT FIRE "
                       "(4/6) because it used a cyclic shift by one layer. The decision rule is "
                       "unchanged; only the mis-specified control was fixed. v1 is retained."
                       if a.derangement == "random" else None),
        "by_transport": {"logit_I": logit, "jlens_P": jp, "jshuf": jsh, "rand_matched": rnd
                         and rd},
        "persist_admitted_mean": {"logit_I": L, "jlens_P": admitted_mean(jp),
                                  "jshuf": admitted_mean(jsh), "rand_matched": admitted_mean(rd)},
        "e28_asymptotes": Yinf,
        "corpora_above_logit": above, "corpora_at_or_below_logit": below,
        "control_C_jp_beats_jshuf_n_sets": nwin_shuf,
        "control_C_fires": nwin_shuf >= 5,
        "VERDICT": verdict,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)

    print(f"\n{'transport':22s} {'persist (5 admitted sets)':>26}")
    print(f"{'logit lens (J=I)':22s} {L:26.5f}")
    for c in CORPORA:
        print(f"{'  J^P: '+c:22s} {Yinf[c]['mean']:26.5f}   {'ABOVE I' if Yinf[c]['mean']>L else 'AT OR BELOW I'}")
    print(f"{'J^shuf (control)':22s} {admitted_mean(jsh):26.5f}")
    print(f"{'rand (control)':22s} {admitted_mean(rd):26.5f}")
    print(f"\ncontrol: J^P beats J^shuf on {nwin_shuf}/6 sets (persist) — "
          f"{'FIRES' if nwin_shuf>=5 else 'DOES NOT FIRE'}")
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
