#!/usr/bin/env python3
"""t61_adjudicate_null.py — E61: how much read survives when there is no computation to read?

PRE-REGISTRATION: specs/experiments/E61_randomized_network_null.md, written before the run.
  PRIMARY   the randomized-model J-lens read AUC under persist, against the real-model logit
            constant (0.0284395) and the layer-deranged floor.
  RULE      FLOOR IS LOW   randomized J < the real-model logit lens on all three corpora. The read
                           reflects computation; the measurement has a floor where it should.
            FLOOR IS HIGH  randomized J >= the real-model logit lens on ANY corpus. Then a
                           substantial part of what the read measures is W_U geometry plus the
                           averaging operation, and every effect size in the paper must be
                           re-expressed against THIS floor rather than against the logit lens.
            UNCLEAR        C1 or C2 does not fire.
  C1  CE of the randomized model > 8.0 nats (real 410M in-stream is 0.98-2.97).
  C2  every layer's J must be finite and non-zero.
  C3  the REAL-model arm, run through the identical code path, must reproduce
      e48_crossover_410m.json's Q0 value for that corpus to <= 1e-3.

    python experiments/t61_adjudicate_null.py
"""
from __future__ import annotations
import argparse, glob, json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from provenance import write_result  # noqa: E402

RES = os.path.join(HERE, "..", "results")
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
CORPORA = ["USPTO_Backgrounds", "Github", "Pile-CC"]
LOGIT_PERSIST = 0.028439503419212996        # e48_crossover_410m.json, arms_admitted_mean.logit_I
LOGIT_MIN = 0.1247396256774664
CE_GATE = 8.0


def read_auc(rec, agg):
    n = max(int(k) for k in rec["by_N"])
    blob = rec["by_N"][str(n)]
    node = blob["reads"] if "reads" in blob else blob
    return st.mean(node[s][agg] for s in ADMITTED), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(RES, "e61"))
    ap.add_argument("--out", default=os.path.join(RES, "e61_randomized_null_410m.json"))
    a = ap.parse_args()

    rand, real = {}, {}
    for f in sorted(glob.glob(os.path.join(a.dir, "tv_*.json"))):
        r = json.load(open(f))
        (rand if r.get("randomization") else real)[(r["corpus"], r["seed"])] = r
    if not rand:
        raise SystemExit(f"ABORT: no randomized cells in {a.dir}")

    rec = {"experiment": "E61 — the randomized-network null (dead salmon)",
           "prereg": "specs/experiments/E61_randomized_network_null.md",
           "status": "PRE-REGISTERED",
           "reference": {"real_model_logit_persist": LOGIT_PERSIST,
                         "real_model_logit_min": LOGIT_MIN,
                         "source": "results/e48_crossover_410m.json arms_admitted_mean.logit_I"},
           "n_randomized_cells": len(rand), "n_real_cells": len(real)}

    # ---- C1 competence destroyed, C2 operator well-formed
    ces, c2_bad = {}, []
    for (c, s), r in sorted(rand.items()):
        n = max(int(k) for k in r["by_N"])
        cp = r["by_N"][str(n)]["corpus"]
        ces[f"{c}|s{s}"] = cp.get("model_cross_entropy")
        op = r["by_N"][str(n)]["operator"]
        for k in ("dispersion_L0", "stable_rank", "unembed_alignment"):
            v = op.get(k)
            if v is None or not math.isfinite(v) or v == 0:
                c2_bad.append(f"{c}|s{s}|{k}={v}")
    finite_ce = [v for v in ces.values() if v is not None and math.isfinite(v)]
    rec["C1_competence_destroyed"] = {
        "ce_by_cell": ces, "gate_nats": CE_GATE,
        "min_ce": min(finite_ce) if finite_ce else None,
        "fires": bool(finite_ce) and all(v > CE_GATE for v in finite_ce),
        "note": ("a randomly initialised network is CONFIDENTLY WRONG, not uniform, so CE sits "
                 "well above the uniform bound ln(50277)=10.83. The registered gate is >8.0 and "
                 "is met as written."),
    }
    rec["C2_operator_well_formed"] = {"n_bad": len(c2_bad), "bad": c2_bad,
                                      "fires": not c2_bad}

    # ---- C3 the real arm reproduces e48
    c3 = {}
    e48 = json.load(open(os.path.join(RES, "e48_crossover_410m.json")))["arms_admitted_mean"]
    for (c, s), r in sorted(real.items()):
        got, n = read_auc(r, "persist")
        key = f"J|{c}|s{s}"
        ref = e48.get(key, {}).get("persist")
        c3[f"{c}|s{s}"] = {"trainval_persist": got, "e48_persist": ref, "N": n,
                           "abs_diff": (abs(got - ref) if ref is not None else None)}
    diffs = [v["abs_diff"] for v in c3.values() if v["abs_diff"] is not None]
    rec["C3_real_arm_reproduces_e48"] = {
        "per_cell": c3, "tolerance": 1e-3,
        "max_abs_diff": max(diffs) if diffs else None,
        "fires": bool(diffs) and max(diffs) <= 1e-3}

    # ---- the null itself
    by_corpus = {}
    for c in CORPORA:
        cells = [(s, rand[(c, s)]) for s in (0, 1, 2) if (c, s) in rand]
        if not cells:
            continue
        p = [read_auc(r, "persist")[0] for _, r in cells]
        m = [read_auc(r, "min")[0] for _, r in cells]
        by_corpus[c] = {
            "persist_per_seed": p, "persist_mean": st.mean(p),
            "persist_sd": st.stdev(p) if len(p) > 1 else 0.0,
            "min_per_seed": m, "min_mean": st.mean(m),
            "persist_vs_logit": st.mean(p) - LOGIT_PERSIST,
            "persist_below_logit": st.mean(p) < LOGIT_PERSIST,
            "min_vs_logit": st.mean(m) - LOGIT_MIN,
            "min_below_logit": st.mean(m) < LOGIT_MIN,
        }
    rec["randomized_by_corpus"] = by_corpus
    n_below = sum(1 for v in by_corpus.values() if v["persist_below_logit"])
    rec["n_corpora_below_the_free_baseline"] = f"{n_below}/{len(by_corpus)}"

    if not (rec["C1_competence_destroyed"]["fires"] and rec["C2_operator_well_formed"]["fires"]):
        verdict = "UNCLEAR — a control did not fire"
    elif n_below == len(by_corpus):
        worst = max(v["persist_vs_logit"] for v in by_corpus.values())
        verdict = (f"FLOOR IS LOW — a J fitted on a weight-randomized model reads BELOW the "
                   f"unfitted logit lens on {n_below}/{len(by_corpus)} corpora (closest approach "
                   f"{worst:+.5f} against the {LOGIT_PERSIST:.5f} free baseline). The readout "
                   f"reflects the model's computation, and the measurement has a floor where it "
                   f"should.")
    else:
        above = [c for c, v in by_corpus.items() if not v["persist_below_logit"]]
        verdict = (f"FLOOR IS HIGH — a J fitted on a weight-randomized model reaches or beats the "
                   f"unfitted logit lens on {above}. A substantial part of what this readout "
                   f"measures is W_U geometry plus the averaging operation rather than the "
                   f"model's computation, and every effect size in the paper must be re-expressed "
                   f"against THIS floor rather than against the logit lens.")
    rec["VERDICT"] = verdict

    write_result(a.out, rec, experiment="E61",
                 inputs=sorted(glob.glob(os.path.join(a.dir, "tv_*.json"))))
    c1, c2, c3f = (rec["C1_competence_destroyed"], rec["C2_operator_well_formed"],
                   rec["C3_real_arm_reproduces_e48"])
    print(f"C1 CE destroyed : min={c1['min_ce']} vs gate {CE_GATE} -> "
          f"{'FIRES' if c1['fires'] else 'DOES NOT FIRE'}")
    print(f"C2 operator ok  : {'FIRES' if c2['fires'] else 'DOES NOT FIRE'} {c2['bad']}")
    print(f"C3 real vs e48  : max|diff|={c3f['max_abs_diff']} -> "
          f"{'FIRES' if c3f['fires'] else 'DOES NOT FIRE'}")
    print(f"\n  free baseline (logit, persist) = {LOGIT_PERSIST:.5f}")
    for c, v in by_corpus.items():
        print(f"  randomized J | {c:20s} persist {v['persist_mean']:.5f} "
              f"({v['persist_vs_logit']:+.5f})  below={v['persist_below_logit']}")
    print(f"\nVERDICT: {verdict}\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
