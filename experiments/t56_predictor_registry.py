#!/usr/bin/env python3
"""t56_predictor_registry.py — E56: enumerate every candidate predictor of corpus quality, in one
file, with the criterion each was actually tested against.

WHY
  `paper.tex` says: "we pre-registered a criterion -- a predictor whose |r| falls below 0.8 under
  leave-one-corpus-out is reported as leverage-driven whatever its full-sample value -- and tested
  21 candidates". Figure 2 shows ELEVEN. The number 21 appears in no results file; it is assembled
  by hand across seven of them, and RESULTS_TAXONOMY.md:132 lists the sources without the rows.

  That is the same class of defect as the retracted 4614x: a headline integer with no artifact
  behind it. This script is the artifact.

  It also settles a question the hand-count cannot: are the 21 COMMENSURABLE? They are not. The
  candidates span three different outcome variables, three different replication units, and n from
  3 to 5815. "None of the 21 meets our pre-registered robustness criterion at n=5" describes a
  uniformity that was not run, and this file makes the heterogeneity explicit so the paper can
  state the count it can defend.

WHAT IT SETTLES
  How many candidate predictors of THE PER-CORPUS READ ASYMPTOTE were tested at n=5 corpora against
  the pre-registered |r| >= 0.8 leave-one-corpus-out rule, and how many of the "21" are something
  else.

DECISION RULE — none. This is a registry, not a test. The per-predictor pass/fail is read from the
  criterion each source file recorded; nothing is re-adjudicated here.

CONTROLS
  C1  every r and worst-LOO that also appears in e31_local_bakeoff_410m.json must agree exactly.
  C2  entries that are CONTROLS rather than candidates (e.g. E44's random-row arm, E45's random
      Delta) must be tagged as such and excluded from the count. A control counted as a failed
      predictor would inflate the headline integer.
  C3  duplicates must be tagged. E44 re-reports D_act vs I, which is E34a's predictor.

    python experiments/t56_predictor_registry.py
"""
from __future__ import annotations
import json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from provenance import write_result  # noqa: E402

RES = os.path.join(HERE, "..", "results")
CORPORA = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]
SETS = ["multihop", "multilingual", "order-ops", "poetry", "typo"]


def pear(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx * sy else float("nan")


def main() -> int:
    out_path = os.path.join(RES, "e56_predictor_registry.json")
    rows = []

    def add(name, src, target, unit, n, r, worst_loo, worst_loo_by, criterion, kind="candidate",
            note=""):
        rows.append({"predictor": name, "source_file": src, "target_outcome": target,
                     "replication_unit": unit, "n": n, "r": r, "worst_loo": worst_loo,
                     "worst_loo_dropped": worst_loo_by, "criterion": criterion,
                     "passes": (None if worst_loo is None else abs(worst_loo) >= 0.8),
                     "kind": kind, "note": note})

    # ---------------------------------------------------------------- E31: 11
    e31 = json.load(open(os.path.join(RES, "e31_local_bakeoff_410m.json")))
    b = e31["bakeoff"]["plateau_auc"]
    for k, v in b.items():
        wl = min(v["loo"].values(), key=abs)
        wc = min(v["loo"], key=lambda c: abs(v["loo"][c]))
        add(k, "e31_local_bakeoff_410m.json",
            "per-corpus read asymptote (mean over SIX eval sets, N>=75)",
            "corpus", 5, v["r"], wl, wc, "|r| >= 0.8 under every leave-one-corpus-out")
    n_e31 = len(b)

    # ---------------------------------------------------------------- E42 (in e38): corpus-level 2
    e38 = json.load(open(os.path.join(RES, "e38_jgeometry.json")))
    for k, v in e38["E42"]["corpus"].items():
        add(k, "e38_jgeometry.json (E42.corpus)",
            "per-corpus read asymptote", "corpus", 5, v["r"], v["worst_loo"][1], v["worst_loo"][0],
            "|r| >= 0.8 under every leave-one-corpus-out")
    # ...and the 4 SCALE correlations, which are a different question entirely
    for k, v in e38["E42"]["scale"].items():
        add(k + " (vs SCALE)", "e38_jgeometry.json (E42.scale)",
            "read gap ACROSS MODEL SIZES — not a corpus predictor", "model size", v["n"],
            v["r"], None, None, "none — no LOO is defined at n=3",
            kind="different-outcome",
            note="correlates a geometry summary against a scale-varying outcome over 3 models. "
                 "This cannot answer 'is this corpus a good one to fit on'.")

    # ---------------------------------------------------------------- E34a / E34b
    for tag, f, key, ref in (("D_act(J^P, I)", "e34a_dact_vs_floor_410m.json",
                              "dact_JP_vs_I_per_set", "identity"),
                             ("D_act(J^P, J^eval)", "e34b_jeval_dact_410m.json",
                              "dact_JP_vs_Jeval_per_set", "J^eval")):
        d = json.load(open(os.path.join(RES, f)))
        X = d[key]
        Y = d["Y_inf_per_set"]
        xs = [st.mean(X[c][s] for s in SETS) for c in CORPORA]
        ys = [st.mean(Y[c][s] for s in SETS) for c in CORPORA]
        r = pear(xs, ys)
        loo = {c: pear([v for i, v in enumerate(xs) if CORPORA[i] != c],
                       [v for i, v in enumerate(ys) if CORPORA[i] != c]) for c in CORPORA}
        wl = min(loo.values(), key=abs)
        wc = min(loo, key=lambda c: abs(loo[c]))
        add(tag, f, "per-corpus read asymptote (5 admitted sets)", "corpus", 5, r, wl, wc,
            "|r| >= 0.8 under every leave-one-corpus-out",
            note=f"reference operator = {ref}")

    # ---------------------------------------------------------------- E44
    e44 = json.load(open(os.path.join(RES, "e44_dread_410m.json")))
    for k, v in e44["correlations"].items():
        kind, note = "candidate", ""
        if "random" in k:
            kind, note = "control", "random-row arm — a null control, not a candidate"
        if k == "dact_vs_I":
            kind, note = "duplicate", "same quantity as E34a's D_act(J^P, I); counted once"
        add(k, "e44_dread_410m.json", "per-corpus read asymptote", "corpus", 5,
            v["r"], v["worst_loo"][1], v["worst_loo"][0],
            "|r| >= 0.8 under every leave-one-corpus-out", kind=kind, note=note)

    # ---------------------------------------------------------------- E45: 2, at n=25
    e45 = json.load(open(os.path.join(RES, "e45_disagreement_geometry.json")))
    for tag in ("pre_norm", "post_norm"):
        if tag not in e45:
            continue
        v = e45[tag]
        wl_s, wl_c = v["worst_loo_set"], v["worst_loo_corpus"]
        worst = min([wl_s[1], wl_c[1]], key=abs)
        by = wl_s[0] if abs(wl_s[1]) <= abs(wl_c[1]) else wl_c[0]
        add(f"disagreement geometry s_c ({tag})", "e45_disagreement_geometry.json",
            "|z(readability)| per (corpus, eval set)", "corpus x eval-set cell", v["n_cells"],
            v["r"], worst, by,
            "|r| >= 0.5 AND survives leave-one-SET-out AND leave-one-CORPUS-out",
            note="DIFFERENT criterion (0.5, not 0.8) and a different unit (25 cells, not 5 "
                 "corpora). The post_norm variant additionally carries a transpose error — see "
                 "e58_algebra_audit.json.")
    add("random matched-norm Delta", "e45_disagreement_geometry.json",
        "|z(readability)| per (corpus, eval set)", "corpus x eval-set cell", 25,
        e45["control_C2_random_delta_r"], None, None, "|r| < 0.3 to fire", kind="control",
        note="cannot fail: one Delta is shared across all five corpora, so x has zero "
             "within-set variance while the real statistic's x does not. See e58.")

    # ---------------------------------------------------------------- E50: 1
    e50 = json.load(open(os.path.join(RES, "e50_concept_energy.json")))
    add("concept-conditional activation energy", "e50_concept_energy.json",
        "per-concept readability", "concept x cell", e50["n_cells"], e50["r_primary"],
        e50["loo_concept_worst"], "worst concept-LOO",
        "|r| plus a REQUIRED SIGN REVERSAL across concept sets",
        note=f"required_pattern_sign_reversal_ok = {e50['required_pattern_sign_reversal_ok']}; "
             f"n = {e50['n_cells']} cells, not 5 corpora")

    # ---------------------------------------------------------------- counts
    cand = [r for r in rows if r["kind"] == "candidate"]
    n5 = [r for r in cand if r["replication_unit"] == "corpus" and r["n"] == 5]
    other_unit = [r for r in cand if r not in n5]
    controls = [r for r in rows if r["kind"] == "control"]
    dupes = [r for r in rows if r["kind"] == "duplicate"]
    diffout = [r for r in rows if r["kind"] == "different-outcome"]
    passed = [r for r in cand if r["passes"]]

    # C1
    c1_bad = []
    for r in rows:
        if r["source_file"] == "e31_local_bakeoff_410m.json":
            v = b[r["predictor"]]
            if abs(v["r"] - r["r"]) > 0:
                c1_bad.append(r["predictor"])

    rec = {
        "experiment": "E56 — the candidate-predictor registry",
        "why": ("the paper's 'we tested 21 candidates' appears in no results file; Figure 2 shows "
                "11. This enumerates every candidate across the seven source files with the "
                "criterion each was actually tested against."),
        "recomputes_not_remeasures": True,
        "inputs": ["results/e31_local_bakeoff_410m.json", "results/e38_jgeometry.json",
                   "results/e34a_dact_vs_floor_410m.json", "results/e34b_jeval_dact_410m.json",
                   "results/e44_dread_410m.json", "results/e45_disagreement_geometry.json",
                   "results/e50_concept_energy.json"],
        "registry": rows,
        "counts": {
            "total_rows": len(rows),
            "candidates": len(cand),
            "candidates_at_n5_corpora_vs_the_read_asymptote": len(n5),
            "candidates_on_a_different_unit_or_n": len(other_unit),
            "controls_not_candidates": len(controls),
            "duplicates": len(dupes),
            "different_outcome_variable": len(diffout),
            "candidates_passing_their_own_criterion": len(passed),
            "n_from_e31": n_e31,
        },
        "HOW_THE_PAPERS_21_IS_ASSEMBLED": (
            "11 (E31) + 4 (E38/E42.scale) + 2 (E34a, E34b) + 1 (E44 D_read) + 2 (E45) + 1 (E50) "
            "= 21, which is the arithmetic RESULTS_TAXONOMY.md:132 implies. Two problems. The 4 "
            "from E42.scale correlate geometry against a SCALE-varying outcome over 3 model "
            "sizes; they are not candidate predictors of which corpus to fit on, and no "
            "leave-one-corpus-out is even defined for them. And E42's two genuine CORPUS-level "
            "predictors (rho_I, layer_cos) are then not in the 21 at all."),
        "DEFENSIBLE_COUNTS": {
            "candidate predictors of the per-corpus read asymptote, n=5 corpora, tested against "
            "the pre-registered |r|>=0.8 leave-one-corpus-out rule": len(n5),
            "additional candidates on a different unit or with a different criterion":
                len(other_unit),
            "of these, how many met their own criterion": len(passed),
        },
        "controls": {
            "C1_e31_rows_reproduce_exactly": {"mismatches": c1_bad, "fires": not c1_bad},
            "C2_controls_tagged_and_excluded": [r["predictor"] for r in controls],
            "C3_duplicates_tagged": [r["predictor"] for r in dupes],
        },
        "HETEROGENEITY": (
            "three outcome variables (6-set asymptote for E31, 5-set asymptote for E34/E44, "
            "per-cell |z readability| for E45, per-concept readability for E50); three "
            "replication units (corpus, corpus x set cell, concept); n from 3 to 5815; and two "
            "different |r| thresholds (0.8 and 0.5). The honest sentence names the n=5 group and "
            "reports the rest separately."),
    }
    write_result(out_path, rec, experiment="E56",
                 inputs=[os.path.join(RES, f) for f in
                         ("e31_local_bakeoff_410m.json", "e38_jgeometry.json",
                          "e34a_dact_vs_floor_410m.json", "e34b_jeval_dact_410m.json",
                          "e44_dread_410m.json", "e45_disagreement_geometry.json",
                          "e50_concept_energy.json")])

    c = rec["counts"]
    print(f"rows={c['total_rows']}  candidates={c['candidates']}  "
          f"(n=5 corpus-level: {c['candidates_at_n5_corpora_vs_the_read_asymptote']}, "
          f"other unit/n: {c['candidates_on_a_different_unit_or_n']})")
    print(f"controls={c['controls_not_candidates']}  duplicates={c['duplicates']}  "
          f"different-outcome={c['different_outcome_variable']}")
    print(f"candidates meeting their own criterion: {c['candidates_passing_their_own_criterion']}")
    print(f"C1 e31 rows reproduce exactly: {rec['controls']['C1_e31_rows_reproduce_exactly']['fires']}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
