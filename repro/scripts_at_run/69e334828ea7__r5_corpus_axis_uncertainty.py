#!/usr/bin/env python3
"""r5_corpus_axis_uncertainty.py — R5: state the uncertainty in the unit the claim is about.

PRE-REGISTRATION: specs/experiments/R5_corpus_axis_uncertainty.md.
Source slate: specs/POSTREVIEW_EXPERIMENTS.md §3 Tier B, item R5.

THE PROBLEM, which two independent reviewers reached separately. The paper prints ONE interval on
its headline variance split. That interval resamples SEED factors and holds the eight corpora
fixed. The claim it is printed beside -- that the fit axis dominates the read axis -- is a claim
about CORPORA. An interval whose unit is not the unit of the claim invites a population reading the
design cannot support.

  adversarial review : "a descriptive decomposition of one 8x8 grid, not a robust scientific estimate"
  reproducibility review (C-3 row 6): "the interval resamples seeds and holds the eight corpora
  fixed; the leave-k-out numbers are the honest corpus-axis uncertainty"

WHAT THIS ADDS. No new measurement. From the stored per-draw cells: the FULL leave-one-corpus-out
and leave-two-corpus-out DISTRIBUTIONS of fit_pct and read_pct -- not merely the count of splits
that preserve the ordering -- their min-max range, and a corpus-as-random-effect variance component
**with its degrees of freedom printed beside it**, so its weakness at n=8 is visible rather than
implied.

DECLARED BIAS, and it is not a footnote: n = 8 corpora, CHOSEN not sampled. A random-effect
variance component on eight non-random units is a description dressed as an inference. It is
captioned as such wherever it appears, including in this file.

    python tools/r5_corpus_axis_uncertainty.py --cells results/e57_factorial_cells_410m.json
"""
from __future__ import annotations
import argparse, itertools, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "src"))
from provenance import sha256_file, write_result  # noqa: E402

R = lambda *p: os.path.join(REPO, *p)
CORP8 = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github",
         "OOD_News_2024", "OOD_arXiv_2023", "OOD_CommonPile"]


def decomp(M, corp):
    """t57's decomposition, duplicated verbatim so this tool stands alone; C1 checks it agrees."""
    n = len(corp)
    g = st.mean(M[(f, q)] for f in corp for q in corp)
    row = {f: st.mean(M[(f, q)] for q in corp) for f in corp}
    col = {q: st.mean(M[(f, q)] for f in corp) for q in corp}
    SSr = n * sum((row[f] - g) ** 2 for f in corp)
    SSc = n * sum((col[q] - g) ** 2 for q in corp)
    SSt = sum((M[(f, q)] - g) ** 2 for f in corp for q in corp)
    return {"fit_pct": 100 * SSr / SSt, "read_pct": 100 * SSc / SSt,
            "resid_pct": 100 * (SSt - SSr - SSc) / SSt,
            "SS_row": SSr, "SS_col": SSc, "SS_total": SSt}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=R("results", "e57_factorial_cells_410m.json"))
    ap.add_argument("--ci", default=R("results", "e57_grid_variance_ci.json"),
                    help="the stored seed-bootstrap interval, for C1 and for side-by-side reporting")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    stripped = "rstrip" in os.path.basename(a.cells)
    if a.out is None:
        a.out = R("results", f"r5_corpus_axis_uncertainty{'_rstrip' if stripped else ''}.json")

    cells = json.load(open(a.cells))
    seeds, pseeds = cells["fit_seeds"], cells["prefix_seeds"]
    stored_ci = json.load(open(a.ci)) if os.path.exists(a.ci) else None
    by_ag, controls = {}, {}

    for ag in ("persist", "min"):
        D = cells["draws"][ag]
        M = {(f, q): st.mean(D[f"{f}|{q}|s{s}|p{p}"] for s in seeds for p in pseeds)
             for f in CORP8 for q in CORP8}
        full = decomp(M, CORP8)

        loo, l2o = {}, {}
        for drop in CORP8:
            sub = [c for c in CORP8 if c != drop]
            loo[f"drop_{drop}"] = decomp(M, sub)
        for d1, d2 in itertools.combinations(CORP8, 2):
            sub = [c for c in CORP8 if c not in (d1, d2)]
            l2o[f"drop_{d1}+{d2}"] = decomp(M, sub)

        def dist(d, key):
            v = sorted(x[key] for x in d.values())
            return {"n": len(v), "min": v[0], "max": v[-1], "range": v[-1] - v[0],
                    "median": st.median(v), "mean": st.mean(v),
                    "sd": st.pstdev(v) if len(v) > 1 else 0.0, "values_sorted": v}

        # corpus as a RANDOM EFFECT on the fit axis. Printed with its df because that is the
        # number that says how little it means at n=8.
        n = len(CORP8)
        row = {f: st.mean(M[(f, q)] for q in CORP8) for f in CORP8}
        g = st.mean(M[(f, q)] for f in CORP8 for q in CORP8)
        MS_fit = (n * sum((row[f] - g) ** 2 for f in CORP8)) / (n - 1)
        SSi = full["SS_total"] - full["SS_row"] - full["SS_col"]
        MS_resid = SSi / ((n - 1) * (n - 1))
        var_comp = max(0.0, (MS_fit - MS_resid) / n)

        by_ag[ag] = {
            "full_panel": full,
            "leave_one_corpus_out": {"per_split": loo, "fit_pct": dist(loo, "fit_pct"),
                                     "read_pct": dist(loo, "read_pct")},
            "leave_two_corpora_out": {"per_split": l2o, "fit_pct": dist(l2o, "fit_pct"),
                                      "read_pct": dist(l2o, "read_pct")},
            "n_loo_preserving_ordering": sum(1 for v in loo.values() if v["fit_pct"] > v["read_pct"]),
            "n_l2o_preserving_ordering": sum(1 for v in l2o.values() if v["fit_pct"] > v["read_pct"]),
            "l2o_splits_that_REVERSE": {k: v for k, v in l2o.items() if v["fit_pct"] <= v["read_pct"]},
            "corpus_random_effect_on_the_fit_axis": {
                "MS_fit": MS_fit, "df_fit": n - 1,
                "MS_residual": MS_resid, "df_residual": (n - 1) * (n - 1),
                "variance_component": var_comp,
                "CAPTION_THAT_MUST_TRAVEL_WITH_IT": (
                    f"estimated on {n} corpora that were CHOSEN, not sampled, on {n-1} degrees of "
                    f"freedom. This is a description of one panel dressed in the notation of an "
                    f"inference about a population of corpora. It is reported so the weakness is "
                    f"visible, not because it licenses a population claim.")},
        }
        if stored_ci and ag in stored_ci.get("by_aggregation", {}):
            b = stored_ci["by_aggregation"][ag].get("bootstrap", {})
            by_ag[ag]["seed_bootstrap_for_comparison"] = {
                "fit_pct": b.get("fit_pct"), "read_pct": b.get("read_pct"),
                "UNIT": "seed factors resampled; the eight corpora HELD FIXED"}

    p = by_ag["persist"]; m = by_ag["min"]
    controls["C1_full_panel_recovered"] = {
        "required": "the full-panel fit_pct must match the stored E57 point estimate to <= 1e-9",
        "stored": (stored_ci or {}).get("by_aggregation", {}).get("persist", {})
                  .get("point", {}).get("fit_pct"),
        "recomputed": p["full_panel"]["fit_pct"]}
    sv = controls["C1_full_panel_recovered"]["stored"]
    controls["C1_full_panel_recovered"]["fires"] = (
        None if sv is None else abs(sv - p["full_panel"]["fit_pct"]) <= 1e-9)
    obs = {"persist_loo": f"{p['n_loo_preserving_ordering']}/8",
           "persist_l2o": f"{p['n_l2o_preserving_ordering']}/28",
           "min_loo": f"{m['n_loo_preserving_ordering']}/8",
           "min_l2o": f"{m['n_l2o_preserving_ordering']}/28"}
    if stripped:
        # SCOPE. C2 checks that this reimplementation reproduces the STORED leave-k-out counts,
        # which were computed at the unstripped readout. Under the corrected readout the counts are
        # SUPPOSED to change, so holding the stripped arm to them would be a control failing for
        # the wrong reason -- the same mis-scoping R1/C1 and t57/C1 avoid by construction. Recorded
        # as inapplicable, with the observed counts reported as a RESULT rather than a violation.
        controls["C2_leave_k_out_counts_reproduce"] = {
            "applicable": False, "fires": None,
            "required_on_the_unstripped_arm": "8/8 LOO and 28/28 L2O under persist; 27/28 under min",
            "observed_here": obs,
            "why": ("this arm is scored at the corrected readout; the stored counts describe the "
                    "unstripped one. The change in these counts IS the finding, not a control "
                    "failure -- see leave_k_out_ordering_collapse below.")}
    else:
        controls["C2_leave_k_out_counts_reproduce"] = {
            "applicable": True,
            "required": "8/8 LOO and 28/28 L2O under persist; 27/28 under min",
            "observed": obs,
            "fires": (p["n_loo_preserving_ordering"] == 8 and p["n_l2o_preserving_ordering"] == 28
                      and m["n_l2o_preserving_ordering"] == 27)}

    rec = {
        "experiment": "R5 — corpus-axis uncertainty, stated as such",
        "prereg": "specs/experiments/R5_corpus_axis_uncertainty.md",
        "prereg_sha256": sha256_file(R("specs", "experiments", "R5_corpus_axis_uncertainty.md")),
        "status": "PRE-REGISTERED", "recomputes_not_remeasures": True,
        "readout_convention": "STRIPPED (anchor rule)" if stripped else "UNSTRIPPED (legacy)",
        "decision_rule_verbatim": (
            "Descriptive; no branch. The deliverable is that the paper stops printing one interval "
            "whose unit is not the unit the claim is about."),
        "PRIMARY": {
            "what": "the range of fit_pct over all 28 leave-two-out splits, beside the seed bootstrap",
            "persist_l2o_fit_pct_range": [p["leave_two_corpora_out"]["fit_pct"]["min"],
                                          p["leave_two_corpora_out"]["fit_pct"]["max"]],
            "min_l2o_fit_pct_range": [m["leave_two_corpora_out"]["fit_pct"]["min"],
                                      m["leave_two_corpora_out"]["fit_pct"]["max"]],
            "CAPTION": ("the leave-two-out range is CORPUS-axis uncertainty on a panel of eight "
                        "chosen corpora; the bootstrap interval is SEED-axis uncertainty with those "
                        "eight held fixed. They are not comparable and must never be printed as "
                        "though one bounded the other.")},
        "declared_bias": ("n = 8 corpora, chosen not sampled. Leave-k-out on a chosen panel measures "
                          "sensitivity to THESE corpora, not sampling error over a population of "
                          "corpora. No resampling of this panel can produce the latter."),
        "leave_k_out_ordering_collapse": {
            "what": ("how often the fit axis still beats the read axis when corpora are dropped. "
                     "Published, at the unstripped readout: 8/8 LOO and 28/28 L2O under persist, "
                     "27/28 under min."),
            "persist": {"loo": f"{p['n_loo_preserving_ordering']}/8",
                        "l2o": f"{p['n_l2o_preserving_ordering']}/28",
                        "reversing_splits": sorted(p["l2o_splits_that_REVERSE"])},
            "min": {"loo": f"{m['n_loo_preserving_ordering']}/8",
                    "l2o": f"{m['n_l2o_preserving_ordering']}/28",
                    "reversing_splits": sorted(m["l2o_splits_that_REVERSE"])}},
        "controls": controls, "by_aggregation": by_ag,
    }
    write_result(a.out, rec, experiment="R5", inputs=[a.cells] + ([a.ci] if stored_ci else []))
    print(f"readout: {rec['readout_convention']}")
    for ag in ("persist", "min"):
        v = by_ag[ag]
        print(f"[{ag}] full panel fit {v['full_panel']['fit_pct']:.3f}% "
              f"read {v['full_panel']['read_pct']:.3f}%")
        for lab, k in (("LOO ", "leave_one_corpus_out"), ("L2O ", "leave_two_corpora_out")):
            d = v[k]["fit_pct"]
            print(f"   {lab} fit_pct over {d['n']:2d} splits: "
                  f"[{d['min']:.2f}, {d['max']:.2f}]  range {d['range']:.2f} pp  median {d['median']:.2f}")
        print(f"   ordering preserved: LOO {v['n_loo_preserving_ordering']}/8, "
              f"L2O {v['n_l2o_preserving_ordering']}/28")
        if v["l2o_splits_that_REVERSE"]:
            for k, s in v["l2o_splits_that_REVERSE"].items():
                print(f"     REVERSES: {k}  fit {s['fit_pct']:.3f}% vs read {s['read_pct']:.3f}%")
    for k, c in controls.items():
        print(f"  {k:34s} {c['fires']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
