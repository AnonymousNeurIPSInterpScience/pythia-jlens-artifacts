#!/usr/bin/env python3
"""postreview_completion.py — map every item in specs/POSTREVIEW_EXPERIMENTS.md to its outcome.

Parses the slate for its own item ids, then resolves each to the pre-registration document, the
results file(s) it produced, and the adjudicated verdict. Nothing here is typed by hand: the item
list comes from the spec, the verdicts come from the results files.

    python tools/postreview_completion.py
"""
from __future__ import annotations
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "src"))
from provenance import sha256_file, write_result  # noqa: E402

R = lambda *p: os.path.join(REPO, *p)
SLATE = "specs/POSTREVIEW_EXPERIMENTS.md"

# item -> (prereg doc, results files, one-line what-it-was)
MAP = {
 "R0":  (None, [], "retract the corpus-axis crossover — OPERATOR ONLY, not an agent task"),
 "P0":  ("specs/experiments/P0_t52_parallel_port.md", ["results/e52_factorial_410m_pooled.json"],
         "port a process pool into t52_factorial.py"),
 "R1":  ("specs/experiments/R1_grid_rstrip.md",
         ["results/e52_factorial_410m_rstrip.json","results/e57_grid_variance_ci_rstrip.json"],
         "re-score the 8x8 grid at the correct readout token"),
 "R2":  ("specs/experiments/R2_e48_rstrip_arm.md", ["results/e48_crossover_410m_rstrip.json"],
         "the declared-but-never-run --rstrip sensitivity arm on E48"),
 "R3":  ("specs/experiments/R3_ladder410_cpu.md", ["results/r3_close_d2.json"],
         "close D2 — re-score ladder410 on CPU and re-derive"),
 "R4":  ("specs/experiments/R4_corrections.md",
         ["results/r4_corrections.json","results/r4e_control_power.json","results/r4g_e28_provenance.json"],
         "R4a/b-table/c/e/f/g/h/i/j/k/l — corrections that are pure recomputation"),
 "R4b": ("specs/experiments/R4b_e36_rstrip.md", ["results/r4b_e36_flatness.json"],
         "re-score E36 — S3's READ-axis rejection is a logit-lens comparison"),
 "R5":  ("specs/experiments/R5_corpus_axis_uncertainty.md",
         ["results/r5_corpus_axis_uncertainty.json","results/r5_corpus_axis_uncertainty_rstrip.json"],
         "corpus-axis uncertainty, stated as such"),
 "R6":  ("specs/experiments/R6_within_source_resampling.md",
         ["results/r6_within_source_410m.json","results/r6_within_source_410m_cpu.json"],
         "within-source resampling — is corpus a factor or a bundle?"),
 "R7":  ("specs/experiments/R7_length_matched_pools.md", ["results/r7_matched_pools_410m.json"],
         "length- and format-matched fitting pools"),
 "R8":  ("specs/experiments/R8_ladder_rstrip_S2.md", ["results/r8_ladder_flatness.json"],
         "slate §5 item S-1, self-promoted to Tier A by R1's QUALIFIED verdict — closes S2"),
 "R9":  ("specs/experiments/R9_permutation_calibrated_min.md",
         ["results/r9_permutation_calibrated_min.json"],
         "permutation-calibrated min — dissolves the metric circularity"),
}


# The adjudication lives in the specs/experiments/ document, not always in the results file:
# e52_factorial_*.json carries E52's OWN verdict (the diagonal-excess D), which is NOT P0's or R1's
# adjudication. Reading VERDICT blindly mislabels both. The verdict is therefore taken from the
# prereg document's "## VERDICT" section, which is where this repository records adjudications.
def verdict_of(paths, prereg):
    if prereg and os.path.exists(R(prereg)):
        txt = open(R(prereg)).read()
        m = re.search(r"^## VERDICT\s*\n+(.+?)(?=\n## |\Z)", txt, re.S | re.M)
        if m:
            body = m.group(1).strip()
            if body and "(pending)" not in body:
                first = re.sub(r"\s+", " ", body.split("\n\n")[0]).strip()
                return first
    for p in paths:
        if os.path.exists(R(p)):
            try:
                d = json.load(open(R(p)))
            except Exception:
                continue
            for k in ("VERDICT", "verdict"):
                if k in d:
                    return str(d[k])
    return None


def main() -> int:
    slate = open(R(SLATE)).read()
    ids_in_slate = sorted(set(re.findall(r"^####\s+(R\d+[a-z]?|P0)\.", slate, re.M)))
    rows, missing = [], []
    for item, (prereg, results, what) in MAP.items():
        pr_ok = bool(prereg) and os.path.exists(R(prereg))
        have = [p for p in results if os.path.exists(R(p))]
        v = verdict_of(results, prereg)
        rows.append({"item": item, "what": what,
                     "prereg": prereg, "prereg_exists": pr_ok,
                     "prereg_sha256": sha256_file(R(prereg)) if pr_ok else None,
                     "results_files_expected": results, "results_files_present": have,
                     "verdict": (v[:300] if v else ("OPERATOR-ONLY — not an agent task"
                                                    if item == "R0" else None))})
        if item != "R0" and (not pr_ok or len(have) != len(results)):
            missing.append(item)
    rec = {
        "experiment": "POSTREVIEW completion ledger",
        "slate": SLATE, "slate_sha256": sha256_file(R(SLATE)),
        "item_headings_parsed_from_the_slate": ids_in_slate,
        "n_items_tracked": len(MAP),
        "n_with_prereg_and_results": len(MAP) - len(missing) - 1,
        "items_incomplete": missing,
        "operator_only_outstanding": ["R0 — retraction of the corpus-axis crossover; the SLATE "
                                      "itself marks it 'OPERATOR ONLY. DO NOT EXECUTE.' Its "
                                      "MEASUREMENT is confirmed twice (R1, R2) but the canonical "
                                      "documents were deliberately not edited."],
        "tier_C_not_started_by_design": [
            "move the task battery (three-factor P x Q x T)", "canonical N=1000 protocol arm",
            "external validation of persist against causal interventions",
            "a second model family / instruction-tuned models",
            "full corpus-resampling hierarchical design (R6 is the two-source version)",
            "E66's fitter discrepancy resolved by refit"],
        "items": rows,
    }
    write_result(R("results", "postreview_completion.json"), rec, experiment="POSTREVIEW",
                 inputs=[R(SLATE)])
    print(f"slate: {SLATE}  sha256 {rec['slate_sha256'][:16]}...")
    print(f"item headings parsed from the slate itself: {ids_in_slate}\n")
    print(f"{'item':6s}{'prereg':8s}{'results':9s} verdict")
    for r in rows:
        v = (r["verdict"] or "MISSING").split("—")[0].split("--")[0].strip()[:66]
        print(f"{r['item']:6s}{('yes' if r['prereg_exists'] else '-'):8s}"
              f"{str(len(r['results_files_present']))+'/'+str(len(r['results_files_expected'])):9s} {v}")
    print(f"\nincomplete: {missing or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
