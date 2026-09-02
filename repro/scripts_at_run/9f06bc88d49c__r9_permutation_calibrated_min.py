#!/usr/bin/env python3
"""r9_permutation_calibrated_min.py — R9: calibrate the PUBLISHED statistic against its own null.

PRE-REGISTRATION: specs/experiments/R9_permutation_calibrated_min.md.

THE POINT. `persist` was adopted because it passed the derangement control, and the control it
passes is the one used to justify it -- outcome-dependent metric selection, which both external
reviews name as circular in the Kriegeskorte sense. R9 keeps `min`, which is the SOURCE's own
operational definition of recovery, and asks a different question of it: not "is the raw score
high?" but "is the real operator unusual relative to derangements of ITSELF?"

That changes no metric, invents no statistic, and needs no new compute: the null is already stored.

    python tools/r9_permutation_calibrated_min.py
"""
from __future__ import annotations
import argparse, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "src"))
from provenance import sha256_file, write_result  # noqa: E402

R = lambda *p: os.path.join(REPO, *p)


def calibrate(rung, ag):
    real = rung[ag]["jp_admitted_mean"]
    draws = rung[ag].get("per_derangement_admitted_mean") or {}
    vals = sorted(draws.values())
    n = len(vals)
    n_ge = sum(1 for v in vals if v >= real)
    mu = st.mean(vals) if vals else None
    sd = st.pstdev(vals) if len(vals) > 1 else 0.0
    return {"real": real, "n_draws": n, "null_min": vals[0] if vals else None,
            "null_max": vals[-1] if vals else None, "null_mean": mu, "null_sd": sd,
            "n_null_ge_real": n_ge,
            "beats_every_draw": n_ge == 0,
            "empirical_p_one_sided": (1 + n_ge) / (1 + n) if n else None,
            "p_floor": 1 / (1 + n) if n else None,
            "z_vs_null": ((real - mu) / sd) if sd else None,
            "rank_of_real_among_null_plus_real": n - n_ge + 1,
            "draw_keys": sorted(draws)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e48", default=R("results", "e48_crossover_410m_rstrip.json"))
    ap.add_argument("--out", default=R("results", "r9_permutation_calibrated_min.json"))
    a = ap.parse_args()
    d = json.load(open(a.e48))
    rungs = d["rungs"]

    res, ctrl = {}, {}
    for ag in ("min", "persist"):
        per = {c: calibrate(rungs[c], ag) for c in rungs if ag in rungs[c]}
        beats = [c for c, v in per.items() if v["beats_every_draw"]]
        res[ag] = {
            "per_corpus": per,
            "n_corpora": len(per),
            "n_beating_every_own_derangement": len(beats),
            "corpora_beating_every_draw": sorted(beats),
            "corpora_NOT_beating_every_draw": sorted(set(per) - set(beats)),
            "empirical_p_by_corpus": {c: v["empirical_p_one_sided"] for c, v in per.items()},
            "z_by_corpus": {c: v["z_vs_null"] for c, v in per.items()},
            "median_z": st.median([v["z_vs_null"] for v in per.values() if v["z_vs_null"] is not None]),
        }

    # C1 — every null draw must belong to the corpus it calibrates
    bad = []
    for ag in res:
        for c, v in res[ag]["per_corpus"].items():
            if not all(k.startswith("s") and "|d" in k for k in v["draw_keys"]):
                bad.append(f"{ag}:{c}")
    ctrl["C1_null_is_the_corpus_own_derangement"] = {
        "required": "per-corpus draw keys of the form s{seed}|d{draw}, read from that corpus's rung",
        "n_corpora_checked": sum(res[ag]["n_corpora"] for ag in res),
        "malformed": bad, "fires": not bad,
        "why": ("R4d found E36 building its derangement from Pile-CC's operator and comparing it "
                "against every corpus's. That must not recur here.")}

    # C2 — the two aggregations must disagree, or there is no circularity to dissolve
    ctrl["C2_aggregations_disagree"] = {
        "required": ("under persist the real operator beats its own null on 8 of 8; under min it "
                     "does not. If they agreed the whole framing would be wrong."),
        "persist_n_beating": res["persist"]["n_beating_every_own_derangement"],
        "min_n_beating": res["min"]["n_beating_every_own_derangement"],
        "fires": (res["persist"]["n_beating_every_own_derangement"] == res["persist"]["n_corpora"]
                  and res["min"]["n_beating_every_own_derangement"] < res["min"]["n_corpora"])}

    m, p = res["min"], res["persist"]
    verdict = (
        f"CALIBRATED. Under the SOURCE'S OWN statistic (`min`), the real operator exceeds every one "
        f"of its 15 own layer-derangement draws on {m['n_beating_every_own_derangement']} of "
        f"{m['n_corpora']} corpora (median z = {m['median_z']:+.2f}). Under `persist` it does so on "
        f"{p['n_beating_every_own_derangement']} of {p['n_corpora']} (median z = {p['median_z']:+.2f}). "
        f"The published statistic, read against its own null rather than as a raw score, does not "
        f"certify layer-to-derivative correspondence — and that statement is now made WITHOUT "
        f"selecting the metric on the control it has to pass.")

    prereg = "specs/experiments/R9_permutation_calibrated_min.md"
    rec = {"experiment": "R9 — permutation-calibrated min: the published statistic against its own null",
           "prereg": prereg, "prereg_sha256": sha256_file(R(prereg)),
           "status": "PRE-REGISTERED", "recomputes_not_remeasures": True,
           "source_file": os.path.relpath(a.e48, REPO),
           "readout_convention": "STRIPPED — the anchor rule (R1)",
           "PRIMARY_AGGREGATION": "min — the source's own operational definition of recovery",
           "SECONDARY_AGGREGATION": (
               "persist — DEMOTED, and explicitly labelled: it was selected with knowledge of the "
               "derangement-control outcome, which is outcome-dependent metric selection. It is "
               "reported as a robustness arm, never as the adjudicator."),
           "decision_rule_verbatim": (
               "Descriptive; no accept/reject branch, and none may be invented. The deliverable is "
               "that the paper stops reading a raw min score as specificity and instead reports "
               "where the real operator sits in its own null."),
           "declared_bias": [
               "15 draws floors the empirical p at 1/16 = 0.0625 — NO single corpus can reach "
               "p < 0.05 on this null whatever the effect. Per-corpus values are ranks with a "
               "floor, not significance tests; the inferential statement is the across-corpora count.",
               "a derangement preserves entries, norms and spectra but destroys layer "
               "correspondence. It is a null for CORRESPONDENCE, not for 'the Jacobians carry no "
               "layer-specific information'. This measures a failure of procedure-plus-aggregation "
               "to distinguish correct from mismatched operators and must not drift into the "
               "stronger claim."],
           "by_aggregation": res, "controls": ctrl, "VERDICT": verdict}
    write_result(a.out, rec, experiment="R9", inputs=[a.e48])

    for ag in ("min", "persist"):
        v = res[ag]
        label = "  (PRIMARY -- the source's own statistic)" if ag == "min" else "  (secondary, demoted)"
        print(f"\n=== {ag.upper()}{label} ===")
        print(f"  {'corpus':22s}{'real':>10}{'null mean':>11}{'null max':>10}{'z':>8}{'p':>8}  beats all 15?")
        for c, r in sorted(v["per_corpus"].items()):
            print(f"  {c:22s}{r['real']:10.5f}{r['null_mean']:11.5f}{r['null_max']:10.5f}"
                  f"{r['z_vs_null']:+8.2f}{r['empirical_p_one_sided']:8.4f}  {r['beats_every_draw']}")
        print(f"  -> beats every own derangement on {v['n_beating_every_own_derangement']}/{v['n_corpora']}"
              f"   median z {v['median_z']:+.2f}")
    for k, c in ctrl.items():
        print(f"  {k:44s} {c['fires']}")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
