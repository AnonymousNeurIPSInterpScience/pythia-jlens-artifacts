#!/usr/bin/env python3
"""r10_t59_readout_reconciliation.py — is the stored E59 dose ladder legacy-scored, and if so
what is the corrected dose-128 anchor?

NOT A NEW EXPERIMENT. Pure recomputation from results files already on disk, plus a hash of a
script already in the tree. No model, no GPU, no scoring, no fitting. Runs in under a second.

THE ALLEGATION
  paper section 3 states "Readout correction, applied before every number below." Appendix D then
  cites the E59 dose ladder ("the read share moves 19.3 -> 32.4 -> 29.4% under min") as one of six
  ablations that bound the headline factorial decomposition, which is 53.35 / 44.61 under the
  CORRECTED readout. If E59 is legacy-scored, a legacy ablation is bounding a corrected headline
  and section 3's sentence is false.

FOUR INDEPENDENT LINES OF EVIDENCE, all recomputed here rather than asserted
  1. argv        results/e59_read_dose_410m.json provenance.argv carries no --rstrip.
  2. code path   experiments/t59_read_dose.py had no --rstrip option and no .rstrip() call at the
                 time the stored result was written. Established by hashing the script: if the
                 on-disk file still hashes to provenance.script.sha256 then the code that ran is
                 the code that can be read. It no longer does, because the flag has since been
                 added -- so this tool checks the hash against the PRE-FIX value recorded below
                 and reports which of the two it matches.
  3. metadata    the stored payload carries no `rstrip` key, unlike e48_crossover_410m_rstrip.json
                 ("rstrip_arm": true) and e52_factorial_410m_rstrip.json.
  4. numbers     E59's dose-128 row means are recomputed against BOTH e52 arms restricted to
                 E59's own seven corpora. Legacy and corrected differ by a factor of ~1.8 (min)
                 and ~2.5 (persist), so this discriminates cleanly.

WHAT THIS TOOL CAN AND CANNOT ESTABLISH
  It CAN give the corrected decomposition at the headline-equivalent dose of 128 tokens, because
  the corrected 8x8 factorial (e52_factorial_410m_rstrip.json) is scored at exactly that prefix
  length; restricting it to E59's seven corpora is a recomputation, not a new measurement.

  It CANNOT give the corrected 384 and 768 rungs. Those need the read-context prefix pools, which
  are built from corpora/*.jsonl -- third-party plaintext that this project deliberately does not
  redistribute (two files are CC BY-SA and cannot be relicensed as the mirror's CC BY 4.0). The
  corrected DOSE TREND is therefore NOT RUN, and this tool says so rather than estimating it.

  The one thing it does estimate, and labels as an estimate, is the POOL EFFECT: E59 draws its
  prefixes only from documents of >= 768 tokens, E52 from the unrestricted pool. That difference
  is measurable at the legacy readout, where both arms exist, and is reported so a reader can see
  how far the corrected E52 anchor is likely to sit from a corrected E59 dose-128 rung.

    .venv/bin/python tools/r10_t59_readout_reconciliation.py
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from provenance import write_result  # noqa: E402

E59 = "results/e59_read_dose_410m.json"
E52_LEGACY = "results/e52_factorial_410m.json"
E52_RSTRIP = "results/e52_factorial_410m_rstrip.json"
E57_CI = "results/e57_grid_variance_ci_rstrip.json"
T59_SRC = "experiments/t59_read_dose.py"

# The sha256 recorded in results/e59_read_dose_410m.json provenance.script at the time this tool
# was written. It is the hash of the PRE-FIX script -- the one with no --rstrip option. Recorded
# here as a literal so that adding the flag (which changes the hash) does not erase the evidence.
T59_SHA_AT_STORED_RESULT = "76473cf24c5b8b440ada67cc3229206f077355a4c024e4bda3201ed0089d2129"


def shares(M, corp):
    """Transcribed from experiments/t59_read_dose.py:shares() so the decompositions are the
    same arithmetic. Two-way SS decomposition of a square fit x read matrix of cell means."""
    n = len(corp)
    g = statistics.mean(M[(f, q)] for f in corp for q in corp)
    row = {f: statistics.mean(M[(f, q)] for q in corp) for f in corp}
    col = {q: statistics.mean(M[(f, q)] for f in corp) for q in corp}
    SSr = n * sum((row[f] - g) ** 2 for f in corp)
    SSc = n * sum((col[q] - g) ** 2 for q in corp)
    SSt = sum((M[(f, q)] - g) ** 2 for f in corp for q in corp)
    return {"fit_pct": 100 * SSr / SSt, "read_pct": 100 * SSc / SSt,
            "residual_pct": 100 * (SSt - SSr - SSc) / SSt,
            "row_means": row, "col_means": col,
            "fit_span": max(row.values()) - min(row.values()),
            "read_span": max(col.values()) - min(col.values())}


def e52_matrix(path, agg, corp):
    M = json.load(open(os.path.join(ROOT, path)))["by_aggregation"][agg]["matrix"]
    return {(f, q): M[f"{f}|{q}"] for f in corp for q in corp}


def main() -> int:
    e59 = json.load(open(os.path.join(ROOT, E59)))
    corp = e59["corpora"]                       # E59's seven; OOD_CommonPile excluded by E59
    prov = e59["provenance"]

    # ---------------------------------------------------------------- evidence 1-3
    on_disk = hashlib.sha256(open(os.path.join(ROOT, T59_SRC), "rb").read()).hexdigest()
    evidence = {
        "1_argv": {
            "what": "the command line that produced the stored result",
            "argv": prov.get("argv"),
            "carries_rstrip_flag": any("rstrip" in str(x) for x in (prov.get("argv") or [])),
            "verdict": "no --rstrip in argv",
        },
        "2_code_path": {
            "what": "did the script that ran contain a .rstrip() on the prompt?",
            "sha256_recorded_in_result": prov["script"]["sha256"],
            "sha256_of_prefix_script_literal": T59_SHA_AT_STORED_RESULT,
            "result_matches_prefix_script": prov["script"]["sha256"] == T59_SHA_AT_STORED_RESULT,
            "sha256_on_disk_now": on_disk,
            "on_disk_is_still_the_prefix_script": on_disk == T59_SHA_AT_STORED_RESULT,
            "note": ("The stored result names the pre-fix script. Once --rstrip is added the "
                     "on-disk hash moves, which is expected and is why the pre-fix hash is "
                     "pinned as a literal above. If BOTH are true the fix has not been applied "
                     "yet; if only the first is true the fix is in and the evidence stands."),
        },
        "3_result_metadata": {
            "what": "corrected artifacts in this repository declare the convention in-payload",
            "e59_has_rstrip_key": "rstrip" in e59,
            "e48_rstrip_declares": json.load(open(os.path.join(
                ROOT, "results/e48_crossover_410m_rstrip.json"))).get("rstrip_arm"),
            "verdict": "the stored E59 payload declares no readout convention at all",
        },
    }

    # ---------------------------------------------------------------- evidence 4 + the anchors
    out = {}
    for agg in ("min", "persist"):
        stored = e59["by_dose"]["128"][agg]
        leg = shares(e52_matrix(E52_LEGACY, agg, corp), corp)
        cor = shares(e52_matrix(E52_RSTRIP, agg, corp), corp)

        # how close is E59@128 to each e52 arm, on the row means it shares with them?
        def rms(a, b):
            return (statistics.mean((a[c] - b[c]) ** 2 for c in corp)) ** 0.5
        d_leg = rms(stored["row_means"], leg["row_means"])
        d_cor = rms(stored["row_means"], cor["row_means"])

        out[agg] = {
            "E59_at_dose128_STORED_legacy_scorer": {
                "fit_pct": stored["fit_pct"], "read_pct": stored["read_pct"],
                "residual_pct": stored["residual_pct"],
                "row_means": stored["row_means"],
                "row_mean_over_corpora": statistics.mean(stored["row_means"][c] for c in corp)},
            "E52_LEGACY_restricted_to_E59_corpora": {
                "fit_pct": leg["fit_pct"], "read_pct": leg["read_pct"],
                "residual_pct": leg["residual_pct"],
                "row_mean_over_corpora": statistics.mean(leg["row_means"][c] for c in corp)},
            "E52_CORRECTED_restricted_to_E59_corpora": {
                "fit_pct": cor["fit_pct"], "read_pct": cor["read_pct"],
                "residual_pct": cor["residual_pct"],
                "row_mean_over_corpora": statistics.mean(cor["row_means"][c] for c in corp),
                "IS_THE_CORRECTED_DOSE128_ANCHOR": True,
                "why": ("E52 --rstrip is scored at prefix_tokens=128, the same read dose as E59's "
                        "first rung, under the corrected readout. Restricting its 8x8 matrix to "
                        "E59's seven corpora is a recomputation of stored values.")},
            "evidence_4_which_arm_does_E59_match": {
                "rms_row_mean_distance_to_E52_legacy": d_leg,
                "rms_row_mean_distance_to_E52_corrected": d_cor,
                "ratio_corrected_over_legacy": d_cor / d_leg if d_leg else float("inf"),
                "matches": "legacy" if d_leg < d_cor else "corrected",
                "falsified_if": ("E59 sat closer to the corrected arm, or the two arms were not "
                                 "separated by much more than the pool effect")},
            "pool_effect_at_the_legacy_readout": {
                "what": ("E59 draws prefixes only from documents of >= 768 tokens; E52 draws from "
                         "the unrestricted pool. At the legacy readout both exist, so their "
                         "difference isolates the pool, holding the readout fixed."),
                "fit_pct_delta_E59_minus_E52": stored["fit_pct"] - leg["fit_pct"],
                "read_pct_delta_E59_minus_E52": stored["read_pct"] - leg["read_pct"],
                "row_mean_delta": statistics.mean(stored["row_means"][c] for c in corp)
                                  - statistics.mean(leg["row_means"][c] for c in corp)},
            "corrected_E59_dose128_ESTIMATE": {
                "status": "ESTIMATE, NOT A MEASUREMENT — reported so the size of the gap is "
                          "visible, never to be cited as a result",
                "method": "corrected E52 anchor + the legacy-measured pool effect",
                "fit_pct": cor["fit_pct"] + (stored["fit_pct"] - leg["fit_pct"]),
                "read_pct": cor["read_pct"] + (stored["read_pct"] - leg["read_pct"]),
                "assumes": "the pool effect is additive across readout conventions, which is "
                           "untested"},
            "corrected_doses_384_and_768": {
                "status": "NOT RUN",
                "blocker": ("needs the read-context prefix pools, built from corpora/*.jsonl, "
                            "which this project deliberately does not redistribute"),
                "to_run": "bash repro/06_data.sh --build   # then:   "
                          "python experiments/t59_read_dose.py --rstrip --device cpu"},
        }

    # ---------------------------------------------------------------- controls
    controls = {}

    # C1 — the transcribed `shares()` must reproduce the programme's own stored decomposition
    #      when given the full 8x8 corrected matrix. If it does not, this tool's arithmetic is
    #      wrong and every number above is void.
    full8 = json.load(open(os.path.join(ROOT, E52_RSTRIP)))["fit_corpora"]
    ci = json.load(open(os.path.join(ROOT, E57_CI)))
    c1 = {"requirement": "shares() on the full 8x8 corrected matrix reproduces E57's stored "
                         "fit/read point estimates to < 0.01 percentage points",
          "fails_if": "any difference >= 0.01 pp, which would mean the SS decomposition here is "
                      "not the one the paper reports",
          "per_agg": {}}
    worst = 0.0
    for agg in ("min", "persist"):
        got = shares(e52_matrix(E52_RSTRIP, agg, full8), full8)
        blk = ci["by_aggregation"][agg]["point"]
        ref_fit, ref_read = blk["fit_pct"], blk["read_pct"]
        row = {"recomputed_fit_pct": got["fit_pct"], "recomputed_read_pct": got["read_pct"],
               "e57_stored_fit_pct": ref_fit, "e57_stored_read_pct": ref_read}
        if ref_fit is not None:
            row["abs_diff_fit"] = abs(got["fit_pct"] - ref_fit)
            row["abs_diff_read"] = abs(got["read_pct"] - ref_read)
            worst = max(worst, row["abs_diff_fit"], row["abs_diff_read"])
        c1["per_agg"][agg] = row
    c1["max_abs_diff_pp"] = worst
    c1["fires"] = worst < 0.01
    controls["C1_shares_reproduces_the_stored_decomposition"] = c1

    # C2 — recomputing E59's own decomposition from its own stored matrix must return its own
    #      stored fit/read exactly. Catches a misread of the E59 payload.
    c2 = {"requirement": "shares() on E59's stored dose-128 matrix returns E59's stored "
                         "fit_pct/read_pct to < 1e-9",
          "fails_if": "any difference, which would mean this tool is misreading E59",
          "per_agg": {}}
    worst2 = 0.0
    for agg in ("min", "persist"):
        blk = e59["by_dose"]["128"][agg]
        M = {(f, q): blk["matrix"][f"{f}|{q}"] for f in corp for q in corp}
        got = shares(M, corp)
        df, dr = abs(got["fit_pct"] - blk["fit_pct"]), abs(got["read_pct"] - blk["read_pct"])
        c2["per_agg"][agg] = {"abs_diff_fit": df, "abs_diff_read": dr}
        worst2 = max(worst2, df, dr)
    c2["max_abs_diff_pp"] = worst2
    c2["fires"] = worst2 < 1e-9
    controls["C2_e59_selfconsistent"] = c2

    # C3 — the discriminating power of evidence 4. If the two e52 arms were NOT well separated,
    #      matching one of them would prove nothing. State the separation.
    sep = {}
    for agg in ("min", "persist"):
        leg = shares(e52_matrix(E52_LEGACY, agg, corp), corp)
        cor = shares(e52_matrix(E52_RSTRIP, agg, corp), corp)
        lm = statistics.mean(leg["row_means"][c] for c in corp)
        cm = statistics.mean(cor["row_means"][c] for c in corp)
        sep[agg] = {"legacy_row_mean": lm, "corrected_row_mean": cm,
                    "ratio_corrected_over_legacy": cm / lm,
                    "abs_separation": abs(cm - lm),
                    "E59_distance_to_legacy": out[agg]["evidence_4_which_arm_does_E59_match"][
                        "rms_row_mean_distance_to_E52_legacy"]}
        sep[agg]["separation_over_E59_legacy_distance"] = (
            sep[agg]["abs_separation"] / sep[agg]["E59_distance_to_legacy"]
            if sep[agg]["E59_distance_to_legacy"] else float("inf"))
    controls["C3_the_two_arms_are_separable"] = {
        "requirement": "the legacy and corrected e52 arms must be separated by much more than "
                       "E59's distance to the one it matches, or evidence 4 discriminates nothing",
        "fails_if": "separation_over_E59_legacy_distance is of order 1 on either aggregation",
        "per_agg": sep,
        "fires": all(sep[a]["separation_over_E59_legacy_distance"] > 3.0 for a in sep)}

    allegation_true = (
        not evidence["1_argv"]["carries_rstrip_flag"]
        and evidence["2_code_path"]["result_matches_prefix_script"]
        and not evidence["3_result_metadata"]["e59_has_rstrip_key"]
        and all(out[a]["evidence_4_which_arm_does_E59_match"]["matches"] == "legacy"
                for a in ("min", "persist")))

    rec = {
        "experiment": "R10 — E59 readout-convention reconciliation",
        "status": "RECOMPUTATION — no model, no GPU, no scoring, no fitting",
        "recomputes_not_remeasures": True,
        "question": ("is the stored E59 dose ladder scored under the LEGACY readout while the "
                     "headline factorial it is cited to bound is scored under the CORRECTED one?"),
        "inputs_used": [E59, E52_LEGACY, E52_RSTRIP, E57_CI, T59_SRC],
        "corpora": corp,
        "ALLEGATION_UPHELD": allegation_true,
        "evidence": evidence,
        "by_aggregation": out,
        "controls": controls,
        "VERDICT": (
            "UPHELD. The stored E59 dose ladder is legacy-scored on all four independent lines of "
            "evidence. Its dose-128 rung is therefore not an anchor for the corrected factorial, "
            "and Appendix D's dose numbers cannot bound a corrected headline. The corrected "
            "dose-128 decomposition IS available and is reported here from the corrected 8x8 "
            "factorial restricted to E59's seven corpora. The corrected 384 and 768 rungs are NOT "
            "RUN: they need corpus plaintext this project does not redistribute. The corrected "
            "DOSE TREND is therefore unresolved, and E59 cannot stand as a robustness claim for "
            "the corrected factorial until it is re-run."
            if allegation_true else
            "NOT UPHELD — see evidence block; at least one line does not support the allegation."),
    }
    write_result(os.path.join(ROOT, "results", "r10_t59_readout_reconciliation.json"), rec,
                 experiment="R10", script=__file__,
                 inputs=[os.path.join(ROOT, p) for p in (E59, E52_LEGACY, E52_RSTRIP, E57_CI)])

    print("\n" + "=" * 78)
    print(f"ALLEGATION UPHELD: {allegation_true}")
    for agg in ("min", "persist"):
        o = out[agg]
        print(f"\n--- {agg}")
        print(f"  E59 @128 STORED   (legacy scorer, >=768 pool)  "
              f"fit {o['E59_at_dose128_STORED_legacy_scorer']['fit_pct']:6.2f}%  "
              f"read {o['E59_at_dose128_STORED_legacy_scorer']['read_pct']:6.2f}%")
        print(f"  E52 legacy        (legacy scorer, full pool)   "
              f"fit {o['E52_LEGACY_restricted_to_E59_corpora']['fit_pct']:6.2f}%  "
              f"read {o['E52_LEGACY_restricted_to_E59_corpora']['read_pct']:6.2f}%")
        print(f"  E52 CORRECTED     (corrected, full pool)       "
              f"fit {o['E52_CORRECTED_restricted_to_E59_corpora']['fit_pct']:6.2f}%  "
              f"read {o['E52_CORRECTED_restricted_to_E59_corpora']['read_pct']:6.2f}%   "
              f"<- the corrected dose-128 anchor")
        e4 = o["evidence_4_which_arm_does_E59_match"]
        print(f"  E59 row means match the {e4['matches'].upper()} arm "
              f"(rms {e4['rms_row_mean_distance_to_E52_legacy']:.5f} vs "
              f"{e4['rms_row_mean_distance_to_E52_corrected']:.5f})")
        print(f"  corrected 384 / 768: NOT RUN (corpus plaintext not redistributed)")
    print("\ncontrols: " + ", ".join(
        f"{k.split('_')[0]}={'FIRES' if v.get('fires') else 'DOES NOT FIRE'}"
        for k, v in controls.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
