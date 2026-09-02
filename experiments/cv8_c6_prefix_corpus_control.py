#!/usr/bin/env python3
"""cv8_c6_prefix_corpus_control.py — CV8's C6, the registered control that was never implemented.

PRE-REGISTRATION: docs/experiments/preregs/CV8_positional_extrapolation.md, CONTROLS table, C6:

    the prefix corpus does not drive the verdict: repeat the primary with the read prefix drawn
    from a second corpus (Github, the extreme arm)
    MUST PRODUCE: the PRIMARY branch must be the same under both prefixes; if it is not, the
    verdict is UNCLEAR regardless of either p-value.

WHY IT EXISTS, plainly. The positional probe G_l(p) is built from prefixes drawn from ONE corpus.
CV8's claim is "position does not explain the corpus ordering". If the instrument itself is
corpus-dependent, that argument is circular. C6 asks one thing: does the instrument care which
corpus built it?

THIS FILE COMPUTES NOTHING NEW. Both arms are produced by the SAME script, experiments/
cv8_positional_extrapolation.py, differing only in --prefix-corpus. This adjudicator reads the two
stored results files, asserts that they differ in the prefix corpus and in NOTHING ELSE that could
move the primary, classifies each arm's PRIMARY branch with the registered thresholds, and applies
C6's rule. It writes a third file and edits neither input.

    .venv/bin/python experiments/cv8_c6_prefix_corpus_control.py

WHAT THIS DOES NOT COVER. It adjudicates C6 only. It cannot change either arm's rho, p, or
alignment profile, and it says nothing about whether n=8 gives CV8 the power to detect a moderate
relationship -- CV8's own registration bounds that and this file re-reports the bound rather than
re-deriving it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

RES = os.path.join(ROOT, "results")

# The registered thresholds, transcribed from CV8's DECISION RULE. Not re-cut, not re-derived.
P_EXPLAINS = 0.05
P_NOT_EXPLAINS = 0.20

# Fields that must be IDENTICAL between the two arms. If any differs, the arms are not "the same
# primary with a different prefix" and C6 cannot be adjudicated on them. `n_sequences_for_G` is in
# here because a shorter arm would estimate G(p) worse and could move the branch for a reason that
# has nothing to do with the prefix corpus.
MUST_MATCH = ["model", "band", "device", "smoke", "p_read", "probe_positions",
              "in_support_positions", "out_of_support_positions", "n_sequences_for_G",
              "read_scores_min_from_r9"]


def branch(rec: dict) -> tuple[str, str]:
    """Classify one arm's PRIMARY into the registered branches, from the stored numbers."""
    if not rec["controls"]["C2_metric_is_not_vacuous"]["fires"]:
        return "VOID", "C2 did not fire: the alignment metric is vacuous on this arm"
    p = rec["PRIMARY"]["exact_two_sided_p"]
    if p <= P_EXPLAINS:
        return "EXPLAINS", f"exact two-sided p = {p:.4f} <= {P_EXPLAINS}"
    if p >= P_NOT_EXPLAINS:
        return "NOT_EXPLAINED", f"exact two-sided p = {p:.4f} >= {P_NOT_EXPLAINS}"
    return "UNCLEAR", f"exact two-sided p = {p:.4f} falls in the registered {P_EXPLAINS}..{P_NOT_EXPLAINS} band"


def arm_summary(rec: dict) -> dict:
    pr = rec["PRIMARY"]
    b, why = branch(rec)
    return {
        "prefix_corpus": rec["prefix_corpus"],
        "n_sequences_for_G": rec["n_sequences_for_G"],
        "rho": pr["rho"],
        "exact_two_sided_p": pr["exact_two_sided_p"],
        "n_permutations": pr["n_permutations"],
        "critical_abs_rho_at_alpha_0.05": pr["critical_abs_rho_at_alpha_0.05"],
        "abs_rho_below_critical": abs(pr["rho"]) < pr["critical_abs_rho_at_alpha_0.05"],
        "corpora_in_order": pr["corpora_in_order"],
        "alignment_at_p_read": pr["alignment_at_p_read"],
        "branch": b, "branch_reason": why,
        "controls_fired": rec["controls_fired"],
        "VERDICT_as_stored": rec["VERDICT"],
    }


def main() -> int:
    import provenance as prov

    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default=os.path.join(RES, "cv8_positional_extrapolation.json"))
    ap.add_argument("--second", default=os.path.join(RES, "cv8_positional_extrapolation_github.json"))
    ap.add_argument("--out", default=os.path.join(RES, "cv8_c6_prefix_corpus_control.json"))
    a = ap.parse_args()

    for p in (a.primary, a.second):
        if not os.path.exists(p):
            raise SystemExit(f"ABORT: missing arm {p}. C6 needs both arms; it is not adjudicable "
                             f"from one.")
    A = json.load(open(a.primary))
    B = json.load(open(a.second))

    # ---- integrity: the arms must differ in the prefix corpus and in nothing else ------------
    if A["prefix_corpus"] == B["prefix_corpus"]:
        raise SystemExit(f"ABORT: both arms used prefix_corpus={A['prefix_corpus']!r}. C6 requires "
                         f"a SECOND corpus; comparing an arm with itself cannot fail.")
    mismatched = {k: [A.get(k), B.get(k)] for k in MUST_MATCH if A.get(k) != B.get(k)}
    if mismatched:
        raise SystemExit(f"ABORT: the arms differ in more than the prefix corpus, so C6 is not "
                         f"adjudicable on them: {json.dumps(mismatched)[:600]}")
    same_script = (A["provenance"]["script"]["sha256"] == B["provenance"]["script"]["sha256"])

    sa, sb = arm_summary(A), arm_summary(B)
    fires = sa["branch"] == sb["branch"]

    # ---- C6's registered rule, applied verbatim ----------------------------------------------
    if fires:
        verdict = (f"C6 FIRES — both prefixes land in the same PRIMARY branch ({sa['branch']}): "
                   f"{sa['prefix_corpus']} p = {sa['exact_two_sided_p']:.4f}, "
                   f"{sb['prefix_corpus']} p = {sb['exact_two_sided_p']:.4f}. The prefix corpus "
                   f"does not drive the verdict, so the instrument is not corpus-dependent in the "
                   f"way that would make CV8's argument circular. CV8's verdict stands as recorded.")
        cv8 = "CV8's verdict STANDS as recorded — C6 fired."
    else:
        verdict = (f"C6 DOES NOT FIRE — the prefixes land in DIFFERENT PRIMARY branches: "
                   f"{sa['prefix_corpus']} -> {sa['branch']} (p = {sa['exact_two_sided_p']:.4f}), "
                   f"{sb['prefix_corpus']} -> {sb['branch']} (p = {sb['exact_two_sided_p']:.4f}). "
                   f"C6 as registered: 'if it is not, the verdict is UNCLEAR regardless of either "
                   f"p-value.' CV8 IS THEREFORE UNCLEAR. The rule is not re-cut and no p-value is "
                   f"reinterpreted to avoid this.")
        cv8 = ("CV8 IS UNCLEAR — a registered control did not fire: C6. This supersedes the "
               "verdict string stored in results/cv8_positional_extrapolation.json, which was "
               "written before C6 existed and could not account for it.")

    rec = {
        "experiment": "CV8 C6 — does the prefix corpus drive the verdict?",
        "prereg": "docs/experiments/preregs/CV8_positional_extrapolation.md",
        "prereg_control": "C6",
        "status": "PRE-REGISTERED — the control was registered 2026-08-29 and implemented now",
        "adjudicates": "results/cv8_positional_extrapolation.json:VERDICT — and nothing else",
        "recomputes_not_remeasures": True,
        "C6_required_verbatim": (
            "the prefix corpus does not drive the verdict: repeat the primary with the read prefix "
            "drawn from a second corpus (Github, the extreme arm). The PRIMARY branch must be the "
            "same under both prefixes; if it is not, the verdict is UNCLEAR regardless of either "
            "p-value."),
        "registered_branch_thresholds": {"EXPLAINS": f"p <= {P_EXPLAINS}",
                                         "NOT_EXPLAINED": f"p >= {P_NOT_EXPLAINS}",
                                         "UNCLEAR": f"{P_EXPLAINS} < p < {P_NOT_EXPLAINS}"},
        "arms": {"primary": sa, "second": sb},
        "arms_differ_only_in_prefix_corpus": True,
        "fields_asserted_identical": MUST_MATCH,
        "same_script_sha256_both_arms": same_script,
        "source_files": {"primary": os.path.relpath(a.primary, ROOT),
                         "second": os.path.relpath(a.second, ROOT)},
        "C6_fires": bool(fires),
        "CV8_VERDICT_AFTER_C6": cv8,
        "VERDICT": verdict,
        "scope": ("Adjudicates CV8's C6 only. It changes neither arm's rho, p, nor alignment "
                  "profile, and it does not revisit CV8's stated power: n=8 detects only a strong "
                  "monotone relationship, and both arms carry that bound."),
    }

    print(verdict, flush=True)
    print(f"  {sa['prefix_corpus']:10s} rho={sa['rho']:+.4f} p={sa['exact_two_sided_p']:.4f} "
          f"branch={sa['branch']}", flush=True)
    print(f"  {sb['prefix_corpus']:10s} rho={sb['rho']:+.4f} p={sb['exact_two_sided_p']:.4f} "
          f"branch={sb['branch']}", flush=True)

    prov.write_result(a.out, rec, script=__file__, experiment="CV8_C6",
                      inputs=[a.primary, a.second])
    print("wrote", os.path.relpath(a.out, ROOT), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
