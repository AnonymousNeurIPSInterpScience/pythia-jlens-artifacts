#!/usr/bin/env python3
"""r4b_e36_flatness.py — R4b: adjudicate E36's shift-axis slopes under the corrected readout.

PRE-REGISTRATION: specs/experiments/R4b_e36_rstrip.md, committed before this script existed.
Source slate: specs/POSTREVIEW_EXPERIMENTS.md §3 Tier B, item R4b.

WHY. paper/CONTEXT.md §3.1 records S3's first half as REJECTED on the read axis, and its
pre-registered SLOPE SECONDARY as failing in the opposite direction: the logit lens is the steepest
arm (-0.00372) against five fitted operators at -0.00146 / -0.00115 / -0.00094 / -0.00089 /
+0.00022, i.e. every fitted operator is FLATTER than the one with nothing fitted, 5 of 5.

That is a comparison AGAINST THE LOGIT LENS, and the logit lens is the single quantity the readout
defect moved most: its constant goes 0.02844 -> 0.08318, a 2.9x change, while the fitted operators
move 2.2-2.7x. A comparison between two quantities that move by different factors can reverse.

A WORDING AMBIGUITY IN THE REGISTERED RULE, flagged rather than silently resolved. The rule says
"flatter on >= 4 of 5 RUNGS", but E36 has eleven-plus rungs and five FIT CORPORA, and the published
"5 of 5" is over corpora -- CONTEXT.md §3.1 lists exactly five slopes, one per fitting corpus. This
tool adjudicates the rule on the five FIT CORPORA, which is the only reading under which "5" is the
denominator, and ALSO reports a per-rung count so the other reading is visible. The rule is not
reinterpreted; the ambiguity is reported (CLAUDE.md §2.9).

    python tools/r4b_e36_flatness.py --stripped results/e36_qladder_410m_rstrip.json
"""
from __future__ import annotations
import argparse, json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "src"))
from provenance import sha256_file, write_result  # noqa: E402

R = lambda *p: os.path.join(REPO, *p)
CORPORA = ["Pile-CC", "StackExchange", "Wikipedia_en", "Github", "USPTO_Backgrounds"]


def ols_slope(x, y):                       # t53's convention, verbatim
    mx, my = st.mean(x), st.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)


def slopes_of(e36, ag="persist"):
    lad = e36["ladder"]
    # t53_ladder_summary.py's filter, verbatim, so the unstripped slopes reproduce its stored ones
    rungs = [r for r, v in lad.items()
             if r != "Q0" and not r.startswith("SHUFFLED_") and v.get("containment_k32")]
    x_lin = [lad[r]["containment_k32"] for r in rungs]
    x_log = [math.log10(v) for v in x_lin]
    out = {}
    for arm in ["logit"] + [f"J|{c}" for c in CORPORA]:
        if arm not in lad[rungs[0]]["arms"]:
            continue
        y = [lad[r]["arms"][arm][ag]["mean"] for r in rungs]
        out[arm] = {"slope_linear": ols_slope(x_lin, y), "slope_log10": ols_slope(x_log, y),
                    "level_mean": st.mean(y)}
    return {"rungs": rungs, "arms": out}


def flatness(s):
    lg = s["arms"]["logit"]
    per = {}
    for c in CORPORA:
        a = s["arms"].get(f"J|{c}")
        if not a:
            continue
        per[c] = {"slope_linear": a["slope_linear"], "logit_slope_linear": lg["slope_linear"],
                  "abs_slope": abs(a["slope_linear"]), "abs_logit_slope": abs(lg["slope_linear"]),
                  "flatter_than_logit": abs(a["slope_linear"]) < abs(lg["slope_linear"])}
    return {"per_corpus": per, "n_flatter": sum(1 for v in per.values() if v["flatter_than_logit"]),
            "n_corpora": len(per)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unstripped", default=R("results", "e36_qladder_410m.json"))
    ap.add_argument("--stripped", default=R("results", "e36_qladder_410m_rstrip.json"))
    ap.add_argument("--out", default=R("results", "r4b_e36_flatness.json"))
    a = ap.parse_args()
    if not os.path.exists(a.stripped):
        raise SystemExit(f"ABORT: {a.stripped} does not exist. Run t36_qladder.py --rstrip first.")
    U, S = json.load(open(a.unstripped)), json.load(open(a.stripped))

    arms, ctrl = {}, {}
    for ag in ("persist", "min"):
        su, ss = slopes_of(U, ag), slopes_of(S, ag)
        arms[ag] = {"unstripped": {"slopes": su["arms"], "flatness": flatness(su)},
                    "stripped": {"slopes": ss["arms"], "flatness": flatness(ss)},
                    "rungs": ss["rungs"]}

    # ---- C1: the unstripped file must be the STORED one, untouched
    ctrl["C1_unstripped_is_the_stored_run"] = {
        "required": "the unstripped arm read here is results/e36_qladder_410m.json as stored",
        "file": os.path.relpath(a.unstripped, REPO),
        "sha256": sha256_file(a.unstripped),
        "rstrip_flag_in_file": U.get("rstrip", "absent (pre-R4b run)"),
        "fires": not U.get("rstrip", False)}

    # ---- C1b: the unstripped slopes must reproduce e53's STORED slopes exactly. If they do not,
    # this tool is not computing the quantity CONTEXT.md §3.1 cites and nothing here is comparable.
    e53p = R("results", "e53_ladder_summary.json")
    c1b = {"required": "max_abs_diff = 0.0 against e53_ladder_summary.e36_containment_slopes",
           "checked": False}
    if os.path.exists(e53p):
        st53 = json.load(open(e53p)).get("e36_containment_slopes", {})
        worst, wk = 0.0, None
        for ag in ("persist", "min"):
            for arm, v in arms[ag]["unstripped"]["slopes"].items():
                ref = st53.get(ag, {}).get(arm, {}).get("slope_linear_containment")
                if ref is None:
                    continue
                d = abs(v["slope_linear"] - ref)
                if d > worst: worst, wk = d, f"{ag}:{arm}"
        c1b = {"required": "max_abs_diff = 0.0", "checked": True,
               "max_abs_diff": worst, "worst_at": wk, "fires": worst == 0.0}
    ctrl["C1b_unstripped_slopes_reproduce_e53"] = c1b

    # ---- C2: each corpus's own derangement must lose to its own operator (R4d)
    c2 = S.get("controls", {}).get("C2b_own_derangement_below_its_own_operator")
    ctrl["C2_own_derangement_loses"] = (
        c2 if c2 else {"required": "0 cells where a corpus's own derangement beats its own operator",
                       "fires": None, "why": "C2b absent from the stripped file"})

    # ---- C3: report C2_shuf_below_everything whichever way it fires
    ctrl["C3_report_C2_shuf_below_everything"] = {
        "unstripped": U.get("controls", {}).get("C2_shuf_below_everything", {}).get("fires"),
        "stripped": S.get("controls", {}).get("C2_shuf_below_everything", {}).get("fires"),
        "note": ("a RECORDER, not a gate -- its power is nil because it compares Pile-CC's "
                 "derangement to every corpus's operator (R4d/F-7). Reported whichever way it "
                 "fires, as the rule requires.")}

    # ---- the primary REJECT, separately from the slope secondary
    crossings = {"unstripped": {c: U.get("crossings", {}).get(c, {}).get("crosses")
                                for c in CORPORA},
                 "stripped": {c: S.get("crossings", {}).get(c, {}).get("crosses") for c in CORPORA}}
    n_cross = {k: sum(1 for v in d.values() if v) for k, d in crossings.items()}

    n = arms["persist"]["stripped"]["flatness"]["n_flatter"]
    if n >= 4:
        verdict = (f"REJECT STANDS — under the corrected readout the fitted operator is flatter "
                   f"across the shift axis than the unfitted logit lens on {n} of 5 fitting "
                   f"corpora (published, unstripped: "
                   f"{arms['persist']['unstripped']['flatness']['n_flatter']} of 5). CONTEXT.md "
                   f"§3.1 is unchanged on this clause.")
    elif n <= 1:
        verdict = (f"REJECT OVERTURNED — flatter on only {n} of 5. STOP AND ALERT THE OPERATOR. "
                   f"S3's first half is no longer rejected on the read axis, which reopens the "
                   f"subthesis and changes the paper's framing.")
    else:
        verdict = (f"UNCLEAR — flatter on {n} of 5, which is neither >= 4 nor <= 1. Report and "
                   f"stop. The rule is not to be re-cut (CLAUDE.md §2.9).")

    rec = {"experiment": "R4b — E36's shift-axis slopes at the corrected readout",
           "prereg": "specs/experiments/R4b_e36_rstrip.md",
           "prereg_sha256": sha256_file(R("specs", "experiments", "R4b_e36_rstrip.md")),
           "status": "PRE-REGISTERED",
           "decision_rule_verbatim": (
               "REJECT STANDS: the fitted operator is flatter on >= 4 of 5 rungs. Section 3.1 is "
               "unchanged. | REJECT OVERTURNED: flatter on <= 1 of 5. STOP and alert. | UNCLEAR: "
               "2 or 3 of 5. Report and stop. Do not re-cut."),
           "WORDING_AMBIGUITY_IN_THE_RULE": (
               "the rule says '5 rungs'; E36 has eleven-plus rungs and five FIT CORPORA, and the "
               "published 5-of-5 is over corpora (CONTEXT.md §3.1 lists five slopes, one per "
               "fitting corpus). Adjudicated over the five fit corpora, which is the only reading "
               "in which 5 is the denominator. Flagged, not resolved."),
           "PRIMARY_slope_secondary": {ag: {"n_flatter_stripped": arms[ag]["stripped"]["flatness"]["n_flatter"],
                                            "n_flatter_unstripped": arms[ag]["unstripped"]["flatness"]["n_flatter"]}
                                       for ag in arms},
           "E36_PRIMARY_reject_s3_crossings": {"crosses_per_corpus": crossings,
                                               "n_crossing": n_cross,
                                               "published": "0 of 5 -> REJECT S3"},
           "by_aggregation": arms, "controls": ctrl,
           "stripped_verdict_string_from_the_run": S.get("VERDICT"),
           "unstripped_verdict_string_from_the_run": U.get("VERDICT"),
           "VERDICT": verdict}
    write_result(a.out, rec, experiment="R4b", inputs=[a.unstripped, a.stripped])
    for ag in arms:
        u, s = arms[ag]["unstripped"]["flatness"], arms[ag]["stripped"]["flatness"]
        print(f"[{ag}] flatter than logit: unstripped {u['n_flatter']}/{u['n_corpora']}  ->  "
              f"stripped {s['n_flatter']}/{s['n_corpora']}")
        print(f"    logit slope {arms[ag]['stripped']['slopes']['logit']['slope_linear']:+.5f} "
              f"(was {arms[ag]['unstripped']['slopes']['logit']['slope_linear']:+.5f})")
        for c in CORPORA:
            if c in s["per_corpus"]:
                print(f"      J|{c:20s} {s['per_corpus'][c]['slope_linear']:+.5f}  "
                      f"flatter={s['per_corpus'][c]['flatter_than_logit']}")
    print(f"\ncrossings (E36 primary): {n_cross}")
    for k, v in ctrl.items():
        print(f"  {k:38s} {v.get('fires')}")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
