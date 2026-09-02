#!/usr/bin/env python3
"""da1b_clustered_t.py — the corpus-clustered paired t on DA1's arms.

NOT A NEW EXPERIMENT and NOT A NEW BAR. This applies the statistic the paper already uses for the
random-derangement gap (results/paper_clustered_derangement_t.json, paper Table 13) to DA1's
cyclic-shift and non-union arms, so the new comparisons sit on exactly the same inferential
footing as the one they adjudicate. Recomputation from results/da1_derangement_adjudication_410m.json.

THE STATISTIC, transcribed
  One-sample paired t over the 8 per-corpus values of (real - null), df = 7, two-sided p from
  Student's t. The corpus is the replication unit because five derangements of one operator share
  an operator and an activation cache, so a draw is not an independent observation. Same reasoning
  applies to twelve cyclic shifts of one operator.

WHAT THIS DOES NOT DO
  It sets no threshold and adjudicates nothing on a p-value. DA1's pre-registration deliberately
  created no pass/fail bar. The t is reported because the claim being adjudicated was reported with
  one, and a reader comparing them is entitled to the same statistic on both sides.

  A control on this tool: the `shuf` rows recomputed here MUST reproduce the stored
  paper_clustered_derangement_t.json values, or DA1 is not scoring the same gaps the paper reports.

    .venv/bin/python tools/da1b_clustered_t.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from provenance import write_result  # noqa: E402

DA1 = "results/da1_derangement_adjudication_410m.json"
REF = "results/paper_clustered_derangement_t.json"
AGGS = ["min", "persist", "best1L", "mean"]


def _t_cdf(t, df):
    """Two-sided Student-t tail via the regularised incomplete beta, stdlib only."""
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    r = front * (f - 1.0)
    return r if x < (a + 1) / (a + b + 2) else 1.0 - _betainc(b, a, 1 - x)


def paired_t(gaps):
    n = len(gaps)
    m = statistics.mean(gaps)
    sd = statistics.stdev(gaps)
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    p = _t_cdf(t, n - 1) if math.isfinite(t) else 0.0
    return {"n_corpora": n, "mean_gap": m, "sd_gap": sd, "t": t, "df": n - 1,
            "p_two_sided": p, "clears_05": p < 0.05}


def main() -> int:
    d = json.load(open(os.path.join(ROOT, DA1)))
    C = d["corpora"]
    out = {}
    for null in ("cyclic", "shuf"):
        key = f"real_minus_{null}_mean"
        out[null] = {}
        for ag in AGGS:
            gaps = [d["by_corpus"][c][key][ag] for c in C]
            r = paired_t(gaps)
            r["per_corpus_gap"] = {c: d["by_corpus"][c][key][ag] for c in C}
            r["n_corpora_real_higher"] = sum(1 for g in gaps if g > 0)
            out[null][ag] = r
    if "bandconst" in d["by_corpus"][C[0]]:
        out["bandconst_EXPLORATORY"] = {}
        for ag in AGGS:
            gaps = [d["by_corpus"][c]["real_minus_bandconst"][ag] for c in C]
            r = paired_t(gaps)
            r["per_corpus_gap"] = {c: d["by_corpus"][c]["real_minus_bandconst"][ag] for c in C}
            r["n_corpora_real_higher"] = sum(1 for g in gaps if g > 0)
            r["caveat"] = ("J_bar is not a permutation of the band's Jacobians; it is a different "
                           "object with a different norm and spectrum. Exploratory only.")
            out["bandconst_EXPLORATORY"][ag] = r

    # CONTROL — the shuf rows must reproduce the paper's stored clustered t
    ctl = {"requirement": "DA1's shuf gaps reproduce paper_clustered_derangement_t.json (corrected "
                          "readout) to < 1e-9 on mean_gap and < 1e-6 on t",
           "fails_if": "any larger difference, which would mean DA1 is not scoring the same gaps",
           "per_agg": {}}
    worst_m = worst_t = 0.0
    ref = json.load(open(os.path.join(ROOT, REF)))["by_readout"]["corrected"]
    for ag in ("min", "persist"):
        mine, theirs = out["shuf"][ag], ref[ag]
        dm = abs(mine["mean_gap"] - theirs["mean_gap"])
        dt = abs(mine["t"] - theirs["t"])
        ctl["per_agg"][ag] = {"da1_mean_gap": mine["mean_gap"], "stored_mean_gap": theirs["mean_gap"],
                              "abs_diff_mean_gap": dm,
                              "da1_t": mine["t"], "stored_t": theirs["t"], "abs_diff_t": dt,
                              "da1_p": mine["p_two_sided"], "stored_p": theirs["p_two_sided"]}
        worst_m, worst_t = max(worst_m, dm), max(worst_t, dt)
    ctl["max_abs_diff_mean_gap"] = worst_m
    ctl["max_abs_diff_t"] = worst_t
    ctl["fires"] = worst_m < 1e-9 and worst_t < 1e-6

    rec = {
        "experiment": "DA1b — corpus-clustered paired t on DA1's arms",
        "status": "RECOMPUTATION — no model, no scoring, no new bar",
        "recomputes_not_remeasures": True,
        "statistic": ("one-sample paired t over the 8 per-corpus values of (real - null), df = 7, "
                      "two-sided p from Student's t. Transcribed from "
                      "results/paper_clustered_derangement_t.json."),
        "sets_no_threshold": ("DA1's pre-registration deliberately created no pass/fail bar. The t "
                              "is reported because the claim being adjudicated was reported with "
                              "one."),
        "source": DA1,
        "corpora": C,
        "by_null": out,
        "control_C1_reproduces_paper_clustered_t": ctl,
    }
    write_result(os.path.join(ROOT, "results", "da1b_clustered_t.json"), rec,
                 experiment="DA1b", script=__file__,
                 inputs=[os.path.join(ROOT, DA1), os.path.join(ROOT, REF)])

    print("\n" + "=" * 86)
    print("corpus-clustered paired t on (real - null), df=7, 8 corpora")
    print("=" * 86)
    for null in [k for k in out]:
        print(f"\n  null = {null}")
        print(f"    {'agg':10} {'mean gap':>11} {'t(7)':>9} {'p 2-sided':>12} {'real higher':>13}")
        for ag in AGGS:
            r = out[null][ag]
            print(f"    {ag:10} {r['mean_gap']:+11.5f} {r['t']:9.2f} {r['p_two_sided']:12.2e} "
                  f"{r['n_corpora_real_higher']:>9}/8")
    print(f"\n  control C1 (shuf rows reproduce the paper's stored t): "
          f"{'FIRES' if ctl['fires'] else 'DOES NOT FIRE'}  "
          f"max|d mean_gap|={ctl['max_abs_diff_mean_gap']:.2e}  max|dt|={ctl['max_abs_diff_t']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
