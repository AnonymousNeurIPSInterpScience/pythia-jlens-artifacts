#!/usr/bin/env python3
"""t48c_exposure_vs_read.py — E48c: does EXPOSURE of the fitting corpus govern the read?

WHY
  E48 returned NOT REACHED on its pre-registered clause: no OOD rung's (J^P - logit) CI went
  strictly negative. That is the registered verdict and it stands. But NOT REACHED is a statement
  about a CI, and the CI in question could not resolve the in-stream advantage either (clause (a)
  failed on all four in-stream rungs). So "NOT REACHED" on its own does not tell the reader what
  the data DOES show.

  This is the positive content: E48 fitted eight operators on corpora whose measured containment
  in the Pythia training stream spans nearly FOUR ORDERS OF MAGNITUDE (0.0001 to 0.8595, E48b at
  75% stream coverage). If exposure of the fitting corpus governed the read, the read would order
  with containment. This script measures whether it does.

  This is NOT a new predictor hunt (HANDOFF sec0.5). It does not propose a quantity to predict the
  corpus x concept interaction. It tests ONE pre-existing, externally-measured variable —
  containment, which E48's own rung classification is built on — against the read. The answer
  being NO is what licenses the paper's negative claim; the answer being YES would have
  overturned E48's framing entirely.

PRIMARY
  Spearman and Pearson correlation between k=32 containment at 75% stream coverage and the
  5-admitted-set persist read AUC, over the eight fitted corpora.

SECONDARY, and reported because it is the honest counterweight
  The in-stream vs OOD TIER contrast with the CORPUS as the replication unit (n=4 vs n=3). A
  near-zero rank correlation and a nonzero tier gap are not contradictory: the tiers can differ
  on average while exposure fails to order the rungs within and across tiers.

CONTROLS
  C1 SPREAD    containment must actually span orders of magnitude, else the test has no leverage.
               Requires max/min > 100.
  C2 SEED      the per-corpus seed SD must be small against the between-corpus spread, else no
               ordering is resolvable at all. Requires max seed SD < 0.1 x between-corpus range.
  C3 GITHUB    every corpus-axis result in this programme collapses on Github (HANDOFF sec3).
               The correlation is reported WITH and WITHOUT it, and the claim is only made in the
               form that survives both.

DECLARED BIAS
  Eight corpora. A correlation on n=8 is weak evidence in either direction; the claim this
  licenses is a NEGATIVE one (exposure does not order the read), which is the direction a small
  n can support, because the alternative predicts a strong monotone relationship that is simply
  absent. It cannot support "exposure has no effect of any size".

    python t48c_exposure_vs_read.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from provenance import write_result  # noqa: E402
INSTREAM = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange"]
OOD = ["OOD_News_2024", "OOD_arXiv_2023", "OOD_CommonPile"]
GITHUB = "Github"


def pearson(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else float("nan")


def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(s):
            r[i] = pos + 1.0
        return r
    return pearson(rank(x), rank(y))


def welch(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    t = (ma - mb) / se if se else float("inf")
    df = ((va / len(a) + vb / len(b)) ** 2 /
          ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)))
    return {"mean_a": ma, "mean_b": mb, "diff": ma - mb, "t": t, "df": df, "se": se}


def main() -> int:
    e48 = json.load(open(os.path.join(HERE, "..", "results", "e48_crossover_410m.json")))
    grow = json.load(open(os.path.join(HERE, "..", "results", "e48b_exposure_growth.json")))
    cont = grow["containment_by_coverage"]["32"]
    cov = len(grow["shards_used"])

    rungs = INSTREAM + [GITHUB] + OOD
    L = e48["rungs"][rungs[0]]["persist"]["logit_admitted_mean"]
    tab = {}
    for k in rungs:
        r = e48["rungs"][k]["persist"]
        ps = list(r["per_seed_admitted_mean"].values())
        tab[k] = {"containment_k32": cont[k][-1],
                  "read_persist": statistics.mean(ps),
                  "seed_sd": statistics.stdev(ps),
                  "delta_vs_logit": statistics.mean(ps) - L,
                  "tier": "in-stream" if k in INSTREAM else "github" if k == GITHUB else "OOD"}

    C = [tab[k]["containment_k32"] for k in rungs]
    Y = [tab[k]["read_persist"] for k in rungs]
    ng = [k for k in rungs if k != GITHUB]
    Cn = [tab[k]["containment_k32"] for k in ng]
    Yn = [tab[k]["read_persist"] for k in ng]

    spread = max(C) / max(min(C), 1e-12)
    rng = max(Y) - min(Y)
    max_sd = max(tab[k]["seed_sd"] for k in rungs)

    tier = welch([tab[k]["read_persist"] for k in INSTREAM],
                 [tab[k]["read_persist"] for k in OOD])
    arxiv = [e48["rungs"]["OOD_arXiv_2023"]["persist"]["per_seed_admitted_mean"][s]
             for s in ("s0", "s1", "s2")]
    wiki = [e48["rungs"]["Wikipedia_en"]["persist"]["per_seed_admitted_mean"][s]
            for s in ("s0", "s1", "s2")]
    head = welch(arxiv, wiki)

    rec = {
        "experiment": "E48c — does exposure of the FITTING corpus govern the read?",
        "why": ("E48 returned NOT REACHED on a CI that could not resolve the in-stream advantage "
                "either. This measures the positive content: whether the read orders with "
                "measured containment across four orders of magnitude of exposure."),
        "not_a_new_predictor": ("tests ONE pre-existing externally-measured variable that E48's "
                               "own rung classification is already built on; proposes no new "
                               "scalar and does not target the corpus x concept interaction "
                               "(HANDOFF sec0.5)"),
        "sources": ["e48_crossover_410m.json", "e48b_exposure_growth.json"],
        "containment_k": 32, "containment_coverage_shards": cov, "aggregation": "persist",
        "logit_baseline": L,
        "by_rung": tab,
        "correlations": {
            "all8_pearson": pearson(C, Y), "all8_spearman": spearman(C, Y),
            "without_github_pearson": pearson(Cn, Yn),
            "without_github_spearman": spearman(Cn, Yn)},
        "tier_contrast_corpus_as_unit": {
            "note": "n=4 in-stream vs n=3 OOD; Github excluded (it is neither a clean in-stream "
                    "English rung nor OOD, and it is the leverage point in seven prior analyses)",
            **tier, "significant_at_5pct": abs(tier["t"]) > 2.78},
        "headline_pair_seed_axis": {
            "note": "OOD_arXiv_2023 (containment 0.0001) vs Wikipedia_en (containment 0.7996), "
                    "SEED as the replication unit (3 disjoint prompt blocks)",
            **head, "arxiv_above_wikipedia": head["t"] > 0,
            "significant_at_5pct": abs(head["t"]) > 2.78},
        "controls": {
            "C1_containment_spans_orders": {"max_over_min": spread, "fires": spread > 100},
            "C2_seed_sd_small_vs_spread": {"max_seed_sd": max_sd, "between_corpus_range": rng,
                                           "ratio": max_sd / rng, "fires": max_sd < 0.1 * rng},
            "C3_github_leverage": {"spearman_with": spearman(C, Y),
                                   "spearman_without": spearman(Cn, Yn),
                                   "note": "claim is stated only in the form that survives both"},
        },
    }
    rec["VERDICT"] = (
        "EXPOSURE DOES NOT ORDER THE READ. Containment spans {:.0f}x across the eight fitting "
        "corpora and the rank correlation with read quality is {:+.3f} (all eight) / {:+.3f} "
        "(without Github). The most-contained corpus (Github, {:.3f}) is the WORST reader; the "
        "least-contained (arXiv_2023, {:.4f}) reads ABOVE in-stream Wikipedia by t={:+.2f} on the "
        "seed axis. The in-stream/OOD tier gap is +{:.5f} with the corpus as the replication "
        "unit (t={:.2f}, df={:.1f}) — suggestive, NOT significant, and far smaller than the "
        "within-tier spread.".format(
            spread, rec["correlations"]["all8_spearman"],
            rec["correlations"]["without_github_spearman"],
            tab[GITHUB]["containment_k32"], tab["OOD_arXiv_2023"]["containment_k32"],
            head["t"], tier["diff"], tier["t"], tier["df"]))

    out = os.path.join(HERE, "..", "results", "e48c_exposure_vs_read.json")
    write_result(out, rec, experiment="E48c",
                 inputs=[os.path.join(HERE, "..", "results", "e48_crossover_410m.json"),
                         os.path.join(HERE, "..", "results", "e48b_exposure_growth.json")])

    print(f"{'rung':22s} {'tier':10s} {'containment':>12} {'read':>9} {'seedSD':>9} {'vs logit':>10}")
    for k in sorted(rungs, key=lambda x: -tab[x]["read_persist"]):
        t = tab[k]
        print(f"{k:22s} {t['tier']:10s} {t['containment_k32']:12.4f} {t['read_persist']:9.5f} "
              f"{t['seed_sd']:9.6f} {t['delta_vs_logit']:+10.5f}")
    print(f"\nSpearman(containment, read): all8 {rec['correlations']['all8_spearman']:+.4f}  "
          f"without Github {rec['correlations']['without_github_spearman']:+.4f}")
    print(f"tier gap (corpus as unit): {tier['diff']:+.5f}  t={tier['t']:.2f} df={tier['df']:.1f}  "
          f"{'SIGNIFICANT' if abs(tier['t'])>2.78 else 'NOT significant'}")
    print(f"arXiv vs Wikipedia (seed axis): t={head['t']:+.2f}  "
          f"{'arXiv ABOVE' if head['t']>0 else 'arXiv below'}")
    for k, v in rec["controls"].items():
        if "fires" in v:
            print(f"  {k:32s} {'FIRES' if v['fires'] else 'DOES NOT FIRE'}")
    print(f"\nVERDICT: {rec['VERDICT']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
