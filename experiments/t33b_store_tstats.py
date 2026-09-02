#!/usr/bin/env python3
"""t33b_store_tstats.py — store the E33 per-corpus t statistics that were only ever in prose.

WHY THIS EXISTS
  Discipline #2: "Never report an unstored number." The figure `Github 0.02808, t = -1.65` is
  cited in SPINE.tex (§ the E33 table), HANDOFF.md §1, CLAUDE.md §4 and PREREG_E48_CROSSOVER.md's
  Amendment 1 — where it is load-bearing, because clause (d) requires Github to sit AT the
  crossover rather than below it. It appears in NO results JSON, and no script in the repo
  computed it. Its two inputs ARE stored, so the number is recoverable, but until this file ran
  it was a prose-only number underwriting a pre-registered clause.

  This does not re-derive or re-measure anything. It recomputes the statistic from
  `e33_logit_baseline_410m_v2.json` and writes it down, with the formula, so the citation has a
  file.

THE STATISTIC, stated explicitly because the prose never did
  One-sample t of the THREE seed-block asymptote means against the logit-lens constant:

      t = (Y_inf(P) - L) / (seed_sd(P) / sqrt(3)),    df = 2

  where Y_inf(P) is E28's per-corpus asymptote (the mean over N>=75 of the 5-admitted-set mean
  persist read, averaged over 3 seed blocks), seed_sd is the sample SD (n-1) across those 3
  blocks, and L is the logit lens's 5-admitted-set persist mean. The logit lens is a CONSTANT
  here, not a second sample: it is unfitted, so it has no seed axis. That is what makes this a
  one-sample test rather than a two-sample one.

DECLARED LIMITATION, and it is not small
  df = 2. The two-sided 5% critical value is 4.303, so |t| < 4.303 is the "indistinguishable"
  region and Github's -1.65 sits inside it. But a t on three points estimates the SD from three
  points; these intervals are wide and unstable, and the three seed-block means themselves are
  NOT persisted (t33_logit_baseline.py:165-168 discards them and keeps only mean and stdev), so
  they cannot be re-derived from this file alone — only from results/ladder410/*.json.
  E48 supersedes this with a hierarchical bootstrap over eval sets and seed blocks. This file
  exists to give the standing citation a home, not to become the preferred instrument.

    python t33b_store_tstats.py
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from provenance import write_result  # noqa: E402
SRC = os.path.join(HERE, "..", "results", "e33_logit_baseline_410m_v2.json")
OUT = os.path.join(HERE, "..", "results", "e33b_tstats_410m.json")
T_CRIT_DF2 = 4.302652729911275          # two-sided 5%, df = 2


def main() -> int:
    d = json.load(open(SRC))
    L = d["persist_admitted_mean"]["logit_I"]
    rows = {}
    for c, v in d["e28_asymptotes"].items():
        se = v["seed_sd"] / math.sqrt(3)
        t = (v["mean"] - L) / se
        rows[c] = {"Y_inf_mean": v["mean"], "seed_sd": v["seed_sd"], "n_seed_blocks": 3,
                   "standard_error": se, "t": t, "df": 2,
                   "abs_t_exceeds_crit_5pct": abs(t) > T_CRIT_DF2,
                   "delta_vs_logit": v["mean"] - L,
                   "pct_vs_logit": 100.0 * (v["mean"] - L) / L}
    rec = {
        "experiment": "E33b — the E33 per-corpus t statistics, stored (they were prose-only)",
        "why": ("discipline #2: 'Github 0.02808, t = -1.65' is cited in SPINE.tex, HANDOFF.md, "
                "CLAUDE.md and PREREG_E48_CROSSOVER.md Amendment 1 — where clause (d) depends on "
                "it — and lived in no results file. No script computed it. This stores it."),
        "source": os.path.basename(SRC),
        "formula": "t = (Y_inf(P) - L) / (seed_sd(P) / sqrt(3)), df = 2, one-sample vs the "
                   "logit constant L (unfitted, so it has no seed axis)",
        "logit_constant_L": L,
        "t_crit_two_sided_5pct_df2": T_CRIT_DF2,
        "recomputes_not_remeasures": True,
        "limitation": ("df = 2. The three seed-block means are NOT persisted by "
                       "t33_logit_baseline.py:165-168 (only their mean and stdev), so they "
                       "cannot be recovered from the source file — only from "
                       "results/ladder410/*.json. E48's hierarchical bootstrap supersedes this "
                       "as the instrument; this file exists to give the standing citation a "
                       "home."),
        "by_corpus": rows,
    }
    write_result(OUT, rec, experiment="E33b", inputs=[SRC])
    print(f"{'corpus':24s} {'Y_inf':>9} {'seed_sd':>9} {'delta':>10} {'%':>8} {'t':>9}  sig(df=2)")
    for c, r in sorted(rows.items(), key=lambda kv: -kv[1]["t"]):
        print(f"{c:24s} {r['Y_inf_mean']:9.5f} {r['seed_sd']:9.6f} {r['delta_vs_logit']:+10.5f} "
              f"{r['pct_vs_logit']:+8.1f} {r['t']:+9.2f}  "
              f"{'yes' if r['abs_t_exceeds_crit_5pct'] else 'NO — indistinguishable'}")
    print(f"logit constant L = {L:.5f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
