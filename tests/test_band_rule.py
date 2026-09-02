#!/usr/bin/env python3
"""test_band_rule.py — the band rule is one rule, and it is the one the anchor states.

WHY THIS EXISTS. Three implementations of "the band" coexisted in this tree and two of them were
wrong:

  experiments/trainval.py      --band-frac default "0.35,0.85"   implements no declared rule, and
                                                                 at 16 layers yields [5,13], the
                                                                 1B ladder's documented deviation
  experiments/t53_ladder_summary.py  round(0.38L)..round(0.92L)  disagrees at 12 layers ([5,9]
                                                                 against the [4,9] fitted)
  experiments/cv6_per_family_ladder.py  int(0.38L)..int(0.92L)   correct

The anchor states the workspace range as beginning "about a third of the way through (~L38)" and
ending "shortly before the output (~L92)" (section 4.1). Normalised, that is 0.38 to 0.92, floored,
intersected with layers strictly below the penultimate target layer.

WHAT THIS DOES NOT COVER. It asserts that the rule is implemented consistently and reproduces the
bands on disk. It says nothing about whether 0.38/0.92 is the right band, whether the anchor's
range transfers to Pythia, or whether band width is confounded with model size. It is not.
`docs/experiments/preregs/CV6_per_family_ladder.md` and the paper's Controls appendix carry that.

    .venv/bin/python tests/test_band_rule.py
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "experiments"))

# The bands this programme actually fitted, read off the stored artifacts, not off a docstring.
# 70M and 160M: results/e49_derangement_stability.json -> cells.<scale>|wikitext.band
# 410M: every e48/e52/e57 arm.  1B: the DECLARED band, results/ladder1b_b613/.
# 1.4B, 2.8B: results/cv6_per_family_ladder.json -> by_model.<m>.band
USED = {
    6:  (2, 3),      # 70M   -- 2 layers, where persist degenerates to min
    12: (4, 9),      # 160M  -- 6 layers
    24: (9, 21),     # 410M and 1.4B -- 13 layers
    16: (6, 13),     # 1B    -- 8 layers, the DECLARED band (the ladder ran [5,13]; see trainval)
    32: (12, 29),    # 2.8B  -- 18 layers
}

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}   [{detail}]")
    if not ok:
        FAILURES.append(name)


def band(lo: float, hi: float, n_layers: int) -> list[int]:
    target_eff = n_layers - 2
    return [l for l in range(int(lo * n_layers), int(hi * n_layers) + 1) if l < target_eff]


def main() -> int:
    from cv6_per_family_ladder import BAND_HI, BAND_LO, band_for  # noqa: E402

    # B1 the canonical constants are the anchor's stated range
    check("B1_constants_are_L38_L92", (BAND_LO, BAND_HI) == (0.38, 0.92),
          f"BAND_LO={BAND_LO}, BAND_HI={BAND_HI}")

    # B2 the canonical implementation reproduces every band on disk
    bad = {n: (tuple((band_for(n)[0], band_for(n)[-1])), want)
           for n, want in USED.items() if (band_for(n)[0], band_for(n)[-1]) != want}
    check("B2_band_for_reproduces_every_fitted_band", not bad,
          "all 6 rungs" if not bad else f"mismatches {bad}")

    # B3 trainval's --band-frac DEFAULT agrees with band_for at every rung. This is the one that
    #    drifted, and the drift produced the 1B [5,13] deviation.
    tv = open(os.path.join(REPO, "experiments", "trainval.py")).read()
    m = re.search(r'"--band-frac",\s*default="([\d.]+),([\d.]+)"', tv)
    check("B3_trainval_default_parses", m is not None, m.group(0) if m else "PATTERN NOT FOUND")
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        mism = {n: (tuple((band(lo, hi, n)[0], band(lo, hi, n)[-1])), w)
                for n, w in USED.items() if (band(lo, hi, n)[0], band(lo, hi, n)[-1]) != w}
        check("B4_trainval_default_matches_band_for", not mism,
              f"default {lo},{hi}" + ("" if not mism else f"; mismatches {mism}"))

        # B5 a control that CAN fail: the OLD default must break B4, or B4 proves nothing.
        old = {n: (band(0.35, 0.85, n)[0], band(0.35, 0.85, n)[-1]) for n in USED}
        broke = {n: (old[n], USED[n]) for n in USED if old[n] != USED[n]}
        check("B5_old_default_would_fail_B4", bool(broke),
              f"0.35/0.85 mismatches {sorted(broke)} layers, e.g. 16 -> {old[16]} against {USED[16]}")

    # B6 t53's reporting rule agrees with band_for. It used round(), which differs at 12 layers.
    t53 = open(os.path.join(REPO, "experiments", "t53_ladder_summary.py")).read()
    check("B6_t53_uses_floor_not_round",
          "int(0.38 * n_layers), int(0.92 * n_layers)" in t53,
          "floor" if "int(0.38 * n_layers)" in t53 else "still round(); disagrees at 160M")

    # B7 every source layer is strictly below the penultimate target, which jlens requires
    check("B7_all_bands_below_target", all(band_for(n)[-1] < n - 2 for n in USED),
          ", ".join(f"{n}:{band_for(n)[-1]}<{n-2}" for n in sorted(USED)))

    print(f"\n=== {7 - len(FAILURES)}/7 PASSED ===")
    if FAILURES:
        print("FAILED: " + ", ".join(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
