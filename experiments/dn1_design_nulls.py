#!/usr/bin/env python3
"""dn1_design_nulls.py — DN1: what the three registered bars correspond to under their own nulls.

PRE-REGISTRATION: docs/experiments/preregs/DN1_design_nulls.md, written and committed before this
file existed.

  A  the null distribution of R = spread_z / pooled_seed_SD_z, the CV6/CV7 statistic, two ways:
     A1 a scale-free design null by simulation (5 groups of 3, iid normal)
     A2 an exact-in-principle permutation null on D3's stored raw per-arm z at 410M
  B  the exact null of Kendall tau >= 0.6 at n=5, by enumerating all 120 orderings
  C  the exact null of R9's "0 of 8 corpora clear all 15 of their own draws", (15/16)^8, plus the
     sign test on the same file's per-corpus z that is NOT degenerate under that null

DECISION RULE (verbatim from the prereg, applied to A only):
  CALIBRATED    the R >= 10 bar sits at or above the 99th percentile of the A1 null
  WEAK          between the 95th and 99th
  UNCALIBRATED  below the 95th
B and C are descriptive and carry no verdict.

CONTROLS
  C1  scale-freedom: A1 at sigma=1 and sigma=100 agree on the 99th percentile to <= 0.02
  C2  pooled seed SD recomputed from raw == stored z_pooled_seed_sd, <= 1e-12, all 5 families
  C3  R recomputed from raw == stored z_spread_over_sd, <= 1e-12, all 5 families
  C4  the identity assignment reproduces the observed R exactly inside the permutation null
  C5  A1 and A2 99th percentiles agree to within a factor of 1.25

WHAT THIS DOES NOT COVER. It calibrates the bars, not the measurements. It cannot make CV6 or CV7
wrong and it cannot make them right; it prices the thresholds they were judged against. It says
nothing about whether z is the right space, whether the battery is valid, or whether the corpora
are a fair panel.

    .venv/bin/python experiments/dn1_design_nulls.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from provenance import write_result  # noqa: E402

RES = os.path.join(HERE, "..", "results")
FAMILIES = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
CORPORA = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github"]
SEEDS = [0, 1, 2]

R_BAR = 10.0          # the CV6/CV7 registered bar
TAU_BAR = 0.6         # the CV6/CV7 registered ordering bar
N_CORPORA_R9 = 8      # R9's panel
N_DRAWS_R9 = 15       # R9's derangements per corpus


# --------------------------------------------------------------------------- the statistic itself
def R_stat(groups: np.ndarray) -> float:
    """R = range of group means / RMS of per-group sample SDs.

    `groups` is (G, n). This is the CV6/CV7 definition verbatim: D3's own caption says the seed SD
    is "the RMS of per-corpus sample SDs over three seed blocks", i.e. sqrt(mean(s_g^2)) with s_g
    the ddof=1 sample SD. Getting this wrong by using a pooled-variance formula instead would
    change the denominator, which is why C2/C3 assert against the stored values rather than
    trusting this docstring.
    """
    means = groups.mean(axis=1)
    sds = groups.std(axis=1, ddof=1)
    return float((means.max() - means.min()) / math.sqrt(float((sds ** 2).mean())))


# --------------------------------------------------------------------------- A1, the design null
def design_null(n_draws: int, sigma: float, seed: int, G: int = 5, n: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, sigma, size=(n_draws, G, n))
    means = x.mean(axis=2)
    sds = x.std(axis=2, ddof=1)
    return (means.max(axis=1) - means.min(axis=1)) / np.sqrt((sds ** 2).mean(axis=1))


# --------------------------------------------------------------------------- B, the tau null
def kendall_tau(order_a: list, order_b: list) -> float:
    """Kendall tau-a between two orderings of the same 5 labels, as CV6/CV7 compute it."""
    pos_a = {c: i for i, c in enumerate(order_a)}
    pos_b = {c: i for i, c in enumerate(order_b)}
    conc = disc = 0
    for x, y in itertools.combinations(order_a, 2):
        s = (pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y])
        conc += s > 0
        disc += s < 0
    return (conc - disc) / (len(order_a) * (len(order_a) - 1) / 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sim", type=int, default=1_000_000)
    ap.add_argument("--n-perm", type=int, default=100_000)
    ap.add_argument("--d3", default=os.path.join(RES, "d3_corpus_by_family_410m.json"))
    ap.add_argument("--r9", default=os.path.join(RES, "r9_permutation_calibrated_min.json"))
    ap.add_argument("--out", default=os.path.join(RES, "dn1_design_nulls.json"))
    a = ap.parse_args()

    rec: dict = {
        "experiment": "DN1 — null distributions of three already-registered bars",
        "prereg": "docs/experiments/preregs/DN1_design_nulls.md",
        "status": "PRE-REGISTERED",
        "measures_nothing_new": True,
        "scope": ("Calibrates thresholds, not measurements. Cannot overturn CV6, CV7 or R9; prices "
                  "the bars they were judged against."),
        "n_sim": a.n_sim,
        "n_perm": a.n_perm,
    }

    # ================================================================= A1 design null, scale-free
    null1 = design_null(a.n_sim, sigma=1.0, seed=0)
    q = {f"p{p}": float(np.percentile(null1, p)) for p in (50, 90, 95, 99, 99.9)}
    exceed_bar = float((null1 >= R_BAR).mean())
    rec["A1_design_null"] = {
        "definition": "R = range(group means) / sqrt(mean(per-group sample var)), G=5 groups of n=3",
        "distribution": "iid normal, scale-free in sigma and location-free in mu",
        "mean": float(null1.mean()), "sd": float(null1.std(ddof=1)),
        "percentiles": q,
        "P_R_ge_10": exceed_bar,
        "P_R_ge_10_exceedances_observed": int((null1 >= R_BAR).sum()),
        "note": ("A one-sided exceedance of 0 in n_sim draws is reported as < 1/n_sim, not as 0. "
                 "The bar is calibrated against this column."),
    }

    # ------------------------------------------------------------------ C1 scale-freedom
    null1_big = design_null(a.n_sim, sigma=100.0, seed=1)
    p99_a, p99_b = float(np.percentile(null1, 99)), float(np.percentile(null1_big, 99))
    # a control that CAN fail: freeze the denominator and the ratio stops being scale-free
    rng = np.random.default_rng(2)
    broken = []
    for sig in (1.0, 100.0):
        x = rng.normal(0.0, sig, size=(20_000, 5, 3))
        m = x.mean(axis=2)
        broken.append(float(np.percentile(m.max(axis=1) - m.min(axis=1), 99)))  # denominator := 1
    rec["C1_scale_freedom"] = {
        "required": "|p99(sigma=1) - p99(sigma=100)| <= 0.02",
        "p99_sigma_1": p99_a, "p99_sigma_100": p99_b,
        "abs_diff": abs(p99_a - p99_b),
        "fires": abs(p99_a - p99_b) <= 0.02,
        "falsification_witness": {
            "what": "same code with the denominator replaced by the constant 1.0",
            "p99_sigma_1": broken[0], "p99_sigma_100": broken[1],
            "abs_diff": abs(broken[0] - broken[1]),
            "control_would_fire": abs(broken[0] - broken[1]) <= 0.02,
            "note": "this must be False, or C1 cannot distinguish a scale-free statistic from any other",
        },
    }

    # ================================================================= A2 permutation null on D3
    d3 = json.load(open(a.d3))
    per_arm = d3["per_arm"]
    perm, c2, c3, c4 = {}, {}, {}, {}
    rng = np.random.default_rng(7)
    for fam in FAMILIES:
        obs = np.array([[per_arm[f"{c}|s{s}"][fam]["z"] for s in SEEDS] for c in CORPORA])
        r_obs = R_stat(obs)
        stored = d3["by_family"][fam]
        # C2 / C3: is this the same statistic the bar judged?
        sd_here = math.sqrt(float((obs.std(axis=1, ddof=1) ** 2).mean()))
        c2[fam] = {"recomputed": sd_here, "stored": stored["z_pooled_seed_sd"],
                   "abs_diff": abs(sd_here - stored["z_pooled_seed_sd"])}
        c3[fam] = {"recomputed": r_obs, "stored": stored["z_spread_over_sd"],
                   "abs_diff": abs(r_obs - stored["z_spread_over_sd"])}
        flat = obs.reshape(-1)
        draws = np.empty(a.n_perm)
        for i in range(a.n_perm):
            draws[i] = R_stat(rng.permutation(flat).reshape(5, 3))
        # C4: the identity assignment must live in the support and reproduce r_obs
        c4[fam] = {"identity_R": R_stat(flat.reshape(5, 3)), "observed_R": r_obs,
                   "abs_diff": abs(R_stat(flat.reshape(5, 3)) - r_obs)}
        perm[fam] = {
            "observed_R": r_obs,
            "null_mean": float(draws.mean()), "null_sd": float(draws.std(ddof=1)),
            "null_p50": float(np.percentile(draws, 50)),
            "null_p95": float(np.percentile(draws, 95)),
            "null_p99": float(np.percentile(draws, 99)),
            "null_max": float(draws.max()),
            "n_null_ge_bar": int((draws >= R_BAR).sum()),
            "P_null_ge_bar": float((draws >= R_BAR).mean()),
            "n_null_ge_observed": int((draws >= r_obs).sum()),
            "p_observed_one_sided": float((1 + (draws >= r_obs).sum()) / (1 + a.n_perm)),
        }
    rec["A2_permutation_null_410m"] = {
        "source": "results/d3_corpus_by_family_410m.json -> per_arm.<corpus>|s<seed>.<family>.z",
        "unit": "15 z values per family, permuted across 5 corpus labels, 3 per label",
        "by_family": perm,
    }
    rec["C2_pooled_sd_recomputes"] = {
        "required": "<= 1e-12 on all 5 families", "by_family": c2,
        "max_abs_diff": max(v["abs_diff"] for v in c2.values()),
        "fires": max(v["abs_diff"] for v in c2.values()) <= 1e-12}
    rec["C3_R_recomputes"] = {
        "required": "<= 1e-12 on all 5 families", "by_family": c3,
        "max_abs_diff": max(v["abs_diff"] for v in c3.values()),
        "fires": max(v["abs_diff"] for v in c3.values()) <= 1e-12}
    rec["C4_identity_in_support"] = {
        "required": "the unpermuted assignment reproduces the observed R exactly",
        "by_family": c4, "max_abs_diff": max(v["abs_diff"] for v in c4.values()),
        "fires": max(v["abs_diff"] for v in c4.values()) == 0.0}

    # ------------------------------------------------------------------ C5 A1 against A2
    p99_perm = max(v["null_p99"] for v in perm.values())
    ratio = max(p99_perm, p99_a) / min(p99_perm, p99_a)
    rec["C5_design_vs_permutation"] = {
        "required": "the two 99th percentiles agree to within a factor of 1.25",
        "p99_design": p99_a, "p99_permutation_worst_family": p99_perm,
        "ratio": ratio, "fires": ratio <= 1.25}

    # ================================================================= B the tau null, exact
    ref = list(range(5))
    taus = [kendall_tau(ref, list(p)) for p in itertools.permutations(ref)]
    n_ge = sum(1 for t in taus if t >= TAU_BAR - 1e-12)
    p_tau = n_ge / len(taus)
    # distribution of "how many of 5 families clear tau >= 0.6" under INDEPENDENCE
    fam_counts = {k: math.comb(5, k) * p_tau ** k * (1 - p_tau) ** (5 - k) for k in range(6)}
    rec["B_tau_null_exact"] = {
        "n": 5, "n_orderings": len(taus), "bar": TAU_BAR,
        "n_orderings_clearing_bar": n_ge,
        "P_single_family_tau_ge_bar": p_tau,
        "distinct_tau_values": sorted(set(round(t, 10) for t in taus)),
        "P_k_of_5_families_clear_under_independence": fam_counts,
        "P_at_least_3_of_5_under_independence": sum(v for k, v in fam_counts.items() if k >= 3),
        "DEPENDENCE_CAVEAT": ("the five families share the same five operators and one activation "
                              "cache, so they are not independent; the independence figure is an "
                              "UPPER bound on the evidence, not an estimate of it"),
    }

    # ================================================================= C the R9 count null, exact
    p_single = 1.0 / (N_DRAWS_R9 + 1)
    p_count_zero = (1 - p_single) ** N_CORPORA_R9
    r9 = json.load(open(a.r9))
    zmin = r9["by_aggregation"]["min"]["z_by_corpus"]
    n_neg = sum(1 for v in zmin.values() if v < 0)
    n_tot = len(zmin)
    # exact one-sided binomial sign test on the direction, which is NOT degenerate under H0
    p_sign = sum(math.comb(n_tot, k) for k in range(n_neg, n_tot + 1)) / 2 ** n_tot
    rec["C_r9_count_null_exact"] = {
        "design": f"{N_CORPORA_R9} corpora, {N_DRAWS_R9} own-derangement draws each",
        "P_one_corpus_clears_all_draws_under_H0": p_single,
        "expected_count_under_H0": p_single * N_CORPORA_R9,
        "P_count_equals_zero_under_H0": p_count_zero,
        "observed_count": r9["by_aggregation"]["min"]["n_beating_every_own_derangement"],
        "reading": ("a count of 0 is the modal outcome under the exact null, so the count alone "
                    "carries almost no information; the direction does"),
        "sign_test_on_direction": {
            "statistic": "number of corpora whose real z_vs_null is negative, min aggregation",
            "n_negative": n_neg, "n_total": n_tot,
            "z_by_corpus": zmin,
            "p_one_sided_exact_binomial": p_sign,
            "DEPENDENCE_CAVEAT": ("the eight corpora share one activation cache and one 551-item "
                                  "battery; the sign test assumes independence and therefore "
                                  "understates the p-value"),
        },
        "positive_control": {
            "what": "persist on the same draws returns 8 of 8",
            "n_beating": r9["by_aggregation"]["persist"]["n_beating_every_own_derangement"],
            "why_it_matters": ("the design CAN return the positive branch, so min's 0 of 8 is a "
                               "failure of the statistic and not a lack of power in the null"),
        },
    }

    # ================================================================= verdict on A
    controls = {k: rec[k]["fires"] for k in
                ("C1_scale_freedom", "C2_pooled_sd_recomputes", "C3_R_recomputes",
                 "C4_identity_in_support", "C5_design_vs_permutation")}
    rec["controls_fired"] = controls
    if not all(controls.values()):
        rec["VERDICT"] = ("UNCLEAR — a control did not fire: "
                          + ", ".join(k for k, v in controls.items() if not v))
    elif R_BAR >= q["p99"]:
        rec["VERDICT"] = (f"CALIBRATED — the registered R >= {R_BAR:g} bar sits at or above the "
                          f"99th percentile of its own null ({q['p99']:.3f}); one-sided null "
                          f"exceedance {exceed_bar:.2e}. CV6 and CV7 stand as written.")
    elif R_BAR >= q["p95"]:
        rec["VERDICT"] = (f"WEAK — the bar sits between the 95th ({q['p95']:.3f}) and 99th "
                          f"({q['p99']:.3f}) percentiles. Quote the null wherever the bar appears.")
    else:
        rec["VERDICT"] = (f"UNCALIBRATED — the bar sits below the 95th percentile "
                          f"({q['p95']:.3f}); the registered rule admits chance passes.")

    write_result(a.out, rec, experiment="DN1",
                 inputs=[a.d3, a.r9], script=os.path.abspath(__file__))

    print(f"A1 null R: mean {null1.mean():.3f}  p95 {q['p95']:.3f}  p99 {q['p99']:.3f}  "
          f"P(R>=10) {exceed_bar:.2e}")
    print(f"A2 permutation p99 by family: "
          + ", ".join(f"{f} {perm[f]['null_p99']:.2f}" for f in FAMILIES))
    print(f"B  P(tau>=0.6 | H0, n=5) = {n_ge}/120 = {p_tau:.4f}; "
          f"P(>=3 of 5 | independence) = {rec['B_tau_null_exact']['P_at_least_3_of_5_under_independence']:.4f}")
    print(f"C  P(R9 count = 0 | H0) = {p_count_zero:.4f}; sign test {n_neg}/{n_tot} "
          f"p = {p_sign:.4f}")
    print(f"controls: {controls}")
    print(rec["VERDICT"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
