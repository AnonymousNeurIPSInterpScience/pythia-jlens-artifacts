#!/usr/bin/env python3
"""t57_grid_variance_ci.py — E57 analysis: put an interval on the paper's headline.

WHY
  `paper.tex` reports 91.2% fit / 7.1% read / 1.6% interaction as a two-way decomposition of 64
  cell MEANS with one observation per cell. The three shares therefore sum to 100 by construction,
  the "interaction" is a RESIDUAL conflated with cell noise, and there is no interval.

  t52_factorial.py --cells-out has now emitted the 9 draws per cell (3 fit seeds x 3 prefix seeds)
  that y() averaged away. This bootstraps the decomposition over those draws and separates the
  interaction from the noise.

  PRE-REGISTRATION: specs/experiments/E57_grid_variance_ci.md.
    PRIMARY  the lower bound of the bootstrap interval on (fit_pct - read_pct) under persist.
    RULE     ACCEPT  interval excludes zero under BOTH aggregations
             NARROW  excludes zero under persist only
             REJECT  includes zero under persist
             UNCLEAR C1 did not fire

TWO THINGS THIS COMPUTES

  1. THE BOOTSTRAP. Resample the 3 fit seed blocks and the 3 prefix seeds independently with
     replacement, rebuild the 8x8 of cell means, redo the decomposition. Percentile interval.
     This is deliberately a SMALL-SAMPLE bootstrap -- 3 levels per factor -- so it is optimistic
     about coverage and is a LOWER bound on the true uncertainty. Declared bias 1 of the prereg.

  2. THE VARIANCE-COMPONENTS CORRECTION. With 9 draws per cell the within-cell mean square
     estimates the draw-level noise sigma^2 directly. A cell mean of 9 draws carries sigma^2/9 of
     that noise, so
         E[MS_residual over the 64 cell means] = theta_interaction + sigma^2 / 9
     and the noise-corrected interaction is MS_resid - MS_within/9. If that is <= 0, the residual
     the paper calls an "interaction" is entirely cell noise.

CONTROLS
  C1  the draws must average back to the stored e52 matrix to <= 1e-12. (C2 of the prereg; the
      measurement-side C1 -- that the re-run reproduces e52 at all -- already fired at 0.0e+00.)
  C2  the bootstrap must produce a non-degenerate spread. A zero-width interval means the
      resampling is broken, not that the estimate is certain.

    python experiments/t57_grid_variance_ci.py
"""
from __future__ import annotations
import argparse, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from provenance import write_result  # noqa: E402

import torch

RES = os.path.join(HERE, "..", "results")
CORP8 = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github",
         "OOD_News_2024", "OOD_arXiv_2023", "OOD_CommonPile"]


def decomp(M, corp):
    n = len(corp)
    g = st.mean(M[(f, q)] for f in corp for q in corp)
    row = {f: st.mean(M[(f, q)] for q in corp) for f in corp}
    col = {q: st.mean(M[(f, q)] for f in corp) for q in corp}
    SSr = n * sum((row[f] - g) ** 2 for f in corp)
    SSc = n * sum((col[q] - g) ** 2 for q in corp)
    SSt = sum((M[(f, q)] - g) ** 2 for f in corp for q in corp)
    SSi = SSt - SSr - SSc
    return {"SS_row": SSr, "SS_col": SSc, "SS_resid": SSi, "SS_total": SSt,
            "fit_pct": 100 * SSr / SSt, "read_pct": 100 * SSc / SSt,
            "resid_pct": 100 * SSi / SSt}


def pct(v, q):
    v = sorted(v)
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--cells", default=os.path.join(RES, "e57_factorial_cells_410m.json"))
    ap.add_argument("--out", default=os.path.join(RES, "e57_grid_variance_ci.json"))
    ap.add_argument("--e52", default=os.path.join(RES, "e52_factorial_410m.json"),
                    help="the matrix C1 checks the draws average back to. MUST match --cells's "
                         "readout convention: pairing stripped cells against the unstripped matrix "
                         "would compare two conventions and C1 would fail for the wrong reason.")
    a = ap.parse_args()

    cells = json.load(open(a.cells))
    seeds, pseeds = cells["fit_seeds"], cells["prefix_seeds"]
    e52 = json.load(open(a.e52))
    rec = {"experiment": "E57 — an interval on the headline variance decomposition",
           "prereg": "specs/experiments/E57_grid_variance_ci.md, written before the run",
           "status": "PRE-REGISTERED",
           "inputs": [os.path.relpath(a.cells, os.path.join(HERE, "..")),
                      os.path.relpath(a.e52, os.path.join(HERE, ".."))],
           "readout_convention": ("STRIPPED (anchor rule)" if "rstrip" in os.path.basename(a.cells)
                                  else "UNSTRIPPED (legacy)"),
           "n_draws_per_cell": len(seeds) * len(pseeds), "n_boot": a.n_boot,
           "measurement_side_C1": cells["control_C1_reproduces_stored_e52_matrix"],
           "by_aggregation": {}}

    for ag in ("persist", "min"):
        D = cells["draws"][ag]

        def cell(f, q, ss, pp):
            return st.mean(D[f"{f}|{q}|s{s}|p{p}"] for s in ss for p in pp)

        M = {(f, q): cell(f, q, seeds, pseeds) for f in CORP8 for q in CORP8}
        point = decomp(M, CORP8)

        # ---- C1: the draws must average back to the stored matrix
        worst = max(abs(M[(f, q)] - e52["by_aggregation"][ag]["matrix"][f"{f}|{q}"])
                    for f in CORP8 for q in CORP8)
        c1 = {"max_abs_diff_vs_stored_matrix": worst, "tolerance": 1e-12, "fires": worst <= 1e-12}

        # ---- variance components: within-cell MS estimates the draw-level noise
        SSw, dfw = 0.0, 0
        for f in CORP8:
            for q in CORP8:
                vals = [D[f"{f}|{q}|s{s}|p{p}"] for s in seeds for p in pseeds]
                m = st.mean(vals)
                SSw += sum((v - m) ** 2 for v in vals)
                dfw += len(vals) - 1
        MS_within = SSw / dfw
        df_resid = (len(CORP8) - 1) ** 2
        MS_resid = point["SS_resid"] / df_resid
        theta = MS_resid - MS_within / (len(seeds) * len(pseeds))
        # re-express the corrected interaction as a share of the corrected total
        SS_int_corr = max(0.0, theta) * df_resid
        SS_tot_corr = point["SS_row"] + point["SS_col"] + SS_int_corr
        vc = {
            "MS_within_cell": MS_within, "df_within": dfw,
            "MS_residual_of_cell_means": MS_resid, "df_residual": df_resid,
            "expected_noise_in_MS_residual": MS_within / (len(seeds) * len(pseeds)),
            "noise_corrected_interaction_MS": theta,
            "residual_is_entirely_noise": theta <= 0,
            "fit_pct_after_noise_correction": 100 * point["SS_row"] / SS_tot_corr,
            "read_pct_after_noise_correction": 100 * point["SS_col"] / SS_tot_corr,
            "interaction_pct_after_noise_correction": 100 * SS_int_corr / SS_tot_corr,
            "note": ("E[MS_resid] = theta_interaction + sigma^2/9. A theta <= 0 means the term "
                     "the paper labels 'fit x read interaction' is not distinguishable from "
                     "cell noise."),
        }

        # ---- bootstrap over the two seed factors
        g = torch.Generator().manual_seed(20260816)
        fit_b, read_b, diff_b, int_b = [], [], [], []
        for _ in range(a.n_boot):
            ss = [seeds[i] for i in torch.randint(0, len(seeds), (len(seeds),),
                                                  generator=g).tolist()]
            pp = [pseeds[i] for i in torch.randint(0, len(pseeds), (len(pseeds),),
                                                   generator=g).tolist()]
            Mb = {(f, q): cell(f, q, ss, pp) for f in CORP8 for q in CORP8}
            db = decomp(Mb, CORP8)
            fit_b.append(db["fit_pct"]); read_b.append(db["read_pct"])
            int_b.append(db["resid_pct"]); diff_b.append(db["fit_pct"] - db["read_pct"])
        boot = {
            "fit_pct": {"point": point["fit_pct"], "ci_lo": pct(fit_b, .025),
                        "ci_hi": pct(fit_b, .975)},
            "read_pct": {"point": point["read_pct"], "ci_lo": pct(read_b, .025),
                         "ci_hi": pct(read_b, .975)},
            "resid_pct": {"point": point["resid_pct"], "ci_lo": pct(int_b, .025),
                          "ci_hi": pct(int_b, .975)},
            "fit_minus_read": {"point": point["fit_pct"] - point["read_pct"],
                               "ci_lo": pct(diff_b, .025), "ci_hi": pct(diff_b, .975)},
            "unit": "fit seed block and prefix seed, resampled independently with replacement",
        }
        boot["fit_minus_read"]["excludes_zero"] = boot["fit_minus_read"]["ci_lo"] > 0
        c2 = {"bootstrap_width_fit_pct": boot["fit_pct"]["ci_hi"] - boot["fit_pct"]["ci_lo"],
              "fires": (boot["fit_pct"]["ci_hi"] - boot["fit_pct"]["ci_lo"]) > 0}

        rec["by_aggregation"][ag] = {"point": point, "bootstrap": boot,
                                     "variance_components": vc,
                                     "controls": {"C1_draws_average_to_stored_matrix": c1,
                                                  "C2_bootstrap_non_degenerate": c2}}
        print(f"[{ag}] fit {point['fit_pct']:.2f}% [{boot['fit_pct']['ci_lo']:.2f},"
              f"{boot['fit_pct']['ci_hi']:.2f}]  read {point['read_pct']:.2f}% "
              f"[{boot['read_pct']['ci_lo']:.2f},{boot['read_pct']['ci_hi']:.2f}]")
        print(f"       fit-read {boot['fit_minus_read']['point']:.2f} "
              f"[{boot['fit_minus_read']['ci_lo']:.2f},{boot['fit_minus_read']['ci_hi']:.2f}]  "
              f"excludes 0: {boot['fit_minus_read']['excludes_zero']}")
        print(f"       residual {point['resid_pct']:.2f}%  -> noise-corrected interaction "
              f"{vc['interaction_pct_after_noise_correction']:.2f}%  "
              f"(entirely noise: {vc['residual_is_entirely_noise']})")
        print(f"       C1 {c1['fires']} (max|diff| {c1['max_abs_diff_vs_stored_matrix']:.2e})  "
              f"C2 {c2['fires']}")

    p = rec["by_aggregation"]["persist"]["bootstrap"]["fit_minus_read"]
    m = rec["by_aggregation"]["min"]["bootstrap"]["fit_minus_read"]
    c1ok = all(rec["by_aggregation"][x]["controls"]["C1_draws_average_to_stored_matrix"]["fires"]
               for x in ("persist", "min"))
    if not c1ok:
        rec["VERDICT"] = "UNCLEAR — C1 did not fire; the draws do not average to the stored matrix"
    elif p["excludes_zero"] and m["excludes_zero"]:
        rec["VERDICT"] = (
            f"ACCEPT — the fit axis exceeds the read axis by "
            f"{p['point']:.1f} points [{p['ci_lo']:.1f},{p['ci_hi']:.1f}] under persist and "
            f"{m['point']:.1f} [{m['ci_lo']:.1f},{m['ci_hi']:.1f}] under min; both intervals "
            f"exclude zero. The headline ordering now carries an interval.")
    elif p["excludes_zero"]:
        rec["VERDICT"] = ("NARROW — the interval excludes zero under persist but not under min; "
                          "the headline must be reported as aggregation-dependent")
    else:
        rec["VERDICT"] = ("REJECT — the interval on fit-read includes zero under persist. The "
                          "paper cannot claim the fit axis dominates; only a point estimate.")

    write_result(a.out, rec, experiment="E57",
                 inputs=[a.cells, os.path.join(RES, "e52_factorial_410m.json")])
    print(f"\nVERDICT: {rec['VERDICT']}\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
