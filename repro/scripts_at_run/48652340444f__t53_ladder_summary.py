#!/usr/bin/env python3
"""t53_ladder_summary.py — E53: store the ladder numbers the paper states and no file contains.

WHY
  EXPERIMENT_VALIDITY_AUDIT v1 §5 specified this script and it was never written; `./lab health`
  H10 has been flagging two dangling references to it. v2 §3.1 then found the paper's N-ladder
  sentence is contradicted by the very file it cites. Both are closed here.

  Three families of number are asserted in `paper/paper.tex` with no results file behind them:
    1. the SHAPE OF THE LADDER GRID itself — "5 corpora x 3 seed blocks x 16 values of N in
       [25,800]". Only 8 of 15 corpus-seed cells reach N=800; the other 7 stop at 400. And the 16
       values are not 16 fits: e28_ladder_410m.py reconstructs them as consecutive-run sums of six
       DISJOINT fitted blocks, so the true count is 83 fitted lenses, not 240.
    2. the CROSS-SCALE ORDERING. Limitation 6 says the corpus effect "replicates at 1B". Whether
       the same corpora are good at both scales is a different claim from whether corpora differ
       at both scales, and only the second is anywhere on disk.
    3. the E36 CONTAINMENT SLOPES on a LOG x-axis. e54 stores the linear-containment slopes with
       leave-one-rung-out; Appendix B's argument is about a figure whose x-axis is log.

  This is a RECOMPUTATION, not a measurement. No model is loaded, no lens is refitted.

WHAT IT SETTLES
  Nothing. It is a storage script. Its only job is to make four paper sentences citable.

DECISION RULE — none. This is an audit artifact, not a hypothesis test.

CONTROLS
  C1  every quantity this script also finds in e54_aggregation_audit.json must agree to 1e-12.
      If it does not, one of the two recomputations is wrong and neither can be cited.
  C2  the reconstructed N grid must be exactly the set of consecutive-run sums of the fitted
      blocks present on disk. Asserted against the actual e28_*.pt filenames, not against the
      ladder JSON, so a missing .pt cannot hide inside an already-scored ladder.

    python experiments/t53_ladder_summary.py
"""
from __future__ import annotations
import glob, json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from provenance import write_result  # noqa: E402

RES = os.path.join(HERE, "..", "results")
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
CORPORA = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]
N_MIN = 75


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for k, i in enumerate(order):
        r[i] = k
    return r


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def ols_slope(x, y):
    mx, my = st.mean(x), st.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)


def consecutive_runs(fitted):
    """The N values reconstruct() can build from disjoint blocks with these cumulative bounds."""
    bounds = [0] + sorted(fitted)
    out = set()
    for i in range(len(bounds) - 1):
        for j in range(i + 1, len(bounds)):
            n = bounds[j] - bounds[i]
            if n > 0:
                out.add(n)
    return sorted(out)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder410-dir", default=os.path.join(RES, "ladder410"),
                    help="R3: point at results/ladder410_cpu to re-derive the summary from "
                         "CPU-scored cells, closing D2.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    L410 = a.ladder410_dir
    out_path = a.out or os.path.join(RES, "e53_ladder_summary.json")

    # ---------------------------------------------------------------- 1. the true grid shape
    grid = {}
    total_fitted = 0
    for c in CORPORA:
        for s in (0, 1, 2):
            fitted = sorted(int(os.path.basename(p).split("_n")[1].split("_")[0])
                            for p in glob.glob(os.path.join(RES, f"e28_{c}_410m_n*_s{s}.pt")))
            total_fitted += len(fitted)
            lad = json.load(open(os.path.join(L410, f"ladder_{c}_s{s}.json")))
            scored = sorted(int(n) for n in lad["by_N"])
            grid[f"{c}|s{s}"] = {
                "fitted_blocks": fitted, "n_fitted_lenses": len(fitted),
                "reconstructed_N": scored, "n_reconstructed": len(scored),
                "max_N": max(scored), "reaches_800": max(scored) == 800,
                "C2_reconstruction_matches_fitted_blocks": consecutive_runs(fitted) == scored,
            }
    c2 = all(v["C2_reconstruction_matches_fitted_blocks"] for v in grid.values())
    n800 = sum(1 for v in grid.values() if v["reaches_800"])

    # ---------------------------------------------------------------- 2. asymptotes per scale
    def ladder_cells(scale, agg):
        out = {}
        for c in CORPORA:
            per_seed = []
            for s in (0, 1, 2):
                if scale == "410m":
                    d = json.load(open(os.path.join(L410, f"ladder_{c}_s{s}.json")))
                    per_seed.append({int(N): st.mean(v[k][agg] for k in ADMITTED)
                                     for N, v in d["by_N"].items()})
                else:
                    d = json.load(open(os.path.join(RES, "ladder1b", f"tv_{c}_s{s}.json")))
                    per_seed.append({int(N): st.mean(v["reads"][k][agg] for k in ADMITTED)
                                     for N, v in d["by_N"].items()})
            out[c] = per_seed
        return out

    scales = {}
    for scale in ("410m", "1b"):
        scales[scale] = {}
        for agg in ("persist", "min"):
            cells = ladder_cells(scale, agg)
            asym, sd = {}, {}
            for c in CORPORA:
                per = [st.mean(v for N, v in ps.items() if N >= N_MIN) for ps in cells[c]]
                asym[c] = st.mean(per)
                sd[c] = st.pstdev(per) * math.sqrt(3 / 2)   # sample sd over 3 seed blocks
            order = sorted(CORPORA, key=lambda c: -asym[c])
            # the N grid the asymptote is actually averaged over, per corpus
            n_used = {c: sorted(N for N in cells[c][0] if N >= N_MIN) for c in CORPORA}
            scales[scale][agg] = {
                "asymptote": asym, "seed_sd": sd, "order_best_to_worst": order,
                "asymptote_N_grid_per_corpus": n_used,
                "asymptote_grids_identical_across_corpora":
                    len({tuple(v) for v in n_used.values()}) == 1,
                "spread": max(asym.values()) - min(asym.values()),
                # t54's convention: the MEAN of the per-corpus seed SDs, not their RMS. Kept
                # identical so C1 compares like with like; the RMS is stored alongside because a
                # ratio to a mean-of-SDs is not a standard error and should not be read as one.
                "pooled_seed_sd": st.mean(sd.values()),
                "pooled_seed_sd_rms": math.sqrt(st.mean([v ** 2 for v in sd.values()])),
            }
            scales[scale][agg]["spread_over_seed_sd"] = (
                scales[scale][agg]["spread"] / scales[scale][agg]["pooled_seed_sd"])

    # ---------------------------------------------------------------- 3. cross-scale ordering
    cross = {}
    for agg in ("persist", "min"):
        a410 = scales["410m"][agg]["asymptote"]
        a1b = scales["1b"][agg]["asymptote"]
        rho = spearman([a410[c] for c in CORPORA], [a1b[c] for c in CORPORA])
        cross[agg] = {
            "order_410m": scales["410m"][agg]["order_best_to_worst"],
            "order_1b": scales["1b"][agg]["order_best_to_worst"],
            "spearman_of_asymptotes": rho, "n": len(CORPORA),
            "worst_corpus_410m": scales["410m"][agg]["order_best_to_worst"][-1],
            "worst_corpus_1b": scales["1b"][agg]["order_best_to_worst"][-1],
            "same_worst_corpus": (scales["410m"][agg]["order_best_to_worst"][-1]
                                  == scales["1b"][agg]["order_best_to_worst"][-1]),
            "note": ("n=5, so rho=+0.7 is p~0.19 and rho=+0.8 is p~0.10. 'The corpus effect "
                     "replicates at 1B' is a claim that corpora DIFFER at both scales, which "
                     "holds; it is not a claim that the SAME corpora are good, which this does "
                     "not establish."),
        }

    # ---------------------------------------------------------------- 4. bands, declared vs used
    bands = {}
    for scale, n_layers in (("410m", 24), ("1b", 16)):
        lo, hi = round(0.38 * n_layers), round(0.92 * n_layers)
        target = n_layers - 2
        declared = [lo, min(hi, target - 1)]
        if scale == "410m":
            used = json.load(open(os.path.join(L410, "ladder_Github_s0.json")))["band"]
        else:
            used = json.load(open(os.path.join(RES, "ladder1b", "tv_Github_s0.json")))["band"]
        bands[scale] = {"n_layers": n_layers, "L38_L92": [lo, hi], "target_layer_index": target,
                        "declared_by_the_papers_own_rule": declared,
                        "used_in_the_ladder": [used[0], used[-1]],
                        "matches": [used[0], used[-1]] == declared}

    # ---------------------------------------------------------------- 5. E36 slopes, lin and log
    e36 = json.load(open(os.path.join(RES, "e36_qladder_410m.json")))["ladder"]
    rungs = [r for r, v in e36.items()
             if r != "Q0" and not r.startswith("SHUFFLED_") and v.get("containment_k32")]
    x_lin = [e36[r]["containment_k32"] for r in rungs]
    x_log = [math.log10(v) for v in x_lin]
    slopes = {}
    for agg in ("persist", "min"):
        arms = ["logit"] + [f"J|{c}" for c in CORPORA]
        slopes[agg] = {}
        for arm in arms:
            y = [e36[r]["arms"][arm][agg]["mean"] for r in rungs]
            loo_lin, loo_log = {}, {}
            for i, r in enumerate(rungs):
                xs_l = [v for j, v in enumerate(x_lin) if j != i]
                xs_g = [v for j, v in enumerate(x_log) if j != i]
                ys = [v for j, v in enumerate(y) if j != i]
                loo_lin[r] = ols_slope(xs_l, ys)
                loo_log[r] = ols_slope(xs_g, ys)
            s_lin, s_log = ols_slope(x_lin, y), ols_slope(x_log, y)
            slopes[agg][arm] = {
                "slope_linear_containment": s_lin,
                "slope_log10_containment": s_log,
                "spearman": spearman(x_lin, y),
                "loo_slope_linear_by_dropped_rung": loo_lin,
                "loo_slope_log10_by_dropped_rung": loo_log,
                "n_loo_sign_flips_linear": sum(1 for v in loo_lin.values() if v * s_lin < 0),
                "n_loo_sign_flips_log10": sum(1 for v in loo_log.values() if v * s_log < 0),
            }
        fitted_arms = [f"J|{c}" for c in CORPORA]
        slopes[agg]["_summary"] = {
            "n_rungs": len(rungs),
            "max_abs_spearman": max(abs(slopes[agg][a]["spearman"]) for a in arms),
            "n_fitted_arms_flipping_sign_when_github_rung_dropped_linear":
                sum(1 for a in fitted_arms
                    if slopes[agg][a]["loo_slope_linear_by_dropped_rung"]["Github"]
                    * slopes[agg][a]["slope_linear_containment"] < 0),
            "n_fitted_arms_flipping_sign_when_github_rung_dropped_log10":
                sum(1 for a in fitted_arms
                    if slopes[agg][a]["loo_slope_log10_by_dropped_rung"]["Github"]
                    * slopes[agg][a]["slope_log10_containment"] < 0),
        }

    # ---------------------------------------------------------------- C1 vs e54
    e54 = json.load(open(os.path.join(RES, "e54_aggregation_audit.json")))
    checks = {}
    for scale in ("410m", "1b"):
        for agg in ("persist", "min"):
            ref = e54["ladder"][scale][agg]
            for c in CORPORA:
                checks[f"{scale}|{agg}|{c}|asymptote"] = (
                    scales[scale][agg]["asymptote"][c], ref["per_corpus_mean"][c])
            checks[f"{scale}|{agg}|spread_over_seed_sd"] = (
                scales[scale][agg]["spread_over_seed_sd"], ref["spread_over_seed_sd"])
    worst = max(abs(a - b) for a, b in checks.values())
    c1 = {"n_quantities_checked": len(checks), "max_abs_diff": worst,
          "tolerance": 1e-12, "fires": worst <= 1e-12,
          "reference": "results/e54_aggregation_audit.json"}

    rec = {
        "experiment": "E53 — the ladder numbers the paper states and no file contains",
        "why": ("EXPERIMENT_VALIDITY_AUDIT v1 section 5 specified this script and it was never "
                "written (lab health H10 flagged it); v2 section 3.1 then found the paper's "
                "N-ladder sentence is contradicted by the file it cites."),
        "recomputes_not_remeasures": True,
        "ladder410_dir": os.path.relpath(L410, os.path.join(HERE, "..")),
        "inputs": [os.path.relpath(L410, os.path.join(HERE, "..")) + "/*.json",
                   "results/ladder1b/*.json",
                   "results/e28_*_410m_n*_s*.pt (filenames only)",
                   "results/e36_qladder_410m.json", "results/e54_aggregation_audit.json"],
        "admitted_sets": ADMITTED, "asymptote_n_min": N_MIN,
        "grid_shape": {
            "per_cell": grid,
            "n_cells": len(grid),
            "n_cells_reaching_N800": n800,
            "n_cells_stopping_at_N400": len(grid) - n800,
            "total_fitted_lenses_410m": total_fitted,
            "C2_every_reconstruction_matches_its_fitted_blocks": c2,
            "PAPER_SAYS": "5 corpora x 3 seed blocks x 16 values of N in [25,800]",
            "WHAT_IS_ON_DISK": (f"{total_fitted} fitted lenses across {len(grid)} corpus-seed "
                                f"cells; {n800} cells reach N=800 with 16 reconstructed values, "
                                f"{len(grid)-n800} stop at N=400 with 11. The N values are "
                                f"consecutive-run sums of disjoint fitted blocks "
                                f"(e28_ladder_410m.py reconstruct()), not independent fits."),
        },
        "by_scale": scales,
        "cross_scale": cross,
        "bands_declared_vs_used": bands,
        "e36_containment_slopes": slopes,
        "control_C1_agrees_with_e54": c1,
    }
    write_result(out_path, rec, experiment="E53",
                 inputs=[os.path.join(RES, "e54_aggregation_audit.json"),
                         os.path.join(RES, "e36_qladder_410m.json")])

    print(f"grid: {total_fitted} fitted lenses, {n800}/{len(grid)} cells reach N=800, "
          f"C2 reconstruction match = {c2}")
    for scale in ("410m", "1b"):
        b = bands[scale]
        print(f"band {scale}: declared {b['declared_by_the_papers_own_rule']} "
              f"used {b['used_in_the_ladder']}  MATCHES={b['matches']}")
    for agg in ("persist", "min"):
        print(f"cross-scale {agg}: rho={cross[agg]['spearman_of_asymptotes']:+.3f} "
              f"same worst corpus={cross[agg]['same_worst_corpus']}")
        s = slopes[agg]["_summary"]
        print(f"  E36 {agg}: max|rho|={s['max_abs_spearman']:.3f}  "
              f"sign flips dropping the Github rung: linear "
              f"{s['n_fitted_arms_flipping_sign_when_github_rung_dropped_linear']}/5, log10 "
              f"{s['n_fitted_arms_flipping_sign_when_github_rung_dropped_log10']}/5")
    print(f"C1 vs e54: {c1['n_quantities_checked']} quantities, max|diff|={c1['max_abs_diff']:.3e} "
          f"-> {'FIRES' if c1['fires'] else 'DOES NOT FIRE'}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
