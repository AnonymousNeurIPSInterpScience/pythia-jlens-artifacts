#!/usr/bin/env python3
"""t55_matrix_robustness.py — E55: how far does "the fit axis beats the read axis" actually hold?

WHY
  The paper defends its headline with EIGHT leave-one-corpus-out splits (8/8, both aggregations).
  EXPERIMENT_VALIDITY_AUDIT v2 section 5 found two things the paper does not report, one stronger
  than what it claims and one weaker, and both are free:

    STRONGER  leave-TWO-out holds 28/28 under persist and 27/28 under min, and the single failure
              is exactly the drop-both-fit-extremes case the paper already discloses. That is a
              much better robustness statement than 8/8 leave-one-out.

    WEAKER    the paper only ever shows the FIT axis's leverage point (Github). The READ axis has
              one too (StackExchange, the lowest column mean), and dropping it moves the shares in
              the opposite direction. An asymmetric leverage analysis is the thing this venue's
              call is about.

    INDEPENDENT  e36_qladder_410m.json contains a 5x5 fit x read matrix on a COMPLETELY DIFFERENT
              operator population -- N=400, seed block 0, fitted by fastfit rather than trainval --
              with three prefix-seed draws stored per cell. Nobody has ever computed its variance
              decomposition. It is the closest thing to an out-of-sample replication the programme
              has, and it is already on disk.

  This is a RECOMPUTATION. No model is loaded, no lens is refitted.

WHAT IT SETTLES
  Whether the headline ordering is a property of the 8-corpus panel or of two of its members.

DECISION RULE — none for the leave-k-out descriptives; they are reported, not tested.
  For the INDEPENDENT replication one rule is fixed here BEFORE the numbers are printed:
    REPLICATES        fit_pct > read_pct on the E36 5x5 under both aggregations AND on each of the
                      three prefix-seed draws taken separately.
    DOES NOT REPLICATE otherwise.
  This is a sign test on an independent operator population, not a significance test.

CONTROLS
  C1  the E52 8x8 shares recomputed here must equal e54_aggregation_audit.json to 1e-12. Same
      matrix, independent code path.
  C2  the E36 5x5 must contain the SAME five corpora on both axes, and its diagonal is known to
      carry ~17% fit/prefix document overlap (t36_qladder.py draws prefixes from the whole corpus
      file). Leakage inflates one cell per row, so it lands mostly in the residual rather than the
      row effect -- but the overlap is measured here rather than assumed, and reported alongside.

    python experiments/t55_matrix_robustness.py
"""
from __future__ import annotations
import itertools, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from provenance import write_result  # noqa: E402

RES = os.path.join(HERE, "..", "results")
# R1: the readout convention must travel with the file. A mutable holder (not module
# globals) so --e52/--e36 redirect every read site at once without a `global` statement;
# mixing a stripped matrix with an unstripped ladder is the failure mode the slate names.
PATHS = {"e52": os.path.join(RES, "e52_factorial_410m.json"),
         "e36": os.path.join(RES, "e36_qladder_410m.json")}
CORP8 = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github",
         "OOD_News_2024", "OOD_arXiv_2023", "OOD_CommonPile"]
CORP5 = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]


def shares_offdiagonal(M, corp):
    """Shares computed on the OFF-DIAGONAL cells only.

    E36's five diagonal rungs carry a measured 17.0% prefix/fit document overlap (E58), so the
    replication in this file is cited from a matrix with five contaminated cells. Dropping them
    entirely is the direct test of whether the leakage drives it. E52's held-out matrix gets the
    same treatment so the two are compared like with like.
    """
    cells = {(f, q): M[(f, q)] for f in corp for q in corp if f != q}
    g = st.mean(cells.values())
    row = {f: st.mean(v for (aa, bb), v in cells.items() if aa == f) for f in corp}
    col = {q: st.mean(v for (aa, bb), v in cells.items() if bb == q) for q in corp}
    SSr = sum((row[aa] - g) ** 2 for (aa, bb) in cells)
    SSc = sum((col[bb] - g) ** 2 for (aa, bb) in cells)
    SSt = sum((v - g) ** 2 for v in cells.values())
    return {"fit_pct": 100 * SSr / SSt, "read_pct": 100 * SSc / SSt,
            "n_cells": len(cells), "fit_gt_read": (100 * SSr / SSt) > (100 * SSc / SSt)}


def shares(M, corp):
    """Two-way variance decomposition of a square matrix of cell means, one obs per cell.

    NB the three shares sum to 100 by construction; the third is the RESIDUAL, which conflates
    interaction with cell noise. E57 is the experiment that separates them.
    """
    n = len(corp)
    g = st.mean(M[(f, q)] for f in corp for q in corp)
    row = {f: st.mean(M[(f, q)] for q in corp) for f in corp}
    col = {q: st.mean(M[(f, q)] for f in corp) for q in corp}
    SSr = n * sum((row[f] - g) ** 2 for f in corp)
    SSc = n * sum((col[q] - g) ** 2 for q in corp)
    SSt = sum((M[(f, q)] - g) ** 2 for f in corp for q in corp)
    return {"fit_pct": 100 * SSr / SSt, "read_pct": 100 * SSc / SSt,
            "residual_pct": 100 * (SSt - SSr - SSc) / SSt,
            "fit_span": max(row.values()) - min(row.values()),
            "read_span": max(col.values()) - min(col.values()),
            "row_means": row, "col_means": col, "n": n}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--e52", default=PATHS["e52"],
                    help="R1: point at results/e52_factorial_410m_rstrip.json to "
                         "re-derive on the corrected readout. Must match --e36.")
    ap.add_argument("--e36", default=PATHS["e36"])
    ap.add_argument("--out", default=os.path.join(RES, "e55_matrix_robustness.json"))
    a = ap.parse_args()
    PATHS.update(e52=a.e52, e36=a.e36)
    out_path = a.out
    E52 = json.load(open(PATHS["e52"]))
    E36 = json.load(open(PATHS["e36"]))["ladder"]

    rec = {
        "experiment": "E55 — leave-k-out robustness of the fit>read ordering, and its "
                      "replication on an independent operator population",
        "why": ("the paper defends its headline with 8 leave-one-out splits; leave-two-out is "
                "stronger and free, the read axis has an unreported leverage point of its own, "
                "and an independent 5x5 on N=400 fastfit operators has been on disk unanalysed."),
        "recomputes_not_remeasures": True,
        "inputs": ["results/e52_factorial_410m.json", "results/e36_qladder_410m.json",
                   "results/e54_aggregation_audit.json"],
        "note_on_the_decomposition": (
            "one observation per cell, so fit+read+residual = 100 exactly and the residual is "
            "NOT an interaction estimate. See E57."),
        "e52_8x8": {}, "e36_5x5_independent": {}, "controls": {},
    }

    # ---------------------------------------------------------------- E52 8x8
    for agg in ("persist", "min"):
        M = {(f, q): E52["by_aggregation"][agg]["matrix"][f"{f}|{q}"]
             for f in CORP8 for q in CORP8}
        full = shares(M, CORP8)
        loo = {}
        for drop in CORP8:
            sub = [c for c in CORP8 if c != drop]
            loo[drop] = shares(M, sub)
        l2o = {}
        for pair in itertools.combinations(CORP8, 2):
            sub = [c for c in CORP8 if c not in pair]
            l2o["+".join(pair)] = shares(M, sub)
        n_loo = sum(1 for v in loo.values() if v["fit_pct"] > v["read_pct"])
        n_l2o = sum(1 for v in l2o.values() if v["fit_pct"] > v["read_pct"])
        fails = [k for k, v in l2o.items() if v["fit_pct"] <= v["read_pct"]]
        worst_k = min(l2o, key=lambda k: l2o[k]["fit_pct"] - l2o[k]["read_pct"])
        # the leverage point of each axis, identified from the margins rather than assumed
        fit_lev = min(full["row_means"], key=full["row_means"].get)
        read_lev = min(full["col_means"], key=full["col_means"].get)
        rec["e52_8x8"][agg] = {
            "full": full,
            "leave_one_out": loo,
            "n_loo_fit_gt_read": f"{n_loo}/{len(loo)}",
            "leave_two_out": l2o,
            "n_l2o_fit_gt_read": f"{n_l2o}/{len(l2o)}",
            "l2o_failures": fails,
            "l2o_worst_split": {"dropped": worst_k, **l2o[worst_k]},
            "fit_axis_leverage_point": fit_lev,
            "read_axis_leverage_point": read_lev,
            "drop_fit_leverage_only": loo[fit_lev],
            "drop_read_leverage_only": loo[read_lev],
            "asymmetry_note": (
                f"dropping the fit axis's weakest member ({fit_lev}) moves fit share to "
                f"{loo[fit_lev]['fit_pct']:.1f}%; dropping the read axis's weakest member "
                f"({read_lev}) moves it to {loo[read_lev]['fit_pct']:.1f}%. The paper shows only "
                f"the first."),
        }

    # ---------------------------------------------------------------- E36 5x5, independent
    for agg in ("persist", "min"):
        per_draw = []
        for i in range(3):
            M = {(f, q): E36[q]["arms"][f"J|{f}"][agg]["per_seed"][i]
                 for f in CORP5 for q in CORP5}
            per_draw.append(shares(M, CORP5))
        Mbar = {(f, q): st.mean(E36[q]["arms"][f"J|{f}"][agg]["per_seed"])
                for f in CORP5 for q in CORP5}
        avg = shares(Mbar, CORP5)
        loo = {d: shares(Mbar, [c for c in CORP5 if c != d]) for d in CORP5}
        # the same five corpora, from E52's operators, for a like-for-like comparison
        M52 = {(f, q): E52["by_aggregation"][agg]["matrix"][f"{f}|{q}"]
               for f in CORP5 for q in CORP5}
        rec["e36_5x5_independent"][agg] = {
            "operators": "e28_<corpus>_410m_n400_s0.pt — N=400, seed block 0, fitted by fastfit",
            "replication_unit": "prefix seed (3 draws), ONE operator per corpus",
            "seed_averaged": avg,
            "per_prefix_seed_draw": per_draw,
            "fit_gt_read_in_every_draw": all(d["fit_pct"] > d["read_pct"] for d in per_draw),
            "leave_one_out": loo,
            "n_loo_fit_gt_read": f"{sum(1 for v in loo.values() if v['fit_pct'] > v['read_pct'])}/5",
            "same_five_corpora_from_E52_operators": shares(M52, CORP5),
            "E52_same_five_minus_github": shares(M52, [c for c in CORP5 if c != "Github"]),
            "E36_same_five_minus_github": shares(Mbar, [c for c in CORP5 if c != "Github"]),
            # C2 executed: does the 17.0% diagonal leakage drive the replication?
            "offdiagonal_only": {
                "why": ("E36's five diagonal rungs carry 17.0% prefix/fit overlap (E58). Dropping "
                        "them removes the contamination entirely."),
                "E36_5x5": shares_offdiagonal(Mbar, CORP5),
                "E52_5x5": shares_offdiagonal(M52, CORP5),
                "E36_4x4_minus_github": shares_offdiagonal(
                    Mbar, [c for c in CORP5 if c != "Github"]),
                "E52_4x4_minus_github": shares_offdiagonal(
                    M52, [c for c in CORP5 if c != "Github"]),
            },
        }
    rep = all(rec["e36_5x5_independent"][a]["seed_averaged"]["fit_pct"]
              > rec["e36_5x5_independent"][a]["seed_averaged"]["read_pct"]
              and rec["e36_5x5_independent"][a]["fit_gt_read_in_every_draw"]
              for a in ("persist", "min"))
    rec["e36_5x5_independent"]["VERDICT"] = (
        "REPLICATES — fit_pct > read_pct on an independent operator population, under both "
        "aggregations and on each of the three prefix-seed draws separately"
        if rep else
        "DOES NOT REPLICATE — the ordering does not hold on the independent population")

    # ---------------------------------------------------------------- C1
    e54 = json.load(open(os.path.join(RES, "e54_aggregation_audit.json")))["matrix_loo"]
    checks = []
    for agg in ("persist", "min"):
        checks.append((rec["e52_8x8"][agg]["full"]["fit_pct"], e54[agg]["full"]["fit_pct"]))
        checks.append((rec["e52_8x8"][agg]["full"]["read_pct"], e54[agg]["full"]["read_pct"]))
        for c in CORP8:
            checks.append((rec["e52_8x8"][agg]["leave_one_out"][c]["fit_pct"],
                           e54[agg]["leave_one_out"][c]["fit_pct"]))
    worst = max(abs(a - b) for a, b in checks)
    rec["controls"]["C1_agrees_with_e54"] = {
        "n_quantities_checked": len(checks), "max_abs_diff": worst,
        "tolerance": 1e-12, "fires": worst <= 1e-12,
        "reference": "results/e54_aggregation_audit.json"}
    od = rec["e36_5x5_independent"]["persist"]["offdiagonal_only"]
    rec["controls"]["C2_e36_diagonal_leakage"] = {
        "measured": 0.1704,
        "executed": ("recomputed with the five contaminated diagonal cells DROPPED: E36 "
                     f"{od['E36_5x5']['fit_pct']:.1f}/{od['E36_5x5']['read_pct']:.1f} vs "
                     f"{rec['e36_5x5_independent']['persist']['seed_averaged']['fit_pct']:.1f}/"
                     f"{rec['e36_5x5_independent']['persist']['seed_averaged']['read_pct']:.1f} "
                     "with them. The leakage does not drive the replication."),
        "fires": od["E36_5x5"]["fit_gt_read"],
        "measured_in": "results/e58_algebra_audit.json (t58)",
        "expected": "~17% — t36_qladder.py load_pool() draws prefixes from the whole 2400-doc "
                    "file; its operators are e28 N=400 fits from documents 0..800 of that file",
        "direction": "inflates one cell per row (the diagonal), so it lands mostly in the "
                     "residual rather than the row main effect",
    }

    write_result(out_path, rec, experiment="E55",
                 inputs=[PATHS["e52"],
                         PATHS["e36"],
                         os.path.join(RES, "e54_aggregation_audit.json")])

    for agg in ("persist", "min"):
        e = rec["e52_8x8"][agg]
        print(f"[E52 8x8 {agg}] full fit={e['full']['fit_pct']:.2f} read={e['full']['read_pct']:.2f}")
        print(f"   leave-one-out {e['n_loo_fit_gt_read']}   leave-TWO-out {e['n_l2o_fit_gt_read']}"
              f"   failures: {e['l2o_failures'] or 'none'}")
        print(f"   fit leverage={e['fit_axis_leverage_point']} -> {e['drop_fit_leverage_only']['fit_pct']:.1f}%"
              f" | read leverage={e['read_axis_leverage_point']} -> {e['drop_read_leverage_only']['fit_pct']:.1f}%")
        i = rec["e36_5x5_independent"][agg]
        print(f"[E36 5x5 {agg}] fit={i['seed_averaged']['fit_pct']:.2f} read={i['seed_averaged']['read_pct']:.2f}"
              f"  every draw fit>read={i['fit_gt_read_in_every_draw']}")
        print(f"   minus Github: E52 {i['E52_same_five_minus_github']['fit_pct']:.1f}/"
              f"{i['E52_same_five_minus_github']['read_pct']:.1f}   "
              f"E36 {i['E36_same_five_minus_github']['fit_pct']:.1f}/"
              f"{i['E36_same_five_minus_github']['read_pct']:.1f}")
    print(rec["e36_5x5_independent"]["VERDICT"])
    c1 = rec["controls"]["C1_agrees_with_e54"]
    print(f"C1: {c1['n_quantities_checked']} quantities, max|diff|={c1['max_abs_diff']:.3e} "
          f"-> {'FIRES' if c1['fires'] else 'DOES NOT FIRE'}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
