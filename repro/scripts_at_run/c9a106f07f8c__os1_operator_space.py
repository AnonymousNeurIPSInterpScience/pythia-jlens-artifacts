#!/usr/bin/env python3
"""os1_operator_space.py — OS1: does the corpus move the operator, or only the score?

PRE-REGISTRATION: docs/experiments/preregs/OS1_operator_space.md, committed before this file.

Recomputes from stored .pt operators. No forward pass, no model weights, no battery, no read
position, no aggregation statistic. That is the point: every other corpus-dependence number in this
programme is a score on a battery whose construct validity the paper itself retires.

PRIMARY   SEP = median(BETWEEN d_rel) / median(WITHIN-SOURCE d_rel), corpus-clustered bootstrap.
RULE      OPERATOR EFFECT  SEP >= 2.0 and CI lower bound > 1.0, under BOTH metrics
          SCORE-ONLY       the interval includes 1.0 under either metric  -> STOP AND ALERT
          UNCLEAR          SEP in (1,2) with CI excluding 1, or the metrics disagree -> STOP

METRICS, per band layer, then averaged over the 13:
  d_rel(A,B)   = ||A-B||_F / sqrt(||A||_F ||B||_F)   scale-aware
  theta_32(A,B)= mean principal angle between top-32 left singular subspaces, in radians. Scale
                 FREE, so it separates "changed how much J transports" from "changed what".

CONTROLS  C1 self-distance exactly 0.  C2 a norm-matched random operator must sit above every
          BETWEEN pair.  C3 fp16 round-trip at least 10x below the within-source median.
          C4 every operator's source_layers == [9..21].  C5 per-layer SEP, reported not gated.

WHAT THIS DOES NOT COVER. It measures displacement of the estimated operator, not whether that
displacement matters for any readout, and not whether J is well estimated at N=200 (en1 bounds
that separately). A large SEP does not make the battery valid; it makes the corpus claim
independent of the battery.

    .venv/bin/python experiments/os1_operator_space.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics as st
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from provenance import write_result  # noqa: E402

RES = os.path.join(HERE, "..", "results")
BAND = list(range(9, 22))
K_SUB = 32
INSTREAM = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github"]
OOD = ["OOD_News_2024", "OOD_arXiv_2023", "OOD_CommonPile"]
SEEDS = [0, 1, 2]
R6_SOURCES = ["Pile-CC", "Wikipedia_en"]
R6_BLOCKS = [0, 1, 2, 3]

SEP_BAR = 2.0
FAIL: list[str] = []


def load(path: str) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    assert d["source_layers"] == BAND, f"{path}: band {d['source_layers'][0]}..{d['source_layers'][-1]}"
    return {l: d["J"][l].float() for l in BAND}


def d_rel(A: torch.Tensor, B: torch.Tensor) -> float:
    na, nb = A.norm().item(), B.norm().item()
    return float((A - B).norm().item() / (na * nb) ** 0.5)


def top_subspace(A: torch.Tensor, k: int) -> torch.Tensor:
    U, _, _ = torch.linalg.svd(A, full_matrices=False)
    return U[:, :k]


def theta(Ua: torch.Tensor, Ub: torch.Tensor) -> float:
    """Mean principal angle, radians. Scale-free: depends only on the subspaces."""
    s = torch.linalg.svdvals(Ua.T @ Ub).clamp(-1.0, 1.0)
    return float(torch.arccos(s).mean().item())


def pair_stats(a: dict, b: dict, sub_a: dict, sub_b: dict) -> tuple[float, float]:
    dr = st.mean(d_rel(a[l], b[l]) for l in BAND)
    th = st.mean(theta(sub_a[l], sub_b[l]) for l in BAND)
    return dr, th


def med(xs) -> float:
    return float(np.median(xs)) if len(xs) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", default=os.path.join(RES, "os1_operator_space_410m.json"))
    a = ap.parse_args()

    rec: dict = {
        "experiment": "OS1 — corpus displacement of the operator itself",
        "prereg": "docs/experiments/preregs/OS1_operator_space.md",
        "status": "PRE-REGISTERED",
        "recomputes_not_remeasures": True,
        "model": "EleutherAI/pythia-410m-deduped", "band": BAND, "N": 200, "k_subspace": K_SUB,
        "scope": ("Displacement of the estimated operator. Says nothing about whether the "
                  "displacement matters for a readout, nor about estimation error at N=200."),
    }

    # ---------------------------------------------------------------- load
    # The paths are enumerated BEFORE loading and handed to write_result as `inputs`, so the
    # provenance block names every operator this result depends on. Until 2026-08-31 this script
    # called write_result with no `inputs`, so `repro/exp/os1_operator_space.sh` was generated with
    # INPUTS=() — and `require_inputs`, the harness gate that refuses to run when an input is
    # missing, had nothing to check. On a clone without the .pt the module died in a raw traceback
    # instead of the one-line diagnostic every other module gives. A control that cannot fire is
    # not a control (CLAUDE.md §3 rule 10).
    op_paths = {f"{c}|s{s}": f"{RES}/e48/lens_INSTREAM_{c}_410m_n200_s{s}.pt"
                for c in INSTREAM for s in SEEDS}
    op_paths.update({f"{c}|s{s}": f"{RES}/e48/lens_{c}_410m_n200_s{s}.pt"
                     for c in OOD for s in SEEDS})
    r6_paths = {f"{src}|b{b}": f"{RES}/r6/lens_R6_{src}_b{b}_410m_n200.pt"
                for src in R6_SOURCES for b in R6_BLOCKS}
    declared_inputs = [op_paths[k] for k in op_paths] + [r6_paths[k] for k in r6_paths]

    missing = [q for q in declared_inputs if not os.path.isfile(q)]
    if missing:
        raise SystemExit(
            f"OS1 needs {len(declared_inputs)} fitted operators; {len(missing)} are absent, e.g.\n"
            f"  {os.path.relpath(missing[0], os.path.join(HERE, '..'))}\n"
            "This is TIER 1 work: it recomputes from stored .pt, which are not in git.\n"
            "Fetch them first:  bash repro/03_fetch_artifacts.sh --all")

    ops: dict[str, dict] = {k: load(v) for k, v in op_paths.items()}
    r6: dict[str, dict] = {k: load(v) for k, v in r6_paths.items()}
    rec["C4_band_matches"] = {"required": "source_layers == [9..21] on all operators",
                              "n_checked": len(ops) + len(r6), "fires": True}
    print(f"  loaded {len(ops)} corpus operators + {len(r6)} within-source refits", flush=True)

    subs = {k: {l: top_subspace(v[l], K_SUB) for l in BAND} for k, v in ops.items()}
    subs_r6 = {k: {l: top_subspace(v[l], K_SUB) for l in BAND} for k, v in r6.items()}
    print("  subspaces done", flush=True)

    CORPORA = INSTREAM + OOD

    # ---------------------------------------------------------------- contrasts
    between, within_seed, within_source = [], [], []
    for c1, c2 in itertools.combinations(CORPORA, 2):
        for s in SEEDS:
            k1, k2 = f"{c1}|s{s}", f"{c2}|s{s}"
            dr, th = pair_stats(ops[k1], ops[k2], subs[k1], subs[k2])
            between.append({"pair": f"{c1}|{c2}", "seed": s, "d_rel": dr, "theta": th,
                            "corpora": [c1, c2]})
    for c in CORPORA:
        for s1, s2 in itertools.combinations(SEEDS, 2):
            k1, k2 = f"{c}|s{s1}", f"{c}|s{s2}"
            dr, th = pair_stats(ops[k1], ops[k2], subs[k1], subs[k2])
            within_seed.append({"corpus": c, "seeds": [s1, s2], "d_rel": dr, "theta": th})
    for src in R6_SOURCES:
        for b1, b2 in itertools.combinations(R6_BLOCKS, 2):
            k1, k2 = f"{src}|b{b1}", f"{src}|b{b2}"
            dr, th = pair_stats(r6[k1], r6[k2], subs_r6[k1], subs_r6[k2])
            within_source.append({"source": src, "blocks": [b1, b2], "d_rel": dr, "theta": th})

    B_dr = [x["d_rel"] for x in between]
    B_th = [x["theta"] for x in between]
    WS_dr = [x["d_rel"] for x in within_source]
    WS_th = [x["theta"] for x in within_source]
    WD_dr = [x["d_rel"] for x in within_seed]
    WD_th = [x["theta"] for x in within_seed]

    rec["contrasts"] = {
        "BETWEEN_corpus": {"n": len(between), "median_d_rel": med(B_dr), "median_theta": med(B_th),
                           "min_d_rel": min(B_dr), "max_d_rel": max(B_dr)},
        "WITHIN_seed_block": {"n": len(within_seed), "median_d_rel": med(WD_dr),
                              "median_theta": med(WD_th)},
        "WITHIN_source_disjoint_docs": {"n": len(within_source), "median_d_rel": med(WS_dr),
                                        "median_theta": med(WS_th),
                                        "note": "R6's four disjoint document blocks per source"},
    }

    sep_dr = med(B_dr) / med(WS_dr)
    sep_th = med(B_th) / med(WS_th)

    # ---------------------------------------------------------------- bootstrap over CORPORA
    rng = np.random.default_rng(0)
    boot_dr, boot_th = [], []
    for _ in range(a.n_boot):
        pick = set(rng.choice(CORPORA, size=len(CORPORA), replace=True).tolist())
        bd = [x["d_rel"] for x in between if x["corpora"][0] in pick and x["corpora"][1] in pick]
        bt = [x["theta"] for x in between if x["corpora"][0] in pick and x["corpora"][1] in pick]
        if not bd:
            continue
        j = rng.integers(0, len(WS_dr), len(WS_dr))
        boot_dr.append(med(bd) / med([WS_dr[i] for i in j]))
        boot_th.append(med(bt) / med([WS_th[i] for i in j]))
    q = lambda v, p: float(np.percentile(v, p))  # noqa: E731
    rec["PRIMARY"] = {
        "SEP_d_rel": sep_dr, "SEP_d_rel_ci": [q(boot_dr, 2.5), q(boot_dr, 97.5)],
        "SEP_theta": sep_th, "SEP_theta_ci": [q(boot_th, 2.5), q(boot_th, 97.5)],
        "n_boot": len(boot_dr),
        "bootstrap_unit": "fitting corpus, resampled with replacement; within-source pairs resampled",
    }

    # ---------------------------------------------------------------- controls
    k0 = f"Pile-CC|s0"
    self_dr = st.mean(d_rel(ops[k0][l], ops[k0][l]) for l in BAND)
    self_th = st.mean(theta(subs[k0][l], subs[k0][l]) for l in BAND)
    rec["C1_self_distance"] = {"required": "exactly 0 on both metrics", "d_rel": self_dr,
                               "theta": self_th, "fires": self_dr == 0.0 and self_th < 1e-6}

    g = torch.Generator().manual_seed(0)
    rand = {}
    for l in BAND:
        A = ops[k0][l]
        R = torch.randn(A.shape, generator=g)
        rand[l] = R * (A.norm() / R.norm())          # norm-matched
    rand_sub = {l: top_subspace(rand[l], K_SUB) for l in BAND}
    r_dr, r_th = pair_stats(ops[k0], rand, subs[k0], rand_sub)
    rec["C2_metric_can_separate"] = {
        "required": "a norm-matched random operator sits above every BETWEEN pair",
        "random_vs_real_d_rel": r_dr, "max_between_d_rel": max(B_dr),
        "random_vs_real_theta": r_th, "max_between_theta": max(B_th),
        "fires": r_dr > max(B_dr) and r_th > max(B_th)}

    rt = st.mean(d_rel(ops[k0][l], ops[k0][l].half().float()) for l in BAND)
    rec["C3_fp16_roundtrip"] = {
        "required": "at least 10x below the within-source median d_rel",
        "roundtrip_d_rel": rt, "within_source_median": med(WS_dr),
        "ratio": med(WS_dr) / rt if rt else float("inf"),
        "fires": rt * 10 <= med(WS_dr)}

    per_layer = {}
    for l in BAND:
        b = [d_rel(ops[f"{c1}|s{s}"][l], ops[f"{c2}|s{s}"][l])
             for c1, c2 in itertools.combinations(CORPORA, 2) for s in SEEDS]
        w = [d_rel(r6[f"{src}|b{b1}"][l], r6[f"{src}|b{b2}"][l])
             for src in R6_SOURCES for b1, b2 in itertools.combinations(R6_BLOCKS, 2)]
        per_layer[l] = {"between": med(b), "within_source": med(w), "SEP": med(b) / med(w)}
    seps = [v["SEP"] for v in per_layer.values()]
    rec["C5_per_layer"] = {"reported_not_gated": True, "by_layer": per_layer,
                           "min_SEP": min(seps), "max_SEP": max(seps),
                           "carried_by_few_layers": (max(seps) / min(seps)) > 3.0}

    controls = {k: rec[k]["fires"] for k in
                ("C1_self_distance", "C2_metric_can_separate", "C3_fp16_roundtrip",
                 "C4_band_matches")}
    rec["controls_fired"] = controls

    # ---------------------------------------------------------------- verdict
    lo_dr, lo_th = rec["PRIMARY"]["SEP_d_rel_ci"][0], rec["PRIMARY"]["SEP_theta_ci"][0]
    if not all(controls.values()):
        rec["VERDICT"] = ("UNCLEAR — a control did not fire: "
                          + ", ".join(k for k, v in controls.items() if not v))
    elif lo_dr <= 1.0 or lo_th <= 1.0:
        rec["VERDICT"] = (f"SCORE-ONLY — an interval includes 1.0 (d_rel CI lo {lo_dr:.2f}, theta "
                          f"CI lo {lo_th:.2f}). The corpus does not measurably move J at this "
                          f"resolution. STOP AND ALERT THE OPERATOR.")
    elif sep_dr >= SEP_BAR and sep_th >= SEP_BAR:
        rec["VERDICT"] = (f"OPERATOR EFFECT — SEP {sep_dr:.2f} on d_rel "
                          f"[{lo_dr:.2f},{rec['PRIMARY']['SEP_d_rel_ci'][1]:.2f}] and {sep_th:.2f} "
                          f"on theta_{K_SUB} [{lo_th:.2f},"
                          f"{rec['PRIMARY']['SEP_theta_ci'][1]:.2f}], both bars met, all controls "
                          f"fired. The corpus moves the operator itself; the claim does not need "
                          f"the battery.")
    else:
        rec["VERDICT"] = (f"UNCLEAR — SEP {sep_dr:.2f} (d_rel) and {sep_th:.2f} (theta), intervals "
                          f"exclude 1.0 but at least one is below the {SEP_BAR} bar, or the metrics "
                          f"disagree. Report and stop.")

    write_result(a.out, rec, experiment="OS1", script=os.path.abspath(__file__),
                 inputs=declared_inputs)
    print(f"\nBETWEEN d_rel median {med(B_dr):.4f}   WITHIN-seed {med(WD_dr):.4f}   "
          f"WITHIN-source {med(WS_dr):.4f}")
    print(f"BETWEEN theta  median {med(B_th):.4f}   WITHIN-seed {med(WD_th):.4f}   "
          f"WITHIN-source {med(WS_th):.4f}")
    print(f"SEP d_rel {sep_dr:.2f} {rec['PRIMARY']['SEP_d_rel_ci']}   "
          f"SEP theta {sep_th:.2f} {rec['PRIMARY']['SEP_theta_ci']}")
    print(f"controls: {controls}")
    print(rec["VERDICT"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
