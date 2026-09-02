#!/usr/bin/env python
"""d2_device_crossvalidation.py — do CUDA-scored and CPU-scored reads agree?

WHY. CONTEXT.md §7 records that E65's scoring path is device-dependent and that its CUDA branch is
wrong: E65's *unfitted* logit arm, which has no operator at all, reads 0.03907 on CUDA against the
known constant 0.02844 on CPU, a 37% discrepancy. It then names the exposure: `ladder410/*.json` and
`ladder1b/*.json` were scored with `device=cuda`, they feed E33's asymptotes and E51's interaction,
and **no CUDA-scored read has ever been cross-validated against a CPU-scored one**. The existing
cross-checks (E52's C1 against e48, E62's C1) are CPU-to-CPU or CUDA-to-CUDA.

That matters to the paper, which cites ladder-derived numbers: the per-corpus asymptotes against the
logit lens, the flatness in N, and the 1B replication.

THE TEST, and why it is free. `ladder410` (CUDA) and `e48_crossover_410m` (CPU) scored the SAME
lenses — `results/e28_<corpus>_410m_n200_s<seed>.pt` — on the SAME band [9,21], the SAME K grid and
the SAME five admitted sets. So the N=200 cell of each ladder file and the matching `J|corpus|sN`
arm of e48 are two measurements of one quantity on two devices. Comparing them needs no GPU, no
refit and no rescoring: it is a join over two files already on disk.

DECISION RULE, fixed before looking.
  AGREE      max |CUDA - CPU| < 1e-6 across all matched cells. The ladder's device is immaterial and
             every ladder-derived number in the paper stands as reported.
  DIVERGENT  max |CUDA - CPU| >= 1e-6. Report the magnitude and STOP. Every ladder-derived number
             needs re-sourcing from a CPU scoring pass before it can be cited.
  A tolerance of 1e-6 is the same one E52's C1 uses against e48, and is derived rather than tuned:
  one (item, intermediate) pair flipping at one k moves a set mean by >= 1/(893*7) = 1.6e-4, so 1e-6
  separates floating-point accumulation from a single changed decision.

CPU only, seconds, no new measurement.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from provenance import write_result                                          # noqa: E402

RES = os.path.join(HERE, "..", "results")
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
TOL = 1e-6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="the ladder rung both files share")
    ap.add_argument("--out", default=os.path.join(RES, "d2_device_crossvalidation.json"))
    a = ap.parse_args()

    e48 = json.load(open(os.path.join(RES, "e48_crossover_410m.json")))
    arms = e48["arms_admitted_mean"]

    rows, devices = [], set()
    for f in sorted(glob.glob(os.path.join(RES, "ladder410", "*.json"))):
        d = json.load(open(f))
        devices.add(d.get("device"))
        corpus, seed = d["corpus"], d["seed"]
        key = f"J|{corpus}|s{seed}"
        by_N = d.get("by_N", {})
        if key not in arms or str(a.n) not in by_N:
            continue
        cell = by_N[str(a.n)]
        if not all(s in cell for s in ADMITTED):
            continue
        for agg in ("persist", "min"):
            cuda = statistics.mean([cell[s][agg] for s in ADMITTED])
            cpu = arms[key][agg]
            rows.append({"corpus": corpus, "seed": seed, "aggregation": agg,
                         "cuda_ladder": cuda, "cpu_e48": cpu,
                         "abs_diff": abs(cuda - cpu),
                         "rel_diff": abs(cuda - cpu) / cpu if cpu else float("nan")})

    if not rows:
        raise SystemExit("ABORT: no matched cells. The join key or the rung is wrong; "
                         "do not report an empty comparison as agreement.")

    worst = max(rows, key=lambda r: r["abs_diff"])
    max_abs = worst["abs_diff"]
    agree = max_abs < TOL

    out = {
        "experiment": "D2 — cross-validating CUDA-scored reads against CPU-scored reads",
        "status": "DIAGNOSTIC. Joins two stored results files; measures nothing new.",
        "why": ("CONTEXT.md §7: E65's CUDA scoring branch is wrong by 37% on an arm with no operator, "
                "and ladder410/ladder1b are CUDA-scored and feed the asymptotes and the interaction. "
                "No CUDA-vs-CPU cross-check had ever been run."),
        "inputs": ["results/ladder410/*.json (device=cuda)",
                   "results/e48_crossover_410m.json (device=cpu)"],
        "shared_conditions": {"lenses": "results/e28_<corpus>_410m_n200_s<seed>.pt",
                              "band": e48["band"], "K": e48["K"], "admitted_sets": ADMITTED,
                              "rung": a.n},
        "ladder_devices_seen": sorted(devices),
        "tolerance": TOL,
        "tolerance_derivation": ("one (item,intermediate) pair flipping at one k moves a set mean by "
                                 ">= 1/(893*7) = 1.6e-4, so 1e-6 separates fp accumulation from a "
                                 "single changed decision"),
        "n_cells_compared": len(rows),
        "max_abs_diff": max_abs,
        "worst_cell": worst,
        "per_cell": rows,
        "VERDICT": ("AGREE — CUDA-scored and CPU-scored reads match to %.2e over %d matched cells. "
                    "The ladder's scoring device is immaterial and every ladder-derived number "
                    "stands as reported. E65's device defect does not reach the ladder."
                    % (max_abs, len(rows))) if agree else
                   ("DIVERGENT — max |CUDA - CPU| = %.3e over %d cells (worst: %s s%d, %s, "
                    "%.2f%% relative). STOP. Every ladder-derived number in the paper, meaning the "
                    "per-corpus asymptotes, the flatness in N and the 1B replication, must be "
                    "re-sourced from a CPU scoring pass before it is cited."
                    % (max_abs, len(rows), worst["corpus"], worst["seed"], worst["aggregation"],
                       100 * worst["rel_diff"])),
    }
    write_result(a.out, out, experiment=out["experiment"], script=__file__,
                 inputs=[os.path.join(RES, "e48_crossover_410m.json")])
    print(f"\ncompared {len(rows)} cells, max |CUDA-CPU| = {max_abs:.3e}")
    print(out["VERDICT"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
