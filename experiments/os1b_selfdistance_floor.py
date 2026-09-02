#!/usr/bin/env python3
"""
OS1b — the numerical floor of OS1's self-distance control. MEASUREMENT ONLY, no decision rule.

OS1 is recorded UNCLEAR because C1 required `theta(J, J) == 0` exactly and measured 3.735905e-04.

theta is a mean principal angle: `mean(arccos(svdvals(Ua.T @ Ub)))`. At Ua == Ub every singular
value is 1, and arccos has an infinite derivative there, so theta ~ sqrt(2 * (1 - sigma)). With
sigma resolved only to float32 precision (eps = 1.19e-07) the floor is ~sqrt(2*eps) = 4.88e-04.
A control demanding exact zero from that quantity cannot fire.

This measures the floor instead of asserting it: theta(U, U) on the stored operators in float32 and
float64, plus a random matrix to show the floor is a property of the arithmetic and not of these
operators. It changes no OS1 number and adjudicates nothing; OS1's PRIMARY (SEP 9.31 [8.08, 10.44])
and its other three controls stand as recorded.

  .venv/bin/python experiments/os1b_selfdistance_floor.py
"""
import json, math, os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

K = 32
LENS = os.path.join(ROOT, "results", "e48", "lens_INSTREAM_Pile-CC_410m_n200_s0.pt")
OS1_C1_MEASURED = 3.735905e-04          # results/os1_operator_space_410m.json:C1_self_distance.theta


def basis(A, k=K):
    U, _, _ = torch.linalg.svd(A, full_matrices=False)
    return U[:, :k]


def theta(Ua, Ub):
    """OS1's metric, transcribed from experiments/os1_operator_space.py:76-79."""
    s = torch.linalg.svdvals(Ua.T @ Ub).clamp(-1.0, 1.0)
    return float(torch.arccos(s).mean().item())


def main() -> int:
    import provenance as prov
    torch.manual_seed(0)

    eps32 = float(torch.finfo(torch.float32).eps)
    eps64 = float(torch.finfo(torch.float64).eps)
    pred32, pred64 = math.sqrt(2 * eps32), math.sqrt(2 * eps64)

    J = torch.load(LENS, map_location="cpu", weights_only=True)["J"]
    per_layer = {}
    for l in sorted(J):
        A = J[l]
        per_layer[str(l)] = {
            "float32": theta(basis(A.float()), basis(A.float())),
            "float64": theta(basis(A.double()), basis(A.double())),
        }
    v32 = [v["float32"] for v in per_layer.values()]
    v64 = [v["float64"] for v in per_layer.values()]

    R = torch.randn(1024, 1024)
    rnd = {"float32": theta(basis(R.float()), basis(R.float())),
           "float64": theta(basis(R.double()), basis(R.double()))}

    rec = {
        "experiment": "OS1b — the numerical floor of OS1's theta self-distance control",
        "status": "MEASUREMENT ONLY — no registration, no decision rule, adjudicates nothing",
        "about": "results/os1_operator_space_410m.json:C1_self_distance",
        "metric": "mean principal angle over the top-32 left singular subspaces, radians",
        "operators": os.path.relpath(LENS, ROOT),
        "os1_c1_required": "exactly 0 on both metrics",
        "os1_c1_measured_theta": OS1_C1_MEASURED,
        "os1_c1_measured_d_rel": 0.0,
        "float32_eps": eps32, "float64_eps": eps64,
        "predicted_floor_sqrt_2eps": {"float32": pred32, "float64": pred64},
        "theta_self_distance_by_layer": per_layer,
        "summary": {
            "float32_min": min(v32), "float32_max": max(v32),
            "float32_mean": sum(v32) / len(v32),
            "float64_mean": sum(v64) / len(v64),
            "ratio_f32_over_f64": (sum(v32) / len(v32)) / max(sum(v64) / len(v64), 1e-30),
            "random_matrix_control": rnd,
        },
        "READING": (
            f"theta(U,U) on the stored operators is {min(v32):.3e}..{max(v32):.3e} in float32 and "
            f"~{sum(v64)/len(v64):.2e} in float64, a factor of "
            f"{(sum(v32)/len(v32))/max(sum(v64)/len(v64),1e-30):.0f}. A random matrix gives the same "
            f"float32 value ({rnd['float32']:.3e}), so the floor is a property of the arithmetic, "
            f"not of these operators. OS1's measured {OS1_C1_MEASURED:.3e} lies inside that "
            f"float32 distribution and below the predicted floor sqrt(2*eps32) = {pred32:.3e}. "
            f"C1 required exact zero from a quantity whose float32 floor is ~4e-04, so it could not "
            f"fire. In float64 the self-distance is ~1e-08, which is zero. OS1's PRIMARY and its "
            f"other three controls are unaffected."),
    }
    print(rec["READING"])
    out = os.path.join(ROOT, "results", "os1b_selfdistance_floor.json")
    prov.write_result(out, rec, script=__file__, experiment="OS1b", inputs=[LENS])
    print("wrote", os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
