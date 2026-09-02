#!/usr/bin/env python3
"""Unit tests for the Jacobian dispersion accumulator (THESIS.md C0)."""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fastfit import jacobian_dispersion  # noqa: E402

P = []
def check(n, c, d=""):
    if not c: raise AssertionError(f"{n} FAILED {d}")
    P.append(n); print(f"  ok  {n}" + (f"   [{d}]" if d else ""))

def build(mats):
    """Emulate the accumulator over a list of per-prompt Jacobians."""
    layers = [0]
    jac_sum = {0: sum(mats)}
    sumsq = {0: sum(float(m.norm()) ** 2 for m in mats)}
    mx = {0: max(float(m.norm()) for m in mats)}
    return jacobian_dispersion(jac_sum, sumsq, mx, len(mats), layers)[0]

def main():
    print("=== jacobian dispersion ===")
    torch.manual_seed(0)
    A = torch.randn(8, 8)

    # 1. identical Jacobians -> zero dispersion
    r = build([A, A, A, A])
    check("identical_prompts_give_zero_dispersion", abs(r["dispersion"]) < 1e-5,
          f"disp={r['dispersion']:.2e}")

    # 2. the algebraic identity E||J-Jbar||^2 = E||J||^2 - ||Jbar||^2
    mats = [torch.randn(8, 8) for _ in range(7)]
    r = build(mats)
    Jbar = sum(mats) / len(mats)
    direct = sum(float((m - Jbar).norm()) ** 2 for m in mats) / len(mats)
    expect = direct / float(Jbar.norm()) ** 2
    check("matches_direct_computation", abs(r["dispersion"] - expect) < 1e-3,
          f"acc={r['dispersion']:.4f} direct={expect:.4f}")

    # 3. dispersion is scale-invariant (it is normalized)
    r1 = build(mats); r2 = build([10 * m for m in mats])
    check("scale_invariant", abs(r1["dispersion"] - r2["dispersion"]) < 1e-3,
          f"{r1['dispersion']:.4f} vs {r2['dispersion']:.4f}")

    # 4. zero-mean ensemble -> dispersion blows up (mean is pure noise)
    r = build([A, -A])
    check("cancelling_mean_gives_huge_dispersion", r["dispersion"] > 1e3 or r["dispersion"] == float("inf"),
          f"disp={r['dispersion']:.3g}")

    # 5. heavy-tail witness fires on one outlier prompt
    r = build([A, A, A, 50 * A])
    check("heavy_tail_ratio_detects_outlier", r["heavy_tail_ratio"] > 1.5,
          f"ratio={r['heavy_tail_ratio']:.3f}")
    r = build([A, A, A, A])
    check("heavy_tail_ratio_near_1_when_uniform", abs(r["heavy_tail_ratio"] - 1.0) < 1e-3,
          f"ratio={r['heavy_tail_ratio']:.4f}")

    print(f"\n=== {len(P)}/{len(P)} PASSED ===")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
