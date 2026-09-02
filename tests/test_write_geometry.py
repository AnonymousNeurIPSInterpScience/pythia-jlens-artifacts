#!/usr/bin/env python3
"""test_write_geometry.py — acceptance tests for the Prompt-9 conditioning controls.

Each test names the Prompt 9 claim it enforces. These are the guard rails that stop a
conditioning artifact being reported as a lens result.
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from write_geometry import (unit, gram_stats, DoseMetric, fit_second_moment, equal_dose,  # noqa: E402
                            swap_delta_normalized, swap_delta_orthogonalized,
                            rank1_contrast_delta, ablation_energy, geometry_record)

_p = _f = 0
def check(name, cond, detail=""):
    global _p, _f
    if cond: _p += 1; print(f"  ok  {name}   [{detail}]")
    else:    _f += 1; print(f"  FAIL {name}   [{detail}]")

g = torch.Generator().manual_seed(0)
d = 64
h = torch.randn(3, 5, d, generator=g)

# ---- unit normalisation (Prompt 9 §2A: mandatory)
v = torch.randn(d, generator=g) * 7.3
check("unit_gives_norm_one", abs(float(unit(v).norm()) - 1.0) < 1e-6,
      f"||unit(v)|| = {float(unit(v).norm()):.8f}")

# ---- Prompt 9 §1 closed forms for kappa
for target_rho in (0.0, 0.5, 0.9, 0.99):
    a = unit(torch.randn(d, generator=g))
    perp = torch.randn(d, generator=g); perp = unit(perp - (perp @ a) * a)
    b = unit(target_rho * a + (1 - target_rho ** 2) ** 0.5 * perp)
    rho, kV, kG = gram_stats(a, b)
    exp_kG = (1 + abs(target_rho)) / (1 - abs(target_rho))
    check(f"kappa_closed_form_rho{target_rho}",
          abs(rho - target_rho) < 1e-4 and abs(kG - exp_kG) / exp_kG < 1e-3,
          f"rho={rho:.4f} kappa_G={kG:.3f} expected {exp_kG:.3f}, kappa_V={kV:.3f}")

# ---- SCALE INVARIANCE: the whole point of mandatory unit normalisation.
a = torch.randn(d, generator=g); b = torch.randn(d, generator=g)
d1, _ = swap_delta_normalized(h, a, b)
d2, _ = swap_delta_normalized(h, a * 13.0, b * 0.07)      # same span, wildly different scales
check("swap_is_scale_invariant_after_unit_norm", torch.allclose(d1, d2, atol=1e-5),
      f"max|diff| = {float((d1-d2).abs().max()):.2e} — an UNnormalised swap would differ here")

# ---- the native swap is rank-1 in the (v_s, v_t) span and linear in alpha
dA, _ = swap_delta_normalized(h, a, b, alpha=1.0)
dB, _ = swap_delta_normalized(h, a, b, alpha=2.0)
check("swap_linear_in_alpha", torch.allclose(dB, 2 * dA, atol=1e-5),
      f"max|d(2)-2d(1)| = {float((dB-2*dA).abs().max()):.2e}")
ua, ub = unit(a), unit(b)
Vspan = torch.stack([ua, ub], dim=1)
proj = dA @ Vspan @ torch.linalg.pinv(Vspan)
check("swap_lives_in_the_span", torch.allclose(proj, dA, atol=1e-4),
      "Delta has no component outside span{v_s, v_t}")

# ---- orthogonalised arm: basis IS orthonormal, and the discarded component is reported
_, meta = swap_delta_orthogonalized(h, a, b)
rho, _, _ = gram_stats(a, b)
check("orth_reports_discarded_component", abs(meta["discarded_frac"] - abs(rho)) < 1e-6,
      f"discarded_frac = {meta['discarded_frac']:.4f} == |rho| = {abs(rho):.4f} (Prompt 9 §2B)")

# ---- orthogonalisation genuinely CHANGES the intervention (so it is a control, not a fix)
dO, _ = swap_delta_orthogonalized(h, a, b)
check("orth_differs_from_native", not torch.allclose(dO, dA, atol=1e-3),
      f"max|orth - native| = {float((dO-dA).abs().max()):.4f} — it is NOT a neutral normalisation")

# ---- collinear pair: native swap blows up, orthogonalised arm degrades gracefully
near = unit(ua + 1e-4 * unit(torch.randn(d, generator=g)))
dN, mN = swap_delta_normalized(h, ua, near)
dOc, mOc = swap_delta_orthogonalized(h, ua, near)
check("collinear_pair_is_flagged", mN["kappa_G"] > 1e3,
      f"kappa_G = {mN['kappa_G']:.3e} at rho = {mN['rho']:.6f}")
check("collinear_native_swap_is_large", float(dN.norm()) > float(dOc.norm()),
      f"||native|| = {float(dN.norm()):.3e} vs ||orth|| = {float(dOc.norm()):.3e} "
      "— this is the conditioning confound, visible")

# ---- rank-1 arm needs NO pseudoinverse and is unit under its metric
euc = DoseMetric("euclidean")
dR, mR = rank1_contrast_delta(h, a, b, alpha=1.0, metric=euc)
check("rank1_has_unit_dose", abs(float(euc(dR).mean()) - 1.0) < 1e-5,
      f"||Delta||_2 = {float(euc(dR).mean()):.6f} at alpha=1")
check("rank1_direction_is_the_contrast",
      abs(abs(float(torch.nn.functional.cosine_similarity(dR[0, 0], unit(b) - unit(a), dim=0))) - 1) < 1e-5,
      "|cos(Delta, v_t_hat - v_s_hat)| = 1")

# ---- equal-dose rescaling: the PRIMARY cross-lens comparison (Prompt 9 §2C)
S_inv = fit_second_moment(h, center=False, shrinkage=1e-2)
whit = DoseMetric("whitened", S_inv)
for name, m in (("euclidean", euc), ("whitened", whit)):
    matched = equal_dose(dA, 0.5, m)
    got = float(m(matched).mean())
    check(f"equal_dose_hits_target_{name}", abs(got - 0.5) < 1e-4,
          f"achieved ||Delta||_M = {got:.6f}, target 0.5")
check("equal_dose_preserves_direction",
      torch.allclose(torch.nn.functional.normalize(equal_dose(dA, 0.5, euc), dim=-1),
                     torch.nn.functional.normalize(dA, dim=-1), atol=1e-5),
      "rescaling changes magnitude only")

# ---- whitened dose is NOT the euclidean dose (else the control is vacuous)
check("whitened_dose_differs_from_euclidean",
      abs(float(whit(dA).mean()) - float(euc(dA).mean())) > 1e-6,
      f"whitened {float(whit(dA).mean()):.4f} vs euclidean {float(euc(dA).mean()):.4f}")

# ---- no centering is performed unless explicitly asked (repo discipline 1)
Hs = h + 100.0                                   # a huge mean offset
S_unc_a = fit_second_moment(h, center=False, shrinkage=1e-2)
S_unc_b = fit_second_moment(Hs, center=False, shrinkage=1e-2)
S_cen_a = fit_second_moment(h, center=True, shrinkage=1e-2)
S_cen_b = fit_second_moment(Hs, center=True, shrinkage=1e-2)
check("uncentered_metric_sees_the_mean", not torch.allclose(S_unc_a, S_unc_b, atol=1e-3),
      "E[hh^T] shifts with the mean — as it must, since no mean is formed")
# compare on a RELATIVE scale: these are inverses of a rank-deficient matrix (N=15 < d=64),
# so absolute entries are large and an atol comparison is meaningless.
rel = float((S_cen_a - S_cen_b).norm() / S_cen_a.norm().clamp_min(1e-12))
check("centered_metric_is_translation_invariant", rel < 1e-4,
      f"relative diff = {rel:.2e} — the center=True path is the LEACE assumption, "
      "offered but not default")

# ---- ablation energy: two UNIT directions remove different energy (Prompt 9 §5)
S = (h.reshape(-1, d).T @ h.reshape(-1, d)) / (h.shape[0] * h.shape[1])
evals, evecs = torch.linalg.eigh(S)
hi_dir, lo_dir = evecs[:, -1], evecs[:, 0]
e_hi, e_lo = ablation_energy(hi_dir, S), ablation_energy(lo_dir, S)
check("ablation_energy_depends_on_alignment",
      e_hi["removed_variance"] > 10 * e_lo["removed_variance"],
      f"top-eigvec removes var {e_hi['removed_variance']:.4f} vs bottom "
      f"{e_lo['removed_variance']:.6f} — ablation-KL is NOT lens-agnostic")
# ERRATUM GUARD. Prompt 9 §5 gives removed energy as v'S^2v/(v'v)^2, which is NOT scale
# invariant and does not match Monte Carlo. The projection ablation IS invariant to ||v||, so
# its removed energy must be too. This test pins the corrected formula.
e_scaled = ablation_energy(hi_dir * 5.0, S)
check("ablation_energy_is_scale_invariant",
      abs(e_scaled["removed_energy"] - e_hi["removed_energy"]) / e_hi["removed_energy"] < 1e-5,
      f"E||h-h_abl||^2 = {e_hi['removed_energy']:.5f} at ||v||=1 and "
      f"{e_scaled['removed_energy']:.5f} at ||v||=5 — projection ablation is scale-free")
check("prompt9_erratum_is_recorded_and_is_NOT_scale_invariant",
      abs(e_scaled["removed_energy_prompt9_erratum"]
          - e_hi["removed_energy_prompt9_erratum"]) > 1e-3,
      f"Prompt 9's v'S^2v/(v'v)^2 gives {e_hi['removed_energy_prompt9_erratum']:.5f} vs "
      f"{e_scaled['removed_energy_prompt9_erratum']:.5f} under a pure rescale — the tell")
check("coordinate_variance_matches_prompt9_verbatim",
      abs(e_hi["removed_variance"] - float(hi_dir @ (S.float() @ hi_dir)) / float(hi_dir @ hi_dir) ** 2) < 1e-9,
      "Var(v'h/v'v) = v'Sv/(v'v)^2 — Prompt 9's second formula IS correct")

# ---- the reporting row carries every Prompt-9 field
rec = geometry_record("jlens", a, b, dA, euc, whit)
need = {"lens","norm_v_s","norm_v_t","rho","kappa_V","kappa_G","discarded_frac",
        "dose_euclidean","dose_whitened","degenerate"}
check("geometry_record_is_complete", need <= set(rec.to_dict()),
      f"{len(need)} required fields present")

# ---- SYMBOLIC DERIVATION of the ablation identity (NOT Monte Carlo).
# Monte Carlo can only show two formulas disagree; it cannot show WHICH is right without
# trusting the simulator. A cold reproducibility review that re-runs a wrong-but-consistent
# test will happily confirm the wrong number. So the identity is PROVED here.
try:
    import sympy as sp
    d3 = 3
    hs = sp.Matrix(sp.symbols("h1:4", real=True))
    vs = sp.Matrix(sp.symbols("v1:4", real=True))
    S3 = sp.Matrix(d3, d3, lambda i, j: sp.Symbol(f"s{min(i,j)}{max(i,j)}", real=True))
    vtv3 = (vs.T * vs)[0]
    removed = ((vs.T * hs)[0] / vtv3) * vs                 # h - h_abl
    lhs = sp.simplify((removed.T * removed)[0])
    rhs = (vs.T * hs)[0] ** 2 / vtv3
    check("ERRATUM_step1_algebraic_identity", sp.simplify(lhs - rhs) == 0,
          "||h - h_abl||^2 = (v'h)^2/(v'v) exactly, for all v and h")

    ours_sym = sp.simplify((vs.T * S3 * vs)[0] / vtv3)
    p9_sym = sp.simplify((vs.T * S3 * S3 * vs)[0] / vtv3 ** 2)
    check("ERRATUM_step2_formulas_are_not_equal", sp.simplify(ours_sym - p9_sym) != 0,
          "v'Sv/(v'v) is not v'S^2v/(v'v)^2")

    aa = sp.Symbol("a", positive=True)
    sub = {vs[i]: aa * vs[i] for i in range(d3)}
    check("ERRATUM_step3_ours_is_scale_invariant",
          sp.simplify(ours_sym.subs(sub, simultaneous=True) - ours_sym) == 0,
          "the projection ablation is invariant to ||v||, so its removed energy must be too")
    check("ERRATUM_step4_prompt9_is_NOT_scale_invariant",
          sp.simplify(p9_sym.subs(sub, simultaneous=True) / p9_sym) == aa ** -2,
          "Prompt 9's form picks up a factor a^-2 under v -> a*v — the structural tell")

    P3 = vs * vs.T / vtv3
    check("ERRATUM_step5_equals_trace_of_projector_times_Sigma",
          sp.simplify(sp.trace(P3 * S3) - ours_sym) == 0,
          "E||Ph||^2 = tr(P S) = v'Sv/(v'v), the standard projector identity")
except ImportError:
    check("ERRATUM_symbolic_derivation", False, "sympy missing — the proof cannot run")

print(f"\n=== {_p}/{_p+_f} PASSED ===")
raise SystemExit(1 if _f else 0)
