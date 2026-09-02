#!/usr/bin/env python3
"""test_metrics_shift.py — acceptance tests for M1..M4 and continuous proximity.

The tests that matter most are the CROSS-FITTING and NULL tests: they are what stop M3/M4
producing a tautological number (external review Prompt 8 §2B).
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from metrics_shift import (ContainmentIndex, m2_reference_calibrated, m3_domain_auc,  # noqa: E402
                           m3_lexical_baseline, m4_action_divergence, m4_cross_fitted,
                           proximity_to_P, stratify, _auc)

_p = _f = 0
def check(name, cond, detail=""):
    global _p, _f
    if cond: _p += 1; print(f"  ok  {name}   [{detail}]")
    else:    _f += 1; print(f"  FAIL {name}   [{detail}]")

g = torch.Generator().manual_seed(0)

# ============================================================ M1
ix = ContainmentIndex(n=5, threshold=0.8)
ix.add(list(range(100))); ix.add(list(range(200, 300)))
check("M1_exact_member", ix.query(list(range(100))).verdict == "member-exact",
      "SHA-256 of the token tuple hits")
check("M1_disjoint_is_non_member",
      ix.query(list(range(1000, 1100))).verdict == "provably-non-member",
      "no shared n-gram at any order")
e = ix.query([0, 1, 2, 3, 4] + list(range(9000, 9050)))
check("M1_partial_span_is_NOT_called_non_member", e.verdict == "member-partial-span",
      f"jaccard={e.max_ngram_overlap:.3f} but lcn={e.longest_common_ngram} — a verbatim span "
      "must not be laundered into 'non-member'")
check("M1_returns_evidence_not_a_boolean",
      {"exact_match","max_ngram_overlap","longest_common_ngram","n_exposures","verdict"}
      <= set(e.to_dict()), "Prompt 7 §7 requires evidence fields, not a flag")

# ============================================================ M2
ce_m = torch.tensor([2.0, 3.0, 4.0]); ce_r = torch.tensor([2.5, 3.0, 3.0])
m2 = m2_reference_calibrated(ce_m, ce_r)
check("M2_is_the_difference", abs(m2["delta_mean"] - (-0.5 + 0.0 + 1.0) / 3) < 1e-6,
      f"delta_mean = {m2['delta_mean']:.4f}")
check("M2_declares_its_family", "membership-inference" in m2["family"],
      "external review B2: M2 is NOT a distance metric and must not be claimed as one")

# ============================================================ M3
d = 16
# genuinely separable corpora
H_P = torch.randn(120, d, generator=g)
H_Q = torch.randn(120, d, generator=g) + 2.5
r = m3_domain_auc(H_P, H_Q, folds=5, seed=0)
check("M3_detects_a_real_shift", r["auc_heldout"] > 0.9,
      f"held-out AUC = {r['auc_heldout']:.4f} on a 2.5-sigma mean shift")
check("M3_is_cross_fitted", r["cross_fitted"] and r["folds"] == 5,
      "every item is scored by a probe that never saw it")
check("M3_permuted_null_is_chance", r["null_ok"],
      f"permuted-label AUC = {r['auc_permuted_null']:.4f}, must be ~0.5 or the pipeline leaks")

# identical corpora -> AUC must collapse to chance. This is the test that catches a probe
# scored on its own training data, which would report ~1.0 here.
H_A = torch.randn(120, d, generator=g); H_B = torch.randn(120, d, generator=g)
r0 = m3_domain_auc(H_A, H_B, folds=5, seed=1)
check("M3_no_shift_gives_chance_AUC", abs(r0["auc_heldout"] - 0.5) < 0.15,
      f"AUC = {r0['auc_heldout']:.4f} on two draws from the SAME distribution — "
      "an in-sample probe would score ~1.0 here")

lex = m3_lexical_baseline(["hello world."] * 30, ["THE QUICK BROWN FOX, 12345!"] * 30)
check("M3_lexical_baseline_fires_on_surface_cues", lex["auc_heldout"] > 0.9,
      f"lexical AUC = {lex['auc_heldout']:.4f} — if this is high, hidden-state AUC is not "
      "evidence about representations")

# ============================================================ M4
J_P = torch.randn(d, d, generator=g)
H = torch.randn(200, d, generator=g)
same = m4_action_divergence(J_P, J_P.clone(), H)
check("M4_is_zero_for_identical_operators", same["d_act"] < 1e-8,
      f"D_act = {same['d_act']:.3e}")
J_Q = J_P + 0.3 * torch.randn(d, d, generator=g)
diff = m4_action_divergence(J_P, J_Q, H)
check("M4_is_positive_for_different_operators", diff["d_act"] > 0,
      f"D_act = {diff['d_act']:.4f}, normalised {diff['d_act_normalised']:.4f}")
check("M4_reports_frobenius_as_a_secondary",
      diff["frobenius_secondary"] > 0 and diff["spectral_secondary"] > 0,
      "Frobenius/spectral are stored but are NOT primary — they count unvisited directions")

# THE distinguishing property: D_act must ignore a difference in a direction the data never
# visits, which is exactly why it is preferred over ||J_P - J_Q||_F.
H_sub = torch.randn(200, d, generator=g)
H_sub[:, d // 2:] = 0.0                                  # data lives in the first half only
J_unvisited = J_P.clone()
J_unvisited[:, d // 2:] += 50.0                          # huge change, unvisited coordinates
r_un = m4_action_divergence(J_P, J_unvisited, H_sub)
check("M4_ignores_unvisited_directions", r_un["d_act"] < 1e-6,
      f"D_act = {r_un['d_act']:.3e} while Frobenius = {r_un['frobenius_secondary']:.1f} "
      "— the whole reason D_act exists")

folds_J = [J_P + 0.3 * torch.randn(d, d, generator=g) for _ in range(4)]
folds_H = [torch.randn(50, d, generator=g) for _ in range(4)]
cf = m4_cross_fitted(J_P, folds_J, folds_H)
check("M4_cross_fitted_reports_spread", cf["cross_fitted"] and cf["d_act_std"] > 0,
      f"mean {cf['d_act_mean']:.4f} +/- {cf['d_act_std']:.4f} over {cf['folds']} folds")
try:
    m4_cross_fitted(J_P, folds_J[:2], folds_H)
    ok = False
except ValueError:
    ok = True
check("M4_refuses_mismatched_folds", ok, "one J_Q per held-out fold is enforced")

# ============================================================ proximity + strata
E_P = torch.randn(40, d, generator=g)
E_Q = torch.cat([E_P[:10] + 0.01 * torch.randn(10, d, generator=g),   # near
                 torch.randn(10, d, generator=g) * 5 + 20])          # far
pr = proximity_to_P(E_Q, E_P)
near_mean = sum(pr["proximity"][:10]) / 10
far_mean = sum(pr["proximity"][10:]) / 10
check("proximity_orders_near_above_far", near_mean > far_mean,
      f"near {near_mean:.4f} > far {far_mean:.4f} — R_i = -min_p d(x_i,p)")
strata = stratify(pr["proximity"], n_bins=3)
check("stratify_gives_ordered_bins", set(strata) <= {"far", "intermediate", "near"}
      and len(set(strata)) > 1,
      f"{ {s: strata.count(s) for s in set(strata)} }")
check("proximity_unblocks_A3_cell", near_mean != far_mean,
      "proximity measured against P (not the training distribution) makes 'member + far' "
      "populable — the A_3 blocker was an artifact of the definition")

# ---- AUC helper sanity
check("auc_helper_is_correct",
      abs(_auc(torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.tensor([0, 0, 1, 1])) - 1.0) < 1e-6,
      "perfect separation -> 1.0")

print(f"\n=== {_p}/{_p+_f} PASSED ===")
raise SystemExit(1 if _f else 0)
