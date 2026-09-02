#!/usr/bin/env python3
"""Unit tests pinning the anchor §2.5 write. Run: .venv/bin/python tests/test_joperator.py

Every test corresponds to a sentence in the anchor's specification or to a discipline in
PLAN.md §3. These are the assertions that would have caught the previous program's errors.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from joperator import (  # noqa: E402
    IdentityHooks, control_orth, control_rand_target, coords,
    make_swap_basis, patch_swap, swap_delta, token_direction,
)

D = 64
PASSED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"{name} FAILED  {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def pair(seed: int = 0, collinear: float = 0.0):
    g = torch.Generator().manual_seed(seed)
    v_s = torch.randn(D, generator=g)
    v_t = torch.randn(D, generator=g)
    if collinear:  # push v_t toward v_s
        v_t = collinear * v_s + (1 - collinear) * v_t
    h = torch.randn(D, generator=g) * 3.0
    return v_s, v_t, h


def main() -> int:
    print("=== anchor §2.5 write — specification tests ===")

    # 1. V† is exactly the pseudoinverse -------------------------------------
    v_s, v_t, h = pair(0)
    b = make_swap_basis(v_s, v_t)
    check("gram_solve_equals_pinv",
          torch.allclose(b.Vpinv, torch.linalg.pinv(b.V), atol=1e-4),
          f"max_diff={float((b.Vpinv - torch.linalg.pinv(b.V)).abs().max()):.2e}")

    # 2. σ is a PERMUTATION, and at α=1 the write is rank-1 along (v_t − v_s) --
    delta = swap_delta(h, b, 1.0)
    payload = v_t - v_s
    cos = torch.nn.functional.cosine_similarity(delta, payload, dim=0)
    check("alpha1_write_is_rank1_along_vt_minus_vs", abs(abs(float(cos)) - 1.0) < 1e-5,
          f"|cos|={abs(float(cos)):.8f}")

    c = coords(h, b)
    expected = (c[0] - c[1]) * payload
    check("alpha1_scalar_is_cs_minus_ct",
          torch.allclose(delta, expected, atol=1e-4),
          f"c=({float(c[0]):.3f},{float(c[1]):.3f})")

    # 3. sigma is NOT softmax — an explicit contrast ---------------------------
    sm = torch.softmax(c, 0)
    delta_softmax = (sm - c) @ b.V.T
    check("softmax_would_be_a_different_operator",
          not torch.allclose(delta, delta_softmax, atol=1e-3),
          f"||diff||={float((delta - delta_softmax).norm()):.3f}")

    # 4. orthogonal component is unchanged -------------------------------------
    h_new, _ = patch_swap(h, b, 1.0)
    Q, _ = torch.linalg.qr(b.V)                     # orthonormal basis of span{v_s,v_t}
    perp = lambda x: x - Q @ (Q.T @ x)
    check("perp_component_unchanged",
          torch.allclose(perp(h), perp(h_new.float()), atol=1e-4),
          f"max_diff={float((perp(h) - perp(h_new.float())).abs().max()):.2e}")

    # 5. the coordinates are exactly exchanged ---------------------------------
    c_post = coords(h_new.float(), b)
    check("coordinates_exactly_exchanged",
          torch.allclose(c_post, torch.stack([c[1], c[0]]), atol=1e-3),
          f"pre=({float(c[0]):.3f},{float(c[1]):.3f}) post=({float(c_post[0]):.3f},{float(c_post[1]):.3f})")

    # 6. alpha scales the WHOLE update: rank-1 at every alpha, linear in alpha ---
    #    (pins the 2026-08-10 fix; the old code put alpha inside sigma, breaking both)
    d1 = swap_delta(h, b, 1.0)
    for al in (0.5, 2.0, 4.0):
        da = swap_delta(h, b, al)
        cosa = abs(float(torch.nn.functional.cosine_similarity(da, payload, dim=0)))
        check(f"alpha{al:g}_still_rank1_along_payload", abs(cosa - 1.0) < 1e-5, f"|cos|={cosa:.8f}")
        check(f"alpha{al:g}_scales_linearly", torch.allclose(da, al * d1, atol=1e-4),
              f"||d({al})-{al}*d(1)||={float((da - al * d1).norm()):.2e}")

    # 7. the 1/(1-rho) gain, on unit vectors -----------------------------------
    u_s = v_s / v_s.norm()
    u_t = v_t / v_t.norm()
    bu = make_swap_basis(u_s, u_t)
    cu = coords(h, bu)
    rho = float(u_s @ u_t)
    a, bb = float(u_s @ h), float(u_t @ h)
    check("coefficient_equals_(a-b)/(1-rho)",
          abs(float(cu[0] - cu[1]) - (a - bb) / (1 - rho)) < 1e-3,
          f"rho={rho:.4f} gain=1/(1-rho)={1/(1-rho):.3f}")

    # 8. the gain blows up as the pair becomes collinear -----------------------
    _, v_t_c, h_c = pair(1, collinear=0.98)
    v_s_c = pair(1)[0]
    bc = make_swap_basis(v_s_c / v_s_c.norm(), v_t_c / v_t_c.norm())
    cc = coords(h_c, bc)
    rho_c = float((v_s_c / v_s_c.norm()) @ (v_t_c / v_t_c.norm()))
    check("gain_grows_with_collinearity", 1 / (1 - rho_c) > 1 / (1 - rho),
          f"rho {rho:.3f}->{rho_c:.3f}, gain {1/(1-rho):.2f}->{1/(1-rho_c):.2f}")

    # 9. matched-dose rescaling hits the requested norm EXACTLY ----------------
    H = torch.randn(3, 5, D)
    target = torch.full((3, 5), 7.5)
    _, dn = patch_swap(H, b, 1.0, match_delta_norm=target)
    check("matched_dose_is_exact", torch.allclose(dn, target, atol=1e-3),
          f"max_abs_err={float((dn - target).abs().max()):.2e}")

    # 9b. match_delta_norm rejects a mismatched shape instead of broadcasting ---
    try:
        patch_swap(torch.randn(3, 5, D), b, 1.0, match_delta_norm=torch.full((3, 4), 1.0))
        check("match_delta_norm_shape_guard", False, "no error raised")
    except ValueError:
        check("match_delta_norm_shape_guard", True, "mismatched dose shape raises")

    # 10. controls: rand_target preserves the payload norm ---------------------
    v_r = control_rand_target(v_s, v_t, seed=1234)
    check("rand_target_preserves_payload_norm",
          abs(float((v_r - v_s).norm() - (v_t - v_s).norm())) < 1e-3,
          f"{float((v_r - v_s).norm()):.4f} vs {float((v_t - v_s).norm()):.4f}")
    # the CRITICAL bug: different seeds must give DIFFERENT payload directions
    pays = []
    vbig_s, vbig_t = v_s * 25, v_t * 25          # realistic ||v|| ~ 50, where the bug appeared
    for sd in (1, 2, 3, 4):
        vr = control_rand_target(vbig_s, vbig_t, sd)
        pays.append((vr - vbig_s) / (vr - vbig_s).norm())
    import itertools as _it
    worst = max(abs(float(torch.dot(x, y))) for x, y in _it.combinations(pays, 2))
    check("rand_target_is_actually_random_across_seeds", worst < 0.30,
          f"worst pairwise |cos| between seeds = {worst:.4f} (was 0.9996 before the fix)")
    neg = abs(float(torch.dot(pays[0], -vbig_s / vbig_s.norm())))
    check("rand_target_is_not_just_minus_v_s", neg < 0.30,
          f"|cos(payload, -v_s_hat)| = {neg:.4f} (was 0.9998 before the fix)")

    # 11. controls: orth is orthogonal to the real payload, same norm ----------
    v_o = control_orth(v_s, v_t, seed=1234)
    p_o, p_r = v_o - v_s, v_t - v_s
    check("orth_payload_is_orthogonal", abs(float(p_o @ p_r)) < 1e-3,
          f"dot={float(p_o @ p_r):.2e}")
    check("orth_preserves_payload_norm",
          abs(float(p_o.norm() - p_r.norm())) < 1e-3,
          f"{float(p_o.norm()):.4f} vs {float(p_r.norm()):.4f}")

    # 12. token_direction is the raw, uncentred J^T W_U[t] ---------------------
    J = torch.randn(D, D)
    wu = torch.randn(D)
    check("token_direction_is_JT_WU", torch.allclose(token_direction(J, wu), J.T @ wu, atol=1e-5))

    # 13. degenerate basis is flagged, not silently wrong ----------------------
    bd = make_swap_basis(v_s, v_s.clone())
    check("degenerate_pair_flagged", bd.degenerate, f"cond={bd.cond:.3e}")

    # 14. batched shapes round-trip -------------------------------------------
    Hb = torch.randn(2, 7, D)
    out, dnb = patch_swap(Hb, b, 1.0)
    check("batched_shapes", out.shape == Hb.shape and dnb.shape == Hb.shape[:-1],
          f"{tuple(out.shape)}, {tuple(dnb.shape)}")

    # 15. dtype is preserved ---------------------------------------------------
    Hh = torch.randn(4, D, dtype=torch.float16)
    outh, _ = patch_swap(Hh, b, 1.0)
    check("dtype_preserved", outh.dtype == torch.float16)

    # 16. identity hooks are a true no-op --------------------------------------
    lin = torch.nn.Linear(D, D)
    blocks = torch.nn.ModuleList([lin])
    x = torch.randn(1, 3, D)
    with torch.no_grad():
        base = lin(x).clone()
        with IdentityHooks(blocks, [0]):
            got = blocks[0](x)
    check("identity_hooks_are_noop", torch.allclose(base, got, atol=1e-6))

    print(f"\n=== {len(PASSED)}/{len(PASSED)} PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
