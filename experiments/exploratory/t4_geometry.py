#!/usr/bin/env python3
"""t4_geometry.py — Claim 2 predictors, computed from the fitted J alone.

Free once a lens exists: no forward passes, no model weights except W_U for the
dictionary-anisotropy term. Every predictor is pre-declared here (PLAN.md §3.8 caps the
family at five for Holm correction) so T6 cannot shop for one after seeing outcomes.

PREDICTORS (the five that enter the regression):
  P1  stable_rank        ||J||_F^2 / sigma_1^2     -- how many directions J really uses
  P2  effective_rank     exp(entropy of the singular-value ENERGY distribution),
                         p_k = sigma_k^2 / sum_j sigma_j^2  (the code has always used squared
                         singular values; the docstring previously said unsquared)
  P3  [REMOVED 2026-08-10] top_sv_share = sigma_1^2 / sum sigma_i^2 is the EXACT RECIPROCAL of
      P1_stable_rank, so the "five predictors" were really four. Retained in the output as a
      derived diagnostic but EXCLUDED from the pre-registered predictor family and from Holm.
  P4  dist_from_identity ||J - I||_F / ||J||_F     -- how far transport is from a no-op
  P5  top_u_participation  PR of J's top LEFT-singular vector, normalised to [1/d, 1]
      *** This is the rogue-coordinate / tied-vs-untied test. ***
      At Gemma-2 (TIED embeddings) the J-lens dictionary was dominated by one residual
      coordinate: |u1|max 0.92-0.98, PR 1.2-1.4 out of 2304, and zeroing that single row
      collapsed the 91-concept cone 0.875 -> 0.123 against an untransported baseline of
      0.121. Pythia is UNTIED. Whether the domination survives is a clean, cheap, novel test.

DIAGNOSTICS (reported, not regressed):
  raw dictionary anisotropy: mean pairwise |cos| over held-out tokens, transported. RAW --
  never centered (mu is the artifact that ended the previous program).
"""
from __future__ import annotations

import argparse, json, os, sys, time

import torch

torch.set_num_threads(min(8, os.cpu_count() or 8))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))

from jlens.hf import Layout, from_hf  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")


def participation_ratio(v: torch.Tensor) -> float:
    """PR = (sum v_i^2)^2 / sum v_i^4, in [1, d]: 1 for one-hot, d for uniform.

    (The docstring previously claimed the range was [1/d, 1]; it is not normalised.)
    """
    v = v.float()
    denom = float((v ** 4).sum())
    if denom <= 0:
        return float("nan")
    return float((v ** 2).sum() ** 2 / denom)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--lens", required=True)
    ap.add_argument("--n-tokens", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--svd-device", default=None,
                    help="device for SVD; CPU MKL thrashes on high-core boxes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    try:
        model = from_hf(hf, tok)
    except ValueError:
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    lens = JacobianLens.load(a.lens)

    WU = model._lm_head.weight.detach().float()
    g = torch.Generator().manual_seed(a.seed)
    # sample WITHOUT replacement: with replacement ~20 of 1500 rows duplicated, each
    # contributing an off-diagonal cosine of exactly 1.0 to the anisotropy estimate.
    hi = min(WU.shape[0], 50000)
    tok_ids = 1000 + torch.randperm(hi - 1000, generator=g)[:a.n_tokens]

    rows = []
    for l in lens.source_layers:
        J = lens.jacobians[l].float()
        d = J.shape[0]
        _sd = a.svd_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        U, S, _ = torch.linalg.svd(J.to(_sd), full_matrices=False)
        U, S = U.cpu(), S.cpu()
        s2 = S ** 2
        p = s2 / s2.sum()
        eff_rank = float(torch.exp(-(p * (p + 1e-30).log()).sum()))

        # transported dictionary, RAW (uncentered), on held-out tokens
        V = WU[tok_ids] @ J                                    # [n, d] rows are v_t^T
        Vn = torch.nn.functional.normalize(V, dim=1)
        C = Vn @ Vn.T
        iu = torch.triu_indices(len(tok_ids), len(tok_ids), offset=1)
        offdiag = C[iu[0], iu[1]]

        # untransported baseline: same tokens, no J
        Wn = torch.nn.functional.normalize(WU[tok_ids], dim=1)
        Cw = Wn @ Wn.T
        base = Cw[iu[0], iu[1]]

        # Figure-28d object: the anchor plots effective dimensionality of W_U J_l, NOT of J_l.
        # We were reporting only rank statistics of J_l and calling it comparable. Compute the
        # real thing (on the held-out token sample, which is what makes it tractable).
        S_wj = torch.linalg.svdvals(V.to(_sd)).cpu()          # V = WU[tok_ids] @ J
        p_wj = (S_wj ** 2) / (S_wj ** 2).sum()
        effdim_wuj = float(torch.exp(-(p_wj * (p_wj + 1e-30).log()).sum()))
        u1 = U[:, 0]
        rows.append({
            "layer": l,
            "depth_pct": round(100 * (l + 1) / model.n_layers, 1),  # blocks record OUTPUTS
            "d_model": d,
            # --- the five pre-declared predictors ---
            "P1_stable_rank": round(float(s2.sum() / s2[0]), 4),
            "P2_effective_rank": round(eff_rank, 4),
            # derived diagnostic only: exactly 1 / P1_stable_rank, NOT an independent predictor
            "D_top_sv_share_reciprocal_of_P1": round(float(p[0]), 6),
            "P4_dist_from_identity": round(
                float((J - torch.eye(d)).norm() / J.norm()), 4),
            "P5_top_u_participation": round(participation_ratio(u1), 4),
            "P5_top_u_max_abs": round(float(u1.abs().max()), 4),
            "P5_top_u_argmax_dim": int(u1.abs().argmax()),
            # --- diagnostics ---
            # the ACTUAL Figure-28d quantity, reported separately from J_l's own rank
            "FIG28D_effective_dim_of_WU_J": round(effdim_wuj, 4),
            "FIG28D_effective_dim_frac_of_d": round(effdim_wuj / d, 6),
            "fro_norm": round(float(J.norm()), 4),
            "top_sv": round(float(S[0]), 4),
            "dict_anisotropy_raw_meanabscos": round(float(offdiag.abs().mean()), 4),
            "dict_anisotropy_raw_median": round(float(offdiag.median()), 4),
            "untransported_baseline_meanabscos": round(float(base.abs().mean()), 4),
            "transport_amplification": round(
                float(offdiag.abs().mean() - base.abs().mean()), 4),
        })
        print(f"  L{l:<3d} srank={rows[-1]['P1_stable_rank']:8.2f}/{d}  "
              f"effrank={rows[-1]['P2_effective_rank']:8.2f}  "
              f"PR(u1)={rows[-1]['P5_top_u_participation']:7.2f}  "
              f"|u1|max={rows[-1]['P5_top_u_max_abs']:.3f}  "
              f"anis={rows[-1]['dict_anisotropy_raw_meanabscos']:.3f} "
              f"(base {rows[-1]['untransported_baseline_meanabscos']:.3f})")

    prs = [r["P5_top_u_participation"] for r in rows]
    out = {
        "experiment": "T4 geometry — Claim 2 predictors from J alone",
        "status": "EXPLORATORY unless a committed prereg is cited here",
        "prereg": None,
        "model": a.model, "lens": a.lens,
        "lens_n_prompts": lens.n_prompts, "d_model": lens.d_model,
        "n_heldout_tokens": a.n_tokens,
        "centered": False,
        "tied_embeddings": False,
        "rogue_coordinate_test": {
            "note": "Gemma-2 (TIED) had PR(u1) 1.2-1.4 of 2304 — near one-hot. Pythia is UNTIED.",
            "min_PR": min(prs), "max_PR": max(prs),
            "median_PR": float(torch.tensor(prs).median()),
            "near_one_hot_layers": [r["layer"] for r in rows
                                    if r["P5_top_u_participation"] < 2.0],
        },
        "env": {"transformers": transformers.__version__, "torch": torch.__version__},
        "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "rows": rows,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n  PR(u1) across layers: min={min(prs):.2f} median="
          f"{float(torch.tensor(prs).median()):.2f} max={max(prs):.2f}  (d={lens.d_model})")
    print(f"  near-one-hot layers (PR<2): {out['rogue_coordinate_test']['near_one_hot_layers']}")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
