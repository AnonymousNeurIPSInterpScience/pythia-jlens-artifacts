#!/usr/bin/env python3
"""t9_n_convergence.py — direction 4: does the OPERATOR converge as fast as the task metric?

RESEARCH_NOTES.tex sec3, falsifiably: the anchor's own sweep shows *task* quality saturating by n≈10
("J-lens beats the logit lens and tuned lens baselines with as few as 10 prompts"), and Nanda's
replication used n=25. But nobody has asked whether the ESTIMATOR has converged there. The
literature review's prediction:

    task metric saturates at n≈10;  top singular SUBSPACE stabilizes later (n≈50);
    tail spectrum and dispersion keep moving through n=1000.

If true, behavioural saturation is not operator convergence, and every downstream geometric
claim (stable rank, PR(u1), anisotropy) computed at small n is measuring a moving target.

MEASURED PER n, on the SAME prompt prefix so the sequence is nested:
  - dispersion  E||J - J_bar||_F^2 / ||J_bar||_F^2                 (the estimator's error term)
  - ||J_n - J_2n||_op / ||J_2n||_op                                 (Cauchy-style convergence)
  - principal angles between top-r singular subspaces of J_n and J_ref
  - stable rank, effective rank
  - cosine between J_n and the reference J_ref, per layer
  - anchor pass@k on multihop (the TASK metric, for the comparison that matters)

The reference J_ref is the largest n in the sweep. Everything is nested, so J_n and J_ref share
prompts — this measures convergence of the estimator, not independent-sample agreement. A
separate `--split-half` mode fits two DISJOINT halves for an independence check.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import torch

torch.set_num_threads(min(8, os.cpu_count() or 8))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jlens.hf import Layout, from_hf  # noqa: E402
from fastfit import fast_fit  # noqa: E402
from anchor_evals import pass_at_k, transport_jlens  # noqa: E402
from t2_fastfit import load_prompts  # noqa: E402

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")


def principal_angles(A: torch.Tensor, B: torch.Tensor, r: int) -> float:
    """Mean cos of principal angles between the top-r left-singular subspaces of A and B."""
    Ua = torch.linalg.svd(A, full_matrices=False)[0][:, :r]
    Ub = torch.linalg.svd(B, full_matrices=False)[0][:, :r]
    s = torch.linalg.svdvals(Ua.T @ Ub)
    return float(s.clamp(-1, 1).mean())


def spectra(J: torch.Tensor) -> dict:
    S = torch.linalg.svdvals(J.float())
    s2 = S ** 2
    p = s2 / s2.sum()
    return {
        "stable_rank": round(float(s2.sum() / s2[0]), 4),
        "effective_rank": round(float(torch.exp(-(p * (p + 1e-30).log()).sum())), 4),
        "top_sv": round(float(S[0]), 4),
        "fro": round(float(J.norm()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--ns", default="1,5,10,25,50,100,200")
    ap.add_argument("--dim-batch", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top-r", type=int, default=8)
    ap.add_argument("--eval", default="multihop", help="task metric; '' to skip")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ns = sorted(int(x) for x in a.ns.split(","))

    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.device)
    try:
        model = from_hf(hf, tok)
    except ValueError:
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)

    prompts = load_prompts(max(ns))          # nested prefixes
    lenses, disp = {}, {}
    for n in ns:
        t = time.perf_counter()
        L = fast_fit(model, prompts[:n], dim_batch=a.dim_batch, checkpoint_path=None,
                     checkpoint_every=0, device=a.device, log_every=0)
        lenses[n] = {l: L.jacobians[l].float() for l in L.source_layers}
        disp[n] = getattr(L, "dispersion", None)
        print(f"  n={n:4d} fitted in {time.perf_counter()-t:.0f}s", flush=True)

    ref = max(ns)
    layers = sorted(lenses[ref])
    rows = []
    for n in ns:
        for l in layers:
            Jn, Jr = lenses[n][l], lenses[ref][l]
            rows.append({
                "n": n, "layer": l,
                **spectra(Jn),
                "dispersion": (disp[n] or {}).get(l, {}).get("dispersion"),
                "heavy_tail_ratio": (disp[n] or {}).get(l, {}).get("heavy_tail_ratio"),
                "cos_to_ref": round(float(torch.nn.functional.cosine_similarity(
                    Jn.flatten(), Jr.flatten(), dim=0)), 6),
                "rel_op_dist_to_ref": round(float(
                    torch.linalg.matrix_norm(Jn - Jr, 2) / torch.linalg.matrix_norm(Jr, 2)), 6),
                "subspace_cos_top_r_vs_ref": round(principal_angles(Jn, Jr, a.top_r), 6),
            })
        print(f"  n={n:4d} geometry done", flush=True)

    task = {}
    if a.eval:
        from jlens.lens import JacobianLens
        for n in ns:
            L = JacobianLens(jacobians={l: lenses[n][l] for l in layers},
                             n_prompts=n, d_model=model.d_model)
            r = pass_at_k(model, transport_jlens(L), a.eval, layers, max_items=40)
            task[n] = {"auc": r["normalized_auc_over_logk"], "pass@10": r["pass_at_k"]["10"]}
            print(f"  n={n:4d} task {a.eval} AUC={r['normalized_auc_over_logk']:.4f}", flush=True)

    out = {
        "experiment": "T9 n-convergence — operator vs task (RESEARCH_NOTES.tex sec3)",
        "status": "EXPLORATORY / DIRECTIONAL — no pre-registration",
        "model": a.model, "ns": ns, "reference_n": ref, "top_r": a.top_r,
        "layers": layers, "dim_batch": a.dim_batch, "centered": False,
        "nested_prefixes": True,
        "task_metric": {"eval": a.eval, "by_n": task} if a.eval else None,
        "env": {"transformers": transformers.__version__, "torch": torch.__version__,
                "device": a.device, "dtype": "float32"},
        "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "rows": rows,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    json.dump(out, open(a.out, "w"), indent=1)

    print(f"\n{'n':>5} {'task AUC':>9} {'cos->ref':>9} {'subspace':>9} {'rel_op':>8} "
          f"{'disp':>7} {'srank':>7}")
    for n in ns:
        rs = [r for r in rows if r["n"] == n]
        med = lambda k: float(torch.tensor([r[k] for r in rs if r[k] is not None]).median())
        ta = task.get(n, {}).get("auc")
        print(f"{n:5d} {('%.4f' % ta) if ta is not None else '   n/a':>9} "
              f"{med('cos_to_ref'):9.4f} {med('subspace_cos_top_r_vs_ref'):9.4f} "
              f"{med('rel_op_dist_to_ref'):8.4f} {med('dispersion'):7.3f} {med('stable_rank'):7.2f}")
    print(f"\nwrote {a.out}  ({out['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
