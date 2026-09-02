#!/usr/bin/env python3
"""t13_transport_controls.py — E1: does a DELIBERATELY BROKEN J still beat the logit lens?

THE GAP THIS CLOSES. `RIGOR_SKILL.md` axis 3 demands controls that empirically fire. T7 — the
headline read result — has **no control transport at all**. Its rubric entry claims axis 9
(pre-registration) is the only failing axis; that is wrong, axis 3 is also 0. This matters because
`rand_rot` once scored 0.996 against a real arm of 1.000 on the self-read, so "an obviously broken
operator fails" is not a safe assumption in this repo.

THE QUESTION. `J_l` maps an early residual into the final-layer basis. In a residual network
`h_final = h_l + (stuff)`, so `J_l ~ I + M`: any competent learned map into the final basis might
beat the raw logit lens, which is exactly the 2023 tuned-lens result. If a broken `J` also wins,
then T7 measures "any learned map into the final basis helps" and NOT "the J-lens reads a
workspace" — a different paper.

FIVE TRANSPORTS, one readout path, same items, same layers:

  logit          T = I                       the free baseline
  jlens          T = J_l                     the real arm
  shuffled       T = J_{pi(l)}, pi a derangement
                                             KEEPS every global property of the J-lens dictionary
                                             (scale, spectrum, basis alignment) and destroys ONLY
                                             the layer-specific content. **The decisive control.**
  spec_random    T = U' S V'^T, S = svd(J_l).S, U'/V' random orthogonal
                                             matched singular spectrum, random directions
  rank1          T = s1 u1 v1^T              only J's dominant direction

  scaled_id      T = c I, c = ||J_l||_F/sqrt(d)
                                             SELF-TEST, not a control: the final LayerNorm is
                                             scale-invariant, so this MUST score identically to the
                                             logit lens. If it does not, the readout path is wrong.

EXPLORATORY. 70m/160m/410m are burned per PREREG_PYTHIA_T7_v2 section 0 and can never be
confirmatory. This is a control on already-burned cells; it constrains interpretation, it does not
test H1.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import torch

torch.set_num_threads(min(8, os.cpu_count() or 8))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jlens.hf import Layout, from_hf  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402
from anchor_evals import EVAL_SETS, pass_at_k, transport_logit  # noqa: E402

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")


def _derangement(layers: list[int], seed: int) -> dict[int, int]:
    """A permutation with NO fixed point, so no layer keeps its own J."""
    g = torch.Generator().manual_seed(seed)
    n = len(layers)
    if n < 2:
        raise ValueError("need >= 2 layers to derange")
    for _ in range(1000):
        perm = torch.randperm(n, generator=g).tolist()
        if all(perm[i] != i for i in range(n)):
            return {layers[i]: layers[perm[i]] for i in range(n)}
    # deterministic fallback: rotate by one
    return {layers[i]: layers[(i + 1) % n] for i in range(n)}


def _rand_orthogonal(d: int, g: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float32))
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)      # fix sign ambiguity


def build_transports(lens: JacobianLens, layers: list[int], seed: int) -> dict:
    """{name: callable(h, layer) -> transported h}. Matrices are precomputed once per layer."""
    g = torch.Generator().manual_seed(seed)
    d = lens.d_model
    J = {l: lens.jacobians[l].float() for l in layers}

    perm = _derangement(layers, seed)
    shuffled = {l: J[perm[l]] for l in layers}

    spec_random, rank1, scaled_id = {}, {}, {}
    for l in layers:
        U, S, Vh = torch.linalg.svd(J[l], full_matrices=False)
        spec_random[l] = _rand_orthogonal(d, g) @ torch.diag(S) @ _rand_orthogonal(d, g).T
        rank1[l] = S[0] * torch.outer(U[:, 0], Vh[0, :])
        scaled_id[l] = torch.eye(d) * (J[l].norm() / (d ** 0.5))

    def mk(mats):
        def T(h, layer):
            return h.float() @ mats[layer].to(h.device).T
        return T

    return ({"logit": transport_logit(), "jlens": mk(J), "shuffled": mk(shuffled),
             "spec_random": mk(spec_random), "rank1": mk(rank1),
             "scaled_id": mk(scaled_id)},
            {str(k): v for k, v in perm.items()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evals", default="all")
    ap.add_argument("--max-items", type=int, default=None)
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
    layers = list(lens.source_layers)

    transports, perm = build_transports(lens, layers, a.seed)
    names = EVAL_SETS if a.evals == "all" else [s.strip() for s in a.evals.split(",")]

    results: dict = {}
    for ev in names:
        results[ev] = {}
        for tname, T in transports.items():
            r = pass_at_k(model, T, ev, layers, max_items=a.max_items)
            results[ev][tname] = {"auc_paper": r["normalized_auc_over_logk_paper"],
                                  "auc_readme": r["normalized_auc_over_logk"],
                                  "median_best_rank": r["median_best_rank"]}
        base = results[ev]["logit"]["auc_paper"]
        line = "  ".join(f"{t}={results[ev][t]['auc_paper']:.4f}" for t in transports)
        print(f"  {ev:13s} {line}", flush=True)

    # n_win vs the logit lens, per transport, on the PRIMARY (paper) metric
    n_win = {t: sum(1 for ev in names
                    if results[ev][t]["auc_paper"] > results[ev]["logit"]["auc_paper"])
             for t in transports if t != "logit"}

    # SELF-TEST: scaled identity must equal the logit lens (final LayerNorm kills the scale)
    worst_id = max(abs(results[ev]["scaled_id"]["auc_paper"] - results[ev]["logit"]["auc_paper"])
                   for ev in names)
    self_test = worst_id < 1e-4

    out = {
        "experiment": "T13 / E1 — transport controls: does a BROKEN J still beat the logit lens?",
        "status": "EXPLORATORY — control on cells already burned per PREREG_PYTHIA_T7_v2 section 0",
        "model": a.model, "lens": a.lens, "seed": a.seed,
        "layers": layers, "n_layers": len(layers),
        "derangement_layer_map": perm,
        "primary_metric": "normalized_auc_over_logk_paper (A.6 per-item binary)",
        "n_win_vs_logit": n_win,
        "n_evals": len(names),
        "SELF_TEST_scaled_identity_equals_logit": self_test,
        "SELF_TEST_worst_abs_auc_diff": round(worst_id, 6),
        "interpretation": (
            "If shuffled/spec_random/rank1 win on as many sets as jlens, T7 measures 'any learned "
            "map into the final basis helps', not J-lens-specific workspace reading."),
        "results": results,
        "env": {"transformers": transformers.__version__, "torch": torch.__version__,
                "device": "cpu", "dtype": "float32"},
        "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n  n_win vs logit (of {len(names)}): " +
          "  ".join(f"{k}={v}" for k, v in n_win.items()))
    print(f"  SELF-TEST scaled_id == logit: {self_test} (worst |Δ| {worst_id:.2e})")
    print(f"wrote {a.out}  ({out['runtime_s']}s)")
    return 0 if self_test else 1


if __name__ == "__main__":
    raise SystemExit(main())
