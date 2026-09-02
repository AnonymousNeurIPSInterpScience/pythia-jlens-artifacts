#!/usr/bin/env python3
"""t18_n_sweep.py — E19. Does eps = sqrt(disp/n) actually hold?

WHY. S2's backbone is an ANALYTIC IDENTITY from external review B1:

    E||J_hat_n - J||_F^2 / ||J||_F^2 = disp / n     =>     eps_F = sqrt(disp / n)

An identity predicting error is not a measurement of error. This measures it.

PRE-REGISTERED (RESEARCH_NOTES sec13 E19):
  PRIMARY OUTCOME  log-log slope of measured relative error against n, per layer.
  DECISION RULE    the law predicts slope -0.5. ACCEPT if the fitted slope lies in
                   [-0.6, -0.4] at EVERY layer. Otherwise the law does not describe this
                   estimator and S2's backbone must be replaced with an empirical curve.
  DECLARED BIAS    using J_hat_400 as the reference makes the n=400 point degenerate, so it is
                   EXCLUDED from the fit. This is the same circularity that invalidated T9's
                   subspace measure (1.00 at n=200 IS the reference), which is why the
                   reference-free diagnostics below are run alongside.

REFERENCE-FREE DIAGNOSTICS (external review B1 sec5), which need no larger-n reference at all:
  split-half   fit two disjoint halves of the SAME n prompts; their disagreement estimates
               2x the error of an n/2 fit. Requires no J_ref.
  odd-even     the same, with an interleaved split, so ordering effects are visible.

Nested prefixes throughout: J_hat_n uses prompts[:n], so the sequence is monotone in data.

EXPLORATORY unless --prereg is cited.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jlens.hf import Layout, from_hf  # noqa: E402
from fastfit import fast_fit  # noqa: E402
from t2_fastfit import load_prompts  # noqa: E402

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")


def relerr(A: torch.Tensor, B: torch.Tensor) -> float:
    return float((A - B).norm() / B.norm())


def loglog_slope(xs, ys) -> float:
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx); mx = sum(lx) / n; my = sum(ly) / n
    den = sum((a - mx) ** 2 for a in lx)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
    ap.add_argument("--ns", default="10,25,50,100,200")
    ap.add_argument("--n-ref", type=int, default=400)
    ap.add_argument("--dim-batch", type=int, default=128)
    ap.add_argument("--target-layer", type=int, default=-2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.device)
    try:
        model = from_hf(hf, tok)
    except ValueError:
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)

    ns = sorted(int(x) for x in a.ns.split(","))
    prompts = load_prompts(a.n_ref, corpus="wikitext")
    fit = lambda P: fast_fit(model, P, dim_batch=a.dim_batch, target_layer=a.target_layer,  # noqa
                             checkpoint_path=None, checkpoint_every=0, device=a.device,
                             log_every=0)

    print(f"reference fit at n={a.n_ref} ...", flush=True)
    ref = fit(prompts)
    layers = list(ref.source_layers)
    disp = {int(k): v["dispersion"] for k, v in (ref.dispersion or {}).items()}

    rows = {}
    for n in ns:
        L = fit(prompts[:n])
        rows[n] = {str(l): round(relerr(L.jacobians[l].float(), ref.jacobians[l].float()), 6)
                   for l in layers}
        print(f"  n={n:4d}  relerr(L0)={rows[n][str(layers[0])]:.4f}  "
              f"relerr(mid)={rows[n][str(layers[len(layers)//2])]:.4f}", flush=True)

    # reference-free: split-half and odd-even at the largest n
    N = ns[-1]
    A_, B_ = fit(prompts[:N // 2]), fit(prompts[N // 2:N])
    O_, E_ = fit(prompts[0:N:2]), fit(prompts[1:N:2])
    free = {str(l): {"split_half_disagreement": round(relerr(A_.jacobians[l].float(),
                                                             B_.jacobians[l].float()), 6),
                     "odd_even_disagreement": round(relerr(O_.jacobians[l].float(),
                                                            E_.jacobians[l].float()), 6)}
            for l in layers}

    per_layer = {}
    for l in layers:
        xs = [n for n in ns]; ys = [rows[n][str(l)] for n in ns]
        s = loglog_slope(xs, ys)
        pred = {n: round(math.sqrt(disp.get(l, float("nan")) / n), 6) for n in ns} if l in disp else {}
        per_layer[str(l)] = {"measured": {str(n): rows[n][str(l)] for n in ns},
                             "predicted_sqrt_disp_over_n": {str(k): v for k, v in pred.items()},
                             "loglog_slope": round(s, 4),
                             "in_decision_band": bool(-0.6 <= s <= -0.4),
                             "dispersion": disp.get(l)}
    slopes = [per_layer[str(l)]["loglog_slope"] for l in layers]
    accept = all(per_layer[str(l)]["in_decision_band"] for l in layers)

    out = {
        "experiment": "T18 / E19 — does eps = sqrt(disp/n) hold for the averaged Jacobian?",
        "status": "CONFIRMATORY" if a.prereg else "EXPLORATORY",
        "prereg": a.prereg,
        "decision_rule": "ACCEPT iff the log-log slope lies in [-0.6,-0.4] at EVERY layer",
        "declared_bias": ("J_ref at n=%d is the reference, so n=%d is degenerate and excluded "
                          "from the fit; reference-free split-half / odd-even are reported "
                          "alongside" % (a.n_ref, a.n_ref)),
        "model": a.model, "ns": ns, "n_ref": a.n_ref, "layers": layers,
        "dim_batch": a.dim_batch, "target_layer_effective": max(layers) + 1,
        "slope_min": round(min(slopes), 4), "slope_max": round(max(slopes), 4),
        "slope_mean": round(sum(slopes) / len(slopes), 4),
        "VERDICT": ("ACCEPT — the sqrt(disp/n) law describes this estimator"
                    if accept else
                    "REJECT — the law does not hold at every layer; S2 needs an empirical curve"),
        "n_layers_in_band": sum(1 for l in layers if per_layer[str(l)]["in_decision_band"]),
        "per_layer": per_layer,
        "reference_free": free,
        "env": {"transformers": transformers.__version__, "torch": torch.__version__},
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n  slope: min {min(slopes):.3f}  mean {sum(slopes)/len(slopes):.3f}  max {max(slopes):.3f}")
    print(f"  layers inside [-0.6,-0.4]: {out['n_layers_in_band']}/{len(layers)}")
    print(f"  VERDICT: {out['VERDICT']}")
    print(f"wrote {a.out} ({out['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
