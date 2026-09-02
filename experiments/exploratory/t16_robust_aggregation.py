#!/usr/bin/env python3
"""t16_robust_aggregation.py — E2. Is the averaged Jacobian an artifact of the MEAN?

**THE EXPERIMENT MOST LIKELY TO SINK THE PAPER.** Everything downstream — dispersion as a
distribution diagnostic, the C2 refit-on-target story, the whole `J^P` vs `J^Q` programme —
assumes that when the J-lens changes between corpora, it changes because the *distribution*
changed. The alternative is that the arithmetic MEAN is a fragile estimator on a heavy-tailed
sample, and what we are measuring is estimator variance wearing a distribution costume.

The anchor itself flags this: it reports per-example Jacobians can be heavy-tailed, and tests
mean vs median aggregation and outlier filtering (`RIGOROUS_ANTHROPIC.md` §A.7).

DESIGN — a control, not a new condition. Same model, same 200 prompts in the same order, same
`skip_first`, same `dim_batch`, same `max_seq_len`, same `target_layer`. **Only the aggregator
changes.** Distribution is held fixed and only the estimator varies, so any difference is
attributable to the estimator alone.

Four aggregators, all from ONE pass over the corpus:
  mean            the anchor's default; identical to `fast_fit`'s output by construction
  median          coordinatewise median over prompts
  trimmed_10      mean after dropping the 10% of prompts with the largest ||J||_F, per layer
  winsor_10       as trimmed, but clipping those prompts' Frobenius norm instead of dropping

REGISTERED PREDICTION (before running; from the heavy-tail analysis in `RESEARCH_NOTES.tex` sec3):
**near-null.** Per-prompt Frobenius norms have a coefficient of variation of only 1.8–12%
(`tail = max/rms` is 1.05–1.70), so there are no magnitude outliers to trim. If the mean and the
median agree closely, the dispersion we measure is genuine DIRECTIONAL heterogeneity and the
distribution programme survives its main confound. **If they disagree, C2 and everything built on
it is estimator instability and must be rebuilt.**

MEMORY. Retaining every per-prompt Jacobian costs `n_source · d² · n_prompts · 2` bytes in fp16 —
23.5 GB for pythia-1b (14 layers, d=2048, n=200). That fits alongside the ~30 GB fit on an 80 GB
card. The naive fp32 figure (3.35 TB) is not the relevant number.

EXPLORATORY.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jlens.hf import Layout, from_hf  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402
from fastfit import fast_fit  # noqa: E402
from t2_fastfit import load_prompts, sha256  # noqa: E402

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / a.norm())


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.flatten() @ b.flatten()) / (a.norm() * b.norm()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1b-deduped")
    ap.add_argument("-n", "--n-prompts", type=int, default=200)
    ap.add_argument("--dim-batch", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--target-layer", type=int, default=-2)
    ap.add_argument("--corpus", default="wikitext")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trim-frac", type=float, default=0.10)
    ap.add_argument("--save-lenses", action="store_true",
                    help="also write median/trimmed lenses as .pt for downstream scoring")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

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

    prompts = load_prompts(a.n_prompts, corpus=a.corpus)
    d = model.d_model
    n_layers = model.n_layers
    target = a.target_layer if a.target_layer >= 0 else n_layers + a.target_layer
    src = list(range(target))
    store = {l: torch.empty(len(prompts), d, d, dtype=torch.float16, device=a.device)
             for l in src}
    gb = sum(t.numel() * 2 for t in store.values()) / 1e9
    print(f"retaining per-prompt Jacobians: {len(src)} layers x {d}x{d} x {len(prompts)} "
          f"= {gb:.1f} GB fp16 on {a.device}", flush=True)

    n_kept = {"i": 0}

    def on_prompt(idx, buf):
        for l in src:
            store[l][n_kept["i"]] = buf[l].to(torch.float16)
        n_kept["i"] += 1

    lens_mean = fast_fit(model, prompts, dim_batch=a.dim_batch, max_seq_len=a.max_seq_len,
                         target_layer=a.target_layer, checkpoint_path=None, checkpoint_every=0,
                         device=a.device, on_prompt=on_prompt)
    N = n_kept["i"]
    print(f"fit done: {N} prompts retained, {time.perf_counter()-t0:.0f}s", flush=True)

    per_layer, med_J, trim_J = {}, {}, {}
    k = max(1, int(round(a.trim_frac * N)))
    for l in src:
        S = store[l][:N]                                   # [N, d, d] fp16
        mean_J = lens_mean.jacobians[l].float().to(a.device)

        med = S.median(dim=0).values.float()               # lower median for even N; noted below
        fro = S.reshape(N, -1).float().norm(dim=1)         # per-prompt ||J||_F at this layer
        keep = torch.argsort(fro)[: N - k]                 # drop the k largest-norm prompts
        trim = S[keep].float().mean(dim=0)
        cap = fro[keep].max()
        scale = torch.clamp(cap / fro, max=1.0).view(N, 1, 1)
        winsor = (S.float() * scale).mean(dim=0)

        per_layer[str(l)] = {
            "n_prompts": N,
            "rel_dist_median_vs_mean": round(rel(mean_J, med), 6),
            "cos_median_vs_mean": round(cos(mean_J, med), 6),
            "rel_dist_trimmed_vs_mean": round(rel(mean_J, trim), 6),
            "cos_trimmed_vs_mean": round(cos(mean_J, trim), 6),
            "rel_dist_winsor_vs_mean": round(rel(mean_J, winsor), 6),
            "cos_winsor_vs_mean": round(cos(mean_J, winsor), 6),
            "fro_mean": round(float(mean_J.norm()), 4),
            "fro_median": round(float(med.norm()), 4),
            "fro_ratio_median_over_mean": round(float(med.norm() / mean_J.norm()), 4),
            "per_prompt_fro_cv": round(float(fro.std() / fro.mean()), 4),
            "per_prompt_fro_max_over_rms": round(
                float(fro.max() / fro.pow(2).mean().sqrt()), 4),
        }
        med_J[l] = med.cpu()
        trim_J[l] = trim.cpu()
        print(f"  L{l:2d}  cos(med,mean)={per_layer[str(l)]['cos_median_vs_mean']:.4f}  "
              f"rel={per_layer[str(l)]['rel_dist_median_vs_mean']:.4f}  "
              f"CV(||J||)={per_layer[str(l)]['per_prompt_fro_cv']:.3f}", flush=True)
        del S

    cosines = [v["cos_median_vs_mean"] for v in per_layer.values()]
    rels = [v["rel_dist_median_vs_mean"] for v in per_layer.values()]
    verdict = ("NULL — median ~ mean; dispersion is directional heterogeneity, not estimator "
               "instability; the distribution programme survives"
               if min(cosines) > 0.95 else
               "NOT NULL — the aggregator materially changes the operator; C2 and everything "
               "built on it must be re-examined as possible estimator instability")

    saved = {}
    if a.save_lenses:
        for name, J in (("median", med_J), ("trimmed", trim_J)):
            p = a.out.replace(".json", f"_{name}.pt")
            JacobianLens(jacobians=J, n_prompts=N, d_model=d).save(p)
            saved[name] = {"path": p, "sha256": sha256(p)}
            print(f"  wrote {p}")

    out = {
        "experiment": "T16 / E2 — robust aggregation control (mean vs median vs trimmed)",
        "status": "EXPLORATORY — control; distribution held fixed, only the estimator varies",
        "registered_prediction": ("near-null: per-prompt Frobenius CV is 1.8-12%, so there are no "
                                  "magnitude outliers to trim"),
        "model": a.model, "corpus": a.corpus, "n_prompts_used": N,
        "dim_batch": a.dim_batch, "max_seq_len": a.max_seq_len,
        "target_layer_arg": a.target_layer, "target_layer_effective": target,
        "source_layers": src, "trim_frac": a.trim_frac, "n_trimmed": k,
        "device": a.device, "dtype": "float32 fit, float16 retention",
        "median_note": ("torch.median on an even N returns the LOWER middle value, not the "
                        "midpoint of the two central order statistics"),
        "summary": {
            "min_cos_median_vs_mean": round(min(cosines), 6),
            "mean_cos_median_vs_mean": round(sum(cosines) / len(cosines), 6),
            "max_rel_dist_median_vs_mean": round(max(rels), 6),
            "VERDICT": verdict,
        },
        "per_layer": per_layer,
        "saved_lenses": saved,
        "env": {"transformers": transformers.__version__, "torch": torch.__version__},
        "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n  min cos(median, mean) over layers = {min(cosines):.4f}")
    print(f"  max relative distance             = {max(rels):.4f}")
    print(f"  VERDICT: {verdict}")
    print(f"wrote {a.out} ({out['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
