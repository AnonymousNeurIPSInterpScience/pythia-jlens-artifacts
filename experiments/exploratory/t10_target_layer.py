#!/usr/bin/env python3
"""t10_target_layer.py — is the anchor's default target FINAL or PENULTIMATE?

The anchor contradicts itself three ways:
  §2.1  writes  J_l = E[d h_final,t' / d h_l,t]                        -> FINAL
  §A.7  prose:  "The default lens on Sonnet 4.5, USED THROUGHOUT THE PAPER, computes dz/dh with
                 z taken at the PENULTIMATE layer"  and  "including the last layer can sometimes
                 increase the number of noisy artifacts in lens-readouts"
  §A.7  pseudocode comment: "z[t] : residual stream at the target layer L (by default final)"
Nanda's replication used the penultimate layer. The vendored library defaults to FINAL
(jlens/fitting.py `_check_layer_indices`: target = n_layers - 1), and every lens we have fitted
used that default.

This fits both and scores them on the anchor's own pass@k. Decides an empirical question the
paper leaves ambiguous.
"""
import argparse, json, os, sys, time
import torch
torch.set_num_threads(min(8, os.cpu_count() or 8))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jlens.hf import Layout, from_hf                      # noqa: E402
from jlens.lens import JacobianLens                       # noqa: E402
from fastfit import fast_fit                              # noqa: E402
from anchor_evals import pass_at_k, transport_jlens       # noqa: E402
from t2_fastfit import load_prompts                       # noqa: E402
L5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm", embed="embed_in", lm_head="lm_head")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
ap.add_argument("-n", type=int, default=50)
ap.add_argument("--evals", default="multihop,typo")
ap.add_argument("--dim-batch", type=int, default=128)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.perf_counter()
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
tok = AutoTokenizer.from_pretrained(a.model)
hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.device)
try: model = from_hf(hf, tok)
except ValueError: model = from_hf(hf, tok, layout=L5)
prompts = load_prompts(a.n)

variants = {"final": model.n_layers - 1, "penultimate": model.n_layers - 2}
res = {}
for name, tl in variants.items():
    t = time.perf_counter()
    lens = fast_fit(model, prompts, target_layer=tl, dim_batch=a.dim_batch,
                    checkpoint_path=None, checkpoint_every=0, device=a.device, log_every=0)
    layers = lens.source_layers
    res[name] = {"target_layer": tl, "n_source_layers": len(layers),
                 "fit_s": round(time.perf_counter() - t, 1), "evals": {}}
    for ev in a.evals.split(","):
        r = pass_at_k(model, transport_jlens(lens), ev, layers, max_items=40)
        res[name]["evals"][ev] = {"auc": r["normalized_auc_over_logk"],
                                  "pass@10": r["pass_at_k"]["10"],
                                  "median_rank": r["median_best_rank"]}
        print(f"  target={name:12s} {ev:10s} AUC={r['normalized_auc_over_logk']:.4f} "
              f"pass@10={r['pass_at_k']['10']:.3f} med_rank={r['median_best_rank']:.0f}", flush=True)
    res[name]["dispersion_median"] = round(float(torch.tensor(
        [v["dispersion"] for v in lens.dispersion.values()]).median()), 4)

out = {"experiment": "T10 target_layer: final vs penultimate",
       "status": "EXPLORATORY / DIRECTIONAL — no pre-registration",
       "why": "the anchor contradicts itself: SS2.1 says final, SSA.7 prose says penultimate is "
              "the default used throughout the paper, SSA.7 pseudocode says final. Nanda used "
              "penultimate. Our lenses all used the library default (final).",
       "model": a.model, "n_prompts": a.n, "centered": False, "device": a.device,
       "env": {"transformers": transformers.__version__, "torch": torch.__version__},
       "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
       "results": res, "runtime_s": round(time.perf_counter() - t0, 1)}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print("\n" + json.dumps({k: v["evals"] for k, v in res.items()}, indent=1))
print(f"wrote {a.out}")
