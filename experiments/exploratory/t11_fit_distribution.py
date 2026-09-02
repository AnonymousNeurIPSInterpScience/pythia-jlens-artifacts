#!/usr/bin/env python3
"""t11_fit_distribution.py — direction 2 (THESIS C2): does the FIT distribution matter?

The literature review found no paper, for ANY lens, that reports the fit-distribution x
eval-distribution factorial with a refit-on-target cell. This runs its cheapest decisive form:

    fit J on corpus A  ->  evaluate pass@k on eval set E,  for A in {wikitext, eval:E}

`eval:E` is a TASK-ADJACENT fit — the Jacobian is averaged over the very prompts the lens is
then scored on. That is the refit-on-target cell.

  If task-adjacent fitting IMPROVES pass@k, the averaged Jacobian is a corpus-specific
  estimator and the J-lens carries a distributional liability the logit lens cannot have.
  If it does NOT, the averaged Jacobian is a model-internal geometric map and F2 is benign.

Both outcomes are reportable; the prediction is registered in the output header before reading.
"""
import argparse, json, os, sys, time
import torch
torch.set_num_threads(min(8, os.cpu_count() or 8))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jlens.hf import Layout, from_hf                       # noqa: E402
from fastfit import fast_fit                               # noqa: E402
from anchor_evals import pass_at_k, transport_jlens, transport_logit  # noqa: E402
from t2_fastfit import load_prompts                        # noqa: E402
L5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm", embed="embed_in", lm_head="lm_head")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
ap.add_argument("-n", type=int, default=50)
ap.add_argument("--evals", default="multihop,typo")
ap.add_argument("--fits", default="wikitext,TASK")
ap.add_argument("--dim-batch", type=int, default=128)
ap.add_argument("--skip-first", type=int, default=None,
                help="override skip_first; needed because most anchor eval prompts "
                     "are shorter than the default 16 skipped positions")
ap.add_argument("--max-items", type=int, default=40)
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

evals = a.evals.split(",")
results = {}
for ev in evals:
    results[ev] = {}
    # logit-lens floor: identical for every fit corpus, since it estimates nothing
    r0 = pass_at_k(model, transport_logit(), ev,
                   list(range(model.n_layers - 1)), max_items=a.max_items)
    results[ev]["logit_lens_floor"] = {"auc": r0["normalized_auc_over_logk"]}
    for fit in a.fits.split(","):
        corpus = f"eval:{ev}" if fit == "TASK" else fit
        t = time.perf_counter()
        prompts = load_prompts(a.n, corpus=corpus)
        kw = {} if a.skip_first is None else {"skip_first": a.skip_first}
        lens = fast_fit(model, prompts, dim_batch=a.dim_batch, checkpoint_path=None,
                        checkpoint_every=0, device=a.device, log_every=0, **kw)
        r = pass_at_k(model, transport_jlens(lens), ev, lens.source_layers,
                      max_items=a.max_items)
        disp = float(torch.tensor([v["dispersion"] for v in lens.dispersion.values()]).median())
        results[ev][fit] = {"corpus": corpus, "auc": r["normalized_auc_over_logk"],
                            "pass@10": r["pass_at_k"]["10"],
                            "median_rank": r["median_best_rank"],
                            "dispersion_median": round(disp, 4),
                            "fit_s": round(time.perf_counter() - t, 1)}
        print(f"  {ev:10s} fit={fit:9s} AUC={r['normalized_auc_over_logk']:.4f} "
              f"disp={disp:.3f}", flush=True)
    w, tk = results[ev].get("wikitext"), results[ev].get("TASK")
    if w and tk:
        results[ev]["task_minus_wikitext_auc"] = round(tk["auc"] - w["auc"], 4)
        print(f"  {ev:10s} TASK-ADJACENT minus WIKITEXT = "
              f"{results[ev]['task_minus_wikitext_auc']:+.4f}\n", flush=True)

out = {"experiment": "T11 fit-distribution (THESIS C2 / direction 2)",
       "status": "EXPLORATORY / DIRECTIONAL — no pre-registration",
       "prediction_registered_before_reading": (
           "If the averaged Jacobian is a corpus-specific estimator, task-adjacent fitting "
           "raises pass@k. If it is a model-internal geometric map, the delta is ~0."),
       "model": a.model, "n_prompts": a.n, "centered": False, "device": a.device,
       "skip_first_override": a.skip_first,
       "env": {"transformers": transformers.__version__, "torch": torch.__version__},
       "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
       "results": results, "runtime_s": round(time.perf_counter() - t0, 1)}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(f"wrote {a.out}  ({out['runtime_s']}s)")
