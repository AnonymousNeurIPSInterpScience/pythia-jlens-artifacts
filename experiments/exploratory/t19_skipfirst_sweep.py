#!/usr/bin/env python3
"""t19_skipfirst_sweep.py — E17. The skip_first confound: 0 vs 4 vs 16.

The paper's default is the unmodified position average (0); the released code hardcodes 16;
Nanda's replication used 4. Three values, no measurement anywhere.

PRE-REGISTERED (RESEARCH_NOTES sec13 E17):
  PRIMARY OUTCOME  n_win vs the logit lens under BEST-SINGLE-LAYER AUC (not min-over-layers,
                   per the E18 finding that min is existential and defective), plus disp(L0).
  DECISION RULE    if n_win differs by <=1 across all three values, skip_first is immaterial on
                   Pythia and 16 stays for continuity. If it differs by >=2, skip_first is a
                   live parameter and D3 must be re-registered.
  CONTROLS         the layer-shuffled J at each value.
  MODEL            410m -- already fully observed, so nothing is burned.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jlens.hf import Layout, from_hf                      # noqa: E402
from fastfit import fast_fit                              # noqa: E402
from t2_fastfit import load_prompts                       # noqa: E402
from t17_reaggregate import collect_ranks, aggregate, band_of  # noqa: E402
from t13_transport_controls import build_transports       # noqa: E402
from anchor_evals import EVAL_SETS                        # noqa: E402

LAYOUT = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                embed="embed_in", lm_head="lm_head")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
ap.add_argument("--skips", default="0,4,16")
ap.add_argument("-n", "--n-prompts", type=int, default=200)
ap.add_argument("--dim-batch", type=int, default=128)
ap.add_argument("--target-layer", type=int, default=-2)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
a = ap.parse_args()

t0 = time.perf_counter()
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
tok = AutoTokenizer.from_pretrained(a.model)
hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.device)
try:    model = from_hf(hf, tok)
except ValueError: model = from_hf(hf, tok, layout=LAYOUT)

prompts = load_prompts(a.n_prompts, corpus="wikitext")
skips = [int(x) for x in a.skips.split(",")]
res = {}
for sk in skips:
    print(f"=== skip_first={sk} ===", flush=True)
    lens = fast_fit(model, prompts, dim_batch=a.dim_batch, target_layer=a.target_layer,
                    skip_first=sk, checkpoint_path=None, checkpoint_every=0,
                    device=a.device, log_every=0)
    layers = list(lens.source_layers); band = band_of(model.n_layers, layers)
    tr, _ = build_transports(lens, layers, 0)
    d = lens.dispersion or {}
    cell = {"n_prompts_used": lens.n_prompts, "n_source_layers": len(layers),
            "disp_L0": d.get(layers[0], {}).get("dispersion"),
            "disp_last": d.get(layers[-1], {}).get("dispersion"), "per_set": {}}
    nw = {"best1L": 0, "min": 0, "persist": 0}
    for ev in EVAL_SETS:
        agg = {t: aggregate(collect_ranks(model, tr[t], ev, layers), layers, band, 0.5)
               for t in ("logit", "jlens", "shuffled")}
        cell["per_set"][ev] = {t: {g: round(agg[t][g], 4) for g in ("min","best1L","mean","persist")}
                               for t in agg}
        for g in nw:
            nw[g] += int(agg["jlens"][g] > agg["logit"][g])
        print(f"  {ev:13s} best1L jlens={agg['jlens']['best1L']:.4f} "
              f"logit={agg['logit']['best1L']:.4f} shuf={agg['shuffled']['best1L']:.4f}", flush=True)
    cell["n_win"] = nw
    res[str(sk)] = cell
    print(f"  -> n_win {nw}  disp(L0)={cell['disp_L0']}", flush=True)

pri = [res[str(s)]["n_win"]["best1L"] for s in skips]
spread = max(pri) - min(pri)
verdict = ("IMMATERIAL — n_win spread <=1 across skip_first; keep 16 for continuity"
           if spread <= 1 else
           "LIVE PARAMETER — n_win spread >=2; D3 must be re-registered")
out = {"experiment": "T19 / E17 — skip_first sweep 0 vs 4 vs 16",
       "status": "EXPLORATORY — 410m is fully observed, nothing burned",
       "decision_rule": "n_win (best-single-layer) spread <=1 => immaterial; >=2 => live parameter",
       "primary_outcome": "n_win vs logit under BEST-SINGLE-LAYER AUC",
       "model": a.model, "skips": skips, "n_prompts": a.n_prompts,
       "n_win_best1L_by_skip": {str(s): res[str(s)]["n_win"]["best1L"] for s in skips},
       "spread": spread, "VERDICT": verdict, "results": res,
       "env": {"transformers": transformers.__version__, "torch": torch.__version__},
       "runtime_s": round(time.perf_counter()-t0, 1)}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(f"\n  n_win(best1L) by skip_first: {out['n_win_best1L_by_skip']}  spread={spread}")
print(f"  VERDICT: {verdict}")
print(f"wrote {a.out} ({out['runtime_s']}s)")
