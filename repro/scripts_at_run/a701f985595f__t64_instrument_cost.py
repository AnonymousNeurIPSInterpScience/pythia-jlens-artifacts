#!/usr/bin/env python3
"""t64_instrument_cost.py — what does a transport operator cost per checkpoint, against a
sparse dictionary?

THE QUESTION (the operator's, restated)
  A J-lens needs N forward+backward passes and N can apparently be small. An SAE or crosscoder
  needs a training run over activations, per checkpoint. For a checkpoint SERIES — where the
  instrument is refitted at every point — does the math favour the transport operator?

REVISED 2026-08-17 against an alphaXiv response that found a real error in v1.
  v1 charged activation generation ONCE PER SAE LAYER. That is wrong when the deliverable is an
  all-layer suite: one model forward exposes every hooked site, so activation capture is amortised
  across layers and only the dictionary optimisation is per-layer:

      C_suite ~= C_capture + sum_l C_opt(l)        NOT   sum_l ( C_capture + C_opt(l) )

  v1's headline "the transport operator is cheaper by 21x to 267x per source layer" was inflated by
  exactly that double-charge and is RETRACTED. The corrected comparison is against an all-layer
  suite and the margin is much smaller.

CALIBRATION AGAINST THE ONE MEASURED NUMBER IN THE LITERATURE
  arXiv:2509.05291 (Crosscoding Through Time) reports ~6 A100-80GB h for a 2-way crosscoder at a
  middle layer of a 1B model, 400M activation tokens, F=16,384. Against that anchor this script's
  FLOP model predicts 25.3 h at A100 fp32 peak (overpredicts 4.2x) and 1.6 h at bf16 peak
  (underpredicts 3.8x). The measured value sits between, which is what mixed precision plus
  sub-peak utilisation looks like. So THE FLOP MODEL IS BRACKETED BY, NOT VALIDATED AGAINST, the
  one real measurement — and its output is reported as a range with that bracket attached.

  Per the source's explicit warning, the 6 h is NOT divided by 2 to get a per-checkpoint cost: a
  crosscoder is a joint object whose cost depends on how many sources it ties.

WHAT IS AND IS NOT COMPARABLE
  A J-lens gives one transport map per source layer, fitted over the whole band in a single pass.
  An SAE gives a reconstruction dictionary for one site. A crosscoder gives a JOINT feature space
  across sources — which is a deliverable neither of the others supplies, and is the reason to pay
  for one. Acquisition cost is the only commensurable axis here; "per explained variance" does not
  normalise against a readout that never optimises reconstruction.

    python experiments/t64_instrument_cost.py
"""
from __future__ import annotations
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from provenance import write_result  # noqa: E402

# (n_layers, d_model, n_params) — n_params from 12*L*d^2 + 2*V*d, V = 50304
V = 50304
MODELS = {"410m": (24, 1024), "1b": (16, 2048), "2.8b": (32, 2560), "6.9b": (32, 4096)}
# the repo's own cost table, L40S column, GPU-hours for ONE fit at N=200 (repro/20_cost_estimate.sh)
JLENS_GPUH_L40S = {"410m": 0.17, "1b": 0.45, "2.8b": 1.42, "6.9b": None}   # 6.9b does not fit
L40S_FP32_TFLOPS = 91.6
UTIL = 1.0          # measured at ~100% of fp32 peak; stated, not assumed silently


def params(L, d):
    return 12 * L * d * d + 2 * V * d


def main() -> int:
    out = {
        "experiment": "E64 — acquisition cost per checkpoint: transport operator vs sparse dictionary",
        "question": ("for a checkpoint series, where the instrument is refitted at every point, "
                     "does a J-lens cost less than an SAE/crosscoder?"),
        "revision": "v2 2026-08-17 — v1 double-charged activation capture per layer; retracted",
        "NOT_COMPARABLE_AS_OBJECTS": (
            "a J-lens gives one transport map per source layer over the whole band in a single "
            "pass; an SAE gives a feature dictionary for ONE layer. This compares acquisition "
            "cost only, and says nothing about what either instrument tells you."),
        "jlens_side": "MEASURED — repo cost table, converted at measured ~100% fp32 peak",
        "sae_side": ("MODELLED, and BRACKETED by one measured anchor (arXiv:2509.05291: ~6 "
                     "A100-80GB h for a 2-way crosscoder at 1B, 400M tokens, F=16384). "
                     "arXiv:2509.17196 reports token budgets (800M) but no hardware or GPU-hours."),
        "assumptions": {
            "activation_flops_per_token": "2 * P_model (forward only, base model frozen)",
            "dictionary_flops_per_token": "6 * d_model * n_features (fwd+bwd+update), PER LAYER",
            "activation_capture": "charged ONCE for the whole suite, not per layer (v2 fix)",
            "n_tokens_swept": [1e8, 1e9],
            "expansion_swept": [8, 32],
            "l40s_fp32_tflops": L40S_FP32_TFLOPS, "utilisation": UTIL,
        },
        "by_model": {},
    }

    for m, (L, d) in MODELS.items():
        P = params(L, d)
        gpuh = JLENS_GPUH_L40S[m]
        jl_flops = (gpuh * 3600 * L40S_FP32_TFLOPS * 1e12 * UTIL) if gpuh else None
        entry = {"n_layers": L, "d_model": d, "n_params": P,
                 "jlens_gpu_h_N200_L40S": gpuh, "jlens_flops": jl_flops,
                 "jlens_covers_n_source_layers": L - 2,
                 "sae": {}}
        nsrc = L - 2
        for ntok in (1e8, 1e9):
            for exp in (8, 32):
                nfeat = d * exp
                cap = ntok * 2 * P                      # ONCE for the whole suite
                opt1 = ntok * 6 * d * nfeat             # per layer
                suite = cap + nsrc * opt1
                key = f"{int(ntok/1e6)}Mtok_x{exp}"
                entry["sae"][key] = {
                    "n_features": nfeat,
                    "flops_activation_capture_once": cap,
                    "flops_dict_opt_one_layer": opt1,
                    "flops_all_layer_suite": suite,
                    "capture_share_of_suite": cap / suite,
                    "gpu_h_suite_L40S": suite / (L40S_FP32_TFLOPS * 1e12 * UTIL) / 3600,
                    "ratio_suite_over_jlens": (suite / jl_flops) if jl_flops else None,
                    "RETRACTED_v1_ratio_per_layer_double_charged":
                        ((ntok * (2 * P + 6 * d * nfeat)) / (jl_flops / nsrc)) if jl_flops else None,
                }
        out["by_model"][m] = entry

    # the headline, at 1b, which is the candidate scale
    e = out["by_model"]["1b"]
    lo = e["sae"]["100Mtok_x8"]["ratio_suite_over_jlens"]
    hi = e["sae"]["1000Mtok_x32"]["ratio_suite_over_jlens"]
    out["MEASURED_ANCHOR"] = {
        "source": "arXiv:2509.05291 Crosscoding Through Time, training details",
        "what": "2-way crosscoder, middle layer of a 1B model, 400M tokens, F=16384, fp32 encoder",
        "measured_a100_80gb_hours": 6.0,
        "three_way_measured": 12.0,
        "this_scripts_flop_model_predicts_h_at_a100_fp32_peak": 25.3,
        "this_scripts_flop_model_predicts_h_at_a100_bf16_peak": 1.6,
        "reading": ("the measurement sits BETWEEN the model's fp32 and bf16 predictions, which is "
                    "what mixed precision plus sub-peak utilisation looks like. The FLOP model is "
                    "bracketed by the one real number, not validated against it."),
        "DO_NOT": ("divide 6 h by 2 to get a per-checkpoint cost. A crosscoder is a joint object "
                   "whose cost depends on how many sources it ties (source's own warning)."),
    }
    out["HEADLINE_1b"] = {
        "jlens_one_fit_all_14_source_layers_gpu_h": e["jlens_gpu_h_N200_L40S"],
        "sae_all_layer_suite_gpu_h_range": [e["sae"]["100Mtok_x8"]["gpu_h_suite_L40S"],
                                            e["sae"]["1000Mtok_x32"]["gpu_h_suite_L40S"]],
        "ratio_range_suite_over_jlens": [lo, hi],
        "capture_share_of_suite_range": [e["sae"]["1000Mtok_x32"]["capture_share_of_suite"],
                                         e["sae"]["100Mtok_x8"]["capture_share_of_suite"]],
        "reading": (
            f"at 1b, one J-lens fit at N=200 covering all {e['jlens_covers_n_source_layers']} "
            f"source layers costs {e['jlens_gpu_h_N200_L40S']} GPU-h. An all-layer SAE suite — "
            f"activation capture charged ONCE plus {e['jlens_covers_n_source_layers']} dictionary "
            f"optimisations — costs {lo:.1f}x to {hi:.0f}x that. v1 said 21x-267x per layer; that "
            f"figure double-charged activation capture and is retracted."),
        "CAVEAT": (
            "the dictionary side remains a FLOP model, and SAE training is typically NOT "
            "compute-bound at near-peak: it is data-loader or storage-throughput constrained "
            "(arXiv:2410.20526 reports ~8 KB/token of latent activations, ~2 TB and >=500 MB/s for "
            "a pre-saving pipeline). Our workload runs at ~100% of fp32 peak, so a wall-clock "
            "comparison would favour the transport operator MORE than these FLOP ratios do. "
            "Quote the range and the accounting, never a point estimate."),
    }

    # what N would have to be for the J-lens to lose
    out["BREAK_EVEN"] = {
        "note": ("J-lens cost is linear in N. Holding the dictionary side at its cheapest modelled "
                 "setting (100M tokens, x8), the J-lens would have to fit this many prompts to "
                 "cost the same as the ALL-LAYER SAE suite:"),
        "by_model": {m: (out["by_model"][m]["sae"]["100Mtok_x8"]["ratio_suite_over_jlens"] * 200)
                     for m in MODELS if out["by_model"][m]["jlens_flops"]},
    }

    dest = os.path.join(HERE, "..", "results", "e64_instrument_cost.json")
    write_result(dest, out, experiment="E64", inputs=[])
    h = out["HEADLINE_1b"]
    print(h["reading"])
    a = out["MEASURED_ANCHOR"]
    print(f"\nmeasured anchor: {a['measured_a100_80gb_hours']} A100-h; model predicts "
          f"{a['this_scripts_flop_model_predicts_h_at_a100_fp32_peak']} h fp32 / "
          f"{a['this_scripts_flop_model_predicts_h_at_a100_bf16_peak']} h bf16 -> bracketed")
    print(f"\nbreak-even N (J-lens = the ALL-LAYER SAE suite, cheapest setting):")
    for m, n in out["BREAK_EVEN"]["by_model"].items():
        print(f"   {m:6s} N = {n:,.0f} prompts   (we use 200)")
    print(f"\n{h['CAVEAT']}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
