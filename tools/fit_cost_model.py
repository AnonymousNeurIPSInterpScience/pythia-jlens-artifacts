#!/usr/bin/env python3
"""
Cost model for J-lens fitting. Reproducible, so the number can be checked rather than trusted.

THE STRUCTURE, read off src/fastfit.py rather than assumed:

  per prompt:
    1 forward   on (dim_batch x seq_len) tokens through the FULL model
    ceil(d_model / dim_batch) backwards, each on (dim_batch x seq_len) tokens,
      through the BAND ONLY -- ActivationRecorder(..., start_graph_at=min(source_layers))
      means the graph does not extend below the lowest source layer.

  so backward tokens per prompt = d_model x seq_len   (NOT x n_layers:
  torch.autograd.grad(outputs=target, inputs=source_activations) returns gradients
  for every source layer from ONE backward)

FLOPs per token:  forward  ~ 2 x params
                  backward w.r.t. INPUTS only ~ 2 x params_traversed
(weight gradients are never needed here, so the usual 4x backward figure does not apply)

  .venv/bin/python tools/fit_cost_model.py
"""
import math

# name: (n_layers, d_model, total_params)
MODELS = {
    "70m":   (6,  512,  70e6),
    "160m":  (12, 768,  160e6),
    "410m":  (24, 1024, 410e6),
    "1b":    (16, 2048, 1.0e9),
    "1.4b":  (24, 2048, 1.4e9),
    "2.8b":  (32, 2560, 2.8e9),
    "6.9b":  (32, 4096, 6.9e9),
    "12b":   (36, 5120, 12e9),
}

N_PROMPTS = 25          # S2 settled that N is flat; CV4 registers N=25
SEQ_LEN = 128
DIM_BATCH = 128
N_OPERATORS = 15        # 5 corpora x 3 seed blocks

# The band rule: normalised L38-L92, intersected with layers strictly below the penultimate target.
BAND_LO, BAND_HI = 0.38, 0.92

# Sustained throughput, FLOP/s. IMPLEMENTATION_RULES R18 records this workload at ~100% of fp32
# peak (arithmetic intensity 482 FLOP/byte against an A100 ridge of 9.6), so fp32 is taken at peak.
# bf16 is given at 80% of dense peak, which is the honest end of what these shapes achieve.
HW = {
    "H100 fp32 (no TF32)":  67e12,
    "H100 bf16 @80%":       989e12 * 0.80,
    "L40S fp32 (no TF32)":  91.6e12,
    "L40S bf16 @80%":       362e12 * 0.80,
    "4090 fp32 (no TF32)":  82.6e12,
}


def band_layers(n_layers):
    lo, hi = math.floor(BAND_LO * n_layers), math.floor(BAND_HI * n_layers)
    hi = min(hi, n_layers - 3)          # strictly below the penultimate target
    return max(1, hi - lo + 1)


def flops_per_operator(n_layers, d_model, params):
    per_layer = 12 * d_model ** 2                       # non-embedding params per block
    nb = band_layers(n_layers)
    band_params = per_layer * nb
    fwd_tokens = N_PROMPTS * DIM_BATCH * SEQ_LEN
    bwd_tokens = N_PROMPTS * d_model * SEQ_LEN
    return fwd_tokens * 2 * params + bwd_tokens * 2 * band_params, nb, bwd_tokens


def fmt(sec):
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 5400:
        return f"{sec/60:.0f}m"
    return f"{sec/3600:.1f}h"


def main():
    print(f"J-lens fitting cost.  N={N_PROMPTS} prompts, seq_len={SEQ_LEN}, "
          f"dim_batch={DIM_BATCH}, {N_OPERATORS} operators (5 corpora x 3 seeds)\n")
    rows = {}
    print(f"{'model':7} {'layers':>6} {'band':>5} {'bwd tokens':>12} {'PFLOP/op':>10} "
          f"{'PFLOP x15':>10}")
    for m, (nl, d, p) in MODELS.items():
        f, nb, bt = flops_per_operator(nl, d, p)
        rows[m] = f
        print(f"{m:7} {nl:6} {nb:5} {bt:12,} {f/1e15:10.2f} {f*N_OPERATORS/1e15:10.1f}")

    for hw, tp in HW.items():
        print(f"\n--- {hw}  ({tp/1e12:.0f} TFLOP/s sustained) ---")
        print(f"{'model':7} {'per operator':>13} {'15 operators':>14}")
        for m in MODELS:
            s = rows[m] / tp
            print(f"{m:7} {fmt(s):>13} {fmt(s*N_OPERATORS):>14}")

    print("\nNOTE. TF32/bf16 are FORBIDDEN by IMPLEMENTATION_RULES R18 — a 10-bit mantissa breaks "
          "the anchor gate's 2e-5 tolerance. CV5 adds an independent reason: at bf16-scale error "
          "(max_rel 1e-2) the rank metric moves 1.834 pooled seed SDs, so a precision change "
          "mid-ladder would confound precision with capability. The bf16 columns are shown because "
          "they were asked for, NOT because they are currently admissible.")


if __name__ == "__main__":
    main()
