#!/usr/bin/env bash
# E61 — the randomized-network null. How much read survives when there is no computation to read?
# PREREG: docs/experiments/preregs/E61_randomized_network_null.md, written before the first number existed.
#
# Every control in this programme randomizes the OPERATOR (J^shuf, norm-matched random transport,
# random-k ablation). None randomizes the MODEL. This does: every transformer block re-initialised
# from the model's own initialiser, with the embedding, the unembedding and the final norm held
# REAL, so the readout is untouched and only the computation is destroyed.
#
# LOAD-BEARING, and it nearly went wrong: transformers >= 5 stamps `_is_hf_initialized` on every
# loaded TENSOR (not module), and transformers/initialization.py's normal_/zeros_/ones_ skip any
# tensor carrying it. Applying `_init_weights` directly is a SILENT NO-OP -- measured, block
# Frobenius 1603.7 -> 1603.7 with every downstream predictor bit-identical to the real model. A
# randomized null that is secretly the real model is the worst failure available here, so
# trainval.py clears the flag on parameters and ABORTS if the norm does not move.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e61"; MODULE_TITLE="randomized-network null (dead salmon)"
MODULE_WHY="how much of the read survives with no computation to read?"
MODULE_COST="gpu:1.25"          # 10 x 410m fits @ 0.17 GPU-h; L40S @ \$0.736/h
MODULE_TIER="T2"
INPUTS=(corpora experiments/trainval.py results/e48_crossover_410m.json)
OUTPUTS=(results/e61/tv_real_USPTO_Backgrounds_s0.json)

CORPORA=(USPTO_Backgrounds Github Pile-CC)   # best, worst, middle by measured read asymptote
SEEDS=(0 1 2)
BAND="9,21"
NMAX=200

main() {
  mkdir -p results/e61
  # --- the null: one randomized model per seed block, fitted on each of the three corpora
  for s in "${SEEDS[@]}"; do
    for c in "${CORPORA[@]}"; do
      info "randomized  $c s$s  (block seed $((61000+s)))"
      run_py experiments/trainval.py --model 410m --corpus "$c" --seed "$s" \
        --randomize-blocks $((61000+s)) --band "$BAND" --n-max "$NMAX" --ckpts "$NMAX" \
        --dim-batch 128 --device cuda \
        --out "results/e61/tv_rand_${c}_s${s}.json"
    done
  done
  # --- C3: the SAME code path on the REAL model. Its Q0 admitted-set read must reproduce
  #     e48_crossover_410m.json. Without this the randomized arm is not known to be on the same
  #     scoring path as everything the paper reports.
  info "C3 control: real model, same path"
  run_py experiments/trainval.py --model 410m --corpus USPTO_Backgrounds --seed 0 \
    --band "$BAND" --n-max "$NMAX" --ckpts "$NMAX" --dim-batch 128 --device cuda \
    --out "results/e61/tv_real_USPTO_Backgrounds_s0.json"
}
module_main "$@"
