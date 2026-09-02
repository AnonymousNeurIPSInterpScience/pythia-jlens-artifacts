#!/usr/bin/env bash
# E62 — refit the 1B ladder on the band the paper actually declares.
# PREREG: docs/experiments/preregs/E62_ladder1b_corrected_band.md, written before the first number existed.
#
# The paper's band rule gives [6,13] at 1b (16 layers, target index 14). Every stored
# results/ladder1b/*.json carries [5,13] -- one layer too low, measured in e53_ladder_summary.json.
# That arm carries Limitation 6's replication claim AND E51's second scale, so it has to be right.
#
# WHY THIS IS A REFIT AND NOT A RESCORE: the original run did not pass --save-lens, so no 1B
# operator was ever retained. --save-lens is therefore the load-bearing flag here: it makes every
# FUTURE band question at 1B free instead of $5.
#
# PREREG GUARD: PREREG_PYTHIA_T7_v2 §2 names pythia-{1b,1.4b,2.8b} confirmatory. 1b's cells are
# ALREADY fully observed (ladder1b scores all six eval sets), so this opens no new door.
# 1.4b and 2.8b are untouched here and must stay that way.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e62"; MODULE_TITLE="1B ladder on the declared band [6,13]"
MODULE_WHY="does the 1B replication survive the band the paper declares?"
MODULE_COST="gpu:4.97"          # 15 x 1b fits @ 0.45 GPU-h; L40S @ \$0.736/h
MODULE_TIER="T2"
INPUTS=(corpora experiments/trainval.py results/ladder1b/tv_Github_s0.json)
OUTPUTS=(results/ladder1b_b613/tv_Github_s0.json)

CORPORA=(Github Wikipedia_en StackExchange Pile-CC USPTO_Backgrounds)
SEEDS=(0 1 2)
BAND="6,13"
CKPTS="10,25,50,75,100,150,200"

main() {
  mkdir -p results/ladder1b_b613
  for c in "${CORPORA[@]}"; do
    for s in "${SEEDS[@]}"; do
      out="results/ladder1b_b613/tv_${c}_s${s}.json"
      if [ -s "$out" ]; then ok "already done: $out"; continue; fi   # resumable
      info "1b  $c  s$s  band $BAND"
      run_py experiments/trainval.py --model 1b --corpus "$c" --seed "$s" \
        --band "$BAND" --n-max 200 --ckpts "$CKPTS" --dim-batch 128 --device cuda \
        --save-lens "results/ladder1b_b613/lens_${c}_1b_n200_s${s}.pt" \
        --out "$out"
    done
  done
}
module_main "$@"
