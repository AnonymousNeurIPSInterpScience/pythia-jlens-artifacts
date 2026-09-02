#!/usr/bin/env bash
# E48c — does exposure of the FITTING corpus order the read?
# This is NOT predictor #22 (HANDOFF sec0.5). It tests ONE pre-existing, externally-measured
# variable that E48's own rung classification is already built on, and it targets the read LEVEL,
# not the corpus x concept interaction. The answer being NO is what licenses the paper's claim.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e48c"; MODULE_TITLE="exposure vs read"
MODULE_WHY="turns E48's NOT REACHED into a positive statement on the axis that has power"
MODULE_COST="free"
MODULE_TIER="T0"
INPUTS=(results/e48_crossover_410m.json results/e48b_exposure_growth.json)
OUTPUTS=(results/e48c_exposure_vs_read.json)
main() { run_py experiments/t48c_exposure_vs_read.py; }
module_main "$@"
