#!/usr/bin/env bash
# e49_stability — how much is one derangement draw worth?
#
# HAND-WRITTEN. results/e49_derangement_stability.json has no `provenance.argv` and no registration
# of any kind; its stored verdict is UNSTABLE and the register says so. Flags below are the script's
# defaults with the stored n_seeds (20) made explicit.
#
# THIS IS THE EXPERIMENT THAT PRICES THE PAPER'S OWN NULL. It measures how much a single derangement
# draw moves the win count, which is what makes the 20-draw appendix paragraph interpretable rather
# than decorative. An UNSTABLE verdict on it is a limitation the paper states, not one it hides.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e49_stability"; MODULE_TITLE="derangement-draw stability, 20 draws"
MODULE_WHY="how much of the derangement result is one draw?"
MODULE_COST="free (CPU)"
MODULE_TIER="T1"
INPUTS=()
OUTPUTS=(results/e49_derangement_stability.json)
main() {
  run_py experiments/t43_derangement_stability.py --seeds 20 --device cpu \
    --out results/e49_derangement_stability.json
}
module_main "$@"
