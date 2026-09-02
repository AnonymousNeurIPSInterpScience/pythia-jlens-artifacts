#!/usr/bin/env bash
# E36 — THE Q-LADDER. Vary the READ distribution, hold the fitted operator fixed.
#
# This is the experiment E48 is not. E48 varied P (the FITTING corpus) and found exposure does not
# order the read. Q never moved. EQ2's unbuyable term is ||J^P - J^Q||, where Q is the
# distribution the activations are drawn from, and nothing in this programme has ever varied it.
#
# FLAGS THAT ARE LOAD-BEARING:
#   --seeds 3       three prefix draws per rung. SPINE A1.1: intervals come from seeds, never
#                   from residual scatter across a nested axis.
#   (no --rungs)    default = Q0 (no prefix) + every corpus with a MEASURED containment from
#                   E48b + a token-shuffled rung. The x-axis is measured exposure, not a rung
#                   number; that is what E35/M1 was built to supply.
#
# PREREQUISITE: e48b_exposure_growth.json supplies the x-axis. Without it the ladder falls back to
# a categorical axis, which the pre-registration allows but which is strictly weaker.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e36"; MODULE_TITLE="the Q-ladder"
MODULE_WHY="does the fitted lens lose its advantage as the READ distribution leaves P?"
MODULE_COST="free (~1.5-2 h CPU)"
MODULE_TIER="T1"
INPUTS=(results/e48b_exposure_growth.json
        results/e33_logit_baseline_410m_v2.json
        results/lenses/e28/e28_Pile-CC_410m_n400_s0.pt
        corpora)
OUTPUTS=(results/e36_qladder_410m_rstrip.json)
# THE PAPER CITES THE CORRECTED-READOUT ARM. Until 2026-08-24 this module ran WITHOUT --rstrip and
# therefore reproduced results/e36_qladder_410m.json, the PRE-CORRECTION file, while the paper's
# register row names e36_qladder_410m_rstrip.json. A canonical invocation that rebuilds a superseded
# number is worse than none: it looks like corroboration. The argv below is the one recorded in
# provenance.argv of the file the paper actually cites.
#   legacy arm, for the readout comparison only:  bash repro/exp/e36_qladder.sh --legacy
main() {
  if [ "${1:-}" = "--legacy" ]; then
    run_py experiments/t36_qladder.py --device cpu --workers 4
  else
    run_py experiments/t36_qladder.py --device cpu --workers 4 --rstrip
  fi
}
module_main "$@"
