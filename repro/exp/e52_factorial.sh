#!/usr/bin/env bash
# E52 — the fit x read factorial. The missing cell: J^Q read on Q.
# PREREG: docs/experiments/preregs/superseded/PREREG_E52_FACTORIAL.md, written before the first number existed.
# Answers the one question neither E36 nor E48 can: is the corpus effect a MATCHING effect
# (fit-read relationship) or a property of the fitting corpus alone?
#
# LOAD-BEARING: every one of the 24 operators comes from ONE fitter (trainval.py, N=200,
# --band 9,21). E48's in-stream arm used fastfit and its OOD arm used jacobian_for_prompt; a
# factorial cannot tolerate a fitter difference confounded with the axis under test. That is what
# the INSTREAM refits were produced for, and lens_path() will not load an E28 fastfit lens.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e52"; MODULE_TITLE="fit x read factorial"
MODULE_WHY="can a lens fitted on an OOD corpus read latent structure on an OOD corpus?"
MODULE_COST="free (~1.3-2 h CPU)"
MODULE_TIER="T1"
INPUTS=(results/e48_crossover_410m.json
        results/e48/lens_INSTREAM_Pile-CC_410m_n200_s0.pt
        results/e48/lens_OOD_arXiv_2023_410m_n200_s0.pt
        corpora)
OUTPUTS=(results/e52_factorial_410m_rstrip.json results/e57_factorial_cells_410m_rstrip.json)
# THE PAPER CITES THE CORRECTED-READOUT ARM (Table 5, row E52/R1), and this module ran without
# --rstrip until 2026-08-24, rebuilding the superseded file instead. The argv below is the one
# recorded in provenance.argv of results/e52_factorial_410m_rstrip.json. It also emits the cells
# file that E57 and R5 consume, which is why both are declared as outputs.
#   legacy arm, for the readout comparison only:  bash repro/exp/e52_factorial.sh --legacy
main() {
  if [ "${1:-}" = "--legacy" ]; then
    run_py experiments/t52_factorial.py --device cpu --workers 6
  else
    run_py experiments/t52_factorial.py --device cpu --workers 6 --rstrip \
      --out results/e52_factorial_410m_rstrip.json \
      --cells-out results/e57_factorial_cells_410m_rstrip.json
  fi
}
module_main "$@"
