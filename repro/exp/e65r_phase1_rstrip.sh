#!/usr/bin/env bash
# e65r_phase1_rstrip — does E65 Phase 1's C1 failure survive the readout correction?
#
# HAND-WRITTEN 2026-09-01. tools/emit_exp_modules.py derives its scope from the PAPER's citation
# set, so it does not emit a module for an adjudication that no paper sentence cites. These
# adjudications are part of the record and a reviewer may want to re-run them, so the module is
# written by hand in the generator's format. INPUTS/OUTPUTS are `provenance.inputs` and the
# result path of the stored file; the invocation is its `provenance.argv`.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e65r_phase1_rstrip"; MODULE_TITLE="E65R — E65 Phase 1 at the corrected readout"
MODULE_WHY="does E65 Phase 1's C1 failure survive the readout correction?"
MODULE_COST="free (CPU) — 10 stored operators + 10 cached checkpoints, ~14 min"
MODULE_TIER="T1"
INPUTS=(results/e65_lenses/lens_410m_step0_trainval.pt results/e65_lenses/lens_410m_step512_trainval.pt results/e65_lenses/lens_410m_step1000_trainval.pt results/e65_lenses/lens_410m_step2000_trainval.pt results/e65_lenses/lens_410m_step4000_trainval.pt results/e65_lenses/lens_410m_step8000_trainval.pt results/e65_lenses/lens_410m_step16000_trainval.pt results/e65_lenses/lens_410m_step32000_trainval.pt results/e65_lenses/lens_410m_step64000_trainval.pt results/e65_lenses/lens_410m_step143000_trainval.pt results/e65_ckpt_readout_410m.json results/e48_crossover_410m.json results/e48_crossover_410m_rstrip.json)
OUTPUTS=(results/e65r_phase1_rstrip_rescore.json)
main() { run_py experiments/e65r_phase1_rstrip_rescore.py; }
module_main "$@"
