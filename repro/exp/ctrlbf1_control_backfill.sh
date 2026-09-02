#!/usr/bin/env bash
# ctrlbf1_control_backfill — record E35 M1-C3, E35 M1-C4 and P0 C2 from existing outputs
#
# HAND-WRITTEN 2026-09-01. tools/emit_exp_modules.py derives its scope from the PAPER's citation
# set, so it does not emit a module for an adjudication that no paper sentence cites. These
# adjudications are part of the record and a reviewer may want to re-run them, so the module is
# written by hand in the generator's format. INPUTS/OUTPUTS are `provenance.inputs` and the
# result path of the stored file; the invocation is its `provenance.argv`.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="ctrlbf1_control_backfill"; MODULE_TITLE="CTRLBF1 — three already-completed controls, recorded"
MODULE_WHY="record E35 M1-C3, E35 M1-C4 and P0 C2 from existing outputs"
MODULE_COST="free (CPU) — recomputation over stored results"
MODULE_TIER="T0"
INPUTS=(results/e48b_exposure_growth.json results/e52_factorial_410m_rstrip.json results/e52_factorial_410m_rstrip_w1.json results/e52_factorial_410m_rstrip_w2.json results/e52_factorial_410m_rstrip_w8.json)
OUTPUTS=(results/ctrlbf1_control_backfill.json)
main() { run_py experiments/ctrlbf1_control_backfill.py; }
module_main "$@"
