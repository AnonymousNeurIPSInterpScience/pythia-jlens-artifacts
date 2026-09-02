#!/usr/bin/env bash
# e65t_threshold — is E65 Phase 0's geometry floor real, or a mis-justified threshold?
#
# HAND-WRITTEN 2026-09-01. tools/emit_exp_modules.py derives its scope from the PAPER's citation
# set, so it does not emit a module for an adjudication that no paper sentence cites. These
# adjudications are part of the record and a reviewer may want to re-run them, so the module is
# written by hand in the generator's format. INPUTS/OUTPUTS are `provenance.inputs` and the
# result path of the stored file; the invocation is its `provenance.argv`.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e65t_threshold"; MODULE_TITLE="E65T — is E65 Phase 0's geometry floor real?"
MODULE_WHY="is E65 Phase 0's geometry floor real, or a mis-justified threshold?"
MODULE_COST="free (CPU) — downloads 19 pythia-410m checkpoints (~30 GB) if not cached, ~1 min cached"
MODULE_TIER="T1"
INPUTS=(results/e65_ckpt_geometry_410m.json results/e38_jgeometry.json)
OUTPUTS=(results/e65t_threshold_adjudication.json)
main() { run_py experiments/e65t_threshold_adjudication.py; }
module_main "$@"
