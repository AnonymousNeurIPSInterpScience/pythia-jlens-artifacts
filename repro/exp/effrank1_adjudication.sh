#!/usr/bin/env bash
# effrank1_adjudication — which effective rank does the paper print, and do the two stored values conflict?
#
# HAND-WRITTEN 2026-09-01. tools/emit_exp_modules.py derives its scope from the PAPER's citation
# set, so it does not emit a module for an adjudication that no paper sentence cites. These
# adjudications are part of the record and a reviewer may want to re-run them, so the module is
# written by hand in the generator's format. INPUTS/OUTPUTS are `provenance.inputs` and the
# result path of the stored file; the invocation is its `provenance.argv`.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="effrank1_adjudication"; MODULE_TITLE="EFFRANK1 — eff_rank 371 vs 725.888"
MODULE_WHY="which effective rank does the paper print, and do the two stored values conflict?"
MODULE_COST="free (CPU) — downloads 4 Pythia checkpoints (~5 GB) if not cached, ~2 min"
MODULE_TIER="T1"
INPUTS=(results/e38_jgeometry.json results/e65_ckpt_geometry_410m.json)
OUTPUTS=(results/effrank1_adjudication.json)
main() { run_py experiments/effrank1_adjudication.py; }
module_main "$@"
