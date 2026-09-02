#!/usr/bin/env bash
# cv8_c6_prefix_corpus — does the prefix corpus drive CV8's verdict?
#
# HAND-WRITTEN 2026-09-01. tools/emit_exp_modules.py derives its scope from the PAPER's citation
# set, so it does not emit a module for an adjudication that no paper sentence cites. These
# adjudications are part of the record and a reviewer may want to re-run them, so the module is
# written by hand in the generator's format. INPUTS/OUTPUTS are `provenance.inputs` and the
# result path of the stored file; the invocation is its `provenance.argv`.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="cv8_c6_prefix_corpus"; MODULE_TITLE="CV8 C6 — does the prefix corpus drive CV8's verdict?"
MODULE_WHY="does the prefix corpus drive CV8's verdict?"
MODULE_COST="free (CPU) — recomputation over stored results"
MODULE_TIER="T0"
INPUTS=(results/cv8_positional_extrapolation.json results/cv8_positional_extrapolation_github.json)
OUTPUTS=(results/cv8_c6_prefix_corpus_control.json)
main() { run_py experiments/cv8_c6_prefix_corpus_control.py; }
module_main "$@"
