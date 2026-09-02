#!/usr/bin/env bash
# e66d_s_quantization — is S quantization-limited, or did the hardware just happen to match?
#
# HAND-WRITTEN 2026-09-01. tools/emit_exp_modules.py derives its scope from the PAPER's citation
# set, so it does not emit a module for an adjudication that no paper sentence cites. These
# adjudications are part of the record and a reviewer may want to re-run them, so the module is
# written by hand in the generator's format. INPUTS/OUTPUTS are `provenance.inputs` and the
# result path of the stored file; the invocation is its `provenance.argv`.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e66d_s_quantization"; MODULE_TITLE="E66d — is S quantization-limited?"
MODULE_WHY="is S quantization-limited, or did the hardware just happen to match?"
MODULE_COST="free (CPU) — reads 2 stored operators, ~2 min"
MODULE_TIER="T1"
INPUTS=(results/lenses/misc/e66_D1_refit_410m_pilecc_s0.pt results/e48/lens_INSTREAM_Pile-CC_410m_n200_s0.pt results/e66b_determinism_floor.json)
OUTPUTS=(results/e66d_s_quantization_limited.json)
main() { run_py experiments/e66d_s_quantization_limited.py; }
module_main "$@"
