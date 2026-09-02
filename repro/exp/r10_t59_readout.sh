#!/usr/bin/env bash
# r10_t59_readout — is the stored E59 dose ladder legacy-scored?
#
# HAND-WRITTEN, not emitted, for the reason given in repro/exp/da1_derangement.sh. The stem is
# registered in tools/emit_exp_modules.py's KNOWN table.
#
# TIER T0: pure recomputation from results files already in the tree. No model, no operators,
# no GPU, no network. Runs in under a second.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="r10_t59_readout"; MODULE_TITLE="is the stored E59 dose ladder legacy-scored?"
MODULE_WHY="is the stored E59 dose ladder legacy-scored while the headline it bounds is corrected?"
MODULE_COST="free (CPU, <1 s)"
MODULE_TIER="T0"
INPUTS=(results/e59_read_dose_410m.json
        results/e52_factorial_410m.json
        results/e52_factorial_410m_rstrip.json
        results/e57_grid_variance_ci_rstrip.json)
OUTPUTS=(results/r10_t59_readout_reconciliation.json)
main() { run_py tools/r10_t59_readout_reconciliation.py; }
module_main "$@"
