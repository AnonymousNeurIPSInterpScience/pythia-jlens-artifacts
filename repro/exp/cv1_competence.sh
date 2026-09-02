#!/usr/bin/env bash
# cv1_competence — can the model do the task at all?
#
# HAND-WRITTEN, not generated. The stored results/cv1_answer_competence.json predated provenance
# stamping and carried no `provenance.argv`, so there was no recorded command to derive a module
# from. Rather than guess, the run was reconstructed and re-executed until it reproduced the stored
# payload BIT-IDENTICALLY; the file now carries a provenance block and this is its argv.
#
# THE FIRST RECONSTRUCTION WAS WRONG AND THE CHECK CAUGHT IT. `--models 70m,160m,410m` looked right
# (it is the script's own default) and produced every number in the paper correctly, but silently
# dropped the 1B rung the stored file contains. Comparing payloads rather than eyeballing the
# printed verdict is what surfaced it. This is why OUTPUTS is verified and not merely present.
#
# WHY IT MATTERS. Answer competence is the PRECONDITION for every absolute recovery claim in the
# paper: Pythia-410M answers 4.7% of the battery. `docs/validity/DIAGNOSIS.md` is where that gets
# cashed out, and it is the reason the absolute readout claims are blocked rather than reported.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="cv1_competence"; MODULE_TITLE="answer competence at 70M/160M/410M/1B"
MODULE_WHY="can the model do the task at all? -- the precondition for any recovery claim"
MODULE_COST="free (CPU, ~10 min; downloads four Pythia checkpoints on first run)"
MODULE_TIER="T1"
INPUTS=()
OUTPUTS=(results/cv1_answer_competence.json)
main() {
  run_py experiments/cv1_answer_competence.py --models 70m,160m,410m,1b
}
module_main "$@"
