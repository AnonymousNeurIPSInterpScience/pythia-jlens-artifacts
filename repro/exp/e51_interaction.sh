#!/usr/bin/env bash
# E51 — the corpus x concept-set interaction decomposition.
# Exists because the programme's HEADLINE number (6.8% / 1.9% / 91.3%) was in six documents and
# in no results file, and no script in ANY commit computed it. This recomputes it from the stored
# ladders and reports whether it reproduces the prose. It does; the "4614x noise floor" does not.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e51"; MODULE_TITLE="corpus x set interaction decomposition"
MODULE_WHY="the paper's most novel claim had no source file; this gives it one"
MODULE_COST="free (CPU, seconds)"
MODULE_TIER="T0"
INPUTS=(results/ladder410 results/ladder1b)
OUTPUTS=(results/e51_interaction_variance.json)
main() { run_py experiments/t51_interaction_variance.py --agg persist --n-min 75 --n-perm 2000; }
module_main "$@"
