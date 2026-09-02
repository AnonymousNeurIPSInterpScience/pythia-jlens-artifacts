#!/usr/bin/env bash
# E48 PREREQUISITE — model competence on every fitting corpus.
# A negative read on a corpus the model cannot model is model-confusion, not distributional bias,
# and the two are NOT separable after the fact. t48_crossover.sh refuses to run without this.
# NOTE trainval.py's own model_cross_entropy is NaN in every fit ever run (swallowed exception at
# trainval.py:89-102); this script is the correct path and does not swallow anything.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e48-gate"; MODULE_TITLE="model-competence gate"
MODULE_WHY="separates operator collapse / model incompetence / distributional bias"
MODULE_COST="free (~40 min CPU)"
MODULE_TIER="T1"
INPUTS=(corpora/Wikipedia_en.jsonl corpora/OOD_News_2024.jsonl)
OUTPUTS=(results/e48_competence_gate_410m.json)
main() { run_py experiments/t48_competence_gate.py --model 410m --device cpu; }
module_main "$@"
