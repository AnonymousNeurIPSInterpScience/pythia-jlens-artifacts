#!/usr/bin/env bash
# E48 — the S3 crossover. Adjudicates PREREG_E48_CROSSOVER.md + Amendment 1 (408fa5c / 5869b63).
# Verdict on record: NOT REACHED.
# FLAGS THAT ARE LOAD-BEARING, and why:
#   (no --rstrip)        PRIMARY is the UNSTRIPPED path. Only it reproduces E33 (control C0), and
#                        clause (d) is defined against E33's numbers. --rstrip is the disclosed
#                        sensitivity arm; it reads a different token on 157/551 items.
#   --n-derangements 5   a derangement is a random object; 5 draws x 3 seeds x 8 corpora = 120.
#   (no --allow-failed-gate)  if the competence gate fails a rung, clause (b) must be RE-SPECIFIED
#                        by the operator BEFORE scoring. Selecting rungs after seeing a read is a
#                        researcher degree of freedom.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e48"; MODULE_TITLE="the S3 crossover"
MODULE_WHY="does the unfitted logit lens overtake J^P when P leaves the training stream?"
MODULE_COST="free (~25 min CPU)"
MODULE_TIER="T1"
INPUTS=(results/e48_competence_gate_410m.json
        results/e33_logit_baseline_410m_v2.json
        results/e48/lens_OOD_News_2024_410m_n200_s0.pt)
OUTPUTS=(results/e48_crossover_410m.json)
main() {
  run_py experiments/t48_crossover.py --device cpu --n-derangements 5 --n-boot 10000
  if [ "${RUN_RSTRIP_ARM:-0}" = "1" ]; then
    info "sensitivity arm: the stripped scoring path (C0 cannot fire here, by construction)"
    run_py experiments/t48_crossover.py --device cpu --rstrip \
      --out results/e48_crossover_410m_rstrip.json
  fi
}
module_main "$@"
