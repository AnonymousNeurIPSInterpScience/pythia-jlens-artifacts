#!/usr/bin/env bash
# e33_logit_v2 — the free-lens (logit) baseline at the corrected readout.
#
# HAND-WRITTEN. results/e33_logit_baseline_410m_v2.json has no `provenance.argv`. Every flag below
# is read off that file's own stored configuration, not from memory:
#   model         "EleutherAI/pythia-410m-deduped"      -> --model 410m
#   band          [9..21]  (13 layers)                  -> --band 9,22   <- SEE BELOW
#   device        "cpu"                                 -> --device cpu
#   derangement   "random"                              -> --derangement random
#   derangement_seed 0                                  -> --derangement-seed 0
#   reference_lens "e28_Github_410m_n400_s0.pt"         -> --ref-lens ...
#
# --derangement random IS LOAD-BEARING. v1 of this experiment used `cyclic` and its control failed
# on 4 of 6 cells; v2 is the re-run that supersedes it. Running this module with the default would
# silently reproduce v1's design, which is why the flag is stated rather than left implicit.
#
# --band 9,22 IS LOAD-BEARING, AND IT LOOKS WRONG. It is not. `--band`'s help string says
# "lo,hi inclusive"; the implementation is `range(lo, hi)` (t33_logit_baseline.py:81), which is
# EXCLUSIVE of hi. The help is wrong and has been since the file was created -- git holds two
# versions of that script and the parsing line is byte-identical in both, so no older code exists
# to recover. `--band 9,22` is therefore the ONLY invocation of this never-changed parser that
# produces the stored band [9..21]; it is derived, not guessed.
#
# This line read `--band 9,21` until 2026-08-30 and that UNDERSHOT the band by one layer, giving
# [9..20] (12 layers). The T1 sweep caught it: 86 numeric fields moved and the experiment's own
# VERDICT flipped from "I INSIDE THE CORPUS SPREAD" to "I BELOW ALL FIVE". The band itself is not
# in question -- [9,21] is `int(0.38L)..int(0.92L)` at L=24, the rule tests/test_band_rule.py
# asserts, normalised from the anchor's section 4.1 workspace range (~L38 to ~L92). The stored
# result was always on the correct band; only this recipe was wrong.
# Full diagnosis: docs/reproducibility/T1_FINDING_E33_BAND.md
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e33_logit_v2"; MODULE_TITLE="free-lens baseline, corrected readout"
MODULE_WHY="what does the unfitted lens score, on the band and readout the paper uses?"
MODULE_COST="free (CPU)"
MODULE_TIER="T1"
INPUTS=(results/lenses/e28/e28_Github_410m_n400_s0.pt)
OUTPUTS=(results/e33_logit_baseline_410m_v2.json)
main() {
  run_py experiments/t33_logit_baseline.py \
    --model 410m --band 9,22 --device cpu \
    --derangement random --derangement-seed 0 \
    --ref-lens e28_Github_410m_n400_s0.pt \
    --out results/e33_logit_baseline_410m_v2.json
}
module_main "$@"
