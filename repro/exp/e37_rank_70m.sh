#!/usr/bin/env bash
# e37_rank_70m — is rank the cause of the small-model floor? (70M)
#
# HAND-WRITTEN. No `provenance.argv`. Flags read off the stored file: model pythia-70m-deduped,
# lens lens_70m_n200_db128_pen.pt. The band ([2,3]) is derived by the script from the model and is
# not a flag.
#
# READ THE OUTPUT CAREFULLY. The paper quotes 71.9% here and that number is the LOGIT arm
# (by_rank.top_1 vs top_full under `min` for J=I), not the fitted lens, whose pair runs the other
# way. The file carries both; the appendix says which.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e37_rank_70m"; MODULE_TITLE="rank-1 unembedding cost at 70M"
MODULE_WHY="is the small-model floor a rank effect in W_U?"
MODULE_COST="free (CPU)"
MODULE_TIER="T1"
INPUTS=(results/lenses/ladder/lens_70m_n200_db128_pen.pt)
OUTPUTS=(results/e37_rank_ablation_70m_wikitext.json)
main() {
  run_py experiments/t37_rank_ablation.py --model 70m --device cpu \
    --lens lens_70m_n200_db128_pen.pt \
    --out results/e37_rank_ablation_70m_wikitext.json
}
module_main "$@"
