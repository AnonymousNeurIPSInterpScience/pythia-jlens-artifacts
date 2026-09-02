#!/usr/bin/env bash
# e37_rank_160m — is rank the cause of the small-model floor? (160M)
#
# HAND-WRITTEN. No `provenance.argv`. Flags read off the stored file: model pythia-160m-deduped,
# lens lens_160m_n200_db128_pen.pt. Band ([4..9]) is derived by the script, not a flag.
#
# The paper's 83.7% at this rung is the LOGIT arm, as at 70M. See e37_rank_70m.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e37_rank_160m"; MODULE_TITLE="rank-1 unembedding cost at 160M"
MODULE_WHY="is the small-model floor a rank effect in W_U?"
MODULE_COST="free (CPU)"
MODULE_TIER="T1"
INPUTS=(results/lenses/ladder/lens_160m_n200_db128_pen.pt)
OUTPUTS=(results/e37_rank_ablation_160m_wikitext.json)
main() {
  run_py experiments/t37_rank_ablation.py --model 160m --device cpu \
    --lens lens_160m_n200_db128_pen.pt \
    --out results/e37_rank_ablation_160m_wikitext.json
}
module_main "$@"
