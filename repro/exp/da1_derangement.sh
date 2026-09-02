#!/usr/bin/env bash
# da1_derangement — what does the layer-derangement control actually identify?
#
# HAND-WRITTEN, not emitted. tools/emit_exp_modules.py derives its scope from the paper tree, and
# this reviewer-facing clone does not carry paper/, so the generator cannot see this result. The
# stem is registered in that tool's KNOWN table so a tree that DOES carry the paper emits an
# identical module. The invocation below is `provenance.argv` of the stored result.
#
# TIER T1: needs the fitted operators from the artifact mirror. Fetch them with
#   bash repro/03_fetch_artifacts.sh --all
# or fetch just the 24 this module reads (410M, n=200, seeds 0-2, eight corpora, ~0.94 GB).
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="da1_derangement"; MODULE_TITLE="what does the layer-derangement control identify"
MODULE_WHY="what does the layer-derangement control actually identify?"
MODULE_COST="free (CPU, ~10 min)"
MODULE_TIER="T1"
INPUTS=(results/e48_crossover_410m_rstrip.json
        results/lenses/e28/e28_Wikipedia_en_410m_n200_s0.pt
        results/lenses/e28/e28_USPTO_Backgrounds_410m_n200_s0.pt
        results/lenses/e28/e28_Pile-CC_410m_n200_s0.pt
        results/lenses/e28/e28_StackExchange_410m_n200_s0.pt
        results/lenses/e28/e28_Github_410m_n200_s0.pt
        results/e48/lens_OOD_News_2024_410m_n200_s0.pt
        results/e48/lens_OOD_arXiv_2023_410m_n200_s0.pt
        results/e48/lens_OOD_CommonPile_410m_n200_s0.pt)
OUTPUTS=(results/da1_derangement_adjudication_410m.json)
main() { run_py experiments/da1_derangement_adjudication.py --device cpu; }
module_main "$@"
