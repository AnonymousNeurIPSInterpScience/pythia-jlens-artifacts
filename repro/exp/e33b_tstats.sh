#!/usr/bin/env bash
# E33b — the per-corpus t statistics from E33, stored.
# "Github 0.02808, t = -1.65" is cited in SPINE.tex, HANDOFF.md, CLAUDE.md and the E48
# pre-registration, where clause (d) DEPENDS on it, and it lived in no results file.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e33b"; MODULE_TITLE="E33 t statistics, stored"
MODULE_WHY="a prose-only number was underwriting a pre-registered clause"
MODULE_COST="free"
MODULE_TIER="T0"
INPUTS=(results/e33_logit_baseline_410m_v2.json)
OUTPUTS=(results/e33b_tstats_410m.json)
main() { run_py experiments/t33b_store_tstats.py; }
module_main "$@"
