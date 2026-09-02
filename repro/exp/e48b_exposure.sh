#!/usr/bin/env bash
# E48b — containment vs stream coverage, for every candidate corpus.
# E48's rung classification rests on a GROWTH RATE that no stored file contained for the OOD
# corpora. This computes the whole curve from the per-shard bitmaps, which is free.
# The index MUST be rebuilt with exactly the corpus set present when the shards were written --
# OOD_News_2024_dedup is a strict SUBSET of OOD_News_2024, so including it leaves the total index
# length unchanged while shifting every owner bit. --exclude handles that; do not remove it.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e48b"; MODULE_TITLE="exposure growth curve"
MODULE_WHY="does the OOD designation earn itself, on a rate rather than a level?"
MODULE_COST="free (~15 min CPU)"
MODULE_TIER="T1"
INPUTS=(results/m1/merged results/m1/m1_5shard.json corpora)
OUTPUTS=(results/e48b_exposure_growth.json)
main() { run_py experiments/t48b_exposure_growth.py --exclude OOD_News_2024_dedup; }
module_main "$@"
