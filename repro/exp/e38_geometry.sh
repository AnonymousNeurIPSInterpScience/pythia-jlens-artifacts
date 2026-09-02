#!/usr/bin/env bash
# e38_geometry — the operator geometry behind the 160M precondition result.
#
# HAND-WRITTEN. results/e38_jgeometry.json has no `provenance.argv`; the script takes only --device
# and --out and the stored file records the cpu run.
#
# WHAT THIS IS FOR, stated because the paper is careful about it: E38 is a PRECONDITION result, not
# a cause. `docs/experiments/descriptions/E38_geometry.md` grades it Tier B and says so in those
# words. Do not read the geometry as explaining the 160M gap.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e38_geometry"; MODULE_TITLE="operator geometry (E38-E42 block)"
MODULE_WHY="what does the fitted operator look like where the free lens beats it?"
MODULE_COST="free (CPU)"
MODULE_TIER="T1"
INPUTS=()
OUTPUTS=(results/e38_jgeometry.json)
main() { run_py experiments/t38_jgeometry.py --device cpu --out results/e38_jgeometry.json; }
module_main "$@"
