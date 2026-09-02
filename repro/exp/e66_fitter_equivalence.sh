#!/usr/bin/env bash
# e66_fitter_equivalence — do the two fitters agree on CUDA, and is the released operator the one
# its own provenance claims?
#
# THIS IS THE ONE OPEN REPRODUCIBILITY QUESTION IN THE PROGRAMME, and it is the only reason this
# module is worth a GPU. The stored verdict is:
#
#     B_vs_C max_rel   2.534676e-03   against C1 rel_tolerance 1e-3   ->  fires false
#     read_differences A-B 0.0   B-C 0.0   A-C 0.0
#     VERDICT: UNCLEAR — the stored lens is not what its own provenance says it is
#
# READ BOTH HALVES OF THAT. The reads are BIT-IDENTICAL, so no number in the paper moves, and cv5
# prices a perturbation of this size as quiet (S_rank 0.092, S_z 0.112). But the `.pt` operators are
# RELEASED, and a reviewer who refits from the recipe lands exactly where E66 landed. Better that
# they find a stated tolerance here than an unexplained mismatch on their own.
#
# THE RUN. 410M, N=200, Pile-CC, seed 0, CUDA, fp32, TF32 OFF. About 20 minutes on one L40S and
# roughly $0.30. Choose the GPU by fp32 TFLOPS — this workload is compute-bound, and
# L40S 91.6 > 4090 82.6 > H100 67 > A100 19.5. An H100 works and is about 40% slower.
#
# TF32 IS FORBIDDEN and must be set in torch, not only via a driver override. The script does this
# itself; if you are porting the invocation elsewhere, carry that with it. TF32 silently truncates
# the mantissa and this experiment measures a 2.5e-3 relative difference.
#
# WHAT AN ANSWER LOOKS LIKE. Either the operators are refit and released consistent with their
# provenance, or the tolerance is stated in ARTIFACTS.md as a known property of the release. Both
# are acceptable; leaving it UNCLEAR after publishing the operators is not.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
MODULE_ID="e66_fitter_equivalence"; MODULE_TITLE="fitter equivalence on CUDA"
MODULE_WHY="is the released operator the one its provenance claims, and do the two fitters agree?"
MODULE_COST="gpu:0.30"
MODULE_TIER="T2"
INPUTS=(corpora/Pile-CC.jsonl)
OUTPUTS=(results/e66_fitter_equivalence_cuda.json)
main() {
  run_py experiments/t66_fitter_equivalence_cuda.py \
    --device cuda \
    --out results/e66_fitter_equivalence_cuda.json
}
module_main "$@"
