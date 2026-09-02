# E65T — is E65 Phase 0's "geometry floor" real, or an artifact of a mis-justified threshold?

**PRE-REGISTERED before the adjudication script was written or run.** Recomputation only.
**No new threshold is invented here.** The only thresholds used are ones already fixed and stored
elsewhere in this repository.

---

## WHY

`docs/experiments/preregs/E65_training_axis_floor.md` §VERDICT carries, verbatim:

> ⚠ **PRE-REGISTRATION DEFECT, FLAGGED NOT FIXED.** The registered rule flags a checkpoint
> degenerate if mean cosine exceeds **0.031** (the 410M value), and 7 mid-training checkpoints do
> (0.033–0.043), so the rule fired "GEOMETRY FLOOR EXISTS". **The threshold was mis-justified in
> this document**: it was defended as "above it the readout is in the regime E38/E39 shows no
> transport can separate", but that regime is cos ≥0.92 **with rank 1**, not "anything above the
> 410M value". The co-measured effective rank settles it independently. The rule fired; the finding
> is not real. Recorded here rather than silently corrected — **the operator rules on it.**

`results/e65_ckpt_geometry_410m.json` stores `n_degenerate = 7` and
`degenerate_checkpoints = [step1000, step2000, step4000, step8000, step16000, step32000, step64000]`.

`paper/audit_paper_5pp.tex:918` restates it as fact: *"At 410M the concept rows of $W_U$ are clean at
initialisation, **degenerate from step 1000 through step 64000**, and below the threshold again at
step 143000."* The paper carries **no disclosure** of the flagged defect.

This registration asks whether the 7 flagged checkpoints are in the degenerate regime **as that
regime is defined by the experiment the threshold borrowed its authority from** — E38/E39.

## DESIGN

For each of the 19 published `pythia-410m-deduped` checkpoints in
`e65_ckpt_geometry_410m.json -> checkpoints_requested`, recompute on the same 1310 concept-token
rows of `W_U`, at float32 on CPU:

`mean_cos`, `er_energy90`, `er_entropy`, `sigma1_energy_share`

using the definitions adjudicated in `EFFRANK1_effective_rank_adjudication.md` and stored in
`results/effrank1_adjudication.json`. All 19 revisions are already in the local HF cache.

## THE REGIME, AS ALREADY DEFINED ELSEWHERE — no new threshold

The degenerate regime is not defined here. It is read off `results/e38_jgeometry.json -> E39`, which
is what the E65 prereg's justification appealed to:

| model | `mean_cos_I` | `eff_rank_I` (90% energy) |
|---|---|---|
| 70m | 0.9585 | **1** |
| 160m | 0.9218 | **1** |
| 410m (final) | 0.0307 | **371** |
| 1b | 0.0199 | **508** |

**The degenerate regime as E38 exhibits it is `mean_cos >= 0.92` AND `er_energy90 == 1`.**
Both conjuncts come from stored values; neither is chosen by me.

## PRIMARY

For the 7 flagged checkpoints: `er_energy90`, and `mean_cos`.

## DECISION RULE — fixed before running

* **FINDING NOT REAL** — none of the 7 flagged checkpoints satisfies the E38 degenerate regime
  (i.e. every one has `er_energy90 > 1`, or `mean_cos < 0.92`). The 0.031 threshold does not select
  the regime it was justified by; the "geometry floor" is an artifact of the threshold and the
  paper's sentence at line 918 must be corrected or disclosed.
* **FINDING REAL** — all 7 satisfy the E38 degenerate regime. The threshold, however
  mis-argued, selected the right set; the paper's sentence stands and the prereg's self-flag is
  withdrawn.
* **PARTIAL** — some but not all 7 satisfy it. Report the split per checkpoint; make no floor claim
  either way and hand the operator the table.

## CONTROLS, each with the number it must produce

* **C1 — the recomputation reproduces E65 Phase 0.** Recomputed `mean_cos` must match
  `e65_ckpt_geometry_410m.json -> by_checkpoint.<step>.mean_cos` to `abs_diff <= 1e-6` at all 19
  checkpoints, and recomputed `er_entropy` to `abs_diff <= 1e-3`. If C1 fails this is not measuring
  E65's object.
* **C2 — the regime definition is not vacuous.** Applied to 70M and 160M final `W_U`, the E38
  degenerate regime must return **TRUE for both**. A regime test that no matrix satisfies would make
  "FINDING NOT REAL" unfalsifiable. *This is the control that makes the rule able to fail.*
* **C3 — the regime definition is not universal.** Applied to 410M final and 1B final `W_U`, it must
  return **FALSE for both**.
* **C4 — the flagged set is reproduced.** The set of checkpoints with `mean_cos > 0.031` must be
  exactly the 7 stored in `degenerate_checkpoints`. If it is not, the stored file and this
  recomputation disagree about the input to the question.

## DECLARED BIAS

This registration expects **FINDING NOT REAL**, because `e65_ckpt_geometry_410m.json` already stores entropy ranks of
736–834 at the 7 flagged checkpoints and the E38 degenerate regime has `er_energy90 == 1`. this registration is
predicting the outcome the E65 prereg predicted. The result that would overturn me is any flagged
checkpoint returning `er_energy90 == 1`.

## COST

Free. CPU, ~1 min, all 19 revisions cached. No GPU, no download, no fit.

## OUTPUT

`results/e65t_threshold_adjudication.json`
