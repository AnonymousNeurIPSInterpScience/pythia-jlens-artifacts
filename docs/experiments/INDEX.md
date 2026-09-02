# INDEX — every experiment document, generated

Written by `tools/build_provenance.py`. One row per document under
`docs/experiments/preregs/`. The narrative account is [`../context/CONTEXT.md`](../context/CONTEXT.md);
every claim with its tier is [`../context/RESULTS_TAXONOMY.md`](../context/RESULTS_TAXONOMY.md).

The verdict is read from **this document's own** results file, matched on the shared
id, not from the first file it happens to cite. A blank means the document cites no
file of its own, which is itself worth knowing.

| document | files | its own result | readout | verdict as stored |
|---|---:|---|---|---|
| [`CV3_margins.md`](preregs/CV3_margins.md) | 3 | `cv3_margins_410m.json` | CLEAN, IMMUNE | ACCEPT — SCORE EFFECT |
| [`CV4_capability_ladder.md`](preregs/CV4_capability_ladder.md) | 4 | `cv4_phase1_capability.json` | CLEAN, IMMUNE | Answer competence, 255 items with a target, 95% Wilson: 70m: top1 1.6% [0.6,4.0] / 160m: top1 1.6% [0.6,4.0] / |
| [`CV5_metric_sensitivity.md`](preregs/CV5_metric_sensitivity.md) | 3 | `cv5_metric_sensitivity_410m.json` | CLEAN, EXPOSED | REJECT — THE OPERATORS ARE FUNCTIONALLY IDENTICAL |
| [`CV6_per_family_ladder.md`](preregs/CV6_per_family_ladder.md) | 33 | `cv6_per_family_ladder.json` | CLEAN, EXPOSED, IMMUNE | REPLICATES |
| [`CV7_1b_rung.md`](preregs/CV7_1b_rung.md) | 3 | `cv7_1b_rung.json` | CLEAN | REPLICATES AT 1B |
| [`DN1_design_nulls.md`](preregs/DN1_design_nulls.md) | 2 | `dn1_design_nulls.json` | CLEAN, UNCLASSIFIED | UNCLEAR — a control did not fire: C5_design_vs_permutation |
| [`E28_read_ladder.md`](preregs/E28_read_ladder.md) | 30 | `ladder_Github_s0.json` | EXPOSED, RESCORED | — |
| [`E31_predictor_bakeoff.md`](preregs/E31_predictor_bakeoff.md) | 1 | `e31_local_bakeoff_410m.json` | EXPOSED | — |
| [`E33_logit_baseline.md`](preregs/E33_logit_baseline.md) | 17 | `e33_logit_baseline_410m_v2.json` | INHERITED, RESCORED | I INSIDE THE CORPUS SPREAD — 1 of 5 corpora fit an operator that is NO BETTER than not fitting at all. The cro |
| [`E36_qladder.md`](preregs/E36_qladder.md) | 2 | `e36_qladder_410m.json` | EXPOSED, RESCORED | REJECT S3 (the publishable null) — no fitting corpus produces a J-lens curve that goes strictly below the logi |
| [`E38_geometry.md`](preregs/E38_geometry.md) | 3 | `e38_jgeometry.json` | CLEAN, EXPOSED, IMMUNE | — |
| [`E48_crossover.md`](preregs/E48_crossover.md) | 2 | `e48_crossover_410m.json` | EXPOSED, IMMUNE | NOT REACHED — (J^P - logit) includes or exceeds 0 on ALL three OOD rungs. Reported straight as a positive-fram |
| [`E48b_containment.md`](preregs/E48b_containment.md) | 1 | `e48b_exposure_growth.json` | INHERITED | — |
| [`E48c_exposure_vs_read.md`](preregs/E48c_exposure_vs_read.md) | 1 | `e48c_exposure_vs_read.json` | INHERITED | EXPOSURE DOES NOT ORDER THE READ. Containment spans 10862x across the eight fitting corpora and the rank corre |
| [`E51_interaction.md`](preregs/E51_interaction.md) | 31 | `e51_interaction_variance.json` | EXPOSED, INHERITED, RESCORED | REPRODUCES — the headline interaction numbers now have a results file and match the prose to within 0.1 percen |
| [`E52_factorial.md`](preregs/E52_factorial.md) | 2 | `e52_factorial_410m.json` | EXPOSED, RESCORED | MATCHING CONFIRMED — the diagonal excess D = +0.00103 has a CI strictly above zero [+0.00001,+0.00277] AND exc |
| [`E53_ladder_summary.md`](preregs/E53_ladder_summary.md) | 1 | `e53_ladder_summary.json` | INHERITED | — |
| [`E55_matrix_robustness.md`](preregs/E55_matrix_robustness.md) | 1 | `e55_matrix_robustness.json` | INHERITED | — |
| [`E56_predictor_registry.md`](preregs/E56_predictor_registry.md) | 1 | `e56_predictor_registry.json` | INHERITED | — |
| [`E57_grid_variance_ci.md`](preregs/E57_grid_variance_ci.md) | 1 | `e57_factorial_cells_410m.json` | EXPOSED | — |
| [`E58_algebra_audit.md`](preregs/E58_algebra_audit.md) | 1 | `e58_algebra_audit.json` | EXPOSED | — |
| [`E59_read_dose_response.md`](preregs/E59_read_dose_response.md) | 28 | `e59_read_dose_410m.json` | EXPOSED | PARTIAL — read share moves +6.2 points (9.5% -> 15.7%), monotone=False. Report the curve; the headline is dose |
| [`E60_fitter_determinism.md`](preregs/E60_fitter_determinism.md) | 1 | `e60_fitter_determinism.json` | CLEAN | — |
| [`E61_randomized_network_null.md`](preregs/E61_randomized_network_null.md) | 1 | `e61_randomized_null_410m.json` | INHERITED | FLOOR IS LOW — a J fitted on a weight-randomized model reads BELOW the unfitted logit lens on 3/3 corpora (clo |
| [`E62_ladder1b_corrected_band.md`](preregs/E62_ladder1b_corrected_band.md) | 17 | `e62_interaction_b613.json` | EXPOSED, INHERITED | REPRODUCES — the headline interaction numbers now have a results file and match the prose to within 0.1 percen |
| [`E65_training_axis_floor.md`](preregs/E65_training_axis_floor.md) | 1 | `e65_ckpt_geometry_410m.json` | EXPOSED | GEOMETRY FLOOR AT step0 — 7 early checkpoint(s) exceed the 410M threshold and are geometrically degenerate. Ph |
| [`OS1_operator_space.md`](preregs/OS1_operator_space.md) | 1 | `os1_operator_space_410m.json` | UNCLASSIFIED | UNCLEAR — a control did not fire: C1_self_distance |
| [`P0_t52_parallel_port.md`](preregs/P0_t52_parallel_port.md) | 3 | `e52_factorial_410m_pooled.json` | EXPOSED | MATCHING CONFIRMED — the diagonal excess D = +0.00103 has a CI strictly above zero [+0.00001,+0.00277] AND exc |
| [`R1_grid_rstrip.md`](preregs/R1_grid_rstrip.md) | 5 | `e52_factorial_410m_rstrip.json` | EXPOSED, INHERITED | NO MATCHING (the publishable null) — D = +0.00085, CI [-0.00090,+0.00279] includes zero. The corpus effect est |
| [`R2_e48_rstrip_arm.md`](preregs/R2_e48_rstrip_arm.md) | 2 | `e48_crossover_410m_rstrip.json` | CLEAN, EXPOSED | NOT REACHED — (J^P - logit) includes or exceeds 0 on ALL three OOD rungs. Reported straight as a positive-fram |
| [`R3_ladder410_cpu.md`](preregs/R3_ladder410_cpu.md) | 20 | `r3_close_d2.json` | INHERITED, RESCORED, UNKNOWN | CLEARED — every paper-cited ladder number moves by less than one pooled seed SD (3.5e-03) and no reported orde |
| [`R4_corrections.md`](preregs/R4_corrections.md) | 17 | `r4_corrections.json` | EXPOSED, INHERITED | PRE-REGISTERED |
| [`R4b_e36_rstrip.md`](preregs/R4b_e36_rstrip.md) | 3 | `r4b_e36_flatness.json` | EXPOSED, INHERITED | REJECT OVERTURNED — flatter on only 1 of 5. FLAGGED FOR ADJUDICATION. S3's first half is no longer rejected |
| [`R5_corpus_axis_uncertainty.md`](preregs/R5_corpus_axis_uncertainty.md) | 1 | `r5_corpus_axis_uncertainty.json` | INHERITED | PRE-REGISTERED |
| [`R6_within_source_resampling.md`](preregs/R6_within_source_resampling.md) | 2 | `r6_within_source_410m.json` | EXPOSED | ACCEPT — corpus identity is a real factor. The between-source share is 0.9723 >= 0.70: holding the source fixe |
| [`R7_length_matched_pools.md`](preregs/R7_length_matched_pools.md) | 1 | `r7_matched_pools_410m.json` | EXPOSED | ACCEPT — the corpus effect is NOT substantially a lexical-composition effect. With the fitting pools matched o |
| [`R8_ladder_rstrip_S2.md`](preregs/R8_ladder_rstrip_S2.md) | 34 | `r8_ladder_flatness.json` | IMMUNE, RESCORED, UNKNOWN | UNCLEAR — the largest range over N is {'persist': 0.14182325681893931, 'min': 0.3188792203845251} of the betwe |
| [`R9_permutation_calibrated_min.md`](preregs/R9_permutation_calibrated_min.md) | 2 | `r9_permutation_calibrated_min.json` | CLEAN, IMMUNE | CALIBRATED. Under the SOURCE'S OWN statistic (`min`), the real operator exceeds every one of its 15 own layer- |

**38 documents.** Regenerate with `tools/build_provenance.py`.
