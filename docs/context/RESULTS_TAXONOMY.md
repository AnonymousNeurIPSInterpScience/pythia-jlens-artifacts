# RESULTS_TAXONOMY — every claim, with its tier and its file

**Rewritten 2026-08-22**, after the post-review slate. The previous version predates the readout
correction and the aggregation ruling and is archived.

**Nothing here is from memory.** Every row names a results file. Every file's hash, producing script
and readout class is stamped into its experiment document by `tools/build_provenance.py`; the
one-row-per-experiment index is [`experiments/INDEX.md`](../experiments/INDEX.md).

**Read [`AGGREGATION_POLICY.md`](AGGREGATION_POLICY.md) before citing any number.** `min` is
primary; `persist` is a secondary arm labelled as selected with knowledge of the control outcome.
Where the two disagree, both are given and `min` leads.

---

## 0. HOW TO READ THIS

### The tiers

| tier | means |
|---|---|
| **A** | recomputes from stored artifacts; every named control fired; registered before it ran; on the corrected readout |
| **B** | recomputes, but carries a named weakness — an unfired control, a small n, an unregistered threshold, or a legacy-readout dependence that has not been re-scored |
| **D** | **retracted.** Do not restate under any framing |

A claim below Tier A is written **`DIRECTIONAL RESULT (REQUIRES <the failing axis>)`** with the tag
**before** the number, never after.

### The ten axes a claim is graded on

Registration · stored artifact · named control · control power · replication unit · effect size
against a noise floor · aggregation robustness · readout correctness · leave-one-out stability ·
disclosed deviations. A claim that fails any axis is at best Tier B, and the failing axis is named.

### Two standing facts that qualify everything

1. **The readout.** 157 of 551 released items were scored at a token that does not occur in the
   tokenised sequence. Every affected result has been re-scored; `tools/readout_exposure.py` says
   which. A claim on the legacy readout is **Tier B at best**, and marked.
2. **The replication unit.** Draw counts overstate evidence when draws share an operator and a
   cache. Where a clustered statistic exists it is the inferential one and the draw count is
   descriptive.

---

## 1. TIER A — cite these

### 1.1 The scale floor (S1)

| claim | value | file |
|---|---|---|
| concept rows of `W_U` are effectively rank-one below 410M | mean pairwise cosine **0.959** (70M), **0.922** (160M) vs **0.031** (410M), **0.020** (1B) | `e38_jgeometry.json` |
| effective rank of that row block | **1, 1, 371, 508** | same |
| both null controls fire | random matrix ρ=0.99999; scaled identity ρ=1.3e-07 | same |
| at 160M the fitted lens does not beat the free baseline | **−0.0207 [−0.0418, −0.0024]** under `min`, excludes zero and is **negative** | `t22_bootstrap_ci_160m.json` |

**Why Tier A:** computed from the unembedding matrix alone — no prompts, no readout, so structurally
immune to §0.1. The consequence interval comes from a script that already stripped correctly.

### 1.2 The metric audit

| claim | value | file |
|---|---|---|
| a layer-deranged operator beats the real one under `min` | **84 of 120** draws at the corrected readout (104 of 120 pre-correction); **0 of 120** under `persist` both ways | `e54_aggregation_audit_rstrip_v2.json` : `C2_derangement.min.shuf_beats_jp_paired_by_seed` |
| clustered properly | **7 of 8** corpora, corpus-clustered paired **t(7) = −1.92, p = 0.096** at the corrected readout, which does NOT clear 5% (legacy: t(7) = −3.28, p = 0.013 over 21 of 24 operators; the operator count is not recomputed at the corrected readout) | `paper_clustered_derangement_t.json` : `by_readout.corrected.min` |
| **verdict** identical at both readouts | yes — `min` INADMISSIBLE, `persist` ADMISSIBLE, 7 of 8 corpora both ways. The **draw count** is not identical (104 → 84) and the seven corpora are not the same seven; cite the count, not the identity | `e54_aggregation_audit{,_rstrip_v2}.json` |
| the effect is a gradient across aggregations, not a binary | real operator wins **2/5 < 3/5 < 4/5 < 5/5** eval sets under `min`, `best1L`, `mean`, `persist`. `best1L` and `mean` were never selected on this control | `e33_logit_baseline_410m_v2.json` — **scope: one operator, one derangement seed, pre-correction readout** |
| the mechanism is a union effect | mean single-layer AUC **0.047 → 0.035**; cross-layer correlation 0.33–0.41 → 0.25–0.27 | `d1_min_union_diagnostic_410m.json` |
| the sign flips with band width | at **7 of 13** layers | same |
| permutation-calibrated: real vs 15 derangements of itself | **`min` 0 of 8** corpora, median z **−0.71**; `persist` 8 of 8, z +12.14 | `r9_permutation_calibrated_min.json` |

**Scoping, and it is load-bearing:** a derangement is a null for **correspondence**, not for "the
Jacobians carry no layer-specific information." And R9's 15 draws floor the empirical p at 1/16, so
the inferential statement is the across-corpora count, never a single corpus.

### 1.3 The corpus effect is real, and it is a source effect

| claim | value | file |
|---|---|---|
| between-source share, holding source fixed and resampling within it | **0.9886** (`min`), 0.9723 (`persist`), against a 0.70 bar | `r6_within_source_410m.json` |
| not substantially a lexical-composition effect | matched spread **22.72** (`min`), 22.24 (`persist`) vs a bar of 20 | `r7_matched_pools_410m.json` |

R6 is **ACCEPT** and answers the strongest criticism in either external review: corpus identity is
not a sampling artifact. **R7 is Tier B**, see §2.

### 1.3b The corpus effect replicates up the ladder, per family — CV6, Tier A

Registered before it ran (`docs/experiments/preregs/CV6_per_family_ladder.md`), corrected readout,
z-space, six named controls all fired. **Verdict as stored: REPLICATES.**

| claim | value | file |
|---|---|---|
| per-family spread over pooled seed SD at **2.8B** | **11.44** multihop, **14.51** multilingual, **6.46** order-ops, **11.93** poetry, **11.41** typo — `R >= 10` on **4 of 5** | `cv6_per_family_ladder.json` : `PRIMARY.R_per_family_2.8b` |
| ordering agreement with 410M, Kendall tau | **+0.60 / +0.80 / +0.80 / +0.60 / +1.00** — `tau >= 0.6` on **5 of 5** | same : `PRIMARY.kendall_tau_vs_410m_2.8b` |
| the effect is **larger** in absolute z at 2.8B than at 410M | z spread grew in **5 of 5** families: 0.4426→0.6007, 0.3826→0.6540, 0.2971→0.3746, 0.2381→0.3792, 0.5255→0.5955 | same, vs `d3_corpus_by_family_410m.json` |
| Github is bottom of the ordering | **last in 4 of 5** families at 2.8B, 4th in the fifth | same : `by_model.2.8b.by_family.<f>.order_by_z` |
| USPTO tops order-ops and **only** order-ops, at a second scale | 2.8B order-ops top corpus = `USPTO_Backgrounds` | same |
| the scorer is the same instrument as D3's | worst \|dz\| **0.000e+00** over 16 arms x 5 families; negative control at the legacy readout separates at **1.997** | `cv6/cv6_c0_scorer_equivalence.json` |
| fp16 storage rounding is immaterial here — **measured, not assumed** | worst \|d z_mean\| **1.171e-04**; ordering unchanged in **10 of 10** (model, family) cells | `cv6_per_family_ladder.json` : `...fp16_roundtrip_diagnostic` |

**Why Tier A:** registered before running, stored artifact, six controls each with the number it
produced, corrected readout, and the replication unit is the 5-corpus panel with 3 disjoint seed
blocks (15/15 distinct fitting-prompt SHAs).

**The one thing this table must not be used for.** `R` is **not comparable across `N`** — the
pre-registration declared this before the run. R falls from 19–43 at 410M (N=200) to 6–15 here
(N=25) purely because the pooled seed SD is **2.29x–5.70x** larger at the smaller N. Row 3 is the
honest cross-scale comparison; **any sentence of the form "the corpus effect weakens with scale" is
unsupported and contradicted by row 3.**

**Open irregularity, recorded not explained:** order-ops is the only family below the bar (6.46),
and at 1.4B it is **R 4.7 with tau −0.40** — an inverted ordering with Github 2nd. It is also the
highest-competence family (14.5% top-1 at 2.8B).

### 1.3c The 1B rung — CV7, Tier A

Registered before it ran (`docs/experiments/preregs/CV7_1b_rung.md`), scoring only from operators
fitted for E62, corrected readout, z-space, five named controls all fired. **Verdict as stored:
REPLICATES AT 1B.**

| claim | value | file |
|---|---|---|
| per-family spread over pooled seed SD at 1B | **29.38** multihop, **25.48** multilingual, **31.71** order-ops, **18.08** poetry, **23.09** typo — `R >= 10` on **5 of 5** | `cv7_1b_rung.json` : `PRIMARY.R_per_family_1b` |
| ordering agreement with 410M | **+0.40 / +0.80 / +0.80 / +0.40 / +0.80** — tau `>= 0.6` on **exactly 3 of 5**, the bar itself | same : `PRIMARY.kendall_tau_vs_410m` |
| the order-ops inversion is **checkpoint-specific, not family-specific** | order-ops tau **+0.80 (1B) / −0.40 (1.4B) / +0.80 (2.8B)** — the only negative tau across four rungs | same : `SECONDARY_order_ops_tau_sign` |
| band read **off the artifact**, not chosen | all 15 lenses carry `source_layers == [6..13] == band_for(16)`, asserted before scoring | same : `controls.C2_band_read_off_artifacts` |
| the free baseline rises monotonically at every rung | `logit_I` pooled flat-mean-7 min-rank **0.19811 → 0.21900 → 0.23942 → 0.26996** (410M/1B/1.4B/2.8B) | `cv7_1b_rung.json`, `cv6_per_family_ladder.json`, `e48_crossover_410m_rstrip.json` |

**Why Tier A:** registered before running with a disclosed ordering, stored artifacts, five controls
each with the number it produced, corrected readout, and the scorer is CV6's — **imported, not
reimplemented** — so it inherits C0's 0.000e+00 agreement with D3 and that control's negative arm
at 1.997.

**Two things this row retires.**

1. **"Github is always worst" is FALSE.** Github is last in 5 of 5 families at 410M, but at 1B
   **Wikipedia_en is last in 3 of 5**. The bottom of the ordering is near-invariant *within* a rung
   and **not stable across rungs**. Restate as within-rung wherever it appears.
2. **R must never rank rungs across mismatched N.** 410M and 1B are N=200; CV6's 1.4B and 2.8B are
   N=25, where the pooled seed SD is 2.29x–5.70x larger. Compare the **z spread**, stored at
   `cv7_1b_rung.json -> CROSS_LADDER_z_spread`.

**Declared, and it survives all four rungs:** 1B is 16 layers at `d_model` 2048 against 1.4B's 24 at
the same width — shallower, not narrower — so band widths are 8 / 13 / 18. Both aggregations are
band-width sensitive (D1's union effect), and **nothing in this design separates band width from
scale.**

### 1.4 Exposure does not order the read

Containment spans **10862x** across the eight fitting corpora and does not order their read quality.
`e48c_exposure_vs_read.json`. Containment is computed over the token stream, so this is immune to
the readout defect.

Corrected coverage figures at 20/20 shards, k=32: post-cutoff Wikipedia **0.26958**, PubMed control
**0.79273**. `e48b_exposure_growth.json`.

### 1.5 The parallel port is exact

The pooled grid reproduces the published matrix at **`max_abs_diff = 0.0`** over 128 cells, and is
identical at workers 1, 2, 6 and 8. `e52_factorial_410m_pooled.json`.

### 1.6 Corrections that are now settled

Random beats top-k on **2 of 5** pairs, both Github. Max |ρ| is **0.28182**, so "≤ 0.28" was false.
The two "exactly zero" sets are **−1.29e-05** and **−2.89e-06**. Fit/prefix overlap **0.17043**.
Effective rank minimum is **725.888** at `step143000`, so "never below 726" was false. "541 items"
is the six-set count; the admitted five hold **449** items and **801** of 893 pairs. The leave-two-out
reversal is **read 47.613 vs fit 42.011**. `r4_corrections.json`.

---

## 2. TIER B — usable, with the weakness named

| claim | value | the weakness |
|---|---|---|
| **the fit x read decomposition** | **53.35 / 44.61** (`min`), 50.70 / 48.13 (`persist`) | the ordering survives both aggregations; **the magnitude does not survive the correction** — it was 91.2/7.1 on the legacy readout. `e57_grid_variance_ci_rstrip.json` |
| the gap excludes zero | **+8.74 [+0.80, +15.87]** under `min` | under `persist` it is +2.57 **[−7.60, +9.80]**, which **includes zero**. The registered rule fires REJECT there |
| independent 5x5 replication | **5/5 replicates** under `min` | **0/5** under `persist` |
| leave-two-out ordering | **20/28** under `min` | 16/28 under `persist`; corpus-axis range is 3.6x the seed interval |
| **R7, length-matched pools** | spread 22.72 vs a bar of 20 | **DIRECTIONAL RESULT (REQUIRES C1)** — C1 failed at 11.1% against its 10% bar, so the pools are not matched to the tolerance the design declared |
| the read-shift slope | fitted operators are flatter on **1 of 5** (`persist`), **3 of 5** (`min`) | aggregation-dependent, and it overturns a prior REJECT rather than establishing anything |
| the 1B replication | 9 of 10 pairs separate either way; **35.52x → 31.03x** on the corrected band | the 1B ladder ran on a band the programme's own rule does not produce; the corrected-band recompute exists |
| D2 device clearance | cited asymptotes move **≤ 1.6e-05** against a 3.5e-3 bar | `r3_close_d2.json`. Cleared, but the underlying divergence is real at the cell level |

### 2.1 The N ladder (S2) — UNCLEAR, and why

R8: N-over-corpus is **31.9% under `min`** against a **25%** bar — a fail — and 14.2% under
`persist`. `r8_ladder_flatness.json`.

**It is a denominator effect.** Four of five corpora are flat in absolute terms while the corpus axis
shrank 42%; only Github's N-range grows.

**Two defects in the surrounding numbers, both Tier D-adjacent:**

* The **58x seed-SD headline has a ragged denominator.** `e28_Wikipedia_en_410m_n800_s2.pt` does not
  exist, so that seed carries 11 rungs where the others carry 16, and `t53` averages each seed over
  its own grid. The SD is **1.121e-04 ragged against 5.820e-04 intersected** under `min` — a ~5x
  difference in the denominator of a printed headline. **Do not cite 58x until it is recomputed on
  the intersected grid.**
* The published **"range over N = 1.2–3.7 seed SD" does not reproduce under any estimand**, and no
  script in the tree computes it. **Unstored. Do not cite.**

### 2.2 Instrument hygiene, measured

**36 of 106** control rows carry no gate, were never evaluated, or are unfalsifiable; **70** have
power, of which **61 fired true and 9 fired false**. *(Corrected 2026-08-24: this line read "36 of
92" and three other documents copied that denominator. `r4e_control_power.json` records
`n_control_rows = 106`. The numerator was always right; the denominator was never in the file.)*
**6** controls named in a specification appear in **no** results file of that specification — including
`PREREG_E36_QLADDER.md`'s **C4**, which the pre-registration itself calls load-bearing (*"without C4
the entire ladder could be a prefix-length artifact"*). `r4e_control_power.json`.

**29 of 225** results files carry a `payload_sha256`; exactly one (`e58_algebra_audit`) genuinely
mismatches. **166 of 166** E28 sidecars misrecorded their corpus and have been backfilled, hashed
into `ARTIFACTS.md` and re-verified with **0 mismatches**. `./lab health` H3 now compares a hash and
was verified to fail on a planted mismatch. `r4g_e28_provenance.json`.

---

## 3. TIER D — RETRACTED. Do not restate.

| claim | why it is dead |
|---|---|
| **the corpus-axis crossover** — Github at or below the free baseline | corrected readout: Github **−1.9% → +21.1%**, logit constant 0.028440 → 0.083179, **all eight corpora above the baseline**. `e48_crossover_410m_rstrip.json`. **R0 is the retraction and it is outstanding** |
| **"the fitting corpus is worth roughly an order of magnitude more than the read corpus"** | 53.4 / 44.6. The fit axis fell only x0.85 in absolute terms; the read axis grew **x10.27**. The share collapse is a denominator effect |
| **the layer-shuffled Jacobian scoring 5/6, as a finding** | an artifact of min-over-layers, which is existential and cannot distinguish a working operator from a broken one. Enforced: `repro/lib/banned_claims.py`, `./lab health` H11 |
| **F19**, r(dispersion, plateau AUC) = −0.938 | killed by its own pre-registered leave-one-out control; its rival predictor died too |
| **the −0.511 sample-size law as a discovery** | close to a mathematical identity — the CLT applied to a mean |
| **"D_act and M1–M4 have never been run"** | false when written; only M1–M3 were unrun |
| **"never falls below 726"** (effective rank) | 725.888 at `step143000` |
| **"~25% previously asserted"** for the E36 diagonal overlap | the stored assertion is **~17%**, and the measurement (0.1704) **vindicated** it. Appeared in two places |

---

## 4. OPEN RULINGS — operator only

1. **R0**, the corpus-axis crossover retraction. Measurement confirmed twice; no canonical document
   edited.
2. **E57's registered rule fires REJECT under `persist`.** Under the discipline forbidding
   reinterpretation of a fired rule, this needs a ruling rather than a rewrite.
3. **P0's wallclock clause is unmet** (~1.6–2x against a registered 3x). Flagged, not re-cut.

## 5. NOT ESTABLISHED

* **S3's crossover.** The fitted lens has never been observed below the free logit lens: **0 of 5**
  corpora, either readout, either aggregation.
* **That `persist` measures readout quality.** It passes a control it was built to pass. R9 is the
  attempt to answer this without changing the metric.
* **Anything about the write side (S4) or second-order structure (S5).** Parked.
* **Task-domain generality.** The concept battery is fixed in every cell of every design here.
* **Any scale above 1B**, and no model family outside Pythia.

## 6. WHERE THE PROGRAMME AND THE SOURCE DISAGREE

| item | the source says | its released code does | we use |
|---|---|---|---|
| `target_layer` | penultimate is the default recipe | final | penultimate primary, final secondary |
| `skip_first` | unmodified position average (0) | 16, hardcoded | 16 (immaterial, measured) |
| `n` | 1000 prompts | — | 200 |
| eval item counts | 50/54/55/52/96/50 | released JSON has 93/107/55/98/96/102 | the released JSON, disclosed |
| readout position | "the token immediately preceding `target`" | **no evaluation code ships** | the stripped token, per the data spec |
| write reference | three families defined | read-side only | ours, from the source text |

**The source ships no evaluation code**, so every scoring decision is ours. That is why §0.1 was
possible and why the deviations table in the paper is load-bearing.
