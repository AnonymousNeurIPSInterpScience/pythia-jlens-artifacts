# CONTEXT — what was run, what it found, what is open

**Rewritten 2026-08-22.** This is the programme's provenance document: every experiment, its
verdict, and the results file behind it. It replaces the version that lived under `paper/`, now in `archive/`. **Nothing here is from memory.** Every number names a file, and every file is
stamped into its experiment document by `tools/build_provenance.py`.

| you want | read |
|---|---|
| one row per experiment, generated | [`experiments/INDEX.md`](../experiments/INDEX.md) |
| every claim, tiered by confidence | [`RESULTS_TAXONOMY.md`](RESULTS_TAXONOMY.md) |
| why `min` is primary | [`AGGREGATION_POLICY.md`](AGGREGATION_POLICY.md) |
| which results sit on the corrected readout | `.venv/bin/python tools/readout_exposure.py` |
| the method itself, transcribed from the source | [`RIGOROUS_ANTHROPIC.md`](../../paper/fundamentals/RIGOROUS_ANTHROPIC.md) |

---

## 0. HOW TO READ EVERY VERDICT BELOW

**A null is a finding.** This programme measures; it does not confirm. S1, S2 and S3 are intuitions
that told us where to point the instrument, and nothing here is improved by any of them being true.
Where a subthesis is marked HOLDS, UNCLEAR or NOT REACHED, that is a statement about **how tightly a
quantity is now pinned down** — never about whether a hypothesis survived. A measured null with a
stated bound closes a question the source left open, which is the whole point.

Nothing in this repository is "dead" because a subthesis failed. The only things that go out of date
are **unstored numbers** and **stale transcriptions**, and both are fixed by recomputing. See
`CLAUDE.md` §1, *The standing rule*.

## 0b. VERDICT FIRST

**The programme measured what it set out to measure, and the answers are legible.** It asked whether
a Jacobian lens's measured quality depends on the corpus it is fitted on. It does — but *which*
corpus, not *whether the model has seen it*, and that dissociation is the finding.

**Where the three subtheses stand** — as bounded measurements, not as wins or losses. Scored in
detail across five axes in the validity synthesis in `DIAGNOSIS.md` §3.

| | what is pinned down | grade |
|---|---|---|
| **S1** — small models | at 160M the fitted lens is measurably behind the free one, **−0.0207 [−0.0418, −0.0024]**. The mechanism is a *precondition* claim — below some scale a transport comparison is weakly identified — not a claim about the Jacobian. `E38_geometry.md` grades itself **Tier B** and says "precondition, not cause" | **B−** |
| **S2** — a justifiable N | **reads are flat in N.** Across a 32× range the largest per-corpus movement is 0.00783 against a between-corpus spread of 0.0246. The registered ratio returns 31.9% against a 25% bar under `min` — **UNCLEAR, standing** — and the entire excursion is attributable to one corpus | **A−** |
| **S3** — distribution theory | **membership is bounded, identity is large.** Across a 10,862× containment range the membership effect is bounded to **[−0.0204,+0.0176]** (fit) and **[−0.0162,+0.0298]** (read), with no matched-tier interaction. The crossover S3 predicts has never been observed: **0 of 5**, either readout, either aggregation | **B** |

**Two things the subtheses did not ask, which the programme answered anyway**, and which travel
further than any of them:

* **Aggregation existentiality governs a statistic's null-sensitivity.** The more a layer-aggregation
  counts *opportunities* rather than *quality*, the more it prefers a scrambled operator: real-operator
  win counts run **2/5 < 3/5 < 4/5 < 5/5** across `min`, `best1L`, `mean`, `persist`, with a
  band-width sign flip at 7 of 13 layers. `best1L` and `mean` were never selected on this control,
  which is what dissolves the circularity objection. Applies to any layerwise probing method.
* **"Distribution shift" separates into membership and identity, and they dissociate.** Membership
  varies 10,862× and orders nothing; identity orders a lot (8–10 draw SD) and none of 17 tested
  predictors recovers it.

**Two standing caveats on the above.** S3's membership result is a **bound, not an absence** — the
eight-corpus panel has a minimum detectable effect of 0.0268 on the fit axis, larger than the
between-corpus spread it is being compared to. And the aggregation gradient currently rests on **one
operator, one derangement seed, on the pre-correction readout**; the ~3.5 CPU-hour re-score that
would generalise it is Tier 0 of the slate.

---

## 1. TWO CORRECTIONS THAT GOVERN EVERYTHING ELSE

### 1.1 The readout defect

Scoring read the **final token of the unstripped prompt**. The released eval prompts end in a
trailing space so that `prompt + target` concatenates as a string, and BPE absorbs that space into
the target's leading space. So the token being scored — id 209, a bare space — **does not occur in
the tokenised sequence at all**. The source's data spec says the readout is "the token immediately
preceding `target`", which is the stripped last token, id 253.

Scope: **157 of 551 released items**, which is 157 of the 449 in the five admitted sets. All 55
order-ops, 83 of 93 multihop, 19 of 107 multilingual, and **exactly zero** on poetry, typo and
association. Those three untouched sets are the internal control: they score identically under both
readouts, which is what proves the two arms differ in the readout token and nothing else.

Measured cost, `persist`, admitted-5, 410M ladder: order-ops **29.8x**, multihop **3.13x**,
multilingual 1.01x, the other three **1.00x exactly**.

**Every affected result has been re-scored.** `tools/readout_exposure.py` is the ledger: it walks
every results file, resolves its producing script, checks whether that script strips, and propagates
exposure through `provenance.inputs` to a fixpoint. Each experiment document carries its files'
class in its generated PROVENANCE block.

**The operators were never affected.** Fitting draws its pool from `corpora/*.jsonl`
(`experiments/trainval.py:241-244`), never from eval prompts. Every correction was a CPU re-score.

### 1.2 The aggregation ruling

**`min`-over-layers is primary. `persist` is a secondary robustness arm, labelled as selected with
knowledge of the control outcome.** Ruled 2026-08-20; see `AGGREGATION_POLICY.md`.

`min` is the source's own operational definition of recovery. `persist` was adopted *because* it
passed the derangement control that motivated it, which is outcome-dependent metric selection. The
switch was a reporting change only: **all 14 results files in the post-review slate carry both
aggregations.** It also happened to help — see §4.

---

## 1d. THE CONSTRUCT-VALIDITY SERIES (CV1–CV6) — 2026-08-22/23

A third correction, larger than the first two, opened here and is **still open**. Entry point:
[`../validity/DIAGNOSIS.md`](../validity/DIAGNOSIS.md). Where the ground truth comes from and the two
routes out: [`../validity/CONSTRUCT_VALIDITY.md`](../validity/CONSTRUCT_VALIDITY.md).

**The question.** The annotated intermediates (`Brazil` for the Carnival multihop, `5` and
`multiplication` for `(2+3)*4`) are **human task decomposition performed by Gurnee et al. for
Claude-scale models and transplanted to Pythia unvalidated**. Nothing in this repository does circuit
analysis on Pythia. So a successful read may be a successful read of *the experimenter's task
decomposition* rather than of *the model's computation*.

**What was measured, and every one of these is stored with provenance:**

| id | question | verdict | file |
|---|---|---|---|
| **CV1** | can the model do the task? | 410M **4.7%** top-1 on the 255 items with a target | `cv1_answer_competence.json` |
| **CV2** | is the operator read inside its positional support? | **CONFIRMED mismatch** — **0 of 551** prefixed reads inside; `Q0` 45.7% | `cv2_position_support.json` |
| **CV3** | is the corpus effect a score effect or a ranking artifact? | **ACCEPT, PRIMARY 2.46** — 27.16σ in z, **2.5× larger than rank** | `cv3_margins_410m.json` |
| **CV4 P1** | capability across the ladder, all six families | top-1 never separates 410M→2.8B; top-10 rises 16.9→32.9%; surprisal falls monotonically | `cv4_phase1_capability.json` |
| **CV5** | was rank blind at E66's perturbation? | **REJECT** — both metrics quiet at `max_rel` 2.535e-03 | `cv5_metric_sensitivity_410m.json` |
| **D3** | the corpus effect by family (**EXPLORATORY**) | 19–43σ in **every** family; USPTO tops order-ops | `d3_corpus_by_family_410m.json` |
| **CV6** | does it replicate at 1.4B/2.8B, per family? | **REPLICATES** — `R(f,2.8B) >= 10` on **4 of 5** families (multihop 11.44, multilingual 14.51, poetry 11.93, typo 11.41; order-ops 6.46) and ordering tau `>= 0.6` on **5 of 5**. All six controls fired. **Do not read R against 410M's 19–43**: the numerator grew in 5 of 5 families and the denominator grew faster, because N=25 has a higher seed noise floor than N=200 (pooled seed SD 2.29×–5.70× larger). In absolute z the effect is **larger** at 2.8B than at 410M in every family | `cv6_per_family_ladder.json` |
| **CV7** | does it hold at the 1B rung, from operators that already exist? | **REPLICATES AT 1B** — `R >= 10` on **5 of 5** (18.08–31.71, at N=200 so comparable to 410M's 19–43, **not** to CV6's N=25), tau `>= 0.6` on **exactly 3 of 5** — the bar itself, not a margin. All five controls fired. Free: scoring only, CPU, no fitting | `cv7_1b_rung.json` |

### 1d.1 What this does to the programme's claims

* **The corpus effect survives, and got larger.** CV3: it is a **score** effect, not a rank artifact,
  and rank *understates* it 2.5×. CV5 confirms rank is both coarse at small perturbations and jumpy
  at medium ones (1.834σ where z moves 0.522σ). **Report in z-space** — two independent reasons.
* **Absolute readout claims are retired.** The model answers ~5% of the battery and the operator is
  read entirely outside its positional support. Neither is fixable by more seeds: the second is
  **misspecification**, not variance.
* **S1 survives one confound.** Competence has no discontinuity at 160M→410M where the `W_U` cliff
  sits (0.9218 → 0.0307), so the scale floor is not a competence artifact. Held lightly: all models
  are near the floor, and a floor can conceal differences.
* **CV4's own STOP condition has fired.** *"If even M\* is <~10% pooled competence, log this as a
  finding."* 2.8B is **5.9%**. **Gurnee et al.'s battery is capability-gated out of Pythia's reach at
  every scale we can run.**

### 1d.2 THE POOLED ESTIMAND IS RETIRED — report per family

By **scored pairs**, which is what the flat-pool convention weights:

| family | pairs | share | 2.8B top-1 | z spread / seed SD (410M) | best → worst (z) |
|---|---:|---:|---:|---:|---|
| multihop | 103 | 12.9% | 6.5% | 25.4 | Pile-CC > StackEx > USPTO > Wiki > Github |
| **multilingual** | **394** | **49.2%** | **0.9%** | 19.4 | StackEx > Pile-CC > USPTO > Wiki > Github |
| order-ops | 110 | 13.7% | **14.5%** | 24.6 | **USPTO** > StackEx > Pile-CC > Wiki > Github |
| poetry | 98 | 12.2% | undefined | **42.7** | Pile-CC > USPTO > StackEx > Wiki > Github |
| typo | 96 | 12.0% | undefined | 27.0 | StackEx > Pile-CC > USPTO > Wiki > Github |

**A pooled figure weights multilingual at 49.2% while the model answers 0.9% of it.** (24% is the
*item* share; multilingual items carry ~4 intermediates each.)

Three structural facts:

1. **The bottom of the ordering is family-invariant WITHIN A RUNG — but which corpus occupies it
   is NOT stable ACROSS rungs.** At 410M, Github is last in all five families, and that is what the
   sentence originally said. **CV7 breaks it at 1B**, where **Wikipedia_en is last in 3 of 5** and
   Github in 2. By rung: 410M Github 5/5 · **1B Wikipedia_en 3/5, Github 2/5** · 1.4B Github 4/5 ·
   2.8B Github 4/5. Any sentence of the form *"Github is always worst"* is false as written.
   `cv7_1b_rung.json`, `cv6_per_family_ladder.json`, `d3_corpus_by_family_410m.json`

The logit lens also **beats every fitted operator on poetry** and roughly ties on order-ops, so the
J-lens advantage is itself family-dependent.

**Ordering cost, disclosed:** D3 ran *before* CV6 was pre-registered, at the operator's direction and
after the cost was flagged. The per-family estimand **may not be described as chosen on capability
grounds alone**.

### 1d.3 Two measurement facts found while scoping the fix

* **The containment index does not transfer to the battery.** 92% of items (505/551) are shorter than
  k=32; median prompt is 15 tokens. The association-vs-computation discriminator needs a
  **(trigger, intermediate) co-occurrence index**, a new build.
* **Our k-summary is flat-mean-7; the source's is reported as log-k AUC.** On identical ranks the two
  differ by **4.3%**. Within-pipeline results are unaffected; any head-to-head must name its summary.

---

## 1e. THE FOUR-RUNG LADDER, ASSEMBLED — write the paper from this section

**2026-08-24, after CV6 and CV7.** This is the corpus-effect result as it now stands, with every
number traced to a file and every trap that a rewrite would otherwise walk into stated inline.

### The headline, in one sentence

> **The per-family corpus effect on the J-lens read is present at every rung of the Pythia ladder
> from 410M to 2.8B, it is a score effect rather than a ranking artifact, and it is larger in
> absolute terms at 2.8B than at 410M in every one of the five admitted families.**

### The table

`R = spread_z / pooled_seed_sd_z`, computed **within** a family. **N is stated on every rung
because R is not comparable across N.**

| family | pairs | 410M R (N=200) | **1B R (N=200)** | 1.4B R (N=25) | **2.8B R (N=25)** | tau@2.8B |
|---|---:|---:|---:|---:|---:|---:|
| multihop | 103 | 25.4 | **29.38** | 7.29 | **11.44** | +0.60 |
| multilingual | 394 | 19.4 | **25.48** | 6.98 | **14.51** | +0.80 |
| order-ops | 110 | 24.6 | **31.71** | 4.72 | **6.46** | +0.80 |
| poetry | 98 | 42.7 | **18.08** | 14.82 | **11.93** | +0.60 |
| typo | 96 | 27.0 | **23.09** | 16.31 | **11.41** | +1.00 |

Verdicts as stored: 410M = D3, **EXPLORATORY, no decision rule**. 1B = **REPLICATES AT 1B**
(`cv7_1b_rung.json`). 2.8B = **REPLICATES** (`cv6_per_family_ladder.json`). **1.4B has no verdict
and may not be given one** — CV6's rule reads 2.8B, and its numbers were seen before any 1.4B rule
could be written.

### Five things a rewrite must not get wrong

1. **Never rank rungs by R across mismatched N.** The pooled seed SD is **2.29x–5.70x** larger at
   N=25 than at N=200. The N=200 column (410M, 1B) and the N=25 column (1.4B, 2.8B) are two
   different scales. The cross-rung comparison is the **z spread**, and CV7 stores it explicitly at
   `cv7_1b_rung.json -> CROSS_LADDER_z_spread`.
2. **The effect does not weaken with scale.** z spread grew 410M→2.8B in **5 of 5** families
   (0.4426→0.6007, 0.3826→0.6540, 0.2971→0.3746, 0.2381→0.3792, 0.5255→0.5955). R falls only
   because the denominator moved. Say "larger", not "smaller".
3. **"Github is always worst" is false.** See §1d. Github is last 5/5 at 410M but **Wikipedia_en is
   last 3/5 at 1B**. State the bottom as within-rung.
4. **The USPTO x order-ops interaction replicated.** USPTO tops order-ops and only order-ops at
   410M (D3, exploratory) and again at 2.8B (CV6, registered). At 1B, order-ops tops with
   StackExchange and USPTO is 2nd. Report as replicated at 2.8B, present but not top at 1B.
5. **The free baseline rises monotonically at every rung** — `logit_I` pooled flat-mean-7 min-rank
   **0.19811 → 0.21900 → 0.23942 → 0.26996** across 410M/1B/1.4B/2.8B. Any J-vs-logit comparison at
   2.8B is against a stronger baseline than at 410M, and a table that omits this overstates the
   fitted operator at scale.

### The 1.4B anomaly, stated at the strength the evidence supports

order-ops Kendall tau by rung: **1B +0.80 · 1.4B −0.40 · 2.8B +0.80.** The inverted ordering is
**specific to the 1.4B checkpoint**, the only negative tau across four rungs. CV4 shows 1.4B also
dips in competence (top-1 3.1% against 1B 4.3% and 2.8B 5.9%). **These are two coincident dips at
one rung and nothing here makes one cause the other.** CV7's pre-registration attached no threshold
to this, so it is a **sign, not a result**. On magnitude the dip is weaker still: 1B exceeds 1.4B
in only **3 of 5** families. Do not write "1.4B is a local dip" without that qualifier.

### The confound that survives all four rungs

**1B is 16 layers at `d_model` 2048; 1.4B is 24 at the same width.** Shallower, not narrower. Band
widths are **8 / 13 / 18** layers at 1B / 1.4B / 2.8B, and both `min`-over-band and max-over-band
are band-width sensitive (D1's union effect). **A rung-to-rung difference may be a band-width
effect and nothing in this design separates it from scale.** Every rung also differs in training,
so no outcome licenses a causal reading of scale.

### What this does NOT do

It does not rehabilitate the absolute claims — cv1 (competence) and cv2 (positional support)
retired those and neither CV6 nor CV7 reopened them. It does not remove the conditional label on
the corpus comparison: position and corpus may interact, and every rung here holds the read
protocol fixed, so the finding remains *"corpus X produces an operator that extrapolates better to
these read positions."* See `../validity/DIAGNOSIS.md` §5 and §8.1.

## 2. S1 — SMALL MODELS. A PRECONDITION RESULT, GRADED B−.

**Below 410M the comparison between transports is weakly identified, for a reason in the readout
matrix rather than in the Jacobian.** The concept rows of the unembedding are ill-conditioned:
mean pairwise cosine 0.959 at 70M and 0.922 at 160M against 0.031 at 410M and 0.020 at 1B; effective
rank 1, 1, 371, 508. Both null controls fire.

**The algebraic endpoint is an identity, not evidence.** If `W_U` were exactly rank-1 then every
concept's logit is one vector times a scalar and ranking is `J`-independent — but that is arithmetic,
and `PREREG_E38_JGEOMETRY.md` §WHY retracted it as an observation. What the geometry establishes is
weaker and still useful: the concept block is ill-conditioned enough that the comparison is poorly
identified. See the three qualifications below for how far that goes.

**Why the readout defect does not touch it:** those numbers are computed from the unembedding matrix
alone, with no prompts and no readout. The two instruments that carry its consequences — the 160M
hierarchical interval and the P-ladder corpus-variance ratio — were both already on the correct
readout.

**The consequence, measured:** at 160M neither aggregation beats the free logit lens, and under
`min` the interval is **negative** (−0.0207 [−0.0418, −0.0024]) — the fitted lens measurably hurts.

**Three qualifications, all measured, none of which touch that interval.**

1. **"Effectively rank-one" overstates it.** A *genuinely* rank-1 `W_U` — E37's `top_1` rung on the
   **logit** arm, no Jacobian involved — costs **71.9%** of the read score at 70M and **83.7%** at
   160M (`e37_rank_ablation_{70m,160m}_wikitext.json`). Those unembeddings are not functionally
   rank-one; `top1_share = 0.9546` leaves 4.5% of the energy carrying all of the ranking.
2. **The geometry statistic is computed on un-centred rows** (`experiments/t38_jgeometry.py:162-169`,
   `geom()`), so a shared row direction inflates `mean_cos` and deflates `eff_rank`. A shared
   direction is rank-irrelevant, because softmax is shift-invariant. The centred recompute is Tier 1
   of the slate; it sits beside `CLAUDE.md` §6.4, which governs the lens read path rather than a
   descriptive statistic on `W_U`, and wants an operator gate.
3. **E37 is silent, not negative, about the mechanism at small scale.** Its decision statistic is
   `max(gap(1),gap(2)) / gap(full)`, and the denominator is **+0.0055 at 70M** (inside the noise
   floor) and **−0.0358 at 160M** (negative). The stored 70M REJECT is a small-denominator artifact
   and the 160M ACCEPT a sign artifact; only the 410M/USPTO run is interpretable (frac 0.214).
   `docs/experiments/preregs/superseded/PREREG_E38_JGEOMETRY.md` §WHY already retracted E37's headline as vacuous and
   noted that frac flips with the *fitting corpus* while `W_U` is identical — so frac cannot be a
   property of `W_U`.

**`E38_geometry.md` grades itself Tier B and says "precondition, not cause — and the paper must keep
saying so."** This section and `RESULTS_TAXONOMY.md` §1.1 have not caught up; see §6 item 6.

**What could knock it down.** Four model scales, two of them degenerate, so the "floor" is located
between 160M and 410M by two points. It is a property of a *trained* small model rather than an
undertrained one (`E65`), which is the stronger version. Extending to 1.4B/2.8B would add scales but
not change the mechanism, which is algebraic.

→ `experiments/E38_geometry.md`, `E65_training_axis_floor.md`

## 3. S2 — THE NUMBER OF FITTING PROMPTS. UNCLEAR.

**The premise is false and that was never in doubt: reads are flat in N.** A 32x change in fitting
prompts moves the read by a handful of seed standard deviations.

**What is unclear is the registered test.** R8 re-ran the ladder at the corrected readout and the
N-over-corpus ratio came in at **31.9% under `min`**, against a 25% bar — a fail — and 14.2% under
`persist`, a pass. The primary metric does not clear it.

**Why, and it is a denominator effect:** four of five corpora are flat in absolute terms while the
*corpus* axis shrank 42%. Only Github's N-range grows. Github again.

**Two documentation defects found alongside it, both real:**

* The ladder's N grid is **ragged** — `results/e28_Wikipedia_en_410m_n800_s2.pt` does not exist, so
  that seed has 11 rungs where the others have 16. `t53` averages each seed over its own grid, so its
  "seed SD" conflates seed with N-grid variation, **and that SD is the denominator of the 58x
  headline**: 1.121e-04 ragged against 5.820e-04 intersected under `min`.
* The published "range over N = 1.2–3.7 seed SD" **does not reproduce under any estimand**, and no
  script in the tree computes it.

→ `experiments/E28_read_ladder.md`, `E53_ladder_summary.md`, `R8_ladder_rstrip_S2.md`

## 4. S3 — DISTRIBUTION THEORY. UNRESOLVED.

Four separate questions that have repeatedly been merged. Keep them apart.

| question | answer | where |
|---|---|---|
| Do both corpus roles matter? | **Yes, comparably.** 53.4 fit / 44.6 read under `min` | `R1_grid_rstrip.md` |
| Is the corpus effect real, or a bundle artifact? | **Real.** Between-source share **0.9886** | `R6_within_source_resampling.md` |
| Is it a lexical-composition effect? | **No**, though C1 missed its bar at 11.1% vs 10% | `R7_length_matched_pools.md` |
| Does the fitted lens degrade faster under read shift than the free lens? | **Directionally yes, unresolved.** Was 5/5 flatter, now 1/5 (`persist`) and 3/5 (`min`) | `R4b_e36_rstrip.md` |
| Does it ever cross **below** the free lens — S3's actual claim? | **No. 0 of 5**, either readout, either aggregation | `E36_qladder.md` |

**What R1 did to the decomposition**, and the shape matters more than the shares:

| | SS_fit | SS_read | ratio |
|---|---|---|---|
| legacy readout, `persist` | 2.899e-03 | 2.267e-04 | — |
| corrected, `persist` | 2.452e-03 | 2.328e-03 | fit **x0.85**, read **x10.27** |

**The fit axis did not weaken. The read axis became real.** Reading at a bare space cannot see a
128-token context prefix, so context looked irrelevant. Saying "the fit effect shrank" is wrong; the
share collapse is a denominator effect.

**Consequences, all measured.** E57's registered rule fires **REJECT under `persist`** (fit−read
includes zero) and **passes under `min`** (+8.74 [+0.80, +15.87]). E55's independent replication
holds **5/5 under `min`**, 0/5 under `persist`. Leave-two-out ordering is 20/28 under `min`.

**Also gone: the corpus-axis crossover.** Github moves from −1.9% to **+21.1%** and all eight corpora
beat the free baseline. R0 retracts it; see §6.

## 5. THE METRIC AUDIT — the most transferable result

**The source's own statistic cannot certify that a readout comes from operator structure.**

A layer-derangement preserves entries, norms and spectra and destroys only the correspondence
between a layer and its own derivative. It is a real perturbation, not a near-no-op: mean pairwise
Jacobian cosine across the band is **0.6291** (`t15_shuffle_diagnostic_410m.json`).

Under `min` the deranged operator beats the real one on **84 of 120** paired draws at the corrected
readout (`e54_aggregation_audit_rstrip_v2.json`, key `C2_derangement.min.shuf_beats_jp_paired_by_seed`);
under `persist`, **0 of 120**.

**The verdict is stable across readouts; the draw count is not.** `min` INADMISSIBLE and `persist`
ADMISSIBLE hold at both readouts, and both give **7 of 8** corpora — but the count moves 104 → 84 and
the `min` gap range narrows from [−0.0321,+0.0011] to [−0.0168,+0.0081], while `persist`'s *widens*.
Do not write "the defect moved nothing"; write that the verdict was unaffected. Note also that the
seven corpora are **not the same seven** across readouts — cite the count, not the identity.

The mechanism is a union effect, and it is measured: `min` is existential over a 13-layer band, a
real operator's per-layer readouts agree while a derangement's do not, so mean single-layer AUC
falls 0.047 → 0.035 while the union rises. The sign flips at a band width of **7 of 13 layers**.

**It is a gradient, not a binary.** Across four aggregations ordered by how existential they are, the
real operator's win count rises monotonically — `min` 2/5 eval sets, `best1L` 3/5, `mean` 4/5,
`persist` 5/5 (`e33_logit_baseline_410m_v2.json`). `best1L` and `mean` were never selected on this
control, which answers the circularity objection without switching metric. **Scope: one operator
(Github, the worst-scoring), one derangement seed, pre-correction readout** — the ~3.5 CPU-hour
re-score that would generalise it is Tier 0 of the slate.

**R9 calibrated it without changing the metric.** Against 15 derangements *of itself*, per corpus:

| | beats all 15 of its own derangements | median z |
|---|---|---|
| **`min`** | **0 of 8 corpora** | **−0.71** |
| `persist` | 8 of 8 | +12.14 |

Under the published statistic a fitted operator is **centred on its own null**. Registered limit: 15
draws floor the empirical p at 1/16, so the inferential statement is the across-corpora count, not
any single corpus. Scoping: a derangement is a null for **correspondence**, not for "the Jacobians
carry no layer-specific information."

**Report the corpus-clustered `t` as the inferential result and the draw count as descriptive**,
because five derangements of one operator share an operator and a cache. At the **corrected**
readout that `t` is **t(7) = −1.92, p = 0.096** over **84/120** draws and 7 of 8 corpora
(`results/paper_clustered_derangement_t.json`); the **t(7) = −3.3 / 104 of 120 / 21 of 24** triple
carried here until 2026-08-24 is a **legacy-readout** number. The 21-of-24 operator count has not
been recomputed at the corrected readout.

→ `experiments/E48_crossover.md`, `R9_permutation_calibrated_min.md`

## 6. OPEN

Full edit list with replacement sentences:
the validity synthesis in `DIAGNOSIS.md` (49 rows).
Ranked experiment slate: the validity synthesis in `DIAGNOSIS.md` §7.

1. **R0 — the corpus-axis crossover retraction. OPERATOR-ONLY, NOT DONE.** Its measurement is
   confirmed twice. No canonical document has been edited under it.
2. **The 58x seed-SD denominator** (§3) is **resolved and needs landing.** The discrepancy was a
   pooling convention, not the ragged grid: the N-grid choice moves the ratio 0.1%, the SD convention
   moves it 37%. Sample SD pooled by RMS is the defensible estimator and gives **19.8**; R8's stored
   22.166 is sample SD with arithmetic-mean pooling. **This does not touch R8's 31.9% verdict**,
   because the registered rule is a ratio with identical estimands on both sides.
3. **Verdict prose in several experiment documents still leads with `persist`.** Every `min` number
   is stored; only the ordering needs refreshing.
4. **P0's wallclock clause is unmet** on this hardware (~1.6–2x, not the registered 3x). Flagged,
   not re-cut.
5. The three independent audits in [`../docs/reproducibility/audit-prompts/`](../reproducibility/audit-prompts/) have not been run.
6. **E37's adjudication has not propagated.** `PREREG_E38_JGEOMETRY.md` §WHY retracted E37's headline
   as vacuous, and `E38_geometry.md` grades itself **Tier B** and says "precondition, not cause" —
   but §2 here and `RESULTS_TAXONOMY.md` §1.1 still present the rank story as Tier A. The experiment
   document is more careful than the two summaries citing it.
7. **`e48c_exposure_vs_read.json` is on the pre-correction readout** (`logit_baseline` 0.0284395), so
   the exposure-ordering claim filed at `RESULTS_TAXONOMY.md` §1.4 is **Tier B**, not Tier A.
8. **`tables_auto.tex:55` captions leave-one-out as "all eight"**; the stored value is **6/8** under
   both aggregations.
9. **The 1B ladder has never been re-scored at the corrected readout** — no cell carries an `rstrip`
   key. All cross-scale claims are pre-correction.
10. **R4b fired STOP-AND-ALERT and the alert never reached a live document. Recorded here, late.**
    `results/r4b_e36_flatness.json` carries
    `VERDICT: REJECT OVERTURNED — flatter on only 1 of 5. FLAGGED FOR ADJUDICATION.`
    That verdict is adjudicated on **`persist`**. Under **`min`**, the paper's declared primary, the
    same registered rule returns **UNCLEAR (3 of 5)**:

    | aggregation | flatter on | the rule's verdict |
    |---|---|---|
    | `persist` (as stored) | 1 of 5 | REJECT OVERTURNED → stop and alert |
    | `min` (declared primary) | 3 of 5 | UNCLEAR → report and stop, do not re-cut |

    **Nothing downstream is void.** The paper makes no shift-axis flatness claim, and E36's crossing
    count is unchanged by the readout correction: `n_crossing` is **0 of 5 fitting corpora under
    both** the stripped and unstripped arms, which is the published "0 of 5 → REJECT S3". The gap
    was procedural, not evidential: the protocol says a STOP-AND-ALERT reaches the operator, and
    this one reached only its own results file.

    The file additionally flags `WORDING_AMBIGUITY_IN_THE_RULE` — whether "5 rungs" means model
    rungs or fitting corpora — and resolves it over the five fitting corpora **without sign-off**.
    That ambiguity is reported here rather than resolved retrospectively, per the rule against
    reinterpreting a registered decision after seeing the result.

## 7. RETRACTED — do not restate

| claim | why it is dead |
|---|---|
| the corpus-axis crossover: Github at or below the free baseline | corrected readout: **+21.1%**, all eight corpora above it |
| "the fitting corpus is worth roughly an order of magnitude more than the read corpus" | 53.4 / 44.6 under `min` |
| the layer-shuffled Jacobian scoring 5/6 as a *finding* | an artifact of min-over-layers. Enforced by `repro/lib/banned_claims.py`; `./lab health` H11 fails if it reappears |
| F19, r(dispersion, plateau AUC) = −0.938 | killed by its own leave-one-out control |
| the −0.511 sample-size law as a discovery | close to a mathematical identity |
