# DIAGNOSIS — why the absolute numbers are low, and what that does and does not mean

**Written 2026-08-22. This is the current state of the problem and the entry point to
[`validity/`](README.md).** It synthesises [`CONSTRUCT_VALIDITY.md`](CONSTRUCT_VALIDITY.md), the
2026-08-22 audit , and the two measurements commissioned out of
them (`results/cv1_answer_competence.json`, `results/cv2_position_support.json`).

**A null is a finding, and so is a validity result.** Nothing below says the programme failed. It
says we have been attributing a number to the wrong cause, and we can now name the candidates and
price them.

---

## 1. THE PROBLEM IN ONE PARAGRAPH

Jacobian-lens recovery on Pythia is low in absolute terms — pass@k AUC around 0.25 under `min`, and
the fitted lens loses to the free logit lens on several arms. The natural reading was *"the J-lens
degenerates on small models"*, and that reading is the programme's lead claim. **Three superimposed
effects are now measured, and none of them is "the lens is bad."** The model fails 95% of the
battery; the operator is evaluated entirely outside the positional range it was estimated for; and
the operator is a 200-sample average of a possibly heavy-tailed quantity. Until those are separated,
low recovery cannot be attributed to the lens.

---

## 2. THE DISTINCTION THAT ORGANISES EVERYTHING: BIAS vs VARIANCE

> **Variance** — the estimate would move if we resampled the fitting data, the seeds, or the items.
> Fixed by more seeds, more windows, wider intervals.
>
> **Misspecification** — the operator is being evaluated at a condition it was not estimated for.
> **Not fixed by any amount of resampling.**

The positional mismatch is the second kind. That is why it outranks everything else on the slate,
including the aggregation question, and why more seeds would not have helped.

---

## 3. WHAT IS NOW MEASURED

### 3.1 Task competence — `results/cv1_answer_competence.json`

Rank of the **expected answer** (not the annotated intermediate), 255 items carrying a `target`.
Nothing had ever measured this: `capability_*` measures the *intermediate's* rank, and
`experiments/t36_qladder.py:483` records the registered competence gate as degenerate and unable to
fire in either direction.

| model | top-1 | top-10 | multihop med. rank | order-ops | multilingual |
|---|---:|---:|---:|---:|---:|
| 70M | 1.6% | 16.9% | 80 | 18 | 3,436 |
| 160M | 1.6% | 22.4% | 42 | 13 | 3,788 |
| **410M** | **4.7%** | **26.3%** | 31 | 9 | 3,044 |
| 1B | 4.3% | 29.4% | 18 | 7 | 1,649 |

* **410M answers 4.7% correctly.** 1B is no better at top-1. Every headline is measured on a battery
  the model overwhelmingly fails.
* **Multilingual is at floor at every scale** — top-1 ~1%, median rank 1,649–3,788 of ~50k — and is
  **107 of 449 admitted items (24%)**.
* **Competence does not explain the S1 floor.** The gradient (1.6 → 1.6 → 4.7 → 4.3) has no
  discontinuity at 160M→410M, where the `W_U` conditioning cliff sits (0.9218 → 0.0307). If the
  floor were a competence artifact, competence would jump there. **S1 survives this specific
  confound.**
* **Caveat, and it matters:** all four models are near the floor, and a floor measurement can
  conceal real differences. Absence of a jump is weaker evidence than a jump would have been.

### 3.2 Positional support — `results/cv2_position_support.json`. CONFIRMED.

`fast_fit` averages the Jacobian over `valid_position_mask(seq_len, skip_first=16)` — positions
**16–126** for a 128-token window.

| condition | median read position | range | **in support** |
|---|---:|---:|---:|
| `Q0` (no prefix) | 14 | 5–49 | **45.7%** (252/551) |
| prefixed rungs | 142 | 133–177 | **0.0%** (0/551) |

**Not one of 551 items reads inside the operator's positional support on any prefixed rung.** Q0
reads *below* support on 54% of items, inside the attention-sink zone the fitter deliberately
excludes. The two arms extrapolate in **opposite directions**, separated by a systematic
~128-position displacement.

**This is conditional on the Jacobian being position-dependent, which is not yet tested.** If it is
near-invariant the effect is small. Testing it is the top priority (§6).

### 3.3 The containment fix does not transfer — measured

The obvious construct check — stratify items by corpus association — cannot use the existing index.
**92% of items (505/551) are shorter than k=32**; median prompt is 15 tokens. A 32-gram cannot be
formed from a 15-token prompt. What is needed is a **(trigger, intermediate) co-occurrence index**,
a new build on the same 20-shard bitmap.

---

## 4. THE FOUR VARIANCE SOURCES, AND WHAT THEY CAN AND CANNOT EXPLAIN

**A. Jacobian-estimation variance.** The operator is a mean of per-example Jacobians over 200
windows. If per-sample Jacobians are heavy-tailed — which the source itself notes — 200 may be a
noisy estimate, especially for small models and rare vocabulary directions. Could produce unstable
ranks, seed-to-seed variation, apparent corpus sensitivity, and inconsistent layer behaviour.

**B. Readout rank variance. TESTED — and the corpus effect survives, larger.** CV3
(`results/cv3_margins_410m.json`, prereg `docs/experiments/preregs/CV3_margins.md`) scored 19 arms on
identical cached activations in three spaces. **PRIMARY = 2.46 → ACCEPT, SCORE EFFECT.**

| space | spread | pooled seed SD | spread/SD |
|---|---:|---:|---:|
| rank (`min` pass@k) | 0.02418 | 0.002581 | **9.37** |
| margin | 0.40060 | 0.017375 | **23.06** |
| z | 0.36420 | 0.013407 | **27.16** |

The corpus effect is **not** a ranking artifact, and rank **understates it by 2.5x**. Ordering agrees
across spaces: StackExchange > Pile-CC > USPTO > Wikipedia_en > Github. All three controls fired;
`logit_I` reproduced the stored 0.19810852520167826 exactly.

**The secondary refines the metric audit.** Derangement contrast, real minus shuffled: rank
**-0.00554**, margin **-0.3942**, z **+0.0855**. The derangement wins on rank and margin and
**loses on z**. It is not scoring the target higher — it is **suppressing the nearest competitor**.
The pre-registration's DECLARED BIAS predicted exactly this and z resolves it. So: **`min` and margin
both reward competitor suppression; z does not.** Report in z-space. Scope: one operator (Pile-CC
s0), three derangements, one model.

**A summary-convention divergence, found by the control.** CV3's first run failed C1 at 0.18966
against the stored 0.19811 — because it summarised pass@k as a trapezoid **AUC over log k** while
this programme's convention is a **flat mean over the 7 k values** (`CONFIG_MATRIX.md`). On identical
ranks the two summaries differ by **4.3%**. Within-pipeline results are unaffected because everything
is scored consistently, but **any head-to-head number quoted against the source must state which
summary it uses** — external review reports Gurnee et al.'s as log-k AUC.

**C. Item and task variance.** The battery is heterogeneous and aggregate pass@k is dominated by
family composition. Multilingual is at answer-floor; multihop and order-ops are less hopeless.
Related: `results/t17_reaggregate_410m.json` flags **poetry, typo and association as
AGGREGATION-DEPENDENT** under a pre-registered rule that says they are *"excluded from any claim"* —
and poetry and typo are 2 of the 5 admitted sets behind every headline.

**D. Aggregation variance.** `min` is an extreme-value statistic over a 13-layer band, so a small
perturbation anywhere can move the minimum. Real, and now demonstrably a **gradient rather than a
binary**: real-operator win counts run **2/5 < 3/5 < 4/5 < 5/5** across `min`, `best1L`, `mean`,
`persist`. Secondary to A–C.

### What variance cannot explain

1. **0% positional support** on the prefixed arm.
2. The systematic ~128-position displacement between `Q0` and prefixed conditions.
3. A battery that is 95% unanswered.
4. Systematic extrapolation bias, if the operator is position-dependent.

Variance can obscure these, inflate or shrink apparent contrasts, and make everything harder to
read. **It cannot turn an out-of-support evaluation into an in-support one.**

---

## 5. WHAT SURVIVES, AND HOW IT MUST NOW BE LABELLED

### CV6 — the comparison replicates up the ladder, per family. 2026-08-24.

The objection CV6 was built to answer: *410M answers 4.7% of this battery, so an effect measured in
a near-degenerate regime may be structured noise.* It is now answered at 2.8B, in the per-family
estimand D3 showed is the honest one.

**VERDICT: REPLICATES.** `R(f,2.8B) >= 10` on **4 of 5** families (multihop 11.44, multilingual
14.51, poetry 11.93, typo 11.41; order-ops 6.46) and ordering tau `>= 0.6` on **5 of 5**. All six
controls fired. `results/cv6_per_family_ladder.json`.

Three things this does and does not license:

1. **It does not rehabilitate the absolute claims.** cv1 and cv2 retired those and CV6 did not
   reopen them. The battery is still ~95% unanswered and the read is still outside the operator's
   positional support. CV6 tests whether the *comparative* effect survives scale, nothing more.
2. **It does not license "the effect weakens up the ladder."** `R` fell from 19–43 to 6–15, and
   that is a denominator artifact of N=25 vs N=200 — the **numerator grew in 5 of 5 families**
   while the pooled seed SD grew 2.29x–5.70x. In absolute z the effect is **larger** at 2.8B than
   at 410M in every family.
3. **It strengthens the conditional label, it does not remove it.** §5 below still stands: position
   and corpus may interact, so the finding remains *"corpus X produces an operator that extrapolates
   better to these read positions."* CV6 holds the read protocol fixed across the ladder, so it
   cannot separate the two. The matched-position experiment (§8.1) is still the way out.

**A corpus x task interaction replicated at a new scale.** D3 found USPTO tops order-ops and only
order-ops at 410M, with a mechanistic story (patent text is number-dense). At 2.8B, USPTO tops
order-ops again. D3 was exploratory; this is the registered replication.

**order-ops is the one family that misses the bar**, R 6.46 at 2.8B and R 4.7 with an **inverted**
ordering (tau -0.40) at 1.4B — and it is also the highest-competence family (14.5% top-1). Recorded
as an open irregularity, not explained.

### Survives — the corpus comparison, conditionally

Every cell of the 8×8 shares the task item, target, model, read position, aggregation and paired
seeds. So it estimates a genuine conditional effect:

> *at this particular, possibly misspecified read position, how does the fitting corpus change
> recovery of the fixed target list?*

Measured at **13–45 pooled seed SD** depending on aggregation and eval-set list. The ground-truth
problem does not touch it, because the battery is byte-identical across cells.

**But the label must change.** Position and corpus may interact: one fitting corpus might yield an
operator that extrapolates more tolerantly to position ~142 than another's. If so the finding is not
*"corpus X produces a better J-lens"* but *"corpus X produces an operator that extrapolates better to
the chosen read positions."* **Do not report the corpus effect as evidence of better workspace
reading until the matched-position experiment is done.**

Also unstable and needing restatement: the fine ordering. Dropping the two aggregation-dependent
sets changes the ordering under **every** aggregation, most sharply under `persist`, where Github
moves from last to second-best. And the independent 5×5 replication gives 5/5 under `min` and 0/5
under `persist`.

### Not yet supported

> **"The J-lens is degenerate at 70M/160M."**

Defensible replacements: *"the J-lens has low recovery on this battery at these scales"*, or *"the
current protocol does not establish latent workspace recovery in these models."*

### Not established

> **"The J-lens reads Pythia's workspace."** No document may drift into this.

---

## 6. THE FIVE THINGS THAT MUST BE KEPT APART

The `W_U` conditioning cliff is a **separate phenomenon** from everything above. Poor conditioning
can make vocabulary ranking unstable even when the underlying J-space directions are meaningful; a
well-conditioned map does not rescue an out-of-support operator or an impossible task.

1. representation / operator quality in residual space
2. mapping quality through `W_U`
3. task competence
4. readout ranking quality
5. causal usefulness

We currently measure 4, partially 2 and 3, and never 1 or 5.

---

## 7. REVISED JUDGMENT — how much each explanation is worth

| explanation | status |
|---|---|
| **position mismatch** | confirmed as a design mismatch; **effect size unknown**; high-priority systematic candidate |
| **task incompetence** | measured; fatal to absolute and mechanistic claims, not to comparative ones |
| **Jacobian estimator variance** | untested; likely a substantial contributor to instability and corpus sensitivity |
| **`min` brittleness** | real, quantified as a gradient, now **secondary** |
| **true small-model degeneracy** | possible, **not established** |
| **corpus-dependent operator differences** | credible as a conditional comparative result |

> **The data show that the protocol behaves badly on this battery. They do not show that Pythia's
> J-lens is intrinsically degenerate.**

---

## 8. WHAT TO DO, IN ORDER

1. **Matched-position pilot — the top priority.** Fit operators at several controlled positions and
   cross them against read positions:

   | operator fitted at | read at | question |
   |---|---|---|
   | in-support | same | baseline validity |
   | in-support | distant | extrapolation penalty |
   | distant | distant | matched-position performance |
   | distant | original | reverse extrapolation |

   Measure: target recovery; **held-out Jacobian reconstruction error**; cosine between operators at
   different positions; singular-value spectra / condition numbers; target-score margins. The
   sharpest single diagnostic is whether an operator fitted near position 16 predicts held-out
   Jacobians at position 143 — that is direct evidence for or against extrapolation bias.

2. ~~**Report continuous scores, not only ranks.**~~ **DONE — CV3.** Corpus effect confirmed as a
   score effect at 27σ in z-space; z is the reporting choice robust to competitor suppression.
   Remaining: extend the derangement arm to 8 corpora x 3 seeds x 5 derangements in z-space (same
   code path).

3. **Stratify by answer rank and task family.** Regress recovery on a continuous competence variable
   (e.g. −log p of the answer) rather than top-1 alone. Recovery on items where the answer sits at
   rank 10–50 is evidence about *partial* computation, which is still interesting, and must be
   labelled as such.

4. **Estimate Jacobian variance directly.** Refit a small set of operators with more windows and more
   seeds; measure how much the operator itself moves.

5. **Build the (trigger, intermediate) co-occurrence index** and stratify by association strength.

6. **Keep the corpus comparison — relabelled as conditional on the current read protocol.**

7. **Do not spend more time on `min` vs `persist` until 1–4 are resolved.**

---

## 9. THE POSTURE

The strongest available contribution may now be the validity result itself:

> **Corpus-dependent J-lens comparisons can be internally reproducible while absolute readout claims
> remain invalidated by task competence and positional support.**

That is a real methodological finding for a measurement-validity venue, and it is arguably a better
fit for *Interpretability as a Science* than the original headline. And the matched-position
experiment may yet show that the low numbers were largely an evaluation artifact rather than a
property of the lens — in which case the absolute claims come back, on a footing they never had.

**Related reading:** [`CONSTRUCT_VALIDITY.md`](CONSTRUCT_VALIDITY.md) for where the ground truth
comes from and the two routes (salvage the annotated battery vs build a causally-grounded synthetic
suite) · the validity synthesis in `DIAGNOSIS.md` for what is quantified
today · [`README.md`](README.md) for what Gate 1 is and how to run it.
