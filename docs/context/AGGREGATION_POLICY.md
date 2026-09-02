# AGGREGATION POLICY — `min` is primary, `persist` is a labelled secondary

**Ruling 2026-08-20, operator-directed. This supersedes `HANDOFF.md`'s "`persist` primary, `min`
never votes" wherever the two conflict.** Every number below is stored; nothing here required a
re-run, because every scoring path in this repository has always emitted both aggregations.


FIRST. BEFORE ANYTHING ELSE, READ THE APPENDIX

---

## THE RULING

1. **`min`-over-layers is PRIMARY throughout.** It is the source's own operational definition of
   recovery — an intermediate counts as recovered if it appears in the top-k at *any* layer,
   motivated by representations being transient and evolving with depth.
2. **`persist` is DEMOTED to a secondary robustness arm, and it carries a label wherever it
   appears:** *selected with knowledge of the derangement-control outcome.*
3. **The permutation-calibrated form of `min` (R9) is what answers the metric question**, not a
   switch of metric.

## WHY — and this is the whole argument

Adopting `persist` because it passed the derangement control, and then citing that control as its
justification, is **outcome-dependent metric selection**. Both external reviews name it
independently as circular in the Kriegeskorte sense. No amount of downstream rigour repairs a metric
chosen on the test it has to pass. `persist`'s "half the band" threshold is itself an unjustified
researcher degree of freedom, which is a second reason it cannot be the adjudicator.

**What the source does and does not do**, confirmed independently by both reviews. `min` is its
published detection rule. The source never derives it from a null-calibrated argument, tests
band-width sensitivity, compares it against persistence/mean/median, permutes the
layer-to-Jacobian correspondence, or discusses union inflation under decorrelation. Its ablations
vary **how `J` is computed** — mean vs median, present vs future positions, frozen attention —
never **how evidence is aggregated across layers**. That axis is genuinely untouched.

**And the source noticed the gap.** At p.33 it counts an item present if its best rank over the
workspace range is top-25, while noting the number represented at an individual layer is smaller.
It saw the difference between "any layer" and "a layer" and did not analyse its null behaviour.
This is not a straw man; it is a question the source raised in passing and left open.

## THE CLAIM, in one sentence

> **The ordering reverses when the aggregation changes from single-layer quality to
> opportunity-counting.**

`min` detects **existential availability**. It cannot certify **layer-to-derivative
correspondence**: a fixed-point-free derangement that preserves entries, norms and spectra scores
higher under it. The **7-of-13 band-width crossover** is the diagnostic that says when the statistic
stops tracking the intended signal.

**Scoping caveat, and it is binding.** This is a failure of *procedure-plus-aggregation* to
distinguish a correct operator from a mismatched one. It is **not** evidence that the Jacobians lack
layer-specific information, and no document may drift into the stronger claim.

## R9 — the calibration that dissolves the circularity

Keep the published statistic; ask whether the observed value is unusual **relative to its own
derangement null**. `results/r9_permutation_calibrated_min.json`, corrected readout, 15 own-operator
derangement draws per corpus:

| aggregation | beats **every** one of its 15 own derangements | median z vs its own null |
|---|---|---|
| **`min` (primary)** | **0 of 8 corpora** | **−0.71** |
| `persist` (secondary) | 8 of 8 | +12.14 |

Under the source's own statistic the real operator is **centred on its own null** — Pile-CC z =
−1.99, Wikipedia_en z = −1.67, only Github positive at +0.59. **Said without selecting the metric on
the control it has to pass.**

*Declared limit:* 15 draws floor the empirical p at 1/16 = 0.0625, so no single corpus can reach
p < 0.05 on this null. Per-corpus values are ranks with a floor; the inferential statement is the
across-corpora count.

## WHAT CHANGES, MEASURED — the switch is favourable on balance

Corrected readout throughout. **These numbers already existed**; only which one leads changes.

| result | `persist` (old primary) | **`min` (new primary)** | |
|---|---|---|---|
| R1 fit / read | 50.70 / 48.13 | **53.35 / 44.61** | |
| R1 fit − read interval | +2.57 **[−7.60, +9.80]**, includes 0 | **+8.74 [+0.80, +15.87], excludes 0** | **better** |
| E55 independent 5×5 replication | fit 37.2 / read 61.5 — the read axis **wins** | fit 67.2 / read 30.9 | **the aggregations disagree on the DIRECTION** |
| R5 leave-two-out ordering | 16/28 | **20/28** | better |
| R6 between-source share | 0.9723 | **0.9886** | better |
| R4b flatness, S3 read axis | 1/5 → REJECT OVERTURNED | 3/5 → **UNCLEAR** | verdict changes |
| **R8 / S2, N ÷ corpus axis** | 14.2% (inside the 25% bar) | **31.9% (outside it)** | **worse** |

**The largest single gain:** under `min`, E57's own pre-registered rule **no longer fires REJECT** —
the interval on `fit − read` excludes zero. The paper can state that the fit axis dominates.

**The one cost, not buried:** S2's N-axis ratio fails its bar under `min`. R8 remains **UNCLEAR**,
now on the primary aggregation rather than the secondary. Its diagnosis is unchanged and is in
`R8_ladder_rstrip_S2.md`: four of five corpora are flat in absolute terms and the ratio moves
because the *corpus* axis shrinks 42% under `min`, not because N starts to matter.

## REPORTING RULES

1. `min` leads every table, figure and sentence. `persist` appears beside it, never instead of it.
2. Every `persist` citation carries the label: *selected with knowledge of the control outcome.*
3. **The clustered `t` is the inferential result; the draw count is descriptive.** At the
   **corrected** readout it is `t(7) = −1.92, p = 0.096`, which does **not** clear 5%
   (`results/paper_clustered_derangement_t.json:by_readout.corrected.min`). The `t(7) = −3.3,
   p = 0.013` this section carried until 2026-08-24 is a **legacy-readout** number; the readout
   correction moved this statistic from significant to not. Say so wherever it appears. The verdict
   does not change: `min` INADMISSIBLE rests on R9 (0 of 8 corpora beat all 15 of their own
   derangements, median z −0.71), which does not require the derangement to win. What weakens is
   only the claim that `min` actively *prefers* it. With 8 clusters, a plot of per-corpus paired
   differences communicates better than the `t`.
4. **Name the estimand for the derangement counts.** At the corrected readout the two estimands
   **coincide**: `shuf_beats_jp_paired_by_seed` = **84** and `shuf_beats_jp_vs_corpus_mean` = **84**
   (`e54_aggregation_audit_rstrip_v2.json`). The `104` vs `103` disambiguation is a statement about
   the pre-correction arm only.
5. The derangement floor used as a *control* in R6/R7 stays anchored on `persist`, and says why:
   `min` is the aggregation that prefers derangements, so a `min`-anchored floor would fail for the
   pathology already established rather than for a defect in the operator under test.

## WHY THIS TRAVELS FURTHER

*"A published existential statistic cannot certify operator correspondence, and here is the band
width at which it breaks"* is usable by any layerwise-probing paper. It was already the most
transferable result here; scoping it properly widens its reach rather than narrowing it. And
**E54 confirms the VERDICT is convention-independent** — `persist` 0/120 ADMISSIBLE, `min`
INADMISSIBLE at both readouts, 7 of 8 corpora both ways. The **counts** are not
convention-independent: the `min` draw count moves 104/120 → **84/120** and the clustered `t` moves
−3.28 → **−1.92**. Cite the verdict as identical, never the count.

---

# APPENDIX: Are we doing this right or no?

The question put to an external model: reading Verbalizable Representations Form a Global Workspace in Language Models by Gurnee et al., what are the appropriate readout success metrics? That work uses min (the lowest rank, where rank 1 is lowest, of the concept token that is successfully read across all reads). On Pythia, min proves brittle as a metric. What is the existing art on min versus persist — not only Gurnee et al., but more broadly on how readouts are structured? What counts as a successful read, and at what layer? How is the performance of the logit and tuned lenses evaluated? Is persist a real trend, is min the right choice, do these aggregations make sense, or is another approach called for?
+the above context

**Response**: 

Short version: **`min` and `persist` are not competing estimators of one quantity. They answer different questions.** I would keep `min` as the primary metric only if your scientific claim is existential:

> “Is the concept available to the readout somewhere in the specified layer band?”

If your claim is instead:

> “Is the concept stably represented, and does the readout track the corresponding operator across layers?”

then `min` is inadequate by construction. Use persistence or a trajectory statistic—but define it before looking at the control outcome.

Your proposed ruling is therefore directionally right: **retain `min` for continuity with Gurnee et al., add persistence as a robustness/diagnostic arm, and calibrate both against an appropriate null.**

## 1. What `min` actually measures

Let $r_{i,\ell}$ be the rank of the target concept for item $i$ at layer $\ell$, with rank 1 best. Then the Gurnee-style statistic is

$$
r_i^{\min} = \min_{\ell \in B} r_{i,\ell},
$$

where $B$ is the selected workspace layer band. A concept is recovered at top-$k$ if

$$
\min_{\ell \in B} r_{i,\ell} \leq k.
$$

This is an **existential** statistic: “the concept appeared at least once.”

That is exactly how [*Verbalizable Representations Form a Global Workspace in Language Models*](https://arxiv.org/abs/2607.15495) defines its quantitative readout evaluation: an intermediate is recovered at $k$ if it appears among the top-$k$ tokens **at any layer**, and pass@k is the fraction of intermediates satisfying that criterion. [Readout Metric](https://arxiv.org/abs/2607.15495)

So `min` is not an accidental implementation detail. It is the paper’s published operational definition.

But it has a predictable weakness. If each layer has a false-hit probability $p$, then under independence

$$
P(\text{at least one false hit}) = 1-(1-p)^{|B|}.
$$

The layers are not independent, but the qualitative point remains: **a larger search band gives the metric more opportunities to succeed**. `min` rewards availability, not stability, layer specificity, or operator correspondence.

The paper itself acknowledges this distinction in its list-loading experiment: counting a word as present if its best rank anywhere in the workspace band is top-25 yields more represented items than looking at one layer alone. [Band Aggregation](https://arxiv.org/abs/2607.15495)

That is the central conceptual distinction in your analysis.

## 2. What does “successful readout” mean in the literature?

There are at least four different meanings.

### A. Predicting the model’s eventual output

This is the usual logit-lens/tuned-lens objective.

The [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) directly applies the final unembedding to an intermediate residual stream. The [tuned lens](https://arxiv.org/abs/2303.08112) instead learns a layer-specific affine translator so that an intermediate representation can be expressed in the final layer’s coordinate system. It is trained by minimizing KL divergence between the intermediate prediction and the model’s final prediction. [Tuned Lens Objective](https://arxiv.org/abs/2303.08112)

The standard metrics here are:

- per-layer cross-entropy or perplexity;
- KL divergence to the final model distribution;
- top-1 or top-$k$ agreement with the final prediction;
- sometimes calibration or entropy;
- sometimes the layer at which prediction stabilizes.

This is **not** usually evaluated with `min`. The natural object is the entire layerwise curve:

$$
\ell \mapsto \operatorname{KL}(p_{\text{final}}\|p_\ell)
$$

or

$$
\ell \mapsto -\log p_\ell(y).
$$

For this task, a best-layer score would often be misleading: it answers whether the model was accurate at least once, whereas early-exit or iterative-refinement claims concern how prediction quality evolves.

The tuned-lens paper explicitly treats the sequence of per-layer distributions as a **prediction trajectory**, and reports layerwise perplexity rather than selecting the layer with the best result. [Prediction Trajectory](https://arxiv.org/abs/2303.08112)

### B. Recovering a known latent intermediate

This is closer to the Gurnee experiment. You have an externally specified target—say “Mars” in a question whose final answer is “red”—and ask whether the target appears in the readout.

Here, top-$k$ hit rate is reasonable. But you must specify the estimand:

- **best-layer hit:** was it ever present?
- **fixed-layer hit:** was it present at layer $\ell$?
- **band-average hit:** was its average score over the band high?
- **trajectory hit:** did it enter, remain, and then perhaps leave?
- **causal hit:** does manipulating the corresponding direction change the computation?

The Gurnee paper uses the first of these. It also makes clear that its readout is a ranked vocabulary list: the J-lens produces a score for every vocabulary token, and ranks those scores to obtain the “top lens tokens.” [J-Lens Readout](https://arxiv.org/abs/2607.15495)

That is a defensible metric for **availability**, but not for “the concept is represented throughout the workspace.”

### C. Recovering information from a hidden state

The [Future Lens](https://arxiv.org/abs/2311.04897) illustrates another convention. It evaluates hidden states by whether a probe can predict the model’s future tokens, using Precision@k and surprisal. It reports performance at each layer, including the layer where performance peaks, rather than using an any-layer recovery criterion. [Future Lens Metrics](https://arxiv.org/abs/2311.04897)

This is important because it shows that a “successful read” is normally defined relative to a **prediction target and a layerwise performance curve**, not merely whether one favorable layer produced the target.

### D. Demonstrating that a readout corresponds to the model’s computation

A readable token is not necessarily the direction the model is using. The J-lens paper makes this distinction explicitly: it evaluates both readout and intervention. Ablating a lens direction measures output KL change; swapping two lens coordinates tests whether the model’s output changes to the corresponding answer. [Causal Validation](https://arxiv.org/abs/2607.15495)

For claims about **operator correspondence**, I would regard these causal or null-control metrics as more important than persistence.

## 3. Is persistence an established trend?

There is a trend toward analyzing **trajectories**, but not, as far as I can tell, a standard field-wide “persist” metric with a canonical threshold.

The tuned-lens literature commonly studies:

- prediction trajectories;
- monotonicity or smoothness of output refinement;
- prediction depth—the point after which the top-1 prediction stops changing;
- per-layer agreement with the final output;
- the relationship between internal prediction depth and example difficulty.

For example, tuned-lens work defines prediction depth using when the top-1 prediction stops changing, and correlates it with when an example was learned during training. [Prediction Depth](https://arxiv.org/abs/2303.08112)

That is conceptually related to persistence, but it is not the same as “the target is in the top-$k$ for at least half the layers.” Prediction depth concerns **stabilization of the model’s output**, while your `persist` statistic concerns **continued presence of a specified latent concept**.

So I would not describe `persist` as an established consensus metric. I would describe it as a sensible, interpretable trajectory statistic that needs a preregistered definition.

The important distinction is:

| Statistic | What it asks |
|---|---|
| `min` / best rank | Was the target available anywhere? |
| Mean or median rank | How strong was the target on a typical layer? |
| Occupancy | What fraction of layers contained the target in top-$k$? |
| Longest run | Did the target remain continuously readable? |
| Entry layer | When did the target first become readable? |
| Exit layer | When did it disappear? |
| Prediction depth | When did the output stop changing? |
| Causal effect | Does manipulating the readout direction affect computation? |

Your half-the-band threshold is therefore not wrong, but it is a researcher-chosen operating point. It should not become the adjudicator merely because it succeeds on a derangement control.

## 4. How I would interpret your Pythia result

Given the numbers in your policy, the pattern is quite informative:

- `min` improves the R1 fit–read interval;
- it changes E55 from non-replication to replication;
- it improves R5 and R6;
- it makes R4b less decisive rather than clearly positive;
- it worsens the R8/S2 $N$-axis result;
- under R9, the real operator is not separated from its own derangement null under `min`, whereas persistence is.

That is not evidence that `min` is “wrong.” It says that **the source’s metric and the correspondence-control metric are testing different properties**.

I would phrase the result as:

> The published `min` statistic supports existential availability of the target concept, but under the permutation-calibrated control it does not distinguish the correctly matched operator from deranged layer correspondences. The persistence statistic is more robust to that control, but was selected after observing the control outcome and therefore cannot replace the primary metric without circularity.

That wording preserves both facts. It does not overclaim that Jacobian representations lack layer-specific information. It says that **the procedure-plus-aggregation combination fails to establish correspondence**.

Your R9 result is especially useful because it avoids the circular choice. The correct question is not simply “which metric passes the derangement?” but:

> For each prespecified metric, is the observed statistic unusual relative to the same metric computed under its own null?

That is the right calibration principle.

## 5. What I would report going forward

I would report a small, fixed panel rather than choose between `min` and `persist`.

### Primary: the full layerwise curve

For every item, retain $r_{i,\ell}$ or the continuous score $s_{i,\ell}$. Report:

- median rank or mean reciprocal rank by layer;
- top-$k$ hit rate by layer;
- mean logit margin or log-probability by layer;
- confidence intervals across corpora or prompts.

This prevents aggregation from hiding where the signal occurs.

### Secondary 1: existential availability

Report

$$
A_k = \frac{1}{N}\sum_i \mathbf{1}\left[\min_{\ell \in B} r_{i,\ell}\leq k\right].
$$

Call it **best-layer**, **any-layer**, or **existential top-$k$ recovery**, rather than relying on the ambiguous label `min`.

### Secondary 2: occupancy

Instead of only a half-band threshold, report the whole occupancy curve:

$$
O_k(q) =
\frac{1}{N}\sum_i
\mathbf{1}\left[
\frac{1}{|B|}
\sum_{\ell\in B}
\mathbf{1}[r_{i,\ell}\leq k]
\geq q
\right],
$$

for prespecified values such as $q \in \{0.25,0.5,0.75\}$.

This lets readers see whether your conclusion depends specifically on “half the band.” A more stable scalar is simply the mean occupancy:

$$
\bar O_k =
\frac{1}{N|B|}
\sum_{i,\ell}
\mathbf{1}[r_{i,\ell}\leq k].
$$

### Secondary 3: temporal structure

Occupancy alone ignores whether the hits are scattered. Add:

- longest contiguous top-$k$ run;
- number of entry/exit transitions;
- first-entry layer;
- last-exit layer;
- normalized run length.

A concept appearing in layers 40, 50, 60, and 70 is not equivalent to one appearing continuously from 40–70, even though they can have identical occupancy.

### Calibration: apply every metric to its own null

For each statistic $T$—best rank, occupancy, longest run, area under the rank curve—compute

$$
z_T =
\frac{T_{\text{real}}-\operatorname{mean}(T_{\text{null}})}
{\operatorname{sd}(T_{\text{null}})}.
$$

Use paired derangements by seed and report the empirical count explicitly, e.g. “104/120 paired-by-seed comparisons,” not just “104/120.” Preserve all layerwise quantities under the null so that the null has the same opportunity structure as the observed statistic.

With only 15 derangements per corpus, your stated empirical-p-value floor of $1/16$ is correct. The across-corpus paired analysis should therefore be the inferential result; per-corpus ranks are descriptive.

## 6. My recommendation

I would use the following hierarchy:

1. **Primary scientific result:** layerwise curves and the preregistered Gurnee-compatible any-layer top-$k$ metric.
2. **Primary inferential control:** the permutation-calibrated version of that same `min` statistic.
3. **Robustness analysis:** occupancy, persistence, longest run, and perhaps area under the rank/score curve.
4. **Mechanistic claim:** intervention or swap success, not readout rank alone.
5. **Interpretation:** distinguish “available to a decoder” from “stably represented” and from “causally used.”

So: **do not abandon `min` merely because it is brittle. Explain precisely what it estimates, expose its multiple-opportunity problem, and supplement it with prespecified trajectory measures.** Your current ruling—`min` primary, persistence secondary and explicitly labeled as selected with knowledge of the control outcome—is methodologically defensible. The strongest result may ultimately be not that persistence is the “correct” metric, but that **aggregation choice changes the estimand and can reverse the apparent ordering of operators.**

**My response**: 
But what constitutes a ROBUST read? 
Also, do Gurnee et al. disavow a final read or identify it as a "property" of the Jacobian lens? 

## What constitutes a **robust** read?

There is no field-standard definition of “robust read.” In this setting, I would reserve **robust** for a read that survives several distinct perturbations:

1. **Layer perturbation:** it is present across a nontrivial interval, not just at one layer.
2. **Rank perturbation:** it remains detectable across reasonable $k$ values.
3. **Readout perturbation:** it survives changes such as logit vs. Jacobian lens, corpus split, Jacobian estimator, or normalization.
4. **Null perturbation:** it exceeds what appears under layer permutations or other matched controls.
5. **Causal perturbation:** intervening on the readout direction changes the relevant computation.
6. **Task perturbation:** it generalizes across prompt templates, positions, and examples.
7. **Semantic perturbation:** nearby tokens, aliases, morphology, or multi-token forms do not arbitrarily determine success.

A one-layer top-10 hit is therefore an **available read**, not necessarily a robust read.

### A useful hierarchy

For a target concept $c$, let $s_{i,\ell}(c)$ be a continuous score and $r_{i,\ell}(c)$ its rank for item $i$ at layer $\ell$.

#### 1. Availability

$$
A_{i,k}=\mathbf{1}\left[\min_{\ell\in B}r_{i,\ell}(c)\leq k\right].
$$

This is the Gurnee-style `min` criterion. It asks:

> Did the concept appear anywhere in the band?

It is appropriate for existential claims, but it is maximally permissive about layer selection.

#### 2. Layerwise reliability

Report

$$
p_{\ell,k} =
\frac{1}{N}\sum_i
\mathbf{1}[r_{i,\ell}(c)\leq k]
$$

for every layer. This shows whether the signal is concentrated in one narrow region or is broadly present.

#### 3. Occupancy

For each item:

$$
o_{i,k} =
\frac{1}{|B|}
\sum_{\ell\in B}
\mathbf{1}[r_{i,\ell}(c)\leq k].
$$

Then report the distribution of $o_{i,k}$, not merely whether it exceeds an arbitrary threshold. A median occupancy of $0.7$ and a median occupancy of $0.08$ can both produce similar `min` scores if the band is sufficiently wide.

#### 4. Continuity

Occupancy does not distinguish scattered hits from a continuous trajectory. Add:

- longest contiguous top-$k$ run;
- first entry layer;
- last exit layer;
- number of entry/exit transitions.

A simple “persistent read” might be defined as a top-$k$ run of at least $m$ consecutive layers, with $m$ fixed before analysis.

#### 5. Null-calibrated robustness

For each statistic $T$, compare it to the same statistic under a matched null:

$$
z_T =
\frac{T_{\mathrm{real}}-\mathbb{E}[T_{\mathrm{null}}]}
{\operatorname{SD}(T_{\mathrm{null}})}.
$$

This is crucial. A `min` score should be compared to a null that also gets to search across all layers. An occupancy score should be compared to a null occupancy score. Otherwise, you are comparing different opportunity structures.

#### 6. Causal robustness

A concept that reads well but has no effect under intervention is not necessarily the representation the model uses. The [Gurnee et al. paper](https://arxiv.org/abs/2607.15495) explicitly separates readout from causal validation: it measures both the output KL change after ablating a lens direction and the success of swapping one concept for another. [Causal Validation](https://arxiv.org/abs/2607.15495)

My preferred terminology would be:

- **available:** passes `min`;
- **reliable:** has good layerwise or occupancy performance;
- **persistent:** satisfies a preregistered duration/occupancy criterion;
- **specific:** beats a matched null;
- **causal:** intervention changes the target computation.

That avoids making one scalar carry all five meanings.

## Does Gurnee et al. disavow a final read?

**No—not exactly.** They do not disavow final-layer readouts in general. They make a more specific claim:

> The final layers are a different functional regime from the intermediate workspace.

The [paper](https://arxiv.org/abs/2607.15495) says that the J-space is defined at every layer, but that it has “workspace-like” properties only in an intermediate band. In the late layers, abstract intermediate concepts give way to representations more directly tied to the imminent output. [Workspace Band](https://arxiv.org/abs/2607.15495)

They call this late regime **“motor”** representation. The final read is still a legitimate J-lens readout; it just usually answers a different question:

- in the workspace: *What concept is the model currently making available for downstream computation or verbal report?*
- in the motor regime: *What token is the model about to emit?*

The paper illustrates this directly. During a copying task, the J-lens readout tracks the instructed internal concept in the intermediate layers, but switches to the next output token in the final layers. [Motor Regime](https://arxiv.org/abs/2607.15495)

Their quantitative boundary analysis uses several signals:

- next-token prediction accuracy rises sharply in the final layers;
- readout kurtosis falls or changes regime;
- cross-position autocorrelation falls as the readout becomes token-local;
- the J-space geometry changes again near the output.

They interpret the late increase in next-token accuracy as marking the end of the workspace proper: the J-lens vectors become motor representations that drive imminent output, rather than intermediate representations available for further processing. [Motor Layers](https://arxiv.org/abs/2607.15495)

So the answer is:

> **They do not reject the final read. They reject treating the final read as evidence of the workspace’s intermediate conceptual content.**

That distinction matters for Pythia. If your target is the latent bridge in a multi-hop computation, the final-layer read may be late-stage answer decoding and therefore not the target of interest. If your target is “what token will the model output?”, the final layer is exactly the relevant read.

## Is the final read a “property” of the Jacobian lens?

It is a property of the J-lens in the limited sense that the lens produces a readout at every layer, including the final layers, and those late readouts converge toward next-token prediction. The paper reports that in the last layers, J-lens, logit-lens, and tuned-lens readouts all collapse toward the same output distribution. [Late Readout Convergence](https://arxiv.org/abs/2607.15495)

But “final-layer readability” is not the defining discovery. The defining claim is that the **intermediate** J-lens readouts expose content that is:

- verbalizable;
- not merely copied from the input;
- not simply the imminent next token;
- causally usable in later computation;
- broadly broadcast across downstream components.

The authors explicitly describe the workspace contents as “neither pure echoes of the input nor predictions of the next token,” while separately saying that the final layers become aligned with the imminent output. [Workspace Content](https://arxiv.org/abs/2607.15495)

## An important caveat: their workspace boundary is partly post hoc

The paper is admirably explicit that the boundary between workspace and motor layers is not yet principled. In its limitations, it says the late boundary was identified empirically and that it did not provide a formal definition of what distinguishes workspace from motor representations. [Boundary Limitation](https://arxiv.org/abs/2607.15495)

That means you should not treat “exclude the final $x\%$ of layers” as a universal law. For each model and task, I would determine the output regime using held-out diagnostics:

1. next-token agreement with the actual model output;
2. similarity to the final output distribution;
3. cross-position versus token-local stability;
4. whether the target intermediate remains causally active;
5. whether the readout is still distinct from the eventual answer.

## My practical definition

For your Pythia experiments, I would define a robust latent read as:

> A target concept is robustly read if it exceeds a preregistered top-$k$ or continuous-score threshold in a contiguous or high-occupancy layer interval, survives reasonable readout and prompt perturbations, beats an identically aggregated derangement/null distribution, and preferably has a corresponding causal intervention effect.

Then report `min` separately:

> `min` recovery measures **existential availability**, not robustness.

And exclude or separately annotate the motor regime rather than silently pooling it with the workspace. In particular, I would report:

- **workspace-only `min`;**
- **workspace-only occupancy/persistence;**
- **motor-regime next-token accuracy;**
- **full layerwise curves;**
- **null-calibrated versions of each.**

That gives Gurnee et al.’s metric its proper role without asking it to establish a stronger claim than it can support.

