# CV8 — POSITIONAL EXTRAPOLATION: is the corpus effect an extrapolation artifact?

**PRE-REGISTERED 2026-08-29, before any CV8 code was written or run.**
Continues the construct-validity series that [`CV2_position_support.md`](CV2_position_support.md)
opened. CV2 measured *that* the read is outside the fitting support. CV8 measures *what that costs*
and *whether it differs by fitting corpus*.

---

## WHY

The paper currently hedges its own headline. It reports a corpus effect measured at read position
142, and then concedes in the Discussion:

> "the label must change, since position and corpus may interact: the finding may be that one corpus
> yields an operator extrapolating more tolerantly to position 142."

That concession is correct and it is unresolved. CV2 established the premise — the fitter averages
the Jacobian over positions 16..126 of a 128-token window, and with a 128-token read prefix **0 of
551 items** are read inside that range (median read position **142**, range **133..177**). So every
number in the paper is produced by an operator applied outside the positional support it was
estimated on.

Both drafts of the paper name this experiment as the fix and do not run it: *"whether an operator
fitted near position 16 predicts held-out Jacobians at position 143 is the sharpest diagnostic
available, and it is cheap."* Naming a cheap decisive experiment and not running it is what makes
the paper read as an audit that stopped early. This registration runs it.

**The estimator makes it cheap, and that is not an accident of our code.** `jlens/fitting.py:198`
computes `grad[:n, positions, :]` — a per-source-position tensor — and then calls `.mean(dim=1)`
over the valid positions. The position-resolved Jacobian is computed by the released fitter and
discarded. Slicing it instead of averaging it costs the same forward and backward passes.

## DESIGN

**Model** Pythia-410M-deduped, fp32, CPU. **Band** `[9,21]` — `floor(0.38L)..floor(0.92L)` at
L=24, the rule `tests/test_band_rule.py` asserts. **Aggregation** `min` (primary, per
`AGGREGATION_POLICY.md`); `persist` reported as labelled secondary.

**The ground truth.** For a probe position `p` and band layer `l`, define

    G_l(p) = E_{s ~ R} [ the position-p slice of the released estimator at layer l ]

where `R` is the set of prefixed evaluation sequences `[BOS] ++ [128 prefix tokens] ++ [item]`, the
exact shape the battery is read on. `G_l(p)` is the transport the lens *would need* at position `p`.
It is a property of the model and the read distribution, and is **independent of which corpus any
operator was fitted on** — which is what makes it a usable reference for all eight arms at once.

**Probe grid**, chosen from CV2's measured distribution before any CV8 number exists:

| class | positions | why |
|---|---|---|
| in-support | 16, 24, 32, 48, 64, 80, 96, 112, 126 | the fitter's own range, 16..126 |
| out-of-support | 133, 142, 150, 160, 177 | the measured prefixed read range; **142 is the median read position** |

Sequence length 192, so position 177 exists. `p_read := 142`.

**The eight arms.** The 8-corpus panel at 410M, `N=200`, seed 0, from `results/e48/`:
`INSTREAM_{Github, Pile-CC, StackExchange, USPTO_Backgrounds, Wikipedia_en}`,
`OOD_{arXiv_2023, CommonPile, News_2024}`. These are the same eight `r9_permutation_calibrated_min`
adjudicates, so the read scores are already stored and are **not recomputed here**.

**The statistic.** Alignment is scale-free by construction, because the estimator's magnitude falls
with `p` purely from summing over fewer later targets (measured in the smoke: `||J(p)||_F` 188 → 47
at 70M). Any magnitude-sensitive metric would report that artifact as a finding.

    A_l(c, p) = <J^c_l , G_l(p)>_F / ( ||J^c_l||_F * ||G_l(p)||_F )
    Abar(c, p) = mean over band layers l of A_l(c, p)

## PRIMARY

**Spearman rho between `Abar(c, p_read)` and the stored read score `real(c)`, over the eight
corpora.** If alignment-at-the-read-position orders the corpora the way the read orders them, the
corpus effect is an extrapolation difference. If it does not, extrapolation is not the mechanism.

Read scores are read from `results/r9_permutation_calibrated_min.json:by_aggregation.min.per_corpus.<c>.real`
and are fixed before this runs (Github 0.2405, Wikipedia_en 0.2416, OOD_News_2024 0.2501,
OOD_CommonPile 0.2529, USPTO_Backgrounds 0.2611, Pile-CC 0.2624, StackExchange 0.2643,
OOD_arXiv_2023 0.2693).

**Significance is calibrated, not assumed.** With n=8 the permutation null over corpus labels is
exactly enumerable (8! = 40,320), so we compute the exact two-sided p-value rather than consult a
table. **DN1 exists because this programme once set thresholds without calibrating them; this one is
calibrated in the registration.**

## DECISION RULE — fixed before running, not to be re-cut

* **EXTRAPOLATION EXPLAINS THE ORDERING** if exact two-sided `p <= 0.05`.
  The positional hedge is **confirmed**: the corpus ordering at position 142 tracks how well each
  operator aligns with the transport actually required there. The paper must retire the unqualified
  corpus claim and report the decomposition instead.
* **EXTRAPOLATION DOES NOT EXPLAIN THE ORDERING** if exact two-sided `p >= 0.20`
  **and the design had power to detect a strong relationship**, i.e. the pre-computed critical
  `|rho|` at alpha=0.05 is reported alongside and the observed `|rho|` is below it.
  The hedge is **retired**: corpora are not ordered by their alignment at the read position, so the
  corpus effect is not an alignment artifact and may be asserted as a property of the operator.
* **UNCLEAR** if `0.05 < p < 0.20`. **Report the number and stop. Do not re-cut.**

**Stated power, so a null is earned rather than defaulted to.** n=8 is a small panel and this rule
can only detect a strong monotone relationship. The exact critical `|rho|` at alpha=0.05 two-sided
for n=8 is computed and reported in the results file. A moderate alignment-read relationship would
**not** be detected by this design, and the NOT-EXPLAINED branch must be read as "no strong monotone
relationship", never as "no relationship". The same eight-corpus panel already carries a stated
minimum detectable effect of 0.0268 on the fit axis for exactly this reason.

## SECONDARY — registered, descriptive, no threshold

1. **The oracle read.** Score the battery using `G(p_read)` itself as the operator, and compare with
   the eight fitted operators. This converts CV2's "0 of 551 are in support" from a fact about
   positions into a number about reads: what the correct local transport recovers, against what the
   corpus-averaged operators recover. **No decision rule is attached**; it is a measurement.
   `G` is built on a **disjoint half** of the items from those scored (see C3).
2. **The alignment profile** `Abar(c, p)` over the whole probe grid, and the between-corpus SD of
   `Abar` in-support versus at `p_read`. Reported as a curve, not adjudicated.

## CONTROLS — each with the number it must produce

| id | control | must produce |
|---|---|---|
| **C1** | scale invariance: `A_l(c,p)` is unchanged when `G_l(p)` is multiplied by any alpha > 0. This is the control for the target-count artifact that motivated a scale-free metric. | `max abs difference == 0.0` exactly, for alpha in {0.5, 2, 100} |
| **C2** | the metric is not vacuous: in-support alignment must exceed alignment against a norm-matched Gaussian matrix | `Abar(c, p=64) >= 0.30` for all 8 corpora, and `Abar` vs random `<= 0.05`. **If C2 fails the instrument is broken and no verdict is issued.** |
| **C3** | no leakage into the oracle read: the items used to build `G` and the items scored are disjoint | `len(build_ids & score_ids) == 0` |
| **C4** | the probe position matches the real read: `p_read` must lie inside CV2's measured prefixed read range | `133 <= 142 <= 177`, and `n_in_support == 0` reproduced from `cv2_position_support.json` |
| **C5** | band is the asserted rule | band `== [9..21]`, equal to `floor(0.38*24)..floor(0.92*24)` |
| **C6** | the prefix corpus does not drive the verdict: repeat the primary with the read prefix drawn from a second corpus (Github, the extreme arm) | the PRIMARY branch must be the same under both prefixes; **if it is not, the verdict is UNCLEAR regardless of either p-value** |

## DECLARED BIAS

**We expect the NOT-EXPLAINED branch, and that expectation favours our own headline.** The reason is
prior evidence, not preference: twenty candidate predictors of the corpus ordering were screened and
none passed its own criterion, so the ordering is already known not to be recoverable from the
geometry of `J`. Alignment is another geometric quantity, so we expect it to fail too. Declaring
this because the branch we expect is the one that lets the paper assert its headline rather than
hedge it, and that is exactly the situation in which a researcher's thumb reaches the scale.

**Two guards against that.** The rule is fixed here with a calibrated threshold. And C2 can void the
experiment outright before any verdict is issued.

**A null here is a finding.** If corpora do not differ in extrapolation, that is a real,
publishable, bounded statement about the instrument, and it is the one that retires a hedge the
paper currently carries into its Discussion.

## COST

**Zero dollars. CPU only.** Measured in a tiny-first smoke on this machine before registration:
410M, 13 band layers, one 192-token sequence, keeping 14 probe positions = **~50 s**. 128 sequences
= **~1.8 h**. The alignment analysis is free. The oracle read is one battery scoring, minutes. No
GPU, no provisioning, no teardown obligation.

**OUTPUT:** `results/cv8_positional_extrapolation.json`.

---

## AMENDMENT — 2026-08-29, before the registered run, after a tiny-first smoke

A 4-sequence CPU smoke was run to exercise the exact script path before committing to the full run,
per the tiny-first discipline. It found three things. **All three are recorded here before the
registered run, and the PRIMARY statistic, its rule and its thresholds are UNCHANGED.**

**1. A fidelity bug the smoke caught, and this is why that rule exists.** The first implementation
computed `G(p)` against target layer `n_layers-1`. The stored operators are fitted with
`target_layer=-2` (`trainval.py:310`), which resolves to `n_layers-2`. Measuring alignment against a
transport the operators were never estimating would have made every number in this experiment
apples-to-oranges, and nothing downstream would have revealed it. Fixed to `n_layers-2`. This is a
code fix, not a change to the design.

**2. C1's threshold as first written is unreachable and is amended.** It required
`max abs difference == 0.0` exactly. `cos_F` sums 1,048,576 products in float64; the smoke measured
**2.184e-11**. The control's intent — that the metric cannot see the target-count artifact — is
unchanged, and the amended tolerance is `< 1e-9`, which is nine orders of magnitude below the
smallest between-corpus alignment difference the primary could turn on. **Amended threshold: `< 1e-9`.**

**3. C5's rule was quoted without its second clause.** The registration wrote the band rule as
`floor(0.38L)..floor(0.92L)`, which at L=24 gives `[9,22]`. The rule `tests/test_band_rule.py`
actually asserts is that intersected with **layers strictly below the penultimate target layer**,
which gives `[9,21]` — the band on disk and the band this experiment uses. The registration's
arithmetic was incomplete; the band was always correct. **C5 now checks the full rule.**

### DISCLOSURE — the smoke emitted a PRIMARY value and we are saying so

The smoke ran the whole pipeline, so it printed a primary result at `n_seq = 4`:
**rho = +0.524, exact two-sided p = 0.1966**, which falls in the registered UNCLEAR band.

We state that rather than leave it to be found, because amending a registration after seeing any
primary value is exactly the researcher-degree-of-freedom this programme's register exists to make
visible. Three things bound what it can have influenced:

* **The PRIMARY rule and both thresholds (0.05, 0.20) are untouched by this amendment.** Only C1's
  floating-point tolerance and C5's transcription changed, and neither can move `rho`.
* `n_seq = 4` against a registered 128. Four sequences cannot estimate `G(p)`; the value is noise.
* The fidelity bug in (1) was **live** when that number was produced, so the smoke's `rho` was
  computed against the wrong target layer and does not estimate the registered quantity at all.

**The registered run proceeds unchanged at `n_seq = 128`.** If it lands in the UNCLEAR band, that is
the verdict and it will be reported as such; the pre-registration is not re-cut to avoid it.
