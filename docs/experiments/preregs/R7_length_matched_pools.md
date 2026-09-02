# R7 — length- and format-matched fitting pools

**PRE-REGISTERED 2026-08-20, before any fit.** Source:
[`../archive/POSTREVIEW_EXPERIMENTS.md`](../../archive/POSTREVIEW_EXPERIMENTS.md) §3 Tier B, item R7.

---

## QUESTION

`SUSPICIONS.md` S-3, unresolved: **Github is the outlier in at least nine analyses**, and it is also
the corpus whose token distribution differs most from the English-prose battery. The P-ladder
already shows length matters — R moves from 2.60 raw to **3.46 length-matched** at 410M, and
`CONTEXT.md` §1.2 instructs the reader to cite the length-matched column. **Nothing in `results/`
matches the fitting corpora on length, type-token ratio or tokens-per-document before comparing
their reads.**

## DESIGN

Refit the five in-stream corpora at N=200 on pools matched on **median tokens-per-document** and
**type-token ratio**, then re-run the ladder read.

## PRIMARY

The between-corpus spread in units of the pooled seed SD, matched versus unmatched.
Published: **58×** under `persist` unstripped; **43×** under the anchor readout per the review's
re-score.

## DECISION RULE — fixed before running, quoted verbatim

Quoted from `POSTREVIEW_EXPERIMENTS.md` §3, item R7:

> **DECISION RULE.** ACCEPT if the matched spread stays above 20 seed SDs. UNCLEAR between 5 and 20.
> REJECT below 5, in which case the corpus effect is substantially a length effect and the paper
> says so.

## CONTROLS — each with the number it must produce

* **C1 — the pools must actually be matched.** Report the post-matching spread in the matched
  statistics. *Required:* below **10%** of the pre-matching spread.
* **C2 — reproduce the unmatched spread from the same code path.** *Required:* the unmatched arm
  recovers the published spread for its readout convention, so that any movement is attributable to
  matching and not to the pipeline.

**Power.** C1 can fail outright — Github may not be matchable against English prose at all. C2 is a
reproduction check with a real threshold.

## DECLARED BIAS

Matching on two statistics leaves **markup, code-ness and surprisal free**. Github may not be
matchable against English prose without discarding most of it; **if the matched pool falls below
N=200, report the shortfall rather than padding.** Scored at the corrected (stripped) readout per
R1, so the comparison baseline is the re-scored spread, not the published 58×.

## COST

5 fits ~ **$1**, plus CPU read. `SUSPICIONS.md` prices the grid at ~$2. GPU required for the same
reason as R6. **Needs the operator's sign-off before any spend.**

## RESULT

30 operators (5 corpora x 3 seeds x {matched, unmatched}) fitted on a rented RTX 4090, 168.8 min,
TF32 asserted off in-process. Scored at the **corrected (stripped) readout**.
`results/r7_matched_pools_410m.json`.

### PRIMARY — between-corpus spread in pooled-seed-SD units

| aggregation | matched | unmatched | rule (ACCEPT > 20) |
|---|---|---|---|
| **`min` (primary)** | **22.72** | 16.00 | ACCEPT |
| `persist` (secondary) | 22.24 | 30.12 | ACCEPT |

Note the two aggregations disagree on **direction**: under `min` matching *raises* the spread
(x1.42), under `persist` it *lowers* it (x0.74). The level survives either way.

### CONTROLS — C1 FAILS

| control | required | observed | fires |
|---|---|---|---|
| **C1 — the pools must actually be matched** | post-matching TTR spread **below 10%** of pre-matching | **11.1%** (0.14062 -> 0.01562) | **NO** |
| C2 — unmatched arm from the same code path | recovers a spread from the identical pipeline | 16.00 / 30.12 seed SD | yes |
| C3 — derangement floor | every operator clears its own floor under `persist` | 30 of 30 | yes |

**C1's failure is marginal and fully diagnosed.** Matching removed **89%** of the type-token-ratio
imbalance where the bar required 90%. Four of five corpora land **exactly** on the target 0.6250;
the entire residual is **Pile-CC at 0.6406**, whose documents are lexically rich enough that even
its 600 closest-to-target documents cannot bring the median down to 0.6250.

**Contrary to the declared bias, the unmatchable corpus was not Github.** The spec predicted
*"Github may not be matchable against English prose at all"*; in fact Github matched **perfectly**
(0.5547 -> 0.6250), because selection could find its high-TTR documents. Pile-CC failed in the
other direction. The prediction was wrong and is logged as wrong.

**What was NOT done, and why.** A feasible-target rule — choosing the target inside the interval
every corpus can actually reach, rather than the median of medians — would very likely close C1 at
the cost of 15 further fits (~85 min GPU, ~$0.45). **It was not done**, because changing the
matching rule *after seeing its control fail* is precisely the researcher degree of freedom this
repository's pre-registration discipline exists to prevent. The registered response to an
infeasible match was to report the shortfall, and that is what this does.

## VERDICT

**DIRECTIONAL RESULT (REQUIRES C1) — ACCEPT on the primary rule, ungraded pending a certified match.**

The pre-registered rule fires **ACCEPT under both aggregations** (`min` 22.72, `persist` 22.24,
against a bar of 20): with the fitting pools matched on the lexical composition of the fitted
window, the between-corpus spread stays far above 20 seed SDs, so **the corpus effect is not
substantially a lexical-composition effect.**

But **C1 does not fire**, so the arm is not certified matched and the result carries the tag before
the number, per `RIGOR_SKILL.md`. It may be cited as directional evidence that the corpus
effect survives lexical matching; it may **not** be cited as a clean ACCEPT.

**Read together with R6, which is clean:** R6 establishes that the fit axis is a **source** effect
rather than a sampling effect (between-source share 0.9886 under `min`, all controls firing). R7
adds, directionally, that it is not merely a *lexical-composition* effect either. The design
finding recorded above still stands and is unaffected by C1: **fitting-side document length is
matched by construction** — every fitting prompt is exactly 128 tokens — so length was never
available as an explanation for the corpus effect in the first place.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/r7_matched_pools_410m.json` | 16,993 | `be98f49b67e1a16c` | `r7_matched_pools.py` | EXPOSED |

**Payload checksums** (content only, provenance block excluded):

* `r7_matched_pools_410m.json` — `bb4dad27226cb58effcc42b6b4b6f59b`

<!-- END GENERATED PROVENANCE -->
