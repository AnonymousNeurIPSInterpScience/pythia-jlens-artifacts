# DA1 — outcome, adjudicated against the pre-registered outcome space

Pre-registration: [`preregs/DA1_derangement_adjudication.md`](preregs/DA1_derangement_adjudication.md).
Results: `results/da1_derangement_adjudication_410m.json`, `results/da1b_clustered_t.json`.
Repro: `bash repro/exp/da1_derangement.sh` (T1), then `.venv/bin/python tools/da1b_clustered_t.py`.

**This record applies the classification fixed in §5 of the pre-registration. It creates no new
threshold and moves none.**

## The gate fired

290 arm/aggregation comparisons against `results/e48_crossover_410m_rstrip.json`,
**max|diff| = 0.000e+00**. DA1's `real` and `shuf` arms are bit-identical to the stored corrected
arms R9 adjudicates, so DA1 is scoring the same objects and not a re-implementation of them.
Re-running the module reproduces `payload_sha256 = fc6fcb31101aa923…` exactly. Controls C1, C2 and
C4 fire; C2 confirms all twelve cyclic shifts have zero fixed points, i.e. **every null compared
here removes absolute layer correspondence completely.**

## The result

Corpus-clustered paired t over the 8 per-corpus gaps (real − null), df = 7 — the statistic the
paper already uses for this comparison (`results/paper_clustered_derangement_t.json`, Table 13):

| null | `min` | `persist` | `best1L` | `mean` |
|---|---|---|---|---|
| **cyclic shift** (mean of 12) | **+0.0162, t=7.70, p=1.2e-4, 8/8** | +0.0269, t=15.83 | +0.0106, t=4.59, 8/8 | +0.0244, t=25.57, 8/8 |
| **random derangement** (mean of 15 draws) | **−0.0052, t=−1.92, p=0.096, 1/8** | +0.0275, t=17.50 | +0.0009, t=0.58, 4/8 | +0.0228, t=25.85, 8/8 |

Cell counts: real beats the cyclic shift on 79 of 96 (corpus × shift) cells under `min` and on
96 of 96 under `mean`; real beats the random derangement on 36 of 120 draws under `min` and on
120 of 120 under `mean`.

## The outcome is §5.3, with the §5.5 qualifier on one summary

> **§5.3 — "Random derangement differs under `min` but not under `mean` / `best1L`. Then the
> derangement interacts with band aggregation, is not an identifying null for correspondence, and
> R9's direction is a fact about `min`'s union and not about the operator."**

Held on `mean` and `persist`, both unambiguously. **Mixed on `best1L`** (§5.5): real beats the
cyclic shift there (8/8, t = 4.59) but is indistinguishable from the derangement (4/8, t = 0.58,
p = 0.58). `best1L` takes a maximum over 13 layers, which is itself a selection over the band, so
it partially inherits the inflation `min` suffers from. Reported mixed, not rewritten.

**The finding that does not depend on any mechanism.** Two nulls, both of which remove absolute
layer correspondence entirely, give **opposite verdicts under `min`**: the real operator beats the
cyclic shift on 8 of 8 corpora at t(7) = +7.70 and loses to the random derangement on 7 of 8 at
t(7) = −1.92. A statistic whose verdict reverses between two nulls that both remove the property in
question is not measuring that property. R9's direction under `min` therefore does not license a
statement about layer correspondence.

**What is licensed positively.** Under `mean` and `persist` the real operator beats **both** nulls
on 8 of 8 corpora, on all 96 cyclic cells and all 120 derangement draws. Because a cyclic shift
preserves every entry, norm and spectrum of the operator and changes only which band activation
meets which Jacobian, this is evidence that the layer-to-activation pairing carries information.
That statement is not selected on the control it must pass: `mean` beats both nulls, and its
definition predates the derangement result (`experiments/t17_reaggregate.py`).

## What the cyclic shift does and does not isolate — measured, not assumed

The pre-registration refused to assume the shift preserves dependence, and it does not:

| arm | `rho_bar` (mean off-diagonal correlation of the per-layer recovery events) | `union_gap` (observed union − independent-layer prediction) |
|---|---|---|
| real | 0.4358 | −0.3535 |
| cyclic (mean of 12) | 0.3371 | −0.2900 |
| random derangement | 0.3282 | −0.2781 |

The cyclic shift recovers only about 8% of the dependence the derangement destroys
(0.337 vs 0.328 against real's 0.436). **So the cyclic shift is not a dependence-preserving
control, and must not be described as one.** What separates the two nulls is `union_gap`: the
derangement sits closest to the independent-layer prediction and therefore extracts the largest
union bonus, which is the axis `min` is sensitive to and `mean` is not.

Within the cyclic family, Spearman(`rho_bar`, median real−shift under `min`) = **−0.797** across the
twelve shifts: the more dependence a shift preserves, the less `min` separates it. **This is a
description, not an identification.** Shift magnitude covaries with both dependence preservation
and per-layer quality preservation — a shift of 1 gives each slot an adjacent layer's Jacobian, so
it degrades both less — and DA1 does not separate those two.

## A4 — band-constant, exploratory only

Every band layer given `J_bar`, the band-average operator. **Not a matched control**: `J_bar` is a
different object with a different norm and spectrum, not a permutation.

| | `min` | `persist` | `best1L` | `mean` |
|---|---|---|---|---|
| real − band-constant | +0.0212, 8/8 | −0.0001, 4/8 | −0.0085, 1/8 | +0.0043, 7/8 |

Absolute: band-constant `persist` = 0.1080 against real 0.1078 and the free lens 0.0832. Under
`persist` **one average transport over the band matches the layer-specific operator**, while both
beat every permuted null. That is consistent with the band's Jacobians being similar enough that
their average substitutes for any one of them, whereas a specific *mismatched* one does not. It is
an open question, carries no verdict, and is not evidence about correspondence.

## What this does not touch

The corpus-sensitivity result. DA1 refits nothing, changes no corpus, and re-scores only
permutations of operators already fitted. `results/e48_crossover_410m_rstrip.json` is unmodified.
