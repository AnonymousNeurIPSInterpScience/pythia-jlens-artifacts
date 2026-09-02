# DA1 — adjudicating what the layer-derangement control identifies

**Written before any DA1 output exists.** Pure re-scoring on CPU from operators already fitted and
already on the artifact mirror. **No refit. No new corpus. No new model. No change to the 8x8
factorial.**

## DISCLOSURE — the ordering, stated up front

This document is written **after** `R9` returned "0 of 8 operators clear all fifteen of their own
random-derangement draws under `min`, median z = -0.71" and after `D1` reported the union
diagnostic (`results/d1_min_union_diagnostic_410m.json`). It is written **because an external audit
argued that the random derangement is not an identifying null for layer correspondence**, since it
removes two things at once:

1. **absolute layer correspondence** — band layer `l` no longer receives its own `J_l`; and
2. **cross-layer dependence** — the per-layer readouts of a random derangement are less correlated
   with each other than the real operator's are.

`min` is an existential union over the band. A union grows as its events decorrelate. So a null
that decorrelates the layers changes `min` through channel (2) whether or not channel (1) matters
at all, and the observed direction cannot be attributed to correspondence.

**This plan cannot rescue the old claim and is not written to.** `D1` already reported the union
mechanism in this repository's own words; what has never been done is to run a null that separates
the two channels. DA1 does that. The registered outcome space in §5 includes withdrawal.

## WHY THIS IS NOT A NEW RESEARCH DIRECTION

Every quantity below is scored from operators already on disk, with the battery, band, admitted
sets, activation cache and scoring function taken unchanged from `t48_crossover.py --rstrip`, the
run that produced the corrected numbers `R9` adjudicates. The only new objects are **permutations
of the band index** applied to an already-fitted operator, and **summaries of an already-computed
rank matrix**. Nothing is refitted and no new evaluation definition is invented.

---

## 1. THE THREE QUESTIONS, FIXED BEFORE ANY DA1 NUMBER IS SEEN

### Q-A1 — does a more structure-preserving disruption of absolute layer correspondence move `min`?

A **cyclic shift** `J_l -> J_{l+k mod |B|}` removes absolute correspondence — after any non-zero
shift, no band layer receives its own Jacobian — while preserving the *ordering* of the Jacobian
sequence, so adjacent band layers continue to receive adjacent Jacobians for all but one wrap
point. A random derangement destroys the ordering as well.

**Q-A1: with correspondence removed but ordering largely preserved, does `min` still move, and in
which direction?**

### Q-A2 — is the derangement's apparent advantage specific to an existential aggregation?

**Q-A2: comparing the same real operators against the same fifteen random derangements, does the
derangement's advantage under `min` persist under summaries that are not unions over the band?**

### Q-A3 — how much does each null actually change cross-layer dependence?

Neither the cyclic shift nor the derangement is assumed to preserve dependence. **Q-A3: for the
real, cyclic-shifted and randomly deranged sequences, what is the cross-layer dependence of the
per-layer recovery events that the aggregation consumes?**

The dependence statistic must be computed on **the recovery events themselves**, not on Jacobian
cosine similarity. The inferential problem is about the correlation of the per-layer indicator
`1[rank_l <= k]` across `l`, because that is what a union over the band consumes. Jacobian
similarity is a different quantity and cannot answer this.

---

## 2. DESIGN

### 2.1 Everything held fixed

Taken unchanged from `experiments/t48_crossover.py` run with `--rstrip`:

| held fixed | value |
|---|---|
| model | `EleutherAI/pythia-410m-deduped`, fp32, TF32 off |
| readout | **corrected**: `item["prompt"].rstrip()`, then the per-set position rule |
| band `B` | `[9..21]`, 13 layers |
| `K` | `1, 2, 5, 10, 20, 50, 100`; flat mean over the seven |
| admitted sets | multihop, multilingual, order-ops, poetry, typo (`association` floored, excluded) |
| operators | the same 8 corpora x 3 seed blocks at `N=200` already used by E48/R9 |
| activation cache | one cache, computed once, shared by every arm |
| scoring | `score()` transcribed from `t48_crossover.py`; 1-indexed strict-greater rank over the full vocabulary, min over the token ids of a multi-token intermediate |

### 2.2 The arms

For each of the 24 operators (8 corpora x 3 seeds), the band rank matrix is computed for **every**
of the 13 x 13 (activation-band-slot, Jacobian-layer) combinations. Every permutation arm below is
then a lookup into that tensor, so all arms are scored from one pass and are **exactly** comparable
by construction — no arm can differ from another by anything except the permutation.

| arm | permutation of the band |
|---|---|
| `real` | identity |
| `cyc{k}` for k = 1..12 | `l -> B[(i+k) mod 13]` where `i` is `l`'s index in `B` |
| `shuf{d}` for d = 0..4 | `derangement(BAND, 7000 + 97*d)`, transcribed from `t48_crossover.py` so the objects are the **same five permutations** E48 used |
| `logit` | no transport (identity transport, not identity permutation) |

`shuf` therefore reproduces E48's fifteen draws per corpus as 3 seeds x 5 permutations. **The
fifteen draws are five distinct permutations applied to three operators, not fifteen distinct
permutations**; that is a property of the E48 design being adjudicated and is recorded here rather
than corrected, because changing it would stop this being an adjudication of R9.

### 2.3 The four summaries

All four consume the same `[n_pairs, 13]` band rank matrix `R`. The two existing ones are
transcribed; the two new-to-this-file ones take their **semantics** from
`experiments/t17_reaggregate.py:aggregate()` and their **k-summary and set-averaging** from
`t48_crossover.py`, so that `min` reproduces the stored corrected value exactly.

Let `v_l` be the per-pair vector `mean_k 1[R[:,l] <= k]`, and let `A(v)` be
`mean over the 5 admitted sets of (mean over that set's pairs of v)`.

| summary | definition | union over the band? |
|---|---|---|
| `min` | `A( mean_k 1[min_l R[:,l] <= k] )` | **yes** — existential |
| `persist` | `A( mean_k 1[ sum_l 1[R[:,l] <= k] >= 6 ] )` | no — occupancy count |
| `best1L` | `max_l A(v_l)` | **no** — one layer, chosen post hoc across layers |
| `mean` | `mean_l A(v_l)` | **no** — average per-layer quality |

**Declared deviation from `t17_reaggregate.py`:** t17 computes `min`, `best1L` and `mean` over
**all** layers and only `persist` over the band, and uses `KS=(1,2,5,10,25,100)` with a log-k
trapezoid AUC. DA1 computes **all four over the band** under `K=(1,2,5,10,20,50,100)` with a flat
mean, because the claim under adjudication (`R9`, paper §4.4) is band-scoped and stated under that
k-summary. This is a change of scope, not of semantics: `best1L` is still "best single layer" and
`mean` is still "average per-layer quality". Recorded here before running.

**`best1L` is not a neutral statistic and is not treated as one.** Taking a maximum over 13 layers
is itself a selection, so `best1L` is reported as a diagnostic of where per-layer quality sits, not
as a corrected recovery metric.

### 2.4 The dependence statistic (Q-A3)

For a permutation arm, and for each `k in K`, form the binary matrix `H_k[p,l] = 1[R[p,l] <= k]`.
Report:

* **`rho_bar(k)`** — the mean off-diagonal Pearson correlation over the 78 band layer pairs of the
  columns of `H_k`. This is the dependence a union consumes: it is computed on the recovery events
  themselves, over the same pairs the score averages over.
* **`union_gap(k)`** — `observed union - independence prediction`, where the observed union is
  `mean_p max_l H_k[p,l]` and the independence prediction is `1 - prod_l (1 - mean_p H_k[p,l])`.
  A sequence whose layers are redundant sits **below** its independence prediction; a decorrelated
  sequence sits near it. This is the same construction as `D1`'s H3 clause and is named so the two
  are comparable.

Both are reported per `k` and averaged over `k`, per arm, per operator. Layers with zero variance
in `H_k` (no pair recovered at that layer and k) contribute no pair to `rho_bar`; the count of such
degenerate pairs is reported alongside, because a correlation over a constant column is undefined
rather than zero.

### 2.5 A4 — band-constant diagnostic, explicitly exploratory

Every band layer receives `J_bar = mean_{l in B} J_l`. This is **not** a matched control: `J_bar`
is a different object with a different norm and spectrum, not a permutation of the existing ones,
so it changes more than the arms in §2.2. It is reported only to answer the descriptive question
*"does layer-specific variation carry anything beyond one average transport over the band?"* and
carries no verdict. It runs only after §2.2–2.4 are complete.

---

## 3. GATE — DA1 IS VOID IF THIS FAILS

The recomputed `min` and `persist` admitted means for every `real` and every `shuf` arm must match
the corresponding `arms_admitted_mean` entry in `results/e48_crossover_410m_rstrip.json` to
**< 1e-6**.

That tolerance is `t48_crossover.py`'s own, derived not tuned: one pair flipping at one k moves a
set mean by at least `1/(394*7) = 3.6e-4`, so 1e-6 sits far below the smallest change a genuine
scoring difference can produce and above float32 accumulation noise.

If the gate does not fire, DA1 reports the disagreement and stops. It does not adopt its own
numbers over the stored ones.

Additional controls, each with the value that makes it fail:

| control | requirement | fails if |
|---|---|---|
| C1 | the identity permutation must reproduce the `J\|c\|s` arm exactly | any non-zero difference |
| C2 | every non-zero cyclic shift must move at least one band layer off its own Jacobian | any `k` with a fixed point (impossible for `k != 0`; the assertion catches an indexing error) |
| C3 | `logit` arm must equal the stored `logit_I` arm to < 1e-6 | otherwise the activation cache differs from E48's |
| C4 | the five `shuf` permutations must equal `t48_crossover.derangement(BAND, 7000+97d)` element-for-element | any mismatch means DA1 is not scoring E48's objects |

---

## 4. WHAT IS REPORTED

Descriptive, paired, per operator and per arm. **No new pass/fail threshold is created**, because
none is justified before seeing the numbers: DA1 exists to establish what the existing control
identifies, and inventing a bar for the replacement control would repeat the error it is
diagnosing. Specifically reported:

* per-corpus and per-(corpus, seed) values of all four summaries for `real`, each `cyc{k}`, each
  `shuf{d}`;
* per-shift and shift-aggregated `real - cyclic` gaps, so no single favourable `k` can be selected;
* the paired `real - shuf` gap under all four summaries, on the same 8 x 15 grid as R9;
* `rho_bar` and `union_gap` for `real`, cyclic and deranged sequences;
* the count of (corpus, arm) cells in each direction, alongside the gaps, since a count over 15
  draws floors an empirical p at 1/16 and is not an inferential statement (`DN1`).

Where the four summaries disagree, the disagreement is the result and is reported as such.

---

## 5. THE OUTCOME SPACE, FIXED IN ADVANCE

All five are reportable outcomes. None is preferred.

1. **Real beats cyclic shifts under `min`.** Correspondence may matter under a more
   structure-preserving null. The statement then permitted is bounded by what a cyclic shift
   actually removes, which §2.4 measures rather than assumes.
2. **Cyclic shifts match or beat real under `min`.** The correspondence-certification reading of
   R9 is withdrawn: neither null supports it.
3. **Random derangement differs under `min` but not under `mean` / `best1L`.** Then the
   derangement interacts with band aggregation, is not an identifying null for correspondence, and
   R9's direction is a fact about `min`'s union and not about the operator.
4. **The derangement's advantage survives every summary.** Then the union explanation of `D1` is
   wrong and that must be said, whatever it does to §4.4.
5. **Mixed.** Reported mixed, per corpus and per summary. A mixed result is not rewritten into a
   PASS or a FAIL.

## 6. DECLARED BIAS

This plan follows an audit which already argued that the random derangement is confounded, so it is
written from a position that predicts outcome 3. Outcomes 1 and 4 both contradict that position and
are reportable exactly as written; the gate in §3 and the controls C1–C4 are what stop a preferred
outcome from being produced by a scoring difference rather than by the data.

## 7. COST

CPU only. 24 operators x 169 (slot, Jacobian) combinations, one unembed pass each, plus one
band-constant pass per operator. No GPU, no fitting, no network beyond fetching operators already
published on the artifact mirror. Estimated single-digit hours on a laptop; no paid compute.
