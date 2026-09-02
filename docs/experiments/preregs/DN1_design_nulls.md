# DN1 — the null distributions of three bars this programme already registered

**Pre-registered 2026-08-24, before any null number exists.** Pure arithmetic on stored values and
on the design. **No model, no GPU, no fitting, no scoring.** Runs in seconds on CPU.

## DISCLOSURE — the ordering

This document is written **after** CV6 and CV7 returned REPLICATES and after `R9` returned 0 of 8.
It does not test any new substantive hypothesis and **cannot overturn any of them**. It computes
what those already-registered bars correspond to under their own nulls. The thresholds being
calibrated (`R >= 10`, `tau >= 0.6` on 3 of 5, "clears all 15 own draws") were all fixed before the
numbers they judge; nothing here moves them.

## WHY

Three decision rules in this programme are stated as bare numbers with no null underneath them.

1. **`R >= 10`** (CV6, CV7). `R = spread_z / pooled_seed_SD_z` over 5 fitting corpora with 3 seed
   blocks each. Ten is a large-looking ratio, but nobody has computed what `R` a design with no
   corpus effect produces. Without that, `R >= 10` is a number, not a threshold, and a reviewer is
   entitled to ask whether 10 is two null SDs or twenty.
2. **`tau >= 0.6` on 3 of 5 families** (CV6, CV7). Kendall's tau on `n = 5` is discrete, taking
   eleven values from `-1.0` to `+1.0` in steps of `0.2`. *(Corrected after the run: this line
   originally said "six values", which is wrong. The error is in the motivation, not in the design
   or the rule, and Rule B enumerates the values rather than assuming them.)* The per-family null
   probability of `tau >= 0.6` has never been stated, so "3 of 5 families agree" has never been
   priced.
3. **"no operator clears all 15 of its own derangement draws", 0 of 8** (R9). Under exchangeability
   of a real score with its own 15 draws the probability that it is strictly largest is exactly
   `1/16`. The expected number of clearing corpora under the null is therefore `0.5`, and a count of
   0 may be the *modal outcome under no effect at all*. `docs/context/STATE.md` and
   `paper/tables_audit.tex` both call this count "the inferential statement". If the count is
   uninformative, that sentence is wrong and the informative statistic is elsewhere in the same file.

None of the three is expensive. All three have been skipped because the direction they are expected
to point is favourable.

## DESIGN

Three independent calculations, no shared state.

**A. The `R` null, two ways.**

*A1, design null by simulation.* `R = range_g(xbar_g) / sqrt(mean_g(s_g^2))` for `G = 5` groups of
`n = 3` iid normal draws, where `s_g` is the sample SD of group `g`. Both numerator and denominator
are homogeneous of degree one in the scale, so the null distribution of `R` is free of `sigma` and of
`mu`, and one simulation calibrates every family at every rung. `10^6` draws, `numpy` default
generator seeded at 0.

*A2, exact permutation null on stored data.* `results/d3_corpus_by_family_410m.json` stores the raw
per-arm `z` at `per_arm.<corpus>|s<seed>.<family>.z` — 15 values per family at 410M. Permute the 15
values across the 5 corpus labels, 3 per label, recompute `R`. `10^5` sampled assignments per
family, seeded. This is the standard one-way permutation null and makes no distributional
assumption; A1 without it is an assumption, A2 without A1 does not generalise to the rungs whose
raw values are not stored.

**B. The `tau` null.** Enumerate all `5! = 120` orderings, compute Kendall's tau against a fixed
reference, and count. Exact, not simulated. Then compute the distribution of "number of families out
of 5 with `tau >= 0.6`" under independence, and state plainly that the five families share five
operators and one activation cache, so independence is an upper bound on the evidence.

**C. The R9 count null.** Exact: `P(count = 0 | H0) = (15/16)^8`. Then the same file's per-corpus `z`
gives a statistic that is *not* degenerate under the null: the sign of `z_vs_null` across the eight
corpora, with an exact binomial sign test.

## PRIMARY

The one-sided null exceedance probability of the registered `R >= 10` bar, under A1, cross-checked
against A2.

## DECISION RULE — fixed here, before the run

* **CALIBRATED** — the `R >= 10` bar sits at or above the 99th percentile of the A1 null. Passing it
  by chance is a sub-1% event and the CV6/CV7 verdicts stand as written.
* **WEAK** — the bar sits between the 95th and 99th percentiles. The verdicts stand but every
  statement of them must quote the null exceedance.
* **UNCALIBRATED** — the bar sits below the 95th percentile. The registered rule admits chance
  passes; CV6 and CV7 must be restated with the null alongside, and no new rule may reuse `10`.

Rule B and Rule C are **descriptive and carry no verdict**: they report exact probabilities. They
cannot fail. They are here because the numbers they produce are load-bearing for prose that already
exists, and that prose is wrong if the numbers come out a particular way.

## CONTROLS — each with the number it must produce

* **C1 scale-freedom.** A1 run at `sigma = 1` and at `sigma = 100` must agree on the 99th percentile
  of `R` to within `0.02` absolute. A gap means `R` is not scale-free and A1 does not transfer
  across rungs. *A control that can fail: the same code with the denominator replaced by a constant
  `1.0` must break it.*
* **C2 recompute the denominator.** For all 5 families, the pooled seed SD recomputed from D3's raw
  per-arm `z` must equal the stored `by_family.<fam>.z_pooled_seed_sd` to `<= 1e-12`. If it does
  not, DN1 is calibrating a different quantity from the one the bar used and every number here is
  void.
* **C3 recompute the statistic.** For all 5 families, `R` recomputed from raw must equal the stored
  `by_family.<fam>.z_spread_over_sd` to `<= 1e-12`.
* **C4 permutation null contains the identity.** For each family, the unpermuted assignment must
  appear in the permutation null's support and reproduce the observed `R` exactly. A null that
  cannot generate the observed statistic is not a null for it.
* **C5 A1 against A2.** The 99th percentile of the design null and of the permutation null must
  agree to within a factor of `1.25`. A larger gap means normality is doing real work and only A2
  may be quoted.

## DECLARED BIAS

**This registration expects this to come out favourably for the programme on A and unfavourably on C.** The `R` bar
is almost certainly far above its null: the null `R` for 5 groups of 3 should sit near 2, so `10` is
likely a wide margin and CV6/CV7 get *stronger*, not weaker. The R9 count is almost certainly
degenerate: `(15/16)^8` is about `0.6`, so "0 of 8" is probably the modal null outcome and the
paper's current framing of it is probably wrong.

Recording the expectation in both directions is the point. A calibration run only when the answer is
feared is not a calibration.

## COST

Seconds. CPU. No model weights, no lenses, no corpora, no network.

## OUTPUT

`results/dn1_design_nulls.json`, stamped by `tools/build_provenance.py` like every other result.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/d3_corpus_by_family_410m.json` | 19,603 | `b4342917486e9329` | `d3_corpus_by_family.py` | CLEAN |
| `results/dn1_design_nulls.json` | 10,329 | `bf771bd0ce2e5313` | `dn1_design_nulls.py` | UNCLASSIFIED |

**Payload checksums** (content only, provenance block excluded):

* `d3_corpus_by_family_410m.json` — `fc353bf005b4e4ab1bf00c84bc40013d`
* `dn1_design_nulls.json` — `201a8d6db3dfd8bcb12d81b23aa253b0`

<!-- END GENERATED PROVENANCE -->
