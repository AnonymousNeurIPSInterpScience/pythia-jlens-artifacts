# GATE 1 — ARE THE EXPERIMENTS CORRECT?

**The question:** *assuming the scripts faithfully implement the methodology and every result
reproduces exactly — are the experiments themselves the right experiments, and do the results mean
what we say they mean?*

This gate takes implementation fidelity as given. That assumption is tested separately and
adversarially in [Gate 2](../reproducibility/). Holding it fixed here is deliberate: it isolates
design error from implementation error, which otherwise mask each other.

**A null is a finding.** Nothing in this gate is looking for a subthesis to survive. It asks what
each experiment pins down, how tightly, and what it cannot resolve. A design that returns a tight
null has passed.

**A null must be earned.** The most consequential finding of the 2026-08-22 audit was not a wrong
number — it was that an apparent null was underpowered. See §"what this gate catches" below.

---

## WHAT THIS GATE CATCHES

Design errors survive perfect implementation and perfect reproducibility. Every item below is real
and was found here, in this repository, on numbers that recompute exactly.

| failure mode | worked example found here |
|---|---|
| **wrong replication unit** | the fit/read variance decomposition's bootstrap resamples seeds while holding the eight corpora fixed — but the claim is about corpora |
| **underpowered null reported as absence** | S3's membership contrast has a minimum detectable effect of **0.0268** at 80% power on the fit axis, which *exceeds* the entire between-corpus spread of 0.0246. The honest statement is a bound of ±0.02, not "exposure does not matter" |
| **decision statistic with a degenerate denominator** | E37's `max(gap(1),gap(2)) / gap(full)` divides by +0.0055 at 70M (inside the noise floor) and −0.0358 at 160M (negative). The stored 70M REJECT is a small-denominator artifact; the 160M ACCEPT is a sign artifact. E37 is **silent, not negative**, about the mechanism at small scale |
| **a result that exists only under one free choice** | the fit>read ordering replicates 5/5 under `min` and 0/5 under `persist` on an independent operator population; dropping one corpus reverses it under `min` |
| **an aggregation that collapses** | at 70M, `persist` is identical to `min` by construction (band `[2,3]`, threshold 1) — 38/38 cells bit-identical. "Survives both aggregations" at 70M has the weight of one |
| **a control that could not have fired** | E33 v1 used a cyclic shift by one layer as its derangement; adjacent-layer Jacobians are the most similar pair, so the control was near-vacuous. v2 amended it to a random derangement and the sign of the `best1L` result flipped |
| **a statistic that does not measure the intended quantity** | the concept-row geometry is computed on un-centred `W_U` rows, so a shared row direction inflates it — and a shared direction is rank-irrelevant, because softmax is shift-invariant |
| **stale transcription** | numbers computed before a correction, still printed. Not wrong-headed; out of date. Fixed by recomputing |

---

## START HERE — [`DIAGNOSIS.md`](DIAGNOSIS.md)

The current state of the open construct-validity problem: why absolute readout numbers are low,
which of the candidate causes are measured, what survives and how it must be relabelled, and the
ordered work queue. Read it before the audit below.

## THE CURRENT AUDIT — 2026-08-22

[`audit-2026-08-22/`](audit-2026-08-22/)

| file | holds |
|---|---|
| **`SCORECARD.md` ** | **start here.** What is quantified, per-subthesis scores on five axes, the currency edit list, the ranked experiment slate |
| `extract.py` | regenerates `DIGEST.json` / `DIGEST.md` from `results/`. Re-runnable — every derived number can be recomputed rather than trusted |
| `DIGEST.json` / `DIGEST.md` | the extracted evidence base. Sections tagged STORED are verbatim; DERIVED are computed in `extract.py` |
| `VERIFY_EXTRACTION.md` | independent recomputation of every DERIVED section by a different route, plus the power analysis |
| `S1_ANALYSIS.md` | the scale floor and the rank-ablation adjudication |
| `CURRENCY.md` | 48-row document-currency table with replacement sentences |
| `SPECS_AND_THESIS.md` | spec scoring, the three open rulings, the recommended spine |

**Method.** Extraction was run centrally and deterministically into `DIGEST.json`; judgement was then
delegated to four independent auditors each given the digest rather than the repository. Every
finding was checked against the underlying file before being recorded. Three of the auditors
corrected the extraction; one fabricated a line citation that did not survive checking, which is why
nothing here is recorded without a verified file-and-key reference.

---

## HOW TO RUN THIS GATE

1. **Regenerate the evidence base.** `.venv/bin/python docs/validity/audit-2026-08-22/extract.py`
   (or a fresh dated copy). Never audit against prose — audit against the digest, and audit the
   digest against the files.
2. **For every headline, name the replication unit.** How many independent operators, corpora,
   seeds. A hierarchical bootstrap over eval sets and items does not create independent corpora.
3. **For every null, state the minimum detectable effect.** If the design could not have detected an
   effect the size of the thing it is being compared to, it is a bound, not an absence.
4. **For every decision rule, check its denominator.** A ratio rule whose denominator can be near
   zero or change sign is not a rule.
5. **For every claim, vary the free choices.** Aggregation (`min` / `persist` / `best1L` / `mean`),
   readout (stripped / unstripped), band, `N`, eval-set list, corpus panel. A claim that exists under
   one setting of a free choice is a claim about that setting.
6. **Check what a result invalidated**, then go and check those claims.

---

## EARLIER PASSES

Earlier validity passes are internal and not published; `DIAGNOSIS.md` is the current synthesis.

Superseded in specifics by the 2026-08-22 audit, retained for the reasoning and for the controls
census (36 of 106 control rows carried no gate, were never evaluated, or were unfalsifiable).
