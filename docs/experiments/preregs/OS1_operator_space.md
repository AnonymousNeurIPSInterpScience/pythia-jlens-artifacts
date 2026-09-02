# OS1 — does the corpus move the operator, or only the score?

**Pre-registered 2026-08-24, before any operator-space distance exists.** Recompute from stored
`.pt` operators. **No model forward pass, no GPU, no scoring, no battery.** CPU, minutes.

## WHY

Every corpus-dependence number this programme has is a **score** on the released battery. Sections
5.3 of the paper establishes that the battery does not identify its construct: Pythia-410M answers
4.7% of it, and 0 of 551 prefixed reads land inside the operator's positional support. A reviewer
is then entitled to the objection that decides the paper:

> A conditional effect measured through an instrument you have shown to be invalid is not a result
> about the instrument. It is a well-controlled measurement of a quantity you cannot name.

The current answer is that every cell shares item, target, model, read position, aggregation and
seeds, so the comparison is internally valid. That establishes internal validity and does not
answer the objection.

**The operator is not the battery.** `J_\ell` is a `d x d` matrix estimated from a corpus, and
whether the corpus moves it is a question with no battery, no read position, no competence
assumption and no aggregation statistic in it. If the between-corpus displacement of `J` is large
against the displacement produced by resampling documents from the same corpus, the claim
"fitting-corpus identity changes the estimated operator" stands on its own, and the battery result
becomes a consequence rather than the evidence.

If it is NOT large, that is the more important finding: it would mean the corpus effect on the score
is produced somewhere other than the operator, and Section 5.1 would need restating.

## DESIGN

**Operators.** 24 stored fp16 operators at `pythia-410m-deduped`, `N=200`, band `[9,21]`, one per
(corpus, seed block): 5 in-stream (`results/e48/lens_INSTREAM_<c>_410m_n200_s<0,1,2>.pt`) and 3
measured-absent (`results/e48/lens_<ood>_410m_n200_s<0,1,2>.pt`). Plus 8 within-source refits
(`results/r6/lens_R6_<Pile-CC|Wikipedia_en>_b<0..3>_410m_n200.pt`), which are R6's four disjoint
document blocks drawn from ONE source.

**The three distances**, per band layer, then averaged over the 13 layers.

* `d_rel(A,B) = ||A - B||_F / sqrt(||A||_F ||B||_F)` — scale-aware relative displacement.
* `theta_k(A,B)` — mean principal angle between the top-`k` left singular subspaces, `k=32`. This
  is scale-FREE and direction-only, so it separates "the corpus changed how much J transports" from
  "the corpus changed what J transports".

**The three contrasts.**

* **BETWEEN**: different corpus, matched seed index. 28 unordered corpus pairs x 3 seeds.
* **WITHIN-SEED**: same corpus, different seed block. 8 corpora x 3 pairs.
* **WITHIN-SOURCE**: same source, different disjoint document block (R6). 2 sources x 6 pairs.
  This is the strictest within baseline, because the blocks share a source but no documents.

## PRIMARY

`SEP = median(BETWEEN d_rel) / median(WITHIN-SOURCE d_rel)`, the separation ratio, with a
corpus-clustered bootstrap interval (resample corpora, not pairs, 10,000 draws).

## DECISION RULE — fixed here, before the run

* **OPERATOR EFFECT** — `SEP >= 2.0` and the bootstrap interval's lower bound exceeds `1.0`, under
  BOTH `d_rel` and `theta_32`. The corpus moves the operator itself, the claim stands without the
  battery, and the paper states it that way.
* **SCORE-ONLY** — the interval on `SEP` includes `1.0` under either metric. The corpus does not
  measurably move `J` at this resolution, the score effect is produced downstream of the operator,
  and Section 5.1 must be restated as a claim about the readout and not the transport. **Stop and
  flag for adjudication.**
* **UNCLEAR** — `SEP` between `1.0` and `2.0` with an interval excluding `1.0`, or the two metrics
  disagree in branch. Report the number, change nothing. **Stop and flag for adjudication.**

## CONTROLS — each with the number it must produce

* **C1 self-distance is zero.** `d_rel(A,A) = 0` and `theta_k(A,A) = 0` exactly, for all 24. A
  non-zero value means the loader or the metric is wrong and nothing else in the file is readable.
* **C2 the metric can separate.** A norm-matched Gaussian random operator must sit ABOVE every
  BETWEEN pair on both metrics. If a random matrix is not further away than another corpus, the
  metric has no resolution and `SEP` is uninterpretable. Required: `d_rel(random, real) >` max
  BETWEEN `d_rel`.
* **C3 fp16 storage is not the effect.** Round-trip one operator fp32 -> fp16 -> fp32 and measure
  `d_rel`. It must be at least 10x below the WITHIN-SOURCE median, or storage precision is
  competitive with the signal.
* **C4 the band is the paper's band.** Every loaded operator must carry `source_layers == [9..21]`.
  A mismatch means these are not the operators the paper's numbers came from.
* **C5 layer homogeneity, reported not gated.** Report `SEP` per band layer. If it is carried by
  one or two layers the summary average is hiding structure, and that is stated.

## DECLARED BIAS

**This registration expects OPERATOR EFFECT, and this is being run because This registration expects it.** The score spread is 19 to
43 pooled seed SD, so a null on the operator would be surprising. That is precisely the condition
under which a check gets skipped. Recording the expectation is what makes the SCORE-ONLY branch
survivable: if it fires, it fires against a stated prior and it is the most useful result in the
programme, because it would relocate the entire effect.

Second declared bias: `theta_32` is scale-free and `d_rel` is not, so the two can disagree, with
`d_rel` large and `theta_32` small if corpora change only the gain of the transport. That
disagreement is a REAL outcome with its own reading and is routed to UNCLEAR rather than resolved
by picking the metric that answers.

## COST

CPU, minutes. 832 MiB of stored operators already pulled. No forward pass, no GPU, no spend.

## OUTPUT

`results/os1_operator_space_410m.json`, stamped like every other result.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/os1_operator_space_410m.json` | 5,547 | `ea08c583748f9020` | `os1_operator_space.py` | UNCLASSIFIED |

**Payload checksums** (content only, provenance block excluded):

* `os1_operator_space_410m.json` — `c1e05ebc9894276d6a779fc2dbc0fcdc`

<!-- END GENERATED PROVENANCE -->
