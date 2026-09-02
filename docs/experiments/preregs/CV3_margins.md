# CV3 — MARGINS: is the corpus effect a score effect or a ranking artifact?

**Pre-registered 2026-08-22, before any margin number existed.** Committed before the code that
computes the result, per `CLAUDE.md` §5.

## WHY

Every number in this programme is a **rank** statistic. `pass@k` asks whether the target's rank is
`<= k`; `min` takes the minimum rank over a band. Nothing has ever looked at the underlying
**scores**.

Rank is unstable when many tokens score similarly. A perturbation far too small to matter can move a
target from rank 9 to rank 30 with almost no change in the logit that produced it. So the programme's
headline — corpus identity moves recovery by 13–45 pooled seed SD — has two possible readings:

1. **score effect.** Different fitting corpora produce operators that genuinely score the target
   differently. The corpus effect is real in the model's own units.
2. **ranking artifact.** The target sits in a crowded region of the vocabulary distribution, all
   operators score it about the same, and small score differences get amplified into large rank
   differences by the density of competitors.

These are very different findings and **we currently cannot tell them apart**. The same question
applies to the derangement result (`min` prefers a layer-deranged operator on 84/120 draws): is the
derangement scoring the target higher, or merely reshuffling a crowded neighbourhood?

`docs/validity/DIAGNOSIS.md` §4B and §8.2 flag this as untested. This closes it.

## DESIGN

Model **Pythia-410M-deduped**, band **[9,21]**, corrected (stripped) readout, admitted-5 eval sets.

Arms, all applied to **identical cached activations** so the only thing that varies is the transport:

| arm | n | source |
|---|---:|---|
| `logit_I` | 1 | identity transport (the free lens) |
| `J\|<corpus>\|s{0,1,2}` | 15 | 5 in-stream corpora x 3 seed blocks, `results/e48/lens_INSTREAM_*.pt` |
| `shuf\|Pile-CC\|s0\|d{0,1,2}` | 3 | fixed-point-free layer derangements of one real operator |

For every (item, intermediate, layer, arm) record **five quantities**, not one:

| quantity | definition |
|---|---|
| `rank` | `1 + #{v : logit_v > logit_target}` — the existing statistic |
| `logit` | the target's raw logit |
| `margin` | `logit_target - max_{v != target} logit_v` (negative iff rank > 1) |
| `z` | `(logit_target - mean(logits)) / sd(logits)` |
| `percentile` | `1 - (rank - 1) / V` |

## PRIMARY

Between-corpus spread on the fit axis, computed **twice** — once in rank space (as
`min`-over-layers pass@k AUC, the existing metric) and once in margin space (mean over layers of the
target margin) — each normalised by **its own** pooled seed SD across the three seed blocks.

```
R_rank   = spread_rank   / pooled_seed_sd_rank
R_margin = spread_margin / pooled_seed_sd_margin
PRIMARY  = R_margin / R_rank
```

Both are dimensionless, so the ratio is interpretable.

## DECISION RULE — fixed before running, three-way, both branches publishable

* **ACCEPT — SCORE EFFECT.** `PRIMARY >= 0.50`. The corpus effect is present in the model's own
  score units at at least half the strength it has in rank units. The headline hardens: it is not an
  artifact of a crowded decision boundary, and margins become a reportable secondary metric.
* **REJECT — RANKING ARTIFACT.** `PRIMARY < 0.25`. The corpus effect largely disappears when scores
  replace ranks. **This is a major measurement-validity finding, not a loss:** a published rank-based
  lens benchmark would be amplifying sub-threshold score differences into large apparent effects, and
  that generalises to every pass@k lens comparison. Stop, flag for adjudication, and do not re-cut.
* **UNCLEAR.** `0.25 <= PRIMARY < 0.50`. Report the number and stop. Do not rerun with new settings.

**Secondary, adjudicated separately and not permitted to override the primary:** the derangement
contrast `J^P - J^shuf` recomputed in margin space. If the derangement's advantage under `min` is
present in rank space and absent in margin space, the metric audit's mechanism is refined — `min`
would be shown to reward rank reshuffling rather than better scoring.

## CONTROLS — each with the number it must produce

* **C1 — the readout path is the one already in use.** The `logit_I` arm's `min` admitted mean must
  reproduce `results/e48_crossover_410m_rstrip.json : arms_admitted_mean.logit_I.min` =
  **0.19810852520167826**, to `<= 1e-6`. A mismatch means this script's readout differs from the
  programme's and every number here is void.
* **C2 — margin and rank agree by construction.** `margin > 0` iff `rank == 1`, on **100%** of
  scored (pair, layer, arm) triples. Any violation is an implementation error.
* **C3 — the derangement is a real perturbation.** Mean pairwise cosine between the real operator's
  band Jacobians must be **< 0.95**; at 0.6291 (`t15_shuffle_diagnostic_410m.json`) it passes, and a
  value near 1 would mean the derangement is a near-no-op and the secondary is uninformative.
* **C4 — seed SD is non-degenerate.** Pooled seed SD must be `> 0` in both spaces; a zero denominator
  makes `PRIMARY` undefined and the run void.

## DECLARED BIAS

Margin is measured against the **top competitor**, which is itself operator-dependent. An operator
that suppresses one strong competitor while leaving the target untouched will show a margin gain that
is not a target-score gain. `z` and `logit` are recorded alongside precisely so this is separable, and
any ACCEPT that holds in margin but not in `z` must be reported as such.

The comparison is at a read position **outside the operator's fitting support** (0/551 on prefixed
rungs, `results/cv2_position_support.json`). This experiment does not fix that and does not claim to;
it holds position fixed across arms, so the contrast is internally valid at that position.

## COST

CPU only. No refitting. ~5,800 cached activations x 19 arms of unembedding; estimated **10–20
minutes**, under $0.

## STATUS

**RUN AND ADJUDICATED — 2026-08-22. `results/cv3_margins_410m.json`.**

**PRIMARY = 2.461243802710653 → ACCEPT — SCORE EFFECT.** The corpus effect is not a ranking
artifact, and rank *understates* it 2.5x: 27.16 pooled seed SD in z-space against 9.37 in rank
space. All three controls fired — `C1_logit_reproduces_stored` (it caught a trapezoid-AUC-for-
flat-mean-7 substitution worth 4.3% on identical ranks), `C2_margin_rank_consistency` at 0
violations, `C4_seed_sd_nondegenerate`.

The SECONDARY landed the way this document's DECLARED BIAS predicted: the derangement wins on rank
(-0.00554) and margin (-0.3942) and **loses on z** (+0.0855) — it suppresses the nearest competitor
rather than scoring the target higher. **Report in z-space.**

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e48_crossover_410m_rstrip.json` | 110,898 | `aaf6ee8aa20c703a` | `t48_crossover.py` | CLEAN |
| `results/cv2_position_support.json` | 5,860 | `e3fbf2816d6c9d92` | `cv2_position_support.py` | IMMUNE |
| `results/cv3_margins_410m.json` | 11,283 | `e5d2a4fb84fa96be` | `cv3_margins.py` | CLEAN |

**Payload checksums** (content only, provenance block excluded):

* `e48_crossover_410m_rstrip.json` — `da1cb4e3e01c5c63465e5654426a1233`
* `cv2_position_support.json` — `1eb1ab9b76e77d6060876485b34291cf`
* `cv3_margins_410m.json` — `38efe87c84be1b8bee5fa680a5a3d354`

<!-- END GENERATED PROVENANCE -->
