# E59 — is the fit/read asymmetry a property of the method, or of the dose?

**PRE-REGISTERED 2026-08-16, before the run.** Written against
`docs/validity/EXPERIMENT_VALIDITY_AUDIT.md` §3.5.

---

## WHY

The paper's headline is that the corpus which **built** the operator explains far more variance
(91%) than the corpus supplying the **read context** (7%). The two axes are not dose-matched:

| axis | manipulation | text the axis actually sees |
|---|---|---|
| fit | `N=200` documents averaged into the operator | **25,600 tokens** |
| read | one 128-token prefix prepended to the concept prompt | **128 tokens** |

and the readout position is inside the **concept prompt**, not inside the corpus text
(`t52_factorial.py:227-234`), so the read context reaches the measured activation only through
attention over a prepended prefix.

**That is a 200× dose ratio, and the axis with 200× the dose wins.** The paper's scope hedge
("fixed-target contextual variation") covers the *target* being fixed; it does not mention the
dose, and the dose is the mechanism of the objection. Nothing on disk bounds it: there is no
prefix-length sweep, so nobody can say whether 7% is the read axis's ceiling or its value at the
smallest dose anyone tried.

This is the first question a referee at a measurement-validity venue will ask, and right now the
paper cannot answer it.

## DESIGN

`experiments/t59_read_dose.py --device cpu`

The E52 grid machinery, unchanged, with **one factor added: the read-context length**.

- **doses** `prefix_tokens ∈ {128, 384, 768}` — a 6× span on the read axis.
- **operators** the same 21 matched-provenance lenses (7 corpora × 3 seed blocks) from
  `results/e48/`, `trainval.py` N=200, band [9,21]. **No refitting.** The fit axis is held at its
  E52 setting deliberately: the question is whether the *read* axis moves.
- **corpora** 7, not 8. **`OOD_CommonPile` is excluded** — measured this session, it has **0
  documents of ≥768 tokens and 147 of ≥384**, so it physically cannot supply a longer read
  context. See DECLARED BIAS 1.
- **the same documents at every dose.** The prefix pool for a rung is drawn once from that
  corpus's **held-out** documents of ≥768 tokens and then truncated to 128 / 384 / 768. Drawing a
  fresh pool per dose would confound dose with document selection, since longer documents are a
  different sample.
- **held-out** the E52 exclusion (`load_pool_heldout`) is kept: no prefix document is one its own
  operator was fitted on.
- 3 prefix seeds, 541 items, 5 admitted sets, both aggregations.

## PRIMARY

**`read_pct`, the read-axis share of the two-way variance decomposition of the 7×7 matrix, as a
function of dose**, under `persist`.

## DECISION RULE — fixed before the run

Let `read_pct(d)` be the read share at dose `d`, and `Δ = read_pct(768) − read_pct(128)`.

- **DOSE-DRIVEN** — `read_pct(768) ≥ 2 × read_pct(128)` **and** monotone across the three doses.
  The 91/7 headline is substantially an artifact of unequal manipulation strength. The paper must
  restate the claim as being about this pair of knob settings and report the dose curve.
- **DOSE-INSENSITIVE** — `Δ < 2 percentage points` and the fit axis still exceeds the read axis at
  every dose. The asymmetry is not about dose; the headline strengthens materially and the paper
  gains its answer to the referee's first question.
- **PARTIAL** — anything between: the read share grows but not by 2×, or grows non-monotonically.
  Report the curve, and state the headline as dose-dependent with the measured slope.
- **UNCLEAR** — C1 does not fire. Stop.

**This rule is not to be reinterpreted after seeing the curve.** In particular, DOSE-DRIVEN is a
result, not a failure: it would mean the paper's contribution is narrower and better specified.

## CONTROLS

- **C1 — dose 128 must reproduce the E52/E57 structure.** The 7×7 at dose 128 is a *different
  document pool* from E52 (it is drawn from the ≥768-token subset), so cell values will not match.
  What must match is the **structure**: `fit_pct > read_pct` at dose 128 under both aggregations,
  and the fit-axis row ordering must agree with E52's 7-corpus row ordering at Spearman ≥ 0.8.
  *Number required to count as firing:* Spearman(row means, E52 row means) ≥ 0.8 at dose 128.
  If C1 fails, the long-document subsample is not the same population and the dose comparison is
  confounded with document selection.
- **C2 — the unfitted baseline must move with dose.** The logit-lens arm (`J=I`) has no fitted
  parameters, so any movement in it across doses is pure context effect with no operator involved.
  It bounds how much the read axis *can* move. Reported at every dose; if the logit arm is flat in
  dose, the model is not using the longer context at all and the experiment is uninformative
  regardless of what the grid does.
- **C3 — prefix-pool disjointness.** Zero prefix documents may come from any fitting pool, at
  every dose. *Number required:* 0.
- **C4 — cycling disclosure.** StackExchange has only **457** held-out documents of ≥768 tokens
  against 541 items, so its pool cycles at ~1.18×. The realised cycling factor is recorded per
  rung; no other rung should cycle.

## DECLARED BIAS

1. **`OOD_CommonPile` is dropped, and it is dropped for a reason correlated with the corpus axis** —
   its documents are short. The dose result is therefore measured on 7 of the paper's 8 corpora,
   and the 8-corpus headline is not itself re-measured here. This is a real limitation and it is
   the corpus's property, not a sampling accident.
2. **Restricting to ≥768-token documents changes the read-context population** for every corpus,
   toward longer documents. C1 is what tests whether that matters; it cannot rule it out entirely.
3. **The fit axis is not dose-matched by this experiment either.** Matching it would mean fitting
   operators at N=1 or reading 25,600-token contexts; the first is degenerate and the second
   exceeds the model's context window. This experiment measures the read axis's *slope* in dose,
   which bounds the argument without equalising the axes.
4. **StackExchange cycles** (C4). One rung with slightly less prefix diversity than the others.
5. The readout position still never enters the corpus text. E59 tests dose, not the fixed-target
   limitation, which stands.

## COST

CPU only, on the laptop, **$0**. 63 activation caches (7 rungs × 3 prefix seeds × 3 doses) against
E57's measured 30 caches in ~45 min, scaled by sequence length (≈1 / 2.25 / 4.2):
**estimated 2.5–3.5 h wallclock.** No GPU, no box, no teardown obligation.

## FLAGGED DELTAS

- Reuses `results/e48/lens_*.pt` — the same 21 operators E52 used, so the fit axis is identical to
  the paper's and only the read axis moves.
- The 8th corpus is absent by physical necessity, not by choice. Any statement of the result must
  say "on seven of the eight corpora".
# E59 RESULT — appended to docs/experiments/preregs/E59_read_dose_response.md

## RESULT (2026-08-17, `results/e59_read_dose_410m.json`)

7 corpora (`OOD_CommonPile` excluded — 0 documents of ≥768 tokens), 3 prefix seeds, 3 fit seeds,
541 items, band [9,21], 63 activation caches, 10 005 s on the laptop CPU, **$0**.

| read dose | fit:read dose ratio | fit % (`persist`) | read % | fit % (`min`) | read % |
|---|---|---|---|---|---|
| 128 tokens | 200:1 | 88.68 | **9.46** | 77.88 | **19.31** |
| 384 tokens | 67:1 | 75.63 | **22.82** | 65.58 | **32.42** |
| 768 tokens | 33:1 | 82.86 | **15.70** | 68.28 | **29.39** |

**Paired-draw movement** (9 draws per dose, paired by fit seed × prefix seed):

| transition | `persist` | `min` |
|---|---|---|
| 128 → 384 | **+10.36 pts, up in 9/9** | **+11.17 pts, up in 9/9** |
| 128 → 768 | +4.34 pts, up in 6/9 | +8.69 pts, up in 6/9 |
| 384 → 768 | −6.02 pts, up in 3/9 | −2.48 pts, up in 3/9 |

**fit > read in 9/9 draws at every dose under both aggregations — 54/54.**

Per-draw read-share SDs: 4.77 / 9.06 / 2.80 (`persist`), 5.79 / 6.47 / 8.45 (`min`).

## CONTROLS

| control | required | produced | fires |
|---|---|---|---|
| **C1** row ordering vs E52 at dose 128 | Spearman ≥ 0.8 | **+1.000** | ✓ |
| **C2** unfitted logit arm moves with dose | range > 0 | **0.00190** (0.03007 → 0.02926 → 0.02817) | ✓ |
| **C3** prefix/fit document overlap | exactly 0 | **0** | ✓ |
| **C4** cycling recorded per rung | — | StackExchange **1.18×**, all others **1.00×** | ✓ |

## VERDICT — **PARTIAL**, as the script adjudicated against the pre-registered rule

`read_pct(768) = 15.70` is **1.66×** `read_pct(128) = 9.46`, short of the registered 2× for
DOSE-DRIVEN, and the curve is **not monotone**. The movement is `+6.2` points, above the 2-point
band for DOSE-INSENSITIVE. So PARTIAL, and the rule says: report the curve, and state the headline
as dose-dependent with this slope.

## WHAT IT MEANS, stated carefully

**The read axis is genuinely dose-sensitive.** 128 → 384 raises its share by ~10–11 points and does
so in **9 of 9 paired draws** under both aggregations. That is unambiguous, and it means
**the 7% in the paper's headline table is the read axis's value at the smallest dose tried, not its
ceiling.** Anyone quoting "91 vs 7" as a property of the method is over-reading it.

**The ordering is not dose-sensitive.** The fit axis is larger in **all 54 draws**, at every dose,
under both aggregations — including at 768 tokens where the dose ratio between the axes has
narrowed from 200:1 to 33:1. The claim the paper actually makes (fit > read) survives a 6× dose
sweep; the *magnitude* of the gap does not transfer across doses.

**Do not read the non-monotonicity as a finding.** 384 → 768 falls by 6.0 points and is lower in
only 6 of 9 paired draws, against per-draw SDs of 2.8–9.1. Three doses with 9 draws each does not
support a shape. The defensible statement is "rises then does not keep rising", not "peaks at 384".

## FLAGGED DELTAS

1. **`OOD_CommonPile` is absent** and absent for a reason correlated with the corpus axis — its
   documents are short (0 at ≥768 tokens, 147 at ≥384). The dose result is 7 of the paper's 8
   corpora. The 8-corpus headline is not itself re-measured here.
2. **Restricting to ≥768-token documents changes the read-context population** toward longer
   documents at every dose. C1 (ρ = +1.000 against E52's row ordering) is what bounds this; it
   cannot exclude it entirely.
3. **`OOD_News_2024`'s read contexts are ~24% duplicated** — 541 prefixes drawn, 405–420 distinct,
   which is that corpus's known 33% duplication surfacing on the read axis. Recorded per rung in
   `pool_meta`; not anticipated in the pre-registration.
4. **StackExchange cycles at 1.18×** (457 held-out documents of ≥768 tokens against 541 items), as
   C4 registered in advance.
5. The readout position still never enters the corpus text. E59 tests dose, not the fixed-target
   limitation, which stands.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e59_read_dose_410m.json` | 40,401 | `7b95f7db997008ea` | `t59_read_dose.py` | EXPOSED |
| `results/e48/tv_INSTREAM_Github_s0.json` | 2,276 | `2dac8fd33a1dfd5f` | `—` | EXPOSED |
| `results/e48/tv_INSTREAM_Github_s1.json` | 2,276 | `929903c91b3ec522` | `—` | EXPOSED |
| `results/e48/tv_INSTREAM_Github_s2.json` | 2,277 | `b3e6a7eaa26bb48b` | `—` | EXPOSED |
| `results/e48/tv_INSTREAM_Pile-CC_s0.json` | 2,280 | `84cd0d2be71c0662` | `—` | EXPOSED |
| `results/e48/tv_INSTREAM_Pile-CC_s1.json` | 2,275 | `6eb775b06a528a01` | `—` | EXPOSED |
| `results/e48/tv_INSTREAM_Pile-CC_s2.json` | 2,276 | `f9d8523fd5256050` | `—` | EXPOSED |
| `results/e48/tv_INSTREAM_StackExchange_s0.json` | 2,286 | `219273f53cfed1db` | `—` | EXPOSED |
| `results/e48/tv_INSTREAM_StackExchange_s1.json` | 2,288 | `985ab0889e960661` | `—` | EXPOSED |
| `results/e48/tv_INSTREAM_StackExchange_s2.json` | 2,296 | `11e0ed7aed307dad` | `—` | EXPOSED |
| `results/e48/tv_INSTREAM_USPTO_Backgrounds_s0.json` | 2,268 | `c74b8fd9982e8ab6` | `—` | EXPOSED |
| `results/e48/tv_INSTREAM_USPTO_Backgrounds_s1.json` | 2,294 | `b72f7f910c00defb` | `—` | EXPOSED |
| `results/e48/tv_INSTREAM_USPTO_Backgrounds_s2.json` | 2,292 | `ed97ffe0e9b1fd2c` | `—` | EXPOSED |
| `results/e48/tv_INSTREAM_Wikipedia_en_s0.json` | 2,263 | `12b8d4f1a0eb28ae` | `—` | EXPOSED |
| `results/e48/tv_INSTREAM_Wikipedia_en_s1.json` | 2,280 | `ce5fcb8a8a89f85f` | `—` | EXPOSED |
| `results/e48/tv_INSTREAM_Wikipedia_en_s2.json` | 2,277 | `77c4b0680508f461` | `—` | EXPOSED |
| `results/e48/tv_OOD_CommonPile_s0.json` | 2,267 | `aa26441ea8c2f089` | `—` | EXPOSED |
| `results/e48/tv_OOD_CommonPile_s1.json` | 2,276 | `a1882c54068e73d1` | `—` | EXPOSED |
| `results/e48/tv_OOD_CommonPile_s2.json` | 2,267 | `852caa1351320ac6` | `—` | EXPOSED |
| `results/e48/tv_OOD_News_2024_dedup_s0.json` | 2,277 | `e6e9e4231c04bff9` | `—` | EXPOSED |
| `results/e48/tv_OOD_News_2024_dedup_s1.json` | 2,294 | `4633c4b5d19531e6` | `—` | EXPOSED |
| `results/e48/tv_OOD_News_2024_dedup_s2.json` | 2,295 | `2e51f29de96391f4` | `—` | EXPOSED |
| `results/e48/tv_OOD_News_2024_s0.json` | 2,267 | `6fb9b3a0badfc05c` | `—` | EXPOSED |
| `results/e48/tv_OOD_News_2024_s1.json` | 2,301 | `cc273f68e16d8b06` | `—` | EXPOSED |
| `results/e48/tv_OOD_News_2024_s2.json` | 2,301 | `de1c4b693a254447` | `—` | EXPOSED |
| `results/e48/tv_OOD_arXiv_2023_s0.json` | 2,275 | `27add57555e6bfcc` | `—` | EXPOSED |
| `results/e48/tv_OOD_arXiv_2023_s1.json` | 2,271 | `a13f1689cf5cd970` | `—` | EXPOSED |
| `results/e48/tv_OOD_arXiv_2023_s2.json` | 2,278 | `8b6d11f5bca9f28c` | `—` | EXPOSED |

**Payload checksums** (content only, provenance block excluded):

* `e59_read_dose_410m.json` — `06272f749e1fbef830d20559f98bc732`

<!-- END GENERATED PROVENANCE -->
