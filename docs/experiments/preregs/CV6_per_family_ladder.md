# CV6 — the per-family corpus contrast at 1.4B and 2.8B

**Pre-registered 2026-08-23, before any corpus number at 1.4B or 2.8B exists.**
Supersedes CV4's Phases 2–3, which specified a pooled estimand. CV4 Phase 1 has run and is retained.

## AMENDMENT — 2026-08-23, AFTER REGISTRATION, BEFORE THE VERDICT

Recorded visibly because a pre-registration must never be edited silently. **Neither item touches
PRIMARY, the DECISION RULE, or the CONTROLS**, which stand exactly as registered.

1. **A background fact in WHY was wrong and is corrected.** This document said CV4 Phase 1's top-10
   competence "rises monotonically". Read off `results/cv4_phase1_capability.json` it rises
   **16.9% → 32.9% end to end but not monotonically** — it dips at 1.4B (1b 29.4% → 1.4b 27.5%).
   The same error was in `docs/context/STATE.md` and the CV4 prereg and is corrected in both.
   Nothing in CV6's design depended on it.

2. **The COST section's card ruling is superseded by measurement.** It says an L40S is the right
   card under R18 and an H100 merely "works and is ~40% slower". Measured on the box with one
   prompt at the registered `dim_batch=128`: **peak VRAM 26.3 GB at 1.4B and 46.3 GB at 2.8B**.
   A 44 GB L40S therefore **cannot run the 2.8B cell at all**, and narrowing the band does not help
   — the graph root is `min(source_layers)`, but every parameter still requires grad, so the forward
   retains a graph from the embedding upward no matter where the band starts. **CV6 ran on an
   H100 NVL (93 GB).** The card is recorded per model in the results file.

## DISCLOSURE — read this before the design

The per-family framing is **informed by `results/d3_corpus_by_family_410m.json`**, an exploratory
diagnostic run at 410M *before* this pre-registration, at the operator's direction and after the
ordering cost was flagged. D3 is not a pre-registered experiment and its numbers do not grade
anything. **This restriction of estimand may not be described as chosen on capability grounds
alone.** It is disclosed here so that no reader has to reconstruct the ordering.

What D3 showed, and why it changed the estimand: the corpus effect is present in **every** family at
19–43σ in z-space, so the pooled figure was not hiding an absent effect — it was hiding *structure*.
Multilingual is **49.2% of scored pairs** (394 of 801) while the model answers 0.9% of it at every
scale. A pooled estimand weights a family the model cannot do at half the total.

## WHY

CV3 established the corpus effect as a score effect at 410M (27.16σ, z-space). One objection stands:

> 410M answers 4.7% of this battery. An effect measured in a near-degenerate regime may be
> structured noise rather than a property of the operator.

CV4 Phase 1 answered part of that and sharpened the rest. Across 70M→2.8B, **pooled top-1 competence
never separates** (all Wilson intervals overlap from 410M up), **top-10 rises**
16.9%→32.9%, and **prompt surprisal falls monotonically in all six families** (5.204→4.013). So the
models become steadily more fluent on the prompts without becoming measurably better at answering
them. The battery is capability-gated out of Pythia's reach at every scale — CV4's own STOP condition
(*"if even M\* is <~10% pooled competence, log this as a finding"*) has fired at 5.9%.

CV6 therefore asks the surviving question in the estimand that D3 showed is the honest one:

> **Does the per-family corpus effect replicate at 1.4B and 2.8B, and does the per-family ordering
> hold?**

## DESIGN

### Phase A — fit

Per model in {1.4B, 2.8B}: **5 in-stream corpora × 3 seed blocks = 15 operators**, **N = 25** prompts
(S2 settled that reads are flat in N; **do not sweep it**), fp32, TF32 off, `skip_first=16`,
`max_seq_len=128`, `dim_batch=128`, penultimate target. Fit protocol byte-identical to the 410M
operators in `results/e48/` except for `N`.

Band by the standing rule — normalised `L38`–`L92` intersected with layers strictly below the
penultimate target: **1.4B → [9,21] (13 layers), 2.8B → [12,29] (18 layers)**. Recorded per model in
the output; **not** tuned.

### Phase B — cache and score

**Cache `h_t` on the box, in the same precision and on the same device as the fitting.** CV4's spec
said to reuse Phase-1 activations; that was wrong and is corrected here — Phase 1 ran on CPU, fitting
runs on GPU, and D2 measured a CUDA-vs-CPU divergence of 2.774e-04 at the cell level. Applying
GPU-fitted operators to CPU-cached activations would bake a device mismatch into the contrast.

Score the admitted five families separately, corrected readout, flat-mean-7 k-summary, no prefix.

## PRIMARY

**Per family `f` and per model `m`:**

```
R(f, m) = spread_z(f, m) / pooled_seed_sd_z(f, m)
```

where the spread is across the 5 corpora and the seed SD is the RMS of per-corpus sample SDs over the
3 seed blocks — **computed within that family**, not pooled across families.

Reported beside it, for every family and model: the **Kendall tau** of the 5-corpus ordering against
that family's 410M ordering (`d3_corpus_by_family_410m.json`).

**There is no pooled primary.** The pooled estimand is retired: it weights multilingual at 49.2% of
pairs at 0.9% competence.

## DECISION RULE — fixed before running, three-way, all branches publishable

Evaluated **per family**, and the verdict is the count across the five:

* **REPLICATES** — `R(f, 2.8B) >= 10` on **>= 4 of 5 families**, and ordering tau >= 0.6 on >= 3 of 5.
  The corpus effect is not an artifact of the low-competence regime at 410M.
* **ATTENUATES** — `R(f, 2.8B) < 5` on **>= 4 of 5 families**. The effect is a small-model phenomenon.
  **Publishable and arguably sharper**: it would make corpus sensitivity itself capability-gated.
* **UNCLEAR** — anything else, including a split where some families replicate and others do not.
  **A split is an informative outcome, not a failure**: it would say the corpus effect is
  family-specific, which the D3 orderings already hint at. Report the per-family table and stop.

**Do not pool to force a verdict. Do not add models to chase a threshold. Do not drop a family.**

## CONTROLS — each with the number it must produce

* **C1 — `logit_I` per model.** The identity arm must be emitted and stored for every model and
  family before any J arm is graded. At 410M it must equal **0.19810852520167826** (flat-mean-7,
  `min`, admitted-5). **No model's numbers are graded if C1 has not fired on that model.**
* **C2 — band matches the standing rule.** Emitted band per model must equal the rule's output.
  A tuned band voids the run.
* **C3 — seed SD non-degenerate per family.** `pooled_seed_sd_z(f, m) > 0` for every (f, m), else
  `R` is undefined there and that cell is void — not silently dropped.
* **C4 — N is 25 and identical across corpora.** Any corpus fitted at a different N voids the
  contrast.
* **C5 — device/precision consistency.** Activations and operators from the same device and dtype.
  Record both.

## DECLARED BIAS

* **Capability and scale are confounded.** Every larger model is also differently trained. No outcome
  licenses "capability causes the effect to persist/shrink" — only "the effect does/does not survive
  up the ladder."
* **The 5 corpora are a fixed panel, not a sample.** n=5 remains the replication unit for any claim
  about *kinds* of corpus.
* **N=25 here vs N=200 at 410M.** The ladder is not N-matched to CV3. S2 established reads are flat
  in N, so this is expected to be immaterial — but it is a difference, and if a family attenuates it
  must not be attributed to scale without checking N.
* **Per-family seed SDs are thin.** 3 seeds, 96–394 pairs per family. Poetry's 410M pooled seed SD is
  0.0056, the smallest of the five, and thin denominators are exactly what inflated the retired 58×
  figure. Report the SD alongside every ratio.

## COST — from `tools/fit_cost_model.py`, reproducible

N=25, 15 operators, fp32, TF32 off:

| model | band | PFLOP × 15 | L40S fp32 | H100 fp32 |
|---|---:|---:|---:|---:|
| 1.4B | 13 | 146 | **27 min** | 36 min |
| 2.8B | 18 | 382 | **70 min** | 1.6 h |

**~1.6 h on one L40S, ~\$1–3.** Inside the `CLAUDE.md` §7 gate (\$40 / 6 h).

**Choose the GPU by fp32 TFLOPS, per R18: L40S 91.6 > 4090 82.6 > H100 67.** An H100 works and is
~40% slower for this workload; it is not the right card under the programme's own rule.

**bf16 is not admissible.** R18 forbids it (10-bit mantissa vs the anchor gate's 2e-5) and CV5 adds
an independent reason: at bf16-scale error the rank metric moves **1.834 pooled seed SDs**, so a
precision change mid-ladder would confound precision with capability. If bf16 is wanted for 6.9B/12B
later, it requires a **matched-precision control at 2.8B** (fit both ways, show operators and reads
agree within seed noise) — ~8 extra minutes.

## WHAT THIS DOES NOT DO

* Does **not** attempt causal validation of the annotated intermediates. That is the synthetic-suite
  programme, `docs/validity/CONSTRUCT_VALIDITY.md` §6 Route B, and a separate paper.
* Does **not** rehabilitate absolute recovery claims. cv1 and cv2 retired those and CV6 does not
  reopen them at any capability level.
* Does **not** sweep N, aggregation, readout position or band.
* Does **not** cover 6.9B/12B. Those need an operator ruling on the budget gate (4.6 h / 10.3 h on
  L40S fp32) and a memory plan — the backward runs through a residual-stream loss and the
  checkpointing for it is not wired.

## OUTPUT

**Adjudicated:** `results/cv6_per_family_ladder.json`.

**Per model and controls, under `results/cv6/`:** `cv6_1.4b_n25.json`, `cv6_2.8b_n25.json` (the
per-arm per-family reads, written after every arm so a crash loses at most one operator) and
`cv6_c0_scorer_equivalence.json` (control C0). Alongside them, not scored from here:
`ht_cache_<model>.pt`, the cached `h_t` so a re-score needs no forward passes, and
`lenses/lens_INSTREAM_<corpus>_<model>_n25_s<seed>.pt`, the 30 fitted operators stored fp16 as the
`results/e48/` convention does.

Fold into `docs/validity/DIAGNOSIS.md`, one line into `docs/context/STATE.md`, and the per-family
estimand into `paper/ARGUMENT.md`. The generated description is
`docs/experiments/descriptions/CV6.md`, emitted by `tools/make_cv6_description.py` so that no number
in it is transcribed by hand.

## STATUS

**RUN AND ADJUDICATED — 2026-08-24. `results/cv6_per_family_ladder.json`.**

### VERDICT: REPLICATES

`R(f, 2.8B) >= 10` on **4 of 5** families and ordering tau `>= 0.6` on **5 of 5**. Both clauses of
the REPLICATES branch are met. **The corpus effect is not an artifact of the low-competence regime
at 410M.** All six controls fired.

| family | pairs | z spread | z pooled seed SD | **R(f, 2.8B)** | tau vs 410M | best -> worst (z) |
|---|---:|---:|---:|---:|---:|---|
| multihop | 103 | 0.6007 | 0.05251 | **11.44** | +0.60 | StackEx > USPTO > Pile-CC > Wiki > Github |
| multilingual | 394 | 0.6540 | 0.04509 | **14.51** | +0.80 | StackEx > USPTO > Pile-CC > Wiki > Github |
| order-ops | 110 | 0.3746 | 0.05798 | **6.46** | +0.80 | **USPTO** > StackEx > Pile-CC > Github > Wiki |
| poetry | 98 | 0.3792 | 0.03180 | **11.93** | +0.60 | USPTO > Pile-CC > Wiki > StackEx > Github |
| typo | 96 | 0.5955 | 0.05217 | **11.41** | +1.00 | StackEx > Pile-CC > USPTO > Wiki > Github |

### READ R WITH THE DECLARED BIAS, NOT AGAINST 410M

**`R` is not comparable across `N`, and this document said so before the run.** R falls from
19–43 at 410M to 6–15 here, and that is a DENOMINATOR effect, not attenuation:

* the **numerator grew in 5 of 5 families** — z spread 410M -> 2.8B: 0.4426 -> 0.6007, 0.3826 ->
  0.6540, 0.2971 -> 0.3746, 0.2381 -> 0.3792, 0.5255 -> 0.5955;
* the **denominator grew faster**, because N=25 has a higher seed noise floor than N=200 — pooled
  seed SD is **2.29x to 5.70x** larger (poetry, the thinnest at 410M, moves 0.00558 -> 0.03180).

So the corpus effect is **larger in absolute z at 2.8B than at 410M in every family**, and
REPLICATES is reached *despite* a harsher denominator. Any statement of the form "the effect
weakens up the ladder" is unsupported by this run and must not be made from R.

### STRUCTURE THAT SURVIVED

* **The bottom of the ordering is near-invariant, as at 410M.** Github is **last in 4 of 5**
  families and 4th in the fifth; Wikipedia_en is 4th in 3 of 5.
* **USPTO tops order-ops and only order-ops at 2.8B**, reproducing D3's corpus x task interaction
  at a scale D3 never saw. (D3 was exploratory; this is the registered replication of that pattern.)
* **order-ops is the one family that misses the bar** — R 6.46 at 2.8B, and at 1.4B R 4.7 with
  **tau -0.40**, an inverted ordering with Github 2nd. It is also the family with the highest
  answer competence (14.5% top-1). Reported, not explained.

### CONTROLS, EACH WITH THE NUMBER IT PRODUCED

| control | required | observed | fired |
|---|---|---|---|
| **C0** (added; not in the original list) | the scorer reproduces D3 on the 15 stored 410M operators | worst \|dz\| **0.000e+00**, worst \|drank\| **0.000e+00** over 16 arms x 5 families; NEGATIVE control at the legacy unstripped readout separates at \|dz\| **1.997** | YES |
| **C1** | identity arm emitted per model and family; 410M anchor 0.19810852520167826 | emitted for all five families at both models; the 410M anchor reproduces to **5.9e-09**. Ladder values: 1.4B **0.23942414368**, 2.8B **0.26996034094** | YES |
| **C2** | emitted band == the rule's output | 1.4B `9..21` (13 layers), 2.8B `12..29` (18 layers), both == rule | YES |
| **C3** | pooled seed SD > 0 per (family, model) | min over all ten cells **0.02063**; no void cells | YES |
| **C4** | N == 25, identical across all 15 arms of both models | `[25]` at both models; 15/15 arms each; the 15 fitting-prompt SHAs are **identical across the two models** and **distinct within each** | YES |
| **C5** | activations and operators same device and dtype | `cuda` / `float32` on an H100 NVL for both; TF32 off and measured off at 1.155e-06 | YES |

### DIAGNOSTIC, NOT A DECISION INPUT — the fp16 storage question, measured

The 410M operators in `results/e48/` are STORED fp16, so D3 scored an fp16-rounded operator while
CV6's primary scores the fp32 accumulator. Rather than assume that gap is immaterial, every arm was
scored both ways: worst `|d z_mean|` over all ten (model, family) cells is **1.171e-04**, and the
corpus ordering is **unchanged in 10 of 10**. The ladder-vs-410M comparison does not rest on an
assumption about storage precision.

### WHAT THE RUN COST, AND THE CARD

**3 h 59 min of fitting** on one **H100 NVL (93 GB)** at \$2.308/h, ~\$10 all-in — inside the
\$40 / 6 h gate. 1.4B: 69.0 min, peak 26.4 GiB. 2.8B: 2 h 50 min, peak 46.3 GiB. See the AMENDMENT
at the top: the 46.3 GiB peak is why a 44 GB L40S could not have run the 2.8B cell at all.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/cv4_phase1_capability.json` | 18,253 | `1b05b8f55679897f` | `cv4_phase1_capability.py` | IMMUNE |
| `results/d3_corpus_by_family_410m.json` | 19,603 | `b4342917486e9329` | `d3_corpus_by_family.py` | CLEAN |
| `results/cv6_per_family_ladder.json` | 30,023 | `f8cdbcc60189e3f9` | `cv6_per_family_ladder.py` | CLEAN |
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
| `results/cv6/cv6_1.4b_n25.json` | 30,228 | `4e8eab1ebedf0785` | `cv6_per_family_ladder.py` | CLEAN |
| `results/cv6/cv6_2.8b_n25.json` | 32,402 | `7f1a6d31f1c8728c` | `cv6_per_family_ladder.py` | CLEAN |
| `results/cv6/cv6_c0_scorer_equivalence.json` | 4,006 | `1d643c0c329014e8` | `cv6_per_family_ladder.py` | CLEAN |

**Payload checksums** (content only, provenance block excluded):

* `cv4_phase1_capability.json` — `43b21658bcc6a1acf619b614a5eb74db`
* `d3_corpus_by_family_410m.json` — `fc353bf005b4e4ab1bf00c84bc40013d`
* `cv6_per_family_ladder.json` — `37c137de5d7788ff265c533db4dabc53`
* `cv6_1.4b_n25.json` — `307871427cdc9dbafbe962115e903a64`
* `cv6_2.8b_n25.json` — `38cc82045d0d27b889fe4cd716dc6f5b`
* `cv6_c0_scorer_equivalence.json` — `ebbd6596928b4d50b4bc0a0034a5708d`

<!-- END GENERATED PROVENANCE -->
