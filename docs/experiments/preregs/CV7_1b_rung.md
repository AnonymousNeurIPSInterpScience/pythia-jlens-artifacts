# CV7 — the 1B rung, from operators that already exist

**Pre-registered 2026-08-24, before any per-family corpus number at 1B exists.**
Scoring only. **No fitting, no GPU.** The 15 operators were fitted for E62 and are on disk.

## DISCLOSURE — the ordering, stated so no reader has to reconstruct it

This document is written **after** CV6's 1.4B and 2.8B tables were seen, at the operator's
direction. Two consequences, both binding:

1. **The thresholds are inherited verbatim from `CV6_per_family_ladder.md`**, which fixed them
   before any ladder number existed. They are not re-derived here and they are not tuned to 1B.
2. **The decision to run 1B at all is informed by CV6's result** — specifically that the ladder is
   non-monotone, with 1.4B lower than both neighbours on multihop and order-ops, and order-ops at
   1.4B carrying the only negative Kendall tau in the run (−0.40). CV7 therefore **may not be
   described as a blind extension of CV6.** It is a disclosed follow-up.

## WHY

CV6 returned REPLICATES at 2.8B. The ladder underneath it is not clean:

* z spread 410M → 1.4B **fell** on multihop (0.4426 → 0.3305) and order-ops (0.2971 → 0.2538),
  while 410M → 2.8B **rose in 5 of 5**;
* order-ops at 1.4B is R 4.72 with tau **−0.40** — an inverted ordering with Github promoted to
  2nd, the only negative tau anywhere in CV6;
* CV4 Phase 1 shows 1.4B also dips in answer competence (top-1 3.1% against 1B 4.3% and 2.8B 5.9%).

With only two rungs between 410M and 2.8B, a dip at one of two is weak evidence for anything.
**1B is the rung that discriminates**, and its operators already exist, so the measurement is free.

## DESIGN

**Scoring only.** No operator is fitted. Score the 15 stored operators
`results/ladder1b_b613/lens_<corpus>_1b_n200_s<seed>.pt` — `EleutherAI/pythia-1b-deduped`, 16
layers, `d_model` 2048, **N=200**, stored fp16, `source_layers` **[6,13]** — plus the unfitted
identity arm, against one `h_t` cache built on the scoring device.

Band **[6,13], 8 layers**, which is `band_for(16)` under the standing rule (normalised L38–L92
intersected with layers strictly below the penultimate target). **Read off the artifact, not
chosen**: C2 asserts the stored `source_layers` equal the rule's output.

Readout, k-summary, pooling and estimand are **byte-identical to CV6 and D3**: stripped prompt, no
prefix, flat-mean-7 over K=(1,2,5,10,20,50,100), flat pool over (item, intermediate) pairs *within*
a family, z = max over band per pair then mean over pairs, seed SD = RMS over the 5 corpora of each
corpus's 3-seed sample SD.

**Device: CPU.** This is the D3 situation, and it is declared rather than hidden — the operators
were fitted on CUDA and are scored on CPU, where D2 measured a cell-level divergence of 2.774e-04.
CV6 avoided that by fitting and scoring on one device; CV7 cannot, because it is not refitting.
D3's 410M reference has the same property, which is what makes 1B-vs-410M the *matched* comparison
here.

## PRIMARY

**Per family `f`:** `R(f, 1B) = spread_z(f) / pooled_seed_sd_z(f)`, computed within the family,
and the **Kendall tau** of the 1B 5-corpus ordering against that family's 410M ordering in
`results/d3_corpus_by_family_410m.json`.

**The cross-ladder comparison is the z SPREAD, not R.** 1B and 410M are both N=200 and their R is
comparable; CV6's 1.4B and 2.8B are N=25 and theirs is not comparable to either. Any statement
ranking rungs by R across different N is void.

## DECISION RULE — fixed before running, three-way, thresholds inherited from CV6 verbatim

* **REPLICATES AT 1B** — `R(f,1B) >= 10` on **>= 4 of 5** families **and** tau `>= 0.6` on
  **>= 3 of 5**.
* **ATTENUATES AT 1B** — `R(f,1B) < 5` on **>= 4 of 5** families.
* **UNCLEAR** — anything else, including a split. Report the table and stop.

**SECONDARY, pre-specified, no threshold and no verdict attached:** does order-ops carry a negative
tau at 1B as it does at 1.4B? A negative tau at both rungs makes the inversion a property of
order-ops; a positive tau at 1B makes it specific to the 1.4B checkpoint. **This is reported as a
sign, not graded**, because with n=1 rung either way it cannot support more.

**Do not pool. Do not drop a family. Do not re-cut R across mismatched N.**

## CONTROLS — each with the number it must produce

* **C1 — identity arm.** `logit_I` emitted for all five families before any J arm is graded. The
  scorer is the one CV6's C0 already validated at **0.000e+00** against D3 on the stored 410M
  operators, with its negative control separating at **1.997**; C0 is not re-run.
* **C2 — band read off the artifact equals the rule.** Every one of the 15 lenses must carry
  `source_layers == [6,7,8,9,10,11,12,13] == band_for(16)`. A lens with any other band voids the run.
* **C3 — seed SD non-degenerate per family.** `pooled_seed_sd_z(f) > 0` for all five, else that
  cell is void, not silently dropped.
* **C4 — N identical across corpora.** All 15 lenses must record `n_prompts == 200`. Any lens at a
  different N voids the contrast.
* **C5 — the 15 arms are 5 distinct corpora x 3 distinct seeds**, and the pair count per family
  must equal D3's exactly: multihop 103, multilingual 394, order-ops 110, poetry 98, typo 96. A
  different battery is a different experiment.

## DECLARED BIAS

* **Ordering, as above.** Written after CV6's tables were seen.
* **1B is not simply "a smaller 1.4B".** It is 16 layers at `d_model` 2048 against 1.4B's 24 at the
  same width — **shallower at equal width**, not narrower. Its band is **8 layers against 13 and
  18**. Both `min`-over-band and max-over-band are band-width sensitive; D1 measured the union
  effect directly. **A 1B-vs-1.4B difference may be a band-width effect rather than a scale
  effect, and nothing in this design can separate them.**
* **N=200 here vs N=25 in CV6.** Stated again because it is the single easiest error to make with
  this table.
* **CPU scoring against CUDA-fitted operators**, as above.
* **The 5 corpora are a fixed panel, not a sample.** n=5 is the replication unit.

## COST

**Free. ~6 TFLOP of scoring, minutes on CPU.** No GPU, no provisioning, no budget gate.

## OUTPUT

`results/cv7_1b_rung.json`, per-model detail at `results/cv7/cv7_1b_n200.json`. Fold into
`docs/context/CONTEXT.md` §1d, `docs/context/STATE.md` and `docs/context/RESULTS_TAXONOMY.md`.

## STATUS

**RUN AND ADJUDICATED — 2026-08-24. `results/cv7_1b_rung.json`. Free, CPU, no fitting.**

### VERDICT: REPLICATES AT 1B — but read the tau column before quoting it

`R(f,1B) >= 10` on **5 of 5** families and tau `>= 0.6` on **exactly 3 of 5**, which is the bar
itself, not a comfortable margin. All five controls fired.

| family | pairs | z spread | z pooled seed SD | R(f,1B) | tau vs 410M | best -> worst (z) |
|---|---:|---:|---:|---:|---:|---|
| multihop | 103 | 0.6069 | 0.02065 | **29.38** | **+0.40** | StackEx > USPTO > Pile-CC > Github > **Wiki** |
| multilingual | 394 | 0.2972 | 0.01167 | **25.48** | +0.80 | StackEx > Pile-CC > USPTO > Github > **Wiki** |
| order-ops | 110 | 0.4765 | 0.01503 | **31.71** | +0.80 | StackEx > USPTO > Pile-CC > Wiki > Github |
| poetry | 98 | 0.2035 | 0.01125 | **18.08** | **+0.40** | USPTO > StackEx > Wiki > Pile-CC > Github |
| typo | 96 | 0.7619 | 0.03300 | **23.09** | +0.80 | StackEx > Pile-CC > USPTO > Github > **Wiki** |

**R is large here because N=200, not because 1B is special.** 18–32 against 410M's 19–43 at the
same N; CV6's 6–15 is at N=25. This is the comparison the DECLARED BIAS said to make and the one
the numbers support.

### THE SECONDARY ANSWERS THE QUESTION THAT MOTIVATED THIS RUN

order-ops Kendall tau by rung: **1B +0.80 · 1.4B −0.40 · 2.8B +0.80.**

The inverted ordering is **specific to the 1.4B checkpoint**, not a property of order-ops. It is
the only negative tau anywhere across four rungs. Recorded as a **sign, not a result** — the
pre-registration attached no threshold to it, and with one rung either side it cannot carry more.
It does not explain *why* 1.4B inverts, and CV4's competence dip at the same rung (top-1 3.1%
against 1B 4.3% and 2.8B 5.9%) is a coincidence this design cannot promote to a cause.

### WHAT THIS RUN BREAKS — the bottom of the ordering is NOT stable across rungs

D3 at 410M had Github last in **5 of 5** families, and CONTEXT, HANDOFF, STATE and
`paper/ARGUMENT.md` all carry that as *"the bottom of the ordering is family-invariant."* At 1B it
is not:

| rung | who is last, by family |
|---|---|
| 410M (N=200) | Github **5/5** |
| **1B (N=200)** | **Wikipedia_en 3/5**, Github 2/5 |
| 1.4B (N=25) | Github 4/5, StackExchange 1/5 |
| 2.8B (N=25) | Github 4/5, Wikipedia_en 1/5 |

The bottom is near-invariant **within** a rung. **Which corpus occupies it is not stable across
rungs**, and 1B is the counterexample. The claim must be restated as within-rung, and any sentence
of the form "Github is always worst" is now false as written. This is a correction CV7 forces on
documents that predate it.

### CONTROLS, EACH WITH THE NUMBER IT PRODUCED

| control | required | observed | fired |
|---|---|---|---|
| **C1** | identity arm emitted for all five families first | emitted 5/5; pooled flat-mean-7 min-rank **0.21900**. The scorer is CV6's, **imported not reimplemented**, and C0 validated it at 0.000e+00 against D3 with a negative control separating at 1.997 | YES |
| **C2** | all 15 lenses carry `source_layers == band_for(16)` | one distinct band across all 15: `[6,7,8,9,10,11,12,13]` == rule. Asserted **before** scoring | YES |
| **C3** | pooled seed SD > 0 per family | min **0.01125** (poetry); no void cells | YES |
| **C4** | all 15 lenses record `n_prompts == 200` | `[200]`, one value. Asserted **before** scoring | YES |
| **C5** | pair counts equal D3's exactly | 103 / 394 / 110 / 98 / 96 — identical | YES |

### THE LADDER, ASSEMBLED

Free baseline (`logit_I`, pooled flat-mean-7 min-rank) rises monotonically at every rung:
**0.19811 (410M) → 0.21900 (1B) → 0.23942 (1.4B) → 0.26996 (2.8B)**.

z spread, N stated because it must be: 1B exceeds 1.4B in **3 of 5** families (multihop
0.6069 vs 0.3305, order-ops 0.4765 vs 0.2538, typo 0.7619 vs 0.7514) and falls below it in 2
(multilingual, poetry). **So "1.4B is a local dip" is supported on the ORDERING (uniquely inverted
tau on order-ops) and only partly on the MAGNITUDE (3 of 5).** Do not state it as a clean dip.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/d3_corpus_by_family_410m.json` | 19,603 | `b4342917486e9329` | `d3_corpus_by_family.py` | CLEAN |
| `results/cv7_1b_rung.json` | 16,173 | `d169a0099812e331` | `cv7_1b_rung.py` | CLEAN |
| `results/cv7/cv7_1b_n200.json` | 16,867 | `558d8d9a96f7a94f` | `cv7_1b_rung.py` | CLEAN |

**Payload checksums** (content only, provenance block excluded):

* `d3_corpus_by_family_410m.json` — `fc353bf005b4e4ab1bf00c84bc40013d`
* `cv7_1b_rung.json` — `a073b38c6bced9536c03b40ba87b15e4`
* `cv7_1b_n200.json` — `39a12fb962ca27e489684090b92af88e`

<!-- END GENERATED PROVENANCE -->
