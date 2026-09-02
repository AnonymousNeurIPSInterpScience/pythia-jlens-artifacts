# E51 — is the corpus effect a level or an interaction?

**Verdict: an INTERACTION. Good for the thesis, and the most structurally novel result the
programme has.**

---

## QUESTION

E33 established that the fitting corpus moves the read by 58× the seed noise. That is compatible
with two very different stories:

- **a level** — some corpora simply make better operators, and one could in principle rank them;
- **an interaction** — the corpus changes *which concepts* are legible, with no consistent
  ranking.

The two have opposite consequences. A level invites a scalar "corpus quality" predictor. An
interaction says no such scalar can exist, because the thing to predict is a matrix.

## PRE-REGISTRATION

**None.** This is a *recomputation*, not a measurement: it re-derives from stored ladder artifacts
a decomposition that had been quoted in six documents and computed by no script in any commit. The
file records this as `"recomputes_not_remeasures"`.

That history is why the experiment exists at all, and it is the honest framing: E51 does not add
evidence, it makes existing evidence checkable. Its own decision content is the **F test**, which
was specified as part of the recomputation.

## DESIGN

Two scales. `pythia-410m-deduped` (band [9,21]) and `pythia-1b-deduped` (band [5,13], see deltas).
`persist`, `N ≥ 75`, five corpora × five admitted concept sets × three disjoint seed blocks.

Variance decomposition of read AUC over (corpus, set) with seed blocks as replication:

```
y[c,s,r] = μ + a_c + b_s + (ab)_{cs} + ε_r
```

Reported both for the five admitted sets and for all six (`association` is floored — 196 of 205
scored cells are exactly zero).

## CONTROLS, and the number each produced

| control | required | produced | fires |
|---|---|---|---|
| **reproduces the prose** | within 0.1 pp | 410M interaction claimed 6.8 / recomputed **6.780626**; corpus main claimed 1.9 / recomputed **1.906450**; set main claimed 91.3 / recomputed **91.278294**. 1B likewise | **yes, all six** |
| **C3** permutation null | — | **MIS-SPECIFIED — DO NOT CITE.** It destroys the corpus main effect along with the interaction, so it cannot separate them | **no** |
| **C4** F test with seed blocks as the error term | F > F_crit(0.001) = 3.3 | **F(16,50) = 611.89** (410M), **567.29** (1B) | **yes** |
| **4614× audit** | reproduce the quoted multiplier | **does not reproduce under any of six natural denominators**: 196× / 235× / 612× / 2548× / 2768× (seed×set) / 10268× (seed main) | **fails — see deltas** |

## RESULT

| | 410M | 1B |
|---|---|---|
| eval-set main effect (task difficulty) | **91.278%** | **88.580%** |
| **corpus × set interaction** | **6.781%** | **9.158%** |
| corpus main effect | 1.906% | 2.212% |
| seed residual | 0.035% | 0.050% |

The interaction is **3.56×** the corpus main effect at 410M and **4.14×** at 1B, and
**F(16,50) = 612 / 567** against a 0.1% critical value of 3.3.

Concretely: Github is **+1.3** on order-ops and **−1.9** on multihop in within-set z-scores. The
sign reverses.

`results/e51_interaction_variance.json`

## VERDICT

**REPRODUCES.** The six variance fractions match the prose to within 0.1 percentage points at both
scales, and the interaction is larger than the seed-level scatter by a factor of 612.

> The fitting corpus does not make the Jacobian lens better or worse. It changes which concepts
> the lens can read.

## FLAGGED DELTAS

1. **"4614× the seed×set noise floor" is RETRACTED.** The variance fractions reproduce exactly;
   the multiplier does not follow from that decomposition under any natural denominator, and **no
   script in any commit ever computed it**. It was in six live citations and is dropped from all of
   them, replaced by the F test — which is what the sentence was reaching for. The file's own
   VERDICT says it "must not be restated until the operator names the intended denominator."
2. **C3 is mis-specified and marked so in the file.** Cite C4, never C3.
3. **The 1B band is off by one layer.** `ladder1b` is scored on **[5,13]** where the anchor's
   normalized L38–L92 rule gives **[6,13]**, and its ceiling reaches 81% of depth rather than 92%
   because the penultimate target sits inside the band at 16 layers. **Disclosed, not corrected**:
   the 1B ladder lenses were never retained, so the fix is a refit rather than a rescore. It
   affects the 1B column here.
4. **Absolute AUCs are not comparable across scales** (13 band layers at 410M, 9 at 1B, so the
   `persist` threshold differs). Orderings and variance decompositions are.
5. **No pre-registration.** Strength comes from the recomputation matching independently-quoted
   numbers, not from a rule fixed in advance.

## MEANING FOR THE PAPER

Supports §"The corpus effect is an interaction, not a level" and — importantly — supplies the
*reason* two other results came out as they did:

- it explains **why E48's registered CI had no power**: between-set heterogeneity at 91.3% swamps
  a 1.9% corpus main effect across five sets;
- it explains **why E31's twenty-one predictors all fail**: a scalar summary of "corpus quality"
  is predicting the 1.9%, and the effect lives in the 6.8%.

That makes E51 the hinge of the paper rather than a supporting table.

## PROVENANCE

| | |
|---|---|
| result | `results/e51_interaction_variance.json` |
| script | `experiments/t51_interaction_variance.py` |
| module | `bash repro/exp/e51_interaction.sh` |
| inputs | `results/ladder410/*.json`, `results/ladder1b/*.json` (30 files hashed) |
| tier | **A** |
| cost | free, seconds on CPU |

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e51_interaction_variance.json` | 29,478 | `2559e4017660a6a5` | `t51_interaction_variance.py` | INHERITED |
| `results/ladder410/ladder_Github_s0.json` | 15,784 | `6309f7636b8ef948` | `—` | RESCORED |
| `results/ladder410/ladder_Github_s1.json` | 15,766 | `aabd00d0ed32678f` | `—` | RESCORED |
| `results/ladder410/ladder_Github_s2.json` | 15,769 | `1feb27ab51615306` | `—` | RESCORED |
| `results/ladder410/ladder_Pile-CC_s0.json` | 10,904 | `dda406c8efc0ae46` | `—` | RESCORED |
| `results/ladder410/ladder_Pile-CC_s1.json` | 10,917 | `8c7335a765473ae8` | `—` | RESCORED |
| `results/ladder410/ladder_Pile-CC_s2.json` | 10,907 | `2720e604dc095ac3` | `—` | RESCORED |
| `results/ladder410/ladder_StackExchange_s0.json` | 15,777 | `8658d3f32c1f2010` | `—` | RESCORED |
| `results/ladder410/ladder_StackExchange_s1.json` | 15,789 | `14615d4d76b66c2e` | `—` | RESCORED |
| `results/ladder410/ladder_StackExchange_s2.json` | 15,780 | `4190fb2dd294b529` | `—` | RESCORED |
| `results/ladder410/ladder_USPTO_Backgrounds_s0.json` | 10,913 | `60246f23acb93238` | `—` | RESCORED |
| `results/ladder410/ladder_USPTO_Backgrounds_s1.json` | 10,901 | `7d3e4e6053174f5f` | `—` | RESCORED |
| `results/ladder410/ladder_USPTO_Backgrounds_s2.json` | 10,910 | `b950a4d857dfcf81` | `—` | RESCORED |
| `results/ladder410/ladder_Wikipedia_en_s0.json` | 15,636 | `48fceb1cf8b910d9` | `—` | RESCORED |
| `results/ladder410/ladder_Wikipedia_en_s1.json` | 15,555 | `c50a33e631f489b0` | `—` | RESCORED |
| `results/ladder410/ladder_Wikipedia_en_s2.json` | 10,911 | `83765d399663a6e9` | `—` | RESCORED |
| `results/ladder1b/tv_Github_s0.json` | 12,878 | `70f08c0c04fae13c` | `—` | EXPOSED |
| `results/ladder1b/tv_Github_s1.json` | 12,927 | `afbee942d86a6667` | `—` | EXPOSED |
| `results/ladder1b/tv_Github_s2.json` | 12,914 | `7b5d8b49c9e50247` | `—` | EXPOSED |
| `results/ladder1b/tv_Pile-CC_s0.json` | 12,843 | `cfae7085a49f4e09` | `—` | EXPOSED |
| `results/ladder1b/tv_Pile-CC_s1.json` | 12,894 | `a3b969ef0bcb1fba` | `—` | EXPOSED |
| `results/ladder1b/tv_Pile-CC_s2.json` | 12,909 | `6bbc136053598f58` | `—` | EXPOSED |
| `results/ladder1b/tv_StackExchange_s0.json` | 12,972 | `808c4f56ea2edb5b` | `—` | EXPOSED |
| `results/ladder1b/tv_StackExchange_s1.json` | 12,962 | `fd8eb1c2f96410e1` | `—` | EXPOSED |
| `results/ladder1b/tv_StackExchange_s2.json` | 12,958 | `764a95eb1b6cd2d4` | `—` | EXPOSED |
| `results/ladder1b/tv_USPTO_Backgrounds_s0.json` | 12,974 | `74a7317e4c90f91e` | `—` | EXPOSED |
| `results/ladder1b/tv_USPTO_Backgrounds_s1.json` | 12,976 | `88af3b2e98111782` | `—` | EXPOSED |
| `results/ladder1b/tv_USPTO_Backgrounds_s2.json` | 12,976 | `9a80132d9f3a7543` | `—` | EXPOSED |
| `results/ladder1b/tv_Wikipedia_en_s0.json` | 13,034 | `2afe54c57a5d16c0` | `—` | EXPOSED |
| `results/ladder1b/tv_Wikipedia_en_s1.json` | 13,083 | `07fdb704221ff92b` | `—` | EXPOSED |
| `results/ladder1b/tv_Wikipedia_en_s2.json` | 13,105 | `90b2485e1df2168f` | `—` | EXPOSED |

**Payload checksums** (content only, provenance block excluded):

* `e51_interaction_variance.json` — `6802b9c2d9ce2a76779e8c63cca0e9d1`

<!-- END GENERATED PROVENANCE -->
