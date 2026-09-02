# E33 — the missing J = I arm: where does the unfitted lens sit among the corpora?

**Verdict: INSIDE the spread. One of five fitting corpora produces an operator no better than not
fitting at all. Good for the thesis — this is the effect in its simplest form.**

---

## QUESTION

E28's read ladder compared fitting corpora against each other and never against the free baseline.
Without a `J = I` arm, "corpus A beats corpus B" says nothing about whether *fitting* was worth
anything. This adds the arm on the same forward passes.

## PRE-REGISTRATION

**None.** E33 predates the programme's pre-registration discipline for this class of question. Its
strength comes from three things instead: it is a rescore of *stored* ladder artifacts, so the
comparison is on identical activations; it has a named control that fires; and its numbers were
independently reproduced to 1.9e-08 by E48's C0.

## DESIGN

`pythia-410m-deduped`, band **[9,21]**, `persist`, five admitted concept sets, `N ≥ 75` averaged,
three disjoint seed blocks. Five fitting corpora against `J = I` and against a layer-deranged
`J^shuf`, all on one shared activation cache.

## CONTROLS, and the number each produced

| control | required | produced | fires |
|---|---|---|---|
| **derangement** — `J^P` beats `J^shuf`, sampled as a **random** derangement | ≥ 5 of 6 sets | fires in v2 | **yes (v2 only)** |
| **norm-matched random transport** | at the floor | **0.00000** | **yes** |

**Cite v2, never v1.** v1's control **did not fire** (4/6) because it used a cyclic shift by one
layer — the weakest member of the derangement family, since adjacent-layer Jacobians are the most
similar pair in the band. The decision rule was unchanged; only the mis-specified control was
fixed. v1 is retained in `results/` on purpose.

The 410M derangement floor was later reproduced properly by E48: **120 of 120** independent draws
under `persist`.

## RESULT

| corpus | Y∞ | seed SD | vs logit | t (df=2) |
|---|---|---|---|---|
| USPTO Backgrounds | **0.04974** | 0.00050 | **+75%** | **+73.37** |
| Pile-CC | 0.04572 | 0.00018 | +61% | **+163.03** |
| StackExchange | 0.04568 | 0.00030 | +61% | **+100.19** |
| Wikipedia (en) | 0.03860 | 0.00050 | +36% | **+35.39** |
| **logit lens (J = I)** | **0.02844** | — | — | — |
| Github | 0.02808 | 0.00037 | −1.3% | **−1.65** |
| norm-matched random | 0.00000 | — | — | floor |

`|t| > 4.303` is p<0.05 at df=2. **Github's −1.65 is inside that**, i.e. statistically
indistinguishable from not fitting at all.

**9 of 10 corpus pairs separate at p<0.05** (Welch, |t| 12.0–73.5), spread **58×** the pooled seed
SD. The one non-separating pair is instructive: StackExchange and Pile-CC differ by **61%** in
dispersion and **0.09%** in performance (t = −0.20), so dispersion is not injective onto
performance.

`results/e33_logit_baseline_410m_v2.json`, `results/e33b_tstats_410m.json`

## VERDICT

**I INSIDE THE CORPUS SPREAD.** The lens's whole value at 410M runs from **zero to +75%** on
fitting-corpus choice alone.

Stated no more strongly than the data allows: Github's operator is not *worse* than not fitting;
it is **statistically indistinguishable** from it.

## FLAGGED DELTAS

1. **df = 2.** The three seed-block means are not persisted by `experiments/t33_logit_baseline.py`, so they
   cannot be recovered from `e33_logit_baseline_410m_v2.json` alone — only from
   `results/ladder410/*.json`. **E48's hierarchical bootstrap supersedes this as the instrument**;
   E33's t statistics are the readable form, not the strongest form.
2. **The t statistics lived only in prose for weeks.** `Github t = −1.65` was load-bearing for a
   pre-registered clause and appeared in **no results file**. E33b (`experiments/t33b_store_tstats.py`) exists
   solely to store them. This is one of the two failures that produced `src/provenance.py`.
3. **Tier B**, not A: v2 predates the provenance stamper. E33b, which stores its t statistics, is
   Tier A.
4. **This is NOT the S3 crossover.** There is no shift axis here and all five corpora are Pile
   components, so the model is in-distribution throughout. The design *cannot* produce a crossover.
   The crossover-shaped fact — that the logit lens sits inside the spread — lives on the **corpus**
   axis, which is not the axis S3 names.
5. **410M.** E28's 1B ladder replicates the ordering only partially (Spearman +0.700).

## MEANING FOR THE PAPER

Supplies §"The fitting corpus decides the read"'s headline table and the abstract's clause (3)
opening: *"the same operator is worth +75% fitted on one Pile component and statistically
indistinguishable from not fitting at all on another."*

**It does not license** a crossover claim of any kind.

## PROVENANCE

| | |
|---|---|
| results | `results/e33_logit_baseline_410m_v2.json` (**cite v2**), `results/e33b_tstats_410m.json` |
| scripts | `experiments/t33_logit_baseline.py`, `experiments/t33b_store_tstats.py` |
| module | `bash repro/exp/e33b_tstats.sh` (t statistics only) |
| tier | **B** (v2) / **A** (E33b) |
| cost | free |

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e33_logit_baseline_410m_v2.json` | 5,912 | `48ee37f6526b7891` | `—` | RESCORED |
| `results/e33b_tstats_410m.json` | 6,372 | `38b5286aba050924` | `t33b_store_tstats.py` | INHERITED |
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

**Payload checksums** (content only, provenance block excluded):

* `e33b_tstats_410m.json` — `db344d95ee0c8c1b208316bcc44fdbf3`

<!-- END GENERATED PROVENANCE -->
