# E48b — n-gram containment against the Pythia stream: is the OOD designation earned?

**Verdict: yes, and with a calibrated scale. This is the instrument the whole exposure axis
stands on.**

---

## QUESTION

Every claim on the exposure axis presupposes that "the model never saw this text" is a
**measurement**. Pythia publishes its deduplicated training stream, so it can be. This builds that
measurement and asks whether it discriminates.

## PRE-REGISTRATION

Section "E35" of `docs/experiments/preregs/superseded/PREREG_E36_QLADDER.md`, which fixes `k` by a rule rather
than by inspection.

## DESIGN

An n-gram containment index against the **actual** deduplicated stream, built by **inverting the
query**: index the candidate corpora (millions of k-grams) and stream the 300B-token corpus past
them, rather than indexing the stream. That is what makes it tractable on commodity hardware —
and note that **this stage is network-bound, so a GPU is useless for it**.

Twenty shards, k ∈ {8, 16, 32}, reported at full 20/20 coverage. Candidate rungs span four
designed tiers: in-stream Pile components, a same-source-family control, a temporal-exclusion
probe, a trap, and the three genuinely-absent corpora.

## CONTROLS, and the number each produced

| control | required | produced | fires |
|---|---|---|---|
| **C1** uniform-random-token floor | ≈ 0 at every k and coverage | **exactly 0.000000** | **yes** |
| **C2** per-corpus k-gram counts reproduce the reference | match | match | **yes** |
| **C3** reproduces the stored 5-shard partial result | exact | **0.000e+00** | **yes** |

## RESULT

At k = 32, full 20/20 shard coverage:

| rung | containment | reading |
|---|---|---|
| in-stream Pile components | **0.92 – 0.93** | present |
| `CONTROL_PubMed_2023` (same source family, post-cutoff) | **0.674** | **this is what calibrates the scale** |
| `OOD_Wikipedia_2023` (post-cutoff Wikipedia) | **0.24** | **not OOD** — see below |
| `TRAP_FineWeb` | 0.05 | mostly absent |
| `OOD_News_2024` | 0.00075 | absent |
| `OOD_CommonPile` | 0.00125 | absent |
| `OOD_arXiv_2023` | 0.000086 | absent |
| uniform random tokens | **0.000000** | the floor |

`results/e48b_exposure_growth.json`

Two findings here generalise beyond this paper:

1. **Containment at fixed k conflates exposure with recurrence.** On a 5% sample Github scores
   0.374 and Wikipedia 0.071 at k=32, though both are Pile components and equally in-distribution —
   formulaic code recurs, a unique sentence does not. This is the same distinction the memorization
   literature draws between *membership* and *extractable verbatim recurrence*.
2. **The discriminator is the growth rate, not the level.** Corpora present in the stream are found
   roughly linearly as coverage rises; genuinely absent corpora stay pinned at zero. By that test,
   **post-cutoff Wikipedia is not out of distribution** (0.24, growing like the in-stream corpora)
   because revisions inherit most of their text from versions that *were* in the Pile.
   **Document-level temporal exclusion does not give textual novelty** — the trap most naive OOD
   constructions fall into.

## VERDICT

**THE OOD DESIGNATION IS EARNED.** Three rungs sit three to four orders of magnitude below the
in-stream tier, against a random floor of exactly zero and a same-family control at 0.67.

## FLAGGED DELTAS

1. **Containment is formatting-sensitive, and this bounds what can be claimed.** Token k-grams are
   destroyed by re-wrapping: re-flowing identical text to a different line width leaves only
   **0.14%–2.4%** of k=32 n-grams. So **any value in the 0.001–0.03 band is uninterpretable as
   absence.** The three OOD rungs sit below that band and PubMed-2023 sits at 0.67, so the
   classification holds — but the correct phrasing is **"no match found"**, never "provably
   non-member". The paper states this as Limitation 3.
2. **CommonPile is the least independent rung.** It is 2400/2400 arXiv abstracts and arXiv is a
   Pile component. Its containment (0.00125) sits below the reflow floor (0.0014 for a single-line
   reflow), so absence is not refuted — but two of three "register-diverse" absent rungs are the
   same scientific domain.
3. **This is a network-bound job.** Do not rent a GPU for it.

## MEANING FOR THE PAPER

Supports §"Exposure is measurable, and mostly not what people assume". Its two generalisable
findings — recurrence ≠ membership, and growth rate ≠ level — are arguably reusable outside this
paper and are worth keeping prominent for a venue focused on measurement validity.

## PROVENANCE

| | |
|---|---|
| result | `results/e48b_exposure_growth.json` |
| script | `experiments/t48b_exposure_growth.py` (index built by `experiments/t35_containment.py`) |
| module | `bash repro/exp/e48b_exposure.sh` |
| tier | **A** — 21 inputs hashed |
| cost | free, ~15 min CPU on the merged index; the index build itself is the expensive part |

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e48b_exposure_growth.json` | 32,715 | `30ba5ecd395f1cdf` | `t48b_exposure_growth.py` | INHERITED |

**Payload checksums** (content only, provenance block excluded):

* `e48b_exposure_growth.json` — `a3904599356aee3c0265c5c1b7b54240`

<!-- END GENERATED PROVENANCE -->
