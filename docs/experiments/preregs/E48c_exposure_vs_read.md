# E48c — does exposure of the fitting corpus ORDER the read?

**Verdict: no. Spearman +0.000 across 10862× in exposure. Bad for the estimator framing, and it
is the substantive content of the whole exposure axis.**

---

## QUESTION

E48's registered CI returned NOT REACHED on an interval that could not have resolved the in-stream
advantage either. That leaves the positive question unanswered: **does the read order with
measured exposure at all?** This asks it on the axis that has power — the seed block, which is
where E28/E33's effects are measured — rather than the eval set.

## PRE-REGISTRATION

**None as a separate registration.** It is a re-analysis of E48's and E48b's stored outputs on a
different replication axis. The file is explicit that this is not a licence to fish:

> `"not_a_new_predictor"`: tests ONE pre-existing, externally-measured variable that E48's own rung
> classification is already built on; proposes no new scalar and does not target the corpus × concept
> interaction.

That matters because the programme carries a standing prohibition on inventing predictor #22.
Containment is not a new scalar — it is the variable E48's design already used to *define* its
rungs.

## DESIGN

Eight fitting corpora, `persist`, band [9,21], containment at **k = 32** over **20/20 shards** of
the Pythia deduplicated stream. Seed block is the replication unit. Read AUC and containment are
both taken from stored files (E48 and E48b), so no new forward pass is involved.

## CONTROLS, and the number each produced

| control | required | produced | fires |
|---|---|---|---|
| **C1** containment spans orders of magnitude | ≥ 100× | **10861.8×** (max/min) | **yes** |
| **C2** seed SD small against the between-corpus spread | ratio ≪ 1 | max seed SD **0.00103** vs between-corpus range **0.02172** → ratio **0.047** | **yes** |
| **C3** Github leverage reported **both ways** | — | Spearman **+0.000** with Github, **+0.500** without | **yes** — and see deltas |

## RESULT

| fitting corpus | tier | containment k=32 | read (persist) | vs logit |
|---|---|---|---|---|
| USPTO Backgrounds | in-stream | 0.9268 | 0.04962 | +0.02118 |
| StackExchange | in-stream | 0.9180 | 0.04573 | +0.01729 |
| Pile-CC | in-stream | 0.9276 | 0.04547 | +0.01703 |
| **arXiv 2023** | **measured absent** | **0.000086** | **0.04106** | **+0.01262** |
| **CommonPile** | **measured absent** | **0.00125** | **0.03824** | **+0.00980** |
| Wikipedia (en) | in-stream | 0.9216 | 0.03821 | +0.00977 |
| **News 2024** | **measured absent** | **0.00075** | **0.03376** | **+0.00532** |
| Github | in-stream | **0.9340** | **0.02790** | **−0.00054** |

logit-lens constant: **0.028440**.

- Pearson (all 8) **+0.266**, **Spearman +0.000**.
- The **most**-contained corpus is the **worst** reader.
- A lens fitted on 2023 arXiv — containment 0.0001, indistinguishable from random tokens — reads
  **above** one fitted on in-stream Wikipedia, at **t = +5.75** on the seed axis.
- All three absent-corpus operators clear their own derangement floor, so they are structured, not
  collapsed.

`results/e48c_exposure_vs_read.json`

## VERDICT

**EXPOSURE DOES NOT ORDER THE READ.**

The honest counterweight, stated in the file rather than left to a reader: with the **corpus** as
the replication unit the four in-stream rungs do average above the three absent ones, by
**+0.00707** (t = 2.22, df = 5.0) — suggestive, **not significant** at n=4 vs 3, and smaller than
the **0.0114** spread *within* the in-stream tier (0.0217 if Github is counted).

The claim is that exposure does not *order* the read, which the rank correlation supports. It is
**not** that exposure has no effect of any size, which n=8 cannot establish.

## FLAGGED DELTAS

1. **The headline number is Github-leveraged, and the paper's abstract does not say so.** The file
   states the claim "only in the form that survives both" — +0.000 with Github, **+0.500 without**.
   The abstract currently reports "+0.000" unqualified. **Spearman +0.5 at n=7 is p ≈ 0.25, so the
   null still stands** — but the paper should carry the both-ways form the results file already
   carries. *This is a paper-fidelity gap, not a data problem.*
2. **Github is the leverage point in seven analyses.** Four broadly similar English-prose corpora
   plus one extreme outlier means dropping Github removes nearly all corpus-axis variance. No
   corpus-axis result at n=5 can survive leave-one-out. That is a design property, not bad luck.
3. **Containment is formatting-sensitive.** See E48b: any value in 0.001–0.03 is uninterpretable as
   absence. The three OOD rungs sit below that floor and the PubMed-2023 control at 0.67 calibrates
   the scale, so the classification is sound — but the correct phrasing is "no match found", never
   "provably non-member".
4. **410M only.**

## MEANING FOR THE PAPER

Carries the abstract's clause (1) and §"Exposure axis I"'s substantive result. It is the reason
the paper can say *what the model was trained on is not the place to look* — which is the
practitioner-facing recommendation, and the most directly useful sentence in the paper.

## PROVENANCE

| | |
|---|---|
| result | `results/e48c_exposure_vs_read.json` |
| script | `experiments/t48c_exposure_vs_read.py` |
| module | `bash repro/exp/e48c_exposure_read.sh` |
| inputs | `e48_crossover_410m.json`, `e48b_exposure_growth.json` (2 hashed) |
| tier | **A** |
| cost | free |

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e48c_exposure_vs_read.json` | 9,875 | `ff344efc91604138` | `t48c_exposure_vs_read.py` | INHERITED |

**Payload checksums** (content only, provenance block excluded):

* `e48c_exposure_vs_read.json` — `77af14c08520331c43a36332f4d18beb`

<!-- END GENERATED PROVENANCE -->
