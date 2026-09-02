# docs/experiments/preregs/ — one document per experiment, pre-registration through verdict

This directory replaces two things that used to be separate and drifted apart: the
pre-registrations (which said what would count as an answer) and the write-ups (which said what
the answer was). Keeping them apart is how a programme ends up with a registered rule nobody
re-reads and a headline nobody can grade. **Each file here carries both, plus everything that
happened in between.**

The raw pre-registrations are preserved verbatim in [`../archive/prereg/`](../archive/prereg/).
They are timestamps and their bodies are never edited. Where a document below quotes a registered
rule, it quotes that file.

## The shape every document has

| section | what goes in it |
|---|---|
| **QUESTION** | what this settles, and why it mattered at the time |
| **PRE-REGISTRATION** | the decision rule **verbatim**, with the date and commit at which it was fixed. If there was no pre-registration, the section says so — that is a fact about the result's strength |
| **DESIGN** | model, corpora, N, band, aggregation, what varies and what is held fixed |
| **CONTROLS** | each control, **the number it actually produced**, and **the number that would have made it fail**. A control you cannot name a number for did not fire; a control you cannot name a *failing* value for had no power and must be labelled a **RECORDER**. Audited repo-wide by `tools/r4e_control_power.py` → `results/r4e_control_power.json` (R4e): 36 of 106 control rows carry no gate, were never evaluated, or are unfalsifiable by construction, and 6 controls named in a spec have no corresponding key in any of that spec's results files — including `PREREG_E36_QLADDER.md`'s **C4**, which the pre-registration itself calls load-bearing (*"without C4 the entire ladder could be a prefix-length artifact"*) |
| **RESULT** | the numbers, each with its file |
| **VERDICT** | graded against the registered rule, in the registered vocabulary. Never softened |
| **FLAGGED DELTAS** | every known defect, disclosure, amendment and open question. This is the section a reviewer should read first |
| **MEANING FOR THE PAPER** | which claim it supports, and what it does *not* license |
| **PROVENANCE** | results file, script, tier, and how to re-run it |

## The experiments

Ordered by the axis they belong to, which is how the paper is organised.

### Axis 1 — exposure: has the *model* seen this text?

| doc | question | verdict |
|---|---|---|
| [`E48b_containment.md`](preregs/E48b_containment.md) | can we *measure* membership in the Pythia stream? | the OOD designation is earned |
| [`E48_crossover.md`](preregs/E48_crossover.md) | does fitting on absent text make the lens worse than not fitting? | **NOT REACHED** (and underpowered — stated) |
| [`E48c_exposure_vs_read.md`](preregs/E48c_exposure_vs_read.md) | does exposure *order* the read? | **no**, Spearman +0.000 |
| [`E36_qladder.md`](preregs/E36_qladder.md) | does shifting the *read* distribution favour the free baseline? | **REJECT S3** |

### Axis 2 — operator-relative: was the *lens* fitted on this text?

| doc | question | verdict |
|---|---|---|
| [`E52_factorial.md`](preregs/E52_factorial.md) | does reading inside the operator's own fitting distribution help? | yes, and **it is small** |

### Axis 3 — identity: *which* corpus is it?

| doc | question | verdict |
|---|---|---|
| [`E28_read_ladder.md`](preregs/E28_read_ladder.md) | does corpus choice or sample size govern the read? | corpus, decisively; `N` barely at all |
| [`E33_logit_baseline.md`](preregs/E33_logit_baseline.md) | where does the unfitted lens sit among the corpora? | **inside** the spread |
| [`E51_interaction.md`](preregs/E51_interaction.md) | is the corpus effect a level or an interaction? | **an interaction** |
| [`E31_predictor_bakeoff.md`](preregs/E31_predictor_bakeoff.md) | can anything predict which corpus is good? | **no** — 21 predictors fail |

### Supporting

| doc | question | verdict |
|---|---|---|
| [`E38_geometry.md`](preregs/E38_geometry.md) | why do small models fail? | concept-token unembedding is rank-1 below 410M |

## Reading order for a reviewer

1. `E33` — the effect, in its simplest form.
2. `E51` — what kind of effect it is. This is the load-bearing structural claim.
3. `E48c` and `E36` — the two shift axes, both negative, both pre-registered.
4. `E52` — the third axis, positive but small, reported as directional.
5. `E31` — the systematic negative.
6. `E28` and `E48b` — the instruments the four above stand on.
