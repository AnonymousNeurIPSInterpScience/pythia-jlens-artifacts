# EFFRANK1 — which "effective rank" does the paper print, and do the two stored values conflict?

**PRE-REGISTERED before the adjudication script was written or run.** Recomputation only: no
operator is fitted, no model is trained, no threshold in any prior experiment is changed.

---

## WHY

Two stored records report a field named `eff_rank` for the **same matrix** — the 1310 concept-token
rows of `W_U` for `pythia-410m-deduped` at its final checkpoint — and disagree by a factor of ~1.96:

| source | field | value |
|---|---|---|
| `results/e38_jgeometry.json -> E39.410m.eff_rank_I` | `eff_rank_I` | **371** |
| `results/e65_ckpt_geometry_410m.json -> by_checkpoint.step143000.eff_rank` | `eff_rank` | **725.887939453125** |

Mean pairwise cosine agrees to 9 decimal places across the two files
(`0.030666086822748184` vs `0.030666090548038483`), and both record
`n_concept_tokens = 1310`, so the *inputs* are not in dispute. Only the statistic is.

`paper/audit_paper_5pp.tex:693` prints **"The effective ranks are 1, 1, 371 and 508."** That
sentence carries S1's mechanism. `docs/experiments/descriptions/E65.md` records both numbers and
states verbatim: *"Both effective-rank numbers are recorded; neither is adjudicated here."*

**This registration does not assume 371 is correct.**

## DESIGN

Load `W_U` for `pythia-70m-deduped`, `pythia-160m-deduped`, `pythia-410m-deduped`,
`pythia-1b-deduped` (the four models the paper's sentence covers), take the same 1310 concept-token
rows via `concept_token_ids`, and compute **both** stored definitions plus three reference
statistics on **one** singular-value spectrum per model, in one process, at float32:

| id | definition | source of the definition |
|---|---|---|
| `er_energy90` | `#{i : cumsum(σ²)_i / Σσ² < 0.90} + 1` | `experiments/t38_jgeometry.py:167-169` |
| `er_entropy` | `exp(−Σ p log p)`, `p = σ / Σσ` | `experiments/t65_ckpt_geometry.py:73-75` |
| `er_energy99` | as `er_energy90` at 0.99 | reference only |
| `er_stable` | `‖W‖_F² / σ₁²` | reference only |
| `sigma1_energy_share` | `σ₁² / Σσ²` | reference only |

CPU, float32, no TF32. All four models are already in the local HF cache.

## PRIMARY

For each model, whether `er_energy90` reproduces the stored `e38_jgeometry.json` value and whether
`er_entropy` reproduces the stored `e65_ckpt_geometry_410m.json` value, on the same spectrum.

## DECISION RULE — fixed before running

* **NAME COLLISION** — both recomputations reproduce their own stored value to the tolerance below.
  The two numbers are **different statistics of the same matrix**, both correct, and the defect is
  that they share the field name `eff_rank`. The paper's sentence is then true *of the definition
  E38 used* and must name that definition.
* **ONE IS WRONG** — exactly one recomputation reproduces its stored value. The other stored number
  is a defect; name it, and state what depends on it.
* **BOTH WRONG** — neither reproduces. Stop and flag for adjudication; the geometry claim is
  unsupported and S1's mechanism sentence must be withdrawn pending a re-measurement.

**Tolerance.** `er_energy90` is an integer: **exact equality** required. `er_entropy` is continuous:
`abs_diff <= 1e-3` against the stored `725.887939453125`.

## CONTROLS, each with the number it must produce

* **C1 — the inputs are the same object.** Recomputed `mean_cos` must reproduce
  `e38_jgeometry.json -> E39.<model>.mean_cos_I` to `abs_diff <= 1e-6` at all four models, and
  recomputed `n_concept_tokens` must equal **1310** at all four. If C1 fails the two files are not
  describing the same matrix and nothing below is interpretable.
* **C2 — the two definitions are not accidentally equal.** On the same 410M spectrum,
  `er_entropy - er_energy90` must be **> 1**. A control that cannot fail otherwise: if the two
  definitions coincided numerically there would be no disagreement to adjudicate.
* **C3 — the rank-one claim is a property of the definition, not of the matrix.** At 70M, where
  `e38_jgeometry.json` stores `eff_rank_I = 1`, `er_entropy` must be **> 1**. This is the control
  that can embarrass the paper: if `er_entropy` also returned 1 at 70M, the choice of definition
  would not matter and the paper's sentence would be robust. **Predicted to fail the "doesn't
  matter" reading**, i.e. predicted `er_entropy(70M) > 1`.
* **C4 — spectrum integrity.** `Σσ² > 0` and `σ` sorted descending at every model.

## DECLARED BIAS

This registration expects **NAME COLLISION**, because Both implementations have been read both implementations and they are textbook-distinct
(90% energy participation vs. Roy–Vetterli entropy rank). The outcome that would surprise me is
`ONE IS WRONG`. It is recorded that The code has already been read the code, so this registration is not blind to the
mechanism — it is blind only to the numbers, which have not been computed.

## COST

Free. CPU, ~1 min, four models already cached. No GPU, no download, no fit.

## OUTPUT

`results/effrank1_adjudication.json`
