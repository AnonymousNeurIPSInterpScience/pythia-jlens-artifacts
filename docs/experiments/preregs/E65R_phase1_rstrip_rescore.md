# E65R — E65 Phase 1 at the corrected readout, against a corrected reference

**PRE-REGISTERED before the rescore script was written or run.** Rescore only: **no operator is
refitted.** The ten 410M operators already exist at `results/e65_lenses/`. The stored Phase-1
outputs are **preserved unchanged**; this writes a new file.

---

## WHY

E65 Phase 1 is the programme's only measurement on the **training axis**, and it is stalled at
**UNCLEAR** because its control C1 failed:

> `e65_ckpt_readout_410m.json -> controls.C1_final_checkpoint_reproduces_e48`:
> `reference_value = 0.04546590892132372`, `final_checkpoint = 0.050229771388694645`,
> `abs_diff = 0.004763862467370927`, `tolerance = 3.7e-4`, `fires = false`.

Three facts, all read off stored files, make that failure worth re-examining:

1. **E65 Phase 1 is unstripped.** `experiments/t65_ckpt_readout.py:112-113` tokenises
   `it["prompt"]` with no `.rstrip()`. Every other axis in the programme was re-scored at the
   corrected readout (R1, R2, R4b, R8 and the whole `_rstrip` family). **E65 never was.**
   `docs/experiments/PROPOSED.md` item E34 tracks that backlog and does not list E65.
2. **The correction is not cosmetic here.** Measured on this battery: **157 of 541** admitted items
   (29.0%) read a *different token* under `.rstrip()` — multihop 83/93, order-ops 55/55,
   multilingual 19/107, poetry/typo/association 0. This reproduces the documented 157 figure.
3. **The C1 reference is not what its own label says.** The stored `reference` string names
   `e48_crossover_410m.json arms_admitted_mean['J|Pile-CC|s0'].persist`, but that field currently
   holds **0.04509159799199551**, not the stored `reference_value` **0.04546590892132372**
   (which is the three-seed mean). `docs/experiments/descriptions/E65.md` records this without
   resolution.

So C1 compared an unstripped measurement against a mislabelled unstripped reference. This
re-runs the comparison at the corrected readout against the corrected reference.

## DESIGN

For each of the 10 Phase-1 checkpoints of `pythia-410m-deduped`
(`0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000`), load the **already-fitted**
operator from `results/e65_lenses/lens_410m_step<STEP>_trainval.pt` and score **two arms** that
differ only in whether the prompt is `.rstrip()`ed before tokenisation:

* **arm U (unstripped)** — reproduces the stored run. This is the control.
* **arm S (stripped)** — the anchor rule, `src/anchor_evals.py` docstring lines 26-34.

Each arm scores the same 7 transports the original did: the fitted `J`, the logit lens (`J = I`),
and **5 layer-derangements** at the original seeds (`7000 + d`, `d = 0..4`), over the 5 admitted
sets, recording `persist` and `min`. Band `[9,21]`, `target_layer = -2`, `skip_first = 16`,
`K = [1,2,5,10,20,50,100]`, float32, CPU.

**Nothing is refitted.** Cost measured before registering: 44.7 s per checkpoint per arm on this
machine, all 10 revisions already in the local HF cache. ~15 min total, $0, no GPU, no download.

## PRIMARY

`C1_S`: at `step143000`, arm S's `jlens_persist` against the **corrected** reference
`e48_crossover_410m_rstrip.json -> arms_admitted_mean['J|Pile-CC|s0'].persist =
0.11054239969234914`, with tolerance `2 x` the stripped Pile-CC three-seed sample SD, computed in
this run from the s0/s1/s2 values in that same file.

## DECISION RULE — fixed before running

* **C1 CLEARS AT THE CORRECTED READOUT** — `C1_S` fires. E65 Phase 1's UNCLEAR was a readout
  artifact. The registered Phase-1 rule (`FLOOR EXISTS` / `NO FLOOR` / `NON-MONOTONE`) is then
  applied to arm S and its outcome reported.
* **C1 STILL FAILS** — `C1_S` does not fire. E65 Phase 1 remains **UNCLEAR** and the readout is not
  the explanation. Report the residual and stop; **do not re-run with new settings** (`CLAUDE.md`
  §"Outcomes are three-way").
* **UNINTERPRETABLE** — control C-U below fails, i.e. this harness does not reproduce the stored
  unstripped numbers. Then nothing in either arm is comparable to the stored file and both are
  discarded.

The Phase-1 rule is applied **only** in the first branch, and is quoted verbatim from
`E65_training_axis_floor.md`; it is not re-thresholded here.

## CONTROLS, each with the number it must produce

* **C-U — arm U reproduces the stored run.** For all 10 checkpoints, arm U's `jlens_persist`,
  `jlens_min`, `logit_persist`, `logit_min`, `shuf_persist_mean` and `shuf_persist_max` must match
  `results/e65_ckpt_readout_410m.json -> by_checkpoint.<rev>` to `abs_diff <= 1e-6`.
  **This control can fail**: the stored run scored on CUDA from a full-precision `J`, while this
  scores on CPU from the `.half()` operator persisted to `e65_lenses/`. A device change, a
  precision change and a code path change all sit between the two numbers.
  *Observed at registration time for step143000 only, during cost measurement:* arm U gave
  `0.050230` against the stored `0.050229771388694645`.
* **C-REF — the two references differ by more than the tolerance.** The stripped reference
  (0.11054239969234914) minus the unstripped one (0.04546590892132372) must exceed the C1
  tolerance. If the corrected reference were within tolerance of the old one, re-running C1 against
  it could not change the verdict and this experiment would be unable to fire.
* **C-DERANGE — the derangement floor is recomputed per checkpoint, per arm.** 5 draws at seeds
  `7000+d` in both arms; `shuf_draws` must contain 5 distinct values per cell at `step143000`.
  Importing the final checkpoint's floor would be wrong (the original C2's reasoning, retained).
* **C-ITEMS — the two arms differ on exactly the expected items.** The count of admitted items whose
  read token differs between arms must equal **157** of **541**. If it does not, the arms are not
  the two conventions.

## DECLARED BIAS

This registration expects **C1 STILL FAILS**, and this is recorded that before running. The stripped reference is
~2.4x the unstripped one (0.1105 vs 0.0455) and the stored unstripped final was 0.0502 — for arm S
to land within ~1.3e-3 of 0.1105, stripping would have to more than double the checkpoint read.
That is considered unlikely; if it happens it is to be treated as a bug report first
(`CLAUDE.md` §3 rule 4) rather than a result. **A null here is still a finding**: it would establish
that E65's C1 failure is not a readout artifact, which is currently unknown and is the reason this
runs.

## COST

Free. CPU, ~15 min, 10 revisions cached, 10 operators already on disk. No GPU, no download, no fit.

## OUTPUT

`results/e65r_phase1_rstrip_rescore.json`. The stored
`e65_ckpt_readout_410m.json`, `e65_ckpt_readout_1b.json` and `e65_ckpt_readout_410m_trainval.json`
are **not modified**.

## SCOPE LIMIT, STATED UP FRONT

**410M only.** Only 1 of the 10 1B operators was persisted (`lens_1b_step0_trainval.pt`), so the 1B
curve cannot be rescored without refitting 9 operators and downloading 20.2 GB of checkpoints. The
1B arm of E65 Phase 1 therefore remains at its stored UNCLEAR and is **out of scope here**.

---

## RESULT — **UNINTERPRETABLE** (2026-09-01)

`results/e65r_phase1_rstrip_rescore.json`. Ran 800 s on CPU, 10 checkpoints x 2 arms, no refit.

**The registered verdict is UNINTERPRETABLE and it stands.** Control **C-U failed**: arm U
reproduces the stored unstripped run to `max abs_diff = 7.252e-05` against the registered `1e-6`
bar, on `step16000.jlens_min`. Per `CLAUDE.md` §3 rule 7 the threshold is **not softened** and the
run is **not repeated with a looser one**. Three of the four controls fired
(`C_REF_differ`, `C_DERANGE_distinct`, `C_ITEMS_157` — the last at exactly 157 of 541).

## DISCLOSURE — the registered control was wider than the question it guarded

Post-result note, **changing no rule and no verdict**. C-U required six fields to agree at `1e-6`.
The PRIMARY depends on exactly one of them, `jlens_persist`. Measured, arm U vs the stored run
across all 10 checkpoints:

| field | max abs_diff | checkpoints over 1e-6 |
|---|---|---|
| `jlens_persist` | **3.166e-09** | 0 / 10 |
| `logit_persist` | 3.539e-09 | 0 / 10 |
| `logit_min` | 4.470e-09 | 0 / 10 |
| `shuf_persist_mean` | 1.304e-09 | 0 / 10 |
| `shuf_persist_max` | 3.725e-09 | 0 / 10 |
| **`jlens_min`** | **7.252e-05** | **1 / 10** (`step16000` only) |

So the CPU/fp16-operator rescore reproduces the stored CUDA/fp32 run essentially bit-exactly on
five fields and on the sixth at nine of ten checkpoints. **I scoped the control to six fields when
the PRIMARY uses one.** That is a control-design defect on my part, recorded rather than corrected
after the fact, in the same form as `OS1`'s C1 and `E65`'s own threshold flag.

**It does not hide a different answer, and this is the reason for disclosing rather than re-running.**
Had C-U been scoped to `jlens_persist` it would have fired at 3.166e-09, the PRIMARY would then have
been evaluated, and it does **not** fire either:

> `C1_S`: arm S final = **0.11742814**, corrected reference **0.11054240**,
> `abs_diff` = **6.886e-03** against `tolerance` = **1.322e-03**. **Would not fire.**

The registered branch under that counterfactual is `C1 STILL FAILS`, whose content is:
**the readout convention is not the explanation for E65 Phase 1's C1 failure.** Both roads reach
"E65 Phase 1 stays UNCLEAR". The operator rules on whether to re-register C-U scoped to the PRIMARY
and re-run; **that has not been done.**

## WHAT THE RUN MEASURED ANYWAY, reported as description, not as an adjudicated result

The corrected readout roughly **doubles** the final-checkpoint read (`0.050230` → `0.117428`) and
raises the free logit lens with it (`0.039072` → `0.084927`). The **qualitative training-axis
picture does not change**: under both conventions only `step143000` clears both its own derangement
floor and the logit lens, so `t*` is the last measured checkpoint in each. At **8 of the 10**
checkpoints the *unfitted* logit lens reads **above** the fitted J under the corrected readout.
None of this is adjudicated by a fired rule and none of it should be cited as a result.

## SCOPE, restated

410M only. 1 of 10 1B operators was persisted, so the 1B arm of E65 Phase 1 is untouched and remains
at its stored UNCLEAR.
