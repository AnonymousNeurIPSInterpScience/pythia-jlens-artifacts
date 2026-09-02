# E48 — the S3 crossover on the fit axis: does fitting on absent text lose to not fitting?

**Verdict: NOT REACHED — and the registered test had no power to return its positive branch.
Both facts are reported. Mixed for the thesis, honest for the paper.**

---

## QUESTION

The estimator framing's central prediction: if the O(1) corpus-bias term is what matters, an
operator fitted on a corpus the model never saw should be worth less than one fitted on a corpus
it did — and at sufficient shift, worth less than not fitting at all. E48b made "never saw"
measurable. This spends that.

## PRE-REGISTRATION

`docs/experiments/preregs/superseded/PREREG_E48_CROSSOVER.md` + **Amendment 1**, commits `408fa5c` / `5869b63`,
fixed **before any OOD lens was fitted**. The results file carries the rule under
`rule_verbatim` and its status as `"PRE-REGISTERED — rule fixed before any OOD lens existed"`.

The rule has two clauses. Clause (a) requires the in-stream rungs' `(J^P − logit)` interval to sit
strictly **positive**; clause (b) requires an OOD rung's to sit strictly **negative** for
CONFIRMED.

## DESIGN

`pythia-410m-deduped`, band **[9,21]**, `persist` adjudication, five admitted concept sets,
`N=200`, **three disjoint seed blocks per corpus**, CPU.

Nine `J^P` fitted on three corpora **measured absent** from the Pythia stream — 2024 news, 2023
arXiv, and an openly-licensed scientific corpus — and scored against the logit lens and against a
layer-deranged `J^shuf` on **one shared activation cache**, so every arm differs only in the
transport. Five in-stream Pile components run as the comparison tier.

**Prerequisite gate.** `e48_competence_gate_410m.json` establishes the model is competent on every
fitting corpus before any read is scored: per-token cross-entropy 2.78–2.83 nats on the three OOD
corpora, *inside* the in-stream range of 1.06–2.93. A negative read therefore could not have been
attributed to model confusion.

## CONTROLS, and the number each produced

| control | required | produced | fires |
|---|---|---|---|
| **C0** reproduces E33 v2 on the shared cache | exact | max abs diff **1.9e-08** across all 12 (transport × aggregation × set) cells; several exactly **0.0** | **yes** |
| **derangement floor** `J^P` clears its own layer-deranged control | majority | **120 of 120** independent draws (8 corpora × 3 seed blocks × 5 derangements) under `persist`; **0/120** the other way | **yes** |
| **derangement floor under `min`** | — | the shuffled operator **beats** the real one on **103 of 120** draws *against the corpus-mean operator* (104/120 paired by seed) and 7 of 8 corpora | *fires as a disproof of `min`, not of the operator* |
| **norm-matched random transport** | at the floor | reads **exactly 0.000000** under `persist` on all eight corpora | **yes** |
| **competence gate** | OOD CE inside in-stream range | 2.78–2.83 vs 1.06–2.93 | **yes** |

## RESULT

No OOD rung's `(J^P − logit)` interval goes strictly negative. **And no in-stream rung's goes
strictly positive either** — so clause (a) failed too, and the registered test could not have
returned CONFIRMED under any data.

The reason is E51: with the eval **set** as the replication unit, between-set heterogeneity is
91.3% of variance against a corpus effect of 1.9%, and five sets cannot resolve it.

`results/e48_crossover_410m.json`, `results/e48_competence_gate_410m.json`

## VERDICT

**NOT REACHED**, reported exactly as registered. The file's own wording, which is binding:

> Reported straight as a positive-framed negative: the bias term's practical magnitude is below
> the estimator worst case; the fitted operator tolerates shift the theory permits it to fail on.
> Publishable exactly as pre-registered and **MUST NOT be re-described as a weaker crossover**.

## FLAGGED DELTAS

1. **The test was underpowered, and this is stated rather than banked.** Clause (a) failing on all
   four in-stream rungs means NOT REACHED is as much a statement about an n=5-eval-set interval as
   about shift. The substantive content of this arm is in **E48c**, which uses the seed block as
   the replication unit — the axis that has power.
2. **A power upgrade exists and is unrun.** The outcome grid is per eval **set** — five cells.
   A per-token rescore gives ~6550. Confounder to handle: per-token log-frequency-in-P correlates
   with both axes, so partial it out or stratify within frequency deciles.
3. **One OOD rung is weaker than the panel implies.** The openly-licensed corpus (CommonPile) is
   **2400/2400 arXiv abstracts**, and arXiv is a Pile component, so two of three "register-diverse"
   absent rungs are the same scientific domain. Its containment does sit below the reflow floor, so
   absence is not refuted, but it is the least independent rung. The pre-registration's amendment
   anticipated exactly this by requiring the register-near, exposure-far rung (2024 news) to carry
   the result, and a declared sensitivity analysis dropping the contested corpus changes no
   conclusion.
4. **`min` prefers a broken operator.** 103/120 *against the corpus-mean operator* (104/120 paired by seed) is not a finding about `J^shuf`; it is a disproof
   of the aggregation statistic, and it is why this programme adjudicates on `persist`. **Never
   restate it as "the shuffled Jacobian works"** — `repro/lib/banned_claims.py` enforces this.
5. **410M only.**

## MEANING FOR THE PAPER

Supports §"Exposure axis I" and Limitation 2. The honest framing is that the *registered* test
returned a null it could not have failed to return, and the *seed-axis* analysis in E48c is where
the resolution actually is — which the paper says, in the limitations, first rather than last.

## PROVENANCE

| | |
|---|---|
| results | `results/e48_crossover_410m.json`, `results/e48_competence_gate_410m.json` |
| scripts | `experiments/t48_crossover.py`, `experiments/t48_competence_gate.py` |
| modules | `bash repro/exp/e48_gate.sh` → `bash repro/exp/e48_crossover.sh` (order matters) |
| tier | **A** — both verify; 26 and 11 inputs hashed respectively |
| cost | free; ~40 min + ~25 min CPU |

> **Taxonomy correction (2026-08-15):** `RESULTS_TAXONOMY.md` §2 lists both of these as Tier B.
> They are Tier **A** — both carry verifying provenance blocks. Corrected in the taxonomy.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e48_crossover_410m.json` | 115,278 | `cb00c6c92899012c` | `t48_crossover.py` | EXPOSED |
| `results/e48_competence_gate_410m.json` | 19,534 | `37b9f18a33810e33` | `t48_competence_gate.py` | IMMUNE |

**Payload checksums** (content only, provenance block excluded):

* `e48_crossover_410m.json` — `5efcb17b7912b358119f7bf899cbcd45`
* `e48_competence_gate_410m.json` — `682a6cbd921db6dd8f9228d287e422bc`

<!-- END GENERATED PROVENANCE -->
