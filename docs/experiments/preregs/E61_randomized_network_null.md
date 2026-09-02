# E61 — the randomized-network null ("dead salmon"): how much read survives with no computation?

**Output:** `results/e61_randomized_null_410m.json`

**PRE-REGISTERED 2026-08-16, before the run. NOT YET RUN — awaiting a GPU box.**

---

## WHY

Every control in this programme randomizes the **operator**: the layer-deranged $J^{\mathrm{shuf}}$
(120 draws, E48/E54), the norm-matched random transport (E48, exactly 0.000000), the random-$k$
ablation (E47). **None randomizes the model.** So no number here answers: *how much of the read
survives when there is no computation to read?*

Méloux et al. (arXiv:2512.18792) show that probing, SAEs, circuits and causal abstraction all
produce plausible-looking explanations on **weight-randomized** networks, and argue the fix is a
hypothesis test against a randomized-computation null. For a workshop whose call names
*"falsifiability and experimental designs that distinguish mechanisms from artifacts"*, this is the
cheapest addition with the highest venue fit.

It is **not predictor #22**. It targets the floor of the measurement, not the corpus effect, so
`HANDOFF.md` §0.5's standing prohibition does not apply.

## DESIGN

`experiments/t61_randomized_null.py --device cuda`

- Take `pythia-410m-deduped`. Re-initialise **every transformer block's weights** from the model's
  own initialiser, holding the **embedding and unembedding fixed**. Holding $W_U$ fixed is the
  whole point: the read is $\mathrm{unembed}(J h)$, and E38 already shows $W_U$'s concept-token
  geometry is what fails below 410M. Randomising $W_U$ too would answer a different question.
- Fit $J$ on the randomized model with the identical fitter and settings: `trainval.py`, $N{=}200$,
  `dim_batch=128`, `max_seq_len=128`, `skip_first=16`, `target_layer=-2`, fp32, no centering, TF32
  off. **Three fit corpora** (USPTO — the best real operator; Github — the worst; Pile-CC — a
  middle) **× 3 seed blocks = 9 operators.**
- Score on the same 449-item admitted battery, band $[9,21]$, both aggregations, at $Q0$ (no read
  context) so nothing but the operator differs from the real arm.
- Arms per corpus: randomized-model $J$, randomized-model logit lens ($J{=}I$ on the randomized
  model), and the stored real-model values for the same corpus and seed.

## PRIMARY

**The randomized-model J-lens read AUC under `persist`, against the real-model logit-lens constant
(0.02844) and against the layer-deranged floor.**

## DECISION RULE — fixed before the run

- **FLOOR IS LOW** — the randomized-model J-lens scores **below the real-model logit lens**
  (< 0.02844) on all three corpora. The read reflects computation; the measurement has a floor
  where it should. This is the expected and reassuring outcome and it is worth one sentence in the
  paper plus a table row.
- **FLOOR IS HIGH** — the randomized-model J-lens scores **at or above** the real-model logit lens
  on any corpus. Then a substantial part of what the read measures is $W_U$ geometry plus the
  averaging operation, not the model's computation, and **the paper's effect sizes must be
  re-expressed against this floor rather than against the logit lens.** This would be a major
  finding and it is the reason to run it.
- **UNCLEAR** — C1 or C2 does not fire.

**Do not reinterpret.** FLOOR IS HIGH is not a failure of the experiment; it is the result.

## CONTROLS

- **C1 — the randomization must actually destroy the computation.** Mean cross-entropy of the
  randomized model on a held-out in-stream sample must be at or near the uniform-token bound
  ($\ln 50277 \approx 10.83$ nats), versus 1.06–2.93 for the real model (`e48_competence_gate`).
  *Number required to fire:* CE > 8.0 nats. If the model still predicts text, it was not randomized.
- **C2 — the fit must not degenerate.** $\lVert J \rVert_F$ per layer must be finite and non-zero,
  and the anchor gate (`tests/test_anchor_fidelity.py`) must pass on the box before fitting.
  *Number required:* all layer norms in $(0, \infty)$, 7/7 fidelity tests green.
- **C3 — the real-model arm must reproduce.** Re-scoring the stored real operator for the same
  corpus and seed at $Q0$ must reproduce `e48_crossover_410m.json`'s `arms_admitted_mean` to
  $\le 10^{-6}$. This is what proves the randomized arm is on the same scoring path.

## DECLARED BIAS

1. **Only the blocks are randomized.** Embedding and unembedding are the real ones, deliberately
   (see DESIGN). So this bounds "no computation", not "no model". A stronger null would randomize
   $W_U$ as well and would score near chance by construction, which is why it is less informative.
2. **Three corpora, not eight.** The floor is a property of the model, not of the corpus, so three
   spanning the observed range is enough to detect a corpus dependence in the floor if one exists.
   If the three floors differ materially, that is itself a finding and the full eight are warranted.
3. **One randomization draw per seed block.** A randomized network is a random object, exactly as a
   derangement is (§metric). Three seed blocks give three draws; that is fewer than the 120 the
   derangement control uses and the interval will be correspondingly wide.
4. Scored at $Q0$ only. The dose and read-context axes are not crossed with this.

## COST

The repo's own model (`repro/20_cost_estimate.sh`): one 410M lens at $N{=}200$ on an **L40S** is
**0.17 GPU-h / \$0.14**. Nine fits = **1.53 GPU-h / \$1.26**, plus ~1 GPU-h provisioning and the
preflight gate ≈ **2.5 GPU-h, ~\$2.** Scoring is free (local CPU).

**Within both budget gates** (\$40, 6 h).

## RESULT

*(unrun — no box provisioned)*

## VERDICT

*(pending)*

## IMPLEMENTATION FINDING FROM THE TINY-FIRST SMOKE (2026-08-16, before any spend)

The obvious implementation — `blk.apply(hf._init_weights)` — is a **silent no-op** on
transformers 5.14. Measured on 70m: block Frobenius **1603.7 → 1603.7**, and dispersion, stable
rank, unembedding alignment and cross-entropy all **bit-identical to the real model**. The cause is
that `transformers/initialization.py`'s `normal_`/`zeros_`/`ones_` each begin
`if not getattr(tensor, "_is_hf_initialized", False)` — the guard is on the **tensor**, not the
module, so clearing it on modules (the obvious reading) still does nothing.

`trainval.py` now clears the flag on **parameters** and **aborts** if the block Frobenius norm does
not move. Post-fix on 70m: **1603.7 → 117.0**, embedding and unembedding verified unchanged,
CE **3.43 → 34.92** against a real-model 3.43.

**Two consequences for the rule.** C1's registered threshold (CE > 8.0 nats) fires. But the
randomized model is far *above* the uniform bound of 10.83, not at it — a randomly initialised
network is confidently wrong rather than uniform. The rule is met as written and **is not being
reinterpreted**; this note exists so the number is not read as anomalous.

Had this not been smoked, E61 would have reported the real model as its own null and every arm
would have "passed".

## FLAGGED DELTAS

- Requires a GPU box, therefore a teardown obligation: pull + SHA-verify + mirror **before**
  `./lab down`, per `COMPUTE.md` §6 and the E28 lesson.
- The randomized model's weights are not an artifact worth storing; the **seed and the
  initialiser** are, and both go in the results file so the fit is regenerable.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e61_randomized_null_410m.json` | 6,583 | `db4b5e64cdb856a1` | `t61_adjudicate_null.py` | INHERITED |

**Payload checksums** (content only, provenance block excluded):

* `e61_randomized_null_410m.json` — `1dcd4fcbfbd93ea178ad62e17f5b1cdc`

<!-- END GENERATED PROVENANCE -->
