# PRE-REGISTRATION — T7 transport comparison, confirmatory extension

**Written 2026-08-10, BEFORE any 1B / 2.8B lens exists.** Commit this file before fitting either.
Branch `pythia`. Supersedes nothing; the exploratory tier stays exploratory forever.

---

## 0. CONTAMINATION DISCLOSURE — read this before anything else

**These results have already been seen: the 70M, 160M and 410M results.** A pre-registration written after seeing
data is not a pre-registration for that data, and pretending otherwise is the exact failure this
document exists to prevent. So:

**Everything already observed is permanently EXPLORATORY and is the SOURCE of the hypothesis, not
evidence for it.** Specifically, these are burned and can never be confirmatory:

| burned | what I saw |
|---|---|
| T7 at 70m / 160m / 410m, all six eval sets, all items | evals won: 3/6, 2/6, **6/6**; 410m margins typo +0.121 … association +0.004 |
| T3 read audit at 70m / 160m / 410m | R1 0.000 / 0.000 / 1.000; rand_rot control 0.996 (near-vacuous) |
| T4 geometry at all three | stable rank 8.5 / 7.6 / 66.9; PR(u1)/d 0.78 / 0.87 / 0.084 |
| T5a capability at four models | 21.3% / 25.0% / 72.1% / (1b 66.2% pre-exclusion) |
| T9 n-convergence at all three | task flat; subspace 0.51–0.69 → 1.00 |
| T10 target layer at all three | final wins 4 of 6 |
| T11 fit distribution at 70m | multihop +0.0091, typo +0.0183, multilingual −0.0242 |
| T12 concept interaction at 410m | ~7× random |
| T8 write battery at 410m | 1 admissible cell at α=1 |

**The confirmatory test below uses only cells this has not observed: Pythia-1B and Pythia-2.8B.**
No lens exists for either. If a 1B or 2.8B lens is fitted before this file is committed, this
pre-registration is void.

---

## 1. Hypothesis

Derived from the exploratory tier above; **not** independent of it.

> **H1.** The J-lens's advantage over the *unfitted* logit lens, measured by the anchor's own
> pass@k on the anchor's own six evaluation sets, is present at ≥410M and absent below it.
> Therefore it will also be present at 1B and 2.8B.

The interesting property of H1 is that it is **easy to falsify**: the exploratory tier could be a
70m/160m/410m fluke, `d_model` rather than parameter count could be the driver, or the advantage
could be non-monotone.

---

## 2. The confirmatory test, fixed now

**Models (both unseen):** `EleutherAI/pythia-1b-deduped`, `EleutherAI/pythia-2.8b-deduped`.

**Lenses:** fit with `pythia/t2_fastfit.py`, `--corpus wikitext`, `-n 200`, `--dim-batch 128`,
`--dtype float32`, `target_layer` = library default (final; justified by `t10_target_layer_*`,
where final wins 4 of 6 — a decision made from exploratory data and frozen here).
The `verify_against_anchor` gate must pass and its report must be in the provenance JSON.

**Transports:** `T = I` (logit lens) and `T = J_ℓ` (J-lens), same operator, same layers, same
items. The tuned lens is **not** part of this test (no Pythia tuned lens is fitted; adding it
later is a separate pre-registration).

**Metric:** `pass@k` exactly as `pythia/anchor_evals.py` implements it — mean over ITEMS of the
per-item fraction of intermediates whose min-over-layers rank ≤ k; normalized AUC over log k,
`ks = (1,2,5,10,25,100)`. **All items in every set. No `--max-items`.**

**Layers:** all fitted source layers. No band selection.

### PRIMARY OUTCOME (one number, declared now)

> `n_win` = the number of the six eval sets on which `AUC(J-lens) > AUC(logit lens)`, per model.

**H1 fires** iff `n_win ≥ 5` at **both** 1B and 2.8B.
**H1 is falsified** iff `n_win ≤ 3` at either.
`n_win = 4` at either model is **inconclusive** and will be reported as such, with no
reinterpretation.

Under a null of no J-lens advantage, `n_win` is Binomial(6, 0.5); `P(n_win ≥ 5) = 0.109` per
model, so `P(both ≥ 5) = 0.0119`. That is the test's whole α. **No other comparison in this
document may be used to claim H1.**

### SECONDARY OUTCOMES (reported, never used to rescue H1)

S1. Per-set AUC deltas, all six, both models, reported whatever their sign.
S2. Median best-rank per set and transport.
S3. Jacobian dispersion per layer (`fastfit` accumulator), both models.
S4. `n_win` at 70m/160m/410m re-stated as EXPLORATORY, for the trend picture only.

---

## 3. Analysis decisions frozen in advance

1. **No set is dropped**, for any reason, including "the model can't do the task". All six, all
   items.
2. **No band, layer subset, α, or `k` is chosen after seeing results.**
3. **No re-fit.** One lens per model at `n=200`. If a fit fails the anchor gate, the run is
   reported as failed — not retried with different settings.
4. **Ties** (`AUC` equal to 4 decimal places) count as a loss for the J-lens.
5. **`n_win` is computed on AUC only.** Not pass@1, not pass@10, not median rank. Those are S1/S2.
6. If a model OOMs or a lens cannot be fitted inside the budget gate, the test is **incomplete**,
   not partial — `n_win ≥ 5 at both` is unevaluable with one model missing.
7. Every number enters a results JSON before it is read (discipline #2). The confirmatory runner
   must be invoked with `--prereg pythia/PREREG_PYTHIA_T7.md`, which stores this file's SHA-256.

## 4. Cost and the budget gate

pythia-1b: d=2048, 15 source layers. pythia-2.8b: d=2560, 31. At the measured db=128 GPU rate
(70m 0.10 s/prompt, 160m 0.59, 410m 2.44), 1B ≈ 8 min and 2.8B ≈ 35 min at n=200, plus ~10 min
of pass@k. **Well under $2 on a 4090 and under the 6 h per-job gate.** 2.8B fp32 is 11 GB of
weights; if it does not fit at db=128, drop to db=64 — `dim_batch` is a batching parameter, and
the whole ladder for THIS test must use one value, chosen before the first fit.

---

## 5. REWARD-HACKING AUDIT of this pre-registration

Written adversarially against myself. Each row names a way I could make H1 fire without the
world cooperating, and the guard that blocks it.

| # | how I could cheat | guard |
|---|---|---|
| 1 | **HARKing** — present the exploratory 6/6 as if it were confirmatory | §0 lists every burned cell by name; the confirmatory claim rests only on 1B+2.8B |
| 2 | Pick eval sets that favour the J-lens | All six of the anchor's sets, no substitutions, no additions |
| 3 | Subsample items until the sign flips | `--max-items` forbidden; all items |
| 4 | Choose the band post hoc (the Gemma-2 program's original sin) | All fitted source layers; no band |
| 5 | Swap AUC → pass@10 → median rank until one wins | `n_win` is AUC-only, fixed in §2 |
| 6 | Add models until the trend appears (optional stopping) | Exactly two models, both named, both required |
| 7 | Drop a model that misbehaves | §3.6: a missing model makes the test **incomplete**, not partial |
| 8 | Re-fit with a different `n`/`dim_batch`/`target_layer` after a bad result | §3.3 one fit; §3.2 no post-hoc knobs; `target_layer` frozen to final with its justification |
| 9 | Call `n_win = 4` a partial win | §2 declares 4 inconclusive, in advance |
| 10 | Report a one-sided p without accounting for the two-model conjunction | The α is stated as 0.0119 for the conjunction, computed here, before the run |
| 11 | Let a tie break my way | §3.4 ties count against the J-lens |
| 12 | Quietly compare against a *fitted* baseline that is weaker than the logit lens | The baseline is `T = I` — it has no parameters and cannot be weakened |
| 13 | Use the tuned lens only if it loses | Tuned lens excluded entirely; adding it needs its own prereg |
| 14 | Fit the lens on data related to the eval sets | `--corpus wikitext` fixed; task-adjacent fitting is T11's separate question |
| 15 | Report dispersion only where it supports the story | S3 requires dispersion for **every** layer of both models |
| 16 | Silently fix a bug mid-run and keep the good half | Any code change after the first confirmatory fit voids the run; re-register |

**Residual risks I cannot fully guard, stated openly:**

- **The hypothesis is not independent of the exploratory data.** No prereg can undo that. The
  most this test can establish is that the pattern *extends* to two unseen models.
- **`n_win ≥ 5` is a coarse statistic.** It ignores effect size deliberately, to remove a
  degree of freedom. Effect sizes are S1 and are not part of the test.
- **Six eval sets are not independent** — they share a model, a lens and a tokenizer, so the
  Binomial null is approximate and probably anti-conservative. It is the anchor's own set of six
  and this registration is not going to invent a different one after the fact.
- **The harness and the pre-registration share an origin.** `anchor_evals.py` is pinned by 28 tests against
  the anchor's published spec, but a shared misreading of that spec would not be caught by them.

## 6. Predictions recorded, both ways

- **If H1 fires:** the claim is "the J-lens's read advantage over an unfitted baseline requires
  ≥410M and holds to at least 2.8B on Pythia" — a scale boundary, on one architecture family,
  with the exploratory tier cited as the source of the hypothesis.
- **If H1 is falsified:** the 6/6 at 410M is a fluke or a `d_model` artifact, and the honest
  paper is the *methodological* one — the anchor's eval data and metric reimplemented, four
  measurement defects documented (double final-norm, per-dose oracle gating, control
  non-independence, refit-on-target being structurally blocked), and no scale claim at all.
- **If inconclusive:** report `n_win` per model and stop. Do not add a third model to break the tie.

Signed off before any 1B or 2.8B lens exists. — operator + Claude Opus 5
