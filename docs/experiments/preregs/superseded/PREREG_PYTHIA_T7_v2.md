# PRE-REGISTRATION v2 — T7 transport comparison across the full Pythia ladder

**Written 2026-08-10, BEFORE any lens exists at 1B / 1.4B / 2.8B, and before any penultimate-target
lens exists at ANY scale.** Commit before fitting. Branch `pythia`.

Supersedes `PREREG_PYTHIA_T7.md` (v1). v1 is **VOID** — see §0.2. Its contamination disclosure is
carried forward in full and nothing it burned becomes unburned.

---

## 0. DISCLOSURE — read before anything else

### 0.1 Everything already observed, and therefore permanently exploratory

v1's burned table, unchanged and still binding:

| burned | what was seen |
|---|---|
| T7 at 70m / 160m / 410m, six sets, all items, **README metric, final target** | n_win 3/6, 2/6, **6/6**; 410m deltas typo +0.121 … association +0.004 |
| T7 paired item bootstrap at all three (added 2026-08-10) | P(n_win≥5) = 0.096 / 0.012 / **0.969**; CIs exclude zero on 1/6, 1/6, **3/6** |
| T3 read audit at all three | R1 0.000 / 0.000 / 1.000; `rand_rot` control 0.996 |
| T4 geometry at all three | stable rank 8.5 / 7.6 / 66.9 |
| T5a capability at four models | 21.3% / 25.0% / 72.1% / (1b 66.2%, on the superseded 80-cell battery) |
| T9 n-convergence at all three | task flat; subspace 0.51–0.69 → 1.00 (**circular**: 1.00 is the n=200 reference) |
| T10 target layer at all three | final beats penultimate 4 of 6 |
| T11 fit distribution at 70m | typo +0.0183, multilingual −0.0242 (multihop +0.0091 is **not stored**) |
| T12 concept interaction at 410m | ~7× random |
| T8 write battery at 410m | 1 of 8 runnable cells admissible at α=1 |

### 0.2 NEW disclosures made while writing this document

1. **v1 §0 contained a false statement.** It asserted "No lens exists for either [1B / 2.8B]."
   `results/lenses/misc/t0_smoke_1b_lens.pt` (N=3, d=2048, 15 source layers) is dated 2026-08-09,
   one day *before* v1 was written. No 1B or 2.8B **pass@k** has ever been computed, so no
   outcome was observed — but v1's own void clause is triggered by its own text. v1 is void.
2. **v1 §3.7 claimed the runner stores the prereg's SHA-256. It did not.** Fixed in
   `t7_lens_comparison.py` on 2026-08-10; this document's hash will be stored in every result.
3. **this has seen three paper-metric numbers at 70m.** While validating the new §A.6 scorer I
   computed J-lens `normalized_auc_over_logk_paper` at pythia-70m on the **final-target** lens:
   multihop **0.0394**, multilingual **0.0385**, order-ops **0.3524**. No logit-lens arm was run,
   so no `n_win` and no delta was observed. **70m is burned under the paper metric too.** It is
   not part of the confirmatory set.
4. **No penultimate-target lens exists at any scale**, and no paper-metric `n_win` exists at any
   scale. Both are clean.

### 0.2b CONTAMINATION EVENT, 2026-08-11 16:5x UTC — pythia-1b is PARTIALLY BURNED

While closing out E2 (`t16`, robust aggregation) run of `t7_lens_comparison.py` on
**`EleutherAI/pythia-1b-deduped`** — a registered confirmatory model — to check whether the
median aggregator changes *reads*. That check needed only *some* model; 410m was already burned
and would have served. **1b was used instead. That was an error.**

**Exactly what was observed** (`t16b_reads_mean_1b.json`, `t16b_reads_median_1b.json`),
paper-metric AUC, penultimate lens, all items:

| set | logit | J-lens (mean) | Δ | J-lens (median) | Δ |
|---|---|---|---|---|---|
| multihop | 0.2602 | 0.3085 | **+0.0415** | 0.3128 | +0.0454 |
| multilingual | 0.1422 | 0.1425 | **+0.0017** | 0.1440 | +0.0036 |
| typo | 0.1736 | 0.4181 | **+0.2445** | 0.4244 | +0.2508 |

**Damage.** Three of the six sets are observed at 1b, all positive. `n_win` at 1b is therefore
known to be **≥ 3 of 6**, and only order-ops, poetry and association remain unseen there. That is
substantial information about the primary outcome and cannot be un-seen.

**pythia-1b can no longer serve as a confirmatory cell.** §3.6 forbids silently rescoring over the
remaining two, so this document's test as written is **unevaluable**.

**1.4b and 2.8b remain completely clean** — no pass@k of any kind has been computed for either,
verified by audit over every results file. An amendment restricting the confirmatory test to those
two is therefore still a genuine pre-registration. **It requires the operator's sign-off and is not
adopted here.** See §7.

### 0.3 Why v1 is void rather than amended

Three of v1's frozen choices are now known to contradict the paper's own recipe, and one is a
factual error. Amending would mean changing a registered primary after the fact, which is the
failure this document exists to prevent. v1 is voided in full and replaced.

| v1 froze | now known | source |
|---|---|---|
| `target_layer` = final | the paper's **default recipe is penultimate**, with a stated rationale and a measured small pass@k gain | §A.7 |
| metric = shipped-README mean-of-fractions | §A.6 defines a **per-item binary** on the min-over-layers rank | §A.6 |
| exactly two models (1B, 2.8B) | the program requires the full ladder to see emergence | operator, 2026-08-10 |
| "no 1B lens exists" | one did | disk |

---

## 1. Hypothesis

Derived from the exploratory tier in §0.1; **not** independent of it.

> **H1.** The J-lens's advantage over the *unfitted* logit lens, measured by the paper's §A.6
> pass@k on the anchor's six released evaluation sets, is present at ≥410M and **persists across
> the next three rungs of the Pythia ladder, to 2.8B.**

H1 is easy to falsify: the advantage could be non-monotone, could vanish once the model is large
enough that the logit lens alone suffices, or could be an artifact of `d_model` rather than scale.

---

## 2. The confirmatory test, fixed now

**Models (all three unseen, no lens exists for any):**
`EleutherAI/pythia-{1b,1.4b,2.8b}-deduped`.

**6.9B and 12B are deliberately NOT in this test.** They are unfundable on the available hardware
(§4) and will require H100-class compute. They are deferred to a **separate pre-registration**,
written after this one resolves. Registering them here and then failing to run them would make
this test permanently incomplete under §3.6. This model set is fixed **before any lens exists at
any of the three scales.**

**Lenses.** `t2_fastfit.py`, `--corpus wikitext`, `-n 200`, `--dim-batch 128`,
`--dtype float32`, `--skip-first` left at the library default.

- **`skip_first = 16`, HARD-CONFIRMED** at `jacobian-lens/jlens/fitting.py:42`, applied by default
  in both `valid_position_mask` and `fit()`. In-code rationale: *"early positions act as attention
  sinks and have atypical residual statistics."* The **paper** reports that skipping gives no
  meaningful improvement and describes the unmodified position average as its default, i.e. 0.
  **These disagree. 16 is the registered primary** because it is what the released reference
  implementation does, what all three existing lenses used, and because Pythia's BOS position
  carries 7–9× the residual norm of any real token. The skip sweep {0, 4, 16} is registered as a
  separate exploratory follow-up on ONE model (§3.9) and can never enter the primary.
- **`--target-layer -2` (PENULTIMATE) is the registered PRIMARY**, matching the paper's default
  recipe. `--target-layer` unset (final) is fitted as well and is **secondary only** (S5).
  Consequence, declared now: sources must be `< target`, so a penultimate lens has **one fewer
  fitted source layer** than a final lens on the same model. `n_win` is computed within a lens,
  so this does not bias the J-vs-logit contrast, but it does make penultimate and final AUCs
  non-comparable in absolute terms.
- The `verify_against_anchor` gate must pass **at the same `target_layer` and `skip_first` being
  fitted** (threaded through 2026-08-10; previously it silently gated the default configuration)
  and its report must appear in the provenance JSON.

**Transports:** `T = I` (logit lens) and `T = J_ℓ` (J-lens). Same operator, same layers, same
items. The tuned lens is **excluded**; adding it requires its own pre-registration.

**Metric — PRIMARY, the paper's §A.6:**

> `recovered_k(i) = 1[∃ℓ : rank_ℓ(v_i) ≤ k]`,  `pass@k = (1/N) Σᵢ recovered_k(i)`,
> then normalized AUC over log k with `ks = (1, 2, 5, 10, 25, 100)`.

**Declared interpretive choice.** §A.6 specifies one intermediate per item. Four of the six
*released* sets carry several (multilingual 4, order-ops 2). The indicator is applied
**per (item, intermediate) pair — a flat pool** — which reduces exactly to §A.6's formula when
each item has one intermediate, and does not reweight items by how many intermediates they carry.
The alternatives ("item counts if ANY / if ALL") are noted and rejected. Implemented as
`normalized_auc_over_logk_paper`; verified to coincide exactly with the shipped-README form on
order-ops, where every item has exactly two intermediates.

**Items:** **all items in every released set**, no `--max-items`.
**Declared mismatch:** the paper reports 50 / 54 / 55 / 52 / 96 / 50 items for
multihop / multilingual / order-ops / poetry / typo / association; the released JSON files contain
**93 / 107 / 55 / 98 / 96 / 102**. Only order-ops and typo agree. **Our absolute AUCs are therefore
not comparable to the paper's reported values.** The J-vs-logit contrast is internal to our run
and is unaffected. This is disclosed, not resolved.

**Layers:** all fitted source layers. No band, no subset.

### PRIMARY OUTCOME, declared now

For each model, `n_win` = the number of the six sets on which
`normalized_auc_over_logk_paper(J-lens) > normalized_auc_over_logk_paper(logit lens)`.

- **H1 FIRES** iff `n_win ≥ 5` at **all three** models.
- **H1 IS FALSIFIED** iff `n_win ≤ 3` at **any** model.
- **Anything else is INCONCLUSIVE**, reported as such, with no reinterpretation.

**α.** Under a null of no J-lens advantage, `n_win ~ Binomial(6, 0.5)`, so `P(n_win ≥ 5) = 0.109`
per model and `P(fire) = 0.109³ = 1.3 × 10⁻³`. That is the test's entire α. **No other comparison
in this document may be used to claim H1.**

**Power, computed in advance from the measured 410m bootstrap.**
- If each unseen model behaves like 410m (measured `P(n_win ≥ 5) = 0.969`, `P(n_win ≤ 3) = 0.001`):
  **power = 0.969³ = 0.910**, with a ~0.3% chance of spurious falsification.
- If poetry and association remain floor cells at every scale — both arms at pass@k = 0 for k ≤ 10,
  as measured at 410m — so that four sets win deterministically and two are coin flips, then
  `P(n_win ≥ 5) = 0.5` and `P(n_win ≤ 3) = 0.25` per model. **Power collapses to 0.125, and the
  chance of outright falsification rises to 1 − 0.75³ = 0.58.** In that world this test declares
  H1 false more often than not, *even though the J-lens wins on every set where the model can do
  the task.* **This is the single largest known threat to the test and it is stated before the run,
  not after.** It is not mitigated by any post-hoc set exclusion; §3.1 forbids that. If it
  materializes, the correct response is a NEW pre-registration with a capability-gated eval set —
  not a reinterpretation of this one.

### SECONDARY OUTCOMES — reported always, never used to rescue H1

- **S1** per-set AUC deltas, all six, all three models, whatever their sign, each with a 95% paired
  item-level bootstrap CI (`t7_bootstrap_audit.py`, 10,000 resamples, seed 0) and the count of
  sets whose CI excludes zero.
- **S2** median best rank per set and transport.
- **S3** Jacobian dispersion per layer for every fitted lens (`fastfit` accumulator). Note: this
  is **absent from all five existing lens provenance files**, because the accumulator postdates
  them; the three sub-1B lenses are refit here so that dispersion exists across the whole ladder.
- **S4** `n_win` at 70m / 160m / 410m under the paper metric, restated as **EXPLORATORY**, for the
  trend picture only.
- **S5** the same `n_win` on **final**-target lenses, as a recipe-sensitivity check against the
  paper-vs-code disagreement on `target_layer`.
- **S6** the shipped-README metric alongside the paper metric everywhere, so every pre-2026-08-10
  number in `results/` remains comparable.

---

## 3. Analysis decisions frozen in advance

1. **No set is dropped**, for any reason, including "the model cannot do the task" and including
   the floor-cell threat named in §2. All six, all items.
2. **No band, layer subset, α, or `k` is chosen after seeing results.**
3. **One lens per (model, target_layer).** If a fit fails its anchor gate, that run is reported as
   **failed**, not retried with different settings.
4. **Ties** (paper-metric AUC equal to 4 decimal places) count as a **loss** for the J-lens.
5. **`n_win` is computed on the paper-metric AUC only.** Not pass@1, not pass@10, not median rank,
   not the README metric. Those are S1/S2/S6.
6. **A missing model makes the test INCOMPLETE, not partial.** `n_win ≥ 5 at all three` is
   unevaluable with fewer than three models. If any of the three cannot be fitted inside the
   budget gate, H1 is **not evaluated** — the run is reported as incomplete and superseded by a
   new document naming the models actually run. It is *not* silently rescored over two.
7. **Every number enters a results JSON before it is read.** The runner must be invoked with
   `--prereg pythia/PREREG_PYTHIA_T7_v2.md`, which now stores this file's SHA-256 in the result.
8. **Any code change after the first confirmatory fit voids the run.** Re-register.
9. **The `skip_first ∈ {0, 4, 16}` sweep and the OOD ladder are separate, later, exploratory work**
   and can never enter this test.

---

## 4. Cost, and the budget gate — THIS RUN IS NOT FULLY FUNDED

Extrapolated from the measured `db=128` GPU rate (70m 0.10 s/prompt, 160m 0.59, 410m 2.44),
scaling as `d_model × params`. **Estimates, not measurements.**

| model | d_model | est. s/prompt | per lens (n=200) | ×2 target layers |
|---|---|---|---|---|
| refit 70m / 160m / 410m | — | — | 20 s / 2 min / 8 min | ~20 min total |
| pythia-1b | 2048 | ~12 | ~40 min | ~1.3 h |
| pythia-1.4b | 2048 | ~17 | ~56 min | ~1.9 h |
| pythia-2.8b | 2560 | ~42 | ~2.3 h | ~4.6 h |
| **pythia-6.9b** | 4096 | ~164 | **~9.1 h** | **~18 h** |
| **pythia-12b** | 5120 | ~357 | **~20 h** | **~40 h** |

**Tier 1 (1b, 1.4b, 2.8b + the three refits): ~8 h of 4090 time across six independent jobs, each
individually under the 6 h per-job gate. ≈ $3.** Proceed on approval.

**Tier 2 (6.9b, 12b): ~58 h, and every single job exceeds the 6 h gate.** It also does not fit:
6.9B fp32 is 27.6 GB of weights and 12B is 48 GB, before the `dim_batch=128` replicated forward.
**This tier cannot be run on the available hardware and is EXCLUDED from this test** (§2), not
deferred within it. It requires H100-class compute and gets its own pre-registration, written
after this one resolves.

That later document must fix, before any 6.9B/12B lens exists, whichever of these it needs — each
changes a parameter that is frozen here, so none may be adopted silently inside this test:

- **(a) bf16** for the large models — a dtype confound against this fp32 ladder.
- **(b) reduced `n`** (the anchor licenses n≈10–100) — an `n` confound, against which C3 is direct
  evidence that the operator has not converged.
- **(c) penultimate-only**, dropping S5 there — costs only a secondary and is the cheapest.

**No dtype, `n`, or model-set change will be made to THIS test after seeing any result.**

---

## 5. REWARD-HACKING AUDIT

| # | how I could cheat | guard |
|---|---|---|
| 1 | HARK the exploratory 6/6 as confirmatory | §0.1 lists every burned cell; the claim rests only on the three unseen models |
| 2 | Pick favourable eval sets | all six released sets, no substitutions |
| 3 | Subsample items until the sign flips | `--max-items` forbidden |
| 4 | Choose the band post hoc | all fitted source layers; no band |
| 5 | Swap metric until one wins | `n_win` is paper-metric AUC only (§3.5); README form is S6 |
| 6 | Add models until the trend appears | exactly three, all named, all required (§3.6); 6.9B/12B need their own prereg |
| 7 | Drop a model that misbehaves | §3.6: missing model ⇒ **incomplete**, not partial |
| 8 | Re-fit with different knobs after a bad result | §3.3 one fit; §3.2 no post-hoc knobs; §4 the Tier-2 contingency is chosen now |
| 9 | Call `N = 2 or 3` a partial win | §2 declares it inconclusive in advance |
| 10 | Quote a per-model p without the conjunction | α = 1.3e-3 for `n_win ≥ 5` at all three, computed here, before the run |
| 11 | Let a tie break my way | §3.4 ties count against the J-lens |
| 12 | Compare against a weakened fitted baseline | the baseline is `T = I`; it has no parameters |
| 13 | Use the tuned lens only if it loses | excluded entirely |
| 14 | Fit on data related to the eval sets | `--corpus wikitext` fixed |
| 15 | Report dispersion only where it helps | S3 requires it for **every** layer of **every** lens |
| 16 | Silently fix a bug mid-run and keep the good half | §3.8 voids the run |
| 17 | **Pick `target_layer` after seeing which wins** | penultimate is primary, fixed here on the paper's stated default; final is S5 and can never claim H1 |
| 18 | **Pick the `skip_first` that helps** | 16, hard-confirmed and fixed here; the sweep is §3.9 exploratory |
| 19 | **Exclude the floor sets once they fail** | §3.1 and §2's power statement forbid it, in advance |

**Residual risks I cannot guard, stated openly:**

- **The hypothesis is not independent of the exploratory data.** No prereg undoes that. The most
  this establishes is that the pattern extends to three unseen models.
- **`n_win ≥ 5` is coarse by design** — it ignores effect size to remove a degree of freedom.
  Effect sizes are S1 and are not part of the test.
- **The six sets are not independent** (shared model, lens, tokenizer), so the Binomial null is
  approximate and probably anti-conservative.
- **The floor-cell threat (§2) may simply defeat this test.** Power is 0.187 in that world.
- **The harness, the metric and this document share an origin.** Mitigation: the read path is now asserted
  bit-identical to the anchor's own `JacobianLens.apply` (`tests/test_anchor_fidelity.py`,
  max|diff| 0.00e+00), and the anchor's own readout is asserted against `hf(...).logits`. A shared
  misreading of §A.6 would still not be caught.
- **Our item counts exceed the paper's on four of six sets** (§2). Absolute AUCs are ours alone.

---

## 6. Predictions recorded, both ways

- **If H1 fires:** "the J-lens's read advantage over an unfitted baseline requires ≥410M and holds
  across the Pythia ladder to 2.8B" — a scale boundary on one architecture family, with the
  exploratory tier cited as the *source* of the hypothesis, not as evidence.
- **If H1 is falsified:** the 410M result is a fluke, a `d_model` artifact, or non-monotone, and
  the honest paper is the methodological one — the anchor's data and metric reimplemented, the
  read path proven faithful, and the measurement defects documented (double final-norm, per-dose
  oracle gating, control non-independence, `probe-swap` run with the wrong operator, the
  paper-vs-code `target_layer` and `skip_first` disagreements) — with no scale claim.
- **If inconclusive:** report `n_win` per model and stop. **Do not add a sixth model.**

Signed off before any lens exists at 1B, 1.4B or 2.8B, and before any penultimate-target lens
exists at any scale. — operator + Claude Opus 5

---

## 7. AMENDMENT A1 — price the peek, do not delete the model (revised 2026-08-11)

Triggered by §0.2b. **An earlier draft of this section proposed dropping pythia-1b from the
confirmatory set. That was an over-correction and is withdrawn.** It discarded a model *and*
produced a worse α than keeping it. The operator identified the error.

### 7.1 What the contamination actually costs

The Binomial null `P(n_win ≥ 5) = 0.109` assumes the outcome was unobserved. Three of six sets at
1b were observed, all wins. Conditional on that, under the same null,

`P(n_win ≥ 5 | 3 wins observed) = P(≥2 of the remaining 3) = C(3,2)/8 + C(3,3)/8 = 0.5`

So 1b's contribution to the conjunction is **0.5, not 0.109**. Nothing else changes: 1.4b and 2.8b
are verified unobserved and keep their 0.109.

| option | α (3-model conjunction) | models retained |
|---|---|---|
| uncontaminated (unattainable) | 0.109³ = **0.00130** | 3 |
| **A1 — retain 1b at its conditional α** | 0.5 × 0.109² = **0.00594** | **3** |
| withdrawn proposal — drop 1b | 0.109² = 0.01188 | 2 |

**A1 is adopted.** It is both more powerful and more informative than deletion.

### 7.2 Why this is legitimate rather than a rescue

The forking-paths risk a pre-registration guards against applies to **choices that remain open**.
At 1b none do: the metric (§2), threshold (§2), item set (§3.1), layer set (§3.1), tie rule (§3.4)
and lens (fitted, gated, SHA-verified) were all frozen before any 1b number existed. The remaining
three sets are produced by a fully determined script. **Observing 3 sets cannot change what the
other 3 return.** What it does change is the sampling distribution of the test statistic, and that
is priced above rather than argued away.

### 7.3 Binding conditions

1. **All six sets at 1b are still run and all six reported**, whatever they show. The three already
   observed are reported as observed-in-advance, with §0.2b cited at the point of use.
2. **The conjunction α is quoted as 0.00594**, never 0.00130, in any document, abstract or figure
   caption.
3. **1.4b and 2.8b remain untouched until scored**, and their per-model α stays 0.109. Any further
   peek at either triggers this same treatment and must be recorded in §0.
4. **A guard is added to the runner** (§3.10) so this cannot recur silently.

### 7.4 §3.10 — new frozen analysis decision

> **Any scoring run against a model named in §2 must be invoked with `--prereg`.** The runner
> refuses to score a §2 model without it. This is the `assert` that would have prevented §0.2b.

### 7.5 Process failure that produced this

`IMPLEMENTATION_RULES.md` R2 ("index against the original prompt before declaring a turn
complete") and the burn discipline both existed, both written in the same session, and neither
fired — because scoring a lens to close out a control was treated as routine rather than as
observing a registered cell. The remedy is mechanical (§7.4), not exhortative.
