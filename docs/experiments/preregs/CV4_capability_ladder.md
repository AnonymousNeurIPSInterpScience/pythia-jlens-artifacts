# CV4 — Does the corpus effect replicate up the capability ladder?

**Pre-registered 2026-08-23, before any number at 1.4B or above existed.**
Numbered CV4, not E60: `E60_fitter_determinism.md` and `e66_*` already exist, and this belongs with
`cv1` (competence), `cv2` (positional support) and `cv3` (margins) as validity work.

## WHY

CV3 established that the corpus effect is a **score effect, not a ranking artifact** — 27.16 pooled
seed SD in z-space at 410M, 2.5x larger than in rank space
(`results/cv3_margins_410m.json`). One objection survives that result and it is the strongest one
left:

> **"Pythia-410M answers 4.7% of this battery. A 27σ effect measured in a near-degenerate regime may
> be structured noise — corpora differ in how they perturb an operator that is reading a computation
> the model is not performing. Show it at a scale where the model can actually do the task."**

That objection is fair, it is not answerable by any recomputation of existing data, and it is the
last thing standing between the corpus finding and a clean claim. **CV4 answers it or concedes it.**

**This does NOT attempt to rehabilitate absolute recovery claims.** Those are retired by cv1 and cv2
and stay retired regardless of the outcome here. The validity frame does not depend on capability
being high anywhere.

## DESIGN

Three phases with gates between. All arms use the corrected (stripped) readout, the flat-mean-7
k-summary, `N=25` fitting prompts (S2 settled that N is flat; do **not** sweep it), and a fit
protocol byte-identical to CV3 so the ladder is comparable.

### Phase 1 — capability calibration (forward-only, cheap)

For each model, on all six Gurnee et al. families: final-answer top-1 and top-10 (the cv1 metric —
rank of the **expected answer**, not the intermediate), per family and pooled. **Cache `h_t` at each
item's readout position for every band layer and reuse it in Phase 3.** Do not recompute.

Existing anchors: 70M 1.6% / 160M 1.6% / 410M 4.7% / 1B 4.3% pooled top-1
(`results/cv1_answer_competence.json`).

### Phase 2 — fit the operators

Per model: 5 in-stream corpora x 3 seed blocks = **15 operators**, `N=25`, same corpora and matched
seeds as CV3.

### Phase 3 — corpus contrast in z-space, from cached `h_t`

Per model, compute spread across the 5 corpora / pooled seed SD, in **z-space primary** (CV3's
finding: z is robust to the competitor-suppression pathology that fools `min` and margin). Record
rank and margin alongside for completeness.

## PRIMARY

`spread_z / pooled_seed_sd_z` per model, plotted against Phase-1 pooled competence, together with
**concordance of the 5-corpus ordering against 410M's** (StackExchange > Pile-CC > USPTO >
Wikipedia_en > Github), reported as Kendall tau.

## DECISION RULE — fixed before running, both branches are findings

* **H1 — REPLICATES.** The z-space effect is **>= 15σ at the highest-capability model reached**, and
  ordering concordance with 410M is **tau >= 0.6**. The degenerate-regime objection is answered: the
  corpus effect is not an artifact of measuring in a regime the model cannot perform.
* **H0 — ATTENUATES.** The effect falls **monotonically** with capability and is **< 5σ** at the
  highest model reached. The corpus effect is a small-model phenomenon. **This is a publishable
  finding, not a loss** — it would mean corpus sensitivity is itself capability-gated, which is a
  sharper claim than "corpora differ," and it must be reported as the headline if it fires.
* **UNCLEAR.** Anything else — non-monotone, or 5–15σ at the top. Report the curve and stop. Do not
  re-cut, do not add models to chase a threshold.

**Ordering concordance is reported in all three branches**, because an effect that persists in
magnitude while scrambling its ordering is a different finding from one that persists in both.

## CONTROLS — each with the number it must produce

* **C1 — `logit_I` reproduces per model.** The identity arm's `min` flat-mean-7 admitted mean must
  match the model's own stored logit constant where one exists (410M: **0.19810852520167826**), and
  for new models must be emitted and stored before any J arm is graded. **No model's corpus number
  is graded if C1 has not fired on that model.**
* **C2 — margin/rank consistency.** `margin > 0` iff `rank == 1`, 100% of triples. Ported from CV3.
* **C3 — seed SD non-degenerate.** Pooled seed SD > 0 in z-space per model, else PRIMARY is undefined
  and that model is void.
* **C4 — competence is actually measured, not assumed.** Phase 1 must emit per-family accuracy for
  every model. A pooled number alone is insufficient: multilingual is at floor at every scale
  measured so far and could dominate a pooled figure.

## DECLARED BIAS

Capability and scale are confounded — every larger model is also differently trained, so an effect
that tracks capability may track any correlate of scale. **CV4 cannot separate them**, and no
outcome licenses "capability causes the corpus effect to shrink/persist." It licenses only
"the effect does/does not survive up the ladder."

The 5 corpora are the same 5 as CV3, so this is a replication of *that panel*, not a sample of
corpora. n=5 remains the replication unit for anything about kinds of corpus.

## SUPERSEDED — 2026-08-23

**Phase 1 HAS RUN** (`results/cv4_phase1_capability.json`) and is retained. **Phases 2–3 are
superseded by [`CV6_per_family_ladder.md`](CV6_per_family_ladder.md)**, which replaces the pooled
estimand with a per-family one and corrects two things in this document:

1. **The cost table below is stale.** It assumed 60% utilisation and did not account for the backward
   traversing only the band. The reproducible model is `tools/fit_cost_model.py`: **1.4B ~27 min,
   2.8B ~70 min on one L40S at fp32**, for 15 operators at N=25 — not 1.7 h and 4.2 h.
2. **The h_t caching instruction in Phase 1 was wrong.** It says to cache on the Phase-1 run and
   reuse in Phase 3. Phase 1 ran on CPU and fitting runs on GPU; D2 measured a CUDA-vs-CPU divergence
   of 2.774e-04, so reuse would bake a device mismatch into the contrast. **CV6 caches on the box.**

Phase 1's STOP condition fired: *"if even M\* is <~10% pooled competence, log this as a finding"* —
2.8B is **5.9%**. Recorded in `docs/context/CONTEXT.md` §1d.

## COST — STALE, see SUPERSEDED above

**Two corrections to the operator's off-the-cuff estimate.**

**(1) The formula overcounts by `n_layers`.** `src/fastfit.py:140` calls
`torch.autograd.grad(outputs=target_activation, inputs=source_activations, ...)` — **one backward
returns gradients for every source layer at once**. So the cost is

```
backward-token-passes per operator  =  N_prompts x d_model x seq_len
```

not `N x n_layers x d_model`. That is a factor of 24–36 saved.

**(2) `CLAUDE.md` §7 forbids TF32 and mandates fp32, and that is the real budget.** GPUs are chosen
by **fp32** TFLOPS: L40S 91.6 > 4090 82.6 > H100 67. At L40S fp32, using ~4 x params FLOPs per
backward token and 60% achieved utilisation:

| model | `d_model` | tokens/operator | est. per operator | **15 operators** |
|---|---:|---:|---:|---:|
| 1.4B | 2048 | 6.6M | ~7 min | **~1.7 h** |
| 2.8B | 2560 | 8.2M | ~17 min | **~4.2 h** |
| 6.9B | 4096 | 13.1M | ~66 min | **~16.5 h** |
| 12B | 5120 | 16.4M | ~143 min | **~36 h** |

**The "one hour on an H100" estimate assumed bf16-class throughput. Under this programme's own
fp32/no-TF32 discipline the ladder is 10–20x that.** This is not a reason to abandon it; it is a
reason to scope it before provisioning.

### The plan this spec registers

**Phase 2/3 at 1.4B and 2.8B, full 5 corpora x 3 seeds: ~5.9 h, one L40S, ~$5–8.** That is inside the
`CLAUDE.md` §7 budget gate (\$40 / 6 h) and yields a **four-point ladder — 410M, 1B, 1.4B, 2.8B — a
~7x capability span**, which is enough to see a trend.

**6.9B and 12B are NOT registered here.** Either needs an explicit operator ruling on one of:

1. **a TF32 exception for this sweep only**, which is a registered-discipline change and must be
   ruled on, not assumed — it would also make CV4's numbers non-comparable to CV3's fp32 ones unless
   a matched-precision control is run;
2. **a budget-gate exception** (~16.5 h for 6.9B);
3. **a scope cut** — 5 corpora x 2 seeds at 6.9B is ~11 h and leaves a 2-point seed SD, which C3
   would pass but weakly.

**Additional friction at 6.9B/12B, flagged not costed:** the backward is taken through a loss defined
on the *residual stream*, not the output logits. At those sizes that needs memory machinery
(gradient checkpointing, careful graph retention) which is **not currently wired**. Budget setup
time, not run time. If it OOMs, drop the model and report the ladder as far as it went.

## WHAT THIS DOES NOT DO — scope, declared

* Does **not** attempt causal validation of intermediates. That is the synthetic-suite programme
  (`docs/validity/CONSTRUCT_VALIDITY.md` §6 Route B) and it is a separate paper.
* Does **not** rehabilitate absolute recovery claims at any capability level.
* Does **not** sweep `N`, aggregation, readout position or band — all settled upstream.
* Does **not** vary the eval battery. The construct-validity problem in `CONSTRUCT_VALIDITY.md`
  stands unchanged whatever CV4 returns.

## OUTPUT

**Output:** `results/cv4_phase1_capability.json` — Phase 1 only.

This registration originally named a single output, `cv4_capability_ladder.json` (written here
without its path on purpose, so the provenance generator does not read it as a citation). **That
file was never produced and never will be**, because Phases 2-3 were superseded by
[`CV6_per_family_ladder.md`](CV6_per_family_ladder.md) before they ran. The ladder they would have
emitted is `results/cv6_per_family_ladder.json`, registered under CV6. Phase 1 ran and adjudicated on
its own; its verdict is folded into `docs/validity/DIAGNOSIS.md` and `docs/context/STATE.md`.

The original name is recorded rather than deleted so the supersession is visible in the registration
itself and not only in the ledger. It is **not** a missing result.

## STATUS

**PHASE 1 RUN AND ADJUDICATED — 2026-08-23. `results/cv4_phase1_capability.json`.**
**PHASES 2–3 SUPERSEDED by [`CV6_per_family_ladder.md`](CV6_per_family_ladder.md)** — see the
SUPERSEDED section above, which is the authoritative account. 6.9B and above still require an
operator ruling and are not registered anywhere.

Phase 1 as stored: answer competence over the 255 items carrying a `target`, 95% Wilson —
70m 1.6% [0.6,4.0] / 160m 1.6% [0.6,4.0] / 410m 4.7% [2.7,8.0] / 1b 4.3% [2.4,7.6] /
1.4b 3.1% [1.6,6.1] / **2.8b 5.9% [3.6,9.5]**. Every interval overlaps from 410M up: pooled top-1
**never separates**. Top-10 rises 16.9 → 32.9% end to end but **not monotonically** — it dips at
1.4B (1b 29.4% → 1.4b 27.5%). Pooled prompt surprisal falls monotonically at every rung,
5.204 → 4.013. **The STOP condition fired at 5.9%.**

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/cv3_margins_410m.json` | 11,283 | `e5d2a4fb84fa96be` | `cv3_margins.py` | CLEAN |
| `results/cv1_answer_competence.json` | 4,967 | `dbdcd4f33687bbae` | `—` | IMMUNE |
| `results/cv4_phase1_capability.json` | 18,253 | `1b05b8f55679897f` | `cv4_phase1_capability.py` | IMMUNE |
| `results/cv6_per_family_ladder.json` | 30,023 | `f8cdbcc60189e3f9` | `cv6_per_family_ladder.py` | CLEAN |

**Payload checksums** (content only, provenance block excluded):

* `cv3_margins_410m.json` — `38efe87c84be1b8bee5fa680a5a3d354`
* `cv4_phase1_capability.json` — `43b21658bcc6a1acf619b614a5eb74db`
* `cv6_per_family_ladder.json` — `37c137de5d7788ff265c533db4dabc53`

<!-- END GENERATED PROVENANCE -->
