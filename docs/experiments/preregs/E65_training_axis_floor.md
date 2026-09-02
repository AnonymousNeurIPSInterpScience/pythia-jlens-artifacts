# E65 — when during training does a transport lens start working?

**PRE-REGISTERED 2026-08-17, before any checkpoint is downloaded.**

---

## WHY

The paper's Limitation 8 says the comparison is degenerate **below 410M parameters**: concept-token
unembedding is effectively rank-1 (mean pairwise cosine **0.959** at 70M, **0.922** at 160M, against
**0.031** at 410M), so no transport can separate what $W_U$ has collapsed. That is a floor on the
**parameter** axis.

Nobody has asked whether the **training** axis has one too. Pythia publishes ~154 intermediate
checkpoints per size, so the question is directly answerable: an early checkpoint of a 1B model
plausibly has the same collapsed readout geometry as a small model has at convergence. If so, every
developmental-interpretability result computed with a vocabulary-grounded readout has a validity
window nobody has measured.

**This is an instrument-validation experiment, not a science claim about concepts.** It does not
compete with crosscoder work on feature emergence (arXiv:2509.17196, arXiv:2509.05291); it asks when
*our* instrument becomes usable. It is the training-axis analogue of the paper's own S1.

**What it cannot do, stated up front:** the causal-validation step that makes the crosscoder papers
credible (RelIE-style intervention) is unavailable to us — W1/W2 writes never produced an admissible
result and are excluded from the paper. E65 is correlational and structural, and any writeup must
say so.

## DESIGN

Two phases. **Phase 0 is free of GPU and can make Phase 1 unnecessary.**

### Phase 0 — the geometry gate, no Jacobian, no fitting

`experiments/t65_ckpt_geometry.py`. For each checkpoint, load `embed_out` only and compute E39's
statistic: mean pairwise cosine and effective rank over the **1310 concept tokens** of the admitted
battery. Band-free, eval-free, CPU.

Models: `pythia-410m-deduped` and `pythia-1b-deduped`.
Checkpoints: the published log-spaced early grid (`step0,1,2,4,…,512`) plus `step1000` and every
subsequent power-of-two-ish rung to `step143000` — **~20 per model, fixed in the results header
before the run**.

### Phase 1 — the paid sweep, only where Phase 0 says it is interesting

`experiments/t65_ckpt_readout.py`. At ~10 log-spaced checkpoints spanning the window Phase 0
identifies:

- **one fitting corpus, held fixed across every checkpoint.** Non-negotiable: our own result is that
  the fitting corpus accounts for more variance than the read context, so varying it between
  checkpoints would swamp the signal. Corpus = **Pile-CC** (mid-range at both scales, not the
  degenerate Github, not the 410M-best USPTO whose advantage does not transfer to 1B).
- $N=200$, `dim_batch=128`, `skip_first=16`, `target_layer=-2`, fp32, TF32 off, declared band.
- arms per checkpoint: **J**, **logit lens** ($J=I$), **$J^{\mathrm{shuf}}$** (5 random derangements).
- `persist` primary, `min` reported, `min` never votes.

## PRIMARY

**$t^\*$ — the training-token count at which the J-lens read first exceeds both (a) its own
layer-deranged floor and (b) the logit lens at the same checkpoint, and remains above both at every
later checkpoint.**

## DECISION RULE — fixed before any checkpoint is downloaded

- **FLOOR EXISTS** — $t^\* > $ the earliest measured checkpoint. There is a training-axis floor;
  the paper's parameter-axis degeneracy has a training-axis analogue, and any checkpoint study using
  this readout has a validity window.
- **NO FLOOR** — J clears both at the **earliest** measured checkpoint. The instrument works from
  very early in training and the parameter-axis floor has no training-axis analogue at this scale.
  **This is a publishable negative and it is the outcome This registration expects to be most likely.**
- **NON-MONOTONE** — clears, then fails again at a later checkpoint. Report the curve; make no floor
  claim. Do not pick the first crossing and call it $t^\*$.
- **UNCLEAR** — C1 fails, or J never clears at any checkpoint including the last.

**Not to be reinterpreted.** In particular, NO FLOOR is a result: it would say the instrument is
safe to use across training, which is exactly what a checkpoint study needs to know.

## CONTROLS

- **C1 — the final checkpoint must reproduce our stored result.** `step143000` is the released model
  we have already measured. Its J read on Pile-CC at $N{=}200$ must reproduce
  `e52_factorial_410m.json`'s corresponding cell to **within 2 seed SD** (410M seed SD on Pile-CC is
  1.84e-4, so the tolerance is ~3.7e-4). *If this fails, the checkpoint pipeline is not the pipeline
  that produced the paper and nothing else in E65 is comparable.* **This is the load-bearing control.**
- **C2 — derangement at every checkpoint.** 5 random derangements per checkpoint, `persist`. The
  floor must be defined per checkpoint, not imported from the final model: a deranged operator on an
  early checkpoint is a different object.
- **C3 — the geometry gate at every checkpoint** (Phase 0). A checkpoint whose mean pairwise
  concept-token cosine exceeds the 410M value of **0.031** is flagged as geometrically degenerate,
  and its readout is reported but excluded from $t^\*$.
- **C4 — capability.** Cross-entropy per checkpoint on held-out in-stream text, via the fixed
  `trainval.py` path. A rising read on a model whose CE is still near-uniform would indicate the
  readout is measuring $W_U$ rather than computation.
- **C5 — the untrained anchor.** E61's randomized-block operators (read **0.0035–0.0040** against a
  free baseline of **0.02844**, CE **11.85** nats) are the $t{=}0$ end of this axis and are plotted
  as such. `step0` should land near them; if it does not, one of the two is wrong.

## DECLARED BIAS

1. **The logit lens is not constant across checkpoints.** $W_U$ and the residual stream both change,
   so the free baseline moves and must be measured at every checkpoint rather than imported. This is
   the same defect the paper corrected on the read-context axis.
2. **One fitting corpus.** Holding it fixed is required (above), but it means E65 measures the floor
   *for Pile-CC operators*. Whether $t^\*$ moves with the fitting corpus is untested and is the
   obvious follow-up.
3. **The battery is fixed and English-targeted**, 449 items over five admitted sets, one of which is
   multilingual with English targets. An early checkpoint may fail on the battery for reasons that
   have nothing to do with transport.
4. **Correlational only.** No intervention, no causal validation. See WHY.
5. **Pythia publishes one training run per size.** Seed variation here is over fitting-prompt draws,
   not over independent training runs, so nothing here speaks to run-to-run variability in $t^\*$.
6. **`step0` is a randomly initialised model with a *trained* tokenizer and embedding table** — not
   the same object as E61's randomized-block model, which holds the real embedding and unembedding.
   C5 compares them deliberately; they are not expected to be identical.

## COST

**Phase 0: $0 of GPU.** Bandwidth only — ~20 checkpoints × 0.8 GB (410M) + ~20 × 2.0 GB (1B)
≈ **56 GB**, deleted as it goes. CPU for the geometry.

**Phase 1:** `repro/20_cost_estimate.sh` — 0.17 GPU-h per 410M fit, 0.45 per 1B.

| arm | fits | GPU-h | \$ at \$0.82/h |
|---|---|---|---|
| 410M × 10 checkpoints | 10 | 1.70 | **1.39** |
| 1B × 10 checkpoints | 10 | 4.50 | **3.69** |
| provisioning + preflight | — | ~1.0 | ~0.82 |
| **total** | | **~7.2** | **≈ \$6** |

Under both gates. Phase 1 is not launched until Phase 0 has run and its window is recorded here.

## RESULT

### Phase 0 (2026-08-17, `results/e65_ckpt_geometry_410m.json`) — no geometry floor

19 checkpoints, `pythia-410m-deduped`, CPU. Controls: **C1b fires at 3.7e-09** (the final
checkpoint reproduces E39's stored 0.030666), **C2b fires** (concept-token id set identical at
every checkpoint).

| checkpoint | mean cos | eff rank |
|---|---|---|
| step0–step64 | −0.0000 | 889.3 |
| step128 | 0.0002 | 889.1 |
| step512 | 0.0245 | 868.7 |
| step1000 | **0.0434** | 833.5 |
| step16000 | 0.0387 | 752.9 |
| step143000 | 0.0307 | 725.9 |

**Effective rank never falls below 726, against 1 at both 70M and 160M.** The concept-token
unembedding is well-conditioned at every checkpoint including random initialisation. The
parameter-axis degeneracy has no training-axis analogue at this scale.

It also inverts the naive expectation: a **randomly initialised $W_U$ is the best-conditioned of
all** (cos −0.0000, rank 889), and the geometry gets slightly *worse* with training. So the
collapse at 70M/160M is a property of **small trained models**, not of undertrained ones.

⚠ **PRE-REGISTRATION DEFECT, FLAGGED NOT FIXED.** The registered rule flags a checkpoint degenerate
if mean cosine exceeds **0.031** (the 410M value), and 7 mid-training checkpoints do (0.033–0.043),
so the rule fired "GEOMETRY FLOOR EXISTS". **The threshold was mis-justified in this document**: it
was defended as "above it the readout is in the regime E38/E39 shows no transport can separate",
but that regime is cos ≥0.92 **with rank 1**, not "anything above the 410M value". The co-measured
effective rank settles it independently. The rule fired; the finding is not real. Recorded here
rather than silently corrected — **the operator rules on it.**

### Phase 1 (2026-08-17, `results/e65_ckpt_readout_{410m,1b}.json`) — UNCLEAR

10 checkpoints × 2 scales, N=200 on Pile-CC seed block 0, one L40S, ~$5.

**C1 DOES NOT FIRE at either scale, in the same direction and by the same margin:**

| scale | final checkpoint | stored reference | diff | tolerance | excess |
|---|---|---|---|---|---|
| 410M | 0.05023 | 0.04547 | 4.76e-3 | 3.7e-4 | **+10.5%** |
| 1B | 0.04486 | 0.04047 | 4.39e-3 | 2.15e-3 | **+10.9%** |

Diagnosed as far as the rule permits without altering the experiment:

- **Scoring reproduces.** The *stored* `lens_INSTREAM_Pile-CC_410m_n200_s0.pt` scored with t65's
  own scoring code gives **0.045092** against the reference 0.045466 — inside the tolerance. The
  scoring path is not the problem.
- **The difference is in the fitting.** `trainval.py` (which produced every stored INSTREAM lens)
  accumulates `jacobian_for_prompt` in its own loop and skips prompts that raise; `t65` calls
  `fast_fit`. E60 established `fast_fit == jlens.fitting.fit` on CPU at ≤8 threads; **nothing has
  ever established `fast_fit == trainval`'s manual accumulation**, and that is the gap.

**The data, which is internally consistent and must not be cited until C1 is resolved.** At both
scales the fitted operator is **beaten by the free logit lens at every checkpoint except the last**,
and clears both controls only at step143000. If that survives the C1 fix it says the J-lens's
advantage is a property of the *converged* model — a real finding for anyone using a transport lens
across training. It is also exactly the shape §2.4 says to treat as a bug report until proven
otherwise, and a failed C1 is such a bug.

## VERDICT

**Phase 0: NO GEOMETRY FLOOR** (with the threshold defect above flagged for the operator).
**Phase 1: UNCLEAR** — C1 fails at both scales; per the decision rule the run is not interpretable
and is not to be re-run with different settings without an operator ruling.

### Options for the operator, costed
1. **Re-run t65 using `trainval.py`'s fitter**, making the series continuous with the paper's
   stored cells. ~$5.
2. **Accept the run as internally valid** and re-baseline C1 against a `fast_fit` reference — the
   within-run comparisons all use one fitter, but the series is then not directly comparable to any
   number in the paper.
3. **Park it.** Phase 0 stands alone and is citable now.

## FLAGGED DELTAS

- Requires a GPU box for Phase 1, therefore a teardown obligation: pull + SHA-verify + mirror
  **before** `./lab down`.
- `--save-lens` on every Phase-1 fit, so any future band or aggregation question on this axis is a
  rescore rather than a refit. E62 was a paid refit for exactly the want of this flag.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e65_ckpt_geometry_410m.json` | 7,520 | `209935237e5fc4ae` | `t65_ckpt_geometry.py` | EXPOSED |

**Payload checksums** (content only, provenance block excluded):

* `e65_ckpt_geometry_410m.json` — `cea1549bbfefad0baac73eadb286493b`

<!-- END GENERATED PROVENANCE -->
