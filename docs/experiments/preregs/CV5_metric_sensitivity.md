# CV5 — At the perturbation E66 found, is rank blind where z is sighted?

**Pre-registered 2026-08-23, before any number existed.**

## WHY

`results/e66_fitter_equivalence_cuda.json` compared three routes to the same operator:

| pair | operator `max_rel` | read difference |
|---|---:|---:|
| fast_fit vs trainval | 1.06e-07 | 0.0 |
| trainval vs **stored** | **2.535e-03** | **0.0** |
| fast_fit vs stored | 2.535e-03 | 0.0 |

Its C1 control **did not fire** and its verdict reads: *"the stored lens is not what its own
provenance says it is, which is a larger problem than either branch of this experiment."*

Two readings of the `0.0`:

1. **The operators are functionally identical.** A 2.5e-3 relative difference genuinely does not
   change what the lens reads, and the provenance mismatch is cosmetic.
2. **The metric was too blunt to see it.** E66 scored in **rank** space (`min` pass@k). CV3 has since
   shown rank is a coarse, threshold-crossing statistic that understates score-space effects by
   **2.5x** (`results/cv3_margins_410m.json`). A logit perturbation that moves nothing past a
   top-k threshold registers as exactly zero.

If (2), then E66's null is an artifact of the same pathology CV3 identified, its verdict needs
restating, and — because the flagged file is `results/e48/lens_INSTREAM_Pile-CC_410m_n200_s0.pt`,
**the exact base operator CV3 used** — the provenance question is live for a result we just won.

E66's arms A and B were **not saved to disk**, so this is not a re-score of E66. It is the isolated
question, asked directly.

## DESIGN

Take the stored operator. Perturb it by controlled relative magnitudes. Score each perturbation in
**both** spaces on identical cached activations, admitted-5, corrected readout, band [9,21],
flat-mean-7 k-summary — the CV3 harness unchanged.

Perturbation magnitudes, chosen to bracket the quantities that matter:

| `max_rel` | what it represents |
|---|---|
| 1.06e-07 | E66's fast_fit-vs-trainval gap — fp32 epsilon |
| **2.535e-03** | **E66's trainval-vs-stored gap — the one that produced 0.0** |
| 1.0e-02 | bf16-scale error, for the CV4 precision question |
| 5.0e-02 | a deliberately large perturbation; the metric must see this or the harness is broken |

Perturbation is elementwise Gaussian scaled per layer so that
`max|dJ| / max|J| = r`, applied to every band layer, three noise seeds per magnitude.

## PRIMARY

For each magnitude, the read shift in each space expressed **in units of that space's own pooled
seed SD**, taken from CV3 (`rank` 0.002581, `z` 0.013407):

```
S_rank(r)   = |read_rank(J+dJ)   - read_rank(J)|   / 0.002581
S_z(r)      = |read_z(J+dJ)      - read_z(J)|      / 0.013407
```

## DECISION RULE — fixed before running, three-way

At **r = 2.535e-03**, the magnitude E66 actually observed:

* **ACCEPT — RANK WAS BLIND.** `S_z >= 1.0` and `S_rank < 0.5`. z resolves a perturbation rank
  cannot see. **E66's `0.0` is a metric-coarseness artifact, its verdict must be restated, and the
  provenance question it raised is unresolved rather than benign.**
* **REJECT — THE OPERATORS ARE FUNCTIONALLY IDENTICAL.** `S_z < 0.5`. Both metrics agree the
  perturbation is immaterial; E66's reading (1) stands and the provenance mismatch is cosmetic for
  read purposes. Equally publishable, and it would also mean **bf16 fitting is probably admissible**,
  which unlocks the 6.9B/12B rungs of CV4.
* **UNCLEAR.** Anything else. Report and stop.

**The r = 5.0e-02 arm is a harness check, not a result:** if `S_rank` and `S_z` are both < 1.0 there,
the scoring path is not responding to the operator at all and every number here is void.

## CONTROLS

* **C1 — zero perturbation is a no-op.** `r = 0` must reproduce the unperturbed read to `0.0` in both
  spaces, exactly.
* **C2 — logit anchor.** The `logit_I` arm must reproduce **0.19810852520167826** (flat-mean-7,
  `min`), as in CV3. Ported unchanged.
* **C3 — monotonicity.** `S_z` must be non-decreasing in `r` across the four magnitudes, within noise
  seeds. A non-monotone response means the perturbation model is not doing what it claims.

## DECLARED BIAS

**Gaussian noise is not a fitter difference.** Two fitting routes differ *structurally* — accumulation
order, batching, device kernels — not by isotropic noise. A structured difference of the same
`max_rel` could move the read more or less than random noise of that magnitude. This experiment
therefore bounds *sensitivity to a perturbation of that size*; it does not reproduce E66's specific
operator delta. Stated so no result here is read as "E66's operators do/do not differ functionally."

The seed SDs used as denominators come from CV3's 5-corpus x 3-seed panel and are properties of that
panel, not of this operator alone.

## COST

CPU only, no refitting: ~13 arms x 449 items x 13 layers. **Under 10 minutes, $0.**

## OUTPUT

`results/cv5_metric_sensitivity_410m.json`.

## STATUS

**RUN AND ADJUDICATED — 2026-08-22. `results/cv5_metric_sensitivity_410m.json`.**

**REJECT — THE OPERATORS ARE FUNCTIONALLY IDENTICAL.** At E66's perturbation magnitude
(`max_rel` 2.535e-03) both metrics are quiet: S_rank 0.092, S_z 0.112. E66's `0.0` is real and was
not a rank-coarseness artifact, so CV3 stands on the operator it flagged. All four controls fired:
`C1_zero_perturbation_is_noop`, `C2_logit_anchor`, `C3_harness_responds`, `C4_monotone_in_r`.

**This REJECT also refuted an aside in this document's own pre-registration.** At bf16-scale error
the rank metric moves **1.834 pooled seed SDs**, so a REJECT does *not* license bf16 mid-ladder — a
precision change would confound precision with capability. `CV6_per_family_ladder.md` carries that
forward as an independent reason for fp32.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e66_fitter_equivalence_cuda.json` | 3,208 | `d6e9bd3cd0c9e942` | `t66_fitter_equivalence_cuda.py` | EXPOSED |
| `results/cv3_margins_410m.json` | 11,283 | `e5d2a4fb84fa96be` | `cv3_margins.py` | CLEAN |
| `results/cv5_metric_sensitivity_410m.json` | 5,478 | `7a5c2d241cc3d3db` | `cv5_metric_sensitivity.py` | CLEAN |

**Payload checksums** (content only, provenance block excluded):

* `e66_fitter_equivalence_cuda.json` — `48e1594bcc1da51992150cd8fcc2b99c`
* `cv3_margins_410m.json` — `38efe87c84be1b8bee5fa680a5a3d354`
* `cv5_metric_sensitivity_410m.json` — `cde8769ac592f81096a80090b47a40f1`

<!-- END GENERATED PROVENANCE -->
