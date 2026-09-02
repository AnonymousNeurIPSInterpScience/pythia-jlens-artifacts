# E66d — is `S` quantization-limited, or did the hardware just happen to match?

**Registered 2026-08-30, before the code that computes the result and before any E66d number
exists. Written knowing exactly one thing that motivated it: that `S` came back bit-identical on a
different day and a possibly different card. Nothing below is written knowing why.**

## WHY

`docs/handoff/E66_C_COMPLETION_SPEC.md` §3 is emphatic, and it was right to be:

> `S` = fp16(D1) vs stored — **no** — kernel-dependent, and that dependence *is* E66b's finding.
> **Therefore do not treat `S != 1.0986e-03` as a reproduction failure.**

The 2026-08-30 L40S re-run returned `S = 0.0010986328125`, **bit-identical** to the value stored on
2026-08-14. The same run measured `F' = 1.072884e-06` against a stored `1.028180e-06`, so the
device-order difference between the two runs is real and non-zero. A quantity declared
device-dependent reproduced exactly while a quantity declared device-dependent did not.

Two explanations, and they are not mutually exclusive:

* **(a) HARDWARE COINCIDENCE.** The 2026-08-14 box was also an L40S. The GPU model was never
  recorded — that absence is E66b's own finding — so this cannot be checked from the record.
* **(b) QUANTIZATION-LIMITED.** `S` compares **fp16(D1)** against an **fp16** stored operator. One
  fp16 step in the `[1,2)` binade is `9.765625e-04`. The measured kernel difference is `~1.07e-06`,
  **three orders of magnitude smaller**. If almost no entry of `D1` sits within `1e-06` of an fp16
  rounding boundary, then `fp16(D1)` is invariant to the kernel difference and `S` is stable *by
  construction*, on any card.

If (b) holds, the spec's caution is prudent but is not the operative reason, and the honest
restatement is bounded rather than absolute: *`S` is reproducible across devices as long as the
fitter's run-to-run difference stays below the fp16 rounding margin, which at `F' ~ 1e-06` it does
by three orders.* It would also mean the `1.0986e-03` gap is a property of the **stored artifact**,
not of the device — which is what E66b concluded, and this would be independent support for it.

**This experiment can establish sufficiency, not exclusivity.** If quantization suffices to explain
the reproduction, (a) may *also* be true and this design cannot rule it out — the GPU model of the
original run is unrecorded and unrecoverable. That limit is stated here, before the run, and the
verdict must carry it.

## DESIGN

**Recomputes only. No refit, no GPU, no model.** Inputs are the two operators already on disk:

* `results/lenses/misc/e66_D1_refit_410m_pilecc_s0.pt` — D1, **fp32**, sha256 `00eebd6ddffff675…`
* `results/e48/lens_INSTREAM_Pile-CC_410m_n200_s0.pt` — the stored operator, **fp16**,
  sha256 `4f00cc3d7450c364…`

Band `[9..21]`, the rule `tests/test_band_rule.py` asserts, intersected with layers strictly below
the penultimate target. 13 layers x 1024 x 1024 = **13,631,488 entries**. CPU, float32/float64.

`F'` is **read from `results/e66b_determinism_floor.json`**, not hardcoded here, so the perturbation
scale is the measured device difference and not a number chosen after the fact.

The perturbation is applied with **random signs at fixed seeds 0-4**, so the test is repeated five
times and a single lucky draw cannot carry it.

## PRIMARY

    S            = max over band of |fp16(D1) - stored|
    S_pert(s)    = max over band of |fp16(D1 + F' * sign_s) - stored|,  sign_s in {-1,+1}, seed s

for `s` in `{0,1,2,3,4}`, where `F'` is the measured `F_prime_changed_reduction_order`.

Mechanism, reported alongside and not adjudicated:

    FRAC(eps)    = fraction of the 13,631,488 entries whose fp16 image changes under +/- eps,
                   i.e. fp16(a+eps) != fp16(a) or fp16(a-eps) != fp16(a)

over the grid `eps in {1e-8, 1e-7, F', 1e-6, 1e-5, 1e-4, 4.883e-04, 9.766e-04}`.

## DECISION RULE — fixed before running, not to be re-cut

* **QUANTIZATION-LIMITED** if `S_pert(s) == S` **exactly, for all five seeds**, at `eps = F'`
  **AND** control Q2 fires. `S`'s reproduction is then explained by fp16 rounding rather than by
  hardware coincidence, and §3 of the completion spec is restated as a bounded claim rather than an
  absolute one. **(a) is not thereby excluded** and the verdict must say so.
* **NOT QUANTIZATION-LIMITED** if `S_pert(s) != S` for **any** of the five seeds at `eps = F'`.
  A device-order perturbation does move `S`, so the exact reproduction is not explained by
  rounding and hardware coincidence becomes the leading explanation. §3 stands exactly as written.
* **UNCLEAR** in any other configuration — in particular if Q2 does not fire, in which case the
  perturbation machinery is inert and **no verdict issues**. Report and stop. **Do not re-cut.**

**This document adjudicates the explanation of `S`'s stability only.** It cannot and does not change
`S`, `F`, `F'`, the floor, E66b's registered rule, its CONFIRMED DEFECT verdict, or any of C1-C5.

## CONTROLS — each with the number it must produce

| id | control | must produce |
|---|---|---|
| **Q1** | `S` recomputes from the persisted D1. If it does not, the object on disk is not what produced the reported `S` and nothing below is interpretable | `max\|fp16(D1) - stored\| == 0.0010986328125` **exactly**, the value the L40S run reported |
| **Q2** | **THE CONTROL THAT CAN FAIL.** The perturbation machinery is capable of moving `S` at all: at `eps` = one full fp16 ULP (`9.765625e-04`) the recomputed `S` **must** change for at least one seed. A test whose perturbation never moves the answer cannot distinguish "invariant" from "inert" | `S_pert(eps=9.765625e-04) != S` for at least one seed. **If Q2 does not fire, no verdict issues** |
| **Q3** | the ULP constant is the right one for this binade, reproducing E66b's C3 | `max\|stored\|` in `[1,2)`, so one fp16 step is `9.765625e-04` and the half-step is `4.8828125e-04` |
| **Q4** | the inputs are the recorded artifacts, asserted before measuring | D1 sha256 `00eebd6ddffff675…`, stored sha256 `4f00cc3d7450c364…`, both `source_layers == [9..21]` |
| **Q5** | `F'` is the measured value, not a chosen one | `F'` read from `results/e66b_determinism_floor.json:F_prime_changed_reduction_order` `== 1.072884e-06`, and it must be strictly less than the fp16 half-ULP |

## DECLARED BIAS

1. **I proposed the quantization explanation in writing before testing it**, and stated the
   mechanism (fp16 ULP three orders above `F'`) as the likely reason. That is a public prior and it
   is exactly the situation in which a thumb reaches the scale. The guards are that **Q2 can void
   the experiment outright**, the rule has an explicit NOT-QUANTIZATION-LIMITED branch, and the
   perturbation is repeated over five seeds rather than one.
2. **The quantization outcome is the more useful one for the programme**, because it converts a
   "this number cannot be compared across boxes" caveat into a bounded, quantified statement. A
   motivated preference for the more useful answer is declared here rather than discovered later.
3. **The arithmetic is close to predetermined and that is stated here.** Whether `1e-06` perturbs an
   fp16 rounding decision is nearly a matter of number theory, not of these operators, and This registration expects
   QUANTIZATION-LIMITED. That expectation is *why* Q2 exists: without a control that fires only
   when the machinery works, a foregone conclusion and a broken script produce the same output.
4. **This cannot exclude hardware coincidence** (see WHY). No result here licenses the claim that
   the 2026-08-14 box was or was not an L40S.

## COST

Free. CPU, minutes. Two stored operators, no model, no refit, no GPU, no download. Tier **T1**.

**OUTPUT:** `results/e66d_s_quantization_limited.json`.

---

## POST-RESULT DISCLOSURE — 2026-08-30, after the run. CHANGES NO RULE.

**The registered rule fired exactly as written and its verdict stands. The *mechanism* I asserted
in WHY was wrong, and this records that rather than leaving it to be found.**

WHY above supposed the reason S would be stable is that *"almost no entry of `D1` sits within
`1e-06` of an fp16 rounding boundary."* **That supposition is false.** The mechanism grid, which the
registration ordered reported-but-not-adjudicated, measured:

| eps | fraction of the 13,631,488 entries whose fp16 image changes |
|---|---|
| `1.000e-08` | 1.07% |
| `1.000e-07` | 8.50% |
| **`1.073e-06` = F'** | **46.35%** (6,318,563 entries) |
| `1.000e-05` | 98.37% |

Nearly **half** the operator's entries do change their fp16 image under a perturbation the size of
the measured device difference — and `S` is nonetheless bit-identical on all five seeds.

**The verdict survives because the PRIMARY was the perturbation test, not the boundary-density
supposition.** Had the rule been written on `FRAC(F')` — which was the intuition — it would have
returned NOT QUANTIZATION-LIMITED, and it would have been wrong. That is an argument for
adjudicating on the quantity you care about rather than on the mechanism you assume produces it.

**The mechanism that is actually consistent with the stored numbers.** fp16 spacing is
magnitude-dependent. `S` is a **max**, and `max|stored| = 1.208984` sits in the `[1,2)` binade where
one fp16 step is `9.765625e-04`; the measured `S = 1.0986328125e-03` is about `1.125` of those
steps. So `S` is set by **large-magnitude** entries, where the spacing is ~1e-03 and `F'` cannot
move a rounding decision. The ~46% of entries that do flip are **small-magnitude** ones, where the
local spacing is comparable to `F'` but whose differences are orders below the max and can never
attain it.

**This explanation is consistent with the stored numbers but was not separately measured.** It is
stated as the reading, not as a result. The clean follow-up, if it is ever worth the time, is to
record the magnitude at which `S` is attained and the local fp16 spacing there; no claim in the
results file depends on it.
