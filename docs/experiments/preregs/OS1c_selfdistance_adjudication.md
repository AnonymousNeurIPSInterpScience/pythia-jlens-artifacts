# OS1c — is OS1's C1 a failed control, or an unfireable one?

**Registered 2026-08-30, before the code that computes the result. Nothing below is written knowing
the answer for the 23 operators and the second metric it turns on.**

## WHY

`results/os1_operator_space_410m.json` is recorded **UNCLEAR**, and the reason is not its science.
Its PRIMARY stands: `SEP_d_rel = 9.309 [8.079, 10.443]` and `SEP_theta = 8.476 [7.520, 9.258]`.
Three of its four controls fired. It is UNCLEAR because **C1** demanded

> `d_rel(A,A) = 0` and `theta_k(A,A) = 0` **exactly**, for all 24

and `theta` measured `3.735905e-04`.

`theta_k(A,B)` is the mean principal angle between top-32 left singular subspaces:
`mean(arccos(svdvals(Ua.T @ Ub)))`. At `Ua == Ub` every singular value is 1, and **arccos has
infinite derivative there**, so `theta ~ sqrt(2*(1 - sigma))`. With `sigma` resolved only to float32
precision (`eps = 1.19e-07`) the floor is `~sqrt(2*eps) = 4.88e-04`. **A control demanding exact
zero from that expression cannot fire — not for these operators, not for any.**

`os1b_selfdistance_floor.json` already smoked this and found `3.736e-04` mean with a **random matrix
at `2.859e-04`**. But OS1b is explicitly `MEASUREMENT ONLY — no registration, no decision rule,
adjudicates nothing`, and it covers **one operator of 24 and one metric of two**: it never measured
`d_rel` at all. It is a pilot. It cannot retire a control whose registered scope was "for all 24, on
both metrics", and using a file that disclaims adjudication to adjudicate would be the exact move
`IMPLEMENTATION_RULES` rule 7 forbids.

This document registers the rule **before** measuring the other 23 operators and the entire `d_rel`
metric, so the retirement — if it happens — is a fired rule and not a re-reading.

**Precedent.** `E66b_determinism_floor.md` retired E66's UNCLEAR the same way: by registering a
decision rule whose branches said "E66's UNCLEAR is retired outright and its C1 is recorded as
mis-specified: it compared fp16 against fp32." That is the mechanism this repository already
accepts. OS1b simply never took the step.

## DESIGN

**Recomputes, does not re-measure the science.** No operator is refitted and no OS1 number is
recomputed. For each of the **same 24 operators OS1 used** (read from
`os1_operator_space_410m.json`'s own `scope`, not re-chosen), at `k = 32`, on CPU:

1. `theta_32(A,A)` and `d_rel(A,A)` in **float32** — the dtype OS1's C1 ran in.
2. The same two in **float64**, to show the value is arithmetic and not structure.
3. The same two on a **norm-matched Gaussian random matrix** `R`, i.e. `theta_32(R,R)`,
   `d_rel(R,R)`, to show the floor is a property of the expression rather than of these operators.

Band `[9..21]`, the rule `tests/test_band_rule.py` asserts. Storage dtype as on disk.

## PRIMARY

For each metric `m` in `{theta_32, d_rel}`:

    FLOOR_m   = max over the 24 operators of m(A,A) in float32
    F64_m     = max over the 24 operators of m(A,A) in float64
    RAND_m    = m(R,R) in float32
    PRED_m    = sqrt(2 * eps_float32) for theta_32 ; 0 for d_rel

## DECISION RULE — fixed before running, not to be re-cut

* **C1 IS MIS-SPECIFIED, OS1's UNCLEAR IS RETIRED** if, for **both** metrics:
  `FLOOR_m <= 10 * PRED_theta` (i.e. at or within one order of the predicted float32 floor,
  `4.883e-03`), **AND** `F64_m <= FLOOR_m / 100`, **AND** `RAND_m` is within a factor of 10 of
  `FLOOR_m`. All three must hold. OS1's C1 is then recorded **MIS-SPECIFIED — it demanded exact zero
  from a float32 quantity whose floor it did not price**; OS1's PRIMARY and its other three controls
  stand exactly as recorded, and OS1's verdict becomes whatever those three controls support.
* **OS1's UNCLEAR STANDS** if any operator's `m(A,A)` exceeds `10 * PRED_theta` on either metric.
  The self-distance is then larger than arithmetic explains and C1 caught something real.
* **UNCLEAR** in any other configuration — in particular if the random matrix does **not** reproduce
  the floor, which would mean the floor is a property of these operators after all. Report and stop.
  **Do not re-cut.**

**This document adjudicates C1 only.** It cannot and does not change `SEP_d_rel`, `SEP_theta`, their
intervals, or OS1's other three controls.

## CONTROLS — each with the number it must produce

| | control | must produce |
|---|---|---|
| **K1** | the random matrix reproduces the floor — the floor is arithmetic, not these operators | `RAND_theta` within 10x of `FLOOR_theta`. OS1b measured `2.859e-04` against `3.736e-04` on one operator; this must hold across the registered comparison |
| **K2** | float64 collapses it — the floor tracks precision, not structure | `F64_theta <= FLOOR_theta / 100`. OS1b saw a factor of `26962` on one operator |
| **K3** | **the metric can still fail.** `theta_32(A,B)` for two DIFFERENT operators must be far above the floor, or this measurement cannot distinguish "identical" from "different" and no verdict issues | `min over distinct pairs theta_32(A,B) >= 100 * FLOOR_theta`. **If K3 fails the instrument is blind and OS1c issues no verdict** |
| **K4** | scope is OS1's scope, not a friendlier one | the 24 operator paths equal `os1_operator_space_410m.json`'s `scope`, asserted before measuring |
| **K5** | band is the asserted rule | band `== [9..21] == int(0.38*24)..int(0.92*24)` |

K3 is the control that can fail and is included because `IMPLEMENTATION_RULES` rule 10 requires one:
a floor measurement that only ever reports small numbers cannot distinguish a real floor from a
broken metric. K3 constructs the input that makes it fail.

## DECLARED BIAS

1. **We already know the magnitude on one operator and one metric.** `os1b_selfdistance_floor.json`
   measured `theta` on `lens_INSTREAM_Pile-CC_410m_n200_s0.pt` and found `3.736e-04` mean with a
   random matrix at `2.859e-04`. That cannot be un-known. **23 of 24 operators and the whole `d_rel`
   metric are unmeasured**, and the thresholds above are set against the *predicted* floor
   `sqrt(2*eps)`, which is derived from float32 eps and not from OS1b's numbers.
2. **We want C1 retired.** The operator wrote, and the executing agent agreed, that a control
   demanding exact zero from `arccos` near 1 is vacuous. That is a motivated position, which is why
   the rule above is written with an explicit STANDS branch and a K3 that can void the whole thing.
3. **`d_rel` may behave differently from `theta`.** `d_rel(A,A) = ||A-A||_F / sqrt(...)` is an exact
   zero in exact arithmetic and has no `arccos` singularity, so it may well return exactly `0.0` in
   float32. If it does, C1's `d_rel` half was fireable and only its `theta` half was not — and the
   rule above then reports **UNCLEAR**, because the registered C1 required both.

## COST

Free. CPU, minutes. 24 stored operators, SVDs at `k=32` over a 13-layer band, no model loaded, no
refit, no GPU, no download. Tier **T1**.

---

## AMENDMENT — 2026-08-30, before any OS1c code was written or run

Two transcription defects in this registration, both found by reading OS1's *implementation*
against this document's description of it. **Neither touches the PRIMARY, the DECISION RULE, or any
threshold in it.** Both are recorded here, in a commit that precedes the code, because the
alternative — silently implementing something the registration does not say — is the move this
programme's register exists to make impossible.

### 1. K5's arithmetic is incomplete, and its literal reading names a band that does not exist

K5 as registered reads `band == [9..21] == int(0.38*24)..int(0.92*24)`. That equality is false:
`int(0.92*24) = int(22.08) = 22`, so the literal right-hand side is `[9..22]`.

The rule `tests/test_band_rule.py` actually asserts is `floor(0.38L)..floor(0.92L)` **intersected
with layers strictly below the penultimate target layer** (`l < L-2 = 22`), which gives `[9..21]` —
the band this document's DESIGN section names, and the band on disk. This is the identical omission
`CV8_positional_extrapolation.md` amended on 2026-08-29; the second clause was dropped in both.

**This cannot move a number.** Every stored operator carries `source_layers == [9..21]`, which
`experiments/os1_operator_space.py:63` asserts on load. Layer 22 is not present in any operator, so
the literal band is not merely a different choice — it is unrealisable. **K5 checks the full rule,
and the results file records both readings.**

### 2. OS1's C1 as *implemented* is narrower than this document describes it

This registration says C1 demanded `d_rel(A,A) = 0` and `theta_k(A,A) = 0` exactly, *"for all 24"*.
`experiments/os1_operator_space.py:190-194` does neither:

```python
k0 = f"Pile-CC|s0"                                    # ONE operator, not 24
self_dr = st.mean(d_rel(ops[k0][l], ops[k0][l]) for l in BAND)
self_th = st.mean(theta(subs[k0][l], subs[k0][l]) for l in BAND)
rec["C1_self_distance"] = {"required": "exactly 0 on both metrics", "d_rel": self_dr,
                           "theta": self_th, "fires": self_dr == 0.0 and self_th < 1e-6}
```

* **Scope.** C1 measured **1 of 24** operators. Its `required` string says "exactly 0 on both
  metrics" and its scope string implies the panel; the code evaluates `Pile-CC|s0` alone.
* **Threshold.** For `theta` the code tests `< 1e-6`, not `== 0`. That is *more* permissive than
  the registration's "exactly", and still **488x below** the predicted float32 floor
  `sqrt(2*eps32) = 4.883e-04`, so the substitution changes nothing about fireability.
* For `d_rel` the code does test `== 0.0` exactly, and it **passed**: `C1_self_distance.d_rel` is
  `0.0` as stored.

**Nothing in OS1c changes.** OS1c was registered to measure all 24 operators on both metrics
precisely because one operator and one metric cannot adjudicate a control whose stated scope is the
panel; this amendment records that the gap is wider than the registration realised, which
strengthens the reason for the experiment rather than weakening it. The results file reports OS1's
implemented threshold (`< 1e-6`) alongside its registered one (`== 0`) so a reader can see that the
verdict does not turn on which is used.

**The DECISION RULE, its three conditions, `PRED_theta`, the `10x` and `/100` factors, and K1–K4 are
untouched. No OS1c number has been computed at the time of this commit.**
