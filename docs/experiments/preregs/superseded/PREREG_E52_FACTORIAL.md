# PRE-REGISTRATION — E52, the fit × read factorial: the missing cell

**Written 2026-08-15, BEFORE any E52 number exists.** Branch `pythia`. Parent commit `1a6f5df`.
Format is CLAUDE.md §3. Nothing here may be reinterpreted after a result.

---

## WHY

The operator's framing, verbatim, and it is the right one:

> "Can the J lens trained on an out of distribution sample read some latent properties of the
> model on an out of distribution corpus?"

Three of the four cells of the fit × read square are now measured and the fourth is not:

| | read on P (in-stream) | read on Q (out-of-stream) |
|---|---|---|
| **J fitted on P** | E28 / E33 ✓ | **E36 ✓** (REJECT S3) |
| **J fitted on Q** | E48 ✓ (NOT REACHED) | **← this experiment** |

Neither E36 nor E48 can answer the question, and the reason is structural. E36 held the operator
fixed and moved the read; E48 moved the operator and held the read fixed. The estimator framing's
last surviving sharp prediction lives only in the joint cell: **if the corpus effect is a MATCHING
effect — if what matters is the relationship between the fitting distribution and the read
distribution rather than the fitting distribution alone — then an operator fitted on Q should read
Q better than an operator fitted on P does.** Nothing measured so far can distinguish that from
"some corpora simply make better operators, full stop."

## DESIGN

### The factorial

| axis | levels |
|---|---|
| **FIT corpus** | 8: Wikipedia_en, USPTO_Backgrounds, Pile-CC, StackExchange, Github, OOD_News_2024, OOD_arXiv_2023, OOD_CommonPile — **× 3 disjoint seed blocks = 24 operators** |
| **READ rung** | Q0 (no prefix) + the same 8 corpora as Q-context prefix + SHUFFLED_Pile-CC = **10 rungs × 3 prefix seeds** |
| transports per cell | 24 fitted + logit (J=I) + J^shuf = **26** |

**Every one of the 24 operators was fitted by the identical code path** — `trainval.py`, N=200,
`--band 9,21`, `dim_batch=128`, `max_seq_len=128`, `skip_first=16`, `target_layer=-2`, fp32,
TF32 off. This is what the in-stream refits were produced for: E48's in-stream arm used
`fastfit.fast_fit` while its OOD arm used `jacobian_for_prompt`, and a factorial cannot tolerate
a fitter difference that is confounded with the axis being tested.

Read distribution is varied by **Q-context prefixing**, exactly as E36 defines and validates it
(E36's control C1 reproduced E33 to 2.6e-10, so the prefixing path provably does not disturb the
base measurement). Readout position is OFFSET, never recomputed.

### Held fixed
model `pythia-410m-deduped`, band `[9,21]`, the eval items and their target tokens, aggregation
(`persist` primary, `min` reported), K grid, readout rule, prefix length 128.

---

## PRIMARY

**The diagonal excess.** Over the 8 × 8 sub-matrix of read AUC `y[f, q]` (fit corpus `f`, read rung
`q`, both ranging over the same 8 corpora), remove the main effects:

```
y[f,q] = m + a_f + b_q + g[f,q]
```

`a_f` is "some corpora make better operators" (the E28/E33/E48 effect). `b_q` is "some read
contexts are easier". **`g[f,q]` is the matching term, and the diagonal `g[f,f]` is the question.**

```
D = mean over f of g[f,f]  −  mean over f≠q of g[f,q]
```

`D > 0` means fitting on the corpus you are about to read on buys something **beyond** operator
quality and read difficulty. That is the matching effect. `D ≈ 0` means the corpus effect is a
property of the fitting corpus alone.

Interval: bootstrap over the 3 prefix seeds and the 5 admitted eval sets, eval set as the outer
replication unit (`bootstrap.hierarchical_bootstrap_seeded`).

---

## DECISION RULE — fixed before running, not reinterpretable

- **MATCHING CONFIRMED** iff the CI on `D` is **strictly > 0**. The fit–read relationship carries
  signal the fitting corpus alone does not.
- **NO MATCHING (the null)** iff the CI on `D` **includes 0**. Then the corpus effect established
  in E28/E33/E48 is a property of the fitting corpus *per se*, the fit–read distance is not the
  operative variable, and EQ2's `‖J^P − J^Q‖` framing — which is a statement about a *gap* — is
  not what the data is measuring. **This is a publishable null and completes the negative.**
- **ANTI-MATCHING** iff the CI is **strictly < 0**: operators read corpora they were NOT fitted on
  *better*. Report separately; do not fold into either branch above.

**The `min` aggregation does not vote** (CLAUDE.md §6.0; and E48 measured the deranged operator
beating the real one on 103/120 draws under `min`). Reported, never decisive.

## SECONDARY — the operator's question in its literal form

For each of the three OOD read rungs `q`, with the matched OOD operator `J^q`:

1. `J^q` vs the **logit lens** on `q`. Answers "can a lens trained on an OOD sample read latent
   structure on an OOD corpus at all?"
2. `J^q` vs the **best in-stream** `J^P` on `q`. Answers "is it *better* to have trained on the
   OOD corpus?"
3. `J^q` vs its own **derangement floor** on `q`. Answers "is it structured or collapsed?"

These are reported per rung with seed intervals, and they are secondary: the pre-registered
adjudication is `D`.

---

## CONTROLS — each with the number it must produce

| id | control | must produce |
|---|---|---|
| **C1** | the Q0 column (no prefix) must reproduce **E48** for these same 24 operators | admitted-mean `persist` agreeing to < 1e-6. Tolerance derived, not tuned: one pair flipping one k moves a set mean by ≥ 1/(394·7) = 3.6e-4, so 1e-6 separates fp32 accumulation from a real difference. A mismatch means the factorial is not on the same scale as E48 and is void. |
| **C2** | `J^shuf` floor per rung | every *non-degenerate* operator above its own derangement floor. **Github is known to sit at/below its floor at every rung (E33 t=−1.65, E48, E36 11/11)** — that is an established result, not a control failure, and C2 is scored excluding it. Recorded here in advance so it is not an excuse later. |
| **C3** | capability — the model's own mean rank of the target at the final layer | no rung above 2× its Q0 value. E36 measured 4224.6 (Q0) → 4539.6 (token-shuffled), so this has headroom. |
| **C4** | **permutation null on the diagonal** — recompute `D` with the fit-corpus labels shuffled relative to the read rungs, 2000 draws | the null must be centred on ≈ 0 and the observed `D` compared against it. This is what makes "the diagonal is special" a measurement rather than an artifact of the decomposition. |
| **C5** | `D_act(J,J) = 0` | exactly 0.0 |

---

## DECLARED BIAS

1. **This is still Q-context prefixing, not a Q-drawn eval battery.** The concept items are the
   same English prompts at every rung; only the 128 tokens preceding them change. A matching
   effect that exists only for *whole-corpus* reads would not appear here.
2. **The diagonal is 8 cells.** `D` averages 8 matched cells against 56 unmatched ones. Eight is
   few, and one of the eight (Github) is a known-degenerate operator. `D` is therefore reported
   both with and without Github, and the claim is made only in the form that survives both.
3. **Three of the eight fit corpora are OOD and five are in-stream**, so the diagonal is not
   balanced across the exposure axis. The per-rung secondary is what speaks to the OOD half.
4. **410M only**, `persist` primary, `n=200` per fit.
5. **A null here does not resurrect S3.** E36 rejected S3 on the slope; a null on `D` says the
   corpus effect is not a matching effect, which is a *different* and additional negative.

---

---

# AMENDMENT 1 — held-out prefix pools (2026-08-15, BEFORE any E52 number existed)

The first launch was **killed after ~10 minutes and produced no number**, because a pre-run check
found a confound that would have manufactured the primary result.

**The defect.** E36's `load_pool()` draws Q-context prefixes from the *whole* corpus file.
`trainval.py` fits each operator on the first 200 qualifying documents of each of 3 seed blocks —
600 documents out of 2400. On the **diagonal** of this factorial the prefix corpus and the fitting
corpus are the same file, so prefix documents can be fitting documents. Measured, before the
result existed:

| corpus | prefix docs that are also FITTING docs |
|---|---|
| Pile-CC | 130 / 541, 142 / 541, 141 / 541 (prefix seeds 0/1/2) — **24–26%** |
| OOD_arXiv_2023 | 131 / 541, 139 / 541, 141 / 541 — **24–26%** |

A quarter of the diagonal's read activations would come from text the operator had already
averaged a derivative over. That elevates the diagonal for a reason that is **not** distributional
matching, and it points in exactly the direction that would produce MATCHING CONFIRMED.

**The fix.** The prefix pool for rung `X` is `X`'s documents **minus every document used to fit any
operator on `X`** (all three seed blocks, not only the ones scored). Verified disjoint: overlap
**0** for all eight corpora, with 541 held-out prefixes available in every case.

The exclusion depends only on the **rung**, so every operator reading a given rung sees identical
prefixes and the fit axis stays comparable. Only the diagonal's unfair advantage is removed.

**Nothing else changed.** The decision rule, the primary, the controls and the declared bias are
as registered above.

**Note for E36.** E36 (already adjudicated, REJECT S3) used the un-held-out pools. Its diagonal
cells carry the same overlap — 5 of its 55 fit×rung cells, one per fitting-corpus curve. E36's
verdict rests on the *slope across rungs*, so a single perturbed point per curve is unlikely to
carry it, but this must be checked and disclosed rather than assumed.

---

## COST

No refits — all 24 operators exist. Forward passes + transports only.
10 rungs × 3 prefix seeds = 30 activation caches; 26 transports each.
Measured reference: E36 did 39 cells × 7 transports in 2796 s on CPU.

**Estimate ≈ 1.4–2.2 h on local CPU, $0.** Budget gate not triggered.

---

# AMENDMENT 2 — three instrument corrections (2026-08-15, BEFORE any E52 number existed)

The second launch was also **killed early and produced no number**, after an adversarial review of
the design surfaced three defects. All three make the test **stricter or more honest**, never
easier to confirm. Recorded with the numbers that motivated them.

**(a) D is not a clean matching instrument — an additive decomposition of a bounded rate has a
link artifact.** For any `y[f,q] = phi(u_f + v_q)` with *no* matching term, a second-order
expansion leaves `g[f,q] = phi'' (u_f − ū)(v_q − v̄)`, so `D` picks up `phi'' · Cov(u_f, v_f)`
purely because operator quality and read-easiness are correlated **across the shared corpus
index**. Measured on E36's own 5×5 matrix: `corr(a_f, b_f) = +0.325`, and a purely
**multiplicative surrogate with the same margins — zero matching by construction — gives
D = +0.000280 against an observed +0.001214.** **23% of a positive D would be link artifact.**

→ New control **C6**: the surrogate `D` is computed and reported. **MATCHING CONFIRMED now
additionally requires `D` to exceed the surrogate null**, in both the all-8 and the
without-degenerate form. A `D` whose CI clears zero but not the surrogate is reported as
**LINK-ARTIFACT**, a new branch that is neither confirmation nor the registered null.

**(b) `without_degenerate` had no interval, so DECLARED BIAS 2 was not executable.** The
permutation null ran only on the all-8 matrix, so "the claim is made only in the form that
survives both" could not be checked. C4 now runs on both forms.

**(c) Per-set `D` was computed and discarded.** `adm` averages the five admitted sets *before* the
contrast, and E51 established a corpus×set interaction (6.8% of variance) larger than the corpus
main effect (1.9%), with signs that reverse per set. **Sign-opposed per-set matching therefore
cancels in `D`**, so a null on `D` is a null on the *set-averaged* contrast only. Per-set `D` and
its sign split are now stored, which makes that statement checkable instead of assumed.

**Bug fixed at the same time:** control C1 read `(x or 1) < 1e-6`. A *perfect* reproduction gives
`x == 0.0`, which is falsy, so `or 1` substituted 1 and the control reported DOES NOT FIRE on a
flawless match. A control that fails hardest when the result is best is worse than no control.

## Disclosure: E52's primary is PARTIALLY UNBLINDED for the in-stream half

`e36_qladder_410m.json` already contains a complete **5×5 in-stream fit × read matrix** (E28 N=400
operators, same aggregation, same admitted sets). Applying this experiment's `diag_excess` to it
gives **D = +0.001214**, all five diagonal residuals positive, permutation **p = 0.134** — positive
but **not significant**. That estimate was known before E52 ran and is recorded in the results file
under `PARTIAL_UNBLINDING_DISCLOSED`. It also carries ~17% fit/prefix leakage (E28 fitted N=400
from the same 800-document block), which is the confound Amendment 1 removes — so its sign is
exactly what leakage predicts, and it must not be read as independent support.
