# E66b — THE FITTER'S DETERMINISM FLOOR AT 410M ON CUDA

**PRE-REGISTERED 2026-08-29, before any E66b code was written or run.**
Retires the programme's only open reproducibility UNCLEAR, or confirms it as a real defect.

---

## WHY

`results/e66_fitter_equivalence_cuda.json` is the one UNCLEAR that is about the *artifact* rather
than the science, and we release those artifacts:

> `B_vs_C max_rel 2.534676e-03` against `C1 rel_tolerance 1e-3` → `fires false`
> **VERDICT: UNCLEAR — the stored lens is not what its own provenance says it is.**

**That verdict is stronger than its evidence, and the evidence points somewhere specific.** Three
facts already on disk narrow it to one missing measurement.

1. **The two fitters agree with each other.** `A_vs_B max_abs = 1.19e-07`. Arm A (`fast_fit`) and
   arm B (the `trainval` loop) are the same to fp32 precision. Whatever differs, it is not the
   reimplementation.
2. **All three arms read identically.** `A_minus_B = B_minus_C = A_minus_C = 0.0`, exactly, and the
   Pile-CC seed SD is `1.84e-04`. No number in the paper moves.
3. **The stored operator is `float16` on disk** — verified 2026-08-29, and it lies exactly on the
   fp16 grid. `max|J|` over the band is `1.208984`, which sits in the `[1,2)` binade where the fp16
   ULP is `9.766e-04` and the half-ULP is **`4.883e-04`**.

So of the observed `B_vs_C max_abs = 1.1066e-03`, **storage dtype accounts for at most `4.883e-04`**
and a residual of **`R = 6.183e-04`** is unexplained. C1 compared an **fp16 artifact against an fp32
refit** under a `1e-3` tolerance that was never calibrated against either the storage dtype or the
fitter's own reproducibility. **That is the same defect this programme demoted E36 to Tier B for: a
cross-convention comparison judged against an assumed constant.**

`results/e60_fitter_determinism.json` already measured the mechanism, at 70M on CPU: the two fitters
agree **exactly** at 1, 4 and 8 threads and differ by `1.09e-04` above that, *the same order as the
released fitter's disagreement with itself* (`1.28e-04`). Its own reading: **"FITS ARE NOT
BIT-REPRODUCIBLE above 8 threads on this machine, which every stored lens SHA depends on."**

**E60 measured that floor at 70M on CPU. The stored operator was fitted at 410M on CUDA.** Nobody has
measured the floor there, and it is the number C1 needed. This experiment measures it.

## DESIGN

Pythia-410M-deduped, corpus **Pile-CC**, seed **0**, `N=200`, band `[9,21]`, `target_layer=-2`,
`skip_first=16`, `max_seq_len=128` — every setting matched to the stored operator
`results/e48/lens_INSTREAM_Pile-CC_410m_n200_s0.pt` and to E66's own arms. **CUDA, fp32, TF32 OFF,
set in torch and not only via the driver.**

Three fits in one session on one card:

| arm | what |
|---|---|
| **D1** | `fast_fit`, the stored configuration, fp32 retained (not cast to fp16) |
| **D2** | **byte-identical invocation to D1**, run again |
| **D3** | same, with a different `dim_batch` (128 against D1's default), which changes reduction order without changing the estimand |

## PRIMARY

**Does the stored operator reproduce, and if not, is the residual inside the fitter's own floor?**

    S  = max_abs( fp16(D1) , stored )        does the artifact reproduce exactly, at its own dtype
    F  = max_abs( D1 , D2 )                  the run-to-run floor, identical invocation
    F' = max_abs( D1 , D3 )                  the floor under a changed reduction order
    R  = 6.183e-04                           E66's unexplained residual, fixed above before running

## DECISION RULE — fixed before running, not to be re-cut

* **RESOLVED, THE ARTIFACT IS EXACT** if `S == 0`. The stored operator is bit-identical to a refit
  at its own storage dtype. E66's UNCLEAR is retired outright and its C1 is recorded as
  mis-specified: it compared fp16 against fp32.
* **RESOLVED, WITHIN THE FLOOR** if `S > 0` and `max(F, F') >= R`. The unexplained residual is
  inside the fitter's own run-to-run variation on this device, so the stored operator is what its
  provenance says to the precision this fitter can deliver. E66's UNCLEAR is retired and **replaced
  by a stated tolerance**, which is what `ARTIFACTS.md` should have carried all along.
* **CONFIRMED DEFECT** if `max(F, F') <= R / 3`. The residual is more than three times the floor, so
  reduction order does not explain it. The stored operators do not reproduce, must be refit before
  release, and the paper must say so.
* **UNCLEAR** if `R/3 < max(F, F') < R`. Report and stop. Do not re-cut.

## CONTROLS — each with the number it must produce

| id | control | must produce |
|---|---|---|
| **C1** | TF32 is actually off, asserted on the observed flag rather than assumed from the setter | `torch.backends.cuda.matmul.allow_tf32 == False` and `torch.backends.cudnn.allow_tf32 == False` |
| **C2** | the refit is the same estimand: D1's reads equal the stored operator's reads | `read_diff == 0.0` under `min` and `persist`, reproducing E66's `A_minus_C = 0.0` |
| **C3** | the fp16 arithmetic is what we claim: half-ULP at `max|J|` | `max|J|` in `[1,2)` and half-ULP `== 4.883e-04` |
| **C4** | the floor is not trivially zero, i.e. the measurement can detect anything at all. If `F == 0` **and** `F' == 0`, the card is fully deterministic and the RESOLVED-WITHIN-FLOOR branch **cannot be taken**; only the exact or defect branches remain | reported either way; a zero floor forces adjudication on `S` alone |
| **C5** | same prompts: D1's prompt pool hashes equal the stored run's | pool SHA-256 identical to `tv_INSTREAM_Pile-CC_s0.json`'s inputs where recorded |

## DECLARED BIAS

**We expect RESOLVED, and that is the convenient answer.** It lets the released artifacts stand and
converts an UNCLEAR into a tolerance. Declared because the convenient branch is the one to distrust.

**C4 is the guard.** If the card turns out to be fully deterministic (`F = F' = 0`), the
within-the-floor branch is unavailable by construction and the result must be adjudicated on `S`
alone, which can only return "exact" or "defect". The rule cannot quietly land on the comfortable
answer by way of a floor that was never really measured.

**A CONFIRMED DEFECT is publishable and would be reported.** It would mean the released operators
must be refit — expensive and late, but the alternative is releasing tensors that do not reproduce.

## COST

**GPU, and small.** The stored run records `runtime_s: 496.3` for one `N=200` 410M fit on CUDA, so
three fits is roughly **25 minutes of card time**. On an L40S (fp32 91.6 TFLOPS; choose by fp32
TFLOPS, L40S 91.6 > 4090 82.6 > H100 67) that is well under **$1**, with provisioning and teardown
perhaps 45 minutes wall clock. Far below the $40 / 6 h budget gate.

**Teardown obligation applies**: pull and SHA-verify before the box is destroyed; `./lab down`
refuses without a verified pull receipt.

**OUTPUT:** `results/e66b_determinism_floor.json`.

## AMENDMENT 2 — 2026-08-29, before the registered run

`dim_batch=128` OOMs a 24 GB RTX 4090 under torch 2.11: the estimator replicates the prompt
`dim_batch` times and retains the graph. **D1/D2 now use 32 and D3 uses 64.**

`dim_batch` changes only the order in which rows of `J` are accumulated, not the estimand, and the
floor being measured (`D1` vs `D2` at an identical setting, `D1` vs `D3` at a different one) is
well defined at any value. `S` — does a refit reproduce the stored operator at its own dtype — is
unaffected. **No threshold moves.** Recorded because the registration named a specific value.
