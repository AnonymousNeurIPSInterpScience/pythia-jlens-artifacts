# R1 — re-score the 8x8 fit x read grid at the correct readout token

**Output:** `results/e52_factorial_410m_rstrip.json`, `results/e57_grid_variance_ci_rstrip.json`

**PRE-REGISTERED 2026-08-19, before the `--rstrip` flag was added to `t52_factorial.py` and before
any run.** Source: [`../archive/POSTREVIEW_EXPERIMENTS.md`](../../archive/POSTREVIEW_EXPERIMENTS.md) §3 Tier A, item
R1. Supersedes nothing; it corrects [`E52_factorial.md`](E52_factorial.md) and
[`E57_grid_variance_ci.md`](E57_grid_variance_ci.md) on the scoring side only.

---

## QUESTION

The paper's headline is a variance decomposition of an 8x8 `fit corpus x read rung` grid:
**fit 91.2% / read 7.1%**. Every cell in that grid was read at a token that **does not occur in the
sequence the anchor's rule refers to**. Does the headline survive the correction?

## WHY IT IS A CORRECTION AND NOT A SENSITIVITY ARM

`jacobian-lens/data/evaluations/README.md` (vendored at `581d398`) specifies, for multihop,
multilingual and order-ops, that readout is at *"the token immediately preceding `target`"*. The
released prompts end in a trailing space. GPT-NeoX BPE absorbs that space into the target's leading
space, so it never survives as a token:

```
prompt tail  ' famously celebrated is the '        target 'Brazil'
  tokens of prompt+target, last 4   ' celebrated'  ' is'  ' the'  ' Brazil'
  stripped   readout token          ' the'   (id 253)   <- "the token immediately preceding target"
  unstripped readout token          ' '      (id 209)   <- DOES NOT OCCUR in prompt+target at all
```

`src/anchor_evals.py:32-34` already says so in the repository's own words — *"`.rstrip()` IS
LOAD-BEARING ... an un-stripped trailing space becomes the readout token and destroys the
baseline"* — and `:228` applies it. `experiments/t52_factorial.py:192` does not. The trap is that
`readout_position(tok, name, prompt)` **ignores `prompt` entirely** for the five "last" sets and
returns `-1`, so it looks as though it handles the prompt and does not; the caller must strip
before tokenising.

Scope: **157 of 551 released items = 157 of the 449 admitted-set items = 35%**, being 55/55
order-ops, 83/93 multihop, 19/107 multilingual, 0 on poetry/typo/association.

**Therefore the stripped arm is the estimate and the unstripped arm is legacy.** There is no branch
of this experiment in which the paper keeps printing a number computed at a token that does not
exist in the sequence its own rule refers to.

## RECORDED PREDICTION — logged before the run so that being wrong about it is visible

The grid is the **least** exposed of the readout-dependent quantities, for a structural reason: it
decomposes cell means in which the five eval sets are **averaged inside every cell**, so the
order-ops explosion (29.8x) never enters the `fit x read` factorisation the way it entered the
ladder's `corpus x set` one; and the recovery is ~90% uniform across corpora while a variance
decomposition is translation-invariant. **Prediction: 91/7 moves little; ACCEPT or QUALIFIED, not
REJECT.**

## DESIGN

`experiments/t52_factorial.py` gains `--rstrip`, implemented exactly as `t48_crossover.py:234`
implements it — `p = it["prompt"].rstrip() if a.rstrip else it["prompt"]`, with
`readout_position(tok, name, p)` then called on the same `p`.

Everything else is held: the same 24 matched-provenance operators
(`results/e48/lens_{INSTREAM_,OOD_}*_410m_n200_s{0,1,2}.pt`, all on local disk), band [9,21],
`K = [1,2,5,10,20,50,100]`, five admitted sets, 10 read rungs, 3 prefix seeds,
`--n-boot 4000 --n-perm 2000`, `--device cpu`. `t57_grid_variance_ci.py` then re-runs on the new
cells.

Outputs `results/e52_factorial_410m_rstrip.json`, `results/e57_factorial_cells_410m_rstrip.json`
and `results/e57_grid_variance_ci_rstrip.json`. **The unstripped files are not overwritten.**

## PRIMARY

`fit_pct` and `read_pct` under `persist` from E57, stripped versus unstripped.
Published (unstripped): fit **91.220%** [87.609, 92.171], read **7.133%** [6.311, 9.752].

## DECISION RULE — fixed before running, quoted verbatim

Quoted from `POSTREVIEW_EXPERIMENTS.md` §3, item R1:

> * **ACCEPT (headline robust):** stripped `fit_pct` exceeds stripped `read_pct` under **both**
>   aggregations, and stripped `fit_pct` lies inside the published bootstrap interval [87.609, 92.171]
>   under `persist`. The headline survives with its magnitude. The unstripped arm is reported once, in
>   the deviations table, as a corrected defect.
> * **QUALIFIED (ordering survives, magnitude does not):** stripped `fit_pct` still exceeds
>   `read_pct` under both aggregations but falls outside the published interval. **The paper's
>   ordering claim stands and its magnitude claim is restated at the corrected value.** The abstract,
>   Table `tab:split`, Figure 1 and the Conclusion all move to the stripped numbers. This is the
>   branch the bracket above makes most likely, and it is a rewrite, not a retraction.
> * **REJECT (ordering does not survive):** stripped `fit_pct` does not exceed `read_pct` under both
>   aggregations. **STOP. Flag for adjudication.** Under `CLAUDE.md` §2.9 nothing is re-thresholded; the
>   paper's central claim is false as stated and the submission is re-planned around the metric audit
>   and the scale floor (`POSTREVIEW_PAPER.md` §2, Plan C).

**This rule is not to be reinterpreted, and none of the three branches may be softened.**

## CONTROLS — each with the number it must produce

* **C1 — recovery of the published run.** The new code run **without** `--rstrip` must reproduce
  `results/e52_factorial_410m.json` to `max_abs_diff = 0.0` across all 128 stored cell values.
  *Required:* exactly 0.0. If it does not fire at exactly 0.0 the flag has changed something it
  should not have and the arm is void. (Shared with P0/C1.)
* **C2 — the convention actually bites.** The count of items whose readout token changes must be
  **157 of 551**, distributed 83/93 multihop, 19/107 multilingual, 55/55 order-ops, 0 on
  poetry/typo/association. *Required:* exactly those six counts, asserted inside the run and stored.
  If it is not 157 the two arms are not the two arms.
* **C3 — the logit-lens constant moves.** The identity-transport arm must differ between the two
  conventions. *Required:* under the unstripped path it is **0.0284395** (E33/E36 agree to 2.6e-10)
  on the `Q0` rung under `persist`; the stripped identity arm must differ from it. A stripped
  identity arm equal to the unstripped one means the flag did not reach the scoring path.
* **C4 — derangement floor holds.** The `shuf` arm must remain below every non-degenerate operator
  under `persist` in the stripped arm, as it does unstripped. *Required:* `C2_shuf_floor_excl_degenerate`
  fires, i.e. an empty `below_floor` list on every rung.

## DECLARED BIAS

1. The 24 operators were **fitted** on unstripped prompts (`trainval.py:253`) and R1 changes only
   the **scoring** readout. A fully anchor-faithful arm would refit. That is deliberate and it is
   the cheap half: it isolates the readout convention from the fitting convention. The fitting-side
   question is out of scope here and is recorded as open.
2. The operators are E48's matched-provenance refits, not the E28 ladder fits. That is already true
   of the published E52 and is why the two families' numbers are not interchangeable.

## COST

$0, CPU only. Runs after P0. Serial ~19 h; pooled, on the wallclock actually measured on this
machine at gate time (recorded in the results file), single-digit hours. Per `CLAUDE.md` §6.6 one
cell is timed before the full 30 are committed; if the observed per-pass cost exceeds ~2x the
slate's 86 s/pass calibration, stop and report rather than starting a two-day job.

## WHAT RE-DERIVES FROM THE NEW MATRIX, IN THE SAME PASS

Table `tab:split` (body), `tab:loo` and the 28 leave-two-out splits (`e54.matrix_loo`, `e55`), the
66/31 four-corpus restriction, Figure 1 (`fig0_grid_single`), the grid tables `tab:grid-persist` /
`tab:grid-min`, and E52's diagonal excess `D`. Mixing conventions across these is the failure mode.

## RESULT

Run 2026-08-19/20, `--device cpu --workers 6 --rstrip`, 30 cells in 95.4 min. Cells:
`results/e57_factorial_cells_410m_rstrip.json`; matrix: `results/e52_factorial_410m_rstrip.json`;
decomposition: `results/e57_grid_variance_ci_rstrip.json`.

### The primary

| | `fit_pct` | `read_pct` | fit > read |
|---|---|---|---|
| unstripped `persist` (published) | **91.220** [87.61, 92.17] | 7.133 [6.31, 9.75] | yes |
| unstripped `min` | 73.417 [68.87, 75.69] | 22.673 [20.29, 26.82] | yes |
| **stripped `persist`** | **50.697** [45.18, 54.19] | **48.132** [44.40, 52.95] | yes |
| **stripped `min`** | **53.353** [49.05, 56.96] | **44.609** [40.96, 48.29] | yes |

Stripped `persist` `fit_pct` = 50.697 lies **outside** the published interval [87.609, 92.171].

### Controls — every one fires

| control | required | observed |
|---|---|---|
| **C1** | the same code **without** `--rstrip` reproduces `e52_factorial_410m.json` at `max_abs_diff = 0.0` on 128 cells | **0.0** (P0 gate run) |
| **C2** | 157 of 551, as 83/93 · 19/107 · 55/55 · 0 · 0 · 0 | **157**, and each per-set count exactly as required |
| **C3** | the identity arm must move; unstripped is 0.0284395 | stripped **0.0831791**, a move of **0.05474** |
| **C4** | `shuf` below every non-degenerate operator under `persist` | fires, empty `below_floor` on every rung |

### THE MECHANISM — and it is not what anyone predicted

Shares move when denominators move, so the decomposition is reported in **absolute sums of
squares**:

| | SS_fit (row) | SS_read (col) | SS_resid | SS_total |
|---|---|---|---|---|
| unstripped `persist` | 2.899e-03 | 2.267e-04 | 5.234e-05 | 3.179e-03 |
| stripped `persist` | 2.452e-03 | **2.328e-03** | 5.662e-05 | 4.836e-03 |
| ratio | **0.85** | **10.27** | 1.08 | 1.52 |

**The fit axis did not weaken. The read axis became real.** SS_fit fell 15%; SS_read grew
**10.3×**. In margin terms the fit spread goes 0.02207 → 0.01717 (×0.78) while the read spread goes
0.00553 → **0.01665 (×3.01)**.

**Why the "translation-invariance" prediction failed, precisely.** The recovery is roughly uniform:
mean gain 0.06168, 13.9% non-uniform across fit corpora and 19.5% across read rungs. But the read
axis's **entire original spread was 0.00553**, while the correction's non-uniformity *across read
rungs* is **0.01201** — **2.17× the effect it perturbs**. A correction that is "≈90% uniform" is
still fatal to an axis whose whole effect is smaller than the correction's residual variation.
Translation-invariance protects an axis only when the shift is uniform *along that axis*; §2b.1
measured uniformity across **fitting corpora** and nobody measured it across **read rungs**.

The physical reading is straightforward and it is a point in the correction's favour: the
unstripped path read a bare space (id 209), a token carrying almost no context-dependent
information, so the 128-token read prefix appeared not to matter. At the anchor's actual readout
token the prefix matters a great deal.

### What else moved

* **The leave-k-out robustness collapses.** Published: 8/8 LOO and **28/28** leave-two-out under
  `persist`. Corrected: **6/8** LOO and **16/28** L2O under `persist` (6/8 and 20/28 under `min`).
  `results/r5_corpus_axis_uncertainty_rstrip.json`.
* **E57's own pre-registered rule now fires REJECT.** The bootstrap interval on `fit_pct −
  read_pct` under `persist` is **2.57 [−7.60, +9.80]**, which **includes zero**. E57's registered
  REJECT branch reads: *"The paper cannot claim the fit axis dominates; it can only claim a point
  estimate, and the abstract must be rewritten to say so."* That is a separate rule from R1's and
  it must be reported in its own terms.

### THE RECORDED PREDICTION WAS WRONG, and this is the record of it

§"RECORDED PREDICTION" above says: *"My prediction is that 91/7 moves little. ACCEPT and QUALIFIED
are both far more likely than REJECT."* The **branch** was called correctly — QUALIFIED — but the
**substance** was wrong: 91/7 did not move little, it moved to 51/48, a ratio of 12.8× collapsing
to 1.05×. The reasoning was wrong for an identifiable reason, given above: uniformity was
established on the fit axis and assumed on the read axis. Logged rather than quietly dropped,
which is the point of recording a prediction.

## VERDICT

**QUALIFIED — the ordering survives, the magnitude does not.**

Stripped `fit_pct` exceeds stripped `read_pct` under **both** aggregations (50.697 vs 48.132 under
`persist`; 53.353 vs 44.609 under `min`), so the ordering claim stands. Stripped `persist`
`fit_pct` = **50.697** falls **outside** the published interval [87.609, 92.171], so the magnitude
claim does not. Per the registered rule this is *"a rewrite, not a retraction"*: the abstract,
Table `tab:split`, Figure 1 and the Conclusion all move to the stripped numbers.

**The ordering now stands on much weaker ground than the point estimate suggests**, and the paper
must say so rather than restate 50.7 vs 48.1 as though it were 91 vs 7:

* the seed bootstrap on `fit_pct − read_pct` under `persist` **includes zero**;
* **12 of 28** leave-two-out splits reverse the ordering under `persist`, against 0 of 28 as
  published;
* the two axes are now within **2.6 percentage points** of each other.

No branch was softened and no threshold was re-cut. The unstripped numbers are legacy from here on:
there is no branch in which the paper keeps printing a number computed at a token that does not
occur in the sequence its own rule refers to.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e52_factorial_410m_rstrip.json` | 43,359 | `d95f684866c8d783` | `t52_factorial.py` | EXPOSED |
| `results/e57_grid_variance_ci_rstrip.json` | 7,566 | `2470403295c7e981` | `t57_grid_variance_ci.py` | INHERITED |
| `results/e57_factorial_cells_410m_rstrip.json` | 82,481 | `9e4ca28fc852c9a4` | `t52_factorial.py` | EXPOSED |
| `results/e52_factorial_410m.json` | 41,190 | `6e81f3cae37d0c19` | `t52_factorial.py` | EXPOSED |
| `results/r5_corpus_axis_uncertainty_rstrip.json` | 39,611 | `b89dbaee921e58e6` | `r5_corpus_axis_uncertainty.py` | INHERITED |

**Payload checksums** (content only, provenance block excluded):

* `e52_factorial_410m_rstrip.json` — `44d43319a7ec0b6145430f38f235084f`
* `e57_grid_variance_ci_rstrip.json` — `688d9a9b5c43fd637cdd1640b189b224`
* `e57_factorial_cells_410m_rstrip.json` — `8bd3566ee0a991bd9839c7a0ee2e17db`
* `e52_factorial_410m.json` — `de0fd855cefa636a23c70be9d4d9d62c`
* `r5_corpus_axis_uncertainty_rstrip.json` — `c3623e4ef8d516db6d44554a7153df37`

<!-- END GENERATED PROVENANCE -->
