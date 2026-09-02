# E57 — an interval on the paper's headline variance decomposition

**PRE-REGISTERED 2026-08-16, before the run.** Written against
`docs/validity/EXPERIMENT_VALIDITY_AUDIT.md` §3.6.

---

## WHY

The paper's single most-quoted number is `91.2% fit / 7.1% read / 1.6% interaction`
(`paper.tex:256-259`). It is a two-way decomposition of **64 cell means with one observation per
cell**. Three things follow and all three are defects:

1. The three shares **sum to exactly 100% by construction** (verified: 100.0000). There is no error
   term.
2. The "fit × read interaction" is **the residual**, conflating any true interaction with all cell
   sampling noise. It is not an interaction estimate.
3. **No interval is possible from the stored file.** Each cell is a mean over **9 draws** (3 fit
   seeds × 3 prefix seeds). `t52_factorial.py`'s `y()` returns `statistics.mean(...)` and only that
   mean reaches `by_aggregation[ag]["matrix"]`. The 576 raw draws existed in memory and were
   discarded.

E52's *matching* statistic D has a hierarchical CI. The paper's headline does not. At a venue whose
call names "measurement validity, identifiability, and evaluation design", shipping a headline
variance share with no interval — when the replicate structure existed and was averaged away — is
the finding a reviewer writes up.

**Secondary motive:** E52 has never been re-run. Nobody knows whether it reproduces.

## DESIGN

`experiments/t52_factorial.py --device cpu --cells-out results/e57_factorial_cells_410m.json`

The `--cells-out` flag is **purely additive**; with it unset the script is byte-identical in
behaviour to the run that produced `e52_factorial_410m.json`. No new measurement is introduced: the
per-arm scores were always computed, they were simply not serialised. Model, band [9,21], N=200,
24 operators from one fitter, 10 read rungs, 3 prefix seeds, 541 items, 5 admitted sets — all
unchanged.

Emits `draws[ag]["<fit>|<read>|s<fit_seed>|p<prefix_seed>"]` = admitted-set mean read AUC, plus the
`logit` and `shuf` arms per (rung, prefix seed). 64 cells × 9 draws × 2 aggregations, plus 60 arm
draws.

**Analysis (separate script, after the run):** bootstrap the variance shares by resampling the 3 fit
seed blocks and the 3 prefix seeds independently, recomputing the full two-way decomposition per
resample. Report a percentile interval on `fit_pct`, `read_pct`, and on `fit_pct − read_pct`.
Separately estimate the true interaction by subtracting the within-cell variance from the residual.

## PRIMARY

**The lower bound of the bootstrap interval on `fit_pct − read_pct`** under `persist`.

## DECISION RULE — fixed before the run

- **ACCEPT** — the interval on `fit_pct − read_pct` excludes zero under **both** aggregations. The
  paper's headline ordering carries an interval and the claim stands as written, now with one.
- **NARROW** — the interval excludes zero under `persist` but not under `min`. The headline is
  reported as aggregation-dependent, in the same sentence, exactly as §metric requires.
- **REJECT** — the interval includes zero under `persist`. The paper cannot claim the fit axis
  dominates; it can only claim a point estimate, and the abstract must be rewritten to say so.
- **UNCLEAR** — C1 does not fire (see below). Stop; nothing here is interpretable.

**This rule is not to be reinterpreted.** If the interval is wide, that is the result.

## CONTROLS

- **C1 — E52 REPRODUCTION.** The recomputed 8×8 matrix must equal the stored
  `e52_factorial_410m.json` matrix to **≤ 1e-12** on all 128 cells (64 × 2 aggregations).
  *Number required to count as firing:* `max_abs_diff ≤ 1e-12`.
  If C1 fails, the re-run is a different measurement and the draws are void — and separately, E52 is
  not reproducible, which is a much larger finding than this experiment.
- **C2 — the draws must average to the cells.** `mean over (s, ps) of draws[f|q|s|p]` must equal
  `matrix[f|q]` to ≤1e-12 by construction. A failure means the indexing is wrong.
  *(Checked in the analysis script, not the measurement script.)*
- **C3 — the bootstrap must not be degenerate.** With 3 fit seeds × 3 prefix seeds there are 9
  draws per cell; the resample must produce a non-zero spread in `fit_pct`. If the interval width
  is exactly 0, the resampling is broken.

## DECLARED BIAS

1. **9 draws per cell is a small replicate count**, and the two resampled factors have 3 levels
   each. A percentile bootstrap on 3 levels is optimistic about coverage. The interval will be a
   lower bound on the true uncertainty, not an upper one. Stated up front so a narrow interval is
   not over-read.
2. **The seed blocks are disjoint thirds of one corpus file**, so this interval covers
   fitting-sample and prefix-sample variation *within* a fixed corpus. It does **not** cover corpus
   sampling — the panel is still n=8, and the leave-two-out analysis (E55) is what speaks to that.
3. **The residual/interaction separation assumes the 9 draws are exchangeable within a cell.** Fit
   seed and prefix seed are different factors; the analysis will report the decomposition both
   pooled and with the two factors separated.
4. This re-run inherits every configuration property of E52, including that the read axis is a
   single 128-token prefix. E58 is the experiment that addresses that.

## COST

CPU only, on the laptop. Smoke (`--smoke`, 3 corpora, 1 seed, 48 items): **37.7 s** measured.
Full run scaled by cache count (84×) and arm-item count (440×): **estimated 3–5 h wallclock, $0.**
No GPU, no box, no teardown obligation.

## RESULT

*(unrun at time of writing)*

## VERDICT

*(pending)*

## FLAGGED DELTAS

- `t52_factorial.py`'s SHA changes with the added flag. The stored `e52_factorial_410m.json`
  records the *old* SHA in its provenance; that file is not modified and its chain stays intact.
  C1 is what establishes that the new script computes the same measurement as the old one.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e57_factorial_cells_410m.json` | 82,248 | `4c3b428670ba1219` | `t52_factorial.py` | EXPOSED |

**Payload checksums** (content only, provenance block excluded):

* `e57_factorial_cells_410m.json` — `8d15b390fc341469cbc37ca5c9e715db`

<!-- END GENERATED PROVENANCE -->
