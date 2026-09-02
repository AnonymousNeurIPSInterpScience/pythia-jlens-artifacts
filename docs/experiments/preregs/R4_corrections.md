# R4 — the corrections that are pure recomputation

**PRE-REGISTERED 2026-08-19, before any of the recomputes ran.** Source:
[`../archive/POSTREVIEW_EXPERIMENTS.md`](../../archive/POSTREVIEW_EXPERIMENTS.md) §3 Tier A, item R4.

> **Note on a colliding id.** The slate uses `R4b` twice: once in the R4 table for *the four-arm
> `t22` table* (finding F-5), and once in Tier B for *re-scoring E36*. This document owns the
> former, referred to throughout as **R4b-table**; the latter has its own document,
> [`R4b_e36_rstrip.md`](R4b_e36_rstrip.md).

---

## QUESTION

Eleven separate numbers in documents a reviewer will read are wrong, missing, or reported at a
coverage the text does not state. None of them needs a measurement — every input is already on
disk. The question each answers is only *"what is the number, actually?"*

## PRE-REGISTRATION

Quoted from `POSTREVIEW_EXPERIMENTS.md` §3, item R4, preamble:

> Grouped because each is minutes of work, each closes a named finding, and none has a losing branch.
> Each still gets a results file and a line in `RESULTS_TAXONOMY.md`; none may be reported from
> memory (§6.2).

**There is therefore no accept/reject branch on this document**, and none may be invented. Each item
below carries instead the *value it must produce to count as done*, taken from the slate, so that a
recompute that disagrees with the slate is visible as a disagreement rather than absorbed.

## THE ITEMS, each with the value that closes it

| id | finding | what is produced | the number that closes it |
|---|---|---|---|
| **R4a** | F-8 | containment `k=32` row recomputed at **20/20 shards** for every corpus, stored as one table keyed by shard count so the two columns can never be mixed again | post-cutoff Wikipedia **0.26958** (not 0.24); PubMed control **0.79273** (not 0.67); in-stream 0.918-0.934 and OOD 0.0001-0.0013 already 20/20 and unchanged |
| **R4b-table** | F-5 | one results file holding all four hierarchical intervals (410M/160M x `min`/`persist`, J-minus-logit and J-minus-shuffled) | the pattern must be visible: **the aggregation that beats the free baseline is the one that fails the derangement control, at both scales**. Unfavourable, and published |
| **R4c** | F-5, second half | all four aggregation arms (`min`, `persist`, `best1L`, `mean`) published from every ladder cell and from `e51`/`e54`/`e55` | the reported interaction share is currently the **argmax of four**; all four appear |
| **R4d** | F-7 | E36's `shuf` recomputed from **each corpus's own** derangement, plus `C2_shuf_below_everything` surfaced | like-for-like evidence on disk goes the other way decisively: `e33..._v2` Github **0.027926** vs its own `jshuf` **0.018142**; `e54` C2 at **0 of 15** and **0 of 120**. Measured in [`R4b_e36_rstrip.md`](R4b_e36_rstrip.md); recorded here |
| **R4e** | JUSTIFIABILITY | the control-power enumeration as a results file, each control marked in place, and the power question added to the experiment-spec template | **27 of 97** control fields could not have failed: 8 compare a quantity to itself, 6 are recorders with no gate, 4 are declared in a docstring and never coded, 9 are arithmetic identities. Includes E47's bottom-k control, computed, stored, never gated, and **would have failed 3 of 5** |
| **R4f** | M-2 | one-line fix in `src/provenance.canonical_payload_sha` (coerce keys to `str` before hashing) plus a regression test; the stamp extended to every results file the post-review work produces | re-keying `e58_algebra_audit`'s payload to int **reproduces the stored hash exactly**, so the defect is understood and the fix is safe. Only **29 of 214** non-sidecar results files carry a `payload_sha256` at all |
| **R4g** | M-3, M-4, F-11 | E28 sidecar provenance repaired: true corpus backfilled from `corpus_file`, gate recorded or explicitly marked unrecorded, all 166 hashed into `ARTIFACTS.md`, stale line 211 deleted, `./lab health` H3 fixed to compare a hash rather than a basename string | **166 of 166** sidecars record `"corpus": "wikitext"` (the `--corpus` default leaking) and `anchor_agreement_gate: null`; **0 of 166** operators are named in `ARTIFACTS.md` |
| **R4h** | M-7, M-16, M-20..M-25 | the small numeric corrections, each verified before it is written | effective-rank minimum **725.888** at `step143000` (so "never falls below 726" is false); "random better on one pair" is **2 of 5**, both Github pairs; `\|rho\| <= 0.28` is **0.2818**; "two sets exactly zero" are **-1.29e-5** and **-2.89e-6**; "~25%" should be **~17%** and the measurement was **0.1704**, in **two** live places — `CONTEXT.md:226` **and** `docs/context/RESULTS_TAXONOMY.md:66`; E52's `D` is a **redundant estimand plus a misleading gloss**, not a prereg break — `PREREG_E52_FACTORIAL.md:69` defines it verbatim as implemented and the C4 permutation p is computed on `D` itself so the 8/7 factor cancels. **Fix the gloss, not the number** |
| **R4i** | new, 2026-08-19 | the 1B claim re-derived from the corrected band, or the deviation stated | `e53_ladder_summary.bands_declared_vs_used` records for 1B `declared_by_the_papers_own_rule [6,13]`, `used_in_the_ladder [5,13]`, **`matches: False`**. The corrected-band recompute already exists as E62 (`results/ladder1b_b613/`, `e62_interaction_b613.json`) |
| **R4j** | new | the leave-two-out reversal published with its magnitude | `e54.matrix_loo.min.drop_both_extremes_github_and_uspto` = fit **42.011** / read **47.613** / residual 10.375, `fit_dominates_read: false` — **the read axis beats the fit axis by 5.6 points**. Also: the abstract's "28/28 leave-two-out splits" carries no aggregation qualifier and is `persist`-only |
| **R4k** | new | the item count relabelled | `t52_factorial.py` builds `items` from `EVAL_SETS`, not `ADMITTED`, so **541** is 551 released minus 10 unscorable association items. The admitted five hold **449** released items and **801** of the 893 pairs |
| **R4l** | C-1 | the "pre-registered" qualifier on `tab:predictors` either dropped, qualified, or the registration located | 6 of 45 registration claims are `UNVERIFIABLE` and **three are rows of `tab:predictors`** (E45, E47, E50 family). The conclusion is unchanged either way, because none of the twenty passes |

## CONTROLS

Each item's control is the recomputation itself against the value tabulated above: **a recompute
that does not reproduce the slate's stated value is a disagreement to report, not a number to
adopt.** Two items carry a genuine self-test beyond that:

* **R4f** — *required:* re-keying `e58_algebra_audit`'s payload to integer keys reproduces the
  **stored** hash exactly, **and** the fixed function is stable under a key-type round-trip. The
  regression test must fail against the unfixed function.
* **R4g** — *required:* `./lab health` H3 must **fail** when a sidecar's operator hash does not
  match, and pass when it does. A basename comparison cannot fail that way, which is the defect.

**Power, stated per R4e's own standard.** The R4 items are corrections, so most of their "controls"
are recomputations rather than gates, and that is declared here rather than discovered later. The
two above can fail; the rest cannot, by construction.

## DECLARED BIAS

R4a, R4c, R4d and R4j all read from files scored at the **unstripped** readout. Where R1 supersedes
the underlying measurement, the corrected value replaces it and both are stored; where it does not
(R4f, R4g, R4i, R4k, R4l are readout-independent), the correction stands on its own.

## COST

$0, CPU, one working day in total.

## RESULT

*(unrun at time of writing)*

## VERDICT

**COMPLETE — 11 items landed, 9 of 10 checkable values reproduce the slate exactly, 1 disagrees.**

`results/r4_corrections.json`, `results/r4e_control_power.json`, `results/r4g_e28_provenance.json`.

Agreeing exactly: R4a (containment 0.26958 / 0.79273 at 20/20 shards), R4b-table (no arm both beats
the free baseline and clears the derangement control), R4h M-7 (725.888), M-20 (2 of 5, both
Github), M-21 (0.28182), M-22 (-1.29e-5, -2.89e-6), M-22b (0.17043), R4i (1B survives its band
defect, 9/10 both ways), R4j (read beats fit by 5.60 pp under `min`), R4k (541 is the six-set count;
admitted five hold 449 items and 801 of 893 pairs).

**R4l DISAGREES and is reported, not adopted:** five candidate predictors rest on in-file docstrings,
not three, and E47 contributes no predictor row at all.

**R4e** found 36 of 92 control rows powerless and **6 controls named in a spec with no corresponding
key in any of that spec's results files** — including `PREREG_E36_QLADDER.md`'s **C4**, which the
pre-registration itself calls load-bearing. **R4f** fixed the `payload_sha256` round-trip defect with
a regression test that fails against the unfixed function. **R4g** backfilled 166 of 166 E28 sidecars,
hashed them into `ARTIFACTS.md` and re-verified every one against its bytes (0 mismatched), and
rewrote `./lab health` H3 from a basename substring test to a hash comparison, verified to fail on a
planted mismatch.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/r4_corrections.json` | 66,242 | `b36c8a2d68e0f5e2` | `r4_recomputes.py` | EXPOSED |
| `results/r4e_control_power.json` | 45,429 | `eea103f84232bb4a` | `r4e_control_power.py` | INHERITED |
| `results/ladder1b_b613/tv_Github_s0.json` | 14,052 | `d7cb38364bd55281` | `—` | EXPOSED |
| `results/ladder1b_b613/tv_Github_s1.json` | 14,032 | `a88461c788cb37d6` | `—` | EXPOSED |
| `results/ladder1b_b613/tv_Github_s2.json` | 14,074 | `ce7908dd1bb119a2` | `—` | EXPOSED |
| `results/ladder1b_b613/tv_Pile-CC_s0.json` | 14,023 | `8d30de01f9cc5fc7` | `—` | EXPOSED |
| `results/ladder1b_b613/tv_Pile-CC_s1.json` | 14,037 | `92d720e2c333aac8` | `—` | EXPOSED |
| `results/ladder1b_b613/tv_Pile-CC_s2.json` | 14,092 | `0cd4ab100c77f936` | `—` | EXPOSED |
| `results/ladder1b_b613/tv_StackExchange_s0.json` | 14,197 | `088ea6d343decfcb` | `—` | EXPOSED |
| `results/ladder1b_b613/tv_StackExchange_s1.json` | 14,149 | `f9ecf5e3b6908a01` | `—` | EXPOSED |
| `results/ladder1b_b613/tv_StackExchange_s2.json` | 14,179 | `a6a603770a0fd601` | `—` | EXPOSED |
| `results/ladder1b_b613/tv_USPTO_Backgrounds_s0.json` | 14,171 | `55ace8cf28534827` | `—` | EXPOSED |
| `results/ladder1b_b613/tv_USPTO_Backgrounds_s1.json` | 14,193 | `b5cb9f7de464affa` | `—` | EXPOSED |
| `results/ladder1b_b613/tv_USPTO_Backgrounds_s2.json` | 14,185 | `98026a284ade37f4` | `—` | EXPOSED |
| `results/ladder1b_b613/tv_Wikipedia_en_s0.json` | 14,187 | `5bf27fde005a80bd` | `—` | EXPOSED |
| `results/ladder1b_b613/tv_Wikipedia_en_s1.json` | 14,261 | `4154b0dabcbcf3d5` | `—` | EXPOSED |
| `results/ladder1b_b613/tv_Wikipedia_en_s2.json` | 14,257 | `26c635061f0a4326` | `—` | EXPOSED |

**Payload checksums** (content only, provenance block excluded):

* `r4_corrections.json` — `9250ac0181df4d119a80dee7cf7fd582`
* `r4e_control_power.json` — `56db754c22f378604b4440ee13ed21dd`

<!-- END GENERATED PROVENANCE -->
