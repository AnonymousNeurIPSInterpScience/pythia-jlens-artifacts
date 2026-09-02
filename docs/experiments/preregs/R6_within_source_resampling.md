# R6 — within-source resampling: is "corpus" a factor, or a bundle?

**PRE-REGISTERED 2026-08-20, before any fit.** Source:
[`../archive/POSTREVIEW_EXPERIMENTS.md`](../../archive/POSTREVIEW_EXPERIMENTS.md) §3 Tier B, item R6.

---

## QUESTION

The single best experimental idea in either external review, and the one the paper's own
Limitations item 3 already concedes: *"A corpus bundles formatting, boundaries, token-frequency
profile, syntax and surprisal; we control prompt length and nothing else."*

If the fit-axis effect survives when the **source is held fixed** and only the **sample** varies,
the effect is not a bundle artifact. If it collapses, the paper's factor is confounded and must be
renamed.

**Why this matters more after R1, not less.** R1 cut the fit axis's *share* from 91.2% to 50.7%,
but in absolute terms the fit axis barely moved (SS ×0.85) — what grew was the read axis (×10.27).
So the fit effect is still the larger of the two and still needs to be shown to be a *source*
effect rather than a *sampling* effect. R6 is the experiment that decides that.

## DESIGN

The two largest in-stream sources by document pool: **Pile-CC** and **Wikipedia_en**. Four
**disjoint** sub-corpora per source at N=200, drawn from disjoint document blocks. Fit one operator
on each — 8 operators — and read all eight against the fixed battery at a fixed read context.
Decompose the read variance into between-source and within-source.

## PRIMARY

`between_source_SS / (between_source_SS + within_source_SS)` under `persist`.

## DECISION RULE — fixed before running, quoted verbatim

Quoted from `POSTREVIEW_EXPERIMENTS.md` §3, item R6:

> * **ACCEPT (corpus identity is a real factor):** between-source share >= 0.70.
> * **REJECT (the effect is sample noise, not source identity):** between-source share <= 0.30.
>   **STOP and alert** — the paper's factor is not what it is called.
> * **UNCLEAR:** anything between. Report the number and stop; do not re-cut it.

## CONTROLS — each with the number it must produce

* **C1 — document disjointness.** The four within-source draws must be verified document-disjoint.
  *Required:* **0** shared documents between any pair. `e58`'s disjointness machinery already
  exists and reports 24–26% overlap *without* the exclusion step, so it has demonstrated power.
* **C2 — shuffled-assignment null.** Relabel the eight operators' source at random and recompute.
  *Required:* the between-source share falls to its chance level, ~1/7 of total (0.143), and the
  observed share must exceed the null's 95th percentile.
* **C3 — derangement floor.** Each new operator must clear its own layer-deranged floor under
  `persist`, as all published 410M operators do on 120/120 draws. *Required:* 8 of 8.

**Power.** All three can fail. C2 in particular is a real null, not an identity: with 8 operators
in 2 groups a random relabelling has a well-defined non-zero expected between-group share, and the
test is whether the observed value exceeds it.

## DECLARED BIAS

1. **Two sources, not eight.** This tests whether *source identity* beats *sampling*; it does not
   test whether it beats *formatting*, which is R7.
2. The fitting pools are drawn under the same `require_full_window` filter, **whose selectivity
   differs by corpus** — the confound `SUSPICIONS.md` S-3 raises. Stated, not solved.
3. **Scored at the corrected (stripped) readout**, per R1. The published operators were scored
   unstripped, so R6's numbers are not comparable to pre-R1 figures and must not be placed beside
   them.

## COST

8 fits at 410M, N=200. `./lab cost 410m 200 L40S` ~ 0.17 GPU-h ~ $0.14 each, so **~$1.20 plus the
read** — call it **under $5**. Requires a rented GPU box: a 410M fit at N=200 is hours on this
laptop's CPU, so 8 of them is not a local job. **Needs the operator's sign-off before any spend.**

## RESULT

8 operators fitted on a rented RTX 4090 (2.0 s/prompt, 99% GPU util, TF32 asserted off in-process),
2026-08-20. Scored at the **corrected (stripped) readout**, 541 items, all five admitted sets.
`results/r6_within_source_410m.json` (GPU-scored) and `results/r6_within_source_410m_cpu.json`
(CPU re-score).

### PRIMARY

**between-source share = 0.9723** under `persist` (0.9886 under `min`).

| | value |
|---|---|
| SS between source | 8.380e-05 |
| SS within source | **2.391e-06** |
| Pile-CC mean read | 0.11148 (within-source SD 0.000682, range 0.001887) |
| Wikipedia_en mean read | 0.10500 (within-source SD 0.000365, range 0.000904) |
| between-source gap | 0.006473 |
| mean within-source SD | 0.000523 |
| **gap / within-source SD** | **12.4×** |

Four disjoint 200-document samples from the *same* source produce operators that read within
0.0009–0.0019 of each other. Two *different* sources are 0.0065 apart. **Sampling moves the read by
about a twelfth of what source identity moves it.**

### Controls — all three fire

| control | required | observed |
|---|---|---|
| **C1 — document disjointness** | 0 shared documents between any within-source pair | **0**, on all 12 pairs |
| **C2 — shuffled-label null** | observed share exceeds the null's 95th percentile; chance ≈ 1/7 = 0.143 | null mean **0.1403** (chance, as predicted), p95 **0.3519**, observed **0.9723** |
| **C3 — derangement floor** | every operator clears its own layer-deranged floor under `persist` | **8 of 8** (reads 0.1045–0.1124 against floors 0.0813–0.0863) |

C2's null landing at 0.1403 against a predicted chance level of 1/7 = 0.1429 is worth noting on its
own: the null behaves exactly as the design said it would, which is what makes the observed 0.9723
interpretable rather than merely large.

### Device cross-check (the D2 protocol)

Fitted and scored on CUDA, then re-scored on CPU from the same stored operators:
**0.9722598723 (GPU) vs 0.9722603430 (CPU), |diff| = 4.7e-07.** Both give the same verdict. The
D2 divergence does not reach this measurement.

## RE-ADJUDICATED UNDER THE 2026-08-20 AGGREGATION RULING

`docs/context/AGGREGATION_POLICY.md` makes **`min` primary** and demotes `persist`. R6's stored file
carries both; only which one leads changes.

| | `min` (**primary**) | `persist` (secondary, labelled) |
|---|---|---|
| between-source share | **0.9886** | 0.9723 |
| verdict against the >= 0.70 bar | **ACCEPT** | ACCEPT |

**The verdict is unchanged and the primary aggregation is the more favourable one.** The
`PRIMARY_between_source_share_persist` key in the stored results file is retained for provenance;
the number that leads is the `min` value in `by_aggregation.min.between_source_share`.

**One control does NOT move to `min`, deliberately.** C3 — every operator clears its own
layer-deranged floor — stays anchored on `persist`, because `min` is the aggregation that *prefers*
derangements (R9: the real operator beats every one of its own 15 derangement draws on 0 of 8
corpora under `min`). A `min`-anchored floor here would fail for the pathology already established
rather than for any defect in the operators under test, and would say nothing about R6's question.
This is recorded in `AGGREGATION_POLICY.md` §REPORTING RULES item 5.

## VERDICT

**ACCEPT — corpus identity is a real factor, not a bundle artifact of sampling.**

The between-source share is **0.9723**, against a pre-registered ACCEPT threshold of >= 0.70 and a
REJECT threshold of <= 0.30. It clears the accept bar by 0.27 and the shuffled-label null by a
factor of 2.8 on the p95.

**What this licenses, and what it does not.** It licenses calling the fit axis a *source* effect
rather than a *sampling* effect — which is the strongest single objection either external review
raised, and it is now answered by measurement rather than by argument. It does **not** license
calling it a "corpus" effect in the sense of *semantic content*: a source still bundles formatting,
markup, token-frequency profile and syntax, and this experiment holds all of them fixed together.
Separating *those* is R7's job, and R7 can only separate the lexical component.

**It also survives R1.** R1 cut the fit axis's *share* from 91.2% to 50.7%, but in absolute terms
the fit axis barely moved (SS ×0.85) — what grew was the read axis. R6 is measured entirely at the
corrected readout and finds the fit axis is a real source effect there, so R1's correction did not
hollow out the axis it shrank the share of.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/r6_within_source_410m.json` | 10,029 | `ce02c58032c3bcfd` | `r6_within_source.py` | EXPOSED |
| `results/r6_within_source_410m_cpu.json` | 11,261 | `e77ff95f4b673a09` | `r6_within_source.py` | EXPOSED |

**Payload checksums** (content only, provenance block excluded):

* `r6_within_source_410m.json` — `461f40c40324d6690adbab7c6aaf42f0`
* `r6_within_source_410m_cpu.json` — `2a4a53d520beb3bca0afc7185a96e943`

<!-- END GENERATED PROVENANCE -->
