# E36 — the Q-ladder: does shifting the READ distribution favour the unfitted lens?

**Verdict: the ladder is FLAT — no fitting corpus crosses below the logit lens at any rung. The
registered *slope* secondary is WITHDRAWN (delta 6): it does not survive dropping one rung and the
baseline moves non-monotonically, so no slope on this design is interpretable as shift.**

---

## QUESTION

Subthesis S3 says the Jacobian lens should degrade under distribution shift *faster* than the
logit lens, because it is fitted and the logit lens is not. E48 moved the corpus the operator is
**fitted** on. This moves the distribution the activations are **read** from — the axis on which
the estimator framing makes its sharpest prediction, and the one nothing in this programme had
varied.

## PRE-REGISTRATION

Fixed **2026-08-13**, before any E36 number existed. Verbatim, from
[`../archive/prereg/PREREG_E36_QLADDER.md`](../../archive/prereg/PREREG_E36_QLADDER.md):

> **PRIMARY** — the crossing rung: the lowest containment at which a fitted operator's read falls
> strictly below the logit lens's.
>
> **ACCEPT S3** if any fitting corpus crosses below the logit lens at some rung.
> **REJECT S3** if no fitting corpus crosses at any measured rung.
>
> **SECONDARY** — *"S3's claim in its weakest testable form is that the J-lens slope is steeper
> than the logit lens's."*

The results file records `"prereg": "pythia/docs/experiments/preregs/superseded/PREREG_E36_QLADDER.md (2026-08-13, before any
E36 number existed)"` — the old path, because payload fields were deliberately not rewritten in
the 2026-08-15 repo flatten. The file is now at `docs/archive/prereg/`.

## DESIGN

`pythia-410m-deduped`, band **[9,21]**, `persist` aggregation, five admitted concept sets,
`N=200`, three prefix draws per rung, **thirteen rungs**.

The concept eval sets are not corpora, so "evaluate on Q" cannot mean swapping the eval set
without also swapping the task. Instead each item is read in a Q context:

```
[BOS] ‖ 128 tokens drawn from Q ‖ the item's own prompt
```

Targets are unchanged, so pass@k is comparable across rungs, and the prefix length matches the
fitting window so activations come from the same window statistics the operator was averaged
over. **The readout position is offset, never recomputed** — `poetry` reads at the last newline
and a Q prefix contains newlines, so recomputing the rule on the concatenated string would read
inside the prefix.

The x-axis is the **measured** containment of E48b, not a rung number. Rungs run from Q0 (no
prefix) through the five in-stream Pile components, a same-family control (PubMed 2023),
post-cutoff Wikipedia, a FineWeb trap, the three OOD corpora, and a token-shuffled Pile-CC.

## CONTROLS, and the number each produced

| control | required | produced | fires |
|---|---|---|---|
| **C1** Q0 reproduces E33's base measurement | to 4 dp | logit `persist` **0.028439503419** here vs **0.028439503163** in E33 — abs diff **2.6e-10** | **yes** |
| **C2** the layer-deranged operator sits below both `J^P` and logit at every rung | all 13 rungs | **false at all 13 rungs** | **NO** |
| **C3** capability: model top-k accuracy does not fall below 50% of its Q0 value | as registered | **the registered form is degenerate** — Q0's own top-10 rate is 0.0202, because the eval targets are latent concepts rather than emitted tokens, so the quantity is floored at Q0 and the rule cannot fire | **substituted** |
| **C3′** substitute: mean rank of the target, which is not floored | monotone, no collapse | **4224.6 → 4539.6** from no prefix to the token-shuffled rung | yes |

## RESULT

No fitting corpus produces a curve that crosses below the logit lens at any rung. The registered
secondary fails **in the opposite direction, five times out of five**:

| transport | slope d(AUC)/d(containment) |
|---|---|
| **logit lens (J = I)** | **−0.00372** |
| `J^P` Github | −0.00146 |
| `J^P` Pile-CC | −0.00115 |
| `J^P` Wikipedia (en) | −0.00094 |
| `J^P` USPTO Backgrounds | −0.00089 |
| `J^P` StackExchange | **+0.00022** |

Every fitted operator is *flatter* across the shift axis than the operator with nothing fitted.
The advantage itself is flat: on the four non-degenerate corpora it runs +0.009 to +0.022 at
containment 0.93 and +0.009 to +0.019 at containment 0.0001 — unchanged across four orders of
magnitude of shift in the read distribution.

`results/e36_qladder_410m.json`

## VERDICT

**REJECT S3**, in the registered vocabulary, with the file's own wording:

> no fitting corpus produces a J-lens curve that goes strictly below the logit lens at any
> measured rung. Shift in the READ distribution degrades both lenses together; the averaged
> derivative is no more shift-fragile than not fitting at all.

## FLAGGED DELTAS

1. **C2 did not fire, at any rung.** The file records two reasons, both prior results reappearing
   rather than a broken floor: Github's operator sits at or below its own floor at all eleven
   substantive rungs (established in E33), and under `persist` the deranged operator sits *at* the
   logit lens rather than below it. **This is a real weakening**: the REJECT rests on C1 plus a
   substituted C3.
2. **The registered C3 was degenerate and a substitute was used.** The file says so explicitly and
   says the operator is asked to confirm rather than reinterpreting silently. **Open: Found
   no record that this confirmation was given.** Under the programme's own rule — *never
   reinterpret a pre-registered decision rule; stop and flag it* — this needs an explicit ruling
   before E36 is cited as fully clean.
3. **Diagonal leakage, disclosed, unfixed.** E36 used prefix pools that were **not held out** from
   the fitting pools. Measured: **24–26%** of a diagonal cell's prefix documents are documents its
   own operator was fitted on. That is 5 of the 13 rungs (one point per curve). E52's Amendment 1
   fixed this for E52; **E36 still carries it, and the slope conclusion has not been rechecked
   without those points.** Cheap to close (~$0) and listed as open work.
4. **Scope.** This shifts the read distribution while holding `J^P` fixed. The bias term is
   ‖J^P − J^Q‖ where `J^Q` is the operator that *would have been fitted* on Q; E36 never fits it.
   E52 supplies that cell.
5. **410M only.** No second scale.

6. **WITHDRAWN FROM THE PAPER 2026-08-16 — the slope secondary does not survive.** The six slopes
   reproduce exactly, but they are an OLS on **raw** containment over a two-cluster design (nine
   rungs above 0.79, three below 0.07, one bridge at 0.27). Dropping the **single Github rung** —
   the maximum-containment point, degenerate by E33 — flips **three of five** fitted slopes to
   positive and halves the logit slope. Under the rank statistic **nothing orders with containment
   at all**: |ρ| ≤ 0.28 at n=11, p > 0.4 for every transport.
   Worse, the direction is wrong: the logit lens's **worst** read is at the most-contained rung
   (@Github, 0.02383) and its **best** at the least (@News24, 0.03684), and four of five fitted
   operators peak at containment 0.27. **Nothing in E36 degrades with shift**, so "REJECT S3" —
   which is a statement about the *shape* of a degradation — is not a claim this design can make.
   The paper now reports the flat ladder as a null and the non-monotone baseline as the reason no
   slope is interpretable. `results/e54_aggregation_audit.json` → `e36`.
7. **The C3 substitution is still unadjudicated** (see delta 2). It is now moot for the paper,
   which no longer rests on E36's registered secondary, but the pre-registration record should say
   so explicitly.

## MEANING FOR THE PAPER

Supports §"Exposure axis II" and the abstract's clause that shift on the read axis "degrades the
fitted lens and the logit lens *together*, and every fitted operator is *flatter* across that axis
than the operator with nothing fitted — the opposite of the pre-registered prediction."

**It does not license** "the Jacobian lens is invariant to distribution shift." The evaluation
*task* never moves; shift enters only through a 128-token prefix.

## PROVENANCE

| | |
|---|---|
| result | `results/e36_qladder_410m.json` |
| script | `experiments/t36_qladder.py` |
| module | `bash repro/exp/e36_qladder.sh` (`--dry-run` prints the contract) |
| tier | **A** — provenance block verifies; 7 inputs hashed |
| cost | free, ~1.5–2 h CPU |

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e36_qladder_410m.json` | 80,656 | `d0cc7b6381ac515f` | `t36_qladder.py` | EXPOSED |
| `results/e54_aggregation_audit.json` | 69,079 | `886885b41660dd01` | `t54_aggregation_audit.py` | RESCORED |

**Payload checksums** (content only, provenance block excluded):

* `e36_qladder_410m.json` — `1811c47b45a5b200ccd14b4795f47534`
* `e54_aggregation_audit.json` — `3d8af6bd411d5767e86f3de415d112a7`

<!-- END GENERATED PROVENANCE -->
