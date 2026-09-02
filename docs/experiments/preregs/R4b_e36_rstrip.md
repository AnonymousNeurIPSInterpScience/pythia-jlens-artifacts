# R4b — re-score E36: S3's READ-axis rejection is a logit-lens comparison and is now unverified

**PRE-REGISTERED 2026-08-19, before `--rstrip` was added to `t36_qladder.py` and before any run.**
Source: [`../archive/POSTREVIEW_EXPERIMENTS.md`](../../archive/POSTREVIEW_EXPERIMENTS.md) §3 Tier B, item R4b.
Amends [`E36_qladder.md`](E36_qladder.md).

---

## QUESTION

`paper/CONTEXT.md` §3.1 records S3's first half as REJECTED on the read axis: the fitted operator is
*flatter* across the shift axis than the unfitted logit lens, **5 of 5**. That is a comparison
**against the logit lens**, produced by `t36_qladder.py`, which does not strip and has never been
re-scored — and the logit lens is the single quantity the readout defect moved most. Its constant
goes 0.02844 to 0.08318, a 2.9x change, while the fitted operators move 2.2-2.7x. **A comparison
between two quantities that move by different factors can reverse.** Until this re-runs, S3's
headline rejection is unverified.

E36 is separately implicated twice, and all three are fixed in one pass:

* `t36_qladder.py:217` builds its `shuf` arm from **Pile-CC's** operator, not from each corpus's
  own, so "a derangement beats Github's operator" is not the like-for-like comparison the prose
  describes;
* its control `C2_shuf_below_everything` records `fires: false` — **a failed pre-registered control
  that is discussed nowhere.**

## DESIGN

Add `--rstrip` to `t36_qladder.py:196` exactly as `t48_crossover.py:234` does it. Re-run the
Q-ladder over the same operators, same band, same rungs. Rebuild `JSHUF` **from each corpus's own
operator** rather than Pile-CC's. Output `results/e36_qladder_410m_rstrip.json`; the stored
unstripped file is not overwritten.

## PRIMARY

The number of read rungs on which the fitted operator is flatter than the logit lens, stripped
versus unstripped. Published: **5 of 5**.

## DECISION RULE — fixed before running, quoted verbatim

Quoted from `POSTREVIEW_EXPERIMENTS.md` §3, item R4b (Tier B):

> * **REJECT STANDS:** the fitted operator is flatter on >= 4 of 5 rungs. §3.1 is unchanged.
> * **REJECT OVERTURNED:** flatter on <= 1 of 5. **STOP and alert.** S3's first half is no longer
>   rejected on the read axis, which reopens the subthesis and changes the paper's framing.
> * **UNCLEAR:** 2 or 3 of 5. Report and stop. Do not re-cut.

## CONTROLS — each with the number it must produce

* **C1 — recovery of the published run.** Running without `--rstrip` and with the original
  Pile-CC-sourced `shuf` must reproduce the stored E36 arms **bit-exactly**. *Required:*
  `max_abs_diff = 0.0`.
* **C2 — the corrected `shuf` arm must lose to its own operator.** Built from each corpus's own
  operator, the derangement must sit **below** that corpus's real operator under `persist`, as the
  like-for-like evidence already on disk does: `e33_logit_baseline_410m_v2` gives Github 0.027926
  against its own `jshuf` 0.018142, and `e54`'s C2 fires at 0 of 15 and 0 of 120.
  *Required:* 0 corpora below their own derangement floor.
* **C3 — report `C2_shuf_below_everything` whichever way it fires.** A recorder, not a gate; its
  power is nil by construction and that is stated here rather than discovered later.

## DECLARED BIAS

Scoring-side only; the operators are unchanged. The fitting corpus for each operator is still the
unstripped pool, as in R1.

## COST

$0, CPU, hours.

## RESULT

Two full E36 runs, 2026-08-20, pooled (39 cells each): the stripped arm
(`results/e36_qladder_410m_rstrip.json`) and an unstripped arm re-run through the *same new code*
(`results/e36_qladder_410m_c1arm.json`). Adjudicated by `tools/r4b_e36_flatness.py` ->
`results/r4b_e36_flatness.json`.

### PRIMARY — the slope secondary

Number of fitting corpora on which the fitted operator is **flatter** across the shift axis than
the unfitted logit lens:

| aggregation | unstripped (published) | **stripped (corrected)** |
|---|---|---|
| `persist` | **5 of 5** | **1 of 5** |
| `min` | 1 of 5 | 3 of 5 |

Slopes under `persist`, corrected (linear in containment):

| arm | slope | flatter than logit? |
|---|---|---|
| logit (J=I) | **−0.00373** (was −0.00372) | — |
| J\|StackExchange | −0.00367 | yes |
| J\|Wikipedia_en | −0.00375 | no |
| J\|USPTO_Backgrounds | −0.00461 | no |
| J\|Pile-CC | −0.00489 | no |
| J\|Github | −0.00630 | no |

**The logit lens barely moved (−0.00372 → −0.00373). The fitted operators got three to six times
steeper**, from ~−0.001 published to ~−0.005 corrected. That is the same mechanism R1 found from
the other direction: at the anchor's readout token the read context genuinely matters, and it
matters *more* for a fitted operator than for the identity.

### Controls — all firing

| control | required | observed |
|---|---|---|
| **C1** | the unstripped arm re-run through the new code reproduces the stored E36 **bit-exactly** | **`max_abs_diff = 0.0`** over 364 values (13 rungs × 7 shared arms × 2 aggregations × 2 statistics) |
| **C1b** | the unstripped slopes reproduce `e53`'s stored slopes at `max_abs_diff = 0.0` | fires — so this tool provably computes the quantity `CONTEXT.md` §3.1 cites |
| **C2** | each corpus's **own** derangement loses to its own operator (R4d) | fires — **0** of 65 corpus×rung cells where a corpus's own derangement beats its own operator |
| **C3** | report `C2_shuf_below_everything` whichever way it fires | recorder, reported |

**R4d is thereby settled.** The stored `shuf` arm builds its derangement from **Pile-CC's**
operator and compares it to *every* corpus's operator, so "a derangement beats Github's operator"
was never like-for-like. Rebuilt per corpus, **no corpus's own derangement beats its own operator
anywhere**, matching the like-for-like evidence already on disk (`e33_v2`: Github 0.027926 vs its
own `jshuf` 0.018142; `e54` C2 at 0 of 15 and 0 of 120).

### The E36 PRIMARY is unchanged

`crossings`: **0 of 5** fitting corpora produce a curve strictly below the logit lens at any rung,
**stripped and unstripped alike**. E36's own primary REJECT of S3 on the read axis therefore
**stands**. What moved is the *slope secondary*.

### A wording ambiguity in the registered rule, flagged not resolved

The rule says *"flatter on >= 4 of 5 **rungs**"*. E36 has eleven-plus rungs and five **fit
corpora**, and the published 5-of-5 is over corpora — `CONTEXT.md` §3.1 lists exactly five slopes,
one per fitting corpus. Adjudicated over the five fit corpora, the only reading in which 5 is the
denominator. Reported rather than silently reinterpreted (`CLAUDE.md` §2.9).

## VERDICT

**REJECT OVERTURNED — FLAGGED FOR ADJUDICATION.**

Flatter on **1 of 5** under `persist`, which is the registered `<= 1 of 5` branch exactly. Per the
rule: *"S3's first half is no longer rejected on the read axis, which reopens the subthesis and
changes the paper's framing."*

**What this does and does not say, stated precisely, because the distinction is the whole result:**

* The evidence that **contradicted** S3 has evaporated. The published finding was that every fitted
  operator is *flatter* than the unfitted one — the opposite of S3's prediction. Corrected, **4 of 5
  are steeper**, which is the direction S3 predicts.
* But S3 is **not confirmed**. Its actual claim is a *crossover* — the fitted lens falling *below*
  the free baseline under shift — and that still never happens: **0 of 5** corpora cross, at either
  readout. The fitted operators are steeper and still strictly above J=I everywhere measured.
* Under `min` the picture inverts (1/5 → 3/5) **and the logit lens's slope changes sign**
  (−0.00107 → +0.00212). The result is aggregation-dependent and must not be quoted without saying so.

So: **S3's read-axis rejection was resting on a comparison that the readout defect had inverted**,
and it can no longer be cited in the form `CONTEXT.md` §3.1 states it. The subthesis is reopened,
not vindicated. No threshold was re-cut and no branch softened.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e36_qladder_410m_rstrip.json` | 111,957 | `6a40a858516cc564` | `t36_qladder.py` | EXPOSED |
| `results/e36_qladder_410m_c1arm.json` | 112,822 | `ab06a3488c667676` | `t36_qladder.py` | EXPOSED |
| `results/r4b_e36_flatness.json` | 18,228 | `5d90536ab71999f8` | `r4b_e36_flatness.py` | INHERITED |

**Payload checksums** (content only, provenance block excluded):

* `e36_qladder_410m_rstrip.json` — `2b970c1743f835d8ce5dffa6394e5c65`
* `e36_qladder_410m_c1arm.json` — `de912bf57dcf05f686d1596f921eba02`
* `r4b_e36_flatness.json` — `ad6aee81ac7bc8ab098176836f2ce4c3`

<!-- END GENERATED PROVENANCE -->
