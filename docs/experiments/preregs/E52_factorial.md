# E52 — the fit × read factorial: does reading inside the operator's own distribution help?

**Verdict: NULL. Significant under `persist` (p=0.0055) and not under the anchor's own `min`
(p=0.66, sign flips without Github); own-corpus rank is at chance under both. Reported as a null
(delta 5). Bad for the estimator framing, which predicted a matching effect.**

---

## QUESTION

E48 moved the **fit** distribution and held the read fixed. E36 moved the **read** distribution
and held the operator fixed. Both are marginal views. The estimator framing's last sharp
prediction lives in the **joint cell**: if the corpus effect is a *matching* effect, an operator
fitted on Q should read Q better than one fitted on P does.

Nothing measured before this separates that from "some corpora just make better operators."

In the operator's own words, recorded in the script: *"Can the J lens trained on an out of
distribution sample read some latent properties of the model on an out of distribution corpus?"*

## PRE-REGISTRATION

`docs/experiments/preregs/superseded/PREREG_E52_FACTORIAL.md` + **Amendments 1 and 2**, written before this ran.
Verbatim, from the script's own header:

> **PRIMARY** — the DIAGONAL EXCESS. Over the 8×8 matrix `y[f,q]` (fit corpus f, read rung q),
> remove main effects: `y[f,q] = m + a_f + b_q + g[f,q]`.
> `a_f` = "some corpora make better operators" (the E28/E33/E48 effect)
> `b_q` = "some read contexts are easier"
> `g[f,q]` = the MATCHING term. `D = mean(g[f,f]) − mean(g[f,q≠f])`.
>
> **RULE** — `CI(D)` strictly > 0 → MATCHING CONFIRMED.

**Amendment 1** is the one that matters: prefix pools must be **held out from every fitting pool**,
verified disjoint. Without it, 24–26% of a diagonal cell's prefix documents are documents its own
operator was fitted on — which would manufacture the result. (E36 does not carry this fix; see
that document.)

## DESIGN

`pythia-410m-deduped`, band [9,21], `persist` (`min` explicitly does not vote), five admitted sets,
CPU.

Eight corpora on the fit axis × three disjoint seed blocks = **24 operators, all from one fitter**
at `N=200`; the same eight corpora as read contexts × three prefix draws; 26 transports per cell.
The eight span both exposure tiers — five in-stream Pile components and the three measured-absent
corpora — so the diagonal includes matched *out-of-stream* cells, which is the novel cell.

## CONTROLS, and the number each produced

| control | required | produced | fires |
|---|---|---|---|
| **C1** Q0 reproduces E48 | exact | fires | **yes** |
| **C2** shuf floor, excluding the known-degenerate Github | below the real operator | fires | **yes** |
| **C3** capability | no model collapse across rungs | fires | **yes** |
| **C4** permutation | p < 0.05 on both all-8 and without-degenerate forms | **p = 0.0055** over 2000 draws | **yes** |
| **C5** self-distance | `D_act(J,J) = 0` exactly | fires | **yes** |
| **C6** link-artifact null: a multiplicative surrogate with the same margins, zero matching by construction | D_surrogate ≪ D | **+0.14×10⁻³** vs observed **+1.03×10⁻³** | **yes** |

C6 exists because an additive decomposition of a bounded rate has a link artifact — the surrogate
recovers 23% of E36's observed D purely from the link, so it must be subtracted conceptually
before D is believed.

## RESULT

**D = +1.03×10⁻³**, hierarchical CI **[+0.015, +2.772]×10⁻³**, permutation **p = 0.0055**.
Survives dropping the degenerate Github operator: **+0.99×10⁻³**.

Per eval set:

| set | mean D | note |
|---|---|---|
| `typo` | carries **84%** of D | dropping it takes D to **+0.20×10⁻³** |
| `multilingual` | +5.6×10⁻⁴ | |
| `multihop` | +2.7×10⁻⁴ | |
| `order-ops` | **−1.3×10⁻⁵** | effectively zero |
| `poetry` | **−2.9×10⁻⁶** | effectively zero |

`results/e52_factorial_410m.json`

## VERDICT

**MATCHING CONFIRMED** by the registered rule — the CI is strictly positive.

**Reported as DIRECTIONAL (REQUIRES REPLICATION AT A SECOND SCALE)**, because two facts bound it:

1. **One evaluation set carries it.** `typo` is 84% of D, and two of five sets are zero. `typo` is
   also the set where a *lexical* explanation is most available: the prompt ends in a misspelling
   and the target is the correct spelling, so a same-corpus prefix can help by supplying
   vocabulary rather than by matching a distribution.
2. **The size.** D is **3.7%** of the matrix span, and the matched out-of-stream operator still
   loses to the best in-stream operator by **−0.008 to −0.010 — eight to ten times D itself**.

*Fitting on what you read helps about a twentieth as much as fitting on a good corpus, and does
not come close to compensating for having fitted out-of-stream at all.*

## FLAGGED DELTAS

1. **The replication target is specific**: does `typo` still carry it at a second scale? If it
   does, there is a real effect with a nameable lexical mechanism. If it does not, D was noise in
   one set. This is the single most informative cheap follow-up in the programme.
2. **`PARTIAL_UNBLINDING_DISCLOSED`** is a field in the results file — read it before citing. Part
   of the matrix was visible before the analysis was finalised.
3. **A pre-existing provenance break, now resolved.** E52 declared `e48_crossover_410m.json` as an
   input and that file's bytes changed after E52 ran — because E48 was *given a provenance block*,
   not because its science changed. Verified from git: E48's `payload_sha256` is **b316c1440c6b634c**
   both before and after. The link was re-anchored on payload by
   `tools/migrate_provenance_paths.py --refresh-input-hashes`, which records the evidence in-file.
4. **410M only.**

5. **DOWNGRADED TO A NULL 2026-08-16 — the effect is aggregation-dependent.** Recomputed under the
   anchor's own `min`: **D = +5.5×10⁻⁴ at p = 0.66**, and **D = −3.2×10⁻⁴ (sign change)** when the
   degenerate Github operator is dropped. Under `persist` the multiplicative-surrogate null
   (+1.4×10⁻⁴), which has zero matching by construction, lies **inside** the confidence interval
   [+0.015, +2.772]×10⁻³. Independently, the full 8×8 grid shows the mean rank of an operator's own
   corpus is **4.12/8** (`persist`) and **4.62/8** (`min`) against a chance rank of 4.5, with only
   2/8 and 1/8 operators reading their own corpus best.
   **The paper now reports "reading your own corpus does not help" as a null**, which is what the
   two metrics agree on. `results/e54_aggregation_audit.json` → `e52`, `matrix_structure`.

## MEANING FOR THE PAPER

Supports §"Operator-relative axis" and abstract clause (2). The load-bearing sentence is the
*comparison*, not the significance: the operator-relative effect is real and is roughly a
twentieth of the identity effect. That is what makes "three distributions, only one of them
matters" a measurement rather than a slogan.

**It does not license** "matching does not matter" — the CI excludes zero and the permutation test
is clean. It licenses "matching matters, and it is small."

## PROVENANCE

| | |
|---|---|
| result | `results/e52_factorial_410m.json` |
| script | `experiments/t52_factorial.py` |
| module | `bash repro/exp/e52_factorial.sh` |
| tier | **A** — 25 inputs hashed |
| cost | free, ~1.3–2 h CPU |

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e52_factorial_410m.json` | 41,190 | `6e81f3cae37d0c19` | `t52_factorial.py` | EXPOSED |
| `results/e54_aggregation_audit.json` | 69,079 | `886885b41660dd01` | `t54_aggregation_audit.py` | RESCORED |

**Payload checksums** (content only, provenance block excluded):

* `e52_factorial_410m.json` — `de0fd855cefa636a23c70be9d4d9d62c`
* `e54_aggregation_audit.json` — `3d8af6bd411d5767e86f3de415d112a7`

<!-- END GENERATED PROVENANCE -->
