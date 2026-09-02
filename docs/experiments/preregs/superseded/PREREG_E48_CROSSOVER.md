# PRE-REGISTRATION — the S3 crossover

**Authored by the operator, 2026-08-14, verbatim below. Committed BEFORE the required $J^P$ fits
exist and before any read is scored on any OOD corpus.** Parent commit `7a20a31`.

> **Naming.** The operator's text labels this "E47". **E47 is already taken** —
> `results/e47_ablation_pairs.json`, the disagreement-subspace knockout that returned REJECT. This
> document therefore registers it as **E48**. Nothing else is changed.

---

## THE RULE, VERBATIM AS AUTHORED

> E47 CROSSOVER — DECISION RULE (pre-registered, commit `7a20a31`)
> Conditions: matched n, length-matched 128 tok/prompt, persist aggregation,
> hierarchical CI (eval-set = replication unit, 3 disjoint seed blocks), fp32, TF32 off.
>
> CONFIRMED  iff all three hold:
>   (a) (J^P − logit) CI strictly > 0 for the in-stream English rungs
>       (Wikipedia, USPTO, Pile-CC, StackExchange); AND
>   (b) (J^P − logit) CI strictly < 0 for ≥2 of 3 genuinely-OOD rungs
>       (News_2024, arXiv_2023, CommonPile); AND
>   (c) those OOD operators read strictly ABOVE the J_shuf floor
>       (structured operator, wrong distribution — not a collapsed one).
>
> NOT REACHED  iff (J^P − logit) CI includes or exceeds 0 for all OOD rungs.
>   → Report straight as a positive-framed negative: the bias term's practical
>     magnitude is below the estimator worst case; the fitted operator tolerates
>     shift the theory permits it to fail on. Publishable. DO NOT reinterpret as
>     a weaker crossover or hunt for a rung where it dips.
>
> DEGENERACY (excluded either way)  iff OOD operators read BELOW J_shuf.
>   → Negative read is operator collapse, not distributional bias. Report
>     separately; does not bear on the crossover claim in either direction.
>
> MONOTONICITY  Spearman(read-advantage, −M2) across the full ladder, reported
>   WITH the partial controlling for M3. If M2 orders the read only by proxying
>   domain identity, the M3 partial kills it — report the partial, not the raw r.

**This rule is fixed. It is not reinterpretable after a result. The NOT REACHED branch is
explicitly a publishable outcome and will be reported as written.**

---

## WHY THIS IS NOW RUNNABLE, AND WAS NOT BEFORE

Every prior corpus in this programme is a Pile component, so the model was in-distribution
throughout and the design *could not* produce a crossover. M1 (`t35_containment.py`, full-stream)
changed that by **measuring** exposure rather than assuming it. At 25% coverage, k=32:

| tier | corpora | containment | growth 5%→25% |
|---|---|---|---|
| in-stream | Github .608, USPTO .408, Pile-CC .360, StackExchange .343, Wikipedia .332, PubMed .290 | high | **3.4–4.8×**, ~linear in coverage |
| partially present | OOD_Wikipedia_2023 .110, TRAP_FineWeb .026 | low but growing | 3.7–5.0× |
| **genuinely absent** | **OOD_News_2024 .0005, OOD_arXiv_2023 .0000, OOD_CommonPile .0002** | ~0 | **flat** |

The discriminator is the **growth rate**, not the level: present corpora are found proportionally
as coverage rises; absent ones stay pinned at zero. That is what licenses calling the three rungs
in (b) "genuinely OOD", and it is measured, not asserted.

**Note that OOD_Wikipedia_2023 is NOT eligible** despite being post-cutoff: it grows 5.0×, tracking
the in-stream corpora, because Wikipedia revisions share most of their text with the 2020 versions
that were in the Pile. Document-level temporal exclusion does not give textual novelty. It was
external review's primary recommended panel and the measurement disqualified it.

---

## PREREQUISITES — none of these exist yet

1. **$J^P$ fitted on the three OOD corpora** at 410M, 3 disjoint seed blocks each, $N$ matched to
   the in-stream fits, `--dim-batch 128`, fp32, TF32 off, penultimate target, `skip_first=16`.
   9 fits. This is the only compute the rule requires.
2. **$J^{\mathrm{shuf}}$ per OOD rung** for clause (c). Must be a **random derangement**, not a
   cyclic shift — E33 v1's control failed at 4/6 precisely because a shift-by-one gives each layer
   its nearest neighbour's Jacobian, the weakest member of the family.
3. **M2 and M3** for the monotonicity clause. In progress.

---

## FIVE THINGS FLAGGED NOW, WHILE THEY ARE STILL DESIGN QUESTIONS

Raised before the rule is executed, because after a result they would be reinterpretation.

**1. `persist` is specified; `min` is the anchor's own metric.** The vendored README defines
pass@k on **min-over-layers** rank, and the paper now reports min as primary with persist
alongside. The rule as written adjudicates on persist. That is defensible — min is existential by
construction — but a reviewer will ask why the crossover was not evaluated on the metric the
method's authors defined. **Recommendation: adjudicate on `persist` exactly as registered, and
report `min` alongside as a disclosed secondary.** No change to the rule.

**2. Github is excluded from clause (a).** The in-stream rungs listed are Wikipedia, USPTO,
Pile-CC, StackExchange — Github is absent. This is almost certainly deliberate (Github is the
high-leverage outlier in six prior analyses, and E33 measured it as *indistinguishable from not
fitting at all*, so it could not satisfy "CI strictly > 0"). **Recorded as intentional. Confirm.**

**3. Clause (b) requires a strictly negative CI, which is a strong bar.** E33's hierarchical
intervals at 410M under persist had half-widths around ±0.03 on read AUC values of ~0.03–0.05.
For an OOD rung to clear "CI strictly < 0" the point estimate must sit well below the logit lens,
not merely at or under it. **This is stated so a near-miss is not later described as "nearly
confirmed".**

**4. The three OOD corpora are not matched to each other on register.** News is ordinary English,
arXiv is LaTeX-heavy technical prose, CommonPile is a licensing-defined mixture. If only one rung
goes negative, register and exposure are confounded within clause (b). The "≥2 of 3" requirement
partially guards against this, and that appears to be its purpose.

**5. Clause (c)'s floor needs the same activations.** $J^{\mathrm{shuf}}$ must be scored on the
identical cached eval activations as $J^P$ and the logit lens, or the floor is not comparable.

---

## COST

9 fits at 410M × 200 prompts ≈ 1800 prompts. Measured 410M rate: 3.07 s/prompt on an L40S.
**≈1.5 GPU-h, ≈$0.30–0.60.** Scoring is CPU and free. Budget gate not triggered.

---

# AMENDMENT 1 — register/exposure de-confound (operator, 2026-08-14, references `408fa5c`)

**Verbatim as authored:**

> E48 AMENDMENT — register/exposure de-confound
>
> The three OOD rungs covary register with exposure. Clause (b) CONFIRMED is
> therefore insufficient alone: a negative on the two most register-distant OOD
> rungs is reattributable to §5's register effect, not to shift.
>
> ADD clause (d), required for CONFIRMED:
>   (d) The register-near/exposure-far rung (OOD_News_2024: English prose,
>       M1-absent) must be among the rungs going strictly < 0.
>       AND the register-far/exposure-near control (Github: code, M1-present)
>       must sit AT the crossover (CI includes 0) or above — NOT strictly
>       below. If Github goes strictly negative, register distance alone
>       reaches the crossover and the exposure attribution is not identified;
>       report as REGISTER-CONFOUNDED, not CONFIRMED.
>
> Rationale: (d) forces the negative to appear where exposure is absent but
> register is held ~English, and forbids the confound explanation where
> register is far but exposure is present. Only the joint pattern identifies
> shift-driven crossover distinct from the §5 corpus effect.

## The 2x2 this creates

|  | **register near (English)** | **register far (code/LaTeX)** |
|---|---|---|
| **exposure far (M1-absent)** | **OOD_News_2024** — must go strictly < 0 | arXiv_2023, CommonPile |
| **exposure near (M1-present)** | Wikipedia, USPTO, Pile-CC, StackExchange — clause (a), must be > 0 | **Github** — must NOT be strictly < 0 |

Exposure-driven crossover predicts the top-left cell goes negative. Register-driven crossover
predicts the bottom-right does. They are distinguishable only jointly, which is the amendment's point.

## Disclosure: Github's condition is ALREADY OBSERVED, so that half of (d) is not blind

`e33_logit_baseline_410m_v2.json` measured Github at **0.02808** against the logit lens's
**0.02844**, t = **−1.65** on three seed blocks — a CI that includes zero. That is precisely what
(d) requires of Github. The condition will be **re-measured** under E48's registered conditions
(3 disjoint seed blocks, hierarchical CI with eval-set as the replication unit), so the number is
not simply reused; but the *direction is known in advance* and this clause is therefore a
confirmation of seen data rather than an independent test. Recorded so it is not later presented
as a blind control.

The forbidding half of (d) — "if Github goes strictly negative, report REGISTER-CONFOUNDED" —
retains full force regardless, because it constrains an outcome that has not been measured under
these conditions.

## Operator answers to the five flags raised at registration

1. **persist vs min** — **both**. Adjudicate on `persist` as registered; report `min` alongside.
2. **Github excluded from clause (a)** — **confirmed intentional.**
3. **Clause (b)'s strong bar** — collect the data; rescoring is possible later.
4. **OOD rungs not register-matched** — **acknowledged; this amendment is the response.**
5. **Clause (c) floor on identical activations** — **required.** Implementation must cache the eval
   activations ONCE and score every arm (logit, each J^P, each J^shuf) against that same cache.
   A single scoring function, one activation set, no exceptions.
