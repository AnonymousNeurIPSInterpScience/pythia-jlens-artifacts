# PRE-REGISTRATION — E38–E42: is the SLM read failure a property of $J$ rather than $W_U$?

**Written 2026-08-14, before any of these numbers exist.** Format is CLAUDE.md §3. A REJECT or
UNCLEAR outcome stops the experiment and goes to the operator; no rule below may be reinterpreted
after a result.

---

## WHY

Two things forced this reframing, both from the operator, both correct.

**1. E37's headline was vacuous.** If $W_U$ has rank 1, then $W_U = u s v^\top$ and
$W_U(J h) = u \cdot s(v^\top J h)$: every token's logit is the *same vector* $u$ scaled by one
scalar, so the token **ranking** is fixed by $u$ alone, identically for every $J$ and every $h$.
"Gap $= 0$ at $r=1$" is therefore algebra, not evidence. It was reported as an observation. It is
retracted as one.

**2. Mutating $W_U$ does not cleanly drive the effect.** E37's intermediate ranks show no monotone
dose-response, and at 410M the verdict flips with the *fitting corpus* (USPTO ACCEPT, ratio 0.214;
wikitext UNCLEAR, ratio 0.729) even though $W_U$ is identical in both runs. A property of $W_U$
cannot depend on which corpus $J$ was averaged over.

The readout is $v_t = J_\ell^\top W_U[t]$, so it composes exactly two objects. If $W_U$ is not the
driver, the mechanism is in $J_{avg}$.

**And the obvious $J$-side hypothesis is already dead.** The natural guess — averaging washes the
Jacobian out at small scale — is **inverted** by our own data. Using the Frobenius dispersion
identity, $\|E[J]\|_F^2 / E\|J\|_F^2 = 1/(1+\mathrm{disp})$:

| model | disp($L_0$) | $\|E[J]\|^2/E\|J\|^2$ | lens |
|---|---|---|---|
| 70M | 0.229 | **0.814** | fails |
| 160M | 0.641 | **0.609** | *hurts* (CI excludes 0, negative) |
| 1B | 1.736 | 0.365 | works |
| 410M | 2.364 | **0.297** | works, +75% on best corpus |
| 2.8B | 4.418 | **0.185** | — |

The mean survives averaging *best* exactly where the lens is *useless*. So the SLM failure is not
an estimation-quality problem. $E[J]$ is cleanest where it is least informative, and the question
becomes what $E[J]$ **is** at small scale.

**Standing prior, stated so a positive is not over-read:** thirteen predictors of the corpus effect
have already failed leave-one-out. The difference here is the hypothesis class — those were scalar
summaries of a corpus or of $J$'s own spectrum; these are structural properties of $J$ *relative to
the identity and to $W_U$*, the two objects the readout actually composes. That is a different urn,
but the base rate justifies a strict control.

---

## THE PRE-REGISTRATION BOUNDARY (binding)

`PREREG_PYTHIA_T7_v2.md` §2 registers `pythia-{1b,1.4b,2.8b}-deduped` as a confirmatory read set;
1b is already contaminated (§0.2b), **1.4b and 2.8b are clean and scoring them consumes the test.**

Therefore, split by whether the measurement touches the eval outcome:

| experiment | touches eval battery? | scales run |
|---|---|---|
| **E38** ($J$ vs scaled identity) | **no** — pure $J$ geometry | 70M, 160M, 410M, 1B, **1.4B, 2.8B** |
| **E41** (layer-dependence of $J$) | **no** — pure $J$ geometry | 70M, 160M, 410M, 1B, **1.4B, 2.8B** |
| **E39** (concept-vector geometry) | yes — uses eval concept tokens | 70M, 160M, 410M, 1B only |
| **E40** ($J$'s action on real activations) | yes — uses eval activations | 70M, 160M, 410M, 1B only |
| **E42** (corpus coupling) | yes | 410M only (the only scale with per-corpus lenses) |

**No pass@k, n_win, or read AUC is computed at 1.4B or 2.8B anywhere in this document.** E38/E41
read only the lens matrices and the identity. That does not observe the primary outcome and does
not burn the test.

---

## DESIGN, common to all

**Lenses.** Cross-scale arm uses `lens_{70m,160m,410m,1b,1.4b,2.8b}_n200_db128_pen.pt` — the
penultimate-target, wikitext, $N{=}200$ lenses, i.e. **corpus- and recipe-matched across scale**,
which is the comparison E37's first run failed to make. Per-corpus arm (E42) uses the E28
`e28_{corpus}_410m_n400_s0.pt` lenses, 5 corpora.

**Band.** Relative depth 0.35–0.85, source layers strictly below the penultimate target — the
convention used throughout. Bands differ in length across scale by construction; every statistic
below is either per-layer or normalised, so no cross-scale comparison depends on band length.

**Declared bias.** All statistics are computed on a *single* lens per (model, corpus) cell, so
none carries a seed interval except where E28 provides three seeds (E42). Cross-scale
comparisons therefore have no error bar and are **descriptive**; only E42 supports inference.

---

## E38 — Is $J_{avg}$ a scaled identity at small scale?

**WHY** The sharpest hypothesis available, and it needs no rank story. If $J \approx cI$ then
$v_t = J^\top W_U[t] = c\,W_U[t]$: the J-lens **is** the logit lens up to a positive scalar, which
does not change token ranking. The advantage would then be *structurally* zero, and S1 would need
no appeal to $W_U$'s geometry at all.

**PRIMARY**
$$\rho_I(\ell) = \frac{\lVert J_\ell - c^\star I\rVert_F}{\lVert J_\ell\rVert_F},\qquad
c^\star = \frac{\mathrm{tr}(J_\ell)}{d}$$
($c^\star$ is the exact minimiser.) Report per layer and as a band median. $\rho_I = 0$ means
exactly a scaled identity; $\rho_I \to 1$ means no identity component.

**DECISION RULE**
- $\rho_I$ **increases monotonically in scale** across the six models, with 70M/160M below and
  410M+ above a gap $\Rightarrow$ **ACCEPT**: the SLM J-lens is a scaled logit lens, and that is
  the mechanism.
- $\rho_I$ is **flat across scale** (range $< 0.10$) $\Rightarrow$ **REJECT**: $J$'s distance from
  the identity is not what changes with scale.
- Non-monotone or partially ordered $\Rightarrow$ **UNCLEAR.** Report and stop.

**CONTROL C1** $\rho_I$ computed on a norm-matched random Gaussian matrix of the same shape must be
$\approx 1$ (a random matrix has essentially no identity component). If it is not, the statistic is
mis-scaled.
**CONTROL C2** $\rho_I(cI) = 0$ exactly for a synthetic scaled identity.

---

## E39 — The concept-vector geometry: the object t21 should have measured

**WHY** `t21_conditioning_*` measured the geometry of $W_U$ **alone** and found it near rank-1 below
410M. But the lens never reads through $W_U$ alone — it reads through $v_t = J_\ell^\top W_U[t]$.
$J$ can spread those transported vectors apart or collapse them together, and *that* is what decides
whether concepts are separable. Measuring $W_U$ without $J$ was measuring the wrong object.

**PRIMARY** Over the 1310 distinct concept token ids of the eval battery, for transport
$T \in \{I, J_\ell\}$:
1. mean pairwise cosine of $\{v_t\}$ (lower = better separated);
2. effective rank of the stacked $\{v_t\}$ at 90% energy;
3. the **spread gain** $G = \overline{\cos}(I) - \overline{\cos}(J)$ — how much the operator
   improves concept separability over the free baseline.

**DECISION RULE**
- $G > 0$ at 410M/1B and $G \le 0$ at 70M/160M $\Rightarrow$ **ACCEPT**: $J$ buys concept
  separability only where the lens works, and fails to at small scale.
- $G$ has the same sign at all scales $\Rightarrow$ **REJECT**: separability gain does not track
  the read failure.
- Mixed $\Rightarrow$ **UNCLEAR.**

**CONTROL** A layer-deranged $J^{\mathrm{shuf}}$ must produce $G$ no greater than $J$'s. If a broken
operator separates concepts as well, the statistic is not measuring what we think.

---

## E40 — Does $J$ act as a scalar on activations that actually occur?

**WHY** $J$ can differ from $cI$ globally (E38) yet still act like a rescaling on the low-dimensional
manifold real activations occupy — in which case it is *behaviourally* the identity even if it is
not structurally one. This is the activation-weighted version of E38 and the same reasoning that
motivated $D_{act}$ over Frobenius distance.

**PRIMARY** $\overline{\cos}(J_\ell h, h)$ over the 541 cached eval-battery activations, per layer.

**DECISION RULE**
- $\overline{\cos} \to 1$ at 70M/160M and clearly below at 410M/1B $\Rightarrow$ **ACCEPT**: at
  small scale $J$ does nothing but rescale the activation, so the lens cannot differ from the logit
  lens.
- Flat across scale $\Rightarrow$ **REJECT.**

**DECLARED BIAS** The eval activations are concept prompts, not corpus text, so this measures $J$'s
action on the distribution we *read* from, not the one we *fitted* on. That is the right choice for
explaining read failure and the wrong one for explaining fitting; stated so it is not conflated.

---

## E41 — Layer-dependence of $J$

**WHY** If $J_\ell \approx J_{\ell'}$ across the band, the operator carries no layer-specific
information, there is nothing for a per-layer transport to exploit, and — separately — a
layer-derangement control *cannot* fire. That would explain the control's scale-dependent behaviour
without any claim about operators being good or bad.

**PRIMARY** median $\cos(J_\ell, J_{\ell'})$ over all band pairs $\ell \ne \ell'$, per model.

**DECISION RULE**
- median $\cos \to 1$ at 70M/160M and clearly below at 410M+ $\Rightarrow$ **ACCEPT**: small models'
  Jacobians are layer-independent, so per-layer transport is vacuous there.
- Flat across scale $\Rightarrow$ **REJECT.**

---

## E42 — The unification test: does one $J$-property explain BOTH scale and corpus?

**WHY** This is the experiment the whole reframing is for. S1 (scale) and S3 (corpus) have been
treated as separate failures. If a single property of $J$ predicts the read gap **across scale**
*and* **across corpus**, they are one mechanism. E37's corpus-flip at fixed $W_U$ is direct evidence
that they are coupled.

**PRIMARY** For each of $\rho_I$, spread gain $G$, activation cosine, and layer cosine:
1. rank correlation with the measured J-vs-logit gap **across the 4 scored scales** (70M, 160M,
   410M, 1B);
2. Pearson correlation with the **per-corpus** gap across the 5 corpora at 410M, with
   leave-one-corpus-out.

**DECISION RULE**
- A property clears **both** — consistent sign across scale, and $|r| \ge 0.8$ surviving every
  leave-one-corpus-out $\Rightarrow$ **ACCEPT**: one mechanism explains S1 and S3.
- Clears scale but not corpus (or vice versa) $\Rightarrow$ **PARTIAL.** Report as scale-only or
  corpus-only; do not claim unification.
- Neither $\Rightarrow$ **NULL.** This becomes the fourteenth failed predictor and the honest
  conclusion is that the read failure has no compact structural summary.

**CONTROL** The leave-one-out threshold is the same $|r| \ge 0.8$ used in E31, so this predictor
family is held to exactly the standard that killed the previous thirteen. No easier bar.

**DECLARED BIAS** $n = 4$ scales and $n = 5$ corpora. At those sizes a single point has enormous
leverage — Github has been that point repeatedly. A positive result here is directional at best and
requires a second scale's corpus grid to confirm.

---

## COST

All five run from artifacts already on disk (240 E28 lenses, 6 wikitext lenses, the cached eval
battery). **CPU, local, $0.** Estimated wall clock ~30 min total; E38 and E41 are seconds.
