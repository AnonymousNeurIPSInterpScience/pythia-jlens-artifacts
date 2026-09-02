# CONSTRUCT VALIDITY — where does the ground truth come from?

**Opened 2026-08-22. This is a MISSION-CRITICAL open question and it outranks every other item on
the slate, including the aggregation question.** Nothing here is resolved. This document exists so
that the reasoning is not lost and so the next person does not have to rediscover it.

**Status: OPEN. The two commissioned measurements have RUN — see §8. Their results are below and
they change §4.2 from suspected to confirmed, and partially clear S1 of the competence confound.**

---

## 0. THE QUESTION IN ONE LINE

> When we score whether `Brazil` appears in a Jacobian-lens readout, **who decided that `Brazil` is
> what Pythia-410M computes at layers 9–21?**

Nobody did. That is the problem.

---

## 1. WHAT WE ACTUALLY TEST ON

Established by reading `experiments/t36_qladder.py:17-31` and the released eval JSONs.

The model's input is:

```
ids = [BOS] ++ [128 tokens drawn from Q] ++ [the item's own prompt tokens]
```

| component | source | varies? |
|---|---|---|
| **the task** — 551 items, 937 intermediates | Anthropic's released concept sets | **never. Byte-identical in all 64 cells** |
| **fit corpus** — 200 × 128-token windows | 8 corpora (5 Pile components + 3 out-of-stream) | rows of the grid |
| **read context** — a 128-token prefix | the same 8, plus `Q0` (none) and token-shuffled | columns of the grid |
| model | Pythia-410M deduped (and 70M/160M/1B for the scale arm) | fixed per arm |

**The corpora are Pythia-native. The task is not.** We vary Pile-derived text on two axes and hold
Anthropic-authored concept prompts fixed on the third. The ground truth attaches to the *fixed*
axis.

### 1.1 The scored target is NOT the next token — verified

`target` is documented in `jacobian-lens/data/evaluations/README.md` as *"defines the readout
position only and is not itself scored."* What is scored is `intermediates`.

| set | prompt ends | answer (`target`) | **what we score** |
|---|---|---|---|
| multihop | "…where Carnival is most famously celebrated is the " | `Atlantic` | **`Brazil`** |
| order-ops | "(2 + 3) * 4 = " | `20` | **`5`, `multiplication`** |
| multilingual | `Lo opuesto de "grande" es "` | `pequeño` | **`Spanish`, `opposite`, `big`, `small`** |
| typo | "…learn a second langauge" | — | **`language`** |
| poetry | end of couplet line 1 | — | **`death`** (line 2's rhyme) |
| association | "…no one had used it in " | — | **`grief`** (never named) |

Measured across all 937 intermediates: **0 equal the answer token** (3 coincidences in multilingual,
0.3%); **7 (0.7%) appear literally in the prompt**, all arithmetic operands like `5` in
`((8 - 5) + 2) * 3`.

**So the design is not output-prediction, and the common objection to it is wrong.** This programme
measures known-intermediate exposure, which is the objective Gurnee et al. state.

---

## 2. WHERE THE GROUND TRUTH COMES FROM

**Human task decomposition, performed by Gurnee et al. for Claude-scale models, transplanted to
Pythia across roughly two orders of magnitude, never validated.**

Their justification is stronger than "a human thought so" but weaker than a measurement. They design
items where the surface prompt does not contain the intermediate, the answer is not directly
associated with the prompt, and the trigger and answer rarely co-occur without the bridge. That is a
**plausibility argument against a shortcut**, not an observation of an internal sequence.

Per family: multihop uses a bridge entity; multilingual uses an unnamed language; order-ops uses the
formal structure of the expression; poetry uses a not-yet-emitted rhyme; typo uses latent
recognition rather than next-token prediction; association is the least crisp and rests entirely on
human semantic reading.

**Nothing in this repository does circuit analysis on Pythia.** No experiment localises a `Brazil`
computation to any layer. No experiment confirms these are the intermediates Pythia computes.

### 2.1 The four labels that must be kept apart

1. **Program ground truth** — the algorithm the task generator specifies.
2. **Semantic intermediate** — what a human decomposition says connects prompt to answer. *This is
   what our battery has.*
3. **Behaviourally necessary intermediate** — a variable whose alteration harms the answer.
4. **Circuit ground truth** — the components whose intervention changes the behaviour.
5. **Readout-recovered intermediate** — what the lens surfaces.

Only 3–5 are model-specific. **We measure 5 and assume 2 licenses it.**

### 2.2 The two explanations we cannot currently separate

For *"the ocean on the coast of the country where Carnival is celebrated"* → `Brazil`:

1. Pythia computes a relationally structured intermediate: `Carnival → Brazil → ocean`.
2. Pythia retrieves a dense lexical association between `Carnival` and `Brazil`.

Both put `Brazil` in a readout. **Semantic correctness of a target does not establish that it was a
computational intermediate.**

---

## 3. THE COMPETENCE GATE DOES NOT EXIST

`capability_top10_rate` and `capability_mean_rank` measure the rank of the **intermediate**, not the
**answer** (`experiments/t36_qladder.py:165-176`). So they say nothing about whether the model can
do the task.

The programme already knows this. `experiments/t36_qladder.py:483-489`, verbatim:

> the pre-registered rule is top-k accuracy < 50% of Q0, but Q0's top-10 rate is 0.0694. **The eval
> targets are latent concepts, not emitted tokens, so this quantity is floored at Q0 and the rule
> cannot fire.** NOT reinterpreted silently — adjudication below uses mean rank, which is not
> floored, and the operator is asked to confirm.

Stored, corrected readout, `Q0`: top-10 rate **0.0694**, mean rank **3,125** (of ~50k).

**Consequence: there is no measurement, at any scale, of whether Pythia answers a single one of
these 551 items correctly.** That is the gap that bites the S1 scale claim hardest — "the J-lens is
degenerate at 70M/160M" is currently confounded with "the task is impossible at 70M/160M."

---

## 4. TWO MEASUREMENT PROBLEMS FOUND WHILE SCOPING THE FIX

### 4.1 The containment index does not transfer to the battery — MEASURED

The obvious fix is to stratify items by corpus containment, separating association from
computation. **It cannot be run as specified.**

Measured with the Pythia-410M tokenizer over all 551 items:

| set | n | median tokens | shorter than k=32 |
|---|---:|---:|---:|
| association | 102 | 28 | 74 |
| multihop | 93 | 16 | 76 |
| multilingual | 107 | 10 | 107 |
| order-ops | 55 | 9 | 55 |
| poetry | 98 | 27 | 97 |
| typo | 96 | 8 | 96 |
| **all** | **551** | **15** | **505 (92%)** |

**92% of items are shorter than k=32.** A 32-gram cannot be formed from a 15-token prompt, so the
existing index (`e48b_exposure_growth.json`, k=32 over 128-token windows) is **undefined for almost
the entire battery**.

What is needed instead is a **(trigger, intermediate) co-occurrence index** — how often `Carnival`
and `Brazil` appear within a window of the Pile — buildable on the same 20-shard bitmap but a new
index, not a re-run. That is the association-vs-computation discriminator; n-gram containment of the
prompt is not.

### 4.2 The readout may fall outside the operator's positional support — UNVERIFIED

Fitting averages the Jacobian over positions within a 128-token window with `skip_first=16`, so
approximately positions 16–127. But:

* **`Q0` rung** (no prefix): the readout lands at roughly position 15 — *below* that range.
* **prefixed rungs**: the readout lands at roughly 128 + 15 ≈ 143 — *above* it.

If the Jacobian is position-dependent, every read in this programme is an extrapolation, and `Q0`
and the prefixed rungs extrapolate in **opposite directions**. This would depress absolute numbers
and could explain pass@k AUCs of ~0.25.

**FLAGGED, NOT ESTABLISHED.** Requires reading `experiments/trainval.py` for `skip_first` semantics
and building the support table in §8. Do not describe it as a confirmed mismatch until that table
exists.

---

## 5. WHAT IS AND IS NOT SUPPORTED

### Supported now

> **Corpus choice affects recovery of fixed annotated concepts under the specified J-lens pipeline.**

Valid as a comparative result. The battery is byte-identical in all 64 cells, so if `Brazil` is an
association artifact it is an *equally* artifactual target for all eight fitting corpora. Measured at
13–45 pooled seed SD depending on aggregation and eval-set list. **The ground-truth problem does not
touch this.**

### Not yet supported

> **The J-lens is degenerate at 70M/160M Pythia.**

Requires the competence measurement (§3) and ideally the positional-support check (§4.2). Currently
confounded with task impossibility. This is the programme's lead claim and it is the one at risk.

### Not established

> **The J-lens reads Pythia's global workspace.**

Requires construct and causal validation. The battery's intermediates are annotations, not
measurements of Pythia's internal variables. **No document may drift into this claim.**

### Potentially salvageable, and already interesting

> **Corpus-dependent J-lens operators differentially recover candidate intermediate concepts, with
> effects that may depend on lexical association, model competence, and positional support.**

---

## 6. THE TWO ROUTES

### Route A — salvage the Anthropic battery

Keep the 551 items; stratify rather than discard.

1. **Answer-competence per item, per scale.** Probability and rank of the *expected answer*, top-1
   accuracy, margin, stability under paraphrase. Report the corpus effect separately on items the
   model can and cannot do.
2. **Association stratification.** Build the (trigger, intermediate) co-occurrence index (§4.1) and
   report low / medium / high association bands separately. If recovery concentrates in the
   high-association band, the result is more plausibly lexical retrieval; if it survives in the low
   band, the workspace reading strengthens.
3. **Positional control.** Evaluate at positions inside the fitting support; a small pilot suffices
   to say whether absolute numbers are suppressed by extrapolation.
4. **Causal spot-checks.** Swap or ablate the candidate direction on the strongest items — low
   association, model competent, target strongly recovered.

**Cost:** hours to days, CPU-dominated. **Risk:** if most items turn out to be high-association or
model-failed, the salvageable subset may be small.

### Route B — manufacture a mechanistically grounded suite

Build tasks where the causal variable is *imposed by construction* rather than inferred by a human.

**Layer 1 — synthetic algorithmic tasks.** Induction (`A B … A` → `B`, randomised tokens, lengths,
distractors). Associative recall over a random nonce dictionary (`The dax is florp`), varying
interference, key–value distance and conflicts. Two-hop nonce composition (`dax → Luma → Nera`) with
counterfactual siblings where only the middle fact changes. Modular arithmetic with objectively
defined partial results. Finite-state tracking where the hidden state is defined by a transition
table.

**Layer 2 — causal controls in every item.** For a program `x → z → y`, generate: change `x` holding
the second relation; change the relation holding `x`; change `z` while preserving surface token
statistics; patch the candidate intermediate at the proposed layer. The expected signature is not
merely "the answer changes" but that unrelated controls do *not* produce the same effect and that
patching transfers downstream behaviour.

**Layer 3 — discover the circuit independently.** **Do not use the J-lens to define the ground truth
and then evaluate the J-lens against it.** Baseline behaviour → activation/path patching → candidate
heads, MLPs, features → ablate the subgraph → compare against matched random and magnitude-matched
controls → test sufficiency by patching into a counterfactual run → test held-out generalisation.
Then ask the separate question: *does the lens read the variable carried by the causally validated
subgraph?*

**Layer 4 — negative tasks, which are essential.** Direct association, memorised single-hop recall,
lexical continuation, distractor-rich prompts, and tasks where a human would posit an intermediate
but the model can shortcut. Without the negative cells a readout looks successful merely by
surfacing semantically plausible concepts.

|  | concept causally used | concept not causally used |
|---|---|---|
| **readout present** | true positive | **false positive** |
| **readout absent** | false negative | true negative |

**A practical pilot:** induction, associative recall, two-hop nonce composition, modular arithmetic,
and one negative shortcut task. 100–200 items each, randomised token identities, 20–50 counterfactual
siblings, all Pythia scales, one or two checkpoints per scale, activation patching on a subset, full
layerwise J-lens output retained.

**Cost:** weeks. **Not a pre-28-August item.** Camera-ready or next paper.

### The routes are complementary, not exclusive

Label them differently and both can stand:

* **naturalistic battery** — recovery of externally annotated candidate intermediates
* **mechanistic suite** — recovery of causally validated task variables
* **factual battery** — recovery of relational variables under corpus-association controls

Which licenses a much sharper sentence: *the J-lens recovers annotated concepts on naturalistic
tasks, but its ability to recover causally validated computational variables is evaluated separately
on controlled tasks.*

---

## 7. PRIOR ART ON PYTHIA — fragmented, and it carries a warning

**CITATIONS BELOW WERE SUPPLIED BY external review AND HAVE NOT BEEN VERIFIED BY US.** NeurIPS desk-rejects
submissions containing hallucinated references. **Every one must be independently confirmed to exist,
and read, before it appears in the paper.** Recorded here as leads, not as bibliography.

| area | status on Pythia | ref (UNVERIFIED) |
|---|---|---|
| induction heads | cleanest cross-scale circuit-like phenomenon; ablation results vary by architecture and scale | 2605.08853 |
| IOI / binding | Pythia-160M IOI circuits reported **sensitive to low-level prompt syntax**, unlike GPT-2's more semantic organisation | 2602.13483 |
| cross-scale motifs | binding and path mechanisms may preserve causal contribution; selection and greater-than can show sign changes | 2605.21303 |
| factual recall | causal tracing / editing canon; substantial cross-architecture variation reported | 2202.05262 (ROME), 2605.08853 |
| sparse feature circuits | features connected into causal graphs, assessed by intervention | 2403.19647 |
| prompt-specific circuits | circuits discovered per prompt then clustered; Pythia-160M structures organise by syntax, not task label | 2602.13483v2 |
| latent multi-hop | whether models use a latent bridge or a shortcut | 2402.16837, 2411.16679, 2411.16353 |
| benchmarking philosophy | causal and behavioural faithfulness as the standard | 2504.13151 |

**What is missing:** no suite simultaneously provides Pythia-specific tasks, a known latent
algorithm, a known layerwise circuit, causal validation, coverage across 70M–1B, and an independent
benchmark for readout methods. **That absence is itself a finding**, and it means no manufactured
suite may be called "ground truth" in the strong sense unless its causal structure is experimentally
imposed or independently verified.

**The warning that matters most:** Pythia circuits appear scale- and prompt-dependent enough that a
single semantic task label is not a safe ground truth. Do not assume GPT-2's canonical IOI circuit
transfers.

---

## 8. RESULTS OF THE TWO COMMISSIONED MEASUREMENTS

### 8.1 CV1 — answer competence. RUN. `results/cv1_answer_competence.json`

Rank of the **expected answer** (not the intermediate) at the corrected readout position, on the
three sets carrying a `target`. n = 255 items (multihop 93, order-ops 55, multilingual 107).
poetry / typo / association carry `target: null`, so answer competence is **undefined** for them,
not zero.

| model | top-1 | top-10 | multihop median rank | order-ops | multilingual |
|---|---:|---:|---:|---:|---:|
| 70M | 1.6% | 16.9% | 80 | 18 | 3,436 |
| 160M | 1.6% | 22.4% | 42 | 13 | 3,788 |
| **410M** | **4.7%** | **26.3%** | 31 | 9 | 3,044 |
| 1B | 4.3% | 29.4% | 18 | 7 | 1,649 |

**Pythia-410M — the model the entire factorial runs on — answers 4.7% of these items correctly.**
1B is no better at top-1. Every headline in this programme is measured on a battery the model
overwhelmingly fails.

**Multilingual is at floor at every scale**: top-1 ~1%, median rank 1,649–3,788 of ~50k. It is
**107 of the 449 admitted items (24%)**, and the model cannot do it. Whatever the lens recovers
there, it is not an intermediate of a successful computation.

**But competence does NOT explain the S1 floor, and this partially clears it.** The competence
gradient (1.6 → 1.6 → 4.7 → 4.3) has no discontinuity at 160M→410M, where the `W_U` conditioning
cliff sits (mean pairwise cosine 0.9218 → 0.0307). If the S1 floor were a competence artifact,
competence would jump there; it rises modestly and then flattens. **The confound flagged in §5 is
measured and does not account for the effect.** S1's floor survives this check.

### 8.2 CV2 — positional support. RUN. CONFIRMED. `results/cv2_position_support.json`

`fast_fit` averages the Jacobian over `valid_position_mask(seq_len, skip_first=16)`, which excludes
the first 16 positions (attention sink) and the final position (no next-token target). For a
128-token window that is **positions 16–126 inclusive**.

| condition | median read position | range | **in support** |
|---|---:|---:|---:|
| `Q0` (no prefix) | 14 | 5–49 | **45.7%** (252/551) |
| prefixed rungs | 142 | 133–177 | **0.0%** (0/551) |

**Not one of 551 items reads inside the operator's positional support on any prefixed rung.** Q0
reads below the support on 54% of items — inside the attention-sink zone the fitter deliberately
excludes. **`Q0` and the prefixed rungs extrapolate in opposite directions**, exactly as suspected.

Consequences, stated conservatively:

* The **8×8 grid remains internally valid**: every cell reads at ~142, so the between-corpus
  contrast is a comparison under a fixed (if out-of-support) condition.
* The **`Q0`-vs-prefixed contrast in E36 is position-confounded**. Any statement of the form "adding
  a read context changes the read" conflates context with a ~128-position shift.
* The **J-vs-logit comparison is systematically unfair to the J-lens.** The logit lens has no fitted
  operator and therefore no positional support to leave; the J-lens is evaluated entirely outside
  its own. This is a candidate mechanical explanation for depressed absolute pass@k, and for the
  J-lens losing to the logit lens on some arms.
* **This assumes the Jacobian is position-dependent.** If it is near-invariant across positions the
  effect is small. That is directly testable — fit at several positions and compare operators — and
  is the obvious follow-up.

---

## 8.3 STILL COMMISSIONED

**1. Answer-competence, all scales, all 551 items.** Rank and probability of the *expected answer*
(not the intermediate) at the readout position, for 70M / 160M / 410M / 1B. Necessary to interpret
the scale comparison. Output: `results/cv1_answer_competence.json`.

**2. Positional-support audit.** Read `experiments/trainval.py` for `skip_first` and window
semantics; produce the table below. **Do not describe the mismatch as confirmed until this exists.**

| condition | read position | fitting-position support | in / out of support |
|---|---|---|---|
| `Q0` | | | |
| prefixed | | | |

Output: `results/cv2_position_support.json`.

## 9. ORDER OF WORK — and why aggregation is last

1. **competence** — is the model capable?
2. **positional-support audit** — is the operator evaluated inside its support?
3. **positional control** — a pilot at matched positions
4. **association index** — are the labels plausible proxies, or lexical shortcuts?
5. **aggregation** (`min` vs `persist` vs `best1L` vs `mean`)
6. **causal swaps** — reserved for low-association items the model answers correctly

**Aggregation is downstream of all of 1–4.** Choosing a layer-reduction is meaningless until we know
whether the model performs the task, whether the operator is in its support, and whether the targets
are proxies for computation or for association. The `min`-vs-`persist` question was occupying the
top of the slate and it should not have been.

## 10. THE POSTURE

Not sunk. Demote the mechanistic headline, promote the precise one:

> Before interpreting J-lens recovery as latent workspace access, establish task competence and
> positional validity. The corpus effect itself remains a valid fixed-battery comparative finding.
