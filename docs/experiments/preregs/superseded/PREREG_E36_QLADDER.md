# PRE-REGISTRATION — E36, the Q-ladder: the S3 crossover

**Written 2026-08-13, before any E36 number exists. Branch `pythia`.**
Format is CLAUDE.md §3. Nothing here may be reinterpreted after a result; a REJECT or UNCLEAR
outcome stops the experiment and goes to the operator.

Companion pre-registration: **E35 (M1 containment index)** below, which builds this experiment's
x-axis and must complete first.

---

## WHY

S3 is the subthesis the programme was opened to test and **it has never been run**. Its claim:

> J-lens read performance degrades under distribution shift *faster than the logit lens*, and the
> degradation is predictable from a quantity computable from `J` itself.

Everything measured so far is **within** the training distribution. E28's five corpora are all Pile
components; its own declared bias says so; the design *cannot* produce a crossover because the
J-lens is never pushed off its fitting distribution. E33 showed the logit lens sits inside the
five-corpus spread, which is suggestive but is a **corpus-identity** result, not a shift result:
nothing gets worse as a function of distance from anything.

The structural asymmetry that makes S3 testable at all:

| lens | fitted? | can it degrade under shift? |
|---|---|---|
| logit lens (`J = I`) | no free parameters | **no** — it estimates nothing, so it has no corpus to be wrong about |
| J-lens (`J = E_P[∂h_final/∂h_ℓ]`) | fitted to corpus P | **yes** — the `‖J^P − J^Q‖` term of EQ2 |

So there must exist a shift magnitude at which the free baseline overtakes the fitted one, **or
there must not**, and either answer is the paper. **A null here is a result**: it would mean an
averaged derivative is robust to shift, and the paper becomes the mechanistic explanation of why.

**Why now.** The blocker was never compute — it was that "evaluate on Q" had no operational
definition (PLAN.tex called this "the one undesigned piece"). §DESIGN below fixes that. And no
refit is required: we already own five `J^P`, so this is forward passes only.

---

## DESIGN

### What varies, and what does not

| held fixed | varied |
|---|---|
| model (pythia-410m-deduped), band `[9,21]`, `target_layer=-2` | **Q — the distribution the read activations are drawn from** |
| the fitted operators `J^P` (five, from E28, N=400) | |
| the concept eval items and their target tokens | |
| aggregation (`persist`), K grid, readout rule | |

**We vary the read distribution, not the fit distribution.** Stated plainly because it is the
design's central limitation: see DECLARED BIAS.

### Operational definition of "evaluate on Q" — Q-context prefixing

The anchor's concept eval sets are **not corpora**, so "evaluate on Q" cannot mean "swap the eval
set" without also swapping the task. Instead:

```
prompt_Q(item) = [ 128 tokens drawn from Q ]  ++  [ the concept item's own prompt, unchanged ]
```

The readout position remains the item's own readout position (shifted by the prefix length). The
target `intermediates` are unchanged, so `pass@k` is comparable across every rung.

This varies `h(x)` — the activation the operator acts on — which is **exactly** the quantity
`D_act = E_{x∼Q}‖(J^P − J^Q)h(x)‖²` integrates over. The prefix length matches the fitting window
(`max_seq_len=128`) so the activations are drawn from the same window statistics the operator was
averaged over.

### The rungs

Ordered by *expected* shift. **The ordering is a hypothesis, not the x-axis** — the x-axis is
measured containment from E35.

| rung | Q | ID/OOD status | certified by |
|---|---|---|---|
| Q0 | **no prefix** | — | the null rung; must reproduce E33 exactly |
| Q1 | the fitting corpus P itself | ID | construction (Pile component) |
| Q2 | a different Pile component | ID | construction |
| Q3 | post-2020 English, same register (recent Wikipedia revisions) | **OOD** | **date** — Pile assembled ~2020 |
| Q4 | post-2020 English, shifted register (recent technical/web text) | **OOD** | date |
| Q5 | non-Pile script / language | **OOD** | date + script |
| Q6 | token-shuffled Q1 (syntax destroyed, unigram statistics preserved) | degenerate | construction |

Q6 is both the maximal-shift rung **and** a floor control: it holds the unigram distribution fixed
while destroying sequence structure, so it separates "unfamiliar tokens" from "unfamiliar syntax".

### Sample size

All 541 scorable eval items × 7 rungs × (5 `J^P` + logit + `J^shuf` + model-output) transports.
Prefix drawn with 3 disjoint seeds per rung so every point carries a genuine error bar, as
SPINE A1.1 binds (intervals from seeds, never from residual scatter).

---

## PRIMARY

**The one number: the containment level `c*` at which the J-lens read AUC curve crosses the logit
lens's.** Reported per fitting corpus P (five values), with the crossing located by linear
interpolation between adjacent rungs and bootstrapped over the 3 prefix seeds.

Secondary: the **slope** `d(AUC)/d(containment)` for each transport. S3's claim in its weakest
testable form is that the J-lens slope is steeper than the logit lens's.

---

## DECISION RULE — fixed before running, not reinterpretable

- **ACCEPT S3** if, for ≥3 of the 5 fitting corpora, the J-lens curve crosses below the logit-lens
  curve at some rung, **AND** the crossing survives leave-one-rung-out, **AND** control C3
  (capability) has not collapsed at or before the crossing rung.
- **REJECT S3** if the two curves remain parallel within their seed intervals across the whole
  ladder — i.e. shift degrades *both* lenses equally. **This is the publishable null**: the fitted
  operator is no more shift-fragile than the unfitted one, and the averaged derivative is robust.
- **UNCLEAR** if capability (C3) collapses at or before any crossing, because lens failure and
  model failure are then not separable. **Stop. Do not re-pick Q to manufacture a crossing.**

**A crossing that appears only under `min` aggregation does not count.** `min` is disqualified
(CLAUDE.md §6.0) and is reported but never votes.

---

## CONTROLS — each with the number it must produce

| id | control | must produce |
|---|---|---|
| **C1** | Q0 (no prefix) reproduces E33 | logit `persist` = **0.02844**, Github `J^P` = **0.02808**, to 4 dp. A mismatch means the prefixing path altered the base measurement and everything downstream is void. |
| **C2** | `J^shuf`, **random derangement** (not cyclic — that is the E33 v1 failure) | below both `J^P` and logit at **every** rung; `J^P > J^shuf` on ≥5 of 6 sets at Q0 |
| **C3** | **capability** — the model's own top-k accuracy on the target token at the final layer | must not fall below 50% of its Q0 value at or before the crossing rung. This is what separates "the lens stopped reading" from "the model stopped knowing". |
| **C4** | **prefix-length control** — a prefix of the same 128 tokens drawn from the *fitting corpus P* | isolates "any prefix at all" from "a prefix from Q". Any degradation present at C4 is subtracted before the crossing is located. |
| **C5** | `D_act(J,J) = 0` exactly | 0.0 (already fires: `t23_pq_ladder_410m.json`) |

**C4 is load-bearing.** Adding 128 tokens of *anything* changes the readout position's activation.
Without C4 the entire ladder could be a prefix-length artifact.

---

## DECLARED BIAS

1. **We shift the read distribution, not the fit distribution.** EQ2's bias term is `‖J^P − J^Q‖`
   where `J^Q` is the operator that *would have been fitted* on Q. We never fit `J^Q`. What we
   measure is how `J^P` performs on activations drawn from Q — a related but not identical
   quantity. **This is the honest scope and it must appear in the paper.**
2. **The concept eval sets are English.** Q5 (non-Pile script) may degrade the *task*, not the
   lens. C3 exists for this and Q5 may well land UNCLEAR on its own.
3. **410M only.** The interaction result (SPINE §A2.6) is one scale; so is this.
4. **`association` is excluded** — floored, 196/205 ladder cells exactly zero, fails E28 admission
   criterion 3. Five admitted sets.
5. **Prior on D_act is now poor.** It has failed as a floor predictor twice (E31-L, E34-a). E36
   gives it a *different* job — predicting a crossing, not a level — but the prior is stated so a
   positive result is not over-read.

---

## COST

**No refits.** Forward passes only: 541 items × 7 rungs × 3 seeds ≈ 11.4k forward passes at 410M,
plus 8 transports × unembed per item. Measured reference: E33 scored 4 transports × 541 items on
**CPU** in ~12 min.

| | est. wall | est. $ |
|---|---|---|
| E36 on local CPU | ~4–6 h | **$0** |
| E36 on one RTX 4090 (offer 47633410, $0.430/h) | ~15 min | **~$0.15** |

Budget gate (CLAUDE.md §6.10, >$40 or >6 h) **not triggered**.

---
---

# PRE-REGISTRATION — E35, the M1 containment index (E36's x-axis)

## WHY

E36's x-axis must be a *measured* shift coordinate, not a rung number. Pythia is the only model
family where this can be a **fact**: the deduped suite trained 300B tokens over a ~207B-token
corpus and the exact token stream is published, so "did the model see this text" is answerable
rather than arguable.

## DESIGN — invert the containment query

Containment asks *what fraction of **Q's** k-grams appear in the training stream*. The naive
implementation indexes the 300-billion-token side. **Index Q instead:**

| side | size |
|---|---|
| Pythia deduped stream | 300B tokens → ~3×10¹¹ k-grams, **601 GB** (20 × 30 GB `.bin`, uint16) |
| all candidate Q corpora | ~6M tokens → **~6M k-grams** |

Build a Bloom filter over **Q's** k-grams (6M entries at FPR 1e-8 ≈ **21 MB, cache-resident**),
stream the Pile once, probe every position, and verify the rare hits exactly against a sorted
48 MB array. Peak disk: one 30 GB shard, deleted after each pass. **Storage cost ≈ 0.**

Two properties this buys for free:
- **More Q's cost nothing** — one Bloom, tagged by Q-id; one stream answers all candidates.
- **The k-sweep costs nothing** — parallel filters for k ∈ {8, 13, 20, 32} in the same pass.

**Second artifact from the same pass:** emit every stream k-gram with `hash mod 100 == 0` to a
sorted `uint64` Parquet (3×10⁹ rows ≈ 24 GB raw, **~13 GB Parquet**). That is a reusable
*approximate* containment oracle for every **future** Q, queryable with DuckDB (zone maps on a
sorted UBIGINT column, hash-join against a few thousand Q rows) with **no further downloads**.
Hash-modulo sampling keeps the same k-grams on both sides, so sampled containment is unbiased;
a 2400-doc Q corpus yields ~3,070 sampled k-grams → binomial SE ≈ 0.9% at containment 0.5.

**Measured, not assumed:** naive 13-pass numpy hashing runs at **0.04 GB/s** on this laptop (4 h for
the full stream). A single-pass Rabin–Karp recurrence in numba is ~1 GB/s (~20 min) and shards 20
ways. Download at 8.4 Gbit ≈ 10–40 min. **The job is network-bound; a GPU is useless here** — it
would be fed through a 0.1 GB/s straw against 800 GB/s of HBM.

## PRIMARY

Containment `c(Q)` for every candidate Q, at the selected k.

## DECISION RULE — k is selected by a rule fixed in advance

**Choose k as the largest value in {8, 13, 20, 32} for which the candidate Q's span a containment
range ≥ 0.3.**

- If **no k** achieves a 0.3 range, M1 cannot rank these Q's → **the x-axis falls back to the
  ordinal rung and E36 runs with a categorical axis, disclosed as such.**
- If **k=8 saturates** every Q at c ≈ 1.0, that is reported, not hidden.

> This rule exists because **the same failure has already happened once in this programme**: E28's
> intended manipulation check was classifier AUC, which measured **1.0000 for every candidate pair**
> and could not rank them. A containment index at small k will saturate the same way.

## CONTROLS

| id | control | must produce |
|---|---|---|
| **M1-C1** | a known Pile component (Pile-CC windows, already ID by construction) | **high** containment — if a corpus we *know* is in the Pile scores low, the index is broken |
| **M1-C2** | random-token documents at matched length | containment ≈ 0 |
| **M1-C3** | post-2020 text | must score **below** M1-C1. If not, either the date argument or the index is wrong |
| **M1-C4** | one shard vs twenty | containment on 1/20 of the stream must be a strict **lower bound** on containment over 20/20 |

## The asymmetry, stated because it decides how far to trust each end

- **ID end — M1 certifies it.** A hit is definitive; positive containment proves exposure.
- **OOD end — M1 *cannot* certify it.** Absence over a sampled stream proves nothing, and even at
  full coverage you are bounded by k. **Use dating**: the Pile was assembled ~2020, so post-2020
  text is *logically* unseen. That is a guarantee, not a statistic.

The five E28 corpora need no M1 at all — they are Pile components by construction.

## TINY-FIRST — one shard before twenty

Shard 1 is 30 GB (5% of the stream) and is enough to (a) prove the pipeline end-to-end and (b)
**select k**, because a partial stream gives a strict lower bound and k-selection only needs the
value where candidates *separate*. Scale to 20 shards only if the k-sweep shows the resolution is
needed. This is CLAUDE.md §6.6 (tiny-first) applied to a data-engineering problem.

## COST

Network-bound, so choose the box by bandwidth and price, **not** by fp32 TFLOPS (CLAUDE.md §7
applies to the Jacobian workload, not this one).

| step | box | wall | $ |
|---|---|---|---|
| shard 1 only (tiny-first) | vast `45415102` RTX A4000, $0.108/h, 8366 Mbit down, 217 GB disk | ~20 min | **~$0.04** |
| full 20 shards | same | ~1.5–2 h | **~$0.22** |
| local laptop alternative | — | 13 h at 100 Mbit | $0 |

Budget gate **not triggered**. Artifacts to pull before teardown (CLAUDE.md §6.7): the containment
JSON and the ~13 GB Parquet sketch — **the sketch must be pulled and SHA-verified before the box is
destroyed**, or the whole stream has to be re-downloaded.
