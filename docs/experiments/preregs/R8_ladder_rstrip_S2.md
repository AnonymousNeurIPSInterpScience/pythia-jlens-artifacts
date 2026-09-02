# R8 — the 410M ladder at the corrected readout: does S2 survive on its own measurement?

**PRE-REGISTERED 2026-08-20, before `--rstrip` was added to `e28_ladder_410m.py` and before any run.**

**This is not a new experiment.** It is `POSTREVIEW_EXPERIMENTS.md` §5's Tier C item S-1 —
*"Re-score E36/E48/E52/E59 and the 1B ladder under `.rstrip()` beyond R1"* — which that section
promotes itself: *"If R1 returns QUALIFIED or REJECT this becomes Tier A immediately."*
**R1 returned QUALIFIED** (`docs/experiments/preregs/R1_grid_rstrip.md`), so it is Tier A.

---

## WHY

S2 — *"reads are FLAT in N"* — is the one surviving subthesis whose corrected-readout status rests
on **someone else's re-derivation rather than a measurement in this repository**. The review reports
the range over N moving from 1.2–3.9 seed SD (unstripped) to 1.9–7.1 (corrected) and concludes the
claim holds either way. Nothing in `results/` contains that corrected number.

R3 cleared the ladder on the **device** axis (CPU vs CUDA, ≤1.6e-05 on cited numbers) but was run at
the **unstripped** readout on purpose, so it says nothing about the readout axis. S2 therefore has a
gap exactly where R1 had one, and R1 moved by 12.8× → 1.05× when it was closed. **The gap must be
closed by measurement before S2 is cited.**

## DESIGN

`experiments/e28_ladder_410m.py` gains `--rstrip`, implemented exactly as `t48_crossover.py:234`
does it. Re-score all 15 `(corpus, seed)` 410M cells **on CPU** from the same stored operators, same
band [9,21], same K grid, same admitted five, same reconstructed N values. Output
`results/ladder410_cpu_rstrip/`. The unstripped CPU ladder (`results/ladder410_cpu/`, R3) is the
comparison arm and is not overwritten.

## PRIMARY

Per corpus, the **range of the admitted-set read across the N axis, in units of that corpus's seed
SD** — `(max_N − min_N) / seed_sd` — and the ratio of the largest such range to the
**between-corpus spread** measured in the same units.

Published, unstripped: range over N **1.2–3.9 seed SD**; between-corpus spread **58×** seed SD.

## DECISION RULE — fixed before running

The strict reading of "flat" — *N moves the read no more than seed noise* — **already fails at the
unstripped readout** (1.2–3.9 seed SD, all above 1). So "flat" has always meant *small relative to
the corpus axis*, and the rule is written on that comparison, which is the one the programme
actually makes:

* **FLAT STANDS** — the largest per-corpus range over N is **below 25%** of the between-corpus
  spread, under **both** aggregations. (Published: 3.9 / 58 = **6.7%**, so this passes with an
  order of magnitude of margin and is a real constraint, not a rubber stamp.)
* **OVERTURNED** — the largest range over N is **at or above** the between-corpus spread. N would
  then matter as much as corpus identity, and S2 is false. **STOP and flag for adjudication.**
* **UNCLEAR** — anything between 25% and 100%. Report and stop; do not re-cut.

**Reported alongside, and not subject to the rule:** the raw per-corpus ranges in seed-SD units
under both conventions, so the strict-noise reading is visible whichever way the relative rule fires.

## CONTROLS — each with the number it must produce

* **C1 — recovery.** The same code run **without** `--rstrip` must reproduce `results/ladder410_cpu/`
  at `max_abs_diff = 0.0` across every `(cell, N, eval set, aggregation)` value.
  *Required:* exactly 0.0. If it is not, the flag changed something it should not have.
* **C2 — the convention bites here too.** The admitted-set mean at the reference rung must move by
  the amount the grid moved. *Required:* the stripped ladder's admitted mean is materially above the
  unstripped one (the grid moved ×2.55); a ladder that does not move means the flag never reached
  the scoring path.
* **C3 — the seed SD must not collapse.** The pooled seed SD must stay non-zero and within an order
  of magnitude of the unstripped 3.7e-04. *Required:* non-zero. A denominator that collapses would
  inflate every ratio in the primary and is the one way this measurement could lie.

**Power.** C1 can fail (it is an exact-equality check on ~2000 values). C2 can fail. C3 can fail.
None is an identity.

## DECLARED BIAS

1. Scoring-side only. The operators were **fitted** on unstripped prompts, exactly as in R1; this
   isolates the readout convention from the fitting convention.
2. CPU only, per R3's finding that the CPU numbers are the ones to print.
3. The N axis is **nested and reconstructed** from disjoint blocks (`reconstruct()`), not
   independently fitted at each N. That is a property of E28's design, unchanged here, and it means
   consecutive N values share prompts.

## COST

$0, CPU, ~1 h at 4 workers.

## RESULT

All 15 `(corpus, seed)` cells re-scored on CPU at the corrected readout, 2026-08-20.
`results/ladder410_cpu_rstrip/`, adjudicated by `tools/r8_ladder_flatness.py` ->
`results/r8_ladder_flatness.json`.

### PRIMARY — the N axis as a fraction of the corpus axis

| aggregation | unstripped | **stripped** | pre-registered bar |
|---|---|---|---|
| `persist` | 13.0% | **14.2%** | < 25% |
| `min` | 8.1% | **31.9%** | < 25% |

**Under `persist` S2 is essentially unmoved. Under `min` it lands in the UNCLEAR band.** The rule
required below 25% under *both*.

### What actually moved, per corpus

The `min` failure is **not** the N axis becoming important. Four of the five corpora barely move in
absolute terms:

| corpus | range over N, `min`, unstripped -> stripped |
|---|---|
| Pile-CC | 0.00235 -> 0.00239 |
| StackExchange | 0.00197 -> 0.00132 |
| USPTO_Backgrounds | 0.00265 -> 0.00256 |
| Wikipedia_en | 0.00147 -> 0.00208 |
| **Github** | **0.00345 -> 0.00783 (2.3x)** |

The driver is the **denominator**: the between-corpus spread under `min` falls **0.04244 ->
0.02456 (x0.58)**, i.e. **49.4 -> 22.2 seed SD**. The corpus effect nearly halves under `min` at the
corrected readout, so the same N-axis movement becomes a much larger *fraction* of it. Under
`persist` the corpus spread holds far better (60.3 -> 47.6 seed SD) and the ratio is stable.

### Controls

| control | required | observed | fires |
|---|---|---|---|
| **C1** | the same code without `--rstrip` reproduces `results/ladder410_cpu` at `max_abs_diff = 0.0` | **0.0** over 384 values | **YES** |
| **C2** | the stripped ladder's admitted mean moves materially | `persist` **×2.66**, `min` **×1.84** | **YES** |
| **C3** | pooled seed SD does not collapse | 4.02e-04 / 1.11e-03, both above the unstripped values | **YES** |
| **C4** | the N grid is identical across seeds of a corpus | **FAILS** — see below | **NO** |

### C4 — a defect this experiment found, which nothing else checks

**`results/e28_Wikipedia_en_410m_n800_s2.pt` does not exist.** Wikipedia_en seed 2 reconstructs 11
N values where seeds 0 and 1 reconstruct 16; it is the **only** corpus affected.
`t53_ladder_summary.py:136` averages **each seed over its own N grid**, so for that corpus the
resulting "seed SD" conflates seed variation with N-grid variation — and it is a term in the
denominator of the programme's **58.4x** headline. Measured under `min`: that corpus's seed SD is
**1.121e-04** on the ragged grid against **5.820e-04** on the intersected one, a **5.2x** difference
in one of five terms, moving `spread_over_seed_sd` from 55.5 to 49.4. `t53` stores
`asymptote_grids_identical_across_corpora` — across **corpora**, and never across **seeds**.
This tool uses the intersected grid, which is the clean estimand.

### DISAGREEMENT — the published range over N does not reproduce

The programme states the range over N as **1.2-3.7 seed SD** (`persist`) and 1.2-2.3 (`min`).
Recomputed on the same cells with `e53`'s own seed-SD convention it is **3.75-7.54** over the full
reconstructed grid and **2.49-6.27** over fitted-N-only; under `min`, 1.53-9.87 and 1.27-7.02.
**No script in the tree computes the published figure.** Reported as a disagreement, not adopted
(R4's standing rule). It does not contaminate this verdict, because the registered rule is a
**relative** comparison computed with identical estimands on both conventions.

## VERDICT

**UNCLEAR — report and stop. The rule is not re-cut.**

`persist` 14.2%, `min` 31.9%, against a bar of "below 25% under **both**". One aggregation passes
comfortably and one lands in the middle band.

**What I would say to a reader, without softening the rule.** S2's claim — that sample size barely
matters next to corpus identity — **survives intact under `persist`**, the aggregation the
programme adjudicates on, and at almost exactly its published strength (13.0% -> 14.2%). It becomes
**UNCLEAR under `min`**, and the reason is diagnosable: the *corpus* axis shrinks by 42% under `min`
at the corrected readout, not that N starts to matter. Four of five corpora are flat in absolute
terms; only Github's N-sensitivity grows.

There is a principled reason the programme weights `persist` — `min` prefers a layer-deranged
operator on 104 of 120 draws *paired by seed*, and **E54 confirms that survives the readout correction unchanged**
(`results/e54_aggregation_audit_rstrip.json`). But that reasoning was **not** in the registered
rule, so it does not convert UNCLEAR into FLAT STANDS. The honest statement is: **S2 holds under the
primary aggregation and is unresolved under the secondary one, and the secondary's failure is driven
by its own corpus-axis collapse.**

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/r8_ladder_flatness.json` | 32,829 | `baf01cf4c304dc65` | `r8_ladder_flatness.py` | IMMUNE |
| `results/e54_aggregation_audit_rstrip.json` | 69,867 | `d723021c6f6ac4e8` | `t54_aggregation_audit.py` | RESCORED |
| `results/ladder410_cpu_rstrip/ladder_Github_s0.json` | 15,830 | `ae3657a11bf48ec3` | `—` | UNKNOWN |
| `results/ladder410_cpu_rstrip/ladder_Github_s1.json` | 15,812 | `bcdb1e83c24ecaea` | `—` | UNKNOWN |
| `results/ladder410_cpu_rstrip/ladder_Github_s2.json` | 15,799 | `65ce1da93e9b79e8` | `—` | UNKNOWN |
| `results/ladder410_cpu_rstrip/ladder_Pile-CC_s0.json` | 10,985 | `620b052881722711` | `—` | UNKNOWN |
| `results/ladder410_cpu_rstrip/ladder_Pile-CC_s1.json` | 10,960 | `f474a277cf4aa3e3` | `—` | UNKNOWN |
| `results/ladder410_cpu_rstrip/ladder_Pile-CC_s2.json` | 10,963 | `71f21d033d1f47e9` | `—` | UNKNOWN |
| `results/ladder410_cpu_rstrip/ladder_StackExchange_s0.json` | 15,799 | `690b1cbfe1020131` | `—` | UNKNOWN |
| `results/ladder410_cpu_rstrip/ladder_StackExchange_s1.json` | 15,786 | `1b78f145b4323034` | `—` | UNKNOWN |
| `results/ladder410_cpu_rstrip/ladder_StackExchange_s2.json` | 15,818 | `4e0271679b54629b` | `—` | UNKNOWN |
| `results/ladder410_cpu_rstrip/ladder_USPTO_Backgrounds_s0.json` | 10,972 | `b0a0a61246ef5f7c` | `—` | UNKNOWN |
| `results/ladder410_cpu_rstrip/ladder_USPTO_Backgrounds_s1.json` | 10,948 | `855e5de3a33d5995` | `—` | UNKNOWN |
| `results/ladder410_cpu_rstrip/ladder_USPTO_Backgrounds_s2.json` | 10,962 | `d0479790f3d402d8` | `—` | UNKNOWN |
| `results/ladder410_cpu_rstrip/ladder_Wikipedia_en_s0.json` | 15,702 | `790b4ec9a5b8a2e2` | `—` | UNKNOWN |
| `results/ladder410_cpu_rstrip/ladder_Wikipedia_en_s1.json` | 15,603 | `96b16e44559f67b8` | `—` | UNKNOWN |
| `results/ladder410_cpu_rstrip/ladder_Wikipedia_en_s2.json` | 10,982 | `c3c6721bedf7e108` | `—` | UNKNOWN |
| `results/ladder410_cpu/e51_interaction_cpu.json` | 29,031 | `67b280e71e18f33f` | `t51_interaction_variance.py` | RESCORED |
| `results/ladder410_cpu/e53_ladder_summary_cpu.json` | 34,803 | `3136274d819c32d7` | `t53_ladder_summary.py` | RESCORED |
| `results/ladder410_cpu/ladder_Github_s0.json` | 15,784 | `4a6e725faf929831` | `—` | UNKNOWN |
| `results/ladder410_cpu/ladder_Github_s1.json` | 15,765 | `800a48a767c095a5` | `—` | UNKNOWN |
| `results/ladder410_cpu/ladder_Github_s2.json` | 15,769 | `2efceaa5838a92c5` | `—` | UNKNOWN |
| `results/ladder410_cpu/ladder_Pile-CC_s0.json` | 10,906 | `4486882a4bf78c6f` | `—` | UNKNOWN |
| `results/ladder410_cpu/ladder_Pile-CC_s1.json` | 10,918 | `f288e7b11b2ab283` | `—` | UNKNOWN |
| `results/ladder410_cpu/ladder_Pile-CC_s2.json` | 10,909 | `766eda99f7f2ea46` | `—` | UNKNOWN |
| `results/ladder410_cpu/ladder_StackExchange_s0.json` | 15,780 | `3922aed1bba3b34b` | `—` | UNKNOWN |
| `results/ladder410_cpu/ladder_StackExchange_s1.json` | 15,791 | `09a2c3a17459e3ab` | `—` | UNKNOWN |
| `results/ladder410_cpu/ladder_StackExchange_s2.json` | 15,781 | `69c734ca706ba24d` | `—` | UNKNOWN |
| `results/ladder410_cpu/ladder_USPTO_Backgrounds_s0.json` | 10,913 | `dca4188fad0ee8ad` | `—` | UNKNOWN |
| `results/ladder410_cpu/ladder_USPTO_Backgrounds_s1.json` | 10,900 | `2dd8d1c35a0f4f8c` | `—` | UNKNOWN |
| `results/ladder410_cpu/ladder_USPTO_Backgrounds_s2.json` | 10,911 | `4c27aca6932a25a0` | `—` | UNKNOWN |
| `results/ladder410_cpu/ladder_Wikipedia_en_s0.json` | 15,637 | `df4f2d634f7a6625` | `—` | UNKNOWN |
| `results/ladder410_cpu/ladder_Wikipedia_en_s1.json` | 15,555 | `d8b27dbb61408fa4` | `—` | UNKNOWN |
| `results/ladder410_cpu/ladder_Wikipedia_en_s2.json` | 10,911 | `97fde88911ffbc01` | `—` | UNKNOWN |

**Payload checksums** (content only, provenance block excluded):

* `r8_ladder_flatness.json` — `e8c57a7d565cd23b79230105c06d17cd`
* `e54_aggregation_audit_rstrip.json` — `3cf2de44dae86fa5cd02084697e76a2b`
* `e51_interaction_cpu.json` — `e22764e0521f0412551dce76560830ff`
* `e53_ladder_summary_cpu.json` — `3b92d96704a1287abaade52db9f69c81`

<!-- END GENERATED PROVENANCE -->
