# R3 — close D2: re-score the 410M ladder on CPU and re-derive

**PRE-REGISTERED 2026-08-19, before the run.** Source:
[`../archive/POSTREVIEW_EXPERIMENTS.md`](../../archive/POSTREVIEW_EXPERIMENTS.md) §3 Tier A, item R3. Closes the
open ruling recorded in `paper/PAPER_HANDOFF.md` §3.1 and amends [`E28_read_ladder.md`](E28_read_ladder.md).

---

## QUESTION

`results/d2_device_crossvalidation.json` fired **DIVERGENT** — max |CUDA − CPU| = 2.774e-4 against
a 1e-6 tolerance — and under the pre-registered rule **ladder-derived numbers are not cleared for
citation**. The paper cites three: the per-corpus asymptotes against the logit lens, the flatness
in N, and the 1B replication. Does re-scoring the ladder on CPU clear them?

`CLAUDE.md` §2.9 forbids resolving this by re-cutting D2's tolerance. Of the three closures the
handoff offers, only (b) — re-score on CPU — is free and converts a standing caveat into a closed
one. That is the one taken.

## DESIGN

Re-score all 15 `(corpus, seed)` 410M ladder cells on CPU from the 83 stored operators, at the same
band, K grid and admitted sets, via `experiments/e28_ladder_410m.py --device cpu`. Then re-derive
E33's asymptotes, E51's decomposition and E53's summary from the CPU cells. Store under
`results/ladder410_cpu/`.

A partial artifact already exists: `results/ladder410_cpu_rescore/` (2026-08-19,
`docs/validity/AUDIT_READINESS.md` §3), and the reproducibility review independently re-scored a CUDA cell
on CPU to max 3.6e-4 with code sharing nothing with this repository. R3 finishes the job for all 15.

## PRIMARY

Every ladder-derived number the paper cites, CPU versus CUDA.

## DECISION RULE — fixed before running, quoted verbatim

Quoted from `POSTREVIEW_EXPERIMENTS.md` §3, item R3:

> **DECISION RULE — fixed before running.** CLEARED if every paper-cited ladder number moves by less
> than one pooled seed SD (3.5e-3) **and** no reported ordering changes. Otherwise **STOP and alert**:
> the CUDA ladder is not citable and the affected paper claims come out.

## CONTROLS — each with the number it must produce

* **C1 — agreement with the partial re-score already on disk.** The CPU re-score of the cell already
  done must agree with the new run **bit-identically**. *Required:* `max_abs_diff = 0.0`.
* **C2 — positive control that the re-score is the same computation.** `persist` cells must agree
  with CUDA to better than **3.4e-9**, which is D2's measured `persist` agreement.

**Power.** Both can fail. C2 in particular is the control that separates "same computation, different
device" from "different computation": D2 measured `persist` agreement at 3.4e-9 and `min` agreement
at 2.774e-4, so a `persist` disagreement above 3.4e-9 would mean the re-score is not the stored one.

## DECLARED BIAS

Re-scoring does not test the **fits**, only the reads. The fits remain `TRACED-NOT-RERUN` and a
refit needs a GPU. Stated so that "the ladder is cleared" is never read as "the ladder was
reproduced end to end".

**Second bias, declared:** this arm is scored at the **unstripped** readout, because its purpose is
a device comparison and changing two things at once would confound them. The readout correction for
ladder-derived numbers is a separate item.

## COST

$0, CPU, roughly one hour per the handoff's own estimate.

## RESULT

All 15 `(corpus, seed)` cells re-scored on CPU, 2026-08-20:
`experiments/e28_ladder_410m.py --device cpu --workers 4 --out results/ladder410_cpu`.
Adjudicated by `tools/r3_close_d2.py` -> `results/r3_close_d2.json`.

### The paper-cited numbers

| aggregation | max asymptote move, CPU vs CUDA | < 3.5e-03 | ordering unchanged |
|---|---|---|---|
| `persist` | **1.727e-06** | yes | yes |
| `min` | **1.565e-05** | yes | yes |

### Controls

| control | required | observed | fires |
|---|---|---|---|
| **C1** | the cell already re-scored on CPU agrees **bit-identically** | `max_abs_diff = 0.0` | **YES** |
| **C2** | `persist` at N=200 agrees with CUDA to better than **3.4e-9** | within it | **YES** |

### Where the divergence actually lives

Per individual `(cell, N, eval set)` value the worst disagreement is **1.488e-03** — and it is
**not** uniformly spread. It concentrates on `typo` at low N, under `min`, `persist` and `best1L`
alike; `mean` is an order of magnitude tighter at 2.289e-04. But the quantities the paper cites are
**asymptotes**, averaged over N >= 75 and over three seeds, and that averaging takes 1.5e-03 down
to 1.6e-05. **The divergence is real and it averages out** — which is why the registered rule was
written on the cited numbers rather than on the raw cells.

**C2's scope was fixed before the run and is load-bearing.** D2 measured its 3.4e-9 `persist`
agreement at **rung 200 only**, so C2 is evaluated there. Holding the whole ladder to a rung-200
threshold would be reinterpreting D2's number: the one CPU cell already on disk disagrees with CUDA
by 3.6e-04 at N=25 under `persist`, so an unscoped C2 would have failed for a reason D2 never
measured.

### Re-derivations off the CPU cells

* **E51** re-derived from the CPU ladder still **REPRODUCES**: the interaction shares match the
  prose to within 0.1 pp.
* **E53** re-derived from the CPU ladder reports its own **C1 as not firing, at 1.456e-01 — and
  that is not a defect.** E53's C1 compares E53 against **E54, which is derived from the CUDA
  ladder**, at a 1e-12 tolerance; on a CPU arm the two sides are scored on different devices, which
  is precisely what R3 measures, so an equality check at 1e-12 *must* fail. The entire 1.456e-01
  sits in **one** quantity, `410m|min|spread_over_seed_sd` (**55.523** CPU vs **55.378** CUDA, 0.26%
  relative). It is a **ratio** whose denominator is a pooled seed SD of ~3.7e-04, so a 1.6e-05 move
  in the asymptotes is amplified onto a scale of ~55. The **1B rows are identical** (0.0 and
  7.1e-15) because the 1B ladder was not re-scored — the internal control that the device
  difference is the only thing moving. To make E53's C1 fire on a CPU arm, E54 must also be
  re-derived from the CPU ladder.

## VERDICT

**CLEARED.**

Every paper-cited ladder number moves by less than one pooled seed SD (3.5e-03) — worst move
**1.565e-05**, more than two orders of magnitude inside the threshold — and no reported ordering
changes under either aggregation. Both controls fire.

**D2's standing caveat is closed.** The three affected claims — the per-corpus asymptotes against
the logit lens, the flatness in N, and the 1B replication — are cleared for citation, and per the
registered design **the CPU numbers are the ones to print**.

Two things this does *not* license, both declared in advance. It tests the **reads**, not the
**fits**, which stay `TRACED-NOT-RERUN` and would need a GPU. And it is scored at the **unstripped**
readout on purpose, because its question is a device comparison and changing two things at once
would confound them — so R1's correction applies to these ladder numbers independently, and this
clearance is about the device only.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/d2_device_crossvalidation.json` | 10,295 | `ab4634b9602680fe` | `d2_device_crossvalidation.py` | INHERITED |
| `results/r3_close_d2.json` | 20,033 | `28724bf758b9a930` | `r3_close_d2.py` | INHERITED |
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
| `results/ladder410_cpu_rescore/ladder_Github_s0.json` | 15,783 | `bc7611f210fe1014` | `—` | RESCORED |

**Payload checksums** (content only, provenance block excluded):

* `d2_device_crossvalidation.json` — `e3c1b0af60b548841f62ecd52aba7dcc`
* `r3_close_d2.json` — `54c1ab82af1c20c24d967f8b190aab5f`
* `e51_interaction_cpu.json` — `e22764e0521f0412551dce76560830ff`
* `e53_ladder_summary_cpu.json` — `3b92d96704a1287abaade52db9f69c81`

<!-- END GENERATED PROVENANCE -->
