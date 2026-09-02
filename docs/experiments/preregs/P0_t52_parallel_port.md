# P0 — port a process pool into `t52_factorial.py`

**Output:** `results/e52_factorial_410m_pooled.json`

**PRE-REGISTERED 2026-08-19, before the port was written and before any run.**
Source: [`../archive/POSTREVIEW_EXPERIMENTS.md`](../../archive/POSTREVIEW_EXPERIMENTS.md) §3 Tier A, item P0.

---

## QUESTION

Can E52's 30 `(read rung, prefix seed)` cells be scored in parallel **without changing a single
number**, so that R1 is an overnight job rather than a two-day one?

This is infrastructure, not science. It is pre-registered anyway because it sits directly upstream
of the paper's headline: a parallelisation that changes a number would put a scheduling artifact
inside `fit_pct`, and the only way that is caught is by fixing the acceptance condition before the
port exists.

## PRE-REGISTRATION — the rule, verbatim

Quoted from `POSTREVIEW_EXPERIMENTS.md` §3, item P0, "DECISION RULE — and it is a gate, not
a target":

> * **ACCEPT:** the pooled run reproduces the serial run's stored matrix at **`max_abs_diff = 0.0`
>   across all 128 cell values**, and wallclock falls by at least 3x.
> * **REJECT:** any non-zero difference. **STOP.** A parallelisation that changes a number is a bug,
>   and shipping it would put a scheduling artifact inside the paper's headline. Do not "investigate
>   the tolerance" — there is no tolerance. The same computation in a different order over the same
>   fp32 inputs must give the same bits here, because nothing is reduced across cells.

## DESIGN

`experiments/t52_factorial.py` gains `--workers`. The per-cell body (`cache` + the 26 `score`
arms) is hoisted to a module-level function so it pickles; the parent keeps ownership of
everything that defines *what* a cell is, and the worker owns only *how* it is computed.

**Deviation from the slate's design note, declared, with its reason.** §3/P0 says "keep the
operators loaded once in the parent and let fork share them copy-on-write". **That is not available
here**: this laptop is `darwin`, where Python's default (and, with PyTorch loaded, only safe) start
method is `spawn`, not `fork` — measured, `multiprocessing.get_start_method()` returns `spawn`.
Forcing `fork` with an already-initialised PyTorch is the documented deadlock case. The port
therefore follows the pattern **`e28_ladder_410m.py` actually uses**, which is the same file the
slate points at: a top-level `cell()` that takes picklable arguments and loads the model and the
operators itself, once per worker, via a `Pool(initializer=...)`. Cost of the deviation: peak RSS
grows with `--workers` instead of staying flat. That is measured below and is what caps the worker
count, and it changes no number.

What the parent still computes, so that determinism is structural rather than hoped for:

* every cell's prefix construction — `build(rung, ps)` runs in the parent, so the
  `torch.Generator` sequence is identical to the serial run's regardless of scheduling;
* the item table, the pair table, `SET_IDX`, `ITEM_PAIRS`;
* all aggregation, bootstrap, permutation and adjudication, unchanged.

The worker returns the same per-pair score vectors the serial loop put into `CELL`, so every
downstream statistic reads an object of identical type and content.

`torch.set_num_threads(1)` is set inside each worker. Without it, `--workers 8` spawns 8x8 BLAS
threads on 14 cores and the run gets slower.

## PRIMARY

Wallclock for the full 30-cell grid, pooled versus serial.

## CONTROLS — each with the number it must produce

* **C1 — bit-identity.** Pooled, **without** `--rstrip`, against `results/e52_factorial_410m.json`.
  *Required:* `max_abs_diff = 0.0` on all 128 stored cell values (64 cells x 2 aggregations).
  This is the check `t52` already implements as E57's `control_C1_reproduces_stored_e52_matrix`
  (tolerance 1e-12, last fired at exactly 0.0).
* **C2 — worker-count invariance.** `--workers 1`, `2` and `8` must give identical output.
  *Required:* `max_abs_diff = 0.0` between every pair, on the same 128 values. A difference means
  shared mutable state and the port is wrong.
* **C3 — no thread oversubscription.** *Required:* `torch.get_num_threads() == 1` observed inside a
  worker and recorded in the results file, not asserted in a comment.

**Power.** C1 and C2 can fail: the scoring path was measured thread-count-invariant on this machine
before the port (smoke matrix at `OMP_NUM_THREADS=1` vs the default 10 threads,
`max|diff| = 0.0` over 18 cells), so an exact-zero requirement is a real constraint rather than a
tolerance chosen to be met. Had that pre-test come back non-zero, this gate would have been
impossible and the port abandoned — which is what makes it a gate.

## DECLARED BIAS

None on the numbers: this changes scheduling only, which is exactly what C1 and C2 verify and why
they are pass/fail rather than tolerance-based. One operational bias: with `spawn`, peak memory is
`workers x (model + operators)`, so `--workers` is bounded by RAM, not by cores, on this machine.

## COST

$0, CPU. An hour to write, minutes to gate.

## RESULT

Run: `t52_factorial.py --device cpu --workers 3 --out results/e52_factorial_410m_pooled.json
--cells-out results/e57_factorial_cells_410m_pooled.json`, 2026-08-19.

| control | required | observed | fires |
|---|---|---|---|
| **C1 — bit-identity** | `max_abs_diff = 0.0` on all 128 stored cell values | **0.0**, worst cell `None` | **YES** |
| **C3 — thread pinning** | `torch.get_num_threads() == 1` observed inside a worker | **1**, on every cell | **YES** |
| C2 — worker-count invariance | identical output at `--workers` 1, 2, 8 | *(full-grid arms outstanding; see below)* | — |

Wallclock: **51.0 min on 3 workers**, 30 cells, 102 s/cell, 3.9 s per score pass.

**Two defects the port surfaced, both found by asserting on observed state rather than on "the call
returned" (`CLAUDE.md` §6.0b), and both would have been invisible to a tolerance-based gate.**

1. **`jlens/hf.py:104-111` mutates the tokenizer in place.** `HFModel.__init__` sets
   `tokenizer.add_bos_token = True` under `force_bos`. Before the port the parent built the model
   *before* tokenising the eval items, so every item carried a BOS as a **side effect of a call
   made for another reason**. With the model moved into the workers, the parent's tokenizer was
   untouched and **48 of 48 smoke items silently lost their BOS**; the smoke matrix moved by
   **1.5e-02** — every cell, in the same direction. Caught because the pre-port smoke matrix had
   been stored first and the post-port one did not match it. The parent now performs the same
   mutation explicitly, and the run **aborts** if any item lacks a BOS.
2. **`torch.set_num_threads(1)` does not hold if called before the model loads.** A first full-grid
   run at 5 workers reported `thr=8` in every worker — 40 threads on 14 cores — despite the call
   being the first statement in the initializer. Something in the model-loading path resets the
   intra-op pool. Pinned *after* loading; observed per cell. The pin is deliberately **not**
   re-applied inside `_run_cell`, because that would make C3 a control that cannot fail, which is
   the defect R4e enumerates 36 instances of.

Neither defect changed a published number — both were caught before the gate — but the first would
have, silently, on every cell.

**Incidental measurement, worth recording:** at one BLAS thread per worker this workload is
**memory-bandwidth bound, not core bound**. Eleven compute processes on 14 cores reach only ~580%
CPU, and a 10-worker run died silently at ~10 minutes (peak RSS with `spawn` is
`workers x (1.6 GB model + 1.3 GB operators)` on a 24 GB machine). Three to six workers is the
usable range here; more is not faster and is eventually fatal.

## VERDICT

**ACCEPT on the decisive clause, with C2's full-grid arms outstanding.**

C1 — the clause that actually protects the paper — fires at **exactly 0.0 across all 128 cell
values**: the pooled path computes bit-for-bit the same measurement as the published serial run.
C3 fires. Worker-count invariance is established **at smoke scale** at `--workers` 1, 2, 3 and 4
(`max_abs_diff = 0.0` between every pair and against the pre-port serial run); the registered rule
asks for it at full grid scale at 1, 2 and 8, and those three arms are **not yet run**. They are
reported as outstanding rather than treated as satisfied by the smoke evidence.

The `>= 3x` wallclock clause likewise awaits the `--workers 1` full-grid arm, which supplies the
serial reference and is the same run as C2's first arm.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e52_factorial_410m_pooled.json` | 44,469 | `1c993ac4ad9ab31a` | `t52_factorial.py` | EXPOSED |
| `results/e52_factorial_410m.json` | 41,190 | `6e81f3cae37d0c19` | `t52_factorial.py` | EXPOSED |
| `results/e57_factorial_cells_410m_pooled.json` | 82,615 | `0b861194d22d8cb4` | `t52_factorial.py` | EXPOSED |

**Payload checksums** (content only, provenance block excluded):

* `e52_factorial_410m_pooled.json` — `466570432a7847b1db9678026018fec3`
* `e52_factorial_410m.json` — `de0fd855cefa636a23c70be9d4d9d62c`
* `e57_factorial_cells_410m_pooled.json` — `8d15b390fc341469cbc37ca5c9e715db`

<!-- END GENERATED PROVENANCE -->
