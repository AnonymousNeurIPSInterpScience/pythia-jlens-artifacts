# COMPUTE.md — the hardware model, the costs, and how to buy them

Everything here is measured or explicitly labelled as extrapolation. The live calculator is
`./lab cost`; this document is why it says what it says.

---

## 1. The one fact that decides everything: this workload is compute-bound

Fitting a Jacobian lens means, for each layer, backpropagating through the model once per output
dimension (in chunks of `dim_batch`). That is an enormous amount of arithmetic against a small
working set.

| quantity | value | consequence |
|---|---|---|
| arithmetic intensity | **482 FLOP/byte** | measured |
| A100 roofline ridge point | **9.6 FLOP/byte** | we are 50× to the right of it |
| observed utilisation | **~100% of fp32 peak** | not memory-starved, not launch-bound |

**So choose GPUs by fp32 TFLOPS, not by memory bandwidth and not by VRAM.** VRAM only decides
whether a model *fits*; once it fits, more VRAM buys nothing.

```
L40S    91.6 TF fp32     RTX 4090  82.6     H100  67.0     A100  19.5
```

**These are spec sheet figures and at least one is badly wrong for this workload.** An RTX 5090
measured **18.4 TFLOPS fp32** here against a spec implying ~105 — a 5.7× overstatement. Treat the
table as a shortlist, not a prediction, and **measure `K` (§2) on any card not already in the
measured column before sizing a fleet.** This is the "never accept a proxy for the thing you care
about" rule in its most expensive form: a spec sheet is a proxy for throughput.

An A100 is a *bad* deal for this workload despite being the reflexive choice for ML — its fp32
rate is a quarter of an L40S's. It is on the list only because 80 GB of VRAM is sometimes the
only way to make a large model fit at all.

### TF32 is forbidden

`NVIDIA_TF32_OVERRIDE=0` on every box, asserted by `repro/13_vast_preflight.sh` and by
`box_bootstrap.sh`. TF32's 10-bit mantissa breaks the anchor gate's 2e-5 tolerance. The failure is
silent: numbers still come out, they are just no longer the anchor's operator. Do not "just try it
and see".

---

## 2. The timing model

```
seconds_per_prompt  =  K_gpu  ×  n_layers × d_model²  /  1e6
```

`K` is **measured per GPU**, not derived from a spec sheet.

| GPU | K | provenance |
|---|---|---|
| L40S | **0.122** | measured — E28 run log, 2026-08-12 |
| H100 | **0.125** | measured — `CLAUDE.md` §7 |
| A100 | **0.320** | measured — `CLAUDE.md` §7 |
| RTX 4090 | 0.135 | *extrapolated*, never measured |

### The anchors

| model | GPU | measured s/prompt | source |
|---|---|---|---|
| pythia-410m | L40S | **3.04 – 3.11** across N ∈ {50,100,200,400} | E28 run log |
| pythia-1b | H100 | **8.375** | `CLAUDE.md` §7 |
| pythia-1b | A100 | **21.44** | `CLAUDE.md` §7 |

**Linearity in N is measured, not assumed**: 3.04–3.11 s/prompt over an 8× range of N is a 2%
spread. Cost estimates are interpolation.

### The thing this model gets wrong, stated up front

L40S and H100 land within 2% of each other (0.122 vs 0.125) despite a 37% gap in fp32 peak. So on
those two the workload is **not purely fp32-limited** — something else (likely memory bandwidth:
L40S 864 GB/s vs H100 3.35 TB/s, pulling the other way) is binding. The A100's 2.6× penalty *is*
consistent with its fp32 rate. Treat `K` as an empirical constant, not a derivation, and re-measure
when you use hardware not in the table.

---

## 3. What a fit costs

`./lab cost --table` regenerates this. One lens at N=200:

| model | layers | d_model | VRAM | L40S | H100 | A100 | RTX 4090 |
|---|---|---|---|---|---|---|---|
| 70m | 6 | 512 | 3 GB | 0.01 h · $0.01 | 0.01 h · $0.03 | 0.03 h · $0.03 | 0.01 h · $0.00 |
| 160m | 12 | 768 | 4 GB | 0.05 h · $0.04 | 0.05 h · $0.12 | 0.13 h · $0.14 | 0.05 h · $0.02 |
| 410m | 24 | 1024 | 8 GB | 0.17 h · $0.14 | 0.17 h · $0.42 | 0.45 h · $0.49 | 0.19 h · $0.08 |
| 1b | 16 | 2048 | 14 GB | 0.45 h · $0.36 | 0.47 h · $1.12 | 1.19 h · $1.31 | 0.50 h · $0.20 |
| 1.4b | 24 | 2048 | 18 GB | 0.68 h · $0.55 | 0.70 h · $1.68 | 1.79 h · $1.97 | 0.75 h · $0.30 |
| 2.8b | 32 | 2560 | 32 GB | 1.42 h · $1.14 | 1.46 h · $3.50 | 3.73 h · $4.10 | — no fit |
| 6.9b | 32 | 4096 | 64 GB | — no fit | 3.73 h · $8.95 | 9.54 h · $10.50 | — no fit |
| 12b | 36 | 5120 | 96 GB | — no fit | — no fit | — no fit | — no fit |

**12B does not fit on a single card at fp32.** It needs multi-GPU sharding, which is not
implemented. That is why the ladder stops at 2.8B, and it is a scope limit, not an oversight.

### Value ranking

| GPU | fp32 TF | typical $/h | **TFLOPS per dollar-hour** |
|---|---|---|---|
| RTX 4090 | 82.6 | 0.40 | **206** *(extrapolated K)* |
| L40S | 91.6 | 0.80 | **115** |
| H100 | 67.0 | 2.40 | 28 |
| A100 | 19.5 | 1.10 | 18 |

L40S is the default: best measured value, and 48 GB clears everything up to 2.8B. The 4090 looks
better on paper but its `K` has never been measured and its 24 GB caps you at 1.4B.

---

## 4. Per-experiment compute — the catalogue

**The single most useful fact in this document: almost none of the paper needs a GPU.** The
expensive step is *fitting* operators, and that happened once. Every experiment on the three shift
axes reads lenses that already exist, on CPU, for nothing. A reviewer reproducing the paper's
claims rents no hardware.

### 4.1 The experiments the paper rests on

Costs are the `MODULE_COST` each `repro/exp/*.sh` declares; run any of them with `--dry-run` to see
the contract without executing. Wall times are single-machine CPU unless stated.

| experiment | module | compute | wall | writes |
|---|---|---|---|---|
| **E28** read ladder — 5 corpora × 16 N × 3 seeds × 2 models | *(no module; the original fits)* | **GPU, 21.4 GPU-h ≈ $17.1** | 5.4 h on 4×L40S | `results/ladder410/`, `results/ladder1b/`, 240 lenses |
| **E33** the missing `J = I` arm | — *(rescore of E28 artifacts)* | free, CPU | minutes | `e33_logit_baseline_410m_v2.json` |
| **E33b** its t statistics, stored | `e33b_tstats.sh` | free | seconds | `e33b_tstats_410m.json` |
| **E35 / M1** n-gram containment index | *(no module)* | **network-bound**, 20 shards | hours | `results/m1/`, `e35_containment_shard0.json` |
| **E48b** containment vs stream coverage | `e48b_exposure.sh` | free, CPU | ~15 min | `e48b_exposure_growth.json` |
| **E48 gate** model competence per corpus | `e48_gate.sh` | free, CPU | ~40 min | `e48_competence_gate_410m.json` |
| **E48** the S3 crossover, fit axis | `e48_crossover.sh` | free, CPU | ~25 min | `e48_crossover_410m.json` |
| **E48 OOD fits** 3 corpora × 3 seeds at 410M, N=200 | *(prerequisite for E48)* | **GPU, ~1.5 GPU-h ≈ $1.2** | ~30 min | `results/e48/*.pt` |
| **E48c** does exposure order the read? | `e48c_exposure_read.sh` | free | seconds | `e48c_exposure_vs_read.json` |
| **E36** the Q-ladder, shifting the read | `e36_qladder.sh` | free, CPU | ~1.5–2 h | `e36_qladder_410m.json` |
| **E52** the fit × read factorial | `e52_factorial.sh` | free, CPU | ~1.3–2 h | `e52_factorial_410m.json` |
| **E51** corpus × set interaction | `e51_interaction.sh` | free, CPU | seconds | `e51_interaction_variance.json` |
| **E31** the 21-predictor bake-off | *(no module — gap)* | free, CPU | minutes | `e31_local_bakeoff_410m.json` + 7 others |
| **E38** why small models fail | *(no module — gap)* | free, CPU | minutes | `e38_jgeometry.json` |

**Two experiments still lack a `repro/exp/` module** (E31, E38). Both are free to run; the gap is
that their exact flags live only in their docstrings. `docs/experiments/preregs/` records this.

**`results/m1/` and the containment index are network-bound, not compute-bound.** A GPU is useless
for that stage — it streams 300B tokens past an in-memory k-gram index. Do not rent one for it.

### 4.2 What a *fresh* fitting campaign costs

`./lab cost --experiment <name>` prices these live.

| campaign | what it is | GPU-h | cost | wall (4×L40S) |
|---|---|---|---|---|
| `ladder` | one lens per Pythia size at N=200 | 4.2 | $3.4 | 1.1 h |
| **`e28`** | the full read ladder, 180 fits | **21.4** | **$17.1** | **5.4 h** |
| `nstar` | every lens refit at its law-prescribed N\* (ε=5%) | 12.4 | $9.9 | 3.1 h |

E28's estimate was produced **before** the run and the completed run matched it, so the timing
model in §2 is calibrated rather than fitted after the fact.

### 4.3 The largest open item, priced

`docs/context/CONTEXT.md` §6 ranks a **second scale at 1B for E48, E48c, E36 and E52** as the biggest
exposure in the paper — four of six Tier-A results are 410M-only. Fitting 1B is **0.45 GPU-h ≈
$0.36 per lens** (§3); the scoring is CPU and free. **Use 1B, which is already burned.** Scoring
reads at 1.4B or 2.8B consumes the last clean confirmatory cells of
`docs/experiments/preregs/superseded/PREREG_PYTHIA_T7_v2.md` — a one-way door, and fitting there is safe but
scoring is not.

### The budget gate

**Pause and report before any job projected above $40 or 6 hours.** `./lab cost` prints
`** BUDGET GATE **` when a run crosses either line. This is a reporting requirement, not a hard
block: the point is that nobody discovers the price afterwards.

---

## 5. Provisioning

```bash
./lab offers L40S            # live offers, ranked by TFLOPS per dollar-hour
./lab up 4 L40S 1.20         # rent 4 under $1.20/h, in parallel, ~4 min
./lab preflight              # the on-box gate — never skip it
```

### What `./lab offers` filters on, and why

- **reliability ≥ 0.95.** A box that dies at hour 4 of a 5-hour ladder costs the whole run.
- **CUDA ≥ 12.4.** Current boxes ship driver 575.x (CUDA 12.9).
- **disk ≥ 40 GB.** A 410M lens is 46 MB; a full E28 shard writes ~1 GB.
- **VRAM ≥ the model's floor**, from the table above.

### The toolchain, and why it is pinned the way it is

Image: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`. On top of it, `box_bootstrap.sh` does:

```bash
pip install --force-reinstall torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

- **All three together.** The 2.4.0 image's `torchvision` and `torchaudio` both break the GPTNeoX
  import. Pinning only `torch` leaves you with a broken model class.
- **cu128, not cu124.** The cu124 recipe is **dead as of 2026-08-12**: `nvidia-cudnn-cu12==9.1.0.70`
  was pulled from the index and torch 2.6.0 pins it exactly. Resolves to torch 2.11.0+cu128.
- **The install is verified, not assumed.** A pip that failed into `/dev/null` once left torch at
  2.4.0 on four boxes; the shard runner's `grep` swallowed the traceback and all 180 fits failed
  silently. `box_bootstrap.sh` now asserts `torch.cuda.is_available()` and imports `jlens` before
  reporting success.

### The remote interpreter

`/opt/conda/bin/python` on this image. **Not** `/venv/main/bin/python` — that path is for vast's
*template* image and appears in older specs.

---

## 6. Teardown — the order is not negotiable

```
run  →  pull  →  hash-verify  →  mirror  →  destroy
```

`./lab down` reads the receipt `./lab pull` writes and **refuses to destroy a box whose artifacts
were not verified**. Two lenses were lost with their boxes before this guard existed.
`ALLOW_UNVERIFIED=1` overrides it and makes you type a reason into `logs/teardown.log`.

A rented box gets a **read-only** HF token or none at all. Mirror from your own machine.

---

## 7. What has actually been spent

From vast.ai's billing records, not estimates.

**Split the number before quoting it.** This repository predates the estimator programme: it
previously hosted four prior research arcs, `recon/` and
`sad-audit/`, all removed from the working tree on 2026-08-12 and still in git history. The
account's all-time spend covers work this programme did not do.

Recomputed from vast.ai's billing records on **2026-08-15**:

| | spend | GPU-h | instances |
|---|---|---|---|
| prior arcs — before 2026-08-09 | $265.62 · **82.7%** | 187.6 | 38 |
| **the estimator programme** — 2026-08-09 onward | **$55.51 · 17.3%** | **66.3** | **22** |
| all-time | $321.13 | 253.9 | 60 |

The boundary is `7e82292`, the first commit adding the programme.

**Do not quote this table without re-running it** — it moves every time a box is rented, and it
was wrong by $24.63 on the programme's own line until 2026-08-15 because it had been transcribed
rather than recomputed. The split is a script, not a claim:

```bash
vastai show invoices --raw -s 2020-01-01 > /tmp/inv.json
python repro/lib/spend_split.py /tmp/inv.json
```

Within the $55.51, the two large items are **E28** (the read ladder, 4 × L40S, 21.4 GPU-h, $17.1 —
estimated in advance and matched) and the **E48 fitting campaign** (the OOD operators plus the
containment index boxes). Everything the paper does *with* those artifacts — E33, E36, E48, E48b,
E48c, E51, E52 — ran on CPU for nothing. See §4.

**60% of the all-time total was one box, and it was a prior arc**: instance `43514780`, 60.3
GPU-hours at $2.85/h = $171.52 on 2026-07-07 — 2.5 days on a single machine, 67% of all prior-arc
spend. Idle boxes are by a wide margin the largest avoidable cost in this account's history.
`./lab status` prints the burn rate in $/h, $/day and $/week for exactly that reason, and
`./lab health` H7 fails loudly if anything is still rented.
