# repro/ — the reproducibility layer

Every script here is **independently runnable** (`bash repro/NN_name.sh`) and also reachable
through the orchestrator (`./lab <verb>`). Use `./lab` day to day; read these when you want to
know exactly what a step does.

Numbering is the order you would meet them in, not a hard sequence.

| script | `./lab` verb | what it does | costs |
|---|---|---|---|
| `00_check_config.sh` | `doctor` | every external dependency, present or missing, with the exact fix | free |
| `01_setup_local.sh` | `setup` | build `.venv`, clone + pin + install `jlens` | free |
| `02_verify_local.sh` | `verify` | **TIER 0** — 196 assertions + bit-identity to the anchor | free |
| `03_fetch_artifacts.sh` | `fetch` | pull operators from the PUBLIC mirror, hash-verified. No corpus text — see `06_data.sh` | free |
| `04_fetch_results.sh` | `results` | pull the results JSON from the mirror; `--check` compares without writing | free |
| `05_mirror_results.sh` | — | push results to the mirror, then **re-list and verify** rather than trusting the upload's return code | free |
| `10_vast_auth.sh` | — | vast.ai API key, SSH key, ground rules | free |
| `11_vast_find_offers.sh` | `offers` | live offers ranked by fp32 TFLOPS per dollar-hour | free |
| `12_vast_provision.sh` | (`up`) | rent, wait, write the SSH alias, install the toolchain | **$$$** |
| `13_vast_preflight.sh` | `preflight` | on-box gate: torch, CUDA, TF32 off, anchor fidelity | minutes of rental |
| `14_vast_pull_and_verify.sh` | `pull` | rsync back, then compare remote and local SHA-256 | free |
| `15_vast_teardown.sh` | `down` | destroy — **refuses without a verified pull receipt** | saves money |
| `20_cost_estimate.sh` | `cost` | price any run before renting anything | free |
| `30_repo_health.sh` | `health` | the standing 10-check audit | free |
| `exp/*.sh` | `exp list` / `exp run` | one module per experiment, carrying the exact flags that produced the stored result. See `exp/README.md` | varies |
| `lib/common.sh` | — | shared helpers + the measured hardware model | sourced |
| `lib/box_bootstrap.sh` | — | runs **on** a box, piped over ssh | — |

## Two different things are called "tier", and they are not the same

This is the single most confusing thing in the repository, it has cost real time in review, and it
is written down here rather than fixed silently because both names are load-bearing in code.

**The `MODULE_TIER` axis answers "can this be run at all".** It is declared by every module in
`exp/*.sh` and it is what `RUN_ALL.sh --tier` selects on. Three values, and only three:

| `MODULE_TIER` | you must already have | what it does | count today |
|---|---|---|---|
| `T0` | this repository **+ the results JSON** | recomputes every statistic from stored JSON | 12 |
| `T1` | + the `.pt` operators | re-scores stored operators against the battery, **CPU only** | 26 |
| `T2` | + a GPU | refits a few operators **in memory** and compares | 7 |

(`exp/cv7_1b.sh` declares no tier on purpose — it is a detached launcher, not a module.
`RUN_ALL.sh` skips it *and names it*, because a silently skipped module is how coverage rots.)

**The `lab doctor` axis answers "what will this cost".** It is printed by `00_check_config.sh` and
appears in the script table above as "**TIER 0**" next to `02_verify_local.sh`. Four values:

| `lab doctor` TIER | means |
|---|---|
| TIER 0 | local CPU, free, ~2 min — the full test suite + bit-identity to the anchor |
| TIER 1 | **one GPU-hour, ~$0.50** — provision a box and run the on-box preflight |
| TIER 2 | a single experiment, priced by `20_cost_estimate.sh` first |
| TIER 3 | a full ladder — E28 is 180 fits on 4 boxes, ~21 GPU-h, ~$17 |

**The collision is at 1, and it inverts the meaning.** `lab doctor` TIER 1 rents a GPU. `MODULE_TIER`
T1 is CPU-only, spends nothing, and a GPU there is *forbidden* — `results/d2_device_crossvalidation.json`
measured CUDA-vs-CPU at `2.774e-04`, above the `1.6e-4` threshold at which one scoring decision
flips, so a GPU T1 sweep returns "does not reproduce" by construction.

**There is no `T3` module tier.** A full ladder is not a module and not one `.sh`: it is the fleet
path, `./lab up <n>` → `./lab shard <file>` (one line per job, spread over the boxes) → `./lab pull`
→ `./lab down`.

### What T2 actually does, and what it does not

T2 is a **spot-check on the fitting step, not a rebuild.** Two facts settle it:

* **No module in `exp/` — T2 included — declares a `.pt` as an output.** Zero of 46, against 382
  `.pt` on disk. Every T2 module emits a JSON comparison.
* **`r6_within_source` and `r7_matched_pools` guard their fits with `if not os.path.exists(path)`.**
  On a tree where `03_fetch_artifacts.sh` has already run, those two refit **nothing** — they load
  the stored operator and re-score, silently behaving like T1.

Seven scripts in `experiments/` can persist an operator at all — `trainval.py`,
`cv6_per_family_ladder.py`, `t65_ckpt_readout.py`, `e66b_determinism_floor.py` (`torch.save`) and
`r6_within_source.py`, `r7_matched_pools.py`, `t2_fastfit.py` (`lens.save`). Only one module ever
passes `trainval.py --save-lens`, and that is `e62_ladder1b_band`.

So **you cannot chain T2 → T1 → T0 to avoid the artifact mirror**, and the reason is measured, not
stylistic: a refit does not reproduce a stored operator bit-identically. Post-provenance operators
refit to `2.441e-04`, inside one fp16 step; the E48 panel refits to `1.0986e-03` and E66b's verdict
on that is CONFIRMED DEFECT. T1's test is payload **bit**-identity, so feeding it your own refits
returns "does not reproduce" by construction — the same manufactured false negative as running T1
on a GPU. The scope is stated plainly: T2 was sampled at n=2, deliberately
straddling the provenance boundary.

To genuinely rebuild from scratch you would drive `trainval.py --save-lens` yourself, once per
operator, outside the module system. That validates the **recipe**; it does not reproduce the
**artifacts**.

### Where the record lives — and why T0 is not repo-only any more

`results/` was untracked from git on **2026-08-22** (`.gitignore` carries a blanket `results/`; the
last commit holding them is `e26993c`). `git ls-files results/` returns **0**.

| surface | carries |
|---|---|
| **GitHub** | code, docs, pre-registrations, `tests/`, this harness, `scripts_at_run/` |
| **the public HF mirror** (`AnonymousInterpScience/pythia-jlens-artifacts`) | **the results JSON *and* the `.pt` operators**. NO corpus plaintext — licence; rebuild with `06_data.sh --build` |

Both halves of the record are on the mirror. GitHub carries the *means* of reproduction.

**Consequence, and it contradicts `CLAUDE.md` §2.1 as written.** That section promises three
commands from a clean clone — `uv sync`, `01_setup_local.sh`, `RUN_ALL.sh --tier T0`. A fresh clone
now has zero results files, so every T0 module stops at `require_inputs` with `MISSING INPUT`. The
sequence needs a fourth command:

```bash
bash repro/04_fetch_results.sh        # the results JSON, ~8 MB, from the mirror
```

And `RUN_ALL.sh`'s preflight checks `.venv`, `jlens`, and `.pt` for T1/T2 — but **never checks for
the results JSON**, so at T0 the failure surfaces as a per-module input error instead of the single
"here is the fix" stop that RUN_ALL exists to give.

### Two stale artefacts of that move, left in place until someone fires them deliberately

* `.gitignore` still carries a **"results/ CUT-OVER, staged 2026-08-15, NOT YET FIRED"** block
  asserting the results JSON "currently lives in git (418 files, ~8 MB)", commented out — while a
  later blanket `results/` rule already untracked it. The cut-over fired by another route and the
  staged block was never removed. **The two blocks contradict each other; the later one is true.**
* `04_fetch_results.sh`'s own header still says it "is a no-op against a clone" until the cut-over
  fires. It is not a no-op any more; it is required.

## Conventions

- `set -euo pipefail` everywhere. A script that cannot do its job stops rather than continuing.
- Anything that spends money or destroys something calls `confirm` first. `FORCE=1` skips the
  prompt for automation; it does **not** skip the pull-receipt guard.
- Read-only by default. `doctor`, `cost`, `offers`, `health`, `status` change nothing.
- Every header comment says what the script does, what it needs, and what it costs.

## Why the guards exist

None of these are hypothetical. Each is the residue of a real failure:

- **the pull-receipt guard** — two lenses were destroyed with their box
- **the verified toolchain install** — a pip that failed into `/dev/null` left torch at 2.4.0 on
  four boxes; the runner's `grep` swallowed the traceback and 180 fits failed silently
- **the TF32 assertion** — a 10-bit mantissa breaks the anchor gate's 2e-5 tolerance, silently
- **the torch/torchvision/torchaudio trio pin** — the image's torchvision and torchaudio both
  break the GPTNeoX import
- **cu128 not cu124** — `nvidia-cudnn-cu12==9.1.0.70` was pulled from the index on 2026-08-12 and
  torch 2.6.0 pins it exactly
- **the budget gate** — a cost estimate was once wrong by 5× because it priced the largest N
  instead of the sum over the N grid

## Adding a check

`30_repo_health.sh` is the place. Use `ok` / `warn` / `bad` from `lib/common.sh` so the summary
counts it: **`bad` blocks, `warn` is a known gap**. Add a row to the table in `REPRODUCIBILITY.md`
§7 and state which real failure the check would have caught.
