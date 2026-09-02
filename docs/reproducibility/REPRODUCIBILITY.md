# REPRODUCIBILITY.md — start here

**If you are here to reproduce or review the paper, read [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) first** — it is
the front door and it is shorter. This page is the deeper reference: the configuration matrix, the
tiers, the layout rules, and a full account of what is *not* reproducible and why.

```bash
./lab doctor     # what you must configure, and the exact fix for each gap   free
./lab setup      # build .venv, vendor the pinned anchor                     free
./lab fetch      # artifacts from the HF mirror — REQUIRED, not optional     ~50 MB
./lab verify     # the gate: 196 assertions + bit-identity to the anchor     free, ~2 min
./lab health     # the standing audit, 10 checks                             free
./lab            # the map: what exists, what is running, what it costs
```

Verified end to end from a virgin clone: **196/196 assertions pass** (10 test files; an 11th,
`tests/test_e28_eval_compliance.py`, runs outside the gate). Skipping
`./lab fetch` fails Tier 0 on `test_anchor_fidelity` with a missing lens — that is a missing
artifact, not a broken repo, and `./lab verify` now says so instead of printing a traceback.

If `./lab verify` fails, stop. Nothing downstream can be trusted, and no GPU spend is justified.

---

## 1. What this repo is

One research programme: **the Jacobian lens is an estimator, so it inherits a distributional
dependence the logit lens cannot have.** Papers are exports of it.

**This repository is older than that programme.** It previously hosted four prior research arcs, removed from the
working tree on 2026-08-12 and still in git history. Two consequences worth knowing before you
read any aggregate: the git log covers work this programme did not do, and **82.7% of the
account's all-time compute spend ($265.62 of $321.13) predates it** — the estimator programme
itself is **$55.51** (recomputed 2026-08-15). See `COMPUTE.md` §7, and recompute rather than quote
it: `repro/lib/spend_split.py`.

- **`src/` and `experiments/`** — the live work. `src/` is the shared library (the modules other
  modules import, which import nothing local themselves); `experiments/` is everything that
  produces a result. **Self-contained**: together they reference nothing outside the repo except
  the vendored `jacobian-lens/`. There is no `pythia/` directory — it was dissolved on 2026-08-15.
- **`docs/context/CONTEXT.md`** and **`docs/context/RESULTS_TAXONOMY.md`** — every claim with its source file
  and its reproducibility tier. **`docs/experiments/preregs/`** carries one document per experiment,
  pre-registration through verdict.
- **`results/*.json`** — the source of truth for every number. If a figure is not in one of
  these files, it does not exist.
- **`repro/`, `lab`, `COMPUTE.md`** — this reproducibility layer.
- the prior research arcs, `experiments/` — **prior research arcs,
  removed from the working tree on 2026-08-12 and still fully in git history.** Nothing in
  the current programme referenced them. Restore any with `git checkout <commit>^ -- <path>`.
  `docs/` joined them on 2026-08-15. See §6.

---

## 2. What you must configure

`./lab doctor` checks all of these and prints the fix for whichever is missing.

| # | thing | needed for | how |
|---|---|---|---|
| 1 | Python ≥ 3.10, `uv` | everything | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| 2 | **`jacobian-lens` @ `581d398`** | **everything** | `./lab setup` |
| 3 | Hugging Face read access | downloading Pythia weights | `huggingface-cli login` |
| 4 | HF **write** token | mirroring artifacts (optional) | laptop only — never on a rented box |
| 5 | `vastai` CLI + API key | GPU tiers only | `pip install vastai` → `./lab doctor` |
| 6 | SSH keypair registered with vast.ai | reaching a rented box | `ssh-keygen -t ed25519` |
| 7 | `tmux` | `./lab watch` | `brew install tmux` |

### The single point of failure

**`jlens` is imported 57 times.** Every fit, read and write routes through the anchor's released
package, vendored at `581d398`, gitignored, never modified in place. Four symbols carry it:

```
jlens.hf.from_hf / Layout           26 call sites
jlens.lens.JacobianLens             21
jlens.fitting.fit / SKIP_FIRST_N     5
jlens.hooks.ActivationRecorder       5
```

If that repository disappears upstream, nothing here runs. It is not currently mirrored anywhere
under our control — the highest-value single fix available to this repo.

---

## 3. The tiers

| tier | what it proves | cost | command |
|---|---|---|---|
| **0** | machinery correct; read path **bit-identical** to the anchor | free, 2 min | `./lab fetch` → `./lab verify` |
| **1** | a rented box is actually usable | ~$0.50 | `./lab up 1` → `./lab preflight` |
| **2** | one experiment reproduces | priced first | `./lab cost …` → `./lab run …` |
| **3** | a full ladder | budget-gated | `./lab cost --experiment e28` |

**Tier 0 is mandatory and never skipped.** Tiny-first: a failed $2 job is fine, a silently-wrong
$50 job is not. Tier 0 exercises the exact code path the GPU jobs use, on CPU-sized input.

### What Tier 0 actually asserts

- our transported activation equals `jlens.lens.JacobianLens.apply` to **max|diff| 0.00e+00**
- the anchor's own readout reproduces `hf(...).logits` to **2.3e-3**
- fitting honours `skip_first` (the released code hardcodes 16 at `jlens/fitting.py:42` while the
  paper text says 0 — we use 16 and disclose it)
- **no centering anywhere**: μ never enters a lens vector or a write

That second-to-last point matters to an auditor: anyone can write *a* lens. This shows ours is the
operator the original authors published, not a lookalike.

---

## 4. Running an experiment

```bash
./lab cost 410m 200 L40S       # price it BEFORE renting. Always.
./lab offers L40S              # live offers, ranked by TFLOPS per dollar-hour
./lab up 4 L40S 1.20           # rent 4 in parallel, ~4 min
./lab push                     # rsync code (artifacts and corpora excluded on purpose)
./lab preflight                # on-box gate: torch, CUDA, TF32 off, anchor fidelity
./lab shard jobs.txt           # one line per job, round-robin across the fleet
./lab watch                    # tmux: dashboard + one window per box
./lab pull                     # rsync back, then hash-verify file by file
./lab down --all               # refuses any box without a verified pull receipt
```

Every remote job runs inside a named remote tmux session (`train`), so it survives ssh dropping and
your laptop closing.

### The teardown order is not negotiable

```
run  →  pull  →  hash-verify  →  mirror  →  destroy
```

`./lab down` reads the receipt `./lab pull` writes and refuses otherwise. Two lenses were lost with
their boxes before this guard existed. `ALLOW_UNVERIFIED=1` overrides it and makes you type a
reason into `logs/teardown.log`.

---

## 5. What is NOT reproducible from this repo alone, and why

Being explicit about this is the point of the document.

Nothing large is in git. **Everything a run needs is either regenerated by a script or pulled
from the mirror**, and the two commands together give you a working tree:

```bash
git clone <ANONYMISED CODE URL> && cd readonly-lens
./lab setup          # code + the pinned anchor
./lab fetch --all    # every lens + the corpus pool (~3 GB; --minimal is ~50 MB)
./lab verify         # the gate
```

| not in git | why | where it comes from |
|---|---|---|
| `*.pt` lenses (74, 3 GB) | too large | **mirror** — 69 Pythia lenses under `results/` |
| **corpus plaintext (57 MB)** | licensed (CC-BY-SA, Common Crawl, GitHub) | **mirror** — `corpora/`. `manifest.json` is *also* in git |
| the anchor's eval sets | third-party | inside `jacobian-lens/data/evaluations`, fetched by `./lab setup` |
| `paper/`, PDFs | gitignored by design | `./lab docs` |

### Why the corpora are mirrored and not merely regenerated

`src/build_corpora.py` is deterministic in its own logic — no seed, no shuffle, it takes the first
2400 documents of each Pile component with ≥600 characters, in stream order. But it calls

```python
load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)   # no revision=
```

with **no revision pin**, so its determinism is borrowed from a third party. Verified 2026-08-12:
stream order still reproduces exactly (first-5 documents identical for all five components)
against dataset sha `3be90335`, last modified 2023-08-31. **That is a snapshot, not a guarantee.**
A re-upload or re-shard upstream changes the order silently, and the manifest's SHA-256 check
would then fail with no way to recover the original pool.

This is not a hypothetical concern for this project specifically: **E28's seed blocks are index
ranges into that exact ordering.** A different pool makes "seed 0" denote different prompts and the
seed variance uninterpretable — precisely what `src/build_corpora.py`'s own docstring exists to
prevent. So the pool is mirrored, and `manifest.json` (per-file SHA-256, window counts, seed-block
ranges) stays in git so you can *verify* whatever you obtain.

**Known non-determinism.** Fits are seeded and the corpus draw is recorded, but GPU kernel
non-determinism means a refit reproduces the *result* to within the sampling floor, not the *bytes*.
That floor is itself measured — it is the denominator of the corpus-variance ratio *R*.

---

## 6. Taxonomy — where everything lives, and where new work goes

Four layers. Each has one job, and the rule for adding to it.

```
┌─ CONTROL ────────────────────────────────────────────────────────────┐
│ lab                    one entry point; delegates, holds no logic     │
│ repro/*.sh             config · setup · verify · fetch · provision    │
│ repro/exp/*.sh         ONE MODULE PER EXPERIMENT, with the exact flags│
│ repro/lib/             shared helpers + the measured hardware model   │
├─ PROGRAMME ──────────────────────────────────────────────────────────┐
│ src/                   shared library: imported, never an experiment  │
│ experiments/           one file per experiment, tNN_<name>.py         │
│ tests/                 one test_<module>.py per module                │
│ tools/                 analysis and audit that is not an experiment   │
│ results/*.json         THE SOURCE OF TRUTH FOR EVERY NUMBER           │
│ corpora/               fitting pools; plaintext gitignored            │
│ figures/               regenerated from results/, never typed in      │
├─ ARGUMENT ───────────────────────────────────────────────────────────┤
│ HANDOFF.md       current state, with provenance per claim       │
│ docs/context/RESULTS_TAXONOMY every claim, tiered A/B/D                      │
│ docs/experiments/preregs/     one doc per experiment: prereg → verdict       │
│ docs/archive/prereg/  the raw pre-registrations, as timestamps       │
│ paper/                 the NeurIPS submission source (TRACKED)        │
├─ EXTERNAL ───────────────────────────────────────────────────────────┤
│ jacobian-lens/         vendored @ 581d398 · gitignored · NEVER edited │
│ thirdparty/            vendored comparisons · gitignored              │
│ HF pythia-jlens-…      results/ + corpora/ — layout matches 1:1       │
└──────────────────────────────────────────────────────────────────────┘
```

| adding… | goes in | and you must also |
|---|---|---|
| a new experiment | `experiments/tNN_<name>.py`, next free NN | write `docs/experiments/preregs/E<NN>_<name>.md` **before** it runs, with a decision rule fixed in advance, and add a `repro/exp/` module |
| a new metric or shared routine | `src/<name>.py` | add `tests/test_<name>.py` — H6 fails without it |
| a result | `results/<runner>_<model>.json`, stamped via `provenance.write_result` | cite the filename in its `docs/experiments/preregs/` doc — H4 checks this |
| an artifact | HF mirror, via `./lab mirror` | record its SHA-256 in `ARTIFACTS.md` — H3 checks this |
| a repro/infra step | `repro/NN_<name>.sh` | give it a `./lab` verb and a row in `repro/README.md` |
| a health invariant | `repro/30_repo_health.sh` | use `ok`/`warn`/`bad`, and name the real failure it would have caught |

**The programme is self-contained.** Audited by AST: it references nothing outside itself except
the vendored `jacobian-lens/`. Keep it that way — a new external dependency is a design regression.

### What was removed, and how to get it back

Prior research arcs were removed from the working tree on **2026-08-12**: four separate
research lines, `experiments/`, the
top-level `tests/` and `tools/`, the eab-era `Makefile`, and five superseded top-level docs. None
was referenced by `pythia/` code. They are cited by published claims and remain **fully in git
history** — the removal is one ordinary commit, not a rewrite:

```bash
git log --oneline --diff-filter=D -- <arc-dir>/  # find the removing commit
git checkout <that-commit>^ -- <arc-dir>/        # restore it, whole
```

Tracked repo went from **82 MB / 700+ files to 8.3 MB / 335 files**. `./lab health` H10 checks
that no surviving document points at a path that no longer exists.

---

## 7. The standing audit

`./lab health` — ten checks, each the residue of something that actually went wrong here.

| check | what it catches |
|---|---|
| H1 | vendored anchor missing, off-pin, or modified in place |
| H2 | model binaries or large files committed to git |
| H3 | artifacts on disk with no SHA in `ARTIFACTS.md` or no provenance JSON |
| H4 | claims in the write-ups that do not name a source file |
| H5 | test suite red |
| H6 | a module with ≥4 importers and no test |
| H7 | **a GPU still billing** |
| H8 | uncommitted work |
| H9 | licensed corpus plaintext tracked in git |
| H10 | documentation pointing at files that no longer exist |

A **FAIL** blocks — fix it before spending. A **warn** is a known gap someone will trip over.

### Known warnings as of 2026-08-12

- **H2/H9** — 55 MB of Pile plaintext (`corpora/*.jsonl`) is tracked. It is fully
  reproducible from `src/build_corpora.py` + `manifest.json`, so untracking it loses nothing. Not done
  yet because `git rm --cached` stops future growth but leaves the blobs in history; the real fix
  is a history rewrite, which is destructive and needs an explicit decision.
- **H3** — 47 of 74 lenses are described in `ARTIFACTS.md` as a group rather than named
  individually; 19 lack a provenance JSON. Root cause: `experiments/t23_pq_ladder.py` writes provenance on the
  unmatched path but not the `--require-full-window` path.
- **H6** — `experiments/t13_transport_controls.py` (8 importers), `experiments/t2_fastfit.py` (7), `experiments/t17_reaggregate.py` (5)
  have no tests. The first is the highest risk in the repo: it supplies the **controls** by which
  every result is graded.
- **H10** — 3 dangling documentation references.
