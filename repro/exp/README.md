# `repro/exp/` — one shell module per executable unit of research

Each `.sh` here is **the canonical invocation** of one experiment. Not "a way to run it" — *the*
way, with the flags that produced the stored result, in git, beside the reason those flags are
those flags.

## Why this exists

An experiment used to be "a `.py` someone ran once with flags they remembered." The flags are
load-bearing and the memory is not:

* `--band 9,21` means `[9,21]` to `experiments/trainval.py` (inclusive) and `[9,20]` to `experiments/t33_logit_baseline.py`
  (exclusive). The 0.35–0.85 depth default gives `[8,20]` at 410M. **E49's 410M cells are on the
  wrong band** because that distinction lived in shell history.
* `--derangement random` vs `cyclic` is the difference between E33 v1 (control failed, 4/6) and
  E33 v2 (control fires).
* `--rstrip` changes the readout token on 157/551 eval items.

None of that was recoverable from a results file. Now it is: every module records its own argv,
its script's SHA-256, the commit, and the SHA-256 of every input, via `src/provenance.py`.

## Contract every module honours

1. **Declares INPUTS and OUTPUTS** and refuses to start if an input is missing.
2. **Checks the real exit code** via `PIPESTATUS` — a `grep` filter in a pipeline masks a
   `SystemExit`, and that has cost this programme a run.
3. **Verifies its outputs afterwards**: present, non-empty, and carrying a provenance block whose
   recorded script hash still matches disk. *Presence is not integrity.*
4. **Prints cost first** and refuses anything above the budget gate without `--yes`.

## Usage

```bash
./lab exp list                 # every module, what it produces, what it costs
./lab exp run e48_crossover    # or: bash repro/exp/e48_crossover.sh
bash repro/exp/e36_qladder.sh --dry-run     # print the contract, execute nothing
```

`--dry-run` prints the full contract without running, which makes each module readable as a spec.

## Provenance, and what it does and does not promise

`src/provenance.py` stamps every results JSON with: script path + SHA-256, git commit + branch +
dirty flag + diff hash, argv, SHA-256 of every declared input, environment versions, TF32 state,
and `payload_sha256` — the hash of the result **content with the provenance block excluded**.

That last field is the reproducibility check. Rerun a module, compare `payload_sha256`: a match
proves the science is identical even though the timestamp is not. Including the timestamp would
make every run differ and the check would be worthless.

This does **not** promise bit-reproducibility. It promises **accountability**: given any results
file you can find the code, the inputs and the command that produced it, and tell whether the tree
was clean at the time. Most results in this programme were produced mid-edit; that is recorded
rather than hidden.

Audit the whole tree with:

```bash
.venv/bin/python tools/migrate_provenance_paths.py --verify
```

## Modules

| module | produces | cost |
|---|---|---|
| `e33b_tstats.sh` | `e33b_tstats_410m.json` — the prose-only t statistics, stored | free |
| `e48_gate.sh` | `e48_competence_gate_410m.json` — model competence per fitting corpus | ~40 min CPU |
| `e48_crossover.sh` | `e48_crossover_410m.json` — the S3 crossover adjudication | ~25 min CPU |
| `e48b_exposure.sh` | `e48b_exposure_growth.json` — containment vs stream coverage | ~15 min CPU |
| `e48c_exposure_read.sh` | `e48c_exposure_vs_read.json` — does exposure order the read? | free |
| `e51_interaction.sh` | `e51_interaction_variance.json` — the corpus×set decomposition | free |
| `e36_qladder.sh` | `e36_qladder_410m.json` — **the Q-ladder** (vary the READ distribution) | ~2–4 h CPU |

Order matters where inputs chain: `e48_gate` → `e48_crossover` → `e48c`, and
`e48b` → `e48c`. Each module fails loudly rather than running on a missing input.
