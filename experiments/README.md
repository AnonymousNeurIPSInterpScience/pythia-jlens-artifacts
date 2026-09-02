# experiments/ — one file per experiment

Each `tNN_<name>.py` here runs one experiment and writes one or more files into `results/`. The
**write-up** for each — question, pre-registered rule, controls with the numbers they produced,
verdict, flagged deltas — is in [`../docs/experiments/preregs/`](../docs/experiments/preregs/). The **canonical
invocation**, with the exact flags that produced the stored result, is in
[`../repro/exp/`](../repro/exp/).

Three files, three jobs. Do not read a number out of a script; read it out of `results/`.

## How these run

```bash
./lab exp list                              # every module, what it produces, what it costs
bash repro/exp/e52_factorial.sh --dry-run   # the full contract, executing nothing
bash repro/exp/e52_factorial.sh             # the run
```

Run them through `repro/exp/` rather than directly. The module declares its inputs and refuses to
start without them, checks the real exit code via `PIPESTATUS`, and verifies afterwards that each
output exists *and* carries a provenance block whose recorded script hash matches disk. Presence
is not integrity.

## Import rules

- The shared library is `../src`, reached by one `sys.path` entry at the top of each file.
- A sibling experiment is imported by bare name and resolves for free, because Python puts a
  script's own directory on `sys.path`. Five files here are dual-role — a runnable experiment that
  is *also* imported by others: `t2_fastfit.py` (24 importers), `t13_transport_controls.py` (8),
  `t17_reaggregate.py` (5), `t35_containment.py`, `t5a_capability_screen.py`. They keep their names
  because stored provenance records point at them.
- `results/` and `corpora/` are one level up, addressed as `os.path.join(HERE, "..", "results")`.

## The load-bearing experiments

Ordered by the axis they belong to. Every one has a write-up in `../docs/experiments/preregs/`.

| script | experiment | writes |
|---|---|---|
| `e28_ladder_410m.py` | E28 — the read ladder: corpus vs `N` | `results/ladder410/`, `results/ladder1b/` |
| `t33_logit_baseline.py` | E33 — the missing `J = I` arm | `e33_logit_baseline_410m_v2.json` |
| `t33b_store_tstats.py` | E33b — its t statistics, stored | `e33b_tstats_410m.json` |
| `t51_interaction_variance.py` | E51 — corpus × set interaction | `e51_interaction_variance.json` |
| `t35_containment.py` | E35 — the n-gram containment index | `e35_containment_shard0.json`, `results/m1/` |
| `t48b_exposure_growth.py` | E48b — containment vs stream coverage | `e48b_exposure_growth.json` |
| `t48_competence_gate.py` | E48 gate — model competence per corpus | `e48_competence_gate_410m.json` |
| `t48_crossover.py` | E48 — the S3 crossover on the fit axis | `e48_crossover_410m.json` |
| `t48c_exposure_vs_read.py` | E48c — does exposure order the read? | `e48c_exposure_vs_read.json` |
| `t36_qladder.py` | E36 — the Q-ladder, shifting the read | `e36_qladder_410m.json` |
| `t52_factorial.py` | E52 — the fit × read factorial | `e52_factorial_410m.json` |
| `t38_jgeometry.py` | E38 — why small models fail | `e38_jgeometry.json` |
| `t31_local_bakeoff.py` | E31 — the predictor bake-off | `e31_local_bakeoff_410m.json` |
| `t17_reaggregate.py` | T17 — re-score under four aggregations | `t17_reaggregate_{160m,410m}.json` |
| `t22_bootstrap_ci.py` | T22 — hierarchical intervals | `t22_bootstrap_ci*.json` |
| `t20_ablation_kl.py` | T20 — ablation writes | `t20_ablation_kl_*.json` |

The remaining `tNN_*.py` are earlier or superseded arms; `../docs/context/RESULTS_TAXONOMY.md` tiers every
one of their results by how confidently it reproduces, including the retracted ones.

## Adding an experiment

1. `experiments/tNN_<name>.py`, next free `NN`.
2. `docs/experiments/preregs/E<NN>_<name>.md` **before it runs**, with the decision rule fixed in advance.
3. Write the result with `provenance.write_result(...)` — two lines, and the file becomes auditable.
4. `repro/exp/<name>.sh` carrying the exact flags.
