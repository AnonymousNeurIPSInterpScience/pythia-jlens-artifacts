# A Jacobian Lens Is an Estimator

**A corpus-averaged derivative is a fitted object. This repository measures what that fitting costs,
on a model family whose training data is published.**

<sub>Last updated 2026-08-22 · not accepting pull requests</sub>

---

## Quick start — four commands, free, about two minutes

```bash
uv sync
bash repro/01_setup_local.sh          # vendors github.com/anthropics/jacobian-lens @ 581d398
bash repro/04_fetch_results.sh        # the results JSON from the public mirror. ~8 MB
bash repro/RUN_ALL.sh --tier T0       # recomputes every statistic in the paper. free, ~1 min
```

**Lines 2 and 3 are both mandatory and are the two steps most often skipped.**

* `jacobian-lens/` is gitignored, so a fresh clone does not have it, and without it five of the
  eight gates fail with `ModuleNotFoundError: No module named 'jlens'`.
* **`results/` is gitignored in full.** A fresh clone has an *empty* `results/`, so without line 3
  every T0 module fails with `FileNotFoundError`. The results tree lives on the public mirror
  because it is the record of record, not because it is large.

**The artifact mirror is public — no account, no login, no token:**

> **https://huggingface.co/AnonymousInterpScience/pythia-jlens-artifacts**

`repro/03_fetch_artifacts.sh` and `repro/04_fetch_results.sh` both default to it; set `HF_REPO`
only to point at a different mirror.

| tier | what it does | needs |
|---|---|---|
| **T0** | recomputes every statistic from the stored results JSON | this repo + `04_fetch_results.sh` (~8 MB) |
| **T1** | re-scores every read from the stored `.pt` operators, on CPU | + `03_fetch_artifacts.sh --all` (~16 GB) |
| **T2** | refits the operators from scratch | a GPU, fp32, TF32 off, and money |

The corpus **plaintext is deliberately not on the mirror** — it is third-party text and two files
are CC BY-SA, which cannot be relicensed as the mirror's CC BY 4.0. Only the manifests ship.
Rebuild the pool at its pinned dataset revisions with `bash repro/06_data.sh --build`.

The longer-form entry points, once T0 is green:

```bash
./lab doctor     # what you must configure, with the exact fix for each gap        free
./lab setup      # build .venv, clone + pin the vendored reference library         free, ~3 min
./lab fetch      # artifacts from the HF mirror — needed for T1 and above          ~50 MB
./lab verify     # the gate: bit-identity to the reference implementation          free, ~2 min
./lab health     # the standing audit                                              free
```

The repository was executed by LLM-based coding agents under human direction and review.

**If `./lab verify` fails, stop.** It proves the read path is bit-identical to the reference
library's own code (`max|diff| = 0.00e+00`), not merely that it runs. **It does not check the readout
*position*** — see §The correction below, which is exactly the defect it stayed green through.

| read next | for |
|---|---|
| **[`docs/context/CONTEXT.md`](docs/context/CONTEXT.md)** | **start here.** Every experiment, its verdict, its results file |
| [`docs/context/RESULTS_TAXONOMY.md`](docs/context/RESULTS_TAXONOMY.md) | every claim, tiered A / B / D |
| [`docs/experiments/INDEX.md`](docs/experiments/INDEX.md) | one row per experiment, generated |
| [`docs/context/AGGREGATION_POLICY.md`](docs/context/AGGREGATION_POLICY.md) | why `min` is primary. Read before citing a number |
| [`REPRODUCIBILITY.md`](docs/reproducibility/REPRODUCIBILITY.md) | tiers, configuration, what is *not* reproducible and why |
| [`COMPUTE.md`](docs/reproducibility/COMPUTE.md) | the measured hardware model, every price, provisioning and teardown |
| the project handoff record (not published) | the submission: what it is, how to build it, what is open |

## The idea

The Jacobian lens reads a model by transporting a mid-layer activation through

$$J_\ell = \mathbb{E}_{x\sim P}\left[\partial h_{\text{final}}/\partial h_\ell\right]$$

a derivative **averaged over a corpus**. That average makes it an *estimator*, and an estimator is
wrong in two ways that behave differently:

```
‖Ĵᴾ_N − J^Q‖   ≤   ‖Ĵᴾ_N − J^P‖    +    ‖J^P − J^Q‖
 total error        sampling            corpus bias
                    O(N^−½)             O(1) in N
                    buy it down         cannot buy it down
```

The logit lens is the `J = I` special case: no fitted parameters, so no distributional dependence to
inherit. That contrast is the whole programme.

**Why Pythia.** All eight sizes saw identical data in identical order, the token stream is published,
and the deduped suite trains ~1.45 epochs, so **corpus membership implies exposure**. It makes
in-distribution a measurement rather than a judgement.

## What was found

Stated flatly, with the caveats that belong to each. Full account in
[`docs/context/CONTEXT.md`](docs/context/CONTEXT.md).

* **Below 410M the comparison is degenerate for an algebraic reason unrelated to the Jacobian.** The
  concept rows of the unembedding are effectively rank-one, so no transport can separate what the
  unembedding has already collapsed. This is computed from the unembedding alone and is the most
  robust result here.
* **Both corpus roles matter, and comparably** — 53.4% of grid variance for the corpus that built the
  operator against 44.6% for the corpus supplying read context. An earlier version of this repository
  reported roughly an order of magnitude; that was a readout defect, see below.
* **Corpus identity is a real source effect, not a sampling artifact** (between-source share 0.989)
  and not substantially a lexical-composition effect.
* **Neither layer-permutation null isolates layer correspondence under `min`.** A random derangement
  beats the real operator on 84 of 120 draws at the corrected readout (corpus-clustered
  t(7) = −1.92, p = 0.096). A pre-registered second null — the twelve non-trivial cyclic shifts of the
  band, which also remove every layer's own Jacobian — loses to the real operator on 8 of 8 corpora
  (t(7) = +7.70). Two nulls that both remove correspondence give opposite conclusions, so the answer
  depends on the aggregation and on the null's cross-layer dependence. Under `persist` and the band
  mean the real operator beats both nulls on 8 of 8. Neither null is used as an identifying control
  and no claim rests on them — see
  [`docs/experiments/DA1_OUTCOME.md`](docs/experiments/DA1_OUTCOME.md).
* **Nothing measured predicts which corpus is good**, and the ranking does not transfer across scale.

## The correction, stated up front

Scoring read the final token of the **unstripped** prompt. The released evaluation prompts end in a
trailing space so that `prompt + target` concatenates; BPE absorbs that space into the target's
leading space, so the scored token — a bare space — **does not occur in the tokenised sequence at
all**. It affected **157 of 551** items, and one evaluation set by a factor of 30.

Every affected result has been re-scored. The ledger is mechanical:

```bash
.venv/bin/python tools/readout_exposure.py     # which results sit on the corrected readout
.venv/bin/python tools/build_provenance.py --check   # every experiment doc's provenance, verified
```

Three evaluation sets are untouched by the change and score **identically** under both readouts.
That is the internal control proving the two arms differ in the readout token and nothing else.

## Layout

| path | what it is |
|---|---|
| `src/` | the shared library: modules other modules import, importing nothing local themselves |
| `experiments/` | one file per experiment, `tNN_<name>.py`. Everything that produces a result |
| `results/` | every number, with provenance. **The source of truth.** Mirrored to Hugging Face; untracked from git |
| `docs/` | **everything non-code**: context, experiment record, validity (gate 1), reproducibility (gate 2), archive. Start at [`docs/README.md`](docs/README.md) |
| `repro/`, `lab` | the reproducibility layer. `docs/reproducibility/audit-prompts/` holds three independent audit prompts |
| `tools/` | generators and ledgers; nothing here is an experiment |
| `paper/` | the submission, its argument, and `fundamentals/` — the source method and PDF it replies to |
| `figures/` | regenerated from `results/`, never typed in |
| `corpora/` | builders and manifests. Plaintext is gitignored, never redistributed |
| `jacobian-lens/` | the vendored reference library, pinned and never modified |

## Discipline

The working discipline, in the form that bites most often: never report a number that is not in a results
file; never accept a proxy for the thing you care about; a control that cannot fail is not a
control; pre-register the decision rule and never reinterpret it afterwards.
