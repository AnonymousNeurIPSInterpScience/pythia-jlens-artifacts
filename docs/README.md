# docs/ — everything that is not code

**Restructured 2026-08-22.** This was `specs/`. It is now the single home for context, experiment
records, validity work, reproducibility work and the archive. Code lives in `src/`, `experiments/`,
`tools/`, `repro/`, `tests/`. The paper lives in `paper/`. Results live in `results/` (mirrored to
Hugging Face; see [`reproducibility/the release procedure`](reproducibility/the release procedure)).

---

## THE STANDING RULE — A NULL IS A FINDING

**The objective is to quantify, not to confirm.** The three subtheses (S1, S2, S3) are intuitions
that told us where to point the instrument. They are not commitments, and nothing here is improved
by any of them being true. A measured null with a stated bound is a result of equal or greater
standing than a positive.

Nothing in this repository is "dead" because a subthesis came out one way. The only things that go
out of date are **unstored numbers** and **stale transcriptions**, and both are fixed by recomputing.

**A null must be earned, not defaulted to.** An underpowered miss is not a null: state the effect
size the design could have detected. See [`validity/`](validity/) for a worked example — S3's
membership null is a *bound* of ±0.02 AUC, not an absence, because the eight-corpus panel has a
minimum detectable effect of 0.0268.

Full statement: `CLAUDE.md` §1.

---

## THE TWO GATES

Rigor here has two independent axes. They fail differently, they are audited differently, and
conflating them is how a programme convinces itself of something false.

| | **GATE 1 — VALIDITY** | **GATE 2 — TRUTH** |
|---|---|---|
| **asks** | *Assuming the code is a faithful implementation and every result reproduces — are the experiments themselves correct?* | *Is the code actually a faithful implementation, and do the results actually reproduce?* |
| **catches** | wrong estimand, wrong replication unit, unfired control, underpowered null, a decision rule whose denominator is near zero, an ordering that only exists under one aggregation | wrong readout token, a scorer that pools differently from the spec, a results file whose `inputs` field lies, a number no script computes |
| **method** | read the design against the stored numbers; recompute; adversarially verify | reimplement from the methodology without seeing the code, then diff; or clone and re-run |
| **lives in** | [`validity/`](validity/) | [`reproducibility/`](reproducibility/) |
| **status** | audited 2026-08-22, see `validity/the validity synthesis in `DIAGNOSIS.md` | one unseal run complete (2026-08-20); clone-audit and post-seamless repro outstanding |

**Neither gate subsumes the other.** Gate 2 found the readout defect — 157 of 551 items scored at a
token that does not occur in the tokenised sequence — which Gate 1 could never have found, because
the design was correct and the implementation was not. Gate 1 found that the S3 membership null was
underpowered and that E37's decision statistic divides by a near-zero denominator, neither of which
any amount of reproducibility checking would surface, because the code faithfully implements a
design with those properties.

---

## WHERE THINGS ARE

### [`context/`](context/) — what is true right now

Read this first. Every number names a results file.

| file | for |
|---|---|
| [`CONTEXT.md`](context/CONTEXT.md) | what was run, what it found, what is open |
| [`RESULTS_TAXONOMY.md`](context/RESULTS_TAXONOMY.md) | every claim, tiered A / B / D(retracted) |
| [`AGGREGATION_POLICY.md`](context/AGGREGATION_POLICY.md) | why `min` is primary. **Read before citing any number** |
| [`CONFIG_MATRIX.md`](context/CONFIG_MATRIX.md) | which configuration produced each cited number |
| [`../experiments/DA1_OUTCOME.md`](experiments/DA1_OUTCOME.md) | the two layer-permutation nulls, adjudicated against their pre-registration |

### [`experiments/`](experiments/) — the experiment record

```
experiments/
  preregs/        one pre-registration per experiment, written BEFORE it ran
  descriptions/   per-experiment digests: what actually happened, in detail
  INDEX.md        one generated row per experiment (tools/build_provenance.py)
  + cross-cutting notes: COVERAGE, DISAGREEMENTS, METRICS, PROVENANCE,
    internal working ledgers (not published)
```

A pre-registration carries exactly: `WHY` · `DESIGN` · `PRIMARY` · `DECISION RULE` (fixed before
running) · `CONTROLS` (named, each with the number it must produce) · `DECLARED BIAS` · `COST`.
Outcomes are three-way — ACCEPT / REJECT / UNCLEAR — and drive control flow. See `CLAUDE.md` §5.

### [`validity/`](validity/) — Gate 1

- **[`DIAGNOSIS.md`](validity/DIAGNOSIS.md) — start here.** The open construct-validity problem:
  task competence, positional support, and what still stands.
- [`CONSTRUCT_VALIDITY.md`](validity/CONSTRUCT_VALIDITY.md) — where the ground truth comes from, and
  the two routes out.
- [`audit-2026-08-22/`](validity/audit-2026-08-22/) — the current audit. `SCORECARD.md` is the
  synthesis; `extract.py` regenerates `DIGEST.json` from `results/` so every derived number can be
  recomputed rather than trusted.
- Earlier validity passes are internal and not published; `DIAGNOSIS.md` is the current synthesis.

### [`reproducibility/`](reproducibility/) — Gate 2

- `REPRODUCIBILITY.md` — how to run it · `COMPUTE.md` — what it costs · `ARTIFACTS.md` — every
  `.pt` with its SHA-256
- [`unseal-2026-08-20/`](reproducibility/unseal-2026-08-20/) — the reimplementation run that found
  the readout defect
- `review/`, `review-snapshot-2026-08-19/`, the reproduction-review protocol, `REVIEWER.md` — external review
- the release procedure — what ships, and how
- The readout-exposure ledger is internal; `tools/readout_exposure.py` regenerates it

### [`archive/`](archive/) — superseded, kept for provenance

Nothing here is current. Do not read it for state. Includes `logs/` (78 run logs),
`prereg/` (original pre-registration texts), the superseded `CLAUDE.md`, and
`repath_2026-08-22.sh` — the exact path rewrite this restructure applied, so any stale link found
later can be traced to a rule rather than guessed at.

### Elsewhere

| path | holds |
|---|---|
| `paper/` | the paper, its argument, and the source material it replies to |
| `paper/fundamentals/` | `RIGOROUS_ANTHROPIC.md` (the source method transcribed) and the source PDF (gitignored) |
| `results/` | every results file; mirrored to Hugging Face, untracked from git |
| `src/`, `experiments/`, `tools/`, `repro/`, `tests/` | code |

---

## TRAPS THAT HAVE COST TIME

Each of these was found the hard way. They are listed here because the next person will otherwise
find them the same way.

1. **A results file's top-level `inputs` field is a hardcoded declaration and does not track CLI
   overrides.** `e54_aggregation_audit_rstrip.json` declares `e36_qladder_410m.json` while its
   `provenance.argv` passes `--e36 …_rstrip.json`. **Read `provenance.argv` and
   `provenance.inputs`.**
2. **`_rstrip` in a filename does not guarantee a corrected readout throughout.**
   `e54_aggregation_audit_rstrip.json` reads a corrected `e36` but a pre-correction `e48`, which is
   why its derangement count is 104/120 where `_rstrip_v2` gives 84/120.
3. **`INDEX.md` truncates stored verdict strings mid-sentence**, which can invert their apparent
   meaning. R9's row reads as though the real operator beats every derangement; the full stored
   string says 0 of 8.
4. **Pooling convention moves headline ratios by ~37%; the N-grid choice moves them by ~0.1%.**
   Sample SD pooled by RMS is the defensible estimator. Name the convention whenever you print a
   "×seed SD" figure.
5. **At 70M, `persist` collapses to `min` by construction** — the band is `[2,3]`, so
   `floor(2/2) = 1` makes "≥1 of 2 layers" identical to "min over layers". All 70M values are
   bit-identical across the two aggregations, so any 70M claim reported as surviving "both
   aggregations" has the weight of one.
