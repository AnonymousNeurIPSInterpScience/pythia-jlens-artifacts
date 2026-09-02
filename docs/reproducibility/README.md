# GATE 2 — IS THE IMPLEMENTATION TRUE?

**The question:** does the code faithfully implement the methodology as written, and do the stored
results actually regenerate?

This is *not* the question of whether the experiments are well-designed. That is
[Gate 1](../validity/), and the two fail in different ways. Gate 2 found the readout defect — 157 of
551 eval items scored at a token that does not occur in the tokenised sequence — which no amount of
design scrutiny would have surfaced, because the design was right and the code was not. Gate 1 found
that S3's membership null is underpowered and that E37's decision statistic divides by a near-zero
denominator, which no amount of reproducibility checking would surface, because the code faithfully
implements a design with those properties.

---

## THE THREE WORKFLOWS

They answer different questions. Do not merge them.

### (a) UNSEAL — blind reimplementation from the methodology

**Question:** *is the methodology, as written, sufficient to determine the implementation?*

An agent receives the Methods section and the pre-registration `DESIGN`/`PRIMARY` fields — and
nothing else from this repository. It builds the pipeline from scratch under
`/tmp/readonly-lens-reimplementation` (the system temp directory, **not** a path inside this repo),
with no read access here. When it is finished it runs a single scripted command that clones this
repository and diffs.

**This is the highest-value workflow and the only one that degrades irreversibly once contaminated.**
It is what caught the readout defect: the source's own data spec says the readout is "the token
immediately preceding `target`", our scorer read the last token of the unstripped prompt, and only
an implementer working from the spec rather than from the code would notice the difference.

Prior run: [`unseal-2026-08-20/`](unseal-2026-08-20/).

### (b) CLONE-AUDIT — reproducibility review of the released artifact

**Question:** *can a reviewer who clones what we release audit it?*

This is the likeliest reviewer behaviour, and it tests something (a) does not: whether the released
tree is navigable, whether every number has a script behind it, whether the provenance stamps
resolve. It assumes the code is what it is and asks whether an outsider can check it.

Prior runs: `review/`, `review-snapshot-2026-08-19/`, the reproduction-review protocol.

### (c) SEAMLESS REPRO — does every JSON regenerate?

**Question:** *is the pipeline deterministic end-to-end from a clean checkout?*

Runs last, after (a) and (b) have stopped changing the code. Full operator fitting need not be
recommissioned; a smaller scale (70M/160M, or a single 410M cell) exercises the same script paths at
a fraction of the cost. What matters is that **every stored JSON is reachable by a command**, not
that every GPU-hour is re-spent.

---

## THE UNSEAL PROTOCOL — three agents, and agent 2 is kept ignorant

**The protection is ignorance, not detection.** Earlier drafts of this document proposed hashing the
reimplementation so that post-hoc edits toward our numbers would be *detectable*. That is weaker than
it looks, and it has a fatal flaw: committing the manifest into this repository hands the implementer
the git history, which is exactly what the blind is meant to withhold.

The right design removes the incentive rather than policing it. **Agent 2's task ends when the
reimplementation is finished. It is never told that a diff will follow.** There is nothing to hack
toward, because as far as agent 2 is concerned the deliverable is a working implementation and
nothing is going to be compared against it.

| | lives in | does | knows about the diff |
|---|---|---|---|
| **Agent 1** | `~/mechinterp-research` | prepares the brief: Methods, prereg `DESIGN`/`PRIMARY`, the source data spec, eval JSONs, config table. Its final act is writing that brief out to the sandbox | yes |
| **Agent 2** | `/tmp/<sandbox>` | reimplements the pipeline from the brief. Has no read path into this repository. Finishes when it works | **no** |
| *(operator)* | — | SHA-256s agent 2's tree, `cat`s the manifest to `/tmp/saved-repro` — **not** into this repo — then `mv`s the tree to `~/mechinterp-research/unseal_<date>/` | — |
| **Agent 3** | `~/mechinterp-research` | diffs our implementation against agent 2's, and against agent 2's own stated objections | yes |

**The manifest still has a job, just a different one.** It is not evidence against agent 2, who never
knew to cheat. It is provenance for *us*: it fixes what agent 2 produced at the moment the sandbox
closed, so nobody can later wonder whether the tree drifted between the `mv` and the diff. Writing it
to `/tmp/saved-repro` keeps it out of the repository and therefore out of the blind.

```
# operator, after agent 2 reports done and before the mv
mkdir -p /tmp/saved-repro
cd /tmp/<sandbox> && find . -type f -not -path './.git/*' -exec shasum -a 256 {} \; \
  | sort -k2 > /tmp/saved-repro/manifest_<date>.sha256
mv /tmp/<sandbox> ~/mechinterp-research/unseal_<date>
```

**Agent 3 is a separate agent for the same reason.** Whoever wrote the brief has a stake in the
implementation matching it. An agent that sees only two trees and a set of objections has none.

---

## SEQUENCING

**No stratification.** An earlier draft here proposed splitting the unseal into five pipeline layers
so that a divergence in one would not contaminate the rest. That was over-engineering, and it also
mistook the subtheses for strata — S1 is a supporting act, S2 is a confirmation, S3 is the research;
they are not stages of a pipeline.

**The methodology section is the thing under test.** Write it as well as it can be written, hand it
over, and see what comes back. If an independent implementation lands on the same readout token, the
methodology is complete. If it does not, the methodology is underspecified — and *that is the
finding*, about the write-up rather than about the implementer. Subdividing the test would only tell
us which paragraph was thin, which the diff tells us anyway.

**Order: (a) unseal, then (b) clone-audit, then (c) seamless repro.** Only (a) loses value
permanently on contamination; (b) and (c) start from our code and are immune. So (a) goes first and
gets the cleanest conditions available.

**(a) is gated on the Methods section existing in final form.** There is nothing to hand agent 2
until then. That gate, not compute, is what sets the schedule.

## WHAT CONTEXT TO GIVE THE REIMPLEMENTER

**Give:**
- the paper's Methods section, in full
- the `DESIGN` and `PRIMARY` fields of the relevant pre-registrations (`../experiments/preregs/`)
- the source's released data spec, `jacobian-lens/data/evaluations/README.md` — this is ground truth
  for the readout position and the pass@k definition, and it is *upstream of us*
- the released eval JSONs themselves (inputs, not implementations)
- the configuration table: Pythia deduped sizes, the normalised `L38`–`L92` band rule intersected
  with layers strictly below the penultimate target, `N`, fp32, TF32 forbidden, no centering
- the corpus manifest: which sources, how many windows, exactly 128 tokens

**Withhold:**
- every line of our code
- every results file and every number, including in prose
- the aggregation definitions beyond what the source states — `min` is the source's published rule
  and may be given; `persist`, `best1L` and `mean` are ours and must not be

**The test that matters:** hand them only what the *source* published plus our stated design, and see
whether they land on the same readout token. If the methodology is complete, they will. If they do
not, the methodology is underspecified — and that is a finding about the write-up, not a failure of
the implementer.

---

## FILES HERE

| path | holds |
|---|---|
| `REPRODUCIBILITY.md` | how to run it |
| `COMPUTE.md` | what it costs |
| `ARTIFACTS.md` | every `.pt` with its SHA-256 |
| the release procedure | what ships and how |
| `unseal-2026-08-20/` | the completed blind reimplementation (workflow a) |
| `audit-prompts/` | the three independent audit prompts (A1 reimplementation, A2 controls and provenance, A3 completeness). Moved here from `repro/audit/` — they are documents, not executables |
| `review/`, `review-snapshot-2026-08-19/` | external review (workflow b) |
| the reproduction-review protocol, `REVIEWER.md` | blinded review protocols |

| `tools/readout_exposure.py` | regenerates which results files sit on which readout |

## STATUS

| workflow | state |
|---|---|
| (a) unseal, full-blind | **done once**, 2026-08-20 — found the readout defect. Next run uses the three-agent protocol above; gated on the Methods section |
| (b) clone-audit | partial — external reviews done; the three audits in `docs/reproducibility/audit-prompts/` have not been run |
| (c) seamless repro | not started; blocked on repro scaffolding (the release procedure) |
