# E66c — do operators fitted UNDER RECORDED PROVENANCE reproduce?

**PRE-REGISTERED 2026-08-29, before any E66c code was written or run.**

## WHY

E66b established that the E48 panel does not reproduce: a refit differs from the stored operator by
`1.1e-3` max element, against a within-box floor of `1.03e-6`. The archaeology explains why the
cause is untraceable rather than what it is. Those lenses were fitted at **20:03–20:04 PDT on
2026-08-14**, and commit `1a6f5df` at **20:05** is the one that added `provenance.py`. They are the
last artifacts made before the provenance layer existed, which is why their sidecars record
`argv: None, env: null`. The producing box (`47732872`) was destroyed and its environment was never
written down.

The question that matters for the release is therefore **not** "why does that panel differ" but
**"does anything fitted with provenance reproduce?"** If yes, the reproducibility claim is narrow
and strong. If no, the harness has a real problem and the paper must say so.

## DESIGN

Target `results/r6/lens_R6_Pile-CC_b0_410m_n200.pt`, fitted 2026-08-20 and recorded in
`results/r6_within_source_410m.json` with a full environment block: `torch 2.11.0+cu128`,
`python 3.11.9`, TF32 off on all three flags, and both corpus inputs SHA-256 hashed.

Refit it exactly: `fast_fit`, `source_layers=[9..21]`, `target_layer=-2`, `max_seq_len=128`,
`skip_first=16`, pool `disjoint_blocks("Pile-CC", tok, 4, 200, 128)[0]` — deterministic, file order,
no RNG. Match `torch 2.11.0+cu128` and TF32 off. `dim_batch` is set to fit the card and is recorded.

## PRIMARY

    S6 = max_abs( fp16(refit) , stored )

## DECISION RULE — fixed before running

* **REPRODUCES** if `S6 == 0`. Operators fitted under recorded provenance are bit-identical at their
  storage dtype. The E48 panel is then a dated exception with a known cause, and the paper says so
  in one appendix paragraph.
* **REPRODUCES TO STORAGE** if `0 < S6 <= 9.766e-04` (one fp16 ULP in the `[1,2)` binade). The
  operator agrees to within a single representable step of its own storage format.
* **DOES NOT REPRODUCE** if `S6 > 9.766e-04`, i.e. the same scale as the E48 panel's `1.1e-3`. Then
  the problem is not the missing provenance and the harness does not reproduce operators at all.

## CONTROLS

| id | control | must produce |
|---|---|---|
| **C1** | TF32 off, asserted on the observed flags | both `False` |
| **C2** | same pool: the 200 prompts are the recorded block | `n == 200`, blocks disjoint, first/last prompt SHA-256 recorded |
| **C3** | torch matches the recorded environment | `torch.__version__ == "2.11.0+cu128"`; a mismatch is reported, not silently accepted |
| **C4** | the comparison can fail: a different block's operator must exceed the threshold | `max_abs(fp16(refit_b0), stored_b1) > 9.766e-04` |

## DECLARED BIAS

We want REPRODUCES, because it makes the release defensible. C4 is the guard: it proves the
comparison can return the adverse answer on data where it should.

## COST

One fit, ~5 min of card time, well under $0.50. **OUTPUT:** `results/e66c_post_provenance_repro.json`.
