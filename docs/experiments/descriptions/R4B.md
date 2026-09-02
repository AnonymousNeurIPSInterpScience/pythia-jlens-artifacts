# R4b — E36 Q-ladder re-scored at the stripped readout, with per-corpus derangements

## WHAT WAS RUN

`--rstrip` was added to `experiments/t36_qladder.py`, and the `shuf` arm was rebuilt from each
corpus's own operator instead of Pile-CC's. Two full E36 runs were then executed over the same
operators, band and shift rungs: a stripped arm (`results/e36_qladder_410m_rstrip.json`) and an
unstripped arm re-run through the same new code (`results/e36_qladder_410m_c1arm.json`). Slopes of
read AUC against containment were fitted per arm and compared to the identity arm's slope by
`tools/r4b_e36_flatness.py` → `results/r4b_e36_flatness.json`, which also stores the crossing count
and three controls.

Note on the id: `docs/experiments/preregs/R4_corrections.md` records that the slate uses `R4b` twice — for
the four-arm `t22` table (documented in `R4.md`) and for this E36 re-score.

## CONFIGURATION

| field | value |
|---|---|
| model | `EleutherAI/pythia-410m-deduped` |
| checkpoint | final |
| fitting corpus | 5 in-stream Pile components (Pile-CC, StackExchange, Wikipedia_en, Github, USPTO_Backgrounds) |
| N | as in the stored E36 operators |
| seeds | as in the stored E36 operators |
| band | `[9,21]` |
| target layer | inherited from E36 |
| skip_first | inherited from E36 |
| eval sets used | 5 admitted |
| aggregations stored | `persist`, `min` (`-> by_aggregation`) |
| device | CPU, pooled, 39 cells per run |
| readout | two arms: stripped (`e36_qladder_410m_rstrip.json`) and unstripped (`e36_qladder_410m_c1arm.json`) |
| shift rungs | Q0, Pile-CC, StackExchange, Wikipedia_en, USPTO_Backgrounds, Github, CONTROL_PubMed_2023, OOD_Wikipedia_2023, TRAP_FineWeb, OOD_News_2024, OOD_arXiv_2023, OOD_CommonPile, SHUFFLED_Pile-CC (13, from `-> controls.C2_own_derangement_loses.per_rung_per_corpus`) |
| date run | `results/e36_qladder_410m_rstrip.json -> provenance.utc` = 2026-08-20T09:35:30+00:00; `results/e36_qladder_410m_c1arm.json -> provenance.utc` = 2026-08-20T09:45:45+00:00; `results/r4b_e36_flatness.json -> provenance.utc` = 2026-08-20T13:19:10+00:00 |

## PRE-REGISTERED RULE (verbatim)

Stored verbatim at `results/r4b_e36_flatness.json -> decision_rule_verbatim`:

> "REJECT STANDS: the fitted operator is flatter on >= 4 of 5 rungs. Section 3.1 is unchanged. | REJECT OVERTURNED: flatter on <= 1 of 5. STOP and alert. | UNCLEAR: 2 or 3 of 5. Report and stop. Do not re-cut."

Registration source: `POSTREVIEW_EXPERIMENTS.md` §3 Tier B item R4b, via
`docs/experiments/preregs/R4b_e36_rstrip.md`; `prereg_sha256` recorded in-file as
`9d6b5847fa32234d38b8486761dce0552ae4749eb75ec2f2ddaa58abfdbdfe31`. Status `"PRE-REGISTERED"`.

The results file records an ambiguity in the rule at `-> WORDING_AMBIGUITY_IN_THE_RULE`:
> "the rule says '5 rungs'; E36 has eleven-plus rungs and five FIT CORPORA, and the published 5-of-5 is over corpora (CONTEXT.md §3.1 lists five slopes, one per fitting corpus). Adjudicated over the five fit corpora, which is the only reading in which 5 is the denominator. Flagged, not resolved."

## CONTROLS (as recorded)

| control | required value | observed value | fired | source |
|---|---|---|---|---|
| C1_unstripped_is_the_stored_run | "the unstripped arm read here is results/e36_qladder_410m.json as stored" | file sha256 `d0cc7b6381ac515f6d53401705e246d8c20c67884cda25e4b1d35d5312f45153`, `rstrip_flag_in_file` "absent (pre-R4b run)" | true | `-> controls.C1_unstripped_is_the_stored_run` |
| C1b_unstripped_slopes_reproduce_e53 | `max_abs_diff = 0.0` | `max_abs_diff` 0.0, `worst_at` null | true | `-> controls.C1b_unstripped_slopes_reproduce_e53` |
| C2_own_derangement_loses | "0 cells where the derangement is at or above its own operator" | `n_cells` 65, `n_cells_where_derangement_wins` 0 | true | `-> controls.C2_own_derangement_loses` |
| C3_report_C2_shuf_below_everything | recorder, no gate | unstripped false, stripped false | reported | `-> controls.C3_report_C2_shuf_below_everything` |

C3's stored note:
> "a RECORDER, not a gate -- its power is nil because it compares Pile-CC's derangement to every corpus's operator (R4d/F-7). Reported whichever way it fires, as the rule requires."

C2's stored cross-reference at `-> controls.C2_own_derangement_loses.like_for_like_evidence_already_on_disk`:
> "e33_logit_baseline_410m_v2: Github 0.027926 vs its own jshuf 0.018142; e54 C2: 0 of 15 and 0 of 120"

### The spec's C1 and the results file's C1 are different checks — both recorded, neither adjudicated
`docs/experiments/preregs/R4b_e36_rstrip.md` §RESULT states:
> | **C1** | the unstripped arm re-run through the new code reproduces the stored E36 **bit-exactly** |
> **`max_abs_diff = 0.0`** over 364 values (13 rungs × 7 shared arms × 2 aggregations × 2 statistics) |

`results/r4b_e36_flatness.json -> controls.C1_unstripped_is_the_stored_run` instead records a
file-identity check (path, sha256, `rstrip` flag). "`max_abs_diff = 0.0` over 364 values" appears as
a key in none of `results/r4b_e36_flatness.json`, `results/e36_qladder_410m_c1arm.json`,
`results/e36_qladder_410m.json`. It is **UNSTORED**.

### Controls stored inside the E36 measurement files themselves
| control | required value | stripped run | unstripped `c1arm` run | stored 2026-08-15 run | source |
|---|---|---|---|---|---|
| `C1_Q0_reproduces_e33` | `logit_e33 = 0.02843950316309929`, tolerance `1e-06` | `logit_here = 0.08317913140635938`, `abs_diff = 0.05473962824326009`, **`fires = false`** | `logit_here = 0.028439503419212996`, `abs_diff = 2.561137066314778e-10`, `fires = true` | `abs_diff = 2.561137066314778e-10`, `fires = true` | `results/e36_qladder_410m{_rstrip,_c1arm,}.json -> controls.C1_Q0_reproduces_e33` |
| `C2_shuf_below_everything` | (Pile-CC-sourced derangement, all rungs) | `fires = false`, `false` on all 13 rungs | `fires = false` | `fires = false` | `-> controls.C2_shuf_below_everything` |
| `C2b_own_derangement_below_its_own_operator` | "0 cells where the derangement is at or above its own operator" | `n_cells = 65`, `n_cells_where_derangement_wins = 0`, `fires = true` | present, same form | key absent (pre-R4b file) | `-> controls.C2b_own_derangement_below_its_own_operator` |
| `C3_capability` | registered form: top-k accuracy < 50% of Q0 | stored under key `REGISTERED_FORM_IS_DEGENERATE`; `q0` top-10 rate `0.06942889137737962` | same structure | `q0` top-10 rate `0.020156774916013438` | `-> controls.C3_capability` |
| `C5_self_distance` | `0.0` | `value = 0.0`, `fires = true` | same | same | `-> controls.C5_self_distance` |
| `C4` (prefix-length control) | see the PREREG quote below | **no C4 key in any E36 results file** | same | same | `-> PREREG_C4_WAS_NEVER_CODED` |

The stripped and `c1arm` files carry, verbatim at `-> PREREG_C4_WAS_NEVER_CODED`:
> "PREREG_E36_QLADDER.md:133 registers C4, the prefix-length control -- a prefix of the same 128
> tokens drawn from the FITTING corpus P -- and :136 calls it load-bearing: 'Without C4 the entire
> ladder could be a prefix-length artifact.' No C4 key exists in any E36 results file. Surfaced by
> R4e; recorded here so the omission travels with the result rather than being rediscovered."

The registering text, quoted from `docs/experiments/preregs/superseded/PREREG_E36_QLADDER.md` §CONTROLS:
> | **C4** | **prefix-length control** — a prefix of the same 128 tokens drawn from the *fitting
> corpus P* | isolates "any prefix at all" from "a prefix from Q". Any degradation present at C4 is
> subtracted before the crossing is located. |
>
> **C4 is load-bearing.** Adding 128 tokens of *anything* changes the readout position's activation.
> Without C4 the entire ladder could be a prefix-length artifact.

`C3_capability`'s stored text in both new runs: "the pre-registered rule is top-k accuracy < 50% of
Q0, but Q0's top-10 rate is 0.06942889137737962. The eval targets are latent concepts, not emitted
tokens, so this quantity is floored at Q0 and the rule cannot fire. NOT reinterpreted silently —
adjudication below uses mean rank, which is not floored, and the operator is asked to confirm."
`results/r4e_control_power.json -> powerless_controls` classes `e36_qladder_410m` C3 as `NO_GATE`,
and `-> declared_in_a_spec_but_absent_from_the_results_file` lists
`docs/experiments/preregs/superseded/PREREG_E36_QLADDER.md` C4 against `results/e36_qladder_410m.json`.

## NUMBERS (as stored)

PRIMARY (the slope secondary) — `-> PRIMARY_slope_secondary`, count of fitting corpora whose
fitted operator is flatter than the identity arm:

| aggregation | unstripped | stripped |
|---|---|---|
| persist | 5 | 1 |
| min | 1 | 3 |

E36's own primary — `-> E36_PRIMARY_reject_s3_crossings.n_crossing`: unstripped **0**, stripped
**0**; `crosses_per_corpus` is `false` for all five corpora on both arms.
`-> E36_PRIMARY_reject_s3_crossings.published` = "0 of 5 -> REJECT S3".
`n_corpora = 5` in every flatness cell (`-> by_aggregation.<agg>.<readout>.flatness.n_corpora`).

The `crossings` blocks inside the E36 measurement files differ between readouts.
`results/e36_qladder_410m_rstrip.json -> crossings`: every corpus records
`rungs_ci_strictly_below_logit = []` **and** `rungs_point_below_logit = []`.
`results/e36_qladder_410m.json -> crossings` and `results/e36_qladder_410m_c1arm.json -> crossings`:
`Github` records `rungs_point_below_logit = ["Pile-CC","StackExchange","Wikipedia_en",
"USPTO_Backgrounds","CONTROL_PubMed_2023","OOD_Wikipedia_2023","TRAP_FineWeb","OOD_News_2024",
"OOD_arXiv_2023","OOD_CommonPile"]`, with `rungs_ci_strictly_below_logit = []` and `crosses = false`;
the other four corpora record both lists empty.

Only `persist` and `min` appear under `results/r4b_e36_flatness.json -> by_aggregation`; `best1L`
and `mean` are **not stored** for this experiment.

The slope fit uses 11 of the 13 rungs — `Q0` and `SHUFFLED_Pile-CC` are absent from
`-> by_aggregation.<agg>.rungs`, which lists `["Pile-CC","StackExchange","Wikipedia_en",
"USPTO_Backgrounds","Github","CONTROL_PubMed_2023","OOD_Wikipedia_2023","TRAP_FineWeb",
"OOD_News_2024","OOD_arXiv_2023","OOD_CommonPile"]`. The x-axis is measured containment:
`results/e36_qladder_410m_rstrip.json -> x_axis = {"source": "e48b_exposure_growth.json", "k": 32,
"coverage_shards": 20, "note": "measured containment, not a rung number — this is what E35/M1 was for"}`.

Slopes, `persist` (`-> by_aggregation.persist.<arm>.slopes`):

| arm | unstripped slope_linear | unstripped slope_log10 | unstripped level_mean | stripped slope_linear | stripped slope_log10 | stripped level_mean |
|---|---|---|---|---|---|---|
| logit | −0.003720899629554985 | −0.0005296062512720639 | 0.031253348919562995 | −0.0037284686515085515 | −5.887753023934386e-05 | 0.07387016320417664 |
| J\|Pile-CC | −0.0011535435993366304 | 0.00013903741903277346 | 0.048237061704452515 | −0.0048877667147762035 | −0.0006226113130866229 | 0.10972794660465848 |
| J\|StackExchange | 0.0002240976423130857 | 0.00043591080099479097 | 0.04689619833829276 | −0.0036664399547795337 | −3.4500043359785385e-05 | 0.11132412637922574 |
| J\|Wikipedia_en | −0.0009354827291769033 | 0.0002869798762885821 | 0.03964683171079466 | −0.003751919371114022 | −0.00021278116380330815 | 0.1012678702841654 |
| J\|Github | −0.0014550596366756292 | −1.0939804428585397e-06 | 0.026799873580845693 | −0.006302421723152167 | −0.000664630277617062 | 0.09380335117845486 |
| J\|USPTO_Backgrounds | −0.000885040945743085 | 0.0002350879878021014 | 0.04893984228838235 | −0.004609631566017049 | −0.00033176743326874674 | 0.11131394055714323 |

`persist` per-corpus `flatter_than_logit` — unstripped: all five true. Stripped: StackExchange
true; Pile-CC, Wikipedia_en, Github, USPTO_Backgrounds false.

Slopes, `min` (`-> by_aggregation.min.<arm>.slopes`):

| arm | unstripped slope_linear | unstripped slope_log10 | unstripped level_mean | stripped slope_linear | stripped slope_log10 | stripped level_mean |
|---|---|---|---|---|---|---|
| logit | −0.0010699323813758122 | 0.0009282505416221876 | 0.12512224580753933 | 0.0021193657987416048 | 0.00200160581744663 | 0.1990344447394212 |
| J\|Pile-CC | −0.004014659692830054 | −0.00033377197965353794 | 0.14627732089861775 | −0.001888259914374931 | 0.0010884349751549287 | 0.2590676621449265 |
| J\|StackExchange | −0.0035589979944138225 | −0.00028825210444100556 | 0.148409635133364 | 0.00014062084151355905 | 0.0018836771421971345 | 0.2655780210075053 |
| J\|Wikipedia_en | −0.004603344091202535 | −0.000310141050106287 | 0.13278038299670725 | −0.0017812876146258553 | 0.0009754492560197017 | 0.23547124836706754 |
| J\|Github | −0.0035714446311215964 | −0.00029611400443362804 | 0.11407741325145418 | 0.00374649675334448 | 0.0028687089581596976 | 0.2293679065437931 |
| J\|USPTO_Backgrounds | −4.247891056195874e-05 | 0.0010588073618437773 | 0.160057055566347 | 0.0028406448262535064 | 0.002393799693174576 | 0.2592142216635473 |

`min` per-corpus `flatter_than_logit` — unstripped: USPTO_Backgrounds true, other four false.
Stripped: Pile-CC, StackExchange, Wikipedia_en true; Github, USPTO_Backgrounds false.

## VERDICT STRING AS STORED

> "REJECT OVERTURNED — flatter on only 1 of 5. FLAGGED FOR ADJUDICATION. S3's first half is no longer rejected on the read axis, which reopens the subthesis and changes the paper's framing."

Stored in: `results/r4b_e36_flatness.json -> VERDICT`

The adjudicating file also stores the verdict strings emitted by the two E36 runs themselves,
identical to each other (`-> stripped_verdict_string_from_the_run` and
`-> unstripped_verdict_string_from_the_run`):

> "REJECT S3 (the publishable null) — no fitting corpus produces a J-lens curve that goes strictly below the logit lens at any measured rung. Shift in the READ distribution degrades both lenses together; the averaged derivative is no more shift-fragile than not fitting at all."

## PROVENANCE

| item | value |
|---|---|
| spec | `docs/experiments/preregs/R4b_e36_rstrip.md` |
| prereg | `POSTREVIEW_EXPERIMENTS.md` §3 Tier B item R4b; `prereg_sha256` `9d6b5847fa32234d38b8486761dce0552ae4749eb75ec2f2ddaa58abfdbdfe31` |
| scripts | `experiments/t36_qladder.py` (both arms), `tools/r4b_e36_flatness.py` (adjudication) |
| results | `results/r4b_e36_flatness.json` sha256 `5d90536ab71999f8`, added in `8614ccd`, utc 2026-08-20T13:19:10+00:00, payload_sha256 `canonical` |
| | `results/e36_qladder_410m_rstrip.json` sha256 `6a40a858516cc564`, added in `8614ccd`, utc 2026-08-20T09:35:30+00:00, payload_sha256 `canonical` |
| | `results/e36_qladder_410m_c1arm.json` sha256 `ab06a3488c667676`, added in `8614ccd`, utc 2026-08-20T09:45:45+00:00, payload_sha256 `canonical` |
| | `results/e36_qladder_410m.json` (the stored unstripped run C1 checks) sha256 `d0cc7b6381ac515f`, added in `1340e76` |

## RELATED / SUPERSEDED FILES

- `results/e36_qladder_410m.json` — the published E36 run; documented at `E36.md`.
- `results/e53_ladder_summary.json -> e36_containment_slopes` — the stored slopes C1b reproduces at `max_abs_diff = 0.0`.
- `results/e33_logit_baseline_410m_v2.json`, `results/e54_aggregation_audit.json` — the like-for-like derangement evidence C2 cites.
- `docs/experiments/preregs/R1_grid_rstrip.md` / `R1.md` — the same readout correction applied to the 8×8 grid.
- `docs/experiments/preregs/R2_e48_rstrip_arm.md` / `R2.md` — the same correction applied to E48.
