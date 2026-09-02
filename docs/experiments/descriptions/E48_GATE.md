# E48_GATE — per-token cross-entropy of pythia-410m-deduped on every E48 fitting corpus

## WHAT WAS RUN

For each of eleven candidate corpora, 200 documents per seed block × 3 seed blocks were tokenised
to `max_seq_len = 128` with `skip_first = 16`, and per-token cross-entropy was computed from the HF
causal-LM head's logits directly. Cross-entropy in nats, perplexity, and bits-per-byte were stored
per (corpus, seed) cell and averaged per corpus. A gate threshold was formed from the in-stream
corpora's own spread and applied to the three OOD rungs, in both CE and bits-per-byte units. Three
controls were evaluated. The run is CPU.

## CONFIGURATION

| field | value | source |
|---|---|---|
| model | `EleutherAI/pythia-410m-deduped` | `results/e48_competence_gate_410m.json -> model` |
| corpora scored | 11 (5 in-stream, 3 OOD, 3 reported-only) | `-> by_corpus` (keys, with `tier`) |
| N per cell | 200 documents; `n_max = 200` | `-> window.n_max`, `-> cells.*.n_docs` |
| seeds | 3 cells per corpus (`s0`, `s1`, `s2`) | `-> cells` (keys) |
| window | `max_seq_len = 128`, `skip_first = 16` | `-> window` |
| scored tokens per cell | 22200 | `-> cells.*.n_scored_tokens` |
| band / target layer / aggregation | not applicable — this is a cross-entropy measurement, no lens is used | — |
| vocab | 50304 | `-> vocab` |
| device | `cpu` | `-> device` |
| readout | not applicable | — |
| date run | `2026-08-15T19:36:44+00:00` | `-> provenance.utc` |

Two implementation notes stored in the file:

> `why_trainval_ce_is_nan`: "trainval.py:89-102 wraps the CE block in `except Exception: pass` and
> calls `model.forward(...).logits`; jlens' HFLensModel.forward returns the TEXT MODULE output
> (hidden states), which has no `.logits`. Every fit in the programme therefore recorded
> model_cross_entropy = NaN."

> `second_bug_avoided_here`: "the obvious fix — unembed the returned last_hidden_state — is ALSO
> wrong: gpt_neox's last_hidden_state is already final-layer-normed and jlens' unembed norms again.
> Measured 6.32 vs the true 3.02 nats at 70M on one English sentence. This script takes logits from
> the HF head directly. The lens read path is NOT affected: it hooks the transformer blocks, which
> are pre-norm."

## PRE-REGISTERED RULE (verbatim)

> "PASS iff CE(Q) <= max_instream_CE + (max_instream_CE - min_instream_CE); one in-stream spread
> above the WORST in-stream corpus"

Stored in: `results/e48_competence_gate_410m.json -> gate_rule`

Status field as stored: `"PRE-REGISTERED (rule in this file's docstring, fixed before the run)"`
(`-> status`).

Registration source: the script docstring, `experiments/t48_competence_gate.py`. No separate file in
`docs/archive/prereg/` — NONE LOCATABLE.

## CONTROLS (as recorded)

| control | required value | observed value | fired | source |
|---|---|---|---|---|
| C1 Pile-CC is lowest or second-lowest in-stream CE | (implied by name) | `pilecc_ce = 2.9277699113536526`; in-stream CEs: Wikipedia_en 2.417942479737886, USPTO_Backgrounds 2.4130186644545546, Pile-CC 2.9277699113536526, StackExchange 2.412771591553101, Github 1.0569410117018083 | **`false`** | `-> controls.C1_pilecc_lowest_or_second` |
| C2 max corpus CE far below the uniform null | (implied by name) | `max_corpus_ce = 3.1196862617698873` vs `uniform_null = 10.825839875788878` | `true` | `-> controls.C2_far_below_uniform_null` |
| C3 seed spread small vs between-corpus spread | (implied by name) | `max_seed_sd = 0.09170711447132196`, `between_corpus_spread = 2.0627452500680787`, `ratio = 0.04445876894798098` | `true` | `-> controls.C3_seed_spread_small` |

## NUMBERS (as stored)

### Gate

| field | value | source |
|---|---|---|
| in-stream CE min / max | 1.0569410117018083 (`Github`) / 2.9277699113536526 (`Pile-CC`) | `-> instream_ce_range` |
| `gate_threshold_ce` | 4.798598811005497 | `-> gate_threshold_ce` |
| `gate_threshold_bpb` | 1.4569084494647655 | `-> gate_threshold_bpb` |
| `uniform_null_ce_nats` | 10.825839875788878 | `-> uniform_null_ce_nats` |
| `ce_bpb_agree` | `true` | `-> ce_bpb_agree` |

| OOD rung | CE (nats) | passes | margin (nats) | bpb | bpb passes |
|---|---|---|---|---|---|
| OOD_News_2024 | 2.783762465468398 | true | 2.014836345537099 | 0.8756680733425302 | true |
| OOD_arXiv_2023 | 2.826561468313406 | true | 1.9720373426920905 | 0.798696141703478 | true |
| OOD_CommonPile | 2.8191345634116782 | true | 1.9794642475938184 | 0.8807126396820312 | true |

Source: `-> gate.<rung>` and `-> gate_bpb.<rung>`.

### Per-corpus means (all 11 corpora)

| corpus | tier | `ce_mean` | `ce_seed_sd` | `ppl_mean` | `bpb_mean` | `bpb_seed_sd` |
|---|---|---|---|---|---|---|
| Wikipedia_en | in-stream | 2.417942479737886 | 0.0232926528101091 | 11.222744519211309 | 0.8121184470645005 | 0.005236773636815465 |
| USPTO_Backgrounds | in-stream | 2.4130186644545546 | 0.054127624845418616 | 11.16762161708706 | 0.6775096843275838 | 0.011680430266261903 |
| Pile-CC | in-stream | 2.9277699113536526 | 0.03656177083296005 | 18.685912754949932 | 0.9546970748516366 | 0.007193550844342514 |
| StackExchange | in-stream | 2.412771591553101 | 0.07272251956827648 | 11.164862741247552 | 0.9206028992960367 | 0.026152248732911738 |
| Github | in-stream | 1.0569410117018083 | 0.09170711447132196 | 2.8775551049055843 | 0.45248570023850765 | 0.040704981108051354 |
| OOD_News_2024 | OOD | 2.783762465468398 | 0.03522452314072801 | 16.179782442562345 | 0.8756680733425302 | 0.0009034577530182717 |
| OOD_arXiv_2023 | OOD | 2.826561468313406 | 0.03148906381317604 | 16.88729338315894 | 0.798696141703478 | 0.0013583006182701558 |
| OOD_CommonPile | OOD | 2.8191345634116782 | 0.046661259745184334 | 16.762337652686245 | 0.8807126396820312 | 0.003708888510801891 |
| OOD_Wikipedia_2023 | reported-only | 2.49233609070649 | 0.03485038634801425 | 12.089485292261289 | 0.7879277209197116 | 0.01042887917511243 |
| TRAP_FineWeb | reported-only | 3.1196862617698873 | 0.06061050721159486 | 22.639275722551698 | 1.0048503261861432 | 0.02189266746899364 |
| CONTROL_PubMed_2023 | reported-only | 2.687820088383672 | 0.04827420022524149 | 14.699597146125235 | 0.8287388738196896 | 0.01876520279719823 |

Source: `-> by_corpus.<corpus>`.

Per-(corpus, seed) cells are stored at `-> cells."<corpus>|s<seed>"` with fields `ce_nats`, `ppl`,
`bits_per_byte`, `n_docs`, `n_scored_tokens`, `n_scored_bytes`, `ce_per_doc_sd`. Example:
`Wikipedia_en|s0` = `ce_nats 2.4447590004860817`, `ppl 11.527771076814997`,
`bits_per_byte 0.8125306169496754`, `n_scored_bytes 96366`, `ce_per_doc_sd 0.5687033895724258`.

### The copy of this block embedded in E48

`results/e48_crossover_410m.json -> competence_gate` stores `file`, `verdict`, `failed_rungs = []`,
and `ce_by_corpus` with the same eleven means listed above.

## VERDICT STRING AS STORED

> "ALL THREE OOD RUNGS PASS — a negative read on them cannot be attributed to model incompetence;
> E48 may be scored as registered"

Stored in: `results/e48_competence_gate_410m.json -> VERDICT`

The same string is stored in `results/e48_crossover_410m.json -> competence_gate.verdict` and in
`results/e48_crossover_410m_rstrip.json -> competence_gate.verdict`.

## PROVENANCE

| | |
|---|---|
| results file | `results/e48_competence_gate_410m.json` |
| results sha256 (recomputed now) | `37b9f18a33810e338c921e998af5eaab98164cc9b0a9848fc549a1d0b0950f8a` |
| sha256 as recorded by E48 | `37b9f18a33810e338c921e998af5eaab98164cc9b0a9848fc549a1d0b0950f8a` (`results/e48_crossover_410m.json -> provenance.inputs[0]`) |
| script | `experiments/t48_competence_gate.py`, `sha256 e38436fa6b1fba57151d4531310999847db0e99a5aa7a20d58777b0cf60fb414`, 305 lines |
| argv | `["pythia/t48_competence_gate.py", "--model", "410m", "--device", "cpu"]` |
| git commit at run | `f7d6908652e33d5927c2a65d605a320e26360da0` (`describe: f7d6908-dirty`) |
| hashed inputs | 11 — `corpora/{Wikipedia_en, USPTO_Backgrounds, Pile-CC, StackExchange, Github, OOD_News_2024, OOD_arXiv_2023, OOD_CommonPile, OOD_Wikipedia_2023, TRAP_FineWeb, CONTROL_PubMed_2023}.jsonl` |
| commit that added the file to git | `1340e76` "Restructure the repo into src/ + experiments/; retire docs/ and FUTURE_IMPROVEMENTS" |
| runner module cited by the E48 spec | `bash repro/exp/e48_gate.sh` |

Spec: this gate is described inside `docs/experiments/preregs/E48_crossover.md` under "Prerequisite gate".
It has no separate document in `docs/experiments/preregs/`.

## RELATED / SUPERSEDED FILES

- `results/e48_crossover_410m.json` and `results/e48_crossover_410m_rstrip.json` — both consume this
  file as hashed input `[0]` and embed its verdict and `ce_by_corpus`; documented in `E48.md`.
- `docs/experiments/preregs/E48_crossover.md` — restates the gate as `"per-token cross-entropy 2.78–2.83
  nats on the three OOD corpora, inside the in-stream range of 1.06–2.93"`, which matches the
  `by_corpus` means above.
- `docs/experiments/preregs/superseded/PREREG_E48_CROSSOVER.md` — the E48 pre-registration; it does not mention this
  gate.

### Recorded observation

Control `C1_pilecc_lowest_or_second` is stored with `fires: false`. The file's `VERDICT` string does
not reference it, and no `controls_fire` aggregate field is stored in this file.
