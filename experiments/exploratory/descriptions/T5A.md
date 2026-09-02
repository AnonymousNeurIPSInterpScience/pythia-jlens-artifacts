# T5A — capability screen gating the write battery

## WHAT WAS RUN
`experiments/t5a_capability_screen.py` was run once per model on `EleutherAI/pythia-{70m,160m,410m,1b}-deduped`, CPU/float32. For each (country, relation) pair in a fixed fact table of 20 countries × 4 relations (capital, language, continent, currency) it runs one clean forward pass on a fixed template prompt and records the model's top-1 token, the rank of the correct answer among the full vocabulary, and the answer's log-probability. A cell counts as `cell_ok` when the correct answer is rank 0. The script compares the number of `cell_ok` cells against a bar fixed in the source (`MIN_CELLS_FOR_WRITE_BATTERY = 32`, `experiments/t5a_capability_screen.py:36`) and writes `WRITE_BATTERY_ADMISSIBLE`. Later versions of the script additionally exclude cells whose answer is multi-token and cells whose answer is copyable from the prompt.

## CONFIGURATION

| field | value |
|---|---|
| model / checkpoint | `EleutherAI/pythia-{70m,160m,410m,1b}-deduped` — `results/t5a_capability_*.json -> model` |
| lens consumed | none. T5A uses no lens and no Jacobian. No `lens` key exists in these files. |
| fitting corpus | not applicable |
| N | not applicable. The unit is a cell: `n_cells` = 61 or 80 depending on file (see NUMBERS). |
| seeds | no seed key is stored; the fact table and templates are fixed in the script |
| band | not applicable |
| target layer | not applicable — final-layer logits only |
| skip_first | not applicable |
| eval sets used | none of the anchor's six. `RELATIONS` and `FACTS` in `experiments/t5a_capability_screen.py`. |
| aggregations stored | none of min / persist / best1L / mean. Counts, an accuracy, and a median rank. |
| device | `cpu` — `-> env.device` |
| dtype | `float32` — `-> env.dtype` |
| readout (stripped or unstripped) | not applicable — templates carry no trailing space and no `.rstrip()` step is in this code path |
| date run | not stored. Git first-add for all seven files: `384565b`, 2026-08-09. Later modifications listed under PROVENANCE. |
| env | `transformers` 5.14.1, `torch` 2.13.0 — `-> env.transformers`, `-> env.torch` |
| bar | `bar_min_cells` = 32 in all seven files — `-> bar_min_cells` |

## PRE-REGISTERED RULE (verbatim)

There is no `prereg` key in any T5A results file. The bar is declared in the script docstring:

> "Pre-declared bar (fixed here, before any model is run):
>   CELL_OK   = correct answer is top-1 among the full vocabulary
>   MODEL_OK  = >= 32 cells pass CELL_OK  ->  write battery is admissible at this scale"

Registration source: `experiments/t5a_capability_screen.py:17–19` (docstring), with the constant at `experiments/t5a_capability_screen.py:36` (`MIN_CELLS_FOR_WRITE_BATTERY = 32  # PLAN.md §3.6 power floor`).

The results file records the same claim as its `status`:

> "GATE — pre-declared bar fixed in-script before any model was run"

`results/t5a_capability_*.json -> status` (identical in all seven files).

No committed pre-registration document for T5A exists in `docs/archive/prereg/`.

## CONTROLS (as recorded)

| control | required value | observed value | fired | source |
|---|---|---|---|---|
| `WRITE_BATTERY_ADMISSIBLE`, `t5a_capability_70m.json` | `n_cells_ok >= 32` | `n_cells_ok` = 13 of `n_cells` = 61 | **FALSE** | `-> n_cells_ok`, `-> bar_min_cells`, `-> WRITE_BATTERY_ADMISSIBLE` |
| `WRITE_BATTERY_ADMISSIBLE`, `t5a_capability_70m_v2.json` | `n_cells_ok >= 32` | `n_cells_ok` = 13 of `n_cells` = 61 | **FALSE** | `-> n_cells_ok`, `-> bar_min_cells`, `-> WRITE_BATTERY_ADMISSIBLE` |
| `WRITE_BATTERY_ADMISSIBLE`, `t5a_capability_160m.json` | `n_cells_ok >= 32` | `n_cells_ok` = 20 of `n_cells` = 80 | **FALSE** | `-> n_cells_ok`, `-> bar_min_cells`, `-> WRITE_BATTERY_ADMISSIBLE` |
| `WRITE_BATTERY_ADMISSIBLE`, `t5a_capability_160m_v2.json` | `n_cells_ok >= 32` | `n_cells_ok` = 17 of `n_cells` = 61 | **FALSE** | `-> n_cells_ok`, `-> bar_min_cells`, `-> WRITE_BATTERY_ADMISSIBLE` |
| `WRITE_BATTERY_ADMISSIBLE`, `t5a_capability_410m.json` | `n_cells_ok >= 32` | `n_cells_ok` = 44 of `n_cells` = 61 | **TRUE** | `-> n_cells_ok`, `-> bar_min_cells`, `-> WRITE_BATTERY_ADMISSIBLE` |
| `WRITE_BATTERY_ADMISSIBLE`, `t5a_capability_410m_v2.json` | `n_cells_ok >= 32` | `n_cells_ok` = 44 of `n_cells` = 61 | **TRUE** | `-> n_cells_ok`, `-> bar_min_cells`, `-> WRITE_BATTERY_ADMISSIBLE` |
| `WRITE_BATTERY_ADMISSIBLE`, `t5a_capability_1b.json` | `n_cells_ok >= 32` | `n_cells_ok` = 53 of `n_cells` = 80 | **TRUE** | `-> n_cells_ok`, `-> bar_min_cells`, `-> WRITE_BATTERY_ADMISSIBLE` |
| multi-token answer exclusion | present only in the 61-cell files | `n_excluded_multitoken_answer` = 16 in `t5a_capability_70m.json`, `_70m_v2`, `_160m_v2`, `_410m.json`, `_410m_v2`; key **absent** from `t5a_capability_160m.json` and `t5a_capability_1b.json` | — | `-> n_excluded_multitoken_answer` |
| answer-copyable-from-prompt exclusion | present only in the 61-cell files | `n_excluded_answer_copyable_from_prompt` = 3 in the same five files; key **absent** from the two 80-cell files | — | `-> n_excluded_answer_copyable_from_prompt` |
| distinct answer tokens | present only in the 61-cell files | `n_distinct_answer_tokens` = 40 | — | `-> n_distinct_answer_tokens` |

## NUMBERS (as stored)

### Scales as rows

| file | model | n_cells | n_cells_ok | accuracy | median_rank_overall | bar_min_cells | WRITE_BATTERY_ADMISSIBLE | runtime_s |
|---|---|---|---|---|---|---|---|---|
| `t5a_capability_70m.json` | 70m | 61 | 13 | 0.2131 | 2 | 32 | false | 14.0 |
| `t5a_capability_70m_v2.json` | 70m | 61 | 13 | 0.2131 | 2 | 32 | false | 15.6 |
| `t5a_capability_160m.json` | 160m | 80 | 20 | 0.25 | 2 | 32 | false | 13.8 |
| `t5a_capability_160m_v2.json` | 160m | 61 | 17 | 0.2787 | 2 | 32 | false | 19.0 |
| `t5a_capability_410m.json` | 410m | 61 | 44 | 0.7213 | 0 | 32 | true | 16.7 |
| `t5a_capability_410m_v2.json` | 410m | 61 | 44 | 0.7213 | 0 | 32 | true | 25.4 |
| `t5a_capability_1b.json` | 1b | 80 | 53 | 0.6625 | 0 | 32 | true | 34.9 |

Source: `results/<file> -> n_cells`, `-> n_cells_ok`, `-> accuracy`, `-> median_rank_overall`, `-> bar_min_cells`, `-> WRITE_BATTERY_ADMISSIBLE`, `-> runtime_s`.

### By relation

Source for every cell: `results/<file> -> by_relation.<relation>.{n,n_ok,median_rank}`.

| file | capital n / n_ok / median_rank | language | continent | currency |
|---|---|---|---|---|
| `t5a_capability_70m.json` | 16 / 8 / 0 | 16 / 2 / 2 | 20 / 0 / 2 | 9 / 3 / 1 |
| `t5a_capability_70m_v2.json` | 16 / 8 / 0 | 16 / 2 / 2 | 20 / 0 / 2 | 9 / 3 / 1 |
| `t5a_capability_160m.json` | 20 / 12 / 0 | 20 / 4 / 2 | 20 / 4 / 2 | 20 / 0 / 17 |
| `t5a_capability_160m_v2.json` | 16 / 9 / 0 | 16 / 4 / 2 | 20 / 4 / 2 | 9 / 0 / 16 |
| `t5a_capability_410m.json` | 16 / 11 / 0 | 16 / 15 / 0 | 20 / 15 / 0 | 9 / 3 / 2 |
| `t5a_capability_410m_v2.json` | 16 / 11 / 0 | 16 / 15 / 0 | 20 / 15 / 0 | 9 / 3 / 2 |
| `t5a_capability_1b.json` | 20 / 13 / 0 | 20 / 19 / 0 | 20 / 16 / 0 | 20 / 5 / 1 |

### Files that are numerically identical

`t5a_capability_70m.json` and `t5a_capability_70m_v2.json` differ only in `runtime_s` (14.0 vs 15.6) and in the `rows` array; every summary key is equal. The same holds for `t5a_capability_410m.json` and `t5a_capability_410m_v2.json` (`runtime_s` 16.7 vs 25.4). `t5a_capability_160m.json` and `t5a_capability_160m_v2.json` differ on `n_cells` (80 vs 61), `n_cells_ok` (20 vs 17), `accuracy` (0.25 vs 0.2787), `by_relation`, `runtime_s`, and the three exclusion keys, which exist only in the `_v2` file.

### DISAGREEMENT, recorded without resolution: three stored versions of the same row

| source | 70m | 160m | 410m | 1b | denominator |
|---|---|---|---|---|---|
| `results/t5a_capability_70m.json`, `_160m.json`, `_410m.json`, `_1b.json` (`-> accuracy`) | 0.2131 | 0.25 | 0.7213 | 0.6625 | 61 / **80** / 61 / **80** |
| `results/t5a_capability_70m_v2.json`, `_160m_v2.json`, `_410m_v2.json` (`-> accuracy`) | 0.2131 | 0.2787 | 0.7213 | no `_v2` file | 61 / 61 / 61 |
| `docs/experiments/preregs/superseded/PREREG_PYTHIA_T7_v2.md` §0.1 burned table | 21.3% | 25.0% | 72.1% | 66.2% *("on the superseded 80-cell battery")* | as quoted |
| `RIGOR_SKILL.md:137` | 21.3 | 25.0 | 72.1 | — | "mixed denominators" |
| `REMOVELOG.md:119` | 21.3 | **27.9** | 72.1 | — | "all on 61" |
| `RESEARCH_NOTES.tex:1908` | 21.3\% | **27.9\%** | 72.1\% | — | "denominator 61 for all three" |

A further stored figure, from the commit that re-measured the screen after a scoring fix:

> "CONSEQUENCE -- the capability floor is re-measured and is MUCH LESS SHARP than reported:
>             OLD (double-normed)        NEW (correct)
>   70m       1/80  (1.2%)  rank 162     13/80 (16.2%)  rank 2
>   160m      2/80  (2.5%)  rank 11      20/80 (25.0%)  rank 2
>   410m     47/80 (58.8%)  rank 0       54/80 (67.5%)  rank 0
>   1b       49/80 (61.3%)  rank 0       53/80 (66.2%)  rank 0"

Stored in: git commit `0da25c3`, 2026-08-10, body. The current 80-cell files store `t5a_capability_160m.json -> n_cells_ok` = 20 and `t5a_capability_1b.json -> n_cells_ok` = 53; no 80-cell file for 70m or 410m remains in `results/`.

And from the commit that introduced the 61-cell denominator:

> "H7/M8 t5a now rejects multi-token answers (16 of 80 were scored on a BPE prefix, including a NEW ' krona'/' krone' -> ' k' collision of the ' yuan'/' yen' class) and answers copyable from the prompt. 61 cells survive: 70m 13/61 (21.3%), 410m 44/61 (72.1%)."

Stored in: git commit `33189b9`, 2026-08-10, body.

## VERDICT STRING AS STORED

> "GATE — pre-declared bar fixed in-script before any model was run"

Stored in: `results/t5a_capability_70m.json -> status` and identically in the other six T5A files.

The per-model outcome flag:

> `"WRITE_BATTERY_ADMISSIBLE": false` — 70m (both files), 160m (both files)
> `"WRITE_BATTERY_ADMISSIBLE": true` — 410m (both files), 1b

Stored in: `results/t5a_capability_*.json -> WRITE_BATTERY_ADMISSIBLE`.

A grading string in a spec document, not a results file:

> "| T5a capability floor | **C — DOWNGRADED** | **mixed denominators**: 70m/410m ran a 61-cell battery, 160m/1b an 80-cell one. 21.3/25.0/72.1% is not a comparable row until 160m is re-run |"

Stored in: `RIGOR_SKILL.md:137`.

> "| T5a \"21.3 / 25.0 / 72.1%\" | **SUPERSEDED** — mixed denominators (70m/410m on 61 cells, 160m on 80). One code path now gives **21.3 / 27.9 / 72.1%**, all on 61 | E6 row in §5.1 |"

Stored in: `REMOVELOG.md:119`.

A stored commit-message string covering the first run:

> "pythia T0/T4/T5a measured: capability floor 160M->410M, and kurtosis-without-readability at 70M"

Stored in: git commit `384565b`, 2026-08-09, subject line.

## PROVENANCE

| item | value |
|---|---|
| spec | none in `docs/experiments/preregs/`. Referenced as "PLAN.md §3.5" in the `experiment` field of every T5A results file. |
| prereg | in-script only (`experiments/t5a_capability_screen.py:17–19`, `:36`). No document in `docs/archive/prereg/`. |
| script | `experiments/t5a_capability_screen.py` (162 lines) |
| shared library | `src/joperator.py` (`final_logits`) |

| file | sha256 | git first-add | later modifications |
|---|---|---|---|
| `results/t5a_capability_70m.json` | `94070f5329706746eaa06133dcfc406ec1bec7a9e573e66698d574f9512b2bef` | `384565b`, 2026-08-09 | `5045a18`, `0da25c3`, `33189b9` (all 2026-08-10); `1340e76` (path move) |
| `results/t5a_capability_70m_v2.json` | `865a143f063eabb40f5e7c517c244ed97aded45b2e7765f7c3ce3d44222ca356` | `384565b`, 2026-08-09 | `5045a18`, `0da25c3`, `33189b9` (2026-08-10), `78fa26f` (2026-08-11); `1340e76` |
| `results/t5a_capability_160m.json` | `69215996a1414f66379fc73c251aab31cc8f7845ae39105b1f36429d32c285e4` | `384565b`, 2026-08-09 | `5045a18`, `0da25c3` (2026-08-10); `1340e76` |
| `results/t5a_capability_160m_v2.json` | `0ce7cbdb42b644c0556f2a2aa3a0ea139f59cc922df8f38388f0443df5cd8b1e` | `384565b`, 2026-08-09 | `5045a18`, `0da25c3`, `33189b9` (2026-08-10), `78fa26f` (2026-08-11); `1340e76` |
| `results/t5a_capability_410m.json` | `4c93e9aa87cf41c015a2567edbaa646f8d8684acdd6d8c076bfc9e4d17a46753` | `384565b`, 2026-08-09 | `5045a18`, `0da25c3`, `33189b9` (all 2026-08-10); `1340e76` |
| `results/t5a_capability_410m_v2.json` | `b79da821e59b859fff6fddc349a01c70d2ab38b02976de3735b0f2d270768f27` | `384565b`, 2026-08-09 | `5045a18`, `0da25c3`, `33189b9` (2026-08-10), `78fa26f` (2026-08-11); `1340e76` |
| `results/t5a_capability_1b.json` | `a5de750c587abfeca6dbad78bf82cdb0353b5e3897d0409c22392a2aca9236b2` | `384565b`, 2026-08-09 | `5045a18`, `0da25c3` (2026-08-10); `1340e76` |

`t5a_capability_160m.json` and `t5a_capability_1b.json` were **not** touched by `33189b9`, the commit whose message reports the 61-cell denominator; the other five were.

## RELATED / SUPERSEDED FILES

- `results/t5_write_410m.json` — the write battery this screen gates; imports `FACTS`, `REL_ORDER` and `RELATIONS` from `experiments/t5a_capability_screen.py`. Documented in `T5.md`.
- `results/t8_probe_swap_410m.json` — the successor write battery named in `t5_write_410m.json -> retracted`.
- `docs/context/RESULTS_TAXONOMY.md:746–752` lists all seven T5A files in its generated appendix with the column value `no`.
- `docs/experiments/preregs/superseded/PREREG_PYTHIA_T7_v2.md` §0.1 lists T5A in its burned-cell table, at the pre-`_v2` denominators.
