# EDATA1 -- reconciliation of paper-reported against released eval item counts

## WHAT WAS RUN

The per-set item counts in the released evaluation JSON of the vendored `anthropics/jacobian-lens`
repository (pinned at `581d398`, `data/evaluations/*.json`) were compared against the per-set counts
transcribed from the anchor paper. For each of the six eval sets the released count, the
paper-reported count and their ratio were recorded, together with a per-set verdict string. No model
was loaded and no measurement was taken.

## CONFIGURATION

| field | value |
|---|---|
| model | none -- `status = "EXPLORATORY — data audit, no model involved"` (`results/edata1_item_count_reconciliation.json -> status`) |
| checkpoint | not applicable |
| fitting corpus | not applicable |
| N | not applicable |
| seeds | not applicable |
| band | not applicable |
| target layer | not applicable |
| skip_first | not applicable |
| eval sets used | `multihop, multilingual, order-ops, poetry, typo, association` (`-> per_set`) |
| aggregations stored | not applicable -- counts and ratios only |
| device | not applicable |
| readout | not applicable |
| source (paper side) | `RIGOROUS_ANTHROPIC.md sec A.6 (operator transcription of 2607.15495)` (`-> source_paper`) |
| source (data side) | `anthropics/jacobian-lens @581d398 data/evaluations/*.json` (`-> source_data`) |
| date run | NOT RECORDED -- the file has no `provenance` block; file mtime is 2026-08-11 09:58 |

## PRE-REGISTERED RULE (verbatim)

NOT PRE-REGISTERED. The file records `status = "EXPLORATORY — data audit, no model involved"` and
stores no decision rule, no threshold and no control block.

Registration source: NONE LOCATABLE.

## CONTROLS (as recorded)

| control | required value | observed value | fired | source |
|---|---|---|---|---|
| (none) | -- | -- | -- | the file stores no `controls` key |

## NUMBERS (as stored)

`results/edata1_item_count_reconciliation.json -> per_set.<set>`

| set | released | paper_reported | ratio | stored per-set verdict |
|---|---|---|---|---|
| multihop | 93 | 50 | 1.86 | "~2x released" |
| multilingual | 107 | 54 | 1.981 | "~2x released" |
| order-ops | 55 | 55 | 1.0 | "EXACT MATCH" |
| poetry | 98 | 52 | 1.885 | "~2x released" |
| typo | 96 | 96 | 1.0 | "EXACT MATCH" |
| association | 102 | 50 | 2.04 | "~2x released" |

No `min`, `persist` or other aggregation is stored in this file.

## VERDICT STRING AS STORED

The file stores no `VERDICT` key. It stores a `finding`, a `rejected_explanations` list and a
`decision`, verbatim:

> `finding`: "Two sets (order-ops 55, typo 96) match the paper EXACTLY. The other four are 1.86-2.04x the paper's reported counts (mean 1.94). Most consistent explanation: the released sets were EXPANDED, roughly doubled, after the paper was written, for four of six sets; order-ops and typo were left at their published size."

> `rejected_explanations`:
> - "tokenizer filtering: would not produce a near-uniform 2x, and would not leave two sets exact"
> - "subsampling for the figures: possible but would not so cleanly halve four sets and leave two intact"
> - "training/fitting divergence: impossible -- our fastfit is asserted bit-identical to jlens.fit"

> `decision`: "USE THE RELEASED JSON. It is what the reference implementation consumes and what any replicator obtains. Consequence, disclosed: our absolute AUCs are NOT comparable to the paper's reported values. The J-vs-logit contrast is internal to our run and unaffected."

Stored in: `results/edata1_item_count_reconciliation.json -> finding`, `-> rejected_explanations`, `-> decision`

## PROVENANCE

| item | value |
|---|---|
| spec | none in `docs/experiments/preregs/` |
| prereg | NONE LOCATABLE |
| script | NOT RECORDED -- the file has no `provenance` block and no producing script was located in `experiments/` |
| results file | `results/edata1_item_count_reconciliation.json`, sha256 `64bab884c8d0250f9efcb384c47687a240b9c52c20ea3675297673955329a443`, 1949 bytes |
| git commit | not recorded in the file. The file first appears in the repository at commit `1340e76` ("Restructure the repo into src/ + experiments/; retire docs/ and FUTURE_IMPROVEMENTS") |
| environment | NOT RECORDED |

## RELATED / SUPERSEDED FILES

- `docs/context/RESULTS_TAXONOMY.md:689` -- index row: "`edata1_item_count_reconciliation.json` \| E-data1 — reconciling paper-reported vs released eval item count \| no".
- `REMOVELOG.md:33` -- records the label change "E-data1 → **E22**", script column `edata1`, description "item-count reconciliation".
- `CLAUDE.md` §5 -- the table row "eval item counts \| 50/54/55/52/96/50 \| released JSON has **93/107/55/98/96/102** \| released JSON, disclosed", which lists the same twelve counts.
- `paper/CONTEXT.md` and `review/reproducibility/*.md` -- documents that reference `edata1` by name; not evaluated here.
