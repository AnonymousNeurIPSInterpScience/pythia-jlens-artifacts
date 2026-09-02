# EN1 -- operator estimation error at n=200, derived from stored dispersion

## WHAT WAS RUN

For six Pythia sizes, the stored layer-0 dispersion value was substituted into the relation recorded
in the file's `rule` field to give a relative RMS Frobenius error `eps` at `n = 200`, and the `n`
required to reach 5% and 10% error at layer 0. Layer-0, mid-layer and last-layer `eps` at `n = 200`
were recorded per model, alongside each model's layer count. No model was run and no new measurement
was taken.

## CONFIGURATION

| field | value |
|---|---|
| model | six sizes: `70m`, `160m`, `410m`, `1b`, `1.4b`, `2.8b` (`results/en1_operator_error_at_n200.json -> per_model`) |
| checkpoint | not applicable |
| fitting corpus | not applicable -- the input is a stored dispersion figure per model |
| N | `n = 200` (`-> n`) |
| seeds | not applicable |
| band | not applicable -- values are given at layer 0, a mid layer and the last layer |
| target layer | not applicable |
| skip_first | not applicable |
| eval sets used | none |
| aggregations stored | not applicable -- `eps` and required-`n` values only |
| device | not applicable -- `status = "EXPLORATORY — analytic, derived from stored dispersion; no model run"` (`-> status`) |
| readout | not applicable |
| stated source | "a review response answer item 2" (`-> source`) |
| date run | NOT RECORDED -- the file has no `provenance` block; file mtime is 2026-08-11 10:50 |

## PRE-REGISTERED RULE (verbatim)

NOT PRE-REGISTERED. The file stores no decision rule with ACCEPT/REJECT/UNCLEAR branches. It stores
a `rule` field describing the relation used, verbatim:

> "E||M_hat_n - M||_F^2 / ||M||_F^2 = disp/n  =>  eps = sqrt(disp/n);  n >= disp/eps^2"

Registration source: NONE LOCATABLE.

## CONTROLS (as recorded)

| control | required value | observed value | fired | source |
|---|---|---|---|---|
| (none) | -- | -- | -- | the file stores no `controls` key |

## NUMBERS (as stored)

`results/en1_operator_error_at_n200.json -> per_model.<model>`

| model | n_layers | disp_L0 | eps_L0_at_200 | eps_mid_at_200 | eps_last_at_200 | n_for_5pct_at_L0 | n_for_10pct_at_L0 |
|---|---|---|---|---|---|---|---|
| 70m | 6 | 0.229049 | 0.0338 | 0.0246 | 0.0169 | 92 | 23 |
| 160m | 12 | 0.641485 | 0.0566 | 0.0374 | 0.0179 | 257 | 64 |
| 410m | 24 | 2.364437 | 0.1087 | 0.053 | 0.0106 | 946 | 236 |
| 1b | 16 | 1.73571 | 0.0932 | 0.0502 | 0.0137 | 694 | 174 |
| 1.4b | 24 | 2.897373 | 0.1204 | 0.0503 | 0.0143 | 1159 | 290 |
| 2.8b | 32 | 4.418416 | 0.1486 | 0.0442 | 0.0079 | 1767 | 442 |

No `min`, `persist` or other read aggregation is stored in this file.

The `disp_L0` values also appear in `results/t14b_dispersion_scaling.json` and
`results/t14_dispersion_ladder.json`.

## VERDICT STRING AS STORED

The file stores no `VERDICT` key. It stores a `finding` and a `caveat`, verbatim:

> `finding`: "n=200 is NOT uniformly adequate. Layer-0 relative RMS Frobenius error runs 3.4% (70m) to 14.9% (2.8b) and grows with DEPTH, tracking dispersion. Late layers are fine (<1% at 2.8b). Reaching 5% error at 2.8b layer 0 would need n~1767 -- which retroactively justifies the anchor's n=1000 and shows our n=200 is marginal at the large end."

> `caveat`: "This is a second-moment expectation for FROBENIUS error only. It is NOT a high-probability guarantee, NOT an operator-norm bound, and NOT valid for singular-subspace error or under heavy tails -- all flagged in the same source."

Stored in: `results/en1_operator_error_at_n200.json -> finding`, `-> caveat`

## PROVENANCE

| item | value |
|---|---|
| spec | none in `docs/experiments/preregs/` |
| prereg | NONE LOCATABLE |
| script | NOT RECORDED -- the file has no `provenance` block and no producing script was located in `experiments/` |
| results file | `results/en1_operator_error_at_n200.json`, sha256 `428edcfa296636ed98d6d68106c8ef84dcfebb57778c655b8a803292909078e8`, 2200 bytes |
| git commit | not recorded in the file. The file first appears in the repository at commit `1340e76` ("Restructure the repo into src/ + experiments/; retire docs/ and FUTURE_IMPROVEMENTS") |
| environment | NOT RECORDED |

## RELATED / SUPERSEDED FILES

- `results/t14b_dispersion_scaling.json` and `results/t14_dispersion_ladder.json` -- the files that also store the `disp_L0` values EN1 uses as inputs.
- `docs/context/RESULTS_TAXONOMY.md:690` -- index row: "`en1_operator_error_at_n200.json` \| E-n1 — operator estimation error at n=200, from measured dispers \| no".
- `REMOVELOG.md:32` -- records the label change "E-n1 → **E21**", script column `en1`, description "operator error at n=200".
- `paper/CONTEXT.md:177` -- a paragraph that names `E-n1` and the relation `ε = √(disp/N)`; not evaluated here.
- `CLAUDE.md` §4 "RETRACTED 2026-08-13" -- records: "The −0.511 sample-size law is close to a mathematical identity (CLT applied to a mean). Not a discovery. Do not present it as one." That retraction is recorded in a markdown document; no results file was located that stores it.
