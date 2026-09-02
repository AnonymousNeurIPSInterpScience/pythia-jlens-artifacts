# METRICS — the definitions every number in this folder is reported under

**Scope.** Definitions taken **from the code**, quoted, with file and line. Where a document and a
results file disagree, both are recorded with both paths and neither is adjudicated. Numbers carry
`results/<file>.json -> <json.path.to.key>`. Quantities not found in a results file are marked
`UNSTORED`, with the code or document that carries them named instead.

Repository state while this file was written: git commit `b07615ef55878bd04678e9a17068145105b4b20f`
(branch `pythia`, working tree dirty). Vendored anchor: `jacobian-lens/` @ `581d398`.

SHA-256 of the files quoted below:

| file | sha256 |
|---|---|
| `src/anchor_evals.py` | `aeb2e9feed448faeed8a572fb11a51a93e7e5d95b6bae42b3515922b898b1f2e` |
| `experiments/trainval.py` | `920a3564c5441b23528e6c44a8e96c33871abfcccf8f70fa4ba56cf19d86d95f` |
| `experiments/e28_ladder_410m.py` | `ad90a25f47f4e4eead20e088d5e355185198324edbf77d1c086b81c44c2bd28f` |
| `experiments/t33_logit_baseline.py` | `eba51672925d8189af516c391b863a5712c9372d92dac10fde45455625a07283` |
| `experiments/t48_crossover.py` | `a76a103f93a2ba7867f349fba0f937863c9c55aba7474e8a60109f49e00a69db` |
| `experiments/t17_reaggregate.py` | `0b4cb9947c74a1f5091dced40378145c6d9a36bcf0f990f55c60ec6940078984` |
| `experiments/t13_transport_controls.py` | `e0cbb418358bec9e20668edc8fc56b577cf10d586663ca2ca6e3d6de7152906d` |
| `experiments/d1_min_union_diagnostic.py` | `22894e1e54db6a3adb12f497f61336054826f61795dd54dda490d17a8ff8281d` |
| `jacobian-lens/data/evaluations/README.md` | `e061d9cce02a1cc651d58a81927833b760d3cef65bf4995126ecbe372a0ebe07` |
| `docs/context/AGGREGATION_POLICY.md` | `a5b73ff998c5a9c7217f5ace4df13dba192313d9884355b19ba36107895c142a` |

---

## 0. The object all four aggregations are computed from

Every aggregation consumes the same matrix `R`. Per eval set, `R` holds one **rank** per
(band layer) × (item, intermediate) pair.

The rank of one intermediate at one layer is the best (lowest) rank over that intermediate's
single-token surface forms. `src/anchor_evals.py:138-143`:

```python
def rank_of(logits: torch.Tensor, ids: list[int]) -> int | None:
    """Best 1-indexed rank of any id in `ids` within a [vocab] logit vector."""
    if not ids:
        return None
    best = logits[ids].max()
    return int((logits > best).sum().item()) + 1
```

Surface forms come from `token_ids_of` (`src/anchor_evals.py:123-135`): each synonym is tried in
six casings/leading-space forms and **multi-token forms are dropped**. `synonyms`
(`:111-120`) expands digits to number words and the four operation names to symbol/word lists for
`order-ops`.

Construction of `R` in the fitting-and-scoring instrument, `experiments/trainval.py:265-272`:

```python
    def evaluate(Jmean):
        per = {}
        with torch.no_grad():
            for name, acts, tgt in items:
                R = torch.tensor([[min((rank_of(model.unembed(acts[l] @ Jmean[l].T).float().cpu(), i)
                                        or 10**9) for i in t) for t in tgt] for l in band],
                                 dtype=torch.float32)
                per.setdefault(name, []).append(R)
```

`R` here is `[len(band), n_pairs]`. Two call conventions for the surface-form minimum coexist and
return the same value: `src/anchor_evals.py:237` passes the whole id list to `rank_of`, which takes
`logits[ids].max()` internally; `experiments/trainval.py:269-270` passes one id at a time and wraps
the call in `min(... for i in t)`. In `experiments/t48_crossover.py:271-285` the same matrix is
built transposed, `[n_pairs, len(band)]`, with the surface-form minimum taken as
`cand.min(0).values`, and the aggregations index accordingly (`R.min(dim=1)`, `.sum(1)`).

---

## 1. `min`

### 1.1 The implementation

`experiments/trainval.py:275-277`:

```python
            R = torch.cat(Rs, dim=1); mn = R.min(dim=0).values
            out[name] = {
                "min":     sum((mn <= k).float().mean().item() for k in K) / len(K),
```

The same expression appears at `experiments/e28_ladder_410m.py:128-129`,
`experiments/t33_logit_baseline.py:115-116`, `experiments/t37_rank_ablation.py:145-146`,
`experiments/t43_derangement_stability.py:107-108`.
`experiments/t48_crossover.py:286-289` is the transposed form:

```python
    def aggregate(R):
        """Per-pair scores under both aggregations. mean over pairs == trainval.py's set AUC."""
        mn = R.min(dim=1).values
        return {"min": torch.stack([(mn <= k).float() for k in K]).mean(0),
```

**What it minimises over:** the layers of the **band** (the rows/columns of `R`, which are the band
layers only), after the per-layer rank has already been minimised over the intermediate's
single-token surface forms. The layer index at which the minimum occurs is not recorded.

**What is then averaged:** the indicator `mn <= k` is averaged over all (item, intermediate) pairs
of that eval set, then averaged **unweighted over the 7 values of `K`** (§4).

`experiments/t17_reaggregate.py:80-83` computes `min` over **all fitted layers** (`R.min(dim=1)`
on the full `R`), while its `persist` is computed on the band sub-matrix `Rb`:

```python
    Rb = R[:, [idx[l] for l in band]]
    ...
    out["min"] = auc([float(hits(R.min(dim=1).values, k).mean()) for k in KS])
```

### 1.2 What the anchor defines

The vendored `jacobian-lens/data/evaluations/README.md` states the metric once per set, verbatim
(identical sentence in all six sections):

> Metric: pass@k = mean over items of the fraction of `intermediates` whose min-over-layers lens
> rank ≤ k.

and, under `## Conventions`:

> - **Workspace band** — the contiguous mid-network layer range where
>   workspace content is read; experiments report over this band, not
>   individual layers.
> - **Hit** — a target token is a *hit* if it appears at lens rank 1 at any
>   (layer, position) in the band over the scored span.

`jacobian-lens/jlens/` ships **no scoring or aggregation code**: `grep` for `pass_at_k` / `pass@k`
over `jacobian-lens/jlens/*.py` returns no hits; the only rank computation in the library is
`jlens/vis.py:98 _ranks_of`, which returns `[seq_len, n_targets]` ranks for the visualiser. The
normative statement of `min` inside the vendored checkout is therefore the README text above.

The operator's transcription of the source paper, `paper/fundamentals/RIGOROUS_ANTHROPIC.md:690`, records:

> The paper reports that at a single layer the number of simultaneously represented unrelated items
> is smaller—around one to two—than the number observed when aggregating across the workspace layer
> range. [Capacity and Generalization](https://arxiv.org/abs/2607.15495)

### 1.3 Two implementations of pass@k in this repo, and which one the ladder uses

`src/anchor_evals.py:212-292` implements the README metric and a second form, and stores both:

- `pass_at_k` / `normalized_auc_over_logk` — "mean over ITEMS of the FRACTION of that item's
  intermediates whose min-over-layers ... rank is <= k" (`:220-222`, `:253-254`).
- `pass_at_k_paper` / `normalized_auc_over_logk_paper` — a flat pool over (item, intermediate)
  pairs (`:256-275`). The docstring at `:264-272` records the choice:

  > INTERPRETIVE CHOICE, flagged rather than buried. The alternative generalizations are
  > "item counts if ANY intermediate is recovered" (more permissive) and "...if ALL are"
  > (stricter). Both are defensible; neither reduces to the paper's formula more naturally
  > than the flat pool. Registered in docs/experiments/preregs/superseded/PREREG_PYTHIA_T7_v2.md §3.

The ladder / crossover / factorial family (`trainval.py`, `e28_ladder_410m.py`, `t33`, `t36`,
`t37`, `t48`, `t52`, `t59`, `r6`, `r7`, `d1`) does **not** call `pass_at_k`. It pools per
(item, intermediate) pair and takes a flat mean over `K` (§4).

---

## 2. `persist`

`experiments/trainval.py:281-282`:

```python
                "persist": sum(((R <= k).float().sum(0) >= (len(band) // 2)).float().mean().item()
                               for k in K) / len(K)}
```

`experiments/t48_crossover.py:256` and `:290-291`:

```python
    HALF = L // 2
...
                "persist": torch.stack([((R <= k).float().sum(1) >= HALF).float()
                                        for k in K]).mean(0)}
```

A pair counts at threshold `k` iff its rank is `<= k` at **at least `len(band) // 2` band layers**.
The count is over band layers only; the indicator is then averaged over pairs and over `K`.

Threshold values, from the stored bands:

| scale | band (stored) | `len(band)` | `len(band)//2` | source of the threshold value |
|---|---|---|---|---|
| 410M | `[9..21]` | 13 | **6** | stored: `results/d1_min_union_diagnostic_410m.json -> half_threshold_for_persist` = `6` |
| 1B, ladder as run | `[5..13]` | 9 | 4 | `len(band)//2` evaluated on `results/ladder1b/tv_Github_s0.json -> band`; not stored as its own field |
| 1B, declared band | `[6..13]` | 8 | 4 | `len(band)//2` evaluated on `results/e62_band_adjudication.json -> C2_band.expected`; not stored as its own field |
| 70M (E28 band) | `[2,3]` | 2 | 1 | `experiments/trainval.py:226-228`, comment quoted below |

`experiments/trainval.py:226-228` records the 70M case:

> `target_layer=-2 resolves to n_layers-2, and jlens requires every source layer to be STRICTLY
> below it. At 70m (6 layers) that caps the band at [2,3] -- the same two-layer band E28 used,
> which is why `persist` degenerates to `min` at that scale.`

`experiments/t17_reaggregate.py:89-92` uses a **fraction-of-band** threshold instead of floor
division, `--kfrac` default `0.5`:

```python
    K = max(1, int(round(kfrac * len(band))))
    out["persist_K"] = K
    out["persist_band_size"] = len(band)
    out["persist"] = auc([float(((Rb <= k).sum(dim=1) >= K).float().mean()) for k in KS])
```

`experiments/t17_reaggregate.py:15-16` states the name:

> `  persist  persistence at K       rank <= k at >= K band layers, K a FRACTION of band depth`
> `                                  (the fix; external review B3 confirms it has no canonical name)`

No definition of `persist` appears in `jacobian-lens/`.

---

## 3. `best1L` and `mean`

`experiments/trainval.py:278-280`:

```python
                "best1L":  max(sum((R[i] <= k).float().mean().item() for k in K) / len(K)
                               for i in range(R.shape[0])),
                "mean":    sum((R <= k).float().mean().item() for k in K) / len(K),
```

- **`best1L`** — for each band layer `i`, the pair-mean hit rate at each `k`, averaged over `K`;
  `best1L` is the **maximum of that quantity over band layers**. The maximising layer index is not
  stored by `trainval.py`. `experiments/t17_reaggregate.py:85-88` stores it as `best1L_layer`
  alongside a full `per_layer` map.
- **`mean`** — the mean of the indicator `R <= k` over **all (layer, pair) cells** of the band,
  averaged over `K`. `R` is rectangular, so this equals the unweighted mean over band layers of the
  per-layer pass@k, averaged over `K`.

Both exist in every file produced by `trainval.py`, `e28_ladder_410m.py`, `t33_logit_baseline.py`,
`t17_reaggregate.py` and `t19_skipfirst_sweep.py` (§9).

Where they are reported: `results/r4_corrections.json -> R4c_all_four_aggregations`, whose own
`finding` field reads:

> "F-5 second half — best1L and mean exist in every ladder cell and are never reported"

Stored values from that block, corpus × eval-set variance decomposition, all four aggregations
(`results/r4_corrections.json -> R4c_all_four_aggregations.by_scale.<scale>.<agg>`):

| scale | agg | `frac_set_main_pct` | `frac_corpus_main_pct` | `frac_interaction_pct` | `frac_residual_seed_pct` | `interaction_over_corpus_main` |
|---|---|---|---|---|---|---|
| 410m | persist | 91.27829399393488 | 1.9064501148312791 | 6.780626428099203 | 0.034629463134648465 | 3.556676555735259 |
| 410m | min | 96.06544098391524 | 1.911884671540871 | 1.9989092146925354 | 0.023765129851367858 | 1.0455176739722107 |
| 410m | mean | 94.18015768908394 | 1.6504107475753969 | 4.145089404552656 | 0.024342158787985454 | 2.511550176610379 |
| 410m | best1L | 92.93385922614715 | 2.553416835551286 | 4.4645987524086 | 0.04812518589297392 | 1.7484801894653001 |
| 1b | persist | 88.57950497345804 | 2.2117202251311614 | 9.158325214755473 | 0.05044958665530746 | 4.1408154208167804 |
| 1b | min | 94.14618752120758 | 1.9252567551293127 | 3.896419437299789 | 0.03213628636333453 | 2.0238440545235643 |
| 1b | mean | 89.08620460630291 | 2.634040417272565 | 8.227329209779771 | 0.05242576664476654 | 3.123463541344903 |
| 1b | best1L | 95.00877329965921 | 1.7035403113063254 | 3.229682343981307 | 0.05800404505314376 | 1.8958649364186106 |

One stored cell under all four aggregations, for reference
(`results/ladder410/ladder_Github_s0.json -> by_N.200.<set>.<agg>`; file fields `device: "cuda"`,
`band: [9..21]`, `n_items: 541`):

| eval set | min | best1L | mean | persist |
|---|---|---|---|---|
| multihop | 0.07628293974058968 | 0.06518724028553281 | 0.01717699766491673 | 0.004160887694784573 |
| multilingual | 0.11167512674416814 | 0.07251631641494376 | 0.031014670551355396 | 0.025743291713297367 |
| order-ops | 0.07922077950622354 | 0.07922077950622354 | 0.023676323671159998 | 0.022077922310147966 |
| poetry | 0.017492711410990784 | 0.011661807474281107 | 0.004821709091109889 | 0.004373177752963134 |
| typo | 0.2633928582072258 | 0.16071428837520735 | 0.09100274728345019 | 0.07886904890515975 |
| association | 0.006211180106869766 | 0.003105590119957924 | 0.0004777830742698695 | 0.0 |

---

## 4. The k-summary: grid, weighting, and whether it is a log-k trapezoid

**Two different summaries exist in this repository.**

### 4.1 The flat mean over `K` — the ladder / crossover / factorial family

`experiments/trainval.py:45`:

```python
K = [1, 2, 5, 10, 20, 50, 100]
```

The identical line `K = [1, 2, 5, 10, 20, 50, 100]` is defined in 15 scripts:
`e28_ladder_410m.py:113`, `t33_logit_baseline.py:44`, `t36_qladder.py:100`, `t37_rank_ablation.py:63`,
`t41_disagreement_ablation.py:72`, `t42_ablation_pairs.py:73`, `t43_derangement_stability.py:49`,
`t48_crossover.py:111`, `t52_factorial.py:59`, `t59_read_dose.py:51`, `t65_ckpt_readout.py:40`,
`t66_fitter_equivalence_cuda.py:60`, `r6_within_source.py:42`, `d1_min_union_diagnostic.py:50`,
`trainval.py:45`. `experiments/r7_matched_pools.py:41` imports `K` from `r6_within_source`.
It is stored as a field: `results/e48_crossover_410m_rstrip.json -> K` = `[1, 2, 5, 10, 20, 50, 100]`;
`results/d1_min_union_diagnostic_410m.json -> K`; `results/d2_device_crossvalidation.json ->
shared_conditions.K`; `results/e52_factorial_410m_rstrip.json -> K`.

The summary is `sum(... for k in K) / len(K)` in every one of the four aggregation expressions
quoted in §1–§3: an **unweighted arithmetic mean over the 7 values of `K`**. There is **no
trapezoid and no log-k weighting** in this path. The quantity is called "read AUC" in the code:
`experiments/trainval.py:15` — "performance : read AUC per eval set under min / best1L / mean /
persist"; `experiments/t48_crossover.py:32` — "metric read AUC over K=[1,2,5,10,20,50,100]".

Per-set values are combined across eval sets by an **unweighted mean over the five admitted sets**,
not weighted by pair count — `experiments/t48_crossover.py:293-297`:

```python
    def set_mean(v, s):
        return float(v[SET_IDX[s]].mean())

    def admitted_mean(v):
        return statistics.mean([set_mean(v, s) for s in ADMITTED])
```

and identically `experiments/t33_logit_baseline.py:126-127`.

### 4.2 The log-k trapezoid — `anchor_evals.pass_at_k` and `t17_reaggregate`

`src/anchor_evals.py:213-214` default grid, and `:245-250` the summary:

```python
def pass_at_k(model, transport: Callable, eval_name: str, layers,
              ks=(1, 2, 5, 10, 25, 100), max_items: int | None = None,
...
    xs = [math.log(k) for k in ks]

    def _auc(curve_ys: list[float]) -> float:
        a = sum((curve_ys[i] + curve_ys[i + 1]) / 2 * (xs[i + 1] - xs[i])
                for i in range(len(ks) - 1))
        return a / (xs[-1] - xs[0])
```

`experiments/t17_reaggregate.py:44-51` is the same construction:

```python
KS = (1, 2, 5, 10, 25, 100)
_LOGK = [math.log(k) for k in KS]
_SPAN = _LOGK[-1] - _LOGK[0]


def auc(curve) -> float:
    a = sum((curve[i] + curve[i + 1]) / 2 * (_LOGK[i + 1] - _LOGK[i]) for i in range(len(KS) - 1))
    return a / _SPAN
```

### 4.3 The two grids side by side

| | grid | weighting | files that use it |
|---|---|---|---|
| flat mean | `[1, 2, 5, 10, 20, 50, 100]` (7 points; includes 20 and 50, excludes 25) | unweighted mean over `K` | the 15 scripts listed in §4.1, and every file they write |
| log-k trapezoid | `(1, 2, 5, 10, 25, 100)` (6 points; includes 25, excludes 20 and 50) | trapezoid in `log k`, normalised by `log(100) − log(1)` | `src/anchor_evals.py:pass_at_k`, `experiments/t17_reaggregate.py` |

`src/anchor_evals.py:18-24` states the README metric it implements:

> METRIC, verbatim from `data/evaluations/README.md`:
>
>     pass@k = mean over ITEMS of the FRACTION of `intermediates` whose min-over-layers lens rank <= k
>
> Note this is a mean of per-item fractions, NOT a flat pool over all (item, intermediate) pairs.
> The two differ whenever items carry different numbers of intermediates — multilingual items have
> four, association items have one.

The vendored `jacobian-lens/data/evaluations/README.md` specifies `pass@k` and does **not** specify
a `k` grid or any summary over `k`.

---

## 5. The BAND

### 5.1 The rule as stated in code

`experiments/t17_reaggregate.py:54-57`:

```python
def band_of(n_layers: int, layers: list[int]) -> list[int]:
    """The anchor's normalised workspace band, L38-L92 of a 0-100 scale, clipped to fitted layers."""
    lo, hi = round(0.38 * n_layers), round(0.92 * n_layers)
    b = [l for l in layers if lo <= l <= hi]
```

`experiments/t53_ladder_summary.py:178-191`:

```python
    for scale, n_layers in (("410m", 24), ("1b", 16)):
        lo, hi = round(0.38 * n_layers), round(0.92 * n_layers)
        target = n_layers - 2
        declared = [lo, min(hi, target - 1)]
```

The same `round(0.38·n) … round(0.92·n)` appears at `experiments/t14_dispersion_ladder.py:102` and
in the generated table text `paper/tools/make_paper_tables.py:344`:
`"$[%d,%d]$, being $\\mathrm{round}(0.38n)\\ldots\\mathrm{round}(0.92n)$ intersected with layers
strictly below the target"`.

`experiments/trainval.py` takes the band from the command line: `--band` (explicit inclusive
`lo,hi`) overriding `--band-frac` (default `"0.35,0.85"`), then intersects with layers strictly
below `target_eff = n_layers - 2` (`:225-234`). `experiments/trainval.py:146-149`:

> `"explicit inclusive band lo,hi. Overrides --band-frac. E48 uses 9,21 so every arm shares the
> E28/E33 geometry and one activation cache."`
> `"band as a fraction of depth; the anchor's normalized L38–L92 convention"`

Where the band is hardcoded: `experiments/t48_crossover.py:110` — `BAND = list(range(9, 22))
# [9..21] — E28/E33, and the OOD lens band`.

### 5.2 The anchor's statement of the range

`paper/fundamentals/RIGOROUS_ANTHROPIC.md:715` (operator transcription of the source):

> The analyses identify a region beginning around layer $L38$ and ending around layer $L92$, using
> a normalized layer scale from $0$ to $100$. In the middle region, readouts are persistent and
> abstract; in the final layers, they become increasingly aligned with imminent output tokens.
> [Workspace Layer Range](https://arxiv.org/abs/2607.15495)

The vendored README's `## Conventions` entry for "Workspace band" (quoted in §1.2) gives no layer
numbers.

### 5.3 Declared vs used, as stored

`results/e53_ladder_summary.json -> bands_declared_vs_used`:

| scale | `n_layers` | `L38_L92` | `target_layer_index` | `declared_by_the_papers_own_rule` | `used_in_the_ladder` | `matches` |
|---|---|---|---|---|---|---|
| 410m | 24 | `[9, 22]` | 22 | `[9, 21]` | `[9, 21]` | `true` |
| 1b | 16 | `[6, 15]` | 14 | `[6, 13]` | `[5, 13]` | `false` |

The 1B difference was re-fitted in E62. `results/e62_band_adjudication.json -> PRIMARY`:

| quantity | `band_6_13` | `band_5_13_stored` |
|---|---|---|
| `frac_interaction` | 9.191520642250026 | 9.158325214755475 |
| `frac_corpus_main` | 2.2424534939728105 | 2.2117202251311614 |
| `frac_set_main` | 88.50952415142129 | — |
| `interaction_over_corpus_main` | 4.098867899358752 | — |
| `delta_pp` | 0.033195427494550955 | |

`results/e62_band_adjudication.json -> VERDICT`:

> "CONFIRMS — interaction 9.192% on [6,13] vs 9.158% on [5,13] (+0.033 pp), still above the 2.242%
> corpus main effect. The band defect was immaterial; report the corrected number and retire the
> deviation."

Controls stored with it: `C2_band` (`expected [6..13]`, `n_cells 15`, `n_matching 15`,
`fires: true`); `C3_prompt_pool` (200 vs 200 prompts in all 15 cells, `fires: true`);
`C1_shared_layer_dispersion` (`n_comparisons 840`, `max_abs_diff 3.6828356636209314e-05`,
`tolerance 0.001`, `fires: true`).

### 5.4 Band width as a swept quantity

`experiments/d1_min_union_diagnostic.py:138-145` scores `min` on **centred sub-bands**:

```python
    def min_auc_on_sub_band(R, w):
        """`min` restricted to the CENTRED sub-band of width w. If min's preference for the
        derangement is a union-size artifact, it must vanish as w shrinks: at w=1 min is a
        single-layer readout and cannot benefit from decorrelation at all."""
        lo = (L - w) // 2
        sub = R[:, lo:lo + w]
        mn = sub.min(dim=1).values
        return admitted_mean(torch.stack([(mn <= k).float() for k in K]).mean(0))
```

Stored per corpus at
`results/d1_min_union_diagnostic_410m.json -> adjudication.<corpus>.H4_artifact_scales_with_band_width.min_gap_jp_minus_shuf_by_band_width.<w>`
for `w = 1..13`, together with `narrowest_width_where_jp_still_wins`. Example, Pile-CC: `w=6`
`+0.007677602892120675`, `w=7` `-0.015596643959482487`. Example, USPTO_Backgrounds: `w=6`
`+0.014086514587203669`, `w=7` `+0.0005327625821034132`, `w=8` `-0.0003358248621225246`,
`narrowest_width_where_jp_still_wins = "13"`.

---

## 6. The eval sets: six released, five admitted

### 6.1 The six sets and the admitted five

`src/anchor_evals.py:63`:

```python
EVAL_SETS = ["multihop", "multilingual", "order-ops", "poetry", "typo", "association"]
```

`ADMITTED` is defined identically in 21 files under `experiments/` and `tools/`, e.g.
`experiments/t48_crossover.py:112`:

```python
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]   # association floored
```

The 21 definition sites: `experiments/d1_min_union_diagnostic.py:51`,
`d2_device_crossvalidation.py:45`, `r6_within_source.py:43`, `t33_logit_baseline.py:45`
(`# association: floored`), `t34_dact_vs_floor.py:74`, `t34b_jeval_dact.py:51`,
`t36_qladder.py:101`, `t37_rank_ablation.py:64`, `t39_dread.py:72`, `t48_crossover.py:112`,
`t51_interaction_variance.py:77`, `t52_factorial.py:60`, `t53_ladder_summary.py:44`,
`t54_aggregation_audit.py:56` (`# association is floored`), `t59_read_dose.py:52`,
`t61_adjudicate_null.py:29`, `t62_adjudicate_band.py:27`, `t65_ckpt_readout.py:41`,
`t66_fitter_equivalence_cuda.py:61`, `tools/r3_close_d2.py:36`, `tools/r8_ladder_flatness.py:29`.
`experiments/r7_matched_pools.py:41` imports `ADMITTED` from `r6_within_source`.

The recorded reason, stored:
`results/e33_logit_baseline_410m_v2.json -> excluded_sets.association` =

> "floored — 196/205 ladder cells exactly zero (E28 crit 3)"

`results/e53_ladder_summary.json -> admitted_sets` = `["multihop", "multilingual", "order-ops",
"poetry", "typo"]`.

### 6.2 Item counts — released JSON vs paper text

`results/edata1_item_count_reconciliation.json -> per_set`:

| set | `released` | `paper_reported` | `ratio` | `verdict` (stored) |
|---|---|---|---|---|
| multihop | 93 | 50 | 1.86 | "~2x released" |
| multilingual | 107 | 54 | 1.981 | "~2x released" |
| order-ops | 55 | 55 | 1.0 | "EXACT MATCH" |
| poetry | 98 | 52 | 1.885 | "~2x released" |
| typo | 96 | 96 | 1.0 | "EXACT MATCH" |
| association | 102 | 50 | 2.04 | "~2x released" |

`results/edata1_item_count_reconciliation.json -> source_paper` = "RIGOROUS_ANTHROPIC.md sec A.6
(operator transcription of 2607.15495)"; `-> source_data` = "anthropics/jacobian-lens @581d398
data/evaluations/*.json". Its stored `decision`:

> "USE THE RELEASED JSON. It is what the reference implementation consumes and what any replicator
> obtains. Consequence, disclosed: our absolute AUCs are NOT comparable to the paper's reported
> values. The J-vs-logit contrast is internal to our run and unaffected."

### 6.3 Released / scorable / admitted, with pair counts

`results/r4_corrections.json -> R4k_item_counts.per_set`:

| set | `released` | `scorable` | `pairs` | `admitted` |
|---|---|---|---|---|
| multihop | 93 | 93 | 103 | true |
| multilingual | 107 | 107 | 394 | true |
| order-ops | 55 | 55 | 110 | true |
| poetry | 98 | 98 | 98 | true |
| typo | 96 | 96 | 96 | true |
| association | 102 | 92 | 92 | false |

`-> R4k_item_counts.all_six` = `{"released": 551, "scorable": 541, "pairs": 893}`
`-> R4k_item_counts.admitted_five` = `{"released": 449, "scorable": 449, "pairs": 801}`
`-> R4k_item_counts.VERDICT`:

> "t52_factorial.py builds `items` by looping EVAL_SETS, not ADMITTED, so n_items = 541 is 551
> released minus the 10 association items with no scorable intermediate. The scored quantity is the
> admitted five, which hold 449 released items and 801 of the 893 pairs. Calling 541 'the readout
> task' overstates coverage by one whole eval set"

Consistent stored fields elsewhere: `results/e33_logit_baseline_410m_v2.json -> n_items` = `541`;
`results/e48_crossover_410m_rstrip.json -> n_items` = `541`, `-> n_pairs` = `893`;
`results/ladder410/ladder_Github_s0.json -> n_items` = `541`;
`results/e52_factorial_410m_rstrip.json -> controls.R1_C2_readout_divergence.n_released_items` = `551`.

Every `by_N.<N>` block in the ladder files stores all six sets, association included; the five
admitted sets are selected downstream by `ADMITTED`.

---

## 7. The READOUT POSITION, and the stripped/unstripped difference

### 7.1 The anchor's rule, quoted

`jacobian-lens/data/evaluations/README.md`, per set:

- multihop: "`target` defines the readout position only and is not itself scored. Readout is at a
  single position — **the token immediately preceding `target`** — across all layers."
- multilingual: identical sentence.
- order-ops: identical sentence, plus "Each intermediate is a key expanded to a synonym set
  (numbers → digit and word forms; operations → symbol and word forms); rank is the min over
  single-token synonyms at each layer."
- poetry: "Readout is at a single position — **the last newline token (end of line 1 of the
  couplet)** — across all layers."
- association: "Readout is at a single position — **the final prompt token — the closing period** —
  across all layers."
- typo: "Readout is at a single position — **the final prompt token, i.e. the last tokenizer
  fragment of the misspelling** — across all layers."

### 7.2 The implementation of the position

`src/anchor_evals.py:65-69` and `:146-153`:

```python
# Readout position rule per set (README). "last" == final prompt token after rstrip.
_POSITION_RULE = {
    "multihop": "last", "multilingual": "last", "order-ops": "last",
    "typo": "last", "association": "last", "poetry": "last_newline",
}
...
def readout_position(tok, eval_name: str, prompt: str) -> int:
    """Position index to read at, per the anchor's per-set convention."""
    rule = _POSITION_RULE[eval_name]
    if rule == "last":
        return -1
    ids = tok(prompt, add_special_tokens=True).input_ids
    newlines = [i for i, t in enumerate(ids) if "\n" in tok.decode([t])]
    return newlines[-1] if newlines else -1
```

For the five "last" sets the function returns `-1` **without reading `prompt`**; the string handed
to the tokenizer therefore decides which token index `-1` is.

### 7.3 Stripped vs unstripped, stated mechanically

`src/anchor_evals.py:32-34`:

> `.rstrip()` IS LOAD-BEARING: the released prompts end in a trailing space (e.g. multihop's
> "... is the "), and an un-stripped trailing space becomes the readout token and destroys the
> baseline.

- **Stripped**: `p = it["prompt"].rstrip()`; the readout token is the last token of the stripped
  prompt.
- **Unstripped**: `p = it["prompt"]`; the trailing space is tokenised as its own token and becomes
  the readout token.

Call sites, verbatim:

| script:line | expression |
|---|---|
| `src/anchor_evals.py:228` | `prompt = item["prompt"].rstrip()          # load-bearing; see module docstring` |
| `experiments/t17_reaggregate.py:66` | `prompt = item["prompt"].rstrip()` |
| `experiments/t13_bootstrap_controls.py:51`, `t15:89`, `t20:225`, `t22:84`, `t23:93` | `.rstrip()` unconditionally |
| `experiments/trainval.py:253` | `p = it["prompt"]` |
| `experiments/t33_logit_baseline.py:90` | `p = it["prompt"]` |
| `experiments/d1_min_union_diagnostic.py:96` | `p = it["prompt"]` |
| `experiments/t48_crossover.py:234` | `p = it["prompt"].rstrip() if a.rstrip else it["prompt"]` |
| `experiments/t52_factorial.py:366` | `p = it["prompt"].rstrip() if a.rstrip else it["prompt"]` |
| `experiments/t36_qladder.py:312` | `p = it["prompt"].rstrip() if a.rstrip else it["prompt"]` |
| `experiments/e28_ladder_410m.py:102` | `p = it["prompt"].rstrip() if rstrip else it["prompt"]` |
| `experiments/r6_within_source.py:66` | `p = it["prompt"].rstrip() if rstrip else it["prompt"]` |

### 7.4 How many items the two conventions differ on, as stored

`results/e52_factorial_410m_rstrip.json -> controls.R1_C2_readout_divergence`:

| field | value |
|---|---|
| `required_total` | 157 |
| `observed_total` | 157 |
| `observed_per_set` | `{"multihop": 83, "multilingual": 19, "order-ops": 55, "poetry": 0, "typo": 0, "association": 0}` |
| `n_released_items` | 551 |
| `fires` | true |
| `examples.multihop` | `unstripped_token_id 209 (' ')` → `stripped_token_id 253 (' the')` |
| `examples.multilingual` | `unstripped_token_id 209 (' ')` → `stripped_token_id 10863 (' ist')` |
| `examples.order-ops` | `unstripped_token_id 209 (' ')` → `stripped_token_id 426 (' =')` |
| `note` | "computed over ALL released items in all six sets, independently of --rstrip: it describes the difference between the two conventions, not this run's arm." |

`results/e52_factorial_410m_rstrip.json -> readout_convention`:

> "STRIPPED — the anchor rule: readout at the token immediately preceding `target`, i.e. the final
> prompt token after .rstrip(). jacobian-lens/data/evaluations/README.md; src/anchor_evals.py:228"

The same 410M cell under the two conventions
(`results/ladder410_cpu/ladder_Github_s0.json` vs `results/ladder410_cpu_rstrip/ladder_Github_s0.json`,
key path `-> by_N.200.<set>.<agg>`, both `device: cpu`):

| set | agg | unstripped | stripped |
|---|---|---|---|
| multihop | min | 0.07628293974058968 | 0.17891817167401314 |
| multihop | persist | 0.004160887694784573 | 0.02635228873363563 |
| order-ops | min | 0.07922077950622354 | 0.6077922050442014 |
| order-ops | persist | 0.022077922310147966 | 0.35974025779536795 |
| typo | min | 0.2633928582072258 | 0.2633928582072258 |
| typo | persist | 0.07886904890515975 | 0.07886904890515975 |

Grand-mean ratio across the 410M ladder, stored:
`results/r8_ladder_flatness.json -> controls.C2_the_convention_bites_here_too.grand_mean_ratio_stripped_over_unstripped`
= `{"persist": 2.660245168042099, "min": 1.844722194673436}`.

Per-set stripped/unstripped ratios (order-ops 29.8x, multihop 3.13x, multilingual 1.01x,
poetry/typo/association 1.00x) are stated in `tools/readout_exposure.py:12-14` (docstring) and in
`docs/experiments/preregs/R1_grid_rstrip.md:48`; **UNSTORED** — not located in any `results/*.json` by this
pass.

`experiments/t48_crossover.py:44-54` records the state of the call sites at the time it was written,
and which path that experiment declared PRIMARY:

```
TWO SCORING-PATH FORKS, BOTH DISCLOSED AND BOTH RUN — declared before any result
  rstrip    `anchor_evals.py:32-34` states that `.rstrip()` on the prompt is LOAD-BEARING (the
            released prompts end in a trailing space, which otherwise becomes the readout token).
            t22/t17 rstrip; trainval.py and t33_logit_baseline.py DO NOT. Measured here: the
            readout token differs on 157/551 items (multihop 83/93, order-ops 55/55,
            multilingual 19/107). So E28's and E33's entire number set — including the 0.02808
            and 0.02844 that clause (d) leans on — was read unstripped.
            PRIMARY is the UNSTRIPPED path, because clause (d) and control C0 are defined
            against E33's numbers and only the unstripped path can reproduce them. The stripped
            path is run as a disclosed sensitivity arm (`--rstrip`). It is CPU and free. If the
            verdict flips between them, that is reported, not resolved.
```

`docs/experiments/preregs/R1_grid_rstrip.md` (pre-registration, §"WHY IT IS A CORRECTION AND NOT A
SENSITIVITY ARM") states the opposite ordering for the grid experiment:

> **Therefore the stripped arm is the estimate and the unstripped arm is legacy.** There is no
> branch of this experiment in which the paper keeps printing a number computed at a token that
> does not exist in the sequence its own rule refers to.

Both statements are recorded here; this file does not adjudicate between them.

`experiments/t54_aggregation_audit.py:48-53` records a mixing case:

> "2026-08-20: e48 was hardcoded at three read sites while --e52/--e36 were redirectable, so
> `--e52 ..._rstrip --e36 ..._rstrip` produced a file that still read the UNSTRIPPED e48. The
> derangement audit is e48-driven, so its numbers were identical at both readouts by construction
> (104/120 in both) and were then cited as measured convention-independence."

---

## 8. The derangement / shuffle nulls

### 8.1 What `J^shuf` is

Each band layer is given **another band layer's Jacobian**, under a permutation with no fixed
point. The operator's entries, per-layer Frobenius norms and per-layer spectra are those of the
real fitted operator; only the layer↔matrix assignment changes.

### 8.2 The constructions, per implementation

| implementation | file:line | construction |
|---|---|---|
| dict-map derangement | `experiments/t13_transport_controls.py:55-66` | `torch.randperm` rejection-sampled up to 1000 times for no fixed point; **deterministic fallback: rotate by one**. Returns `{layer: layer}`. Used by `t15`, `t17`, `t19`, `t21`. |
| list derangement | `experiments/t48_crossover.py:145-153` | `torch.randperm` in a `while True` loop until `all(p != l ...)`; docstring: "A RANDOM derangement. Not a cyclic shift: adjacent-layer Jacobians are the most similar pair in the band, so a shift-by-one is the weakest member of this control family, and that is exactly what made E33 v1's control fail at 4/6." Draw seeds `7000 + 97*dr`, `--n-derangements` default 5. |
| same, transcribed | `experiments/d1_min_union_diagnostic.py:62-72` | docstring: "Transcribed verbatim from t48_crossover.py so the objects are the SAME derangements." Seeds `7000 + 97*dr`, `--n-derangements` default 3. |
| selectable cyclic / random | `experiments/t33_logit_baseline.py:56-146` | `--derangement {random,cyclic}`; `cyclic` = `perm = BAND[1:] + BAND[:1]`; `random` = rejection-sampled with `--derangement-seed` (default 0). |
| E36 arms | `experiments/t36_qladder.py:114-146, 205-208` | one permutation `derangement(BAND, 7000)`; `JSHUF = {l: JP["Pile-CC"][q] ...}` (built from **Pile-CC's** operator, scored against every corpus's) **and** `JSHUF_OWN = {c: {l: JP[c][q] ...}}` (each corpus deranged against itself). |
| stability sweep | `experiments/t43_derangement_stability.py:12-18` | "scores MANY independent derangements per (model, ...) cell", "n_seeds independent random derangements per cell (rejection-sampled: no layer keeps its own J)". |
| trainval-side | `experiments/trainval.py:150-151` | `--save-lens` exists so "the operator can be reused for controls (e.g. the J^shuf derangement in t33)". |

`experiments/t36_qladder.py:451-455` writes the distinction into the results file, under
`controls.C2_shuf_below_everything.WHY_IT_DOES_NOT_FIRE_AND_WHY_THAT_IS_NOT_LIKE_FOR_LIKE`:

> "this control builds its derangement from PILE-CC's operator and then compares it to EVERY
> corpus's operator (R4d / F-7). 'A derangement beats Github's operator' is therefore a comparison
> between two different corpora's operators, not between an operator and its own derangement. C2b
> below is the like-for-like form."

### 8.3 Related non-derangement nulls in the same control family

| null | file:line | construction |
|---|---|---|
| norm-matched random | `experiments/t33_logit_baseline.py:150-155`; `experiments/t48_crossover.py:385-391` | `M = torch.randn(J.shape, generator=g)`, then `M * (J.norm() / M.norm())`, seed 0 |
| spectral random | `experiments/t13_transport_controls.py:86` | `Q1 diag(S) Q2^T` with `Q1,Q2` random orthogonal from QR, `S` the real singular values |
| rank-1 | `experiments/t13_transport_controls.py:87` | `S[0] * outer(U[:,0], Vh[0,:])` |
| scaled identity | `experiments/t13_transport_controls.py:88` | `eye(d) * (J.norm() / sqrt(d))` |
| identity (`J = I`) | `src/anchor_evals.py:173-177` | `transport_logit()`: "T = I. The logit lens: no fitting, no corpus, no distribution dependence." |
| randomized transformer blocks | `experiments/trainval.py:175-220` | `--randomize-blocks`: blocks re-initialised with `hf._init_weights`, embedding/unembedding/final-norm held real; aborts if the block Frobenius norm is unchanged |

### 8.4 Derangement outcomes as stored, both aggregations

`results/e54_aggregation_audit*.json -> C2_derangement.<agg>`:

| file | agg | `n_draws` | `shuf_beats_jp_paired_by_seed` | `shuf_beats_jp_vs_corpus_mean` | `n_corpora_where_shuf_beats_jp_on_the_mean` | `admissible_as_an_operator_comparator` |
|---|---|---|---|---|---|---|
| `e54_aggregation_audit.json` | persist | 120 | 0 | 0 | 0 | true |
| `e54_aggregation_audit.json` | min | 120 | 104 | 103 | 7 | false |
| `e54_aggregation_audit_rstrip.json` | persist | 120 | 0 | 0 | 0 | true |
| `e54_aggregation_audit_rstrip.json` | min | 120 | 104 | 103 | 7 | false |
| `e54_aggregation_audit_rstrip_v2.json` | persist | 120 | 0 | 0 | 0 | true |
| `e54_aggregation_audit_rstrip_v2.json` | min | 120 | **84** | **84** | 7 | false |

The three files' `provenance.argv` differ in which `e48` file was read: `e54_aggregation_audit.json`
and `e54_aggregation_audit_rstrip.json` read `results/e48_crossover_410m.json` (the `--e48` flag did
not exist / was not passed); `e54_aggregation_audit_rstrip_v2.json` was run with
`--e48 results/e48_crossover_410m_rstrip.json`. `docs/context/AGGREGATION_POLICY.md` cites `104/120`.
Both counts are recorded here; this file does not adjudicate between them.

`results/r9_permutation_calibrated_min.json` scores each corpus's real operator against **its own**
15 derangement draws (`source_file` = `results/e48_crossover_410m_rstrip.json`,
`readout_convention` = "STRIPPED — the anchor rule (R1)"):

| field (`-> by_aggregation.<agg>.<key>`) | `min` | `persist` |
|---|---|---|
| `n_corpora` | 8 | 8 |
| `n_beating_every_own_derangement` | 0 | 8 |
| `median_z` | −0.7057803833531182 | 12.136987784699809 |
| `empirical_p_by_corpus` range | 0.375 (Github) … 1.0 (Wikipedia_en, Pile-CC) | 0.0625 on all 8 |
| `z_by_corpus` range | −1.9874216306081265 (Pile-CC) … +0.593144297659444 (Github) | +7.166770934621293 (Github) … +16.76387293651787 (OOD_CommonPile) |

(`-> controls.C2_aggregations_disagree.min_n_beating` = `0`, `.persist_n_beating` = `8`,
`fires: true`; `-> controls.C1_null_is_the_corpus_own_derangement.n_corpora_checked` = `16`,
`malformed: []`, `fires: true`.)
`results/r9_permutation_calibrated_min.json -> VERDICT`:

> "CALIBRATED. Under the SOURCE'S OWN statistic (`min`), the real operator exceeds every one of its
> 15 own layer-derangement draws on 0 of 8 corpora (median z = -0.71). Under `persist` it does so on
> 8 of 8 (median z = +12.14). The published statistic, read against its own null rather than as a
> raw score, does not certify layer-to-derivative correspondence — and that statement is now made
> WITHOUT selecting the metric on the control it has to pass."

`results/r9_permutation_calibrated_min.json -> declared_bias` records the draw floor: 15 draws give
`p_floor = 0.0625` per corpus (stored per corpus as `p_floor`).

---

## 9. Which results files store which aggregation, and under which key path

**Method.** All 511 `.json` files under `results/` (recursive) were scanned for dictionary keys
literally named `min`, `persist`, `best1L`, `mean`, and the shortest key path recorded for each.
This is a file inventory, not a measurement.

**Raw counts.** Files containing at least one key with that name: `min` 174, `persist` 172,
`best1L` 132, `mean` 148. Files with **no** such key: 336.

**By key-set.** 132 files carry exactly `{min, persist, best1L, mean}`; 25 carry exactly
`{min, persist}`; 18 carry another combination, and in each of those 18 the `mean`/`min` hit is a
**summary-statistic key, not an aggregation** (verified individually; listed at the end of this
section).

**Per aggregation, plainly.** The premise "every results file stores all four" does not hold as a
statement about files: it holds for the four-aggregation scoring functions (§9.1) and not for the
two-aggregation ones (§9.2).

| aggregation | files storing it (of 511 under `results/`) | key-path shapes it lives under |
|---|---|---|
| `min` | 174 (includes 2 files where a key named `min` is a series minimum and no aggregation is present, §9.3) | `.by_N.<N>[.reads].<set>.min` · `.by_transport.<arm>.<set>.min` · `.results.<set>.<arm>.min` · `.rungs.<corpus>.min` · `.arms_admitted_mean.<arm>.min` · `.by_aggregation.min` · `.by_dose.<dose>.min` · `.draws.min` · `.arms.<arm>.min` · `.cells.<cell>.min` · `.by_scale.<scale>.min` · `.ladder.<rung>.arms.<arm>.min` |
| `persist` | 172 | the same shapes as `min`, with `.persist` in place of `.min` |
| `best1L` | 132 | `.by_N.<N>[.reads].<set>.best1L` · `.by_transport.<arm>.<set>.best1L` · `.results.<set>.<arm>.best1L` · `.by_aggregation.best1L` · `.R4c_all_four_aggregations.by_scale.<scale>.best1L` |
| `mean` | 148 (includes 16 files where the `mean` key is a summary statistic, not the aggregation, §9.3) | `.by_N.<N>[.reads].<set>.mean` · `.by_transport.<arm>.<set>.mean` · `.results.<set>.<arm>.mean` · `.by_aggregation.mean` · `.R4c_all_four_aggregations.by_scale.<scale>.mean` |

### 9.1 Files carrying all four aggregations

| file family | count | key path |
|---|---|---|
| `results/ladder410/*.json` | 15 | `.by_N.<N>.<set>.<agg>` |
| `results/ladder410_cpu/*.json` | 15 | `.by_N.<N>.<set>.<agg>` |
| `results/ladder410_cpu_rstrip/*.json` | 15 | `.by_N.<N>.<set>.<agg>` |
| `results/ladder410_cpu_rescore/*.json` | 1 | `.by_N.<N>.<set>.<agg>` |
| `results/ladder1b/*.json` | 15 | `.by_N.<N>.reads.<set>.<agg>` |
| `results/ladder1b_b613/*.json` | 15 | `.by_N.<N>.reads.<set>.<agg>` |
| `results/e48/*.json` | 27 | `.by_N.<N>.reads.<set>.<agg>` |
| `results/e61/*.json` | 10 | `.by_N.<N>.reads.<set>.<agg>` |
| `results/r8_c1_unstripped/*.json` | 1 | `.by_N.<N>.<set>.<agg>` |
| `results/e28_*_reads.json` | 11 | `.results.<set>.<arm>.<agg>` |
| `results/e33_logit_baseline_410m.json`, `..._v2.json` | 2 | `.by_transport.<arm>.<set>.<agg>` |
| `results/t17_reaggregate_{160m,410m}.json` | 2 | `.results.<set>.<arm>.<agg>` |
| `results/t19_skipfirst_410m.json` | 1 | `.results.<skip>.per_set.<set>.<arm>.<agg>` (all four) and `.results.<skip>.n_win.<agg>` (`min`, `best1L`, `persist` only — `experiments/t19_skipfirst_sweep.py:63`) |
| `results/r3_close_d2.json` | 1 | `.by_aggregation.<agg>` |
| `results/r4_corrections.json` | 1 | `.R4c_all_four_aggregations.by_scale.<scale>.<agg>` |

The four-aggregation block is emitted by four scoring functions: the three quoted above —
`experiments/trainval.py:273-283`, `experiments/e28_ladder_410m.py:128-136`,
`experiments/t33_logit_baseline.py:112-122` — and `experiments/t17_reaggregate.py:77-92`, whose
definitions differ as recorded in §1.1 and §2. `experiments/t19_skipfirst_sweep.py:24, 65` imports
and calls `t17_reaggregate.aggregate` with `kfrac=0.5`, so `results/t19_skipfirst_410m.json` is on
the `t17` definitions, not the `trainval` ones.

### 9.2 Files carrying `min` and `persist` only

| file | key path |
|---|---|
| `results/e48_crossover_410m.json`, `results/e48_crossover_410m_rstrip.json` | `.rungs.<corpus>.<agg>`; also `.arms_admitted_mean.<arm>.<agg>` |
| `results/e55_matrix_robustness.json`, `..._rstrip.json` | `.e52_8x8.<agg>` |
| `results/e57_grid_variance_ci.json`, `..._rstrip.json` | `.by_aggregation.<agg>` |
| `results/e57_factorial_cells_410m{,_pooled,_rstrip}.json` | `.draws.<agg>` |
| `results/r5_corpus_axis_uncertainty.json`, `..._rstrip.json` | `.by_aggregation.<agg>` |
| `results/r6_within_source_410m.json`, `..._cpu.json` | `.by_aggregation.<agg>` |
| `results/r7_matched_pools_410m.json` | `.by_aggregation.<agg>` |
| `results/r8_ladder_flatness.json` | `.by_aggregation.<agg>` |
| `results/r9_permutation_calibrated_min.json` | `.by_aggregation.<agg>` |
| `results/r4b_e36_flatness.json` | `.by_aggregation.<agg>` |
| `results/e59_read_dose_410m.json` | `.by_dose.<dose>.<agg>` |
| `results/e49_derangement_stability.json` | `.cells.<model>\|<corpus>.<agg>` |
| `results/d1_min_union_diagnostic_410m.json` | `.arms.<arm>.<agg>` |
| `results/e37_rank_ablation_{70m,160m,410m}*.json` | `.by_rank.<rank>.{logit,jlens,gap}.<agg>`, `.VERDICT.<agg>` |
| `results/e53_ladder_summary.json` | `.by_scale.<scale>.<agg>`, `.cross_scale.<agg>` |
| `results/e66_fitter_equivalence_cuda.json` | `.arms.<arm>.read.<agg>` |
| `results/ladder410_cpu/e53_ladder_summary_cpu.json` | `.by_scale.<scale>.<agg>`, `.cross_scale.<agg>` |

Produced by scoring functions that compute only these two: `experiments/t48_crossover.py:286-291`,
`experiments/t37_rank_ablation.py:145-148`, `experiments/t43_derangement_stability.py:107-110`,
`experiments/d1_min_union_diagnostic.py:149-152`, and the recompute tools
(`tools/r8_ladder_flatness.py`, `tools/r9_*`, `tools/r5_*`, `t53`, `t55`, `t57`, `t59`, `t61`, `t62`).

### 9.3 Files where a `min`/`mean` key is **not** an aggregation

| file | key path | what the key is |
|---|---|---|
| `results/e36_qladder_410m{,_c1arm,_rstrip}.json` | `.ladder.<rung>.arms.<arm>.<agg>.mean` | `min`/`persist` are aggregations; the nested `.mean` is the mean over prefix seeds |
| `results/e52_factorial_410m*.json` | `.by_aggregation.<agg>` (aggregations) and `.per_set_D.<set>.mean` | `.per_set_D.<set>.mean` is the mean over the three prefix seeds of that set's `D`, stored beside `per_prefix_seed` and `sd` |
| `results/e54_aggregation_audit*.json` | `.<block>.<agg>` (aggregations) and `.e36.<agg>.<arm>.mean` | nested `.mean` is a series mean |
| `results/r5_corpus_axis_uncertainty*.json` | `.by_aggregation.<agg>.leave_one_corpus_out.<field>.{min,mean}` | leave-one-out summary statistics |
| `results/e62_band_adjudication.json` | `.asymptotes.band_<lo>_<hi>.<corpus>.mean` | mean over cells; no aggregation key |
| `results/t14_dispersion_ladder.json` | `.collapse_across_models.<frac>.min` | minimum of a numeric series |
| `results/e48_competence_gate_410m.json` | `.instream_ce_range.min` | minimum of a cross-entropy range |

---

## 10. `docs/context/AGGREGATION_POLICY.md` — the operator ruling, quoted

`docs/context/AGGREGATION_POLICY.md` (sha256 `a5b73ff998c5a9c7217f5ace4df13dba192313d9884355b19ba36107895c142a`)
is titled "AGGREGATION POLICY — `min` is primary, `persist` is a labelled secondary" and opens:

> **Ruling 2026-08-20, operator-directed. This supersedes `HANDOFF.md`'s "`persist` primary, `min`
> never votes" wherever the two conflict.** Every number below is stored; nothing here required a
> re-run, because every scoring path in this repository has always emitted both aggregations.

Its "THE RULING" section, verbatim:

> 1. **`min`-over-layers is PRIMARY throughout.** It is the source's own operational definition of
>    recovery — an intermediate counts as recovered if it appears in the top-k at *any* layer,
>    motivated by representations being transient and evolving with depth.
> 2. **`persist` is DEMOTED to a secondary robustness arm, and it carries a label wherever it
>    appears:** *selected with knowledge of the derangement-control outcome.*
> 3. **The permutation-calibrated form of `min` (R9) is what answers the metric question**, not a
>    switch of metric.

Its "WHY" section, verbatim:

> Adopting `persist` because it passed the derangement control, and then citing that control as its
> justification, is **outcome-dependent metric selection**. Both external reviews name it
> independently as circular in the Kriegeskorte sense. No amount of downstream rigour repairs a
> metric chosen on the test it has to pass. `persist`'s "half the band" threshold is itself an
> unjustified researcher degree of freedom, which is a second reason it cannot be the adjudicator.
>
> **What the source does and does not do**, confirmed independently by both reviews. `min` is its
> published detection rule. The source never derives it from a null-calibrated argument, tests
> band-width sensitivity, compares it against persistence/mean/median, permutes the
> layer-to-Jacobian correspondence, or discusses union inflation under decorrelation. Its ablations
> vary **how `J` is computed** — mean vs median, present vs future positions, frozen attention —
> never **how evidence is aggregated across layers**. That axis is genuinely untouched.
>
> **And the source noticed the gap.** At p.33 it counts an item present if its best rank over the
> workspace range is top-25, while noting the number represented at an individual layer is smaller.
> It saw the difference between "any layer" and "a layer" and did not analyse its null behaviour.
> This is not a straw man; it is a question the source raised in passing and left open.

Its "REPORTING RULES", verbatim:

> 1. `min` leads every table, figure and sentence. `persist` appears beside it, never instead of it.
> 2. Every `persist` citation carries the label: *selected with knowledge of the control outcome.*
> 3. **`t(7) = −3.3` is the inferential result; `104/120` is descriptive.** With only 8 clusters, a
>    plot of per-corpus paired differences communicates better than the t-statistic — a Figure 2
>    candidate.
> 4. **Name the estimand for the derangement counts.** `104/120` is *paired by seed*; `103/120` is
>    *against the corpus-mean operator*. Both are in `e54`; a bare number is ambiguous. Every live
>    citation was disambiguated on 2026-08-20.
> 5. The derangement floor used as a *control* in R6/R7 stays anchored on `persist`, and says why:
>    `min` is the aggregation that prefers derangements, so a `min`-anchored floor would fail for the
>    pathology already established rather than for a defect in the operator under test.

Its "WHAT CHANGES, MEASURED" table, reproduced verbatim from the document (the document states
"Corrected readout throughout"; the results paths for each row are in the corresponding
`docs/experiments/descriptions/` file):

| result | `persist` (old primary) | `min` (new primary) | |
|---|---|---|---|
| R1 fit / read | 50.70 / 48.13 | 53.35 / 44.61 | |
| R1 fit − read interval | +2.57 **[−7.60, +9.80]**, includes 0 | **+8.74 [+0.80, +15.87], excludes 0** | **better** |
| E55 independent 5×5 replication | **0/5, DOES NOT REPLICATE** | **5/5, REPLICATES** | **much better** |
| R5 leave-two-out ordering | 16/28 | **20/28** | better |
| R6 between-source share | 0.9723 | **0.9886** | better |
| R4b flatness, S3 read axis | 1/5 → REJECT OVERTURNED | 3/5 → **UNCLEAR** | verdict changes |
| **R8 / S2, N ÷ corpus axis** | 14.2% (inside the 25% bar) | **31.9% (outside it)** | **worse** |

The stored values behind the first two rows,
`results/e57_grid_variance_ci_rstrip.json -> by_aggregation.<agg>` (`readout_convention` field
present in the file):

| key path | `persist` | `min` |
|---|---|---|
| `.by_aggregation.<agg>.point.fit_pct` | 50.697184677169346 | 53.35344735234028 |
| `.by_aggregation.<agg>.point.read_pct` | 48.13211438274321 | 44.60936175509925 |
| `.by_aggregation.<agg>.bootstrap.fit_minus_read.point` | 2.5650702944261354 | 8.744085597241032 |
| `.by_aggregation.<agg>.bootstrap.fit_minus_read.ci_lo` | −7.602188958362582 | 0.796194967922311 |
| `.by_aggregation.<agg>.bootstrap.fit_minus_read.ci_hi` | 9.795918277187646 | 15.872143452423032 |
| `.by_aggregation.<agg>.bootstrap.fit_minus_read.excludes_zero` | false | true |

The same two percentages are also stored at
`results/e54_aggregation_audit_rstrip_v2.json -> matrix_structure.<agg>.{variance_fit_axis_pct,
variance_read_axis_pct}` with identical values.

`results/r8_ladder_flatness.json -> VERDICT`:

> "UNCLEAR — the largest range over N is {'persist': 0.14182325681893931, 'min': 0.3188792203845251}
> of the between-corpus spread, between 25% and 100% under at least one aggregation. Report and
> stop; do not re-cut (CLAUDE.md §2.9)."

`HANDOFF.md` and `docs/context/RESULTS_TAXONOMY.md` also carry aggregation statements; where they
conflict with the above, `docs/context/AGGREGATION_POLICY.md` states in its own first line that it
supersedes `HANDOFF.md`. No further reconciliation is made here.

---

## 11. Quick reference

| term | one-line definition, from code | file:line |
|---|---|---|
| `min` | `(R.min over band layers <= k)`, meaned over pairs, meaned over the 7 `K` | `experiments/trainval.py:275-277` |
| `persist` | `(count of band layers with rank <= k) >= len(band)//2`, meaned over pairs, meaned over `K` | `experiments/trainval.py:281-282` |
| `best1L` | max over band layers of that layer's `K`-mean pair hit rate | `experiments/trainval.py:278-279` |
| `mean` | mean of the hit indicator over all (layer, pair) cells, meaned over `K` | `experiments/trainval.py:280` |
| `K` (ladder family) | `[1, 2, 5, 10, 20, 50, 100]`, unweighted mean, no trapezoid | `experiments/trainval.py:45` |
| `ks` (anchor_evals) | `(1, 2, 5, 10, 25, 100)`, trapezoid in `log k`, normalised | `src/anchor_evals.py:214, 245-250` |
| band | `round(0.38n)…round(0.92n)` ∩ layers strictly below `target = n-2`; 410M `[9,21]`, 1B declared `[6,13]`, 1B used `[5,13]` | `experiments/t53_ladder_summary.py:178-191`; `results/e53_ladder_summary.json -> bands_declared_vs_used` |
| admitted sets | `multihop, multilingual, order-ops, poetry, typo` (association excluded) | `experiments/t48_crossover.py:112`; `results/e33_logit_baseline_410m_v2.json -> excluded_sets` |
| item counts | released 551 / scorable 541 / admitted-five 449 items, 801 of 893 pairs | `results/r4_corrections.json -> R4k_item_counts` |
| readout | five sets: final prompt token (index `-1`); poetry: last newline token | `src/anchor_evals.py:65-69, 146-153` |
| stripped | `it["prompt"].rstrip()`; 157 of 551 items get a different readout token | `src/anchor_evals.py:228`; `results/e52_factorial_410m_rstrip.json -> controls.R1_C2_readout_divergence` |
| `J^shuf` | band layers permuted by a fixed-point-free permutation of the same operator's Jacobians | `experiments/t48_crossover.py:145-153` |
