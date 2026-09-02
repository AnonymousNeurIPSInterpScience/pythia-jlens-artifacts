# E62 — the 1B ladder on the band the paper actually declares

**Output:** `results/e62_interaction_b613.json`, `results/e62_band_adjudication.json`

**PRE-REGISTERED 2026-08-16, before the run. NOT YET RUN — awaiting a GPU box.**

---

## WHY

`paper.tex` §Setup declares the band rule: the anchor's normalized L38–L92 taken literally,
$\mathrm{round}(0.38n)\ldots\mathrm{round}(0.92n)$ intersected with layers strictly below the
target. For pythia-1b ($n{=}16$ layers, target index 14) that is **[6,13]**.

Every `results/ladder1b/*.json` carries **[5,13]** — one layer too low at the bottom edge, measured
and stored in `e53_ladder_summary.json` (`bands_declared_vs_used.1b.matches = false`).

That arm is load-bearing twice over: it carries **Limitation 6's entire replication claim** and
**E51's second scale** (the corpus×concept-set interaction at 1B, 9.16% vs a 2.21% main effect).
The 1B ladder lenses were never retained, so the fix is a refit, not a rescore.

**This does not burn a registered cell.** `PREREG_PYTHIA_T7_v2` §2 names
`pythia-{1b,1.4b,2.8b}-deduped` as confirmatory, and CLAUDE.md §6.8 forbids scoring them without
`--prereg`. But 1b is **already fully observed**: `ladder1b/*.json` scores all six eval sets at
seven values of N across 15 corpus–seed cells. The door is open and this walks back through it on
a corrected band. **1.4b and 2.8b remain untouched and must stay that way.**

## DESIGN

`repro/exp/e62_ladder1b_band.sh` — 15 cells, 5 corpora × 3 seed blocks, on one L40S.

`trainval.py --model 1b --corpus <c> --seed <s> --band 6,13 --n-max 200
--ckpts 10,25,50,75,100,150,200 --dim-batch 128 --save-lens ... --out results/ladder1b_b613/...`

Everything except the band is identical to the run that produced `ladder1b/`. **`--save-lens` is
new and is the point**: had the original run saved its operators, this experiment would be a free
rescore instead of a paid refit. Every future band question at 1B is then free.

## PRIMARY

**The 1B corpus×concept-set interaction share (E51's statistic) recomputed on band [6,13]**, and
the five per-corpus asymptotes.

## DECISION RULE — fixed before the run

Let $I_{[5,13]} = 9.158\%$ be the stored 1B interaction share under `persist` and $I_{[6,13]}$ the
corrected one.

- **CONFIRMS** — $|I_{[6,13]} - I_{[5,13]}| < 2$ percentage points **and** the interaction still
  exceeds the corpus main effect. Limitation 6's claim stands; the band defect was immaterial and
  the paper reports the corrected number with the deviation retired.
- **WEAKENS** — the interaction still exceeds the main effect but moves by ≥2 points. Report the
  corrected number and state that the 1B replication is band-sensitive.
- **REVERSES** — the interaction no longer exceeds the corpus main effect at 1B. Then **the paper's
  second scale does not replicate**, Limitation 6 must be rewritten, and the concept-set
  interaction becomes a 410M-only result.
- **UNCLEAR** — C1 or C2 does not fire.

**Not to be reinterpreted.** REVERSES is a real possible outcome and it costs the paper its only
second-scale replication; that is the risk being bought with $5.

## CONTROLS

- **C1 — the overlap layers must be identical.** J for layers 6–13 is fitted per layer and cannot
  depend on whether layer 5 was also fitted. Refitting with the same corpus, seed and prompt pool
  must reproduce the stored `ladder1b` **per-layer dispersion** for layers 6–13 to ≤1e-3.
  *Number required to fire:* max abs difference ≤ 1e-3 across the 8 shared layers, all 15 cells.
  If it fails, something other than the band changed between the two runs and nothing here is
  comparable.
- **C2 — the band must be what was asked for.** Every output must record `band == [6,...,13]`.
  *Number required:* 15/15.
- **C3 — the prompt pool must be the same.** `n_used` per cell must equal the stored run's.

## DECLARED BIAS

1. **Only the band changes.** Fitter, prompts, seeds, checkpoints and eval battery are held. So
   this isolates the band, and cannot speak to any other difference between the scales.
2. **The 410M arm is not re-run**, because its band [9,21] already matches the declared rule
   (`e53_ladder_summary.json` confirms). The two scales therefore remain on different N grids
   (1B tops out at 200, 410M at 800) — that is a separate disclosed limitation and E62 does not
   fix it.
3. **1b is a confirmatory model whose cells are already spent.** Re-scoring is defensible only
   because of that. If it were unspent this experiment would not be run.

## COST

`repro/20_cost_estimate.sh`: one 1b lens at N=200 on an **L40S** is **0.45 GPU-h / \$0.36**.
15 cells = **6.75 GPU-h**. At the rented price of \$0.736/h that is **\$4.97**.

**Under the \$40 gate; over the 6-hour gate.** Reported to the operator before provisioning and
approved 2026-08-16.

## RESULT

*(unrun)*

## VERDICT

*(pending)*

## ON-BOX GATE DEVIATIONS, recorded before the run (box0, L40S, 2026-08-17)

Two, both measured rather than argued.

**1. `repro/13_vast_preflight.sh` §3 (TF32) failed on `cudnn.allow_tf32`.** Measured on the box:
`matmul.allow_tf32 = False` (the flag that governs this workload) and `cudnn.allow_tf32 = True`.
`NVIDIA_TF32_OVERRIDE=0` is a **driver-level** override and does not change torch's own flags, so
that check cannot pass on default torch. GPT-NeoX has no convolutions, so cudnn is unused here —
but "unused" is an argument, so `trainval.py` now sets **both** flags to `False` explicitly, making
the asserted state the actual state. The decisive check is preflight §4, the measured anchor
fidelity, run on the box: see below.

**2. `tests/test_anchor_fidelity.py` A5 failed on the box: 1 top-1 disagreement (0 locally).**
A1/A2 are exact (`0.00e+00`), A3 is *tighter* than local (`1.34e-03` vs `2.32e-03`), and A6 is
exact at both thread counts. A5's `max|logit diff|` on the box is **4.88e-04**, i.e. smaller than
local — the single rank flip is a near-tie broken the other way at the fp32 floor, not a broken
path. **A5's `== 0` assertion is knife-edge by construction**: it flips whenever two candidate
logits fall within the fp32 floor, so it is a coin flip on hardware, not a stable gate.

Proceeding, because: the disagreement is in the direction of *smaller* logit error; E62's C1
control checks **per-layer dispersion**, which is computed from the operator and not from ranks, so
it independently catches any real numerical problem; and every comparison here is against numbers
produced by the same batched path, so a systematic path difference cancels.

**Corpus integrity:** all five pools were shipped from the laptop rather than rebuilt on the box
(`src/build_corpora.py` pins no dataset revision) and **verified SHA-256-identical to
`corpora/manifest.json`, 5/5**, so the box draws from exactly the pool the operators were fitted on.

## FLAGGED DELTAS

- Teardown obligation: pull + SHA-verify + mirror **before** `./lab down`, per COMPUTE.md §6.
- `--save-lens` writes 15 fp16 1B lenses; they must be hashed into `ARTIFACTS.md` and mirrored,
  or this experiment has to be paid for twice.

---

<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->

## PROVENANCE

Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the
exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no
re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.

| results file | bytes | sha256 (first 16) | produced by | readout |
|---|---:|---|---|---|
| `results/e62_interaction_b613.json` | 28,056 | `d6f9aa7cf1f3ea55` | `t51_interaction_variance.py` | INHERITED |
| `results/e62_band_adjudication.json` | 8,434 | `39820894aca707c6` | `t62_adjudicate_band.py` | INHERITED |
| `results/ladder1b/tv_Github_s0.json` | 12,878 | `70f08c0c04fae13c` | `—` | EXPOSED |
| `results/ladder1b/tv_Github_s1.json` | 12,927 | `afbee942d86a6667` | `—` | EXPOSED |
| `results/ladder1b/tv_Github_s2.json` | 12,914 | `7b5d8b49c9e50247` | `—` | EXPOSED |
| `results/ladder1b/tv_Pile-CC_s0.json` | 12,843 | `cfae7085a49f4e09` | `—` | EXPOSED |
| `results/ladder1b/tv_Pile-CC_s1.json` | 12,894 | `a3b969ef0bcb1fba` | `—` | EXPOSED |
| `results/ladder1b/tv_Pile-CC_s2.json` | 12,909 | `6bbc136053598f58` | `—` | EXPOSED |
| `results/ladder1b/tv_StackExchange_s0.json` | 12,972 | `808c4f56ea2edb5b` | `—` | EXPOSED |
| `results/ladder1b/tv_StackExchange_s1.json` | 12,962 | `fd8eb1c2f96410e1` | `—` | EXPOSED |
| `results/ladder1b/tv_StackExchange_s2.json` | 12,958 | `764a95eb1b6cd2d4` | `—` | EXPOSED |
| `results/ladder1b/tv_USPTO_Backgrounds_s0.json` | 12,974 | `74a7317e4c90f91e` | `—` | EXPOSED |
| `results/ladder1b/tv_USPTO_Backgrounds_s1.json` | 12,976 | `88af3b2e98111782` | `—` | EXPOSED |
| `results/ladder1b/tv_USPTO_Backgrounds_s2.json` | 12,976 | `9a80132d9f3a7543` | `—` | EXPOSED |
| `results/ladder1b/tv_Wikipedia_en_s0.json` | 13,034 | `2afe54c57a5d16c0` | `—` | EXPOSED |
| `results/ladder1b/tv_Wikipedia_en_s1.json` | 13,083 | `07fdb704221ff92b` | `—` | EXPOSED |
| `results/ladder1b/tv_Wikipedia_en_s2.json` | 13,105 | `90b2485e1df2168f` | `—` | EXPOSED |

**Payload checksums** (content only, provenance block excluded):

* `e62_interaction_b613.json` — `a6bab0d8687c9a48bc1041bd536b874b`
* `e62_band_adjudication.json` — `5c112697381b3b5e2a3a0e2783df6928`

<!-- END GENERATED PROVENANCE -->
