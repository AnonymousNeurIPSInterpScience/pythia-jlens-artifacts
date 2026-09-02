# ARTIFACTS.md — model-artifact manifest (discipline #1; see README §4 + §6)

Every `.pt` in the repo is **gitignored** (large; exceeds GitHub's blob limit) and
must be **backed up off the GPU box and SHA-256-verified before any instance is
destroyed**. History: the N=500 2B and N=150 9B lenses were once destroyed with
their boxes. This file is the ground-truth hash registry.

Hashed 2026-07-21. Verify with `shasum -a 256 <path>` (macOS) / `sha256sum` (Linux).

| Artifact | Size | SHA-256 | What it is | Status |
|---|---|---|---|---|

### Pythia ladder (branch `pythia`, added 2026-08-09)

Fitted with the **anchor's own** `jlens` at pinned `581d398`, corpus wikitext-103-raw-v1,
`dim_batch` 64, `max_seq_len` 128, **CPU/fp32**, **uncentered** (μ never enters). Pythia is
**untied** (`embed_in` ≠ `lm_head`), which is the tied-vs-untied control for the Gemma-2
rogue-coordinate finding. Note the port fix: transformers 5.14.1 renamed
`GPTNeoXForCausalLM.embed_out` → `lm_head`, so the anchor's shipped Pythia `Layout` is stale
and an explicit one is supplied (`experiments/t0_smoke.py`).

| Artifact | Size | SHA-256 | What it is | Status |
|---|---|---|---|---|
| `results/lenses/ladder/lens_70m_n200.pt` | 2.6 MB | `28e03a9982b3d6070f27065d19140e5fe93b97d15f1d541dfefcf38853ce0c3f` | **Pythia-70m-deduped J-lens, N=200**, d=512, 5 source layers. Fit 574 s (2.87 s/prompt). Stable rank **3.3–8.6 of 512** — near-rank-5, and unchanged from the N=3 smoke, so this is a property of the averaged Jacobian rather than an unconverged fit. Provenance: `lens_70m_n200_provenance.json`. | **local + HF mirror, SHA-verified 2026-08-09** |
| `results/lenses/ladder/lens_160m_n200.pt` | 13.0 MB | `af0e17b4c9305e0b54c3c538805cbb7fec71e847f676e1001ca1844d935b8143` | **Pythia-160m-deduped J-lens, N=200**, d=768, 11 source layers. Fit 2470 s (12.35 s/prompt). Stable rank **5.0–7.6 of 768**. Provenance: `lens_160m_n200_provenance.json`. | **local + HF mirror, SHA-verified 2026-08-09** |

#### db=128 ladder — THE LENSES EVERY CURRENT RESULT USES (registered 2026-08-10)

The rows above are the superseded `dim_batch=64` fits. **Every T3/T4/T7/T9/T10/T12 number in
`results/` is computed from the `_db128` lenses below**, which went unregistered until the
2026-08-10 adherence audit — a discipline-#1 violation, now closed. All are on the HF mirror.

| Artifact | Size | SHA-256 | What it is | Status |
|---|---|---|---|---|
| `results/lenses/ladder/lens_70m_n200_db128.pt` | 2.6 MB | `bf8de535f1420effec69636094169eb709461a688003af93206b0b220e4309ea` | Pythia-70m-deduped, N=200, db=128, **final** target, 5 source layers | local + HF mirror |
| `results/lenses/ladder/lens_160m_n200_db128.pt` | 13.0 MB | `0022d6297697477dc152ffc6722a29be22464844de8197aa88a9daa3bd4df197` | Pythia-160m-deduped, N=200, db=128, **final**, 11 source layers | local + HF mirror |
| `results/lenses/ladder/lens_410m_n200_db128.pt` | 48.2 MB | `f74f66f5ac1244b030a339508d66b295496a19a4d39470a4c44b9a21bd5147de` | Pythia-410m-deduped, N=200, db=128, **final**, 23 source layers. Source of the exploratory 6/6. | local + HF mirror |
| `results/lenses/ladder/lens_1b_n200_db128_pen.pt` | 117.4 MB | `fccbb2a6520759d5e5043e56b0fc6c6ab8f10204b27d0225b546e73a414957fa` | **Pythia-1b-deduped, N=200, db=128, PENULTIMATE target** — the first lens fitted under `PREREG_PYTHIA_T7_v2` (primary recipe) and **the first lens in the program carrying a Jacobian-dispersion profile** (1.736 → 0.038 over 14 source layers; at layer 0, 40% of the per-prompt magnitude cancels in the average). A100-80GB, fp32, 4288 s (21.44 s/prompt), 14 source layers. Anchor gate `ALL_LAYERS_AGREE=True` **run at the penultimate target** (previously the gate silently validated the library default). Provenance: `lens_1b_n200_db128_pen_provenance.json`. | **local == box, SHA-verified 2026-08-10; HF mirror** |
| `results/lenses/ladder/lens_1b_n200_db128_fin.pt` | 125.8 MB | `1b1f36a46322f6078200fc0caa030be5cc0f0499fa04c8caf82b6ee41ffaf1cf` | pythia-1b-deduped, N=200, db=128, **final** target (PREREG v2 **S5 secondary**), 15 source layers, 4573 s. **Stable rank is ~1.6× LOWER than the penultimate lens at every layer** (17.4 vs 40.7 at L0; 75.2 vs 151.9 at the last) — the evidence that T4's rank ladder may be a target-layer artifact (E13). Gate passed at the same target. | **local == box, SHA-verified 2026-08-11; HF mirror** |
| `results/lenses/ladder/lens_1.4b_n200_db128_pen.pt` | 184.6 MB | `74da881b4e2d9c8b7b88b5b66907cc498150dd15c132a814ff421b6e7614da28` | **pythia-1.4b-deduped, N=200, db=128, PENULTIMATE** (PREREG v2 primary), 22 source layers, 2651 s on H100. Dispersion **2.897 → 0.041** — at layer 0, **49% of the per-prompt magnitude cancels**, higher than 1b's 40%. Gate passed at the same target. | **local == box, SHA-verified 2026-08-11; HF mirror** |
| `results/lenses/ladder/lens_1.4b_n200_db128_fin.pt` | 192.9 MB | `c513a1ee383df6c8393e780ef4ee2a446d809bdc738ecb78dbeb151d9ec627c8` | pythia-1.4b-deduped, N=200, db=128, **final** (S5 secondary), 23 source layers, 2772 s on H100. Gate passed at the same target. | **local == box, SHA-verified 2026-08-11; HF mirror** |

| `results/lenses/ladder/lens_2.8b_n200_db128_pen.pt` | 393.2 MB | `3437c2bcf657a0663e3a9ca9de35bfa2ba0f6e1bcbcd05dcb2d59ce3b0d0e4f9` | **pythia-2.8b-deduped, N=200, db=128, PENULTIMATE** (PREREG v2 primary), 30 source layers, 7050 s (35.25 s/prompt) on H100. Dispersion **4.418 → 0.012**; at layer 0, **57% of per-prompt magnitude cancels** — the highest on the ladder. Peaked at **76.2 GB of 81.5**, clearing the OOM risk at the frozen `dim_batch=128`. Gate passed at the same target. | **local == box, SHA-verified 2026-08-11; HF mirror** |
| `results/lenses/ladder/lens_2.8b_n200_db128_fin.pt` | 406.3 MB | `569395f9f23828eb0347712ebab3124dd25f4cebf4db9160ecfd6ab96ee4c6b6` | pythia-2.8b-deduped, N=200, db=128, **final** (S5), 31 source layers, 7276 s on H100. Dispersion 4.355 → 0.007. Gate passed at the same target. | **local == box, SHA-verified 2026-08-11; HF mirror** |

#### Penultimate backfill — the small rungs (E13 / T14), added 2026-08-11

Fitted so the dispersion ladder and the stable-rank comparison span all six models at **one**
target layer. **Not** part of `PREREG_PYTHIA_T7_v2`'s confirmatory set — these models are burned.

| Artifact | Size | SHA-256 | What it is | Status |
|---|---|---|---|---|
| `results/lenses/ladder/lens_70m_n200_db128_pen.pt` | 2.1 MB | `b53b45cd4f89a5174109f70fcffb7d6bce639093afc9b6dcecb34a88ab71a390` | pythia-70m-deduped, PENULTIMATE, 4 source layers, **14 s**. Dispersion 0.229 → 0.057 — the **lowest** on the ladder, refuting the C0 prediction that small models would be the most dispersed. | **SHA-verified 2026-08-11; HF mirror** |
| `results/lenses/ladder/lens_160m_n200_db128_pen.pt` | 11.8 MB | `65373e75993d2342ba95142055fcaea1c71206b3a4c8faa7de9a8e984cbd5b78` | pythia-160m-deduped, PENULTIMATE, 10 source layers, 89 s. Dispersion 0.641 → 0.064. **Highest stable rank on the ladder (78.0)** at this target. | **SHA-verified 2026-08-11; HF mirror** |
| `results/lenses/ladder/lens_410m_n200_db128_pen.pt` | 46.1 MB | `4bad084ffea3a88708029f5b916bc9bc56a46600532ede1bd0604947ea45872a` | pythia-410m-deduped, PENULTIMATE, 22 source layers, 398 s. Dispersion 2.364 → 0.023 — **higher than 1b despite 2.4× fewer parameters**, the case showing dispersion tracks **depth** not scale. | **SHA-verified 2026-08-11; HF mirror** |

**14 Pythia lenses now on the HF mirror.** The ladder is complete at the penultimate target for all
six models, and at the final target for 70m/160m/410m/1b/1.4b/2.8b.
⧗ Not fitted: 6.9b, 12b (excluded from PREREG v2 as unfundable — see §4 of that document).
⧗ Excluded from PREREG v2 as unfundable on available hardware: 6.9b, 12b — see `PREREG_PYTHIA_T7_v2.md` §4.
T0 smoke lenses (N=3) exist for 70m/160m/410m/1b and are **not citable**.

Per-fit provenance (corpus, N, dim_batch, dtype, wall time, VRAM) lives in the
`fit_provenance.json` alongside each result (e.g. `prior-arc/results/gate1/fit_provenance.json`);
the Pythia line writes `<lens>_provenance.json` beside each `.pt`.

## Standing rule (teardown order — this is where two lenses were lost)
```bash
vastai copy $(cat .instance_id):/workspace/logs/          ./logs/
vastai copy $(cat .instance_id):/workspace/prior-arc/results/   ./prior-arc/results_remote/
shasum -a 256 prior-arc/results_remote/**/*.pt      # verify against this table
vastai destroy instance $(cat .instance_id)   # billing stops HERE only
```

## Third-copy durability — HuggingFace mirror (`${HF_REPO}`, private)
Policy: anything personally trained gets a durable third copy off-laptop/off-box, because
these artifacts have a destruction history.

**STATUS 2026-08-03 — CLOSED. All 10 `.pt` artifacts in the table above are mirrored and
SHA-256-verified (10/10) under *path-preserving* names.** Mirror holds 11 objects / 3.75 GB,
private. Verification compares HF's LFS `sha256` metadata against this table — no download
required.

> ⚠ **Bug found and fixed, 2026-08-03.** The previous procedure uploaded with
> `$(basename $f)`, but **two distinct lenses share the basename `lens_2b_n200.pt`** —
> `prior-arc/results/gate1/…` (`3477c934…`) and `prior-arc/results/e8/…` (`6a1583e5…`). Under the flat
> scheme the second silently overwrote the first, so the E8 cone-formation lens was never
> actually mirrored despite the procedure appearing to succeed. **Always upload with the
> repo-relative path as `path_in_repo`.** The 5 legacy flat objects were deleted after their
> path-preserving copies verified.

```bash
export HF_TOKEN=$(python3 -c "import configparser,os;c=configparser.ConfigParser();\
c.read(os.path.expanduser('~/hf_cache/stored_tokens'));print(c['mechinterp-work']['hf_token'])")
python - <<'PY'
import os, re
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); RID = "${HF_REPO}"
paths = [m.group(1) for m in (re.match(r"\|\s*`([^`]+\.pt)`", l) for l in open("ARTIFACTS.md")) if m]
for p in paths:                      # path_in_repo == repo-relative path: collision-proof
    if os.path.exists(p): api.upload_file(path_or_fileobj=p, path_in_repo=p,
                                          repo_id=RID, repo_type="model")
PY
```
Verify by comparing each sibling's `lfs.sha256` against the table above (see
`HANDOFF.md` §0.3 for the one-shot verification script).

## Third-party lenses (not ours; pinned, not mirrored)

Source repo **`neuronpedia/jacobian-lens`** (MIT), pinned at revision **`a4114d7752d11eb546e6cf372213d7e75526d3a1`** (last modified 2026-07-06). These are **not** re-uploaded to our HF mirror — they are someone else's artifacts; we pin and hash-verify instead. Fetch with `hf_hub_download(..., revision=<sha above>)`.

| file (in that repo) | size | SHA-256 | used for |
|---|---|---|---|

**Verified 2026-08-03:** the 2B reference lens hash recorded in `prior-arc/results/e1_box/e1_operator_reflens_refop.json` (`lens_sha256`) is byte-identical to the file currently served by the pinned revision, so the paper's independently-fitted-lens result is reproducible from source.

✅ **Gap CLOSED 2026-08-08.** `b1_cone_compare_9b.json` records `reference_source` as a *path only* — no hash was captured at run time — so the 9B cross-fitter delta (Δ0.0015) rested on an unverified file identity. Two checks now settle it:

1. **Pinned-revision hash re-verified.** All four lenses we use or plan to use (2B, 9B, 27B, Qwen3.6-27B-n1000) were re-queried against revision `a4114d77…` and every SHA-256 in the table above **matches** the file the revision currently serves.
2. **The 9B file has exactly one version in the repo's history.** The full commit log of `neuronpedia/jacobian-lens` is 9 commits; the 9B lens was uploaded in the 2026-06-17 pair (`343ab3dc`, `4f30bb8c`) and **never modified since** — every later commit touches only `README.md`, `CREDIT.md`, or the Qwen3.6 paths. The B1 9B run's artifact was persisted 2026-08-03, so any download of `main` at run time necessarily returned the same bytes as the pinned revision.

Together these mean the file identity behind Δ0.0015 is determined even though no run-time hash was captured. **Δ0.0015 is citable.** (The general lesson stands: capture the hash at run time — the recovery above only worked because the upstream file happened never to be revised.)

## D1 result JSONs (committed in git; hashes recorded 2026-07-21)
These are git-tracked (git is their integrity record); hashes below for cross-machine verification.

| Result | Size | SHA-256 |
|---|---|---|

Box lens `lens_2b_n200.pt` re-verified against this file on wind-down: **✓ match** (`3477c934…`).
All box result JSONs + logs pulled local before instance teardown.

## Mini-run result JSONs (2026-07-22)
| Result | SHA-256 |
|---|---|

## P-ladder lenses (E27/F12–F17), fitted 2026-08-12 on vast.ai 47516820 (L40S)

42 lenses, all pulled and SHA-256 verified against the box **before** it was destroyed
(42 match / 0 mismatch / 0 missing). Naming: `plad_{model}_{corpus}[_LM|_DOC|_DOCLM]_{half}.pt`
where `LM` = length-matched (`--require-full-window`, every prompt exactly 128 tokens),
`DOC` = document-level split (disjoint articles), `DOCLM` = both.
Hashes are in the fit logs and each lens's `_provenance.json`; the box is gone, so these files
are the only copies. **Mirror before the next teardown.**

### Backfill 2026-08-12 — the 12 length-matched lenses that had no provenance JSON

Found by audit: of 53 `plad_*.pt`, 41 carried a `_provenance.json` with a SHA-256 that
**re-verified clean today (41 match / 0 mismatch)**. Twelve had **no provenance file at all** and
therefore no recorded hash. These are the length-matched (`LM`) corpus-variance fits behind the
410M `R = 3.46` result, and the box that produced them is destroyed, so they are the only copies.

**Read this hash for what it is.** It was computed on 2026-08-12, not at fit time, so it cannot
detect corruption that already occurred — it establishes a baseline from here forward and lets a
mirror be verified. The absence of a provenance file is itself the finding: `experiments/t23_pq_ladder.py`
writes one for the unmatched path and not for the `--require-full-window` path.

| Artifact | Size | SHA-256 |
|---|---|---|
| `results/lenses/plad/plad_410mLM_code_A.pt` | 46 MB | `f30afa8ead88116edf5eb29f691b5583aadd9cd7177b6df62c1353ccdee643c6` |
| `results/lenses/plad/plad_410mLM_code_B.pt` | 46 MB | `0f834d586ad588383c753fb73c8be8105530b419d7bded06d5511dc7f118e5e6` |
| `results/lenses/plad/plad_410mLM_pile_A.pt` | 46 MB | `22543312f585298839b191357f1b934cda2a11b5767677e1d369da8a08b9e19a` |
| `results/lenses/plad/plad_410mLM_pile_B.pt` | 46 MB | `88c415a2d9b914e44758483b1ae450a176a87832cbffbd470b69598093888bfa` |
| `results/lenses/plad/plad_410mLM_wikitext_A.pt` | 46 MB | `d895cafeba61e2bcb3e831054cc23288f7f6af704ba289fa9304fa9d1fa013d4` |
| `results/lenses/plad/plad_410mLM_wikitext_B.pt` | 46 MB | `8ab6958198f5c001301c699364ce3018947eccd7a5311ff8d1e607da626a3ec2` |
| `results/lenses/plad/plad_70mLM_code_A.pt` | 2 MB | `bdb0194df77fdb77c4ae72cc11f723c752ef82947e6c3b42547e47be9d4d23cb` |
| `results/lenses/plad/plad_70mLM_code_B.pt` | 2 MB | `c5205c2cacc78d7653ff4eadd7bae088393e207b03dd100c0ee509247f8c9f9e` |
| `results/lenses/plad/plad_70mLM_pile_A.pt` | 2 MB | `8e3a8a210ebd4ed42b5050a7b55cf7928181c3fe9ae95068fc7a6975c99434d8` |
| `results/lenses/plad/plad_70mLM_pile_B.pt` | 2 MB | `12cbd2c37ac458b26008143fc5c47105c50581e4591dbef20bb4d93e3537832a` |
| `results/lenses/plad/plad_70mLM_wikitext_A.pt` | 2 MB | `2d49641422dfb1efbc05811bd7769537586150da7817ac7313d50e0c49f173d0` |
| `results/lenses/plad/plad_70mLM_wikitext_B.pt` | 2 MB | `a81cb137cef05dc9e38539c91270826da99d2789d207886a1b3a85a8dac8147e` |

**RESOLVED 2026-08-12.** All 53 `plad_*` lenses plus the 2 `t16_robust_aggregation_1b_*`
lenses (55 files, 1.35 GB) and 41 provenance JSONs are now mirrored to
`${HF_REPO}` under `results/`, verified 55/55 present after upload.
The mirror holds **69 pythia lenses** total.

Five local `.pt` are deliberately NOT mirrored: `t0_smoke_{70m,160m,410m,1b}_lens.pt`
(smoke-test artifacts, reproducible in minutes, not results) and `lens_410m_n200_ckpt.pt`
(a resumable checkpoint whose finished lens is already on the mirror).

**Resolved 2026-08-19 (R4g):** E28's 166 fits are pulled, on local disk, and hashed
into the table below. The teardown obligation they were subject to is discharged.
Teardown order is unchanged --- pull, SHA-verify locally, mirror, *then* destroy.

---

## Mirror reorganisation, 2026-08-12 — provenance record

The Hugging Face mirror was renamed and scoped to the current programme.

| | |
|---|---|
| **repo** | `${HF_REPO_OLD}` → **`${HF_REPO}`** |
| **old id** | still resolves — HF keeps a redirect |
| **layout** | `results/…` → **`results/…`** · `corpora/…` → **`corpora/…`** |
| **at HEAD** | 69 Pythia lenses + 6 corpus files. Nothing else. |

Mirror paths no longer carry a `pythia/` prefix, because Pythia is now the whole repo.
Local paths are unchanged (`results/…`). `repro/03_fetch_artifacts.sh` and
`./lab mirror` map between the two explicitly.

### Prior-arc artifacts: UNTRACKED, NOT DESTROYED

Ten files totalling **3.75 GB** are absent from the mirror's current commit and remain in its
git history — the same semantics as `git rm`. They back published claims, and several are the
only surviving copies: `lens_2b_c4.pt` and `lens_2b_n25.pt` are marked *box DESTROYED* above,
and `lens_9b.pt` was itself a refit replacing a lens lost with its box.

### 🔑 PROVENANCE ANCHOR — WITHHELD FOR DOUBLE-BLIND REVIEW

The revision hash at which all ten are present, and the ready-to-run download recipe that used it,
**are removed from this document for the review period.**

They are not removed because the files are sensitive. They are removed because they are from
**separate prior research lines** (two prior arcs), and a pinned revision that reaches
them hands a reviewer a route from this anonymous artifact to separately attributable work. The anonymous
proxy serves one named branch, so the current tree is safe on its own; a published revision pin
would have gone around that. This is the one place in this repository where the artifact surface
leaked past its own proxy, and it is closed here rather than at camera-ready.

**Nothing is destroyed and nothing is unrecoverable.** The hash is held by the authors, the files
remain in the mirror's history with `git rm` semantics, and the anchor is restored to this document
at camera-ready. The inventory below stands, so the claim that these artifacts exist and are
recoverable is still checkable in outline; only the shortcut to fetching them is withheld.

| untracked file | size |
|---|---|

**Verified before running the reorganisation**, by canary: a file uploaded, then deleted in a
later commit, still downloaded successfully at its pre-delete revision on this repo. The
recoverability above is tested, not assumed.

---

## E35 / M1 containment — shard 0 (2026-08-13)

Pulled from `vast-box` (RTX 5080, offer 33735065) and hash-verified **before** teardown; receipt
at `.pull_receipt_vast-box.json`, 2/2 files matched.

| artifact | SHA-256 | size | what it is |
|---|---|---|---|
| `results/e35_containment_shard0.json` | `e1684b2d6fe1c650d573b7fd5568ca7deaea2a88d6942050ac43fff89eceaf1c` | 3.7 KB | containment of 5 Pile components + a random-token control against **5% of the Pythia deduped stream** (shard 0 of 20), k ∈ {8,13,20,32} |
| `results/found_shard0.tgz` | `8e804ce17a7ccec9c91c8839d1a447be723708c7925d49efc65711c0451b6e08` | 13.0 MB | the 48 per-worker found-bitmaps (`.npz`). Keeping these means shard 0 **never has to be re-streamed** — new Q corpora can be merged against them |

**Not mirrored to HF.** Both are small and regenerable in ~15 min (2.5 min download + ~9 min
compute) from `EleutherAI/pile-deduped-pythia-preshuffled` `document-00000-of-00020.bin`. The 601 GB
source shard itself is upstream and was never stored.

⚠ **Do not cite the k-selection in this file.** It reports `k_selected=32`, but the rule was applied
to a candidate set containing **only Pile components — all 100% in-distribution** — so it ranked
redundancy, not exposure. See `PREREG_E36_QLADDER.md`.md`.
Also: `tokens_streamed` reads 0 because the merge pass does not stream; actual coverage is
15,000,000,000 tokens across 48 workers.


## E62 — the 1B ladder refit on the declared band [6,13] (2026-08-17)

Fitted on vast.ai instance 47910871 (L40S, $0.817/h), 15 cells, 6.75 GPU-h. Pulled and
hash-verified before teardown (`.pull_receipt_box0.json`: 443 files, 443 match, 0 mismatch);
instance destroyed and absence confirmed against `vastai show instances`.

**Why these exist.** `results/ladder1b/*.json` was fitted on band [5,13]; the paper's own
L38–L92 rule gives [6,13] at 16 layers with a penultimate target. That arm carries the paper's
only second-scale replication, so it had to be right. The original run passed no `--save-lens`,
which is why this was a paid refit rather than a free rescore — fixed here permanently.

**Adjudication: CONFIRMS.** Interaction 9.19% on [6,13] against 9.16% on [5,13]; per-layer
dispersion on the eight shared layers reproduces the original fits to 3.7e-05 over 840
comparisons (`results/e62_band_adjudication.json`).

| file | size | sha256 | what it is | status |
|---|---|---|---|---|
| `results/ladder1b_b613/lens_Github_1b_n200_s0.pt` | 67.1 MB | `459f997d6661ff07494e3ac0829864e02d71b15e0609875a705e032371a30158` | **E62 pythia-1b-deduped J-lens, N=200, band [6,13]** — Github, seed block 0. Refit of the 1B ladder on the band the paper's own L38–L92 rule gives; the stored `ladder1b/` run used [5,13] and retained no operator. `--save-lens` added so every future 1B band question is a rescore, not a refit. | **local == box, SHA-verified 2026-08-17; HF mirror** |
| `results/ladder1b_b613/lens_Github_1b_n200_s1.pt` | 67.1 MB | `187bd291fc66a6f3e5cf9a35c4640d9d963e1fd47434124dc3e7b7832f1a7950` | **E62 pythia-1b-deduped J-lens, N=200, band [6,13]** — Github, seed block 1. Refit of the 1B ladder on the band the paper's own L38–L92 rule gives; the stored `ladder1b/` run used [5,13] and retained no operator. `--save-lens` added so every future 1B band question is a rescore, not a refit. | **local == box, SHA-verified 2026-08-17; HF mirror** |
| `results/ladder1b_b613/lens_Github_1b_n200_s2.pt` | 67.1 MB | `e37d288145a6438be3c571ad3fac769b7fd314c00675573dc8e4e9b3812a2754` | **E62 pythia-1b-deduped J-lens, N=200, band [6,13]** — Github, seed block 2. Refit of the 1B ladder on the band the paper's own L38–L92 rule gives; the stored `ladder1b/` run used [5,13] and retained no operator. `--save-lens` added so every future 1B band question is a rescore, not a refit. | **local == box, SHA-verified 2026-08-17; HF mirror** |
| `results/ladder1b_b613/lens_Pile-CC_1b_n200_s0.pt` | 67.1 MB | `3bf4ce26087e107fa74da45922f6ecc486d6eab6fb29b7fbbf9b356a0ca4eabd` | **E62 pythia-1b-deduped J-lens, N=200, band [6,13]** — Pile-CC, seed block 0. Refit of the 1B ladder on the band the paper's own L38–L92 rule gives; the stored `ladder1b/` run used [5,13] and retained no operator. `--save-lens` added so every future 1B band question is a rescore, not a refit. | **local == box, SHA-verified 2026-08-17; HF mirror** |
| `results/ladder1b_b613/lens_Pile-CC_1b_n200_s1.pt` | 67.1 MB | `f41713739a1f278fd0d1c5b5c23db6ecfe1df0991945aa6b83ad716200e9a8f0` | **E62 pythia-1b-deduped J-lens, N=200, band [6,13]** — Pile-CC, seed block 1. Refit of the 1B ladder on the band the paper's own L38–L92 rule gives; the stored `ladder1b/` run used [5,13] and retained no operator. `--save-lens` added so every future 1B band question is a rescore, not a refit. | **local == box, SHA-verified 2026-08-17; HF mirror** |
| `results/ladder1b_b613/lens_Pile-CC_1b_n200_s2.pt` | 67.1 MB | `fe7a6a5b4579d5f47f0119d943efc700eb70220fb16acdec55c1f724dcde0ac2` | **E62 pythia-1b-deduped J-lens, N=200, band [6,13]** — Pile-CC, seed block 2. Refit of the 1B ladder on the band the paper's own L38–L92 rule gives; the stored `ladder1b/` run used [5,13] and retained no operator. `--save-lens` added so every future 1B band question is a rescore, not a refit. | **local == box, SHA-verified 2026-08-17; HF mirror** |
| `results/ladder1b_b613/lens_StackExchange_1b_n200_s0.pt` | 67.1 MB | `5048cd087976e256a7ff9f30d8066dd95d442229f8c2b0ff5f4d273eef512c5b` | **E62 pythia-1b-deduped J-lens, N=200, band [6,13]** — StackExchange, seed block 0. Refit of the 1B ladder on the band the paper's own L38–L92 rule gives; the stored `ladder1b/` run used [5,13] and retained no operator. `--save-lens` added so every future 1B band question is a rescore, not a refit. | **local == box, SHA-verified 2026-08-17; HF mirror** |
| `results/ladder1b_b613/lens_StackExchange_1b_n200_s1.pt` | 67.1 MB | `905a7e26d7af75d7e9a64a49ba5319156b0e220236c0bf8ec5c15d8989db2330` | **E62 pythia-1b-deduped J-lens, N=200, band [6,13]** — StackExchange, seed block 1. Refit of the 1B ladder on the band the paper's own L38–L92 rule gives; the stored `ladder1b/` run used [5,13] and retained no operator. `--save-lens` added so every future 1B band question is a rescore, not a refit. | **local == box, SHA-verified 2026-08-17; HF mirror** |
| `results/ladder1b_b613/lens_StackExchange_1b_n200_s2.pt` | 67.1 MB | `78e10c625c615a13dc475fef35688ce8a90b8ab368ec84e1c8c7f88d70d3a040` | **E62 pythia-1b-deduped J-lens, N=200, band [6,13]** — StackExchange, seed block 2. Refit of the 1B ladder on the band the paper's own L38–L92 rule gives; the stored `ladder1b/` run used [5,13] and retained no operator. `--save-lens` added so every future 1B band question is a rescore, not a refit. | **local == box, SHA-verified 2026-08-17; HF mirror** |
| `results/ladder1b_b613/lens_USPTO_Backgrounds_1b_n200_s0.pt` | 67.1 MB | `4acda201f3fbc46f77e96d3f6f6840ab6cc53b8d80f694ec5080dab7321b054a` | **E62 pythia-1b-deduped J-lens, N=200, band [6,13]** — USPTO_Backgrounds, seed block 0. Refit of the 1B ladder on the band the paper's own L38–L92 rule gives; the stored `ladder1b/` run used [5,13] and retained no operator. `--save-lens` added so every future 1B band question is a rescore, not a refit. | **local == box, SHA-verified 2026-08-17; HF mirror** |
| `results/ladder1b_b613/lens_USPTO_Backgrounds_1b_n200_s1.pt` | 67.1 MB | `4be9599eae66c8895479b2d0151fde7975929aedc420916b1b29d7415234eac1` | **E62 pythia-1b-deduped J-lens, N=200, band [6,13]** — USPTO_Backgrounds, seed block 1. Refit of the 1B ladder on the band the paper's own L38–L92 rule gives; the stored `ladder1b/` run used [5,13] and retained no operator. `--save-lens` added so every future 1B band question is a rescore, not a refit. | **local == box, SHA-verified 2026-08-17; HF mirror** |
| `results/ladder1b_b613/lens_USPTO_Backgrounds_1b_n200_s2.pt` | 67.1 MB | `0f730745e88e5dae78b5f73b98a00ab55c28ed32dae8b7800ba8823445e090ff` | **E62 pythia-1b-deduped J-lens, N=200, band [6,13]** — USPTO_Backgrounds, seed block 2. Refit of the 1B ladder on the band the paper's own L38–L92 rule gives; the stored `ladder1b/` run used [5,13] and retained no operator. `--save-lens` added so every future 1B band question is a rescore, not a refit. | **local == box, SHA-verified 2026-08-17; HF mirror** |
| `results/ladder1b_b613/lens_Wikipedia_en_1b_n200_s0.pt` | 67.1 MB | `f60c8ee8186e4d53e9a0195b5f0535506160d2d313b3bc174d3129b80d9426b2` | **E62 pythia-1b-deduped J-lens, N=200, band [6,13]** — Wikipedia_en, seed block 0. Refit of the 1B ladder on the band the paper's own L38–L92 rule gives; the stored `ladder1b/` run used [5,13] and retained no operator. `--save-lens` added so every future 1B band question is a rescore, not a refit. | **local == box, SHA-verified 2026-08-17; HF mirror** |
| `results/ladder1b_b613/lens_Wikipedia_en_1b_n200_s1.pt` | 67.1 MB | `25a07e77788297cc89a4603c0a1771d9cf2f8580a481c015bf77d039adcd0464` | **E62 pythia-1b-deduped J-lens, N=200, band [6,13]** — Wikipedia_en, seed block 1. Refit of the 1B ladder on the band the paper's own L38–L92 rule gives; the stored `ladder1b/` run used [5,13] and retained no operator. `--save-lens` added so every future 1B band question is a rescore, not a refit. | **local == box, SHA-verified 2026-08-17; HF mirror** |
| `results/ladder1b_b613/lens_Wikipedia_en_1b_n200_s2.pt` | 67.1 MB | `89838efaec6ce3b2f991ed142b320fd28c8f608e5205a8c934e0ac44675dd0fa` | **E62 pythia-1b-deduped J-lens, N=200, band [6,13]** — Wikipedia_en, seed block 2. Refit of the 1B ladder on the band the paper's own L38–L92 rule gives; the stored `ladder1b/` run used [5,13] and retained no operator. `--save-lens` added so every future 1B band question is a rescore, not a refit. | **local == box, SHA-verified 2026-08-17; HF mirror** |

## E28 read-ladder operators (166) — added by R4g

Recorded here because **0 of 166 were named in this file** while the paragraph above
asserted they were still on rented boxes. They are on local disk. `corpus` is the
**backfilled** value: every sidecar recorded the `--corpus` default `wikitext`, and the
true corpus came from `corpus_file` (M-3).

| artifact | sha256 | corpus | N | seed |
|---|---|---|---|---|
| `e28_Github_410m_n100_s0.pt` | `985f6d7aede6680ac4921866b6071fa89e4a72c9b15b494298f5cf10f3df909b` | Github | 100 | 0 |
| `e28_Github_410m_n100_s1.pt` | `1733c165a94867d64c3bcc7fcb2388461b8aabfdf7700112823a2c3466b34e29` | Github | 100 | 1 |
| `e28_Github_410m_n100_s2.pt` | `ac24e4cf9728827f63e99b145bd0291f72aa43ceaea363c90a78cdb967082536` | Github | 100 | 2 |
| `e28_Github_410m_n200_s0.pt` | `3895668dce09dae91c6300f6adbba2de8359c63bf45fd5676d554f6ed90e9dc8` | Github | 200 | 0 |
| `e28_Github_410m_n200_s1.pt` | `ca08e40ea0973004e8c21790e59334ea3fa6839be08e4753570c8ccfa3784283` | Github | 200 | 1 |
| `e28_Github_410m_n200_s2.pt` | `fabb7cc8e515078c224375f23949ba336c33a4f86f35bf3e96df6cfbdba97bfc` | Github | 200 | 2 |
| `e28_Github_410m_n25_s0.pt` | `5b229a4f060278dd2c95c9ce8a9db97396385907c0742f4f0bc14ea782e62ba2` | Github | 25 | 0 |
| `e28_Github_410m_n25_s1.pt` | `7436c9a8282f2c2ff743271fb84efafdb1c78de5449110ed77020b1c849096d0` | Github | 25 | 1 |
| `e28_Github_410m_n25_s2.pt` | `ed2d4e34a916474f0db2c3e5eaebbcf8b7c1d19901e73846533ffc6d7feb9005` | Github | 25 | 2 |
| `e28_Github_410m_n400_s0.pt` | `4c87ed9a5f778ef0280a17ff0df5e623e8732fb7b706936fcc7caf624979511a` | Github | 400 | 0 |
| `e28_Github_410m_n400_s1.pt` | `a5c4f4c5a773a337be04a0269f0d75746b3286a64ff1a6908abe8067626f6f4b` | Github | 400 | 1 |
| `e28_Github_410m_n400_s2.pt` | `d3bff4c7693cbc54d171edb7fc4affd07abc8ca7a75982d25773a79d153d9140` | Github | 400 | 2 |
| `e28_Github_410m_n50_s0.pt` | `31b20e81de8508703f25faf1111fe4db0c1820f0189dcf1b7be0d8c1a9430280` | Github | 50 | 0 |
| `e28_Github_410m_n50_s1.pt` | `b9b85e54f1e56ab1f30f28f9aaa0a6424b180e9f5756ce0bc903d915e27be209` | Github | 50 | 1 |
| `e28_Github_410m_n50_s2.pt` | `551df46a3acdc70eb26d222db228bbea280b15cdfb5e9e3534d97e294342469e` | Github | 50 | 2 |
| `e28_Github_410m_n800_s0.pt` | `e1667fe86a60a0cecdf86f7204c3f9734fd1e94f92b15df0159bb3194930ccc2` | Github | 800 | 0 |
| `e28_Github_410m_n800_s1.pt` | `da8d4fbac1f4c3251cbe2e24251e8dd12bd6cc50db1c1a640c344c6a9d4abd76` | Github | 800 | 1 |
| `e28_Github_410m_n800_s2.pt` | `448959b64da304821a20fb59e5ee485bf6185efb9394e1d2f1ffd75ee825a505` | Github | 800 | 2 |
| `e28_Github_70m_n100_s0.pt` | `7cbae5ce45c68678d6484f81907307fd79f16cb3914548dbb3dcf18224a66ae6` | Github | 100 | 0 |
| `e28_Github_70m_n100_s1.pt` | `019c76267d59a4c58a993ba26f69bdf4a707d5874be4f7eb219e439cfd9d344a` | Github | 100 | 1 |
| `e28_Github_70m_n100_s2.pt` | `b4ef9329e51dcd511a33f909aae382071abda3821f9b4e01d1af4732f9724197` | Github | 100 | 2 |
| `e28_Github_70m_n200_s0.pt` | `39342f80a279888a8a04cc096969cd05e387afe2a5b7ae9c90b6a03df0d906b6` | Github | 200 | 0 |
| `e28_Github_70m_n200_s1.pt` | `d483e10ac99485d770ed5c7ec3b3704664fc6d8fbae582afb2968819f87f9417` | Github | 200 | 1 |
| `e28_Github_70m_n200_s2.pt` | `6404ec1c9bdd298426724313253d2300a5950026aeaa7ff5dddd5e090c2d3dac` | Github | 200 | 2 |
| `e28_Github_70m_n25_s0.pt` | `0cd9589c619ef6be3d7bce67bc371185f732b8972dc59a9d05a5318bb0981cb5` | Github | 25 | 0 |
| `e28_Github_70m_n25_s1.pt` | `977fee0bb6a6d13a53037b14dab0c366d8e3243416d38940dbac346226a22847` | Github | 25 | 1 |
| `e28_Github_70m_n25_s2.pt` | `2131a7b899d2562c83fbf87bbbe633f48322b71ddc8c5ece93ad3d5785e83965` | Github | 25 | 2 |
| `e28_Github_70m_n400_s0.pt` | `ed6ae13e4867a77af882f2eca7913b78e4dc00b030716cb0aae4dc64c5908b7a` | Github | 400 | 0 |
| `e28_Github_70m_n400_s1.pt` | `4cfc88ad4ae8063c5cc821f24b4acf9c2f0dd42c03e50f6c2d2086b1d6c3ad82` | Github | 400 | 1 |
| `e28_Github_70m_n400_s2.pt` | `c4d15b0f8320df9a459cfb88bf43df3c45cba27dcc471810c599b488d0da9d6f` | Github | 400 | 2 |
| `e28_Github_70m_n50_s0.pt` | `120b35b89571b816d3c1e199a9bd9da978a8cf569c19042656bd28eb1e996b10` | Github | 50 | 0 |
| `e28_Github_70m_n50_s1.pt` | `5b831f1444b7cd8a4f9afc49b4b2ac70effadcc31d24baab983b66fcd66110f4` | Github | 50 | 1 |
| `e28_Github_70m_n50_s2.pt` | `6e3192762cfb069f17a261abadbc9d1e73b2088ab5fe8ce80e4ae87947f78dbd` | Github | 50 | 2 |
| `e28_Github_70m_n800_s0.pt` | `56a1ccb5f2f437f158e0376944f74aa5e6a270535c4667af655e9783e145ce89` | Github | 800 | 0 |
| `e28_Github_70m_n800_s1.pt` | `5a69469ee71f893c26b08795aacadae6a4b48c3757b0296736d7e13c683c5a33` | Github | 800 | 1 |
| `e28_Github_70m_n800_s2.pt` | `793f38bfd6b5ce319189af994612d9bf6b141603ef4746a114a677c05c443dd7` | Github | 800 | 2 |
| `e28_Pile-CC_410m_n100_s0.pt` | `7592ccff0fe57aac7d2fa9b694d0ed6df64d590c681ce7a19a9b05e38d2fc02b` | Pile-CC | 100 | 0 |
| `e28_Pile-CC_410m_n100_s1.pt` | `cbd295aa65cbce257e7fd5670cf0e40ee0d10902af3ab74937fc70edc37654f2` | Pile-CC | 100 | 1 |
| `e28_Pile-CC_410m_n100_s2.pt` | `474269538b22f5b1bc84fe7ec0cbcccefa81efb2913a6537226846864e23aa27` | Pile-CC | 100 | 2 |
| `e28_Pile-CC_410m_n200_s0.pt` | `666e119f8cdcc6b218c020862e3de834c6920dcbed894121a108defa90877325` | Pile-CC | 200 | 0 |
| `e28_Pile-CC_410m_n200_s1.pt` | `6c8e7e8900cdc01e54107cd6d4d786d12d199a7c50110f29606ea3ddbe64e0bc` | Pile-CC | 200 | 1 |
| `e28_Pile-CC_410m_n200_s2.pt` | `fb4fc80ba2990f27e9343260491cfafbdb1d3bcaf9abc79cfc66b51a5fd38037` | Pile-CC | 200 | 2 |
| `e28_Pile-CC_410m_n25_s0.pt` | `1745452027ba4e8bf117b5c261080000e145240126e432dcffd7278ad522c9f9` | Pile-CC | 25 | 0 |
| `e28_Pile-CC_410m_n25_s1.pt` | `a2017712b8cd2247cd421bb25e95cd0a6a248973ce4f07ff440e7679fe0b08b9` | Pile-CC | 25 | 1 |
| `e28_Pile-CC_410m_n25_s2.pt` | `2e6ab007579ede452eff34d350c419b76e39a0c456db3f4137a041a1f36dd090` | Pile-CC | 25 | 2 |
| `e28_Pile-CC_410m_n400_s0.pt` | `4754d23b9e6eb16ac8159661cc393328374d8cc8d286bbb9c971ee95166fca62` | Pile-CC | 400 | 0 |
| `e28_Pile-CC_410m_n400_s1.pt` | `31e9a962ee48623d7cf0ed646f4a59119c2a9c6eca02e4e4c809b5ef7659a4dd` | Pile-CC | 400 | 1 |
| `e28_Pile-CC_410m_n400_s2.pt` | `a53750f0a8695fa3ddcd996a2be068f1dcabf986a9ae76ec8949c3fcb5097197` | Pile-CC | 400 | 2 |
| `e28_Pile-CC_410m_n50_s0.pt` | `c76f2b9fb699bcf929b34ebe1279d413d651d33f6fb689584a1dadd64b5ca03c` | Pile-CC | 50 | 0 |
| `e28_Pile-CC_410m_n50_s1.pt` | `4b9ca06df6d95d1b3fbb858527b178a25bfd74f5126808d87cf9a024523f9bda` | Pile-CC | 50 | 1 |
| `e28_Pile-CC_410m_n50_s2.pt` | `e8e5b9cbb43aab61486d074c2ea49cb2b6f618d74dd55a04716102ef76c4cf5d` | Pile-CC | 50 | 2 |
| `e28_Pile-CC_70m_n100_s0.pt` | `7ede4b79ebc17a69258af76df58082c6fd9c11449c70b1b2a5d4e382f7565fa0` | Pile-CC | 100 | 0 |
| `e28_Pile-CC_70m_n100_s1.pt` | `8a82d874ea184b7517eadcca719de154b062e4c59b5af44c159b38cb7af93f36` | Pile-CC | 100 | 1 |
| `e28_Pile-CC_70m_n100_s2.pt` | `2539133caa26f816a0e515ddbd9445a0cc25291b48454f55a9d63381aee328b1` | Pile-CC | 100 | 2 |
| `e28_Pile-CC_70m_n200_s0.pt` | `116203e39ec6e3ded97c6a73eced00fd5dba5167f9768ec391b0f8fde2b5cc54` | Pile-CC | 200 | 0 |
| `e28_Pile-CC_70m_n200_s1.pt` | `fb5030d9e08d326066082498ab79a57d63300759302278effd06f7836868363a` | Pile-CC | 200 | 1 |
| `e28_Pile-CC_70m_n200_s2.pt` | `54058be9e1be2d69022b6fd2e195c6f54a67f3448da9027e34d0bb40ee45c83a` | Pile-CC | 200 | 2 |
| `e28_Pile-CC_70m_n25_s0.pt` | `eab6ad784d04a60a3136cd353c13fcdc53a74f3532fd5070a2a39968514b97dc` | Pile-CC | 25 | 0 |
| `e28_Pile-CC_70m_n25_s1.pt` | `a50f1a44aae9ea253a2f4398efaab6eab18e703230ecb1f132cb2814cc920138` | Pile-CC | 25 | 1 |
| `e28_Pile-CC_70m_n25_s2.pt` | `aa51d2307b90c4714003208a6b71b8999766b3f07f3097daac89ed9176b61dcb` | Pile-CC | 25 | 2 |
| `e28_Pile-CC_70m_n400_s0.pt` | `cfd9f23e4f0cfd7eb8cb0377910c9623c6aac0719d83941411a7f5cae31f5b79` | Pile-CC | 400 | 0 |
| `e28_Pile-CC_70m_n400_s1.pt` | `153570cb3400f928ff3d46be8f57e970e3c8cce02bcb7340be3ff13b652a04f6` | Pile-CC | 400 | 1 |
| `e28_Pile-CC_70m_n400_s2.pt` | `e0fc84c5ae790396e6f2d38342dc4c167144891fa1527912a666807fc4ccf4f9` | Pile-CC | 400 | 2 |
| `e28_Pile-CC_70m_n50_s0.pt` | `58672012ff8f98f94a440d51b8821552d8533092e95d4220191d8f7f6eea3398` | Pile-CC | 50 | 0 |
| `e28_Pile-CC_70m_n50_s1.pt` | `ae020623e076bde097c5ff93d129b93fadb5f5027ace3e70da21f04588fe74f4` | Pile-CC | 50 | 1 |
| `e28_Pile-CC_70m_n50_s2.pt` | `faf52c6c09fdc0d0a4bdac4f2e212509663d9177690b243345f0883ccd71674c` | Pile-CC | 50 | 2 |
| `e28_StackExchange_410m_n100_s0.pt` | `947a4c69d53ceb80445f452147a531512731b616c9f99ee4d4889d10d8a33f89` | StackExchange | 100 | 0 |
| `e28_StackExchange_410m_n100_s1.pt` | `f3a026b1da50466405ffd4d7ad480aec1597e02f1442b7e9320b0f450f8ec9b5` | StackExchange | 100 | 1 |
| `e28_StackExchange_410m_n100_s2.pt` | `4a2f90dcddc685307ddf94d0671a67bc70983e411aee7e518643cb6dc4336262` | StackExchange | 100 | 2 |
| `e28_StackExchange_410m_n200_s0.pt` | `fba4f9a51cbd39b523043bea1286f8efc61ec72df542dc224a419c0e2e0a3a9f` | StackExchange | 200 | 0 |
| `e28_StackExchange_410m_n200_s1.pt` | `7552cfc76417a4b50971284df5944f731e579c196137b8c32a3ae1c2d949d582` | StackExchange | 200 | 1 |
| `e28_StackExchange_410m_n200_s2.pt` | `20e1867f9f0c7a5528f7b115ea67479c8875ff496ca35ee2323576e2aae81a63` | StackExchange | 200 | 2 |
| `e28_StackExchange_410m_n25_s0.pt` | `b85c0ebf98b77ffa4a8cd5cb7558e82cdece403f65662e78cb94d6c0ff8806a3` | StackExchange | 25 | 0 |
| `e28_StackExchange_410m_n25_s1.pt` | `2125ffb5668dd2c12e455a8b82321206229b82b77d9fd428ad2a6f08cb59a2d3` | StackExchange | 25 | 1 |
| `e28_StackExchange_410m_n25_s2.pt` | `969cd4e7ae1261962c71eafda0cfcd29cf8bf58f9462bae0fe7c1e767a0e71a8` | StackExchange | 25 | 2 |
| `e28_StackExchange_410m_n400_s0.pt` | `3c3bbbca13ef5891fb19ce2c552334ebf12263c905f25545bc36a831b4bd2a00` | StackExchange | 400 | 0 |
| `e28_StackExchange_410m_n400_s1.pt` | `35924fd85fb01d40f761f59711f0a3e968d3f6bfdcd59a83000cb51888a6ce26` | StackExchange | 400 | 1 |
| `e28_StackExchange_410m_n400_s2.pt` | `eb267e4d8e622d0fa6bd4039cff285dc01f57a121cf99a7c4c1247d3d68b0c43` | StackExchange | 400 | 2 |
| `e28_StackExchange_410m_n50_s0.pt` | `0b59b1cc3e164aa7ee10f647c45e8c466c0fbdf190064039c5d2740e4a3c0323` | StackExchange | 50 | 0 |
| `e28_StackExchange_410m_n50_s1.pt` | `bfe59d22b735a6c975651fa751a0496e470337655b5af5692d1947c7c1eb1c49` | StackExchange | 50 | 1 |
| `e28_StackExchange_410m_n50_s2.pt` | `54e88c7f45291fbb4056efc5b69049cb997c3c86aa61889931f6bbc060511e5d` | StackExchange | 50 | 2 |
| `e28_StackExchange_410m_n800_s0.pt` | `8daeaf9ab1d0e38698a5bcb0452791802cbd82b35a7173609ed0f669a8f4aa7b` | StackExchange | 800 | 0 |
| `e28_StackExchange_410m_n800_s1.pt` | `0df04165a7304f0c91270fee186ef7f3a9d1985e161903acce3ffcc5b1eaadbe` | StackExchange | 800 | 1 |
| `e28_StackExchange_410m_n800_s2.pt` | `5c4fc7abbe22780a9a87b67464bc4eb76000a8767be000a5a3034e0a8dcc0425` | StackExchange | 800 | 2 |
| `e28_StackExchange_70m_n100_s0.pt` | `6af7ced13a0cabe7d11a7534cd4b4353ac797c91a99c12c597d18a82339a46ff` | StackExchange | 100 | 0 |
| `e28_StackExchange_70m_n100_s1.pt` | `c9540618f31a8c1f59da2ec570bb6645c8267585d4b664898b816c914cd79782` | StackExchange | 100 | 1 |
| `e28_StackExchange_70m_n100_s2.pt` | `b41cf8049c9aedc60bea9640ea68f8af57829dc648d1251e47dcd003c4749d57` | StackExchange | 100 | 2 |
| `e28_StackExchange_70m_n200_s0.pt` | `39b04ea2223f0288ddb706d85180c76938f23f2b586fbe4a90d9d1189c9c1a04` | StackExchange | 200 | 0 |
| `e28_StackExchange_70m_n200_s1.pt` | `894433e6c8294b3bec6ccfca49a2f29a2b3550380c90a698539b92f115a7dd21` | StackExchange | 200 | 1 |
| `e28_StackExchange_70m_n200_s2.pt` | `d8eec1900568e7f1392a93607de7babc4b4b3b73a791860265a6e01cfbeac60f` | StackExchange | 200 | 2 |
| `e28_StackExchange_70m_n25_s0.pt` | `df32a3544d11b4b0e44393f92d41d65a5a663c24d02d151459d7defab0a1d83b` | StackExchange | 25 | 0 |
| `e28_StackExchange_70m_n25_s1.pt` | `8057cdb51ba6346fac90ec6e914382a792cae57db9eeaf7b606a4be6e1c1a501` | StackExchange | 25 | 1 |
| `e28_StackExchange_70m_n25_s2.pt` | `bac3cae3f219841b65dfd31705f8b5e9ec38a2104cd26fe8bb12edd6b4401f03` | StackExchange | 25 | 2 |
| `e28_StackExchange_70m_n400_s0.pt` | `d13dd95f375ae43663f35ae06db99627030a090d827e2b4fb6806003006d7ec8` | StackExchange | 400 | 0 |
| `e28_StackExchange_70m_n400_s1.pt` | `58ad9fc98ed1823b12921b5fc30a222a80b1ca91ee036fb01f8d35955cd1521c` | StackExchange | 400 | 1 |
| `e28_StackExchange_70m_n400_s2.pt` | `8ed1b087708fa07fc5f83308e4a13104827891494f46f6262a43b138fd2c8542` | StackExchange | 400 | 2 |
| `e28_StackExchange_70m_n50_s0.pt` | `cba6014f368782ffe274fbf359aeafb5850b9f5b349709a38f581637f7faee89` | StackExchange | 50 | 0 |
| `e28_StackExchange_70m_n50_s1.pt` | `c48d88b881f4a63e1a9e78983f11626c723ea1bdb66879a34998fe2bd2153774` | StackExchange | 50 | 1 |
| `e28_StackExchange_70m_n50_s2.pt` | `746f1b29f48b998c2b88ef6b302afb5d617fec1e9e0791da3efdfd92b352878f` | StackExchange | 50 | 2 |
| `e28_StackExchange_70m_n800_s0.pt` | `8b1bd5a6391796d8f2c853a013cbca76e4174b8bbf03a36d2c41d1c32acc74dc` | StackExchange | 800 | 0 |
| `e28_StackExchange_70m_n800_s1.pt` | `557f07a3c09ad86da08a7ee2cc71ddea139056c7b2e17262431b42745a7ae710` | StackExchange | 800 | 1 |
| `e28_StackExchange_70m_n800_s2.pt` | `ec32cd8954e8d39f7bfb9bf74bfea6fec52baef31d0462399e0d47d183c95c5a` | StackExchange | 800 | 2 |
| `e28_USPTO_Backgrounds_410m_n100_s0.pt` | `257c8c60058eafbfb51b97fb1fee2e97b05910316d887fc65b33457d68700a29` | USPTO_Backgrounds | 100 | 0 |
| `e28_USPTO_Backgrounds_410m_n100_s1.pt` | `4826bb37268b663e21b44e65ff7b6cfda452612034020814245a1cc806007d44` | USPTO_Backgrounds | 100 | 1 |
| `e28_USPTO_Backgrounds_410m_n100_s2.pt` | `8142513be6cf969377e2b9671f7a23a1130fea92bcfcb13c0c3a2fcdca6ece83` | USPTO_Backgrounds | 100 | 2 |
| `e28_USPTO_Backgrounds_410m_n200_s0.pt` | `6f8f06ad3ce7ef6c3a0dc44afa63838df505ac8cd1617b25fda0aa9747a31524` | USPTO_Backgrounds | 200 | 0 |
| `e28_USPTO_Backgrounds_410m_n200_s1.pt` | `2a2a3f4524ac4db804501e412d4ebc188935058687f1a468e2b5bfe0cdfbbfcd` | USPTO_Backgrounds | 200 | 1 |
| `e28_USPTO_Backgrounds_410m_n200_s2.pt` | `a1e5dfd9f19145467500b542c74c1103bd90a44d430e4e3528aca22cdb589922` | USPTO_Backgrounds | 200 | 2 |
| `e28_USPTO_Backgrounds_410m_n25_s0.pt` | `d070039c4d7a38016e6cb2d50e302368b1a3207ed87f1f4bb534e789e70890d5` | USPTO_Backgrounds | 25 | 0 |
| `e28_USPTO_Backgrounds_410m_n25_s1.pt` | `207874b5cc9417c1b2c8c8dca5ac4393d4bf0a696c3fd6b4dcc97b5f65de3d88` | USPTO_Backgrounds | 25 | 1 |
| `e28_USPTO_Backgrounds_410m_n25_s2.pt` | `d7fc83d478f118f991670728226f22b31a88356400cb8f5ac7d1ddb9c44560df` | USPTO_Backgrounds | 25 | 2 |
| `e28_USPTO_Backgrounds_410m_n400_s0.pt` | `4bc59371a760c85c5570f68e00a6e41a65a1ff7168079fc9ab5a695e2d70987b` | USPTO_Backgrounds | 400 | 0 |
| `e28_USPTO_Backgrounds_410m_n400_s1.pt` | `ec859ffabba4ee9fd77e271d3468a6a48543e460059a20372a417472f28984d6` | USPTO_Backgrounds | 400 | 1 |
| `e28_USPTO_Backgrounds_410m_n400_s2.pt` | `f8ac3e2549cda84008e03a18d1cb7033688247b61df83eb101519832580aa2bd` | USPTO_Backgrounds | 400 | 2 |
| `e28_USPTO_Backgrounds_410m_n50_s0.pt` | `5da1a5918c276fbaf3534076f6ba77954be94549d894c15badc78bf830ce5399` | USPTO_Backgrounds | 50 | 0 |
| `e28_USPTO_Backgrounds_410m_n50_s1.pt` | `02e2b60567d2a6578ce15cec0d0c1bcde20fc7831b6df4df0914c15c944d5f70` | USPTO_Backgrounds | 50 | 1 |
| `e28_USPTO_Backgrounds_410m_n50_s2.pt` | `36e82a189185caa8b3b0d97605c5118bb554b516f78a2fd919c9dfc51acd2ef6` | USPTO_Backgrounds | 50 | 2 |
| `e28_USPTO_Backgrounds_70m_n100_s0.pt` | `4dce47af9d07fca3c7ea3ab5f28f95f5d1d2403a9cd43d6daff2181f6268eff3` | USPTO_Backgrounds | 100 | 0 |
| `e28_USPTO_Backgrounds_70m_n100_s1.pt` | `55269aaec6ca8db12a9d7767688d397003dbde3b5b74fcc469d51696e0175bd0` | USPTO_Backgrounds | 100 | 1 |
| `e28_USPTO_Backgrounds_70m_n100_s2.pt` | `30dbd33314dc2e0adc3fcc0b59b9c2f012cbf45afaea5a409e4270633083edce` | USPTO_Backgrounds | 100 | 2 |
| `e28_USPTO_Backgrounds_70m_n200_s0.pt` | `39a7873e0c4d9609a6aec275305586c7508f7f049b0430c5b04e5d58bc2b6657` | USPTO_Backgrounds | 200 | 0 |
| `e28_USPTO_Backgrounds_70m_n200_s1.pt` | `9cd2b8d183898a85123f5c3bba871bb07147a3cab16ea56d08397f3894b2f278` | USPTO_Backgrounds | 200 | 1 |
| `e28_USPTO_Backgrounds_70m_n200_s2.pt` | `58a24bc75783b4c955660a48dab07303177ebb283ed9371e25a09a06acacb4d1` | USPTO_Backgrounds | 200 | 2 |
| `e28_USPTO_Backgrounds_70m_n25_s0.pt` | `eda0a62dc263c99a2b0a0d068acbe049a1a9461092db0ca10016d07c05bd8bb0` | USPTO_Backgrounds | 25 | 0 |
| `e28_USPTO_Backgrounds_70m_n25_s1.pt` | `cbdaa281c2cf024ab8ed65fb5c056ff5c36a00931288b7d7fe31bbc5b1e29e54` | USPTO_Backgrounds | 25 | 1 |
| `e28_USPTO_Backgrounds_70m_n25_s2.pt` | `87f0395b1b967ca436bb3b601317f478b3a60eaf78f7a4d3fe50c45fb1287eb5` | USPTO_Backgrounds | 25 | 2 |
| `e28_USPTO_Backgrounds_70m_n400_s0.pt` | `574ae9e09c17e469c14104999d66fe98b86491cb4d4d7b44c969ebd5d0f8a246` | USPTO_Backgrounds | 400 | 0 |
| `e28_USPTO_Backgrounds_70m_n400_s1.pt` | `a573e328a5f61071be015d456fd6ff84fbbdaa15825538bb58f4f002dd28d056` | USPTO_Backgrounds | 400 | 1 |
| `e28_USPTO_Backgrounds_70m_n400_s2.pt` | `db924e59c27de003785f19abc31810c70613d1d1d924044bd3dea6baaf484ec0` | USPTO_Backgrounds | 400 | 2 |
| `e28_USPTO_Backgrounds_70m_n50_s0.pt` | `4e22172c87840636861d0e9376dd37f0434fdc7e3db35f25d331321c9e6de695` | USPTO_Backgrounds | 50 | 0 |
| `e28_USPTO_Backgrounds_70m_n50_s1.pt` | `2578682de58f4808d7a2786f0cabe96d9686bd55f4fd4a0ead55e2755e0c9d1f` | USPTO_Backgrounds | 50 | 1 |
| `e28_USPTO_Backgrounds_70m_n50_s2.pt` | `795a74b02d7b3590fcbbd89f2cdc55eb36676e8785bc4d4923ded1cf359a2b29` | USPTO_Backgrounds | 50 | 2 |
| `e28_Wikipedia_en_410m_n100_s0.pt` | `48e9a4ef08fc7636956b2b70791e71ce128c3a52ee2560b21a0109f3ca302c38` | Wikipedia_en | 100 | 0 |
| `e28_Wikipedia_en_410m_n100_s1.pt` | `e29405b8043a5564eabbadf48f310a05e6cf758c060201d6d33395b28e60892a` | Wikipedia_en | 100 | 1 |
| `e28_Wikipedia_en_410m_n100_s2.pt` | `1187f5d000b701566a366acc23a44da8a44f9030dd1844782fdb976429a34ace` | Wikipedia_en | 100 | 2 |
| `e28_Wikipedia_en_410m_n200_s0.pt` | `efcf8b669825016233690a19162ff645dc58ed2e2628a2e113f26713e27cf998` | Wikipedia_en | 200 | 0 |
| `e28_Wikipedia_en_410m_n200_s1.pt` | `bfa5aee55a338b05ccb9fc7d66a631da83b97c6e48cfa387105689bd30805033` | Wikipedia_en | 200 | 1 |
| `e28_Wikipedia_en_410m_n200_s2.pt` | `33d97ade0445171e2329dc86da6ef93a218f178e3390f182e511c237ca2f1ed6` | Wikipedia_en | 200 | 2 |
| `e28_Wikipedia_en_410m_n25_s0.pt` | `88bc80e8f4ca4c18f0053c96d5e73c80418ffa2583a78d76e430eb5d72f4a80f` | Wikipedia_en | 25 | 0 |
| `e28_Wikipedia_en_410m_n25_s1.pt` | `640063636e75a733a5220793dbfe6bfd266319f701f1bfbebc8439dcaf37049f` | Wikipedia_en | 25 | 1 |
| `e28_Wikipedia_en_410m_n25_s2.pt` | `d13827efcdc248d16ec92e4fb117d48a1b776f93bd8fb46fdd7b4bd569f01a03` | Wikipedia_en | 25 | 2 |
| `e28_Wikipedia_en_410m_n400_s0.pt` | `5e2af80098d34c57e6cdbfcf680a9c9303e18daae91b1577d3e9222e7eb4952b` | Wikipedia_en | 400 | 0 |
| `e28_Wikipedia_en_410m_n400_s1.pt` | `6e7c60a21498755c793c8d3206a0d805ae5348572d7a51ce393c1574bf218217` | Wikipedia_en | 400 | 1 |
| `e28_Wikipedia_en_410m_n400_s2.pt` | `8eb1664fa377f21fc868166697c03a31592782ba9c0eac0fc67ecf52b357a01b` | Wikipedia_en | 400 | 2 |
| `e28_Wikipedia_en_410m_n50_s0.pt` | `60d160e9c92c8679c2a25f99a4c47381fe48c930294d70420349be74acca805f` | Wikipedia_en | 50 | 0 |
| `e28_Wikipedia_en_410m_n50_s1.pt` | `ae4513372f15515fbc7e5ef33371d5a19ea0358fc036532684dbfb25fd0caf3d` | Wikipedia_en | 50 | 1 |
| `e28_Wikipedia_en_410m_n50_s2.pt` | `32dea8de66bf5edaa80b24e76761715815dd2b0a6874b85d0ceb45b22f2e4dd4` | Wikipedia_en | 50 | 2 |
| `e28_Wikipedia_en_410m_n800_s0.pt` | `6244533db3ed6740e6cc1640ef7ce71a7c85f3834047ab097b73284880f8adcd` | Wikipedia_en | 800 | 0 |
| `e28_Wikipedia_en_410m_n800_s1.pt` | `22e1d8eb49c6f6d9fd7cbf370f128b907d9f77ae87f36680055469c0e4bebb11` | Wikipedia_en | 800 | 1 |
| `e28_Wikipedia_en_70m_n100_s0.pt` | `791dae2001d54780df16c24e528c2d37b8850b3ae18636bc049456368ab949b0` | Wikipedia_en | 100 | 0 |
| `e28_Wikipedia_en_70m_n100_s1.pt` | `d0045fd42d96e2569b8cd600dfd69658ac9905473d047bc585427308dfbbc104` | Wikipedia_en | 100 | 1 |
| `e28_Wikipedia_en_70m_n100_s2.pt` | `81ce6b3178708e6cc0a3cf26ba46743db50113b28b8ac02ec33e5d815267b209` | Wikipedia_en | 100 | 2 |
| `e28_Wikipedia_en_70m_n200_s0.pt` | `6b8eca1a4c0cc2137b3b83b22ccf3542629097011045603dc45d898bb2c8e62a` | Wikipedia_en | 200 | 0 |
| `e28_Wikipedia_en_70m_n200_s1.pt` | `d5ff2d95106e40a28ab80792cca129bdfa56976900f1c5dc59dbb1770c69857d` | Wikipedia_en | 200 | 1 |
| `e28_Wikipedia_en_70m_n200_s2.pt` | `c10f206bb41dd5ff2eda75cebf8c1aa25a978ae3520ff6b65b7546a92221b45f` | Wikipedia_en | 200 | 2 |
| `e28_Wikipedia_en_70m_n25_s0.pt` | `f2a6fefbcb065aa660edb02278cda05fa0f108ea57727b9d62f008f016b378ba` | Wikipedia_en | 25 | 0 |
| `e28_Wikipedia_en_70m_n25_s1.pt` | `7e6181c3cf9fc15032ca046ae60cae559eefce70b5ef9b0b2ca03b4fc131cff4` | Wikipedia_en | 25 | 1 |
| `e28_Wikipedia_en_70m_n25_s2.pt` | `19a317acca9852e9b50c041e1b2b509520e66083315806138a264f319f68f3b8` | Wikipedia_en | 25 | 2 |
| `e28_Wikipedia_en_70m_n400_s0.pt` | `63de5057134684e0fa7d1829e63e4a24ce8693d58d837800b13e2d3135b60d7c` | Wikipedia_en | 400 | 0 |
| `e28_Wikipedia_en_70m_n400_s1.pt` | `c6ea7428b4c500f7da5c7c3a1a79cd080a3b197a1d10420c9e00a61a425e8d8d` | Wikipedia_en | 400 | 1 |
| `e28_Wikipedia_en_70m_n400_s2.pt` | `ee7c003f2a4f5303f8450d25a1a99f15ac6c2220ff674157011b065c451b385b` | Wikipedia_en | 400 | 2 |
| `e28_Wikipedia_en_70m_n50_s0.pt` | `47c7d8884935cfa16add972a8ebe08ee6267ea5d62ef4d02877de42692e4f2ec` | Wikipedia_en | 50 | 0 |
| `e28_Wikipedia_en_70m_n50_s1.pt` | `0bbda35c32b666a608cb6177a52a238c86f9e9ac08db38c0372d28ce09fa0546` | Wikipedia_en | 50 | 1 |
| `e28_Wikipedia_en_70m_n50_s2.pt` | `cd01517b76d37e7affe7890621a4ae1c5b18bc126641c74dd6d0faa527b24153` | Wikipedia_en | 50 | 2 |
| `e28_Wikipedia_en_70m_n800_s0.pt` | `331d3030a6652c2b4960eb5abb29bc378c0a39e727913f4a421f402c3a8758b0` | Wikipedia_en | 800 | 0 |
| `e28_Wikipedia_en_70m_n800_s1.pt` | `1a858ad5c6a133a98cd1702bdc807089ddd9225c12058d3b6208bedb342eb176` | Wikipedia_en | 800 | 1 |

## R6 within-source operators (8) — added 2026-08-20

Fitted on a rented RTX 4090 for R6 (`docs/experiments/preregs/R6_within_source_resampling.md`):
four **disjoint** 200-document samples from each of two sources. Pulled and SHA-verified
against the box before it was destroyed (8/8 identical), per `CLAUDE.md` §1.

| artifact | sha256 | bytes |
|---|---|---|
| `lens_R6_Pile-CC_b0_410m_n200.pt` | `33dae3b0d750d08e1cc7236e477e442e503d99f669a0f3ad0cfa4ef490e272e8` | 27268171 |
| `lens_R6_Pile-CC_b1_410m_n200.pt` | `10e7a7af7110bb0515adc8cc8b27c638c7cb040aa47c3224280a1b53db0e46d9` | 27268171 |
| `lens_R6_Pile-CC_b2_410m_n200.pt` | `8fc8cc67463f391d6e0e5fbe83230263309c13c7babeb27b5331c79588432e75` | 27268171 |
| `lens_R6_Pile-CC_b3_410m_n200.pt` | `4ebb1fce54b6c476f46cb5b38067b98af5f7bc85331f042471bbc7e5efab9b66` | 27268171 |
| `lens_R6_Wikipedia_en_b0_410m_n200.pt` | `b9bd821d39b93b9353433600c62f0b7c0711dc02d8c3622db934e90f66829886` | 27268266 |
| `lens_R6_Wikipedia_en_b1_410m_n200.pt` | `b24b4903c5ef156ae1a08afda45f6dc3ff89451d1ffe03c0350b9382ce7d0d27` | 27268266 |
| `lens_R6_Wikipedia_en_b2_410m_n200.pt` | `51c9fde7e588d29c445bfe2b9067da68ad018609f362de42790d3ded25a4a529` | 27268266 |
| `lens_R6_Wikipedia_en_b3_410m_n200.pt` | `49ad6f5c39d632a4c2b0bc96c10362b3a6f3168ad1ee9b3889a3f0b694bbc55f` | 27268266 |

## R7 length/format-matched operators (30) — added 2026-08-20

Fitted on rented RTX 4090 `48212997` for R7 (`docs/experiments/preregs/R7_length_matched_pools.md`):
five in-stream corpora x three seeds, in a **matched** and an **unmatched** arm.

Pulled and SHA-verified **30/30** against the box's own `/workspace/r7.sha256`, which was
confirmed byte-identical to `results/r7_box_sha256.txt` before scoring. Receipt:
`.pull_receipt_vast-box.json` (31 verified / 0 mismatch / 0 missing, including
`r7_matched_pools_410m.json` at `be98f49b...`).

**The transfer failed twice before it succeeded**, and the failure mode is worth recording:
the `vast-box` alias in `~/.ssh/config` pointed at `<redacted-vast-ssh-endpoint>`, which timed out during
banner exchange on every new connection while the box remained reachable at
`root@<redacted-box-ip>:<redacted-port>`. A whole-directory `tar cf - r7` retry re-streams every file from the
start, so each retry transiently truncates copies that were already intact — file **presence** was
never evidence of file **integrity** here (`CLAUDE.md` §6.0b). Two files were observed full-size
with a wrong hash mid-rewrite and both resolved on completion. Verify against the manifest, never
against `ls`.

| artifact | sha256 | bytes |
|---|---|---|
| `lens_R7_matched_Github_s0_410m_n200.pt` | `cbf143643e1ccded5f111e45a44abc3ad3987234b91ef0df2c638bbe0de926f9` | 27268304 |
| `lens_R7_matched_Github_s1_410m_n200.pt` | `2b81e8e18dc6b6bbc2a0296ceaf2fa0bf1f612b19af2e6fd4f4cf09584fa5d97` | 27268304 |
| `lens_R7_matched_Github_s2_410m_n200.pt` | `c15445f647b3d0d233d75d290bdffe12cacc29bbd1cc0d1268d4c3289cbd7bfd` | 27268304 |
| `lens_R7_matched_Pile-CC_s0_410m_n200.pt` | `257d2fe462d5fad6c0b9544b4eb04ef67f58de7a155f76a133267502901fb307` | 27268323 |
| `lens_R7_matched_Pile-CC_s1_410m_n200.pt` | `e9eb00a20a026ee835e8ef29c6c11c7ba417a5efc910f9a98efd221adef5275b` | 27268323 |
| `lens_R7_matched_Pile-CC_s2_410m_n200.pt` | `42fd87602328b1931e2d2827d30f71ec250ea93455134d9f6a6a46be1fd31d08` | 27268323 |
| `lens_R7_matched_StackExchange_s0_410m_n200.pt` | `9452d30c8a399af5eb90367fb6877358ca1527ab2401d3a25895419d353ef404` | 27268437 |
| `lens_R7_matched_StackExchange_s1_410m_n200.pt` | `be0042ae80e91ea9ab3928877389f37bc6d19c170d7076d642e2001de6853d32` | 27268437 |
| `lens_R7_matched_StackExchange_s2_410m_n200.pt` | `1c2cacc90d6fbf7a38d84d3d372793d070aea548b2e5a659f632e74a04bb2c4e` | 27268437 |
| `lens_R7_matched_USPTO_Backgrounds_s0_410m_n200.pt` | `d7cb96c7c790c8b260720756a2067e1b44e4192f4e39dc9ccd58069c764ee96a` | 27268513 |
| `lens_R7_matched_USPTO_Backgrounds_s1_410m_n200.pt` | `9df86067524293ae11dc6a849f6b79bc7dd0a5863aaadf184a0bee17ea6e5ca1` | 27268513 |
| `lens_R7_matched_USPTO_Backgrounds_s2_410m_n200.pt` | `dc23ef148a56c0463a7f88af418c6ea04a403a0a6f33b452dff7ed1cd1a5c7ff` | 27268513 |
| `lens_R7_matched_Wikipedia_en_s0_410m_n200.pt` | `c2f6153d750dceb248d1050adccb792c74b8137d129eeb27a236d5a7288f19a1` | 27268418 |
| `lens_R7_matched_Wikipedia_en_s1_410m_n200.pt` | `da38a7badcbf45eed064963864befa803da85551dc40e4d679a996ecbab86301` | 27268418 |
| `lens_R7_matched_Wikipedia_en_s2_410m_n200.pt` | `2e87a8aaec3fc4abdda5ea33e417a1303deb3e8ce9135d51d796d8f6800ffd88` | 27268418 |
| `lens_R7_unmatched_Github_s0_410m_n200.pt` | `49a278ac0cf1b26aaaa8834e6638e968f90204077b22b3a5f67b3ff98e48d495` | 27268342 |
| `lens_R7_unmatched_Github_s1_410m_n200.pt` | `d4714d69a2f8c6f5e9e8d32e21f5f3af476e6054ddb24279e38a8128968a9e07` | 27268342 |
| `lens_R7_unmatched_Github_s2_410m_n200.pt` | `958112fcc4d90dce85e1beefa815354694f7a2ffd7d9e1d058e3c2b948a9c44e` | 27268342 |
| `lens_R7_unmatched_Pile-CC_s0_410m_n200.pt` | `3d90dab194cdc620eeed5533f99e0c4908fc98b5622d2252db05979807f64220` | 27268361 |
| `lens_R7_unmatched_Pile-CC_s1_410m_n200.pt` | `3d996d6da417511ca26a0c3a8c87ad5cb58c29a24d64dc60928c930951ac7985` | 27268361 |
| `lens_R7_unmatched_Pile-CC_s2_410m_n200.pt` | `37cf8cb50602c1d319df7061e2c5b584977628a60ba6e0556c6d86432cc0364f` | 27268361 |
| `lens_R7_unmatched_StackExchange_s0_410m_n200.pt` | `efe02b31dc3c5cb83a10f732921bd60b1a60f6baa7c69a0827615e20dc74d065` | 27268475 |
| `lens_R7_unmatched_StackExchange_s1_410m_n200.pt` | `77279ef0d58e134f86899f2062055c8d850493222bd0cf7b4c112494b2108df0` | 27268475 |
| `lens_R7_unmatched_StackExchange_s2_410m_n200.pt` | `a308cb623d4fee979c3fbd19f0cdec9e3589dae1d1f3f804c690a72eaf7f7c82` | 27268475 |
| `lens_R7_unmatched_USPTO_Backgrounds_s0_410m_n200.pt` | `961a6e747fa37130ca039fab945b659e593c43b1907e1832498860a575907efb` | 27268551 |
| `lens_R7_unmatched_USPTO_Backgrounds_s1_410m_n200.pt` | `3267710f269fea1caa1d646e1935f3ca60af9bc2d79b63737ba7228d263ca97e` | 27268551 |
| `lens_R7_unmatched_USPTO_Backgrounds_s2_410m_n200.pt` | `906b3c272fe52e2abb9818e69e5293f731d988e238df48e0949401add764f413` | 27268551 |
| `lens_R7_unmatched_Wikipedia_en_s0_410m_n200.pt` | `1067b44c08b7f59ac25b19a6f59933c5fd89b36f8fc78e2df578a63fcf1c7876` | 27268456 |
| `lens_R7_unmatched_Wikipedia_en_s1_410m_n200.pt` | `cee4952b3a420dcfbd807048cd685917811f53bb6973fc9d5839e3a17aa8e1af` | 27268456 |
| `lens_R7_unmatched_Wikipedia_en_s2_410m_n200.pt` | `b4538e53b11faa1a6942130a73981a805a92cd35aba5693a88270ec59c7a74b0` | 27268456 |


---

## CV6 — the per-family ladder, 2026-08-24

Pulled from the H100 NVL and hash-verified **before** teardown: `.pull_receipt_box0.json` records
**35 files, 35 match, 0 mismatch, 0 missing**. Produced by `experiments/cv6_per_family_ladder.py`
at commit `518d192`, fp32, TF32 off and measured off (fp32 matmul rel err 1.155e-06).

### Cached `h_t` — a re-score needs no forward passes

| artifact | size | sha256 |
|---|---:|---|
| `results/cv6/ht_cache_1.4b.pt` | 45.6 MB | `a3d5f7ecfb9c8bacbea1822275e28d3faf73940426c0456183b3c52fb4d83095` |
| `results/cv6/ht_cache_2.8b.pt` | 79.0 MB | `c4cabaca7f8ca0d38312ce257855d1e8870c1429a2f1c56e088608056881da37` |

### The 30 fitted operators

`results/cv6/lenses/lens_INSTREAM_<corpus>_<model>_n25_s<seed>.pt` — 5 in-stream corpora x
{1.4b, 2.8b} x 3 disjoint seed blocks, **4.82 GB** total. Stored **fp16**, the `results/e48/`
convention; CV6's PRIMARY scores the **fp32** accumulator, and the fp16 round-trip is recorded
separately as a diagnostic (worst `|d z_mean|` **1.171e-04**, ordering unchanged 10/10 cells).

| artifact | sha256 |
|---|---|
| `results/cv6/lenses/lens_INSTREAM_Github_1.4b_n25_s0.pt` | `5c3df30f285ca6a2632ddf5ccda1c0bbeb72b288167df1f5bf63e48aadfa3f68` |
| `results/cv6/lenses/lens_INSTREAM_Github_1.4b_n25_s1.pt` | `3aade78c84bf70764a74a54018f1fa39966756618d5b74522194dc89e684d834` |
| `results/cv6/lenses/lens_INSTREAM_Github_1.4b_n25_s2.pt` | `3b406658637da8ba432429bb262353698d1118b7efb531cb955328a95d5dbf01` |
| `results/cv6/lenses/lens_INSTREAM_Github_2.8b_n25_s0.pt` | `84f242a1abcd7293a4472f5a98fd7b5f3d047cae90890a31aa562dfb94decf5c` |
| `results/cv6/lenses/lens_INSTREAM_Github_2.8b_n25_s1.pt` | `5f2d16e6011c4f2bdc865b401983bd9a490e98e5316ff0ae507d4b2e4096fd77` |
| `results/cv6/lenses/lens_INSTREAM_Github_2.8b_n25_s2.pt` | `6734844c920ecfb5f3f18a4135a88679c6b50d6fe59f676f828917fc484fbab4` |
| `results/cv6/lenses/lens_INSTREAM_Pile-CC_1.4b_n25_s0.pt` | `28fb8e54aae2bea11a122fdba06f6c30b9601a006d282d7872afd0d43b2c3e70` |
| `results/cv6/lenses/lens_INSTREAM_Pile-CC_1.4b_n25_s1.pt` | `af172234b25addf33a0a086a1ecfd995e7914e4d1fdf21d588d74830e7073863` |
| `results/cv6/lenses/lens_INSTREAM_Pile-CC_1.4b_n25_s2.pt` | `c6fb07ab61cb06a78913afb87427c9ca324f27a03b60f473edb83291c5c0ac00` |
| `results/cv6/lenses/lens_INSTREAM_Pile-CC_2.8b_n25_s0.pt` | `e955f5d339ac9043539b179e56a9a0c59808955179af8e3aec85afc5ed384d73` |
| `results/cv6/lenses/lens_INSTREAM_Pile-CC_2.8b_n25_s1.pt` | `ca7836772ad75f173f11aebfc423b61297da58c356e054cc67b0a1ec71416b5f` |
| `results/cv6/lenses/lens_INSTREAM_Pile-CC_2.8b_n25_s2.pt` | `545bdf63efc4706e446d87f8e813e992fcc3461a3a6ff074ed90564fb3ab06f1` |
| `results/cv6/lenses/lens_INSTREAM_StackExchange_1.4b_n25_s0.pt` | `469b8052d4a296314823529130a6bb8610870b860ef2c4ecbc2970c3b480b90a` |
| `results/cv6/lenses/lens_INSTREAM_StackExchange_1.4b_n25_s1.pt` | `1a9f037db52482669dc656456f0643a2c290e246866875fae8e36de261ae8a52` |
| `results/cv6/lenses/lens_INSTREAM_StackExchange_1.4b_n25_s2.pt` | `07992cfcbb3d32fbeb1871f4aa79f003c6f43c4ba24ce977ae0b874f1f08a8b3` |
| `results/cv6/lenses/lens_INSTREAM_StackExchange_2.8b_n25_s0.pt` | `c38d60e2daff2203100167c2b4862a31486e3fc8e70c972af4441438f7e6418e` |
| `results/cv6/lenses/lens_INSTREAM_StackExchange_2.8b_n25_s1.pt` | `45fc266499b5b97f6d5ef4d5671558a30f561430db0cd200870213950106b05d` |
| `results/cv6/lenses/lens_INSTREAM_StackExchange_2.8b_n25_s2.pt` | `3438dec229cfcacf26aef81d273c2080373e5d8d5ca3dc43eb1d1280de351c1e` |
| `results/cv6/lenses/lens_INSTREAM_USPTO_Backgrounds_1.4b_n25_s0.pt` | `9d068456087b18d29fd38f37345ca881c14311fb97d80a280ec3adaf12cad3c5` |
| `results/cv6/lenses/lens_INSTREAM_USPTO_Backgrounds_1.4b_n25_s1.pt` | `046eeef5dfb737093c822bec3607c8b22c56d5abb39e450d5aa0f7b4f8ff4265` |
| `results/cv6/lenses/lens_INSTREAM_USPTO_Backgrounds_1.4b_n25_s2.pt` | `3f0cd894627cc85caffa5e712da65069534ac996256fbd075a25e938b85e59c8` |
| `results/cv6/lenses/lens_INSTREAM_USPTO_Backgrounds_2.8b_n25_s0.pt` | `09013abe933c8754d1ce9dcf55df4f4fc597bde7502ffb6641341b25e24242ce` |
| `results/cv6/lenses/lens_INSTREAM_USPTO_Backgrounds_2.8b_n25_s1.pt` | `58fd8c7f4fb4d6e589cc559140d205d0fcda3a03ecaa95eaab83bc8e4ba2f743` |
| `results/cv6/lenses/lens_INSTREAM_USPTO_Backgrounds_2.8b_n25_s2.pt` | `c9f632a13fb0b1d0eb26b639bfe046d27e7f54980f9df4a09595d9a0287527da` |
| `results/cv6/lenses/lens_INSTREAM_Wikipedia_en_1.4b_n25_s0.pt` | `7db6b6e4605bd974a732b9b3501c060be612e0ba39c21c5a780658a1b64a6250` |
| `results/cv6/lenses/lens_INSTREAM_Wikipedia_en_1.4b_n25_s1.pt` | `7fa1e2a931ca8454368ae8bcd67a13bc053c89e617b0c005f970b8ebbc813f76` |
| `results/cv6/lenses/lens_INSTREAM_Wikipedia_en_1.4b_n25_s2.pt` | `b1763b5e8c3ca65bc26749be894aeaeb8961230491410e7853ffe45e41e77f4e` |
| `results/cv6/lenses/lens_INSTREAM_Wikipedia_en_2.8b_n25_s0.pt` | `1f3e2850461bf1e416585181c361f0a5018f163e3af388735e668f2c977a5a64` |
| `results/cv6/lenses/lens_INSTREAM_Wikipedia_en_2.8b_n25_s1.pt` | `29f72e060566702cc51497528e6935786b949a095f1322a4832843ab633a861e` |
| `results/cv6/lenses/lens_INSTREAM_Wikipedia_en_2.8b_n25_s2.pt` | `a319a3aebd47c030cc5d3861f3455e91c80812eac3c7711a5d64a970206c9d6c` |

Concatenated SHA-256 of the 30 hashes in filename order, so the set can be verified as a set:
`2bdee2a43517bfb1288c23628d535abc40f4caecfe9fd681f23a0baf8bd165f2`

### E66 D1 — the refit that was never persisted (added 2026-08-30)

E66b's registered controls **C2** and **C5** both reference **D1**, the first refit, and D1 was
never written to disk: the 2026-08-14 run computed it in memory, compared it, and dropped it
(`grep -c torch.save` on that version of the script returns **0**). With nothing on disk there was
nothing to diff, which is why both controls went unimplemented for sixteen days. This row exists so
that cannot recur.

| Artifact | Size | SHA-256 | What it is | Status |
|---|---|---|---|---|
| `results/lenses/misc/e66_D1_refit_410m_pilecc_s0.pt` | 54.5 MB | `00eebd6ddffff67532bb954e8d51674f0cd8d5021c0919d7139a942c607fff21` | **E66 D1, regenerated 2026-08-30 on an L40S.** Pythia-410m-deduped, Pile-CC seed-block 0, N=200, band [9,21], `target_layer=-2`, `dim_batch=32`, **fp32**, TF32 off and *measured* off (fp32 matmul rel err 2.473e-07). torch 2.11.0+cu128, matching the stored run's recorded version. Carries its own `pool_sha256` `708336f679b4fb14…`, the field whose absence made C5 unwritable. Reads **identically** to the stored operator: `read_diff == 0.0` under both `min` and `persist` (C2). | **local == box, SHA-verified 2026-08-30; box 49341043 destroyed after verification** |

**Why it is fp32 and 54.5 MB while the E48 operators are fp16 and 27 MB.** D1 is stored at the
precision it was *fitted* at. The fp16 cast used for `S` is derived from it at comparison time and
is deliberately not what is on disk — casting down before storage is exactly the conflation that
made E66's original C1 compare an fp16 artifact against an fp32 refit.
