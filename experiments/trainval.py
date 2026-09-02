#!/usr/bin/env python3
"""trainval.py — fit and evaluate in ONE pass, with predictors recorded at every checkpoint.

WHY THIS SUPERSEDES THE E28 APPROACH
  jlens accumulates a running sum over prompts and divides by the count. So at ANY prompt
  index N the operator J_N = sum/N already exists in memory. E28 fitted six separate lenses
  and we later reconstructed intermediates by algebra; this instead evaluates *during* the
  single fit, at every checkpoint we ask for. One pass to N_max gives the whole ladder, with
  no refit, no reconstruction, and no fp16 round-trip error.

  Cost: one fit to N_max, plus one eval per checkpoint. Activations for the eval battery are
  cached once (they do not depend on the lens), so a checkpoint costs a matmul + unembed.

WHAT IT RECORDS AT EACH CHECKPOINT
  performance : read AUC per eval set under min / best1L / mean / persist
  operator    : dispersion, stable rank, top-k spectral share, condition number,
                per-layer dispersion slope, unembedding alignment
  corpus      : mean prompt length, type-token ratio, eval-vocabulary overlap,
                model cross-entropy on the fitting prompts
  These are the candidate predictors of the plateau. Dispersion is one of them, not the
  privileged one -- E28 left dispersion confounded with eval-distance and this is the fix.

    python trainval.py --model 70m --corpus Github --seed 0 --n-max 800 --device cuda
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

# The env var above is a DRIVER-level override; it does not change torch's own flags, so
# `torch.backends.cudnn.allow_tf32` still reports True and repro/13_vast_preflight.sh's flag
# check fails on it. GPT-NeoX has no convolutions so cudnn is unused here, but "unused" is an
# argument and this is a measurement repo: set both flags explicitly so the asserted state is
# the actual state, and let the on-box anchor-fidelity test be what decides.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))

MODELS = {"70m": "EleutherAI/pythia-70m-deduped", "160m": "EleutherAI/pythia-160m-deduped",
          "410m": "EleutherAI/pythia-410m-deduped", "1b": "EleutherAI/pythia-1b-deduped",
          "1.4b": "EleutherAI/pythia-1.4b-deduped", "2.8b": "EleutherAI/pythia-2.8b-deduped"}
DEFAULT_CKPTS = [10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800]
K = [1, 2, 5, 10, 20, 50, 100]


# ---------------------------------------------------------------- predictors
def operator_predictors(Jsum: dict, sumsq_fro: dict, n: int, band: list[int], WU: torch.Tensor):
    """Everything computable from the operator alone -- no eval, no labels.

    Dispersion uses the CANONICAL definition from fastfit.jacobian_dispersion so these
    numbers are comparable with every prior result:
        disp = E||J - Jbar||_F^2 / ||Jbar||_F^2 = (E||J||_F^2 - ||Jbar||_F^2) / ||Jbar||_F^2
    An elementwise Var/E^2 is NOT the same quantity -- it explodes wherever a matrix entry
    has near-zero mean, and on a first attempt here it returned ~6400 against the ~0.33 the
    Frobenius form gives at 70m.
    """
    out = {}
    disp_per_layer = {}
    for l in band:
        mean_fro_sq = (Jsum[l] / n).float().pow(2).sum().item()
        var = sumsq_fro[l] / n - mean_fro_sq
        disp_per_layer[l] = (var / mean_fro_sq) if mean_fro_sq > 0 else float("inf")
    out["dispersion_L0"] = disp_per_layer[band[0]]
    out["dispersion_mean"] = sum(disp_per_layer.values()) / len(band)
    # how fast dispersion decays across the band (a shape, not a level)
    xs = list(range(len(band))); ys = [disp_per_layer[l] for l in band]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    out["dispersion_slope"] = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den else float("nan")

    mid = band[len(band) // 2]
    J = (Jsum[mid] / n).float()
    sv = torch.linalg.svdvals(J)
    out["stable_rank"] = (sv.pow(2).sum() / sv[0].pow(2)).item()
    out["top8_spectral_share"] = (sv[:8].pow(2).sum() / sv.pow(2).sum()).item()
    out["condition_number"] = (sv[0] / sv[-1].clamp_min(1e-12)).item()
    # how much of J's output space lands where the unembedding can read it
    k = min(64, J.shape[0])
    Uj = torch.linalg.svd(J, full_matrices=False).U[:, :k]
    Uw = torch.linalg.svd(WU.float().T, full_matrices=False).U[:, :k]
    out["unembed_alignment"] = (Uj.T @ Uw).pow(2).sum().item() / k
    return out, disp_per_layer


def corpus_predictors(prompts, tok, model, eval_concept_ids, device, hf=None):
    """Corpus-side quantities -- the deconfounders. Github is both high-dispersion AND
    furthest from the English eval sets; these separate those two explanations."""
    lens_tok = [tok(p).input_ids for p in prompts]
    lengths = [len(t) for t in lens_tok]
    flat = [t for s in lens_tok for t in s]
    types = len(set(flat))
    corpus_vocab = set(flat)
    overlap = len(corpus_vocab & eval_concept_ids) / max(1, len(eval_concept_ids))
    # CE MUST come from the HF model, not the jlens wrapper. `model.forward` returns HIDDEN
    # STATES, so `lg.logits` was absent, `lg` was [1,T,d_model], and cross_entropy against token
    # ids raised every time -- caught by the bare `except` below and recorded as NaN in every
    # trainval result ever written. That is why RESULTS_TAXONOMY tiers this field D. Fixed here,
    # and the exception is no longer swallowed silently.
    # The window and weighting are t48_competence_gate.py's, deliberately, so this field is the
    # SAME QUANTITY as the one canonical CE path in the repo and not merely "a" cross-entropy:
    # positions skip_first..T-1 scored (matching the fitter's valid_position_mask), aggregated
    # token-weighted rather than as a mean of per-prompt means.
    ce, ce_error, n_tok = float("nan"), None, 0
    SKIP_FIRST = 16
    if hf is None:
        ce_error = "no HF model passed; the jlens wrapper returns hidden states, not logits"
    else:
        try:
            with torch.no_grad():
                tot_nll = 0.0
                for p in prompts[:32]:
                    ids = model.encode(p, max_length=128)
                    lg = hf(ids).logits[0].float()
                    x = ids[0] if ids.dim() > 1 else ids
                    if lg.shape[0] - 1 <= SKIP_FIRST:
                        continue
                    pred, tgt = lg[SKIP_FIRST:-1], x[SKIP_FIRST + 1:]
                    tot_nll += torch.nn.functional.cross_entropy(
                        pred, tgt, reduction="sum").item()
                    n_tok += tgt.numel()
                ce = tot_nll / n_tok if n_tok else float("nan")
        except Exception as e:                  # recorded, never swallowed (rule 0b)
            ce_error = f"{type(e).__name__}: {e}"
            print(f"  WARNING cross-entropy failed: {ce_error}", flush=True)
    return {"mean_prompt_tokens": sum(lengths) / len(lengths),
            "type_token_ratio": types / max(1, len(flat)),
            "eval_vocab_overlap": overlap,
            "model_cross_entropy": ce,
            "model_cross_entropy_n_scored_tokens": n_tok,
            "model_cross_entropy_convention": "t48_competence_gate: skip_first=16, token-weighted",
            "model_cross_entropy_error": ce_error}


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-max", type=int, default=800)
    ap.add_argument("--ckpts", default=",".join(map(str, DEFAULT_CKPTS)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dim-batch", type=int, default=128)
    ap.add_argument("--band", default=None,
                    help="explicit inclusive band lo,hi. Overrides --band-frac. E48 uses 9,21 so\n                          every arm shares the E28/E33 geometry and one activation cache.")
    ap.add_argument("--band-frac", default="0.38,0.92",
                    help="band as a fraction of depth; the anchor's relative-depth convention. "
                         "0.38/0.92 is the anchor's stated workspace range (~L38 to ~L92, its "
                         "section 4.1), and matches cv6_per_family_ladder.band_for().")
    # WHY THIS DEFAULT CHANGED, 2026-08-24. It was "0.35,0.85", which implements NO rule this
    # programme ever declared, and the drift was load-bearing: at 16 layers it yields [5,13],
    # which is exactly the 1B ladder's documented deviation from the declared [6,13]. The number
    # that deviation moved was measured (E62: interaction 9.158% -> 9.192%) and retired, but the
    # CAUSE stayed in the tree, so any run without an explicit --band reproduced it.
    #   floor(0.38L)..floor(0.92L) reproduces every band this programme actually used:
    #   70M [2,3]  160M [4,9]  410M [9,21]  1B [6,13]  1.4B [9,21]  2.8B [12,29].
    # tests/test_band_rule.py asserts exactly that table and fails on any further drift.
    # CONSEQUENCE, stated rather than hidden: E49 (derangement stability) ran under the OLD
    # default and its stored band is [8,20] at 410M. Re-running it now gives [9,21], so its stored
    # numbers are not reproducible from current code without --band 8,20. E49 is cited in the
    # paper with its band named for this reason.
    ap.add_argument("--save-lens", default=None,
                    help="also persist the final J as a .pt (fp16), so the operator can be\n                          reused for controls (e.g. the J^shuf derangement in t33)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--randomize-blocks", type=int, default=None,
                    help="E61: re-initialise EVERY transformer block from the model's own\n"
                         "                          initialiser with this seed, holding the "
                         "embedding, the unembedding and\n                          the final "
                         "norm fixed. The readout stays real; only the computation is\n"
                         "                          destroyed. This is the randomized-network "
                         "null -- how much read\n                          survives when there is "
                         "no computation to read.")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from jlens.fitting import jacobian_for_prompt
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of, rank_of

    mid = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(mid)
    hf = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(a.device)

    randomization = None
    if a.randomize_blocks is not None:
        blocks = hf.gpt_neox.layers
        before = float(sum(p.detach().float().pow(2).sum() for p in blocks.parameters()).sqrt())
        emb_before = float(hf.gpt_neox.embed_in.weight.detach().float().norm())
        un_before = float(hf.get_output_embeddings().weight.detach().float().norm())
        torch.manual_seed(a.randomize_blocks)
        # transformers >= 5 stamps `_is_hf_initialized = True` on every module it loaded weights
        # into, and `_init_weights` SILENTLY SKIPS those. Applying it directly is a no-op: measured,
        # block Fro 1603.7 -> 1603.7 and every downstream predictor bit-identical to the real
        # model. A "randomized null" that is secretly the real model is the worst possible failure
        # here, so the flag is cleared explicitly and the change is ASSERTED below, not recorded.
        # The guard lives on the TENSOR, not the module: transformers/initialization.py's
        # normal_/zeros_/ones_ all begin `if not getattr(tensor, "_is_hf_initialized", False)`.
        # Clearing it on modules (the obvious reading) leaves _init_weights a silent no-op.
        n_cleared = 0
        for blk in blocks:
            for prm in blk.parameters():
                if getattr(prm, "_is_hf_initialized", False):
                    prm._is_hf_initialized = False
                    n_cleared += 1
            blk.apply(hf._init_weights)
        after = float(sum(p.detach().float().pow(2).sum() for p in blocks.parameters()).sqrt())
        if after == before:
            raise SystemExit(
                f"ABORT: randomization was a no-op — block Frobenius norm unchanged at {before:.4f} "
                f"after clearing _is_hf_initialized on {n_cleared} modules. Do not proceed: the "
                f"'randomized' null would be the real model.")
        randomization = {
            "seed": a.randomize_blocks,
            "initialiser": "hf._init_weights (the model's own), initializer_range="
                           f"{getattr(hf.config, 'initializer_range', None)}",
            "n_blocks_reinitialised": len(blocks),
            "n_modules_uninitialised_flag_cleared": n_cleared,
            "block_param_fro_before": before, "block_param_fro_after": after,
            "blocks_actually_changed": before != after,
            "embedding_unchanged": emb_before == float(
                hf.gpt_neox.embed_in.weight.detach().float().norm()),
            "unembedding_unchanged": un_before == float(
                hf.get_output_embeddings().weight.detach().float().norm()),
            "final_norm_held_real": True,
            "why_readout_is_held_fixed": (
                "the read is unembed(norm(J h)); E38 shows W_U's concept-token geometry is what "
                "fails below 410M, so randomising it would answer a different question. This "
                "bounds 'no computation', not 'no model'."),
        }
        print(f"  RANDOMIZED {len(blocks)} blocks, seed {a.randomize_blocks}: block Fro "
              f"{before:.1f} -> {after:.1f}, embed/unembed held fixed", flush=True)

    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    n_layers = len(model.layers)
    lo, hi = (float(x) for x in a.band_frac.split(","))
    # target_layer=-2 resolves to n_layers-2, and jlens requires every source layer to be
    # STRICTLY below it. At 70m (6 layers) that caps the band at [2,3] -- the same two-layer
    # band E28 used, which is why `persist` degenerates to `min` at that scale.
    target_eff = n_layers - 2
    if a.band:
        b0, b1 = (int(x) for x in a.band.split(","))
        band = [l for l in range(b0, b1 + 1) if l < target_eff]
    else:
        band = [l for l in range(int(lo * n_layers), int(hi * n_layers) + 1) if l < target_eff]
    if len(band) < 2:
        band = [max(0, target_eff - 2), target_eff - 1]
    ckpts = sorted({int(x) for x in a.ckpts.split(",") if int(x) <= a.n_max})
    WU = model.unembed_matrix if hasattr(model, "unembed_matrix") else hf.get_output_embeddings().weight

    # ---- corpus
    texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "..", "corpora", f"{a.corpus}.jsonl"))]
    b = len(texts) // 3
    pool = texts[a.seed * b:(a.seed + 1) * b]
    pool = [t for t in pool if len(tok(t).input_ids) >= 128][:a.n_max]
    if len(pool) < a.n_max:
        print(f"  NOTE: corpus supplies {len(pool)}/{a.n_max} full-window prompts", flush=True)

    # ---- cache eval activations ONCE (lens-independent)
    items, concept_ids = [], set()
    with torch.no_grad():
        for name in EVAL_SETS:
            for it in load_eval(name):
                p = it["prompt"]
                ids = model.encode(p, max_length=256)
                pos = readout_position(tok, name, p)
                with ActivationRecorder(model.layers, at=band) as rec:
                    model.forward(ids)
                    acts = {l: rec.activations[l][0][pos].detach().float() for l in band}
                tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
                if tgt:
                    items.append((name, acts, tgt))
                    for t in tgt: concept_ids.update(t)
    print(f"  {a.model}/{a.corpus}/s{a.seed}: band={band} items={len(items)} ckpts={ckpts}", flush=True)

    def evaluate(Jmean):
        per = {}
        with torch.no_grad():
            for name, acts, tgt in items:
                R = torch.tensor([[min((rank_of(model.unembed(acts[l] @ Jmean[l].T).float().cpu(), i)
                                        or 10**9) for i in t) for t in tgt] for l in band],
                                 dtype=torch.float32)
                per.setdefault(name, []).append(R)
        out = {}
        for name, Rs in per.items():
            R = torch.cat(Rs, dim=1); mn = R.min(dim=0).values
            out[name] = {
                "min":     sum((mn <= k).float().mean().item() for k in K) / len(K),
                "best1L":  max(sum((R[i] <= k).float().mean().item() for k in K) / len(K)
                               for i in range(R.shape[0])),
                "mean":    sum((R <= k).float().mean().item() for k in K) / len(K),
                "persist": sum(((R <= k).float().sum(0) >= (len(band) // 2)).float().mean().item()
                               for k in K) / len(K)}
        return out

    # ---- the single fit pass, evaluating at each checkpoint
    Jsum = {l: torch.zeros(model.d_model, model.d_model, dtype=torch.float32, device=a.device)
            for l in band}
    sumsq_fro = {l: 0.0 for l in band}      # sum of ||J_i||_F^2, the canonical accumulator
    rec_out = {"model": mid, "short": a.model, "corpus": a.corpus, "seed": a.seed,
               "randomization": randomization,
               "band": band, "n_layers": n_layers, "ckpts": ckpts, "device": a.device,
               "n_items": len(items), "by_N": {}}
    t0, n, skipped = time.time(), 0, 0
    for i, prompt in enumerate(pool):
        try:
            pj, _, _ = jacobian_for_prompt(model, prompt, band, target_layer=-2,
                                           dim_batch=a.dim_batch, max_seq_len=128, skip_first=16)
        except ValueError as exc:
            skipped += 1
            if skipped <= 3:
                print(f"    skip prompt {i}: {exc}", flush=True)
            if skipped == 25 and n == 0:
                raise SystemExit(f"ABORT: first 25 prompts all failed -- {exc}")
            continue
        for l in band:
            # jacobian_for_prompt returns CPU tensors regardless of --device; move them to the
            # accumulator's device explicitly. The CUDA path of this script had never been run on
            # hardware before 2026-08-13 and failed here on the first GPU cell -- the same latent
            # shape as the W3/t20 CUDA bug recorded in SPINE A1.4.
            v = pj[l].float().to(Jsum[l].device)
            Jsum[l] += v; sumsq_fro[l] += v.pow(2).sum().item()
        n += 1
        if n in ckpts:
            Jm = {l: Jsum[l] / n for l in band}
            op, dpl = operator_predictors(Jsum, sumsq_fro, n, band, WU)
            cp = corpus_predictors(pool[:n], tok, model, concept_ids, a.device, hf=hf)
            rec_out["by_N"][str(n)] = {"reads": evaluate(Jm), "operator": op, "corpus": cp,
                                       "dispersion_per_layer": {str(k): v for k, v in dpl.items()}}
            el = time.time() - t0
            print(f"    N={n:<5} {el:7.0f}s  disp={op['dispersion_L0']:.3f} "
                  f"srank={op['stable_rank']:.1f} align={op['unembed_alignment']:.4f}", flush=True)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(rec_out, open(a.out, "w"), indent=1)   # crash-safe: write every checkpoint
    if a.save_lens and n > 0:
        os.makedirs(os.path.dirname(a.save_lens) or ".", exist_ok=True)
        torch.save({"J": {l: (Jsum[l] / n).half().cpu() for l in band}, "n_prompts": n,
                    "source_layers": band, "d_model": model.d_model,
                    "model": mid, "corpus": a.corpus, "seed": a.seed}, a.save_lens)
        print(f"  saved lens -> {a.save_lens}", flush=True)
    rec_out["runtime_s"] = round(time.time() - t0, 1)
    rec_out["n_used"] = n
    rec_out["n_skipped"] = skipped
    json.dump(rec_out, open(a.out, "w"), indent=1)
    print(f"  DONE {n} prompts ({skipped} skipped), {rec_out['runtime_s']:.0f}s -> {a.out}", flush=True)
    if n == 0: raise SystemExit("ABORT: zero prompts contributed -- nothing was fitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
