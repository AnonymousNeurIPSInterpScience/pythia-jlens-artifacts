#!/usr/bin/env python3
"""t31_local_bakeoff.py — E31's predictor bake-off, computed from artifacts already on disk.

WHY THIS EXISTS
  PLAN.tex §5 pre-registers E31 as a ~$7 GPU run: refit 15 cells at 410M with trainval.py and
  correlate 11 candidate predictors against the read plateau. But E31's PRIMARY output is

      r(predictor, plateau AUC) for each of 11 predictors, with leave-one-out on every corpus

  and every input to that number already exists locally:

    * plateau AUC per (corpus, seed)   -> results/ladder410/*.json          (E28 read ladder)
    * per-layer dispersion             -> results/e28_*_410m_n*_provenance.json
    * the fitted operators themselves  -> results/lenses/e28/e28_*_410m_n400_s*.pt     (240 lenses, 6.7 GB)
    * the fitting corpora              -> corpora/*.jsonl
    * the eval battery                 -> vendored jacobian-lens @ 581d398

  So the spectral predictors are an SVD of a matrix we own, and the corpus predictors are a
  tokenizer pass over text we own. No GPU is required and no lens is refitted.

WHAT THIS IS NOT
  This is NOT all of E31. E31 as specced records every predictor at 11 checkpoints along N, so it
  can ask how a predictor MOVES with N. This asks only the primary question, at the single matched
  sample size N=400. Differences, declared:

    (a) N=400, not 800. Matched across all five corpora -- Pile-CC and USPTO have no N=800 fit
        (SPINE A1.2), and dispersion carries an O(1/N) finite-sample bias, so comparing a
        dispersion measured at N=800 against one at N=400 compares two different estimators.
        CLAUDE.md rule 0b: do not accept a proxy for the thing you care about.
    (b) The stored lenses are fp16. Measured round-trip floor ~3e-3 relative, an order of
        magnitude under the seed SD on the downstream metric (e28_ladder_410m.py header).
    (c) One predictor value per corpus, not a trajectory in N.

DECISION RULE — quoted verbatim from PLAN.tex §5 (E31), fixed before this ran, NOT reinterpreted:
    * dispersion survives and eval_vocab_overlap does not  => F19 promotes to a finding
    * eval_vocab_overlap predicts as well or better        => F19 is an eval-distance effect
                                                              wearing dispersion's clothes; RETRACT
    * neither, and no predictor clears leave-one-out       => the ceiling is corpus identity with
                                                              no compact summary; report as a null
    CONTROL: any predictor whose |r| collapses below 0.8 when its extreme corpus is dropped is
    reported as LEVERAGE-DRIVEN whatever its full-sample r.

    python t31_local_bakeoff.py --out results/e31_local_bakeoff_410m.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))

MODEL = "EleutherAI/pythia-410m-deduped"
CORPORA = ["Github", "Pile-CC", "StackExchange", "USPTO_Backgrounds", "Wikipedia_en"]
SEEDS = [0, 1, 2]
N_MATCH = 400
BAND = list(range(9, 22))          # the band the E28 read ladder scored
MID = BAND[len(BAND) // 2]         # trainval.py evaluates spectra at the band midpoint
TOPK_SUBSPACE = 64                 # trainval.py: unembed_alignment uses the top-64 subspaces

# critical |r| for a two-sided p<0.05 Pearson test, by n
CRIT_R = {3: 0.997, 4: 0.950, 5: 0.878}


# ------------------------------------------------------------------ statistics
def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):                       # average ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    return pearson(rank(x), rank(y))


# ------------------------------------------------------------------ outcome
def read_ladder(results_dir):
    """plateau AUC per (corpus, seed): mean over the 6 eval sets, under `persist`, for N>=75.

    `persist` = concept readable at >= half the band's layers. It is the aggregation the
    programme trusts; `min` is disqualified (CLAUDE.md §6.0).
    """
    cells = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "ladder410", "*.json"))):
        d = json.load(open(f))
        cells[(d["corpus"], d["seed"])] = d
    out, per_N = {}, {}
    for (c, s), d in cells.items():
        vals = {}
        for nstr, blk in d["by_N"].items():
            vals[int(nstr)] = sum(blk[k]["persist"] for k in blk) / len(blk)
        per_N[(c, s)] = vals
        plateau = [v for n, v in vals.items() if n >= 75]
        out[(c, s)] = {"plateau_auc": sum(plateau) / len(plateau),
                       "auc_at_400": vals.get(400),
                       "auc_at_25": vals.get(25),
                       "max_auc": max(vals.values())}
    return out, per_N


# ------------------------------------------------------------------ predictors
def dispersion_predictors(results_dir):
    """dispersion at L0, band mean, and band slope -- from the E28 provenance JSONs.

    NOTE the layer mismatch, which is a real inconsistency in the prior analysis: every published
    dispersion number in this programme is dispersion at SOURCE LAYER 0, while the reads are
    scored on band [9,21]. Both are recorded here so the choice is visible rather than implicit.
    """
    out = {}
    for f in glob.glob(os.path.join(results_dir, f"e28_*_410m_n{N_MATCH}_s*_provenance.json")):
        d = json.load(open(f))
        c = os.path.basename(d["corpus_file"]).replace(".jsonl", "")
        if d["n_prompts_used"] != N_MATCH:
            continue
        jd = {int(k): v["dispersion"] for k, v in (d.get("jacobian_dispersion") or {}).items()}
        ys = [jd[l] for l in BAND]
        xs = list(range(len(BAND)))
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        out[(c, d["seed_block"])] = {
            "dispersion_L0": jd[0],
            "dispersion_band_mean": my,
            "dispersion_slope": sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den,
        }
    return out


def spectral_predictors(results_dir, WU):
    """stable rank / top-8 share / condition number / unembedding alignment, from the lenses."""
    Uw = torch.linalg.svd(WU.float().T, full_matrices=False).U[:, :TOPK_SUBSPACE]
    out = {}
    for c in CORPORA:
        for s in SEEDS:
            p = os.path.join(results_dir, f"e28_{c}_410m_n{N_MATCH}_s{s}.pt")
            blob = torch.load(p, map_location="cpu")
            assert blob["n_prompts"] == N_MATCH, f"{p} holds n={blob['n_prompts']}, want {N_MATCH}"
            J = blob["J"][MID].float()
            sv = torch.linalg.svdvals(J)
            U = torch.linalg.svd(J, full_matrices=False).U[:, :TOPK_SUBSPACE]
            out[(c, s)] = {
                "stable_rank": (sv.pow(2).sum() / sv[0].pow(2)).item(),
                "top8_spectral_share": (sv[:8].pow(2).sum() / sv.pow(2).sum()).item(),
                "condition_number": (sv[0] / sv[-1].clamp_min(1e-12)).item(),
                "unembed_alignment": (U.T @ Uw).pow(2).sum().item() / TOPK_SUBSPACE,
            }
            print(f"    {c:20s} s{s} srank={out[(c,s)]['stable_rank']:6.2f} "
                  f"align={out[(c,s)]['unembed_alignment']:.5f}", flush=True)
    return out


def corpus_predictors(corpora_dir, tok, hf, concept_ids, n_ce=64):
    """Length, lexical diversity, eval-vocabulary overlap, and model cross-entropy.

    Every prompt is truncated to 128 tokens because that is `max_seq_len` in the fit -- the
    fitting run never saw a 129th token, so a corpus statistic computed over the full document
    is a statistic of something the operator was not fitted on.
    """
    out = {}
    with torch.no_grad():
        for c in CORPORA:
            texts = [json.loads(l)["text"] for l in open(os.path.join(corpora_dir, f"{c}.jsonl"))]
            b = len(texts) // 3
            for s in SEEDS:
                pool = [t for t in texts[s * b:(s + 1) * b]
                        if len(tok(t).input_ids) >= 128][:N_MATCH]
                ids = [tok(t).input_ids for t in pool]
                win = [t[:128] for t in ids]
                flat = [x for t in win for x in t]
                vocab = set(flat)
                tot = 0.0
                for p in pool[:n_ce]:
                    x = torch.tensor([tok(p).input_ids[:128]])
                    lg = hf(x).logits
                    tot += torch.nn.functional.cross_entropy(
                        lg[0, :-1].float(), x[0, 1:], reduction="mean").item()
                out[(c, s)] = {
                    "mean_prompt_tokens": sum(len(t) for t in ids) / len(ids),
                    "type_token_ratio": len(vocab) / len(flat),
                    "eval_vocab_overlap": len(vocab & concept_ids) / len(concept_ids),
                    "model_cross_entropy": tot / min(n_ce, len(pool)),
                }
                print(f"    {c:20s} s{s} ce={out[(c,s)]['model_cross_entropy']:.4f} "
                      f"overlap={out[(c,s)]['eval_vocab_overlap']:.4f}", flush=True)
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "..", "results"))
    ap.add_argument("--corpora", default=os.path.join(HERE, "..", "corpora"))
    ap.add_argument("--n-ce", type=int, default=64, help="prompts per cell for cross-entropy")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e31_local_bakeoff_410m.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from anchor_evals import EVAL_SETS, load_eval, token_ids_of

    print("[1/5] read ladder", flush=True)
    outcome, per_N = read_ladder(a.results)
    print(f"      {len(outcome)} (corpus, seed) cells", flush=True)

    print("[2/5] dispersion from provenance", flush=True)
    disp = dispersion_predictors(a.results)

    print("[3/5] model + eval battery", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    concept_ids, n_items = set(), 0
    for name in EVAL_SETS:
        for it in load_eval(name):
            tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
            if tgt:
                n_items += 1
                for t in tgt:
                    concept_ids.update(t)
    print(f"      {n_items} eval items, {len(concept_ids)} distinct concept token ids", flush=True)

    print("[4/5] spectral predictors from the stored lenses", flush=True)
    spec = spectral_predictors(a.results, hf.get_output_embeddings().weight)

    print("[5/5] corpus predictors", flush=True)
    corp = corpus_predictors(a.corpora, tok, hf, concept_ids, a.n_ce)

    # ---- assemble: one row per (corpus, seed), then average seeds to one row per corpus
    PRED = ["dispersion_L0", "dispersion_band_mean", "dispersion_slope", "stable_rank",
            "top8_spectral_share", "condition_number", "unembed_alignment",
            "mean_prompt_tokens", "type_token_ratio", "eval_vocab_overlap",
            "model_cross_entropy"]
    OUTCOMES = ["plateau_auc", "auc_at_400", "max_auc"]
    cells = {}
    for c in CORPORA:
        for s in SEEDS:
            row = {}
            row.update(disp.get((c, s), {}))
            row.update(spec.get((c, s), {}))
            row.update(corp.get((c, s), {}))
            row.update({k: v for k, v in outcome.get((c, s), {}).items()})
            cells[f"{c}|s{s}"] = row
    by_corpus = {}
    for c in CORPORA:
        rows = [cells[f"{c}|s{s}"] for s in SEEDS]
        by_corpus[c] = {k: statistics.mean([r[k] for r in rows if r.get(k) is not None])
                        for k in PRED + OUTCOMES}
        by_corpus[c]["plateau_auc_seed_sd"] = statistics.stdev([r["plateau_auc"] for r in rows])

    # ---- the bake-off
    table = {}
    for outc in OUTCOMES:
        ys = [by_corpus[c][outc] for c in CORPORA]
        for p in PRED:
            xs = [by_corpus[c][p] for c in CORPORA]
            r = pearson(xs, ys)
            loo = {}
            for i, c in enumerate(CORPORA):
                loo[c] = pearson([v for j, v in enumerate(xs) if j != i],
                                 [v for j, v in enumerate(ys) if j != i])
            worst = min(loo.items(), key=lambda kv: abs(kv[1]))
            table.setdefault(outc, {})[p] = {
                "r": r,
                "spearman": spearman(xs, ys),
                "significant_n5": abs(r) > CRIT_R[5],
                "loo": loo,
                "worst_loo_corpus": worst[0],
                "worst_loo_r": worst[1],
                "leverage_driven": abs(worst[1]) < 0.8,
                "survives_control": abs(r) > CRIT_R[5] and abs(worst[1]) >= 0.8,
            }

    rec = {
        "experiment": "E31-L (local reanalysis of E31's primary outcome)",
        "written": "2026-08-13",
        "model": MODEL,
        "n_match": N_MATCH,
        "band": BAND,
        "spectral_layer": MID,
        "aggregation": "persist",
        "outcome_definition": "mean read AUC over 6 eval sets, N>=75, seeds averaged",
        "decision_rule_source": "PLAN.tex section 5, E31 -- quoted in this file's docstring, not reinterpreted",
        "control": "|r| must stay >= 0.8 under every leave-one-corpus-out refit",
        "declared_deviations_from_E31": [
            "single matched N=400 rather than an 11-point trajectory in N",
            "fp16 stored lenses (~3e-3 relative floor, << seed SD)",
            "no refit: operators are the E28 artifacts",
        ],
        "inputs": {
            "ladder_cells": len(outcome),
            "lenses": len(spec),
            "eval_items": n_items,
            "concept_token_ids": len(concept_ids),
        },
        "by_corpus": by_corpus,
        "by_cell": cells,
        "bakeoff": table,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")

    # ---- report
    print(f"\n{'predictor':24s} {'r':>8} {'rho':>7} {'sig':>5} {'worst LOO':>10} {'drop':>18} {'verdict':>16}")
    for p, v in sorted(table["plateau_auc"].items(), key=lambda kv: -abs(kv[1]["r"])):
        verdict = ("SURVIVES" if v["survives_control"]
                   else "leverage-driven" if v["significant_n5"] else "n.s.")
        print(f"{p:24s} {v['r']:+8.3f} {v['spearman']:+7.3f} "
              f"{'yes' if v['significant_n5'] else 'no':>5} {v['worst_loo_r']:+10.3f} "
              f"{v['worst_loo_corpus']:>18} {verdict:>16}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
