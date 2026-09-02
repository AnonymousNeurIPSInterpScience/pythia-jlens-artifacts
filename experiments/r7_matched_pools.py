#!/usr/bin/env python3
"""r7_matched_pools.py — R7: is the corpus effect substantially a length/lexical effect?

PRE-REGISTRATION: docs/experiments/preregs/R7_length_matched_pools.md, committed before this ran.
Source slate: POSTREVIEW_EXPERIMENTS.md §3 Tier B, item R7.

WHY. SUSPICIONS.md S-3, unresolved: Github is the outlier in at least nine analyses and is also the
corpus whose token distribution differs most from the English-prose battery. The P-ladder already
shows length matters -- R moves 2.60 raw -> 3.46 length-matched at 410M. Nothing in results/ matches
the fitting corpora on length or lexical statistics before comparing their reads.

A DESIGN FINDING, STATED UP FRONT BECAUSE IT CHANGES WHAT "MATCHED" CAN MEAN HERE.
Fitting truncates every prompt to max_seq_len=128 AND filters to documents of >= 128 tokens
(require_full_window). **Every fitting prompt is therefore already exactly 128 tokens long.**
Fitting-side document length is matched BY CONSTRUCTION and cannot be the corpus effect. The
P-ladder's length effect is a property of the READ prompts, not the fitting pool.

What is NOT matched, and is what this experiment matches: the lexical composition of that
128-token window. We match on **type-token ratio** over the fitted window, which is the leading
free statistic once length is fixed, and report mean-surprisal-proxy and doc length alongside so
the residual imbalance is visible rather than implied.

READOUT: STRIPPED, per R1.

    python r7_matched_pools.py --device cuda --out ../results/r7_matched_pools_410m.json
"""
from __future__ import annotations
import argparse, json, os, statistics as st, sys, time

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))
from provenance import sha256_file, write_result  # noqa: E402
from r6_within_source import (BAND, K, ADMITTED, MODEL, WINDOW, build_items,  # noqa: E402
                              make_scorer, derangement)

CORPORA = ["Pile-CC", "StackExchange", "Wikipedia_en", "Github", "USPTO_Backgrounds"]
SEEDS = [0, 1, 2]
N_FIT = 200


def doc_stats(ids):
    """Statistics of the FITTED window: the first WINDOW tokens, which is all the fit ever sees."""
    w = ids[:WINDOW]
    return {"n_tokens_full_doc": len(ids), "ttr": len(set(w)) / len(w)}


def pools_for(corpus, tok, target_ttr, n_needed):
    """Qualifying docs with their window statistics, plus a TTR-matched ordering."""
    path = os.path.join(HERE, "..", "corpora", f"{corpus}.jsonl")
    texts = [json.loads(l)["text"] for l in open(path)]
    recs = []
    for t in texts:
        ids = tok(t).input_ids
        if len(ids) >= WINDOW:
            recs.append((t, doc_stats(ids)))
    if len(recs) < n_needed:
        raise SystemExit(f"ABORT: {corpus} supplies {len(recs)} full-window docs, need {n_needed}. "
                         f"Reporting the shortfall rather than padding, per the declared bias.")
    unmatched = recs[:n_needed]
    matched = sorted(recs, key=lambda r: abs(r[1]["ttr"] - target_ttr))[:n_needed]
    return {"unmatched": unmatched, "matched": matched, "n_qualifying": len(recs)}


def spread_over_seed_sd(per_corpus):
    """e53's convention, verified against its stored 35.52x: pooled SD is the MEAN of per-corpus SDs."""
    means = {c: st.mean(v) for c, v in per_corpus.items()}
    sds = {c: st.pstdev(v) for c, v in per_corpus.items()}
    pooled = st.mean(sds.values())
    spread = max(means.values()) - min(means.values())
    return {"per_corpus_mean": means, "per_corpus_seed_sd": sds, "pooled_seed_sd": pooled,
            "spread": spread, "spread_over_seed_sd": (spread / pooled) if pooled else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dim-batch", type=int, default=128)
    ap.add_argument("--n-fit", type=int, default=N_FIT)
    ap.add_argument("--lens-dir", default=os.path.join(HERE, "..", "results", "r7"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "r7_matched_pools_410m.json"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from fastfit import fast_fit
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of

    corpora = CORPORA[:2] if a.smoke else CORPORA
    seeds = [0] if a.smoke else SEEDS
    n_fit = a.n_fit
    os.makedirs(a.lens_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    items = build_items(tok, load_eval, readout_position, token_ids_of, EVAL_SETS, rstrip=True)
    if a.smoke:
        items = items[:60]
    print(f"{len(items)} eval items (STRIPPED readout)", flush=True)

    # ---------------------------------------------------------------- the matching target
    need = n_fit * len(seeds)
    prelim = {}
    for c in corpora:
        path = os.path.join(HERE, "..", "corpora", f"{c}.jsonl")
        texts = [json.loads(l)["text"] for l in open(path)]
        ttrs = []
        for t in texts:
            ids = tok(t).input_ids
            if len(ids) >= WINDOW:
                ttrs.append(doc_stats(ids)["ttr"])
            if len(ttrs) >= need * 3:
                break
        prelim[c] = st.median(ttrs)
        print(f"  {c:22s} median window TTR = {prelim[c]:.4f}", flush=True)
    target = st.median(prelim.values())
    print(f"  matching target TTR (median of medians) = {target:.4f}", flush=True)

    pools = {c: pools_for(c, tok, target, need) for c in corpora}

    # ---------------------------------------------------------------- C1: are the pools matched?
    def med_ttr(recs): return st.median([r[1]["ttr"] for r in recs])
    pre = {c: med_ttr(pools[c]["unmatched"]) for c in corpora}
    post = {c: med_ttr(pools[c]["matched"]) for c in corpora}
    pre_spread = max(pre.values()) - min(pre.values())
    post_spread = max(post.values()) - min(post.values())
    c1 = {"required": "post-matching spread in the matched statistic below 10% of pre-matching",
          "statistic": "median type-token ratio over the fitted 128-token window",
          "pre_matching_median_ttr": pre, "post_matching_median_ttr": post,
          "pre_matching_spread": pre_spread, "post_matching_spread": post_spread,
          "ratio": (post_spread / pre_spread) if pre_spread else None,
          "fires": bool(pre_spread and post_spread / pre_spread < 0.10)}
    print(f"C1 matching: TTR spread {pre_spread:.4f} -> {post_spread:.4f} "
          f"({100*post_spread/pre_spread:.1f}% of pre)  -> {'FIRES' if c1['fires'] else 'FAILS'}",
          flush=True)

    # ---------------------------------------------------------------- fit both arms
    ops, meta = {}, {}
    t0 = time.time()
    for arm in ("matched", "unmatched"):
        for c in corpora:
            recs = pools[c][arm]
            for s in seeds:
                key = f"{arm}|{c}|s{s}"
                blk = recs[s * n_fit:(s + 1) * n_fit]
                path = os.path.join(a.lens_dir, f"lens_R7_{arm}_{c}_s{s}_410m_n{n_fit}.pt")
                if not os.path.exists(path):
                    ts = time.time()
                    lens = fast_fit(model, [r[0] for r in blk], source_layers=BAND, target_layer=-2,
                                    dim_batch=a.dim_batch, max_seq_len=WINDOW, skip_first=16,
                                    device=a.device)
                    lens.save(path)
                    print(f"  fitted {key:34s} {time.time()-ts:.0f}s", flush=True)
                J = torch.load(path, map_location="cpu", weights_only=True)["J"]
                ops[key] = {l: J[l].float() for l in BAND}
                meta[key] = {"path": os.path.relpath(path, os.path.join(HERE, "..")),
                             "n_prompts": len(blk),
                             "median_ttr": st.median([r[1]["ttr"] for r in blk]),
                             "median_doc_tokens": st.median([r[1]["n_tokens_full_doc"] for r in blk])}
    print(f"  {len(ops)} operators in {(time.time()-t0)/60:.1f} min", flush=True)

    # ---------------------------------------------------------------- read
    score = make_scorer(model, items, a.device)
    reads = {k: score(J) for k, J in ops.items()}
    reads["__logit__"] = score(None)

    res = {}
    for ag in ("persist", "min"):
        res[ag] = {}
        for arm in ("matched", "unmatched"):
            per = {c: [reads[f"{arm}|{c}|s{s}"][ag] for s in seeds] for c in corpora}
            res[ag][arm] = spread_over_seed_sd(per)
    obs = res["persist"]["matched"]["spread_over_seed_sd"]
    unm = res["persist"]["unmatched"]["spread_over_seed_sd"]
    c2 = {"required": ("the unmatched arm must reproduce a spread from the SAME code path, so any "
                       "movement is attributable to matching and not to the pipeline"),
          "unmatched_spread_over_seed_sd_persist": unm,
          "matched_spread_over_seed_sd_persist": obs,
          "note": ("the published 58x is UNSTRIPPED; this arm is stripped, so the comparison that "
                   "matters is matched-vs-unmatched WITHIN this run, both at the corrected readout"),
          "fires": unm is not None}
    perm = derangement(BAND, 7000)
    floors = {k: score({l: ops[k][q] for l, q in zip(BAND, perm)}) for k in ops}
    c3 = {"required": "every operator clears its own layer-deranged floor under persist",
          "n_clearing": sum(1 for k in ops if reads[k]["persist"] > floors[k]["persist"]),
          "n_operators": len(ops)}
    c3["fires"] = c3["n_clearing"] == c3["n_operators"]

    if obs is None:
        verdict = "UNCLEAR — degenerate."
    elif obs > 20:
        verdict = (f"ACCEPT — the corpus effect is NOT substantially a lexical-composition effect. "
                   f"With the fitting pools matched on window type-token ratio the between-corpus "
                   f"spread is {obs:.1f} seed SDs, above the pre-registered 20.")
    elif obs < 5:
        verdict = (f"REJECT — the corpus effect is substantially a lexical effect. Matched spread "
                   f"{obs:.1f} seed SDs, below 5. The paper must say so.")
    else:
        verdict = (f"UNCLEAR — matched spread {obs:.1f} seed SDs, between 5 and 20. Report and "
                   f"stop; do not re-cut (CLAUDE.md §2.9).")

    prereg = "docs/experiments/preregs/R7_length_matched_pools.md"
    rec = {"experiment": "R7 — length- and format-matched fitting pools",
           "prereg": prereg, "prereg_sha256": sha256_file(os.path.join(HERE, "..", prereg)),
           "status": "PRE-REGISTERED",
           "decision_rule_verbatim": (
               "ACCEPT if the matched spread stays above 20 seed SDs. UNCLEAR between 5 and 20. "
               "REJECT below 5, in which case the corpus effect is substantially a length effect "
               "and the paper says so."),
           "readout_convention": "STRIPPED — the anchor rule (R1). Not comparable to pre-R1 figures.",
           "DESIGN_FINDING_length_is_matched_by_construction": (
               "fitting truncates to max_seq_len=128 and filters to documents of >= 128 tokens, so "
               "EVERY fitting prompt is already exactly 128 tokens. Fitting-side document length is "
               "matched by construction and cannot be the corpus effect; the P-ladder's length "
               "effect is a property of the READ prompts. What this experiment therefore matches is "
               "the lexical composition of the fitted window, via type-token ratio."),
           "tf32": {"allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                    "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)},
           "model": MODEL, "band": BAND, "K": K, "admitted_sets": ADMITTED,
           "corpora": corpora, "seeds": seeds, "n_fit": n_fit, "window_tokens": WINDOW,
           "matching_target_ttr": target, "per_corpus_median_ttr_prelim": prelim,
           "operators": meta, "reads": reads,
           "PRIMARY_matched_spread_over_seed_sd_persist": obs,
           "by_aggregation": res,
           "controls": {"C1_pools_are_matched": c1, "C2_unmatched_arm_same_code_path": c2,
                        "C3_derangement_floor": c3},
           "declared_bias": [
               "matching on one statistic leaves markup, code-ness and surprisal free",
               "Github may not be matchable against English prose; if its matched pool falls short "
               "the run ABORTS and reports the shortfall rather than padding",
               "document length is matched by construction (see DESIGN_FINDING), so this is a "
               "lexical-composition control, not a length control"],
           "VERDICT": verdict}
    write_result(a.out, rec, experiment="R7",
                 inputs=[os.path.join(HERE, "..", "corpora", f"{c}.jsonl") for c in corpora])
    for k, v in rec["controls"].items():
        print(f"  {k:34s} {v['fires']}")
    print(f"\nmatched spread/seed SD (persist) = {obs}   (unmatched: {unm})")
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
