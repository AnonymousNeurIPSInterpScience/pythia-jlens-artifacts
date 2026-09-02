#!/usr/bin/env python3
"""t65_ckpt_readout.py — E65 Phase 1: when during training does the transport lens start working?

PRE-REGISTRATION: docs/experiments/preregs/E65_training_axis_floor.md, written before this ran.
  PRIMARY  t* = the training-token count at which the J-lens read first exceeds BOTH its own
           layer-deranged floor and the logit lens at the same checkpoint, and stays above both.
  RULE     FLOOR EXISTS / NO FLOOR / NON-MONOTONE / UNCLEAR — see the prereg.
  C1  the FINAL checkpoint must reproduce e52_factorial_410m.json's Pile-CC|Q0 cell to within 2
      seed SD. Load-bearing: if it fails, this pipeline is not the paper's pipeline.
  C2  a derangement floor computed PER CHECKPOINT (a deranged operator on an early checkpoint is a
      different object; importing the final model's floor would be wrong).
  C4  cross-entropy per checkpoint, so a dead readout is not confused with a dead model.
  C5  E61's randomized-block operators are the t=0 anchor and are printed alongside.

ONE FITTING CORPUS, HELD FIXED. Non-negotiable: our own result is that the fitting corpus accounts
for more variance in the read than the read context does, so varying it across checkpoints would
swamp whatever training signal exists. Pile-CC, chosen before the run: mid-range at both scales,
not the degenerate Github, not the 410M-best USPTO whose advantage does not transfer to 1B.

    python experiments/t65_ckpt_readout.py --model 410m --smoke
    python experiments/t65_ckpt_readout.py --model 410m --device cuda
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))
from provenance import write_result  # noqa: E402

MODELS = {"410m": "EleutherAI/pythia-410m-deduped", "1b": "EleutherAI/pythia-1b-deduped"}
BANDS = {"410m": list(range(9, 22)), "1b": list(range(6, 14))}   # the declared L38-L92 rule
CKPTS = [0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
CORPUS = "Pile-CC"
K = [1, 2, 5, 10, 20, 50, 100]
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
N_FIT = 200
SEED_BLOCK = 0
N_DERANGE = 5
# e52_factorial_410m.json by_aggregation.persist.matrix["Pile-CC|Pile-CC"] is a READ-CONTEXT cell;
# the Q0 (no-context) comparison lives in e48_crossover_410m.json arms_admitted_mean.
# 410m: e48_crossover_410m.json arms_admitted_mean["J|Pile-CC|s0"].persist (Q0, band [9,21]).
# 1b:   ladder1b_b613/tv_Pile-CC_s0.json by_N["200"] admitted mean (band [6,13]) — E62's refit,
#       which is the only 1B measurement on the band this script uses.
C1_REF = {"410m": 0.04546590892132372, "1b": 0.04046863599547318}
C1_TOL = {"410m": 3.7e-4,              # 2 x the 1.84e-4 Pile-CC seed SD at 410m
          "1b":   2.146e-3}            # 2 x the 1.073e-3 Pile-CC seed SD at 1b


def derangement(band, seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        perm = [band[i] for i in torch.randperm(len(band), generator=g).tolist()]
        if all(p != l for p, l in zip(perm, band)):
            return perm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="410m", choices=list(MODELS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--items-per-chunk", type=int, default=24)
    ap.add_argument("--fitter", default="trainval", choices=["trainval", "fast"],
                    help="trainval = the manual jacobian_for_prompt accumulation that produced "
                         "every stored INSTREAM lens (and therefore every number C1 compares "
                         "against). fast = fastfit.fast_fit, which v1 used and whose CUDA "
                         "equivalence to trainval has never been established.")
    ap.add_argument("--save-lens-dir", default=None,
                    help="persist each checkpoint's J. The prereg required this and v1 omitted "
                         "it, which is why the C1 discrepancy could not be diagnosed offline.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5, fast_fit
    from jlens.fitting import jacobian_for_prompt
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of

    mid, BAND = MODELS[a.model], BANDS[a.model]
    tok = AutoTokenizer.from_pretrained(mid)
    L, HALF = len(BAND), len(BAND) // 2
    ckpts = [0, 143000] if a.smoke else CKPTS
    n_fit = 8 if a.smoke else N_FIT

    # ---- fitting prompts: the SAME documents at every checkpoint, seed block 0
    texts = [json.loads(l)["text"]
             for l in open(os.path.join(HERE, "..", "corpora", f"{CORPUS}.jsonl"))]
    b = len(texts) // 3
    pool = [t for t in texts[SEED_BLOCK * b:(SEED_BLOCK + 1) * b]
            if len(tok(t).input_ids) >= 128][:n_fit]
    if len(pool) < n_fit:
        raise SystemExit(f"ABORT: {CORPUS} supplies {len(pool)}/{n_fit} full-window prompts")

    # ---- eval items, fixed
    items, per_set = [], {}
    for name in EVAL_SETS:
        for it in load_eval(name):
            tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
            if not tgt:
                continue
            if a.smoke and per_set.get(name, 0) >= 8:
                continue
            per_set[name] = per_set.get(name, 0) + 1
            items.append({"set": name, "ids": tok(it["prompt"], add_special_tokens=True).input_ids,
                          "pos": readout_position(tok, name, it["prompt"]), "tgt": tgt})
    pair_set, pair_tgt, pair_item = [], [], []
    for ii, it in enumerate(items):
        for t in it["tgt"]:
            pair_set.append(it["set"]); pair_tgt.append(t); pair_item.append(ii)
    SET_IDX = {s: torch.tensor([i for i, p in enumerate(pair_set) if p == s], dtype=torch.long)
               for s in EVAL_SETS}
    ITEM_PAIRS = {}
    for pi, ii in enumerate(pair_item):
        ITEM_PAIRS.setdefault(ii, []).append(pi)
    P_n = len(pair_tgt)
    print(f"{mid} band={BAND} | {len(items)} items {P_n} pairs | {len(pool)} fit prompts "
          f"| {len(ckpts)} checkpoints", flush=True)

    rows = {}
    t0 = time.time()
    for step in ckpts:
        rev = f"step{step}"
        try:
            hf = AutoModelForCausalLM.from_pretrained(mid, revision=rev,
                                                      dtype=torch.float32).to(a.device).eval()
        except Exception as e:
            rows[rev] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            print(f"  {rev:12s} LOAD FAILED: {type(e).__name__}", flush=True)
            continue
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
        d = model.d_model

        # ---- fit J on the FIXED prompts
        if a.fitter == "fast":
            res = fast_fit(model, pool, dim_batch=128, max_seq_len=128, checkpoint_path=None,
                           checkpoint_every=0, log_every=0, target_layer=-2, skip_first=16)
            J = {l: res.jacobians[l].float() for l in BAND}
            n_fit_used = len(pool)
        else:
            # trainval.py's exact loop — the one that produced every stored INSTREAM lens.
            Jsum, n_fit_used, n_skip = None, 0, 0
            for pr in pool:
                try:
                    pj, _, _ = jacobian_for_prompt(model, pr, BAND, target_layer=-2,
                                                   dim_batch=128, max_seq_len=128, skip_first=16)
                except ValueError:
                    n_skip += 1
                    continue
                if Jsum is None:
                    Jsum = {l: torch.zeros_like(pj[l].float()) for l in BAND}
                for l in BAND:
                    Jsum[l] += pj[l].float().to(Jsum[l].device)
                n_fit_used += 1
            if not n_fit_used:
                raise SystemExit(f"ABORT: every fitting prompt failed at {rev}")
            J = {l: Jsum[l] / n_fit_used for l in BAND}
        if a.save_lens_dir:
            os.makedirs(a.save_lens_dir, exist_ok=True)
            torch.save({"J": {l: J[l].half() for l in BAND}, "band": BAND, "model": mid,
                        "revision": rev, "corpus": CORPUS, "seed": SEED_BLOCK,
                        "n": n_fit_used, "fitter": a.fitter},
                       os.path.join(a.save_lens_dir, f"lens_{a.model}_{rev}_{a.fitter}.pt"))

        # ---- C4 capability: CE on the fitting prompts, t48_competence_gate convention
        tot_nll, n_tok = 0.0, 0
        with torch.no_grad():
            for p in pool[:32]:
                ids = model.encode(p, max_length=128)
                lg = hf(ids).logits[0].float()
                x = ids[0] if ids.dim() > 1 else ids
                if lg.shape[0] - 1 <= 16:
                    continue
                tot_nll += torch.nn.functional.cross_entropy(
                    lg[16:-1], x[17:], reduction="sum").item()
                n_tok += x[17:].numel()
        ce = tot_nll / n_tok if n_tok else float("nan")

        # ---- activations, once per checkpoint
        A = torch.empty(len(items), L, d)
        with torch.no_grad():
            for ii, it in enumerate(items):
                t = torch.tensor([it["ids"]], device=a.device)
                with ActivationRecorder(model.layers, at=BAND) as rec:
                    model.forward(t)
                    A[ii] = torch.stack([rec.activations[l][0][it["pos"]].detach().float()
                                         for l in BAND])

        def score(T):
            H = A if T is None else torch.stack([A[:, j, :] @ T[BAND[j]].T for j in range(L)], 1)
            flat = H.reshape(-1, d)
            R = torch.empty(P_n, L, dtype=torch.float32)
            with torch.no_grad():
                for i0 in range(0, len(items), a.items_per_chunk):
                    i1 = min(i0 + a.items_per_chunk, len(items))
                    lg = model.unembed(flat[i0 * L:i1 * L]).float()
                    for ii in range(i0, i1):
                        blk = lg[(ii - i0) * L:(ii - i0 + 1) * L]
                        for pi in ITEM_PAIRS[ii]:
                            cand = torch.stack([(blk > blk[:, i:i + 1]).sum(1) + 1
                                                for i in pair_tgt[pi]])
                            R[pi] = cand.min(0).values.float()
            mn = R.min(dim=1).values
            return {"min": torch.stack([(mn <= k).float() for k in K]).mean(0),
                    "persist": torch.stack([((R <= k).float().sum(1) >= HALF).float()
                                            for k in K]).mean(0)}

        def adm(v):
            return statistics.mean([float(v[SET_IDX[s]].mean()) for s in ADMITTED])

        j_s, i_s = score(J), score(None)
        shufs = []
        for di in range(1 if a.smoke else N_DERANGE):
            Js = {l: J[q] for l, q in zip(BAND, derangement(BAND, 7000 + di))}
            shufs.append(adm(score(Js)["persist"]))
        rows[rev] = {
            "step": step, "fitter": a.fitter, "n_fit_used": n_fit_used,
            "jlens_persist": adm(j_s["persist"]), "jlens_min": adm(j_s["min"]),
            "logit_persist": adm(i_s["persist"]), "logit_min": adm(i_s["min"]),
            "shuf_persist_mean": statistics.mean(shufs),
            "shuf_persist_max": max(shufs), "shuf_draws": shufs,
            "cross_entropy": ce,
            "clears_derangement": adm(j_s["persist"]) > max(shufs),
            "clears_logit": adm(j_s["persist"]) > adm(i_s["persist"]),
        }
        r = rows[rev]
        print(f"  {rev:12s} J={r['jlens_persist']:.5f} logit={r['logit_persist']:.5f} "
              f"shuf_max={r['shuf_persist_max']:.5f} CE={ce:6.2f} "
              f"clears(dr,lg)=({r['clears_derangement']},{r['clears_logit']}) "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del hf, model, A, J
        if a.device == "cuda":
            torch.cuda.empty_cache()

    if a.smoke:
        print("\nsmoke only; no file written")
        return 0

    ok = [v for v in rows.values() if "error" not in v]
    clears = [v for v in ok if v["clears_derangement"] and v["clears_logit"]]
    # t*: the first step from which every LATER step also clears
    tstar, steps_sorted = None, sorted(ok, key=lambda v: v["step"])
    for i, v in enumerate(steps_sorted):
        if all(w["clears_derangement"] and w["clears_logit"] for w in steps_sorted[i:]):
            tstar = v["step"]; break

    final = rows.get("step143000", {})
    ref, tol = C1_REF.get(a.model), C1_TOL.get(a.model)
    c1 = {"reference": "e48_crossover_410m.json arms_admitted_mean['J|Pile-CC|s0'].persist",
          "reference_value": ref, "final_checkpoint": final.get("jlens_persist"),
          "abs_diff": (abs(final["jlens_persist"] - ref)
                       if ref and "jlens_persist" in final else None), "tolerance": tol}
    c1["fires"] = c1["abs_diff"] is not None and c1["abs_diff"] <= tol

    rec = {
        "experiment": "E65 Phase 1 — the transport lens across training checkpoints",
        "prereg": "docs/experiments/preregs/E65_training_axis_floor.md", "status": "PRE-REGISTERED",
        "model": mid, "band": BAND, "K": K, "admitted_sets": ADMITTED,
        "fitting_corpus": CORPUS, "fitting_corpus_held_fixed": True,
        "why_fixed": ("the fitting corpus accounts for more variance in the read than the read "
                      "context does; varying it across checkpoints would swamp the training signal"),
        "N": n_fit, "seed_block": SEED_BLOCK, "fitter": a.fitter, "n_derangements": N_DERANGE,
        "n_items": len(items), "n_pairs": P_n,
        "checkpoints": ckpts, "by_checkpoint": rows,
        "t_star_steps": tstar,
        "n_checkpoints_clearing_both": len(clears),
        "e61_untrained_anchor": {"randomized_block_persist": [0.00376, 0.00395, 0.00348],
                                 "free_baseline_persist": 0.02844,
                                 "source": "results/e61_randomized_null_410m.json"},
        "controls": {"C1_final_checkpoint_reproduces_e48": c1},
    }
    if not c1["fires"]:
        rec["VERDICT"] = ("UNCLEAR — the final checkpoint does not reproduce the stored result; "
                          "this pipeline is not the paper's pipeline and nothing here is comparable")
    elif not clears:
        rec["VERDICT"] = ("UNCLEAR — the lens clears its controls at NO checkpoint, including the "
                          "released model, which contradicts our own stored result")
    elif tstar == min(v["step"] for v in ok):
        rec["VERDICT"] = (
            f"NO FLOOR — the lens clears both its derangement floor and the logit lens at the "
            f"EARLIEST measured checkpoint (step{tstar}) and stays clear. The parameter-axis "
            f"degeneracy has no training-axis analogue at this scale: the instrument is usable "
            f"across training, which is what a checkpoint study needs to know.")
    elif tstar is None:
        rec["VERDICT"] = ("NON-MONOTONE — no step from which the lens clears at every later "
                          "checkpoint. Report the curve; make no floor claim.")
    else:
        rec["VERDICT"] = (
            f"FLOOR EXISTS at step{tstar} — before it the lens does not clear both controls. Any "
            f"checkpoint study using this readout has a validity window starting there.")

    out = a.out or os.path.join(HERE, "..", "results", f"e65_ckpt_readout_{a.model}_{a.fitter}.json")
    write_result(out, rec, experiment="E65-P1", inputs=[])
    print(f"\nC1 final vs stored: {c1['abs_diff']} vs tol {tol} -> "
          f"{'FIRES' if c1['fires'] else 'DOES NOT FIRE'}")
    print(f"t* = {tstar}\n\nVERDICT: {rec['VERDICT']}\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
