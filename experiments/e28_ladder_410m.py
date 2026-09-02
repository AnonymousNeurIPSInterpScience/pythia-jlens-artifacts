#!/usr/bin/env python3
"""e28_ladder_410m.py — the 410M read ladder: reconstruct, cache, score, in parallel on CPU.

WHAT IT DOES
  1. RECONSTRUCT. J_N is a plain unweighted mean over prompts and the E28 grid is nested
     (t2_fastfit takes pool[:n]), so  sum_{i=a+1..b} J_i = b*J_b - a*J_a.  Consecutive-run
     sums of the six disjoint blocks give 16 distinct N instead of the 6 fitted.
     Verified exact: an independent 3-lens fit reproduced the reconstruction to rel err
     1.6e-07, cosine 1.0000000.  fp16 storage leaves a ~3e-3 floor, an order of magnitude
     under the 3.5e-3 seed SD on the downstream metric.
  2. CACHE. Activations at the band layers are LENS-INDEPENDENT, so one forward sweep of
     the eval battery serves every lens. Measured 32 s for 551 items at 410M.
  3. SCORE. transport -> unembed -> rank -> pass@k under all four aggregations.
  4. PARALLEL. One worker per (corpus, seed) cell; torch pinned to 1 thread per worker so
     14 processes do not oversubscribe 14 cores.

  Reads only. W3 cannot share the cache (ablation changes the forward pass) and is ~50x
  the cost; it runs separately once we know whether there is a curve.

    python e28_ladder_410m.py --device cuda --cells "Github:0,Github:1"
    python e28_ladder_410m.py --device cuda --smoke      # one cell, proves the GPU path
"""
from __future__ import annotations
import argparse, json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")   # 10-bit mantissa breaks the anchor gate
import torch
torch.set_num_threads(1)
if torch.cuda.is_available():
    assert not torch.backends.cuda.matmul.allow_tf32, "TF32 is ON — forbidden"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))

MODEL = "EleutherAI/pythia-410m-deduped"
FITTED = [25, 50, 100, 200, 400, 800]
CORPORA = ["Pile-CC", "StackExchange", "Wikipedia_en", "Github", "USPTO_Backgrounds"]
SEEDS = [0, 1, 2]


def blocks_available(have: dict[int, str]) -> list[int]:
    """Fitted N present for this cell, ascending."""
    return sorted(have)


def reconstruct(J: dict[int, dict], ns: list[int]) -> dict[int, dict]:
    """{N: {layer: tensor}} for every consecutive-run sum of the disjoint blocks.

    Blocks are [1..n0], (n0..n1], (n1..n2], ...  A run from block i to block j covers
    prompts (n_{i-1} .. n_j], with count n_j - n_{i-1}, and sum  n_j*J_{n_j} - n_{i-1}*J_{n_{i-1}}.
    Only runs are valid -- non-adjacent unions are not recoverable.
    """
    layers = list(J[ns[0]])
    out: dict[int, dict] = {}
    bounds = [0] + ns                       # cumulative prompt boundaries
    S = {n: {l: n * J[n][l].float() for l in layers} for n in ns}   # running sums
    S[0] = {l: torch.zeros_like(J[ns[0]][l], dtype=torch.float32) for l in layers}
    for i in range(len(bounds) - 1):
        for j in range(i + 1, len(bounds)):
            lo, hi = bounds[i], bounds[j]
            n = hi - lo
            if n <= 0 or n in out:
                continue
            out[n] = {l: (S[hi][l] - S[lo][l]) / n for l in layers}
    return out


def cell(args):
    corpus, seed, band, outdir, dev, rstrip = args
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from jlens.lens import JacobianLens
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of, rank_of

    have = {}
    for n in FITTED:
        p = os.path.join(HERE, "..", "results", "lenses", "e28", f"e28_{corpus}_410m_n{n}_s{seed}.pt")
        if os.path.exists(p):
            have[n] = p
    if not have:
        return (corpus, seed, 0, "no lenses")

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)

    J = {n: torch.load(have[n], map_location="cpu", weights_only=True)["J"] for n in have}
    ladder = reconstruct(J, blocks_available(have))
    ladder = {n: {l: m.to(dev) for l, m in d.items()} for n, d in ladder.items()}

    # ---- one forward sweep, cached
    items = []
    with torch.no_grad():
        for name in EVAL_SETS:
            for it in load_eval(name):
                # R8. anchor_evals.py:32-34 and :228 strip; this scorer did not. The unstripped
                # path reads token id 209 (a bare space), which does not occur in the tokenisation
                # of prompt+target at all. 157 of 551 released items.
                p = it["prompt"].rstrip() if rstrip else it["prompt"]
                ids = model.encode(p, max_length=256)
                pos = readout_position(tok, name, p)
                with ActivationRecorder(model.layers, at=band) as rec:
                    model.forward(ids)
                    acts = {l: rec.activations[l][0][pos].detach().float() for l in band}
                tgt = [token_ids_of(tok, w) for w in it.get("intermediates", [])]
                tgt = [t for t in tgt if t]
                if tgt:
                    items.append((name, acts, tgt))

    K = [1, 2, 5, 10, 20, 50, 100]
    def score(Jm):
        per = {}
        with torch.no_grad():
            for name, acts, tgt in items:
                ranks = []
                for l in band:
                    lg = model.unembed(acts[l] @ Jm[l].T.float()).float().cpu()
                    ranks.append([min((rank_of(lg, ids) or 10**9) for ids in t) for t in tgt])
                R = torch.tensor(ranks, dtype=torch.float32)          # [layers, concepts]
                per.setdefault(name, []).append(R)
        out = {}
        for name, Rs in per.items():
            R = torch.cat(Rs, dim=1)                                   # [layers, all concepts]
            mn = R.min(dim=0).values
            aggs = {
                "min":     sum((mn <= k).float().mean().item() for k in K) / len(K),
                "best1L":  max(sum((R[i] <= k).float().mean().item() for k in K) / len(K)
                               for i in range(R.shape[0])),
                "mean":    sum((R <= k).float().mean().item() for k in K) / len(K),
                "persist": sum(((R <= k).float().sum(0) >= (len(band) // 2)).float().mean().item()
                               for k in K) / len(K),
            }
            out[name] = aggs
        return out

    res = {"corpus": corpus, "seed": seed, "model": MODEL, "band": band, "device": dev,
           "rstrip": bool(rstrip),
           "readout_convention": ("STRIPPED — the anchor rule: readout at the token immediately "
                                  "preceding `target`" if rstrip else
                                  "UNSTRIPPED (LEGACY) — readout at the trailing-space token"),
           "fitted_N": sorted(have), "ladder_N": sorted(ladder),
           "n_items": len(items), "by_N": {}}
    t0 = time.time()
    for n in sorted(ladder):
        res["by_N"][str(n)] = score(ladder[n])
    res["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"ladder_{corpus}_s{seed}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return (corpus, seed, len(ladder), f"{res['runtime_s']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cells", default="", help='comma list "Corpus:seed"; default = all 15')
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "ladder410"))
    ap.add_argument("--rstrip", action="store_true",
                    help="R8. Apply the anchor's readout rule: strip the prompt's trailing space "
                         "BEFORE tokenising. A CORRECTION, not a sensitivity arm.")
    a = ap.parse_args()
    band = list(range(9, 22))
    if a.cells:
        want = [(c.split(":")[0], int(c.split(":")[1])) for c in a.cells.split(",")]
    else:
        want = [(c, s) for c in CORPORA for s in SEEDS]
    if a.smoke:
        want = want[:1]
    cells = [(c, s, band, a.out, a.device, a.rstrip) for c, s in want]
    print(f"  {len(cells)} cell(s), device={a.device}, workers={a.workers}, band={len(band)}",
          flush=True)
    t0 = time.time()
    if a.workers == 1:
        for cl in cells:
            corpus, seed, nN, msg = cell(cl)
            print(f"  done {corpus}/s{seed}: {nN} N-values  {msg}"
                  f"   [{time.time()-t0:.0f}s]", flush=True)
    else:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            for corpus, seed, nN, msg in pool.imap_unordered(cell, cells):
                print(f"  done {corpus}/s{seed}: {nN} N-values  {msg}"
                      f"   [{time.time()-t0:.0f}s]", flush=True)
    print(f"  DONE in {(time.time()-t0)/60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
