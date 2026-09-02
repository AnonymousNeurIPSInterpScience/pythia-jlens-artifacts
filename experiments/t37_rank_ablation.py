#!/usr/bin/env python3
"""t37_rank_ablation.py — E37: does unembedding rank CAUSE the J-lens read collapse?

THE GAP THIS CLOSES
  S1 says the J-lens fails below 410M *because* the unembedding is near rank-1 (top singular
  value share 95.5% at 70M, 91.5% at 160M, 9.8% at 410M; t21_conditioning_*). Every number
  supporting that is CORRELATIONAL and, worse, structurally confounded: across the Pythia ladder
  every small model is both small AND rank-degenerate. No observational comparison across models
  can separate "small" from "rank-1", because the ladder never varies one without the other.

  This experiment breaks the confound by intervening on the geometry INSIDE ONE MODEL. We take
  410M -- where the lens demonstrably works (+75% over the logit lens on the best corpus, E33) --
  and progressively destroy the rank of its unembedding, holding the model, the operator, the
  activations, the eval battery and the metric fixed. Rank becomes a dose, and the read advantage
  becomes a dose-response.

  If the mechanism is real, the J-lens advantage over the logit lens must vanish as r -> 1,
  because a rank-1 output space has exactly one readable direction and no rotation of the
  activation can help. If the advantage SURVIVES at low rank, the rank-1 story is wrong and S1's
  mechanism must be retracted.

PRE-REGISTRATION -- fixed before this ran.
  PRIMARY   gap(r) = readAUC(J^P; W_U^(r)) - readAUC(I; W_U^(r)), over the 5 admitted eval sets,
            as a function of effective rank r. Reported under BOTH aggregations: the anchor's own
            `min` (their published metric) and our `persist`.
  RULE      * gap(r) decreases monotonically-in-trend and gap(r<=2) <= 0.25 * gap(full)
              => ACCEPT: unembedding rank is the causal bottleneck for the read advantage.
            * gap(r) is flat in r, i.e. gap(r<=2) >= 0.75 * gap(full)
              => REJECT: rank is not the mechanism. S1's mechanistic claim is retracted and the
                 scale trend needs a different explanation.
            * anything between => UNCLEAR. Report and stop; do NOT re-pick the rank grid.
  CONTROL   C1 rank-r RANDOM subspace, norm-matched, instead of the top-r singular subspace.
            If the top-r and random-r curves coincide, the result is about dimensionality alone;
            if top-r is much better at equal r, the specific principal subspace matters too.
            Either is informative, but they are different claims and must not be conflated.
  CONTROL   C2 at r = full, every number must reproduce E33 exactly (logit 0.02844, Github
            0.02808 under persist). A mismatch means the truncation path altered the base case.
  BIAS      Truncating W_U changes the model's actual output distribution, so at low r this is no
            longer "the model". We are NOT claiming the truncated model is a good LM; we are
            asking whether a fitted transport can beat an unfitted one when the readable output
            space is small. That is exactly the question S1's mechanism poses.

    python t37_rank_ablation.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))

MODELS = {"70m":"EleutherAI/pythia-70m-deduped","160m":"EleutherAI/pythia-160m-deduped",
          "410m":"EleutherAI/pythia-410m-deduped","1b":"EleutherAI/pythia-1b-deduped"}
BAND = list(range(9, 22))
K = [1, 2, 5, 10, 20, 50, 100]
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, None]      # None = full rank (the E33 base case)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="410m", choices=["70m","160m","410m","1b"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lens", default="e28_USPTO_Backgrounds_410m_n400_s0.pt",
                    help="the operator to test. USPTO is the BEST corpus at 410M (+75%% over "
                         "logit), so it is where a collapse is most visible.")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e37_rank_ablation_410m.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of, rank_of

    MODEL = MODELS[a.model]
    MODEL = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    global BAND, RANKS
    nl = len(model.layers)
    BAND = [l for l in range(int(.35*nl), int(.85*nl)+1) if l < nl-2]
    if len(BAND) < 2: BAND = [max(0, nl-4), nl-3]
    dm = model.d_model
    RANKS = [r for r in (1,2,4,8,16,32,64,128,256,512) if r < dm] + [None]
    print(f"model={a.model} n_layers={nl} band={BAND} d_model={dm}", flush=True)
    W_full = hf.get_output_embeddings().weight.data.clone()
    print(f"W_U {tuple(W_full.shape)}", flush=True)

    # ---- activations are independent of W_U, so cache once
    items = []
    with torch.no_grad():
        for name in EVAL_SETS:
            for it in load_eval(name):
                p = it["prompt"]
                ids = model.encode(p, max_length=256)
                pos = readout_position(tok, name, p)
                with ActivationRecorder(model.layers, at=BAND) as rec:
                    model.forward(ids)
                    acts = {l: rec.activations[l][0][pos].detach().float() for l in BAND}
                tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
                if tgt:
                    items.append((name, acts, tgt))
    print(f"cached {len(items)} eval items", flush=True)

    U, S, Vh = torch.linalg.svd(W_full.float(), full_matrices=False)
    print(f"SVD done; top-1 share = {(S[0]**2/S.pow(2).sum()).item():.4f}", flush=True)

    def set_rank(r, mode="top"):
        """Replace W_U in place with a rank-r version, so the readout path is untouched."""
        if r is None:
            hf.get_output_embeddings().weight.data.copy_(W_full)
            return
        if mode == "top":
            W = (U[:, :r] * S[:r]) @ Vh[:r]
        else:                                   # C1: a RANDOM r-dim subspace of the row space
            g = torch.Generator().manual_seed(1234 + r)
            Q, _ = torch.linalg.qr(torch.randn(W_full.shape[1], r, generator=g))
            W = (W_full.float() @ Q) @ Q.T
        W = W * (W_full.norm() / W.norm())      # norm-match so logit scale is comparable
        hf.get_output_embeddings().weight.data.copy_(W.to(W_full.dtype))

    def evaluate(T):
        per = {}
        with torch.no_grad():
            for name, acts, tgt in items:
                rows = []
                for l in BAND:
                    h = acts[l] if T is None else acts[l] @ T[l].T
                    lg = model.unembed(h).float().cpu()
                    rows.append([min((rank_of(lg, i) or 10 ** 9) for i in t) for t in tgt])
                per.setdefault(name, []).append(torch.tensor(rows, dtype=torch.float32))
        out = {}
        for name, Rs in per.items():
            R = torch.cat(Rs, dim=1); mn = R.min(dim=0).values
            out[name] = {
                "min": sum((mn <= k).float().mean().item() for k in K) / len(K),
                "persist": sum(((R <= k).float().sum(0) >= (len(BAND) // 2)).float().mean().item()
                               for k in K) / len(K)}
        return out

    blob = torch.load(os.path.join(HERE, "..", "results", a.lens), map_location=a.device)["J"]
    JP = {l: blob[l].float() for l in BAND}

    def am(d, agg):
        return statistics.mean([d[s][agg] for s in ADMITTED])

    rec = {"model_short": a.model, "experiment": "E37 — unembedding rank ablation: is rank the CAUSE of the read collapse?",
           "prereg": "in-file docstring, fixed before running", "model": MODEL, "band": BAND,
           "lens": a.lens, "ranks": [x if x else "full" for x in RANKS], "admitted_sets": ADMITTED,
           "d_model": W_full.shape[1], "top1_share": (S[0] ** 2 / S.pow(2).sum()).item(),
           "by_rank": {}}
    for mode in ["top", "random"]:
        for r in RANKS:
            if mode == "random" and r is None:
                continue
            set_rank(r, mode)
            lo, jp = evaluate(None), evaluate(JP)
            key = f"{mode}_{r if r else 'full'}"
            rec["by_rank"][key] = {
                "rank": r, "mode": mode,
                "logit": {ag: am(lo, ag) for ag in ("min", "persist")},
                "jlens": {ag: am(jp, ag) for ag in ("min", "persist")},
                "gap": {ag: am(jp, ag) - am(lo, ag) for ag in ("min", "persist")}}
            g = rec["by_rank"][key]["gap"]
            print(f"  {mode:6s} r={str(r if r else 'full'):>5}  "
                  f"gap(min)={g['min']:+.5f}  gap(persist)={g['persist']:+.5f}", flush=True)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(rec, open(a.out, "w"), indent=1)
    set_rank(None)

    full = rec["by_rank"]["top_full"]["gap"]
    verdicts = {}
    for ag in ("min", "persist"):
        low = max(rec["by_rank"][f"top_{r}"]["gap"][ag] for r in (1, 2))
        frac = low / full[ag] if full[ag] else float("nan")
        verdicts[ag] = ("ACCEPT — rank is the causal bottleneck" if frac <= 0.25 else
                        "REJECT — rank is not the mechanism; retract S1's mechanistic claim"
                        if frac >= 0.75 else "UNCLEAR — report and stop")
        rec[f"gap_ratio_lowrank_over_full_{ag}"] = frac
    rec["VERDICT"] = verdicts
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"\nfull-rank gap: min={full['min']:+.5f} persist={full['persist']:+.5f}")
    for ag in ("min", "persist"):
        print(f"  {ag:8s} gap(r<=2)/gap(full) = {rec[f'gap_ratio_lowrank_over_full_{ag}']:+.3f}"
              f"  -> {verdicts[ag]}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
