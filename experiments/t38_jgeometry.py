#!/usr/bin/env python3
"""t38_jgeometry.py — E38-E42: is the SLM read failure a property of J rather than W_U?

PRE-REGISTRATION: docs/experiments/preregs/superseded/PREREG_E38_JGEOMETRY.md, fixed before any of these numbers existed.
Decision rules are quoted there and are NOT reinterpreted here.

WHY (short form; the prereg has the full argument)
  E37's "gap = 0 at rank 1" was vacuous: a rank-1 W_U makes every token's logit the same vector
  scaled by one scalar, so token RANKING is identical for any J. And mutating W_U does not cleanly
  drive the effect -- at 410M the E37 verdict flips with the FITTING CORPUS while W_U is identical.
  A property of W_U cannot depend on which corpus J was averaged over. The readout composes exactly
  two objects, v_t = J^T W_U[t]; if it is not W_U, it is J.

  The obvious J-side hypothesis is already dead, and inverted: ||E J||^2/E||J||^2 = 1/(1+disp) is
  0.814 at 70M (mean survives) and 0.297 at 410M (mostly cancellation) -- the average is CLEANEST
  exactly where the lens is USELESS. So this is not an estimation-quality problem.

THE PRE-REGISTRATION BOUNDARY (binding, see prereg)
  1.4b and 2.8b are the last clean cells of PREREG_PYTHIA_T7_v2 §2. E38 and E41 read ONLY the lens
  matrices and the identity -- no eval battery, no pass@k, no read outcome -- so they run at every
  scale without burning it. E39 and E40 touch the eval battery and therefore STOP AT 1B.

    python t38_jgeometry.py --device cpu
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

MODELS = {"70m": "EleutherAI/pythia-70m-deduped", "160m": "EleutherAI/pythia-160m-deduped",
          "410m": "EleutherAI/pythia-410m-deduped", "1b": "EleutherAI/pythia-1b-deduped",
          "1.4b": "EleutherAI/pythia-1.4b-deduped", "2.8b": "EleutherAI/pythia-2.8b-deduped"}
GEOM_ONLY = ["70m", "160m", "410m", "1b", "1.4b", "2.8b"]     # E38/E41: no eval contact
EVAL_OK = ["70m", "160m", "410m", "1b"]                        # E39/E40: 1.4b/2.8b are barred
CORPORA = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]


def band_of(n_layers):
    b = [l for l in range(int(.35 * n_layers), int(.85 * n_layers) + 1) if l < n_layers - 2]
    return b if len(b) >= 2 else [max(0, n_layers - 4), n_layers - 3]


# ---------------------------------------------------------------- E38
def rho_identity(J: torch.Tensor) -> float:
    """||J - c*I||_F / ||J||_F with c* = tr(J)/d, the exact minimiser. 0 = scaled identity."""
    d = J.shape[0]
    c = torch.diagonal(J).sum() / d
    return (torch.linalg.matrix_norm(J - c * torch.eye(d, dtype=J.dtype)) /
            torch.linalg.matrix_norm(J)).item()


# ---------------------------------------------------------------- E41
def layer_coupling(Js: dict) -> float:
    ls = sorted(Js)
    cs = []
    for i, a in enumerate(ls):
        for b in ls[i + 1:]:
            x, y = Js[a].flatten(), Js[b].flatten()
            cs.append((x @ y / (x.norm() * y.norm())).item())
    return statistics.median(cs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e38_jgeometry.json"))
    a = ap.parse_args()

    rec = {"experiment": "E38-E42 — J geometry: is the SLM read failure a property of J, not W_U?",
           "prereg": "docs/experiments/preregs/superseded/PREREG_E38_JGEOMETRY.md",
           "boundary": "E38/E41 touch no eval data and run at all 6 scales; E39/E40 stop at 1b "
                       "because 1.4b/2.8b are the last clean cells of PREREG_PYTHIA_T7_v2 §2",
           "E38": {}, "E39": {}, "E40": {}, "E41": {}, "E42": {}, "controls": {}}

    # ---- CONTROLS for E38, before any real number
    g = torch.Generator().manual_seed(0)
    R = torch.randn(512, 512, generator=g)
    rec["controls"]["E38_C1_random_matrix_rho"] = rho_identity(R)
    rec["controls"]["E38_C2_scaled_identity_rho"] = rho_identity(3.7 * torch.eye(512))
    rec["controls"]["E38_C1_fires"] = rho_identity(R) > 0.95
    rec["controls"]["E38_C2_fires"] = rho_identity(3.7 * torch.eye(512)) < 1e-6
    print(f"C1 random matrix rho_I = {rec['controls']['E38_C1_random_matrix_rho']:.4f} "
          f"(must be ~1)   C2 scaled-I rho_I = {rec['controls']['E38_C2_scaled_identity_rho']:.2e} "
          f"(must be 0)", flush=True)

    # ================= E38 + E41 : pure J geometry, all six scales =================
    print("\n[E38/E41] corpus- and recipe-matched wikitext penultimate lenses, all scales",
          flush=True)
    print(f"{'model':8s}{'d_model':>8}{'band':>6}{'rho_I(band med)':>17}{'layer cos(med)':>16}")
    for m in GEOM_ONLY:
        p = os.path.join(HERE, "..", "results", "lenses", "ladder", f"lens_{m}_n200_db128_pen.pt")
        if not os.path.exists(p):
            print(f"{m:8s}  MISSING {p}"); continue
        blob = torch.load(p, map_location="cpu")
        J = {int(k): v.float() for k, v in blob["J"].items()}
        nl = max(J) + 3                                   # penultimate target => sources < nl-2
        bd = [l for l in band_of(nl) if l in J] or sorted(J)
        rho = [rho_identity(J[l]) for l in bd]
        lc = layer_coupling({l: J[l] for l in bd})
        rec["E38"][m] = {"rho_I_per_layer": dict(zip(map(str, bd), rho)),
                         "rho_I_band_median": statistics.median(rho),
                         "d_model": J[bd[0]].shape[0], "band": bd}
        rec["E41"][m] = {"layer_cos_median": lc, "band": bd}
        print(f"{m:8s}{J[bd[0]].shape[0]:8d}{len(bd):6d}{statistics.median(rho):17.4f}{lc:16.4f}",
              flush=True)

    # per-corpus rho_I at 410M (E42 input)
    print("\n[E38 per-corpus, 410M]", flush=True)
    for c in CORPORA:
        p = os.path.join(HERE, "..", "results", "lenses", "e28", f"e28_{c}_410m_n400_s0.pt")
        if not os.path.exists(p):
            continue
        J = {int(k): v.float() for k, v in torch.load(p, map_location="cpu")["J"].items()}
        bd = [l for l in range(9, 22) if l in J]
        rho = statistics.median([rho_identity(J[l]) for l in bd])
        lc = layer_coupling({l: J[l] for l in bd})
        rec["E38"].setdefault("by_corpus_410m", {})[c] = {"rho_I_band_median": rho}
        rec["E41"].setdefault("by_corpus_410m", {})[c] = {"layer_cos_median": lc}
        print(f"  {c:22s} rho_I={rho:.4f}  layer_cos={lc:.4f}", flush=True)

    # ================= E39 + E40 : touch the eval battery, so 1b and below only ============
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of

    for m in EVAL_OK:
        p = os.path.join(HERE, "..", "results", "lenses", "ladder", f"lens_{m}_n200_db128_pen.pt")
        if not os.path.exists(p):
            continue
        print(f"\n[E39/E40] {m}", flush=True)
        tok = AutoTokenizer.from_pretrained(MODELS[m])
        hf = AutoModelForCausalLM.from_pretrained(MODELS[m], dtype=torch.float32).to(a.device).eval()
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
        WU = hf.get_output_embeddings().weight.data.float()
        J = {int(k): v.float() for k, v in torch.load(p, map_location="cpu")["J"].items()}
        nl = len(model.layers)
        bd = [l for l in band_of(nl) if l in J] or sorted(J)

        cids = set()
        for name in EVAL_SETS:
            for it in load_eval(name):
                for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])):
                    if t:
                        cids.update(t)
        cids = sorted(cids)
        Wc = WU[cids]                                        # [n_concepts, d]

        def geom(V):
            Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-9)
            G = Vn @ Vn.T
            n = G.shape[0]
            mc = ((G.sum() - n) / (n * (n - 1))).item()
            s = torch.linalg.svdvals(V)
            e = (s.pow(2).cumsum(0) / s.pow(2).sum())
            return mc, int((e < 0.90).sum().item()) + 1

        mcI, erI = geom(Wc)
        per = {}
        for l in bd:
            mcJ, erJ = geom(Wc @ J[l])                       # v_t = J^T W_U[t]  => rows W_U[t]^T J
            per[str(l)] = {"mean_cos_J": mcJ, "eff_rank_J": erJ, "spread_gain": mcI - mcJ}
        rec["E39"][m] = {"n_concept_tokens": len(cids), "mean_cos_I": mcI, "eff_rank_I": erI,
                         "per_layer": per,
                         "spread_gain_band_median": statistics.median(
                             [v["spread_gain"] for v in per.values()])}
        print(f"  concept vectors: mean_cos(I)={mcI:+.4f} effrank(I)={erI}  "
              f"spread_gain(median)={rec['E39'][m]['spread_gain_band_median']:+.4f}", flush=True)

        # E40: J's action on activations that actually occur
        acts = {l: [] for l in bd}
        with torch.no_grad():
            for name in EVAL_SETS:
                for it in load_eval(name):
                    if not [t for t in (token_ids_of(tok, w)
                                        for w in it.get("intermediates", [])) if t]:
                        continue
                    ids = model.encode(it["prompt"], max_length=256)
                    pos = readout_position(tok, name, it["prompt"])
                    with ActivationRecorder(model.layers, at=bd) as r:
                        model.forward(ids)
                        for l in bd:
                            acts[l].append(r.activations[l][0][pos].detach().float())
        cosl = {}
        for l in bd:
            H = torch.stack(acts[l])
            JH = H @ J[l].T
            cs = (H * JH).sum(1) / (H.norm(dim=1) * JH.norm(dim=1)).clamp_min(1e-9)
            cosl[str(l)] = cs.mean().item()
        rec["E40"][m] = {"act_cos_per_layer": cosl,
                         "act_cos_band_median": statistics.median(list(cosl.values())),
                         "n_activations": len(acts[bd[0]])}
        print(f"  cos(Jh, h) band median = {rec['E40'][m]['act_cos_band_median']:+.4f}", flush=True)
        del hf, model
        json.dump(rec, open(a.out, "w"), indent=1)

    # ================= E42 : does one property explain BOTH scale and corpus? ==========
    def pear(x, y):
        n = len(x); mx, my = sum(x) / n, sum(y) / n
        sx = math.sqrt(sum((v - mx) ** 2 for v in x)); sy = math.sqrt(sum((v - my) ** 2 for v in y))
        return sum((p - mx) * (q - my) for p, q in zip(x, y)) / (sx * sy) if sx * sy else float("nan")

    # measured J-vs-logit gaps, from E37's corpus-matched wikitext runs (min = the anchor's metric)
    gap_scale = {}
    for m in EVAL_OK:
        f = os.path.join(HERE, "..", "results", f"e37_rank_ablation_{m}_wikitext.json")
        if os.path.exists(f):
            gap_scale[m] = json.load(open(f))["by_rank"]["top_full"]["gap"]["min"]
    # per-corpus gaps at 410M from E33 (persist and min available)
    e33 = json.load(open(os.path.join(HERE, "..", "results", "e33_logit_baseline_410m_v2.json")))
    L = e33["persist_admitted_mean"]["logit_I"]
    gap_corpus = {c: e33["e28_asymptotes"][c]["mean"] - L for c in CORPORA
                  if c in e33.get("e28_asymptotes", {})}

    props_scale = {"rho_I": {m: rec["E38"][m]["rho_I_band_median"] for m in gap_scale
                             if m in rec["E38"]},
                   "layer_cos": {m: rec["E41"][m]["layer_cos_median"] for m in gap_scale
                                 if m in rec["E41"]},
                   "spread_gain": {m: rec["E39"][m]["spread_gain_band_median"] for m in gap_scale
                                   if m in rec["E39"]},
                   "act_cos": {m: rec["E40"][m]["act_cos_band_median"] for m in gap_scale
                               if m in rec["E40"]}}
    props_corpus = {"rho_I": {c: v["rho_I_band_median"]
                              for c, v in rec["E38"].get("by_corpus_410m", {}).items()},
                    "layer_cos": {c: v["layer_cos_median"]
                                  for c, v in rec["E41"].get("by_corpus_410m", {}).items()}}
    rec["E42"]["gap_by_scale"] = gap_scale
    rec["E42"]["gap_by_corpus_410m"] = gap_corpus
    print("\n[E42] does one J-property track the gap ACROSS SCALE and ACROSS CORPUS?")
    for name, d in props_scale.items():
        ks = [k for k in d if k in gap_scale]
        if len(ks) >= 3:
            r = pear([d[k] for k in ks], [gap_scale[k] for k in ks])
            rec["E42"].setdefault("scale", {})[name] = {"r": r, "n": len(ks), "models": ks}
            print(f"  scale  {name:14s} r={r:+.3f}  (n={len(ks)})")
    for name, d in props_corpus.items():
        ks = [k for k in d if k in gap_corpus]
        if len(ks) >= 4:
            xs, ys = [d[k] for k in ks], [gap_corpus[k] for k in ks]
            r = pear(xs, ys)
            loo = {k: pear([v for j, v in zip(ks, xs) if j != k],
                           [v for j, v in zip(ks, ys) if j != k]) for k in ks}
            worst = min(loo.items(), key=lambda kv: abs(kv[1]))
            rec["E42"].setdefault("corpus", {})[name] = {
                "r": r, "loo": loo, "worst_loo": worst, "survives": abs(r) >= 0.8 and abs(worst[1]) >= 0.8}
            print(f"  corpus {name:14s} r={r:+.3f}  worst LOO {worst[1]:+.3f} (-{worst[0]})  "
                  f"{'SURVIVES' if rec['E42']['corpus'][name]['survives'] else 'fails control'}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
