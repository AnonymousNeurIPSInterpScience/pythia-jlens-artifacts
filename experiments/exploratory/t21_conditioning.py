#!/usr/bin/env python3
"""t21_conditioning.py — E31. Does the J-lens produce better-conditioned swap frames?

WHY. external review Prompt 9 establishes that the coordinate-swap write is structurally biased toward
whichever lens defines the basis: c = V†h depends on (VᵀV)⁻¹, so the edit magnitude is governed
by rho = <v_s, v_t> through kappa_2(V) = sqrt((1+|rho|)/(1-|rho|)). The J-lens induces the metric
JJᵀ on output-space directions, so it MAY produce systematically less-collinear concept pairs and
therefore win every swap benchmark for a reason unrelated to semantic faithfulness.

This is cheap to settle: rho needs no forward pass, only the lenses and W_U. If the confound is
real, every published swap number -- including the anchor's 40-88% -- is partly a conditioning
artifact, and that is a methodological finding in its own right.

================================ PRE-REGISTRATION ================================
Written before the first number. Do not reinterpret; if the outcome contradicts the rule, STOP
and report.

PRIMARY OUTCOME
  delta_med = median|rho|(jlens) - median|rho|(logit), over SEMANTIC swap pairs, at band layers.

DECISION RULE
  delta_med <= -0.05  -> CONFOUND CONFIRMED. The J-lens frames are better conditioned. Every
                         swap result must be reported at matched conditioning/dose, and E25's
                         native-success number is not interpretable on its own.
  |delta_med| < 0.05  -> CONFOUND ABSENT. The swap benchmark is fair on this axis; E25 may
                         report native success as primary.
  delta_med >= +0.05  -> CONFOUND INVERTED. The J-lens is WORSE conditioned, so any swap
                         advantage it shows is achieved DESPITE a geometric handicap, which
                         strengthens rather than weakens a positive swap result.

CONTROLS, each with the number it must produce to count as firing
  C1 RANDOM VOCAB PAIRS. Same measurement on uniformly drawn vocabulary pairs.
     Prompt 9 §4: "If only semantically related pairs decorrelate, that is more interesting than
     global whitening." If the jlens shift is the same size on random pairs, the operator is
     globally whitening and the effect is not about concepts.
  C2 SHUFFLED OPERATOR. If J^shuf shows the same conditioning shift as J^P, conditioning is
     GENERIC to any fitted Jacobian and is not a property of layer identity.
  C3 NORM RATIO. ||J^T w|| / ||w||. A pure rescale cannot change rho (rho is scale-free), so if
     the norms move a lot while rho does not, that is evidence the operator is not whitening.

DECLARED BIAS
  rho is measured on UNIT-normalised directions, which is mandatory (Prompt 9 §2A) but means the
  result says nothing about dose. Dose is E25's problem, not this one.
  Band layers only, penultimate-target lenses only, one fit corpus.
==================================================================================
"""
from __future__ import annotations

import argparse, ast, json, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "jacobian-lens"))

from jlens.hf import Layout, from_hf                                     # noqa: E402
from jlens.lens import JacobianLens                                      # noqa: E402
import anchor_evals as ae                                               # noqa: E402
from joperator import token_direction                                   # noqa: E402
from t13_transport_controls import _derangement                         # noqa: E402
from t17_reaggregate import band_of                                     # noqa: E402
from write_geometry import gram_stats, unit                             # noqa: E402

PYTHIA_LAYOUT = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                       embed="embed_in", lm_head="lm_head")
EVALS = ["multihop", "multilingual", "order-ops", "poetry", "typo", "association"]


def _first_token(tok, word: str) -> int | None:
    ids = tok(" " + str(word).strip(), add_special_tokens=False)["input_ids"]
    return ids[0] if ids else None


def semantic_pairs(tok, limit: int) -> list[tuple[int, int, str]]:
    """(source_token, target_token, provenance) — the pairs a swap would actually use.

    Within an item: the intermediate concept and the item's target. This is the anchor's own
    construction, so these are the pairs whose conditioning matters.
    """
    out = []
    for name in EVALS:
        try:
            items = ae.load_eval(name)
        except Exception:
            continue
        for it in items:
            inter = it.get("intermediates")
            if isinstance(inter, str):
                try:
                    inter = ast.literal_eval(inter)
                except Exception:
                    inter = [inter]
            if not inter:
                continue
            s = _first_token(tok, inter[0])
            t = _first_token(tok, it.get("target", ""))
            if s is None or t is None or s == t:
                continue
            out.append((s, t, name))
            if len(out) >= limit:
                return out
    return out


def random_pairs(vocab: int, n: int, gen: torch.Generator) -> list[tuple[int, int, str]]:
    a = torch.randint(0, vocab, (n,), generator=gen)
    b = torch.randint(0, vocab, (n,), generator=gen)
    return [(int(x), int(y), "random") for x, y in zip(a, b) if int(x) != int(y)]


def rho_for(pairs, WU, J, layer):
    """|rho| for every pair under one transport at one layer. J=None => the logit lens."""
    out = []
    for s, t, _ in pairs:
        ws, wt = WU[s].float(), WU[t].float()
        vs = ws if J is None else token_direction(J, ws)
        vt = wt if J is None else token_direction(J, wt)
        if float(vs.norm()) < 1e-8 or float(vt.norm()) < 1e-8:
            continue
        r, kV, kG = gram_stats(vs, vt)
        out.append((abs(r), kV, float(vs.norm()) / max(float(ws.norm()), 1e-12)))
    return out


def med(xs):
    if not xs:
        return None
    t = torch.tensor(xs, dtype=torch.float)
    return float(t.median())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--max-pairs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    try:
        model = from_hf(hf, tok)
    except ValueError:
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT)
    lens = JacobianLens.load(a.lens)
    WU = model._lm_head.weight.detach().float()          # [vocab, d]
    layers = list(lens.source_layers)
    band = band_of(model.n_layers, layers)
    gen = torch.Generator().manual_seed(a.seed)

    sem = semantic_pairs(tok, a.max_pairs)
    rnd = random_pairs(WU.shape[0], len(sem), gen)
    perm = _derangement(layers, a.seed)
    print(f"model={a.model}  layers={len(layers)}  band={band}  "
          f"semantic pairs={len(sem)}  random pairs={len(rnd)}")

    # ---- ANISOTROPY DIAGNOSTIC (added after the smoke run surfaced med|rho| ~ 0.97 on RANDOM
    # vocabulary pairs, which is impossible for isotropic directions and had to be explained
    # before any lens comparison could be interpreted).
    ridx = torch.randint(0, WU.shape[0], (2000,), generator=torch.Generator().manual_seed(a.seed))
    R = WU[ridx]
    Rn = torch.nn.functional.normalize(R, dim=1)
    C = Rn @ Rn.T
    off = C[~torch.eye(len(C), dtype=torch.bool)]
    mu = WU.mean(0)
    Rc = torch.nn.functional.normalize(R - mu, dim=1)
    Cc = Rc @ Rc.T
    offc = Cc[~torch.eye(len(Cc), dtype=torch.bool)]
    Z = torch.nn.functional.normalize(
        torch.randn(2000, WU.shape[1], generator=torch.Generator().manual_seed(a.seed + 1)), dim=1)
    Cz = Z @ Z.T
    offz = Cz[~torch.eye(len(Cz), dtype=torch.bool)]
    mun = mu / mu.norm().clamp_min(1e-12)
    aniso = {
        "raw_mean_cos": float(off.mean()),
        "mean_removed_cos_DIAGNOSTIC_ONLY": float(offc.mean()),
        "isotropic_control_mean_cos": float(offz.mean()),
        "common_component_frac_of_row_norm":
            float((R @ mun).abs().mean() / R.norm(dim=1).mean()),
        "d_model": int(WU.shape[1]), "vocab": int(WU.shape[0]),
        "note": ("W_U is ANISOTROPIC: a single common direction carries most of every row's "
                 "norm, so ALL lenses inherit near-collinear concept pairs. The mean-removed "
                 "figure is a DIAGNOSTIC and is never applied to a lens vector or a write "
                 "(repo discipline 1: no centering, ever)."),
    }
    print(f"  anisotropy: raw mean cos={aniso['raw_mean_cos']:+.4f}  "
          f"mean-removed={aniso['mean_removed_cos_DIAGNOSTIC_ONLY']:+.4f}  "
          f"isotropic control={aniso['isotropic_control_mean_cos']:+.4f}  "
          f"common component={aniso['common_component_frac_of_row_norm']:.1%} of row norm")

    res = {}
    for lname, pairs in (("semantic", sem), ("random", rnd)):
        per_arm = {}
        for arm in ("logit", "jlens", "shuffled"):
            rows = []
            for l in band:
                J = None if arm == "logit" else \
                    lens.jacobians[l if arm == "jlens" else perm[l]].float()
                rows.extend(rho_for(pairs, WU, J, l))
            per_arm[arm] = {
                "median_abs_rho": med([r[0] for r in rows]),
                "mean_abs_rho": float(torch.tensor([r[0] for r in rows]).mean()) if rows else None,
                "median_kappa_V": med([r[1] for r in rows]),
                "frac_rho_above_0.9": (float(sum(1 for r in rows if r[0] > 0.9)) / len(rows))
                                      if rows else None,
                "median_norm_ratio": med([r[2] for r in rows]),
                "n": len(rows),
            }
        res[lname] = per_arm

    d_sem = res["semantic"]["jlens"]["median_abs_rho"] - res["semantic"]["logit"]["median_abs_rho"]
    d_rnd = res["random"]["jlens"]["median_abs_rho"] - res["random"]["logit"]["median_abs_rho"]
    d_shuf = res["semantic"]["shuffled"]["median_abs_rho"] - res["semantic"]["logit"]["median_abs_rho"]

    if d_sem <= -0.05:
        verdict = (f"CONFOUND CONFIRMED — median|rho| is {-d_sem:.4f} LOWER for the J-lens; "
                   "swap results must be reported at matched conditioning")
    elif abs(d_sem) < 0.05:
        verdict = (f"CONFOUND ABSENT — median|rho| differs by only {d_sem:+.4f}; "
                   "the swap benchmark is fair on this axis")
    else:
        verdict = (f"CONFOUND INVERTED — median|rho| is {d_sem:+.4f} HIGHER for the J-lens; "
                   "any swap advantage is achieved despite a geometric handicap")

    out = {
        "experiment": "T21 / E31 — swap-frame conditioning across lenses",
        "status": "EXPLORATORY",
        "prereg": "in-file header; decision rule fixed before the first number",
        "decision_rule": ("delta_med = median|rho|(jlens) - median|rho|(logit) on SEMANTIC pairs; "
                          "<=-0.05 confound confirmed / |.|<0.05 absent / >=+0.05 inverted"),
        "primary_outcome": "delta_med_semantic",
        "VERDICT": verdict,
        "delta_med_semantic": d_sem,
        "delta_med_random_C1": d_rnd,
        "delta_med_shuffled_C2": d_shuf,
        "C1_interpretation": ("if delta_med_random ~ delta_med_semantic the operator is GLOBALLY "
                              "whitening and the effect is not about concepts"),
        "C2_interpretation": ("if delta_med_shuffled ~ delta_med_semantic, conditioning is GENERIC "
                              "to any fitted Jacobian, not a property of layer identity"),
        "model": a.model, "lens": a.lens, "band": band, "n_layers": len(layers),
        "n_semantic_pairs": len(sem), "n_random_pairs": len(rnd), "seed": a.seed,
        "derangement_layer_map": {str(k): v for k, v in perm.items()},
        "anisotropy": aniso,
        "results": res,
        "env": {"torch": torch.__version__},
        "argv": sys.argv, "runtime_s": round(time.time() - t0, 1),
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)

    for lname in ("semantic", "random"):
        print(f"  {lname:>9}  " + "  ".join(
            f"{arm}: med|rho|={res[lname][arm]['median_abs_rho']:.4f} "
            f"kappa={res[lname][arm]['median_kappa_V']:.3f}"
            for arm in ("logit", "jlens", "shuffled")))
    print(f"\n  delta_med semantic = {d_sem:+.4f}   random(C1) = {d_rnd:+.4f}   "
          f"shuffled(C2) = {d_shuf:+.4f}")
    print(f"  VERDICT: {verdict}")
    print(f"wrote {a.out} ({out['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
