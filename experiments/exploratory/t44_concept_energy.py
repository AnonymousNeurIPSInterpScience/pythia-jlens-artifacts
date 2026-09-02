#!/usr/bin/env python3
"""t44_concept_energy.py — E50: concept-conditional activation energy. Targeting the 6.8%.

THE STRUCTURAL DIAGNOSIS THIS IMPLEMENTS
  Twenty predictors of the corpus effect have failed. F20 says the effect is an INTERACTION
  (corpus x concept-set = 6.8% of variance at 410M, 9.2% at 1B) against a corpus MAIN EFFECT of
  1.9% / 2.2%. Nearly every failed predictor was a scalar per corpus, which has nowhere to put
  "which concept" and can therefore only address the 1.9%.

  E45 was the exception -- s_c = u_c^T Sigma u_c with u_c = Delta^T W_U[c] DOES vary jointly in
  corpus and concept -- and it still died. But it died for a locatable reason: the corpus entered
  ONLY through the operator it produced. Sigma was the EVAL activation second moment, identical for
  every corpus. The corpus's own data distribution never appeared in the predictor.

  This is the first predictor in which it does.

THE HYPOTHESIS
  Concept c is readable under J^P iff the computation that produces c was EXERCISED by the prompts
  in P. Not "is the token frequent in P" -- that is eval_vocab_overlap, r = +0.118, already dead,
  and dead because frequency is a corpus marginal. The sharper claim: does P contain contexts that
  activate c's read-direction. Measured as

        A(P, c) = E_{x ~ P} [ <h_l(x), v_c>^2 ]      at the band layers

  High when P contains contexts that light up c's direction, low when it does not. It varies
  jointly in P (which activations) and c (which direction), so it is structurally an interaction
  rather than a marginal.

THREE DESIGN DECISIONS, fixed before running
  1. NO CIRCULARITY. Primary uses v_c = W_U[c], the raw unembedding row, so the predictor contains
     no operator at all. If v_c were (J^P)^T W_U[c] the read-direction would itself depend on P and
     the predictor would be partly circular. The J^P-transported form is a DISCLOSED SECONDARY.
  2. NORMALIZATION. Both sides z-scored WITHIN concept across corpora. That deletes the concept
     main effect (task difficulty, token frequency, direction norm) and leaves exactly the
     interaction term -- matching how the outcome grid is normalised.
  3. GRANULARITY. The outcome is PER TOKEN log-rank, not per eval set. Per-set gives 25 cells and
     5 leave-one-concept-out folds, which is too thin to be a real test. Per-token gives
     ~1310 x 5 = 6550 cells. Binary per-token pass@k would be too noisy (most tokens appear in one
     item), so the outcome is log(min-over-band rank), which is continuous.

WHY LEAVE-ONE-CONCEPT-OUT, NOT LEAVE-ONE-CORPUS-OUT
  Every prior failure collapsed on Github because Github carried nearly all the corpus-axis
  variance, so dropping it removed the signal. Those predictors had their signal IN the corpus
  axis, which made leave-one-corpus-out unpassable by construction. This predictor's signal is in
  the CONCEPT axis -- it predicts, within a corpus, which concepts get bought -- so
  leave-one-concept-out is the powered test and Github stops being load-bearing.

PRE-REGISTRATION — fixed before any number below existed
  PRIMARY   r( z(A) , z(-log rank) ) over the ~6550 (corpus, token) cells, both z-scored within
            concept. Positive r means: corpora whose activations energise a concept's direction
            read that concept better.
  RULE      * |r| >= 0.8 under leave-one-CONCEPT-out => ACCEPT.
            * |r| >= 0.8 full-sample but fails LOO-concept => LEVERAGE-DRIVEN.
            * |r| < 0.8 => NULL, reported as the strongest failed OBJECT-LEVEL account. That is a
              stronger negative than the corpus-scalar negatives and deepens the paper's section 7
              rather than weakening it.
  REQUIRED PATTERN (named in advance, not chosen after)
            Github must show HIGH A on order-ops' concept directions and LOW A on multihop's,
            matching the measured z of +1.3 / -1.9. A global correlation that does not reproduce
            this specific sign reversal does NOT count as ACCEPT.
  CONTROL C1 RANDOM DIRECTION: replace v_c with a random unit vector. Must be null (|r| < 0.3).
  CONTROL C2 DOSE: raw activation energy E||h||^2 with no concept projection. Must be WEAKER than
            the primary, or the effect is just "Github has big activations".
  BIAS      A is computed on the corpus's own activations at the fit positions (skip_first=16,
            max_seq_len=128) -- the same tokens J was averaged over. The outcome is measured on the
            EVAL battery. So the link is P's activation statistics -> J^P -> eval readout, a real
            causal chain but a three-step one.

    python t44_concept_energy.py --device cpu
"""
from __future__ import annotations

import argparse
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
BAND = list(range(9, 22))
CORPORA = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]
SETS = ["multihop", "multilingual", "order-ops", "poetry", "typo"]


def pear(x, y):
    n = len(x)
    if n < 3: return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in x)); sy = math.sqrt(sum((v - my) ** 2 for v in y))
    return sum((p - mx) * (q - my) for p, q in zip(x, y)) / (sx * sy) if sx * sy else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-docs", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e50_concept_energy.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of, rank_of

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    WU = hf.get_output_embeddings().weight.data.float()

    # ---- concept tokens, and which set each belongs to
    tok2set, cids = {}, []
    eval_items = []
    with torch.no_grad():
        for name in EVAL_SETS:
            if name not in SETS: continue
            for it in load_eval(name):
                tg = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
                if not tg: continue
                for t in tg:
                    for i in t:
                        tok2set.setdefault(i, name)
                ids = model.encode(it["prompt"], max_length=256)
                pos = readout_position(tok, name, it["prompt"])
                with ActivationRecorder(model.layers, at=BAND) as r:
                    model.forward(ids)
                    acts = {l: r.activations[l][0][pos].detach().float() for l in BAND}
                eval_items.append((name, acts, tg))
    cids = sorted(tok2set)
    V = WU[cids]
    V = V / V.norm(dim=1, keepdim=True).clamp_min(1e-9)          # unit read-directions
    print(f"{len(cids)} concept tokens, {len(eval_items)} eval items", flush=True)

    # ---- A(P, c): corpus activation energy along each concept direction
    g = torch.Generator().manual_seed(0)
    Vr = torch.randn(len(cids), WU.shape[1], generator=g)         # C1 random directions
    Vr = Vr / Vr.norm(dim=1, keepdim=True)
    A, Arand, dose = {}, {}, {}
    for c in CORPORA:
        texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "..", "corpora", f"{c}.jsonl"))]
        pool = [t for t in texts if len(tok(t).input_ids) >= 128][:a.n_docs]
        se = torch.zeros(len(cids)); sr = torch.zeros(len(cids)); nrm = 0.0; n = 0
        with torch.no_grad():
            for t in pool:
                ids = torch.tensor([tok(t).input_ids[:128]])
                with ActivationRecorder(model.layers, at=BAND) as r:
                    model.forward(ids)
                    for l in BAND:
                        H = r.activations[l][0][16:].float()       # skip_first=16, as fitted
                        se += (H @ V.T).pow(2).mean(0)
                        sr += (H @ Vr.T).pow(2).mean(0)
                        nrm += H.pow(2).sum(1).mean().item()
                        n += 1
        A[c] = (se / n); Arand[c] = (sr / n); dose[c] = nrm / n
        print(f"  A[{c:22s}] mean={A[c].mean():.3f}  dose={dose[c]:.1f}", flush=True)

    # ---- outcome: PER TOKEN log min-rank under each J^P
    def per_token_rank(J):
        acc = {i: [] for i in cids}
        with torch.no_grad():
            for name, acts, tg in eval_items:
                lg = [model.unembed(acts[l] @ J[l].T).float().cpu() for l in BAND]
                for t in tg:
                    for i in t:
                        rs = [rank_of(x, [i]) or 10**9 for x in lg]
                        acc[i].append(min(rs))
        return {i: math.log(statistics.median(v)) for i, v in acc.items() if v}

    R = {}
    for c in CORPORA:
        J = {int(k): v.float() for k, v in torch.load(
            os.path.join(HERE, "..", "results", "lenses", "e28", f"e28_{c}_410m_n400_s0.pt"), map_location="cpu")["J"].items()}
        R[c] = per_token_rank(J)
        print(f"  ranks scored for {c}", flush=True)

    # ---- z within concept across corpora, both sides
    def zwithin(vals):
        out = {}
        for j, i in enumerate(cids):
            v = [vals[c][j] if isinstance(vals[c], torch.Tensor) else vals[c].get(i) for c in CORPORA]
            v = [float(x) if x is not None else float("nan") for x in v]
            if any(math.isnan(x) for x in v): continue
            m, sd = statistics.mean(v), statistics.pstdev(v)
            for ci, c in enumerate(CORPORA):
                out[(c, i)] = (v[ci] - m) / sd if sd > 0 else 0.0
        return out

    zA, zAr = zwithin(A), zwithin(Arand)
    zY = zwithin({c: {i: -R[c].get(i, float("nan")) for i in cids} for c in CORPORA})
    keys = [k for k in zA if k in zY]
    print(f"\n{len(keys)} (corpus, token) cells with complete data", flush=True)

    r_main = pear([zA[k] for k in keys], [zY[k] for k in keys])
    r_rand = pear([zAr[k] for k in keys if k in zAr], [zY[k] for k in keys if k in zAr])
    # C2 dose: same value for every token within a corpus -> z within concept is degenerate,
    # so correlate raw dose against the per-cell outcome instead
    r_dose = pear([dose[k[0]] for k in keys], [zY[k] for k in keys])

    # leave-one-CONCEPT-out
    loo = []
    import random as _rnd
    _rnd.seed(0)
    toks = sorted({k[1] for k in keys})
    for t in _rnd.sample(toks, min(200, len(toks))):
        kk = [k for k in keys if k[1] != t]
        loo.append(pear([zA[k] for k in kk], [zY[k] for k in kk]))
    worst = min(loo, key=abs) if loo else float("nan")

    # REQUIRED PATTERN: Github high on order-ops, low on multihop
    def mean_z(corpus, setname, z):
        v = [z[(corpus, i)] for i in cids if tok2set.get(i) == setname and (corpus, i) in z]
        return statistics.mean(v) if v else float("nan")
    patt = {s: mean_z("Github", s, zA) for s in SETS}
    sign_ok = (patt.get("order-ops", 0) > 0) and (patt.get("multihop", 0) < 0)

    rec = {"experiment": "E50 — concept-conditional activation energy, targeting the 6.8% interaction",
           "prereg": "in-file docstring, fixed before running", "model": MODEL, "band": BAND,
           "n_cells": len(keys), "n_concept_tokens": len(cids),
           "r_primary": r_main, "control_C1_random_direction_r": r_rand,
           "control_C2_dose_r": r_dose,
           "loo_concept_worst": worst, "loo_concept_n": len(loo),
           "required_pattern_github_zA_by_set": patt,
           "required_pattern_sign_reversal_ok": sign_ok,
           "controls_fire": abs(r_rand) < 0.3 and abs(r_dose) < abs(r_main)}
    rec["VERDICT"] = (
        "ACCEPT — corpus activation energy along a concept's read-direction explains which "
        "concepts that corpus buys"
        if (abs(r_main) >= 0.8 and abs(worst) >= 0.8 and sign_ok and rec["controls_fire"]) else
        "LEVERAGE-DRIVEN — full-sample clears but leave-one-concept-out does not"
        if abs(r_main) >= 0.8 else
        "NULL — the strongest object-level account also fails; reported as such")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"\n  PRIMARY  r = {r_main:+.3f}  over {len(keys)} cells")
    print(f"  C1 random direction r = {r_rand:+.3f}  ({'fires' if abs(r_rand)<0.3 else 'DOES NOT FIRE'})")
    print(f"  C2 dose            r = {r_dose:+.3f}  ({'weaker, fires' if abs(r_dose)<abs(r_main) else 'NOT weaker'})")
    print(f"  worst LOO-concept  r = {worst:+.3f}")
    print(f"  required sign reversal (Github order-ops>0, multihop<0): {patt} -> {sign_ok}")
    print(f"\nVERDICT: {rec['VERDICT']}\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
