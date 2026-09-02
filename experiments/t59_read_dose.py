#!/usr/bin/env python3
"""t59_read_dose.py — E59: does the fit/read variance asymmetry survive matching the DOSE?

THE QUESTION
  The paper reports 91% fit / 7% read. The fit axis is manipulated with 200 documents x 128 tokens
  = 25,600 tokens per operator; the read axis with ONE 128-token prefix. That is a 200x dose ratio
  in favour of the axis that wins, and no prefix-length sweep exists, so nobody can say whether 7%
  is the read axis's ceiling or its value at the smallest dose anyone tried.

  E59 varies the read dose over 128 / 384 / 768 tokens with the operators, the battery, the band
  and the aggregation all held fixed, and reports the read share as a function of dose.

PRE-REGISTRATION: docs/experiments/preregs/E59_read_dose_response.md, written before this ran.
  PRIMARY   read_pct of the 7x7 two-way decomposition, per dose, under persist.
  RULE      DOSE-DRIVEN      read_pct(768) >= 2 x read_pct(128) AND monotone
            DOSE-INSENSITIVE delta < 2 points AND fit > read at every dose
            PARTIAL          anything between
  C1  dose-128 row ordering must match E52's 7-corpus row ordering at Spearman >= 0.8
  C2  the unfitted logit arm must move with dose, or the model is not using the longer context
  C3  zero prefix documents from any fitting pool, at every dose
  C4  cycling factor recorded per rung (StackExchange has 457 held-out >=768-token docs vs 541 items)

THE ONE DESIGN POINT THAT MATTERS
  The prefix pool is drawn ONCE per rung from documents of >= 768 tokens and then TRUNCATED to each
  dose. Drawing a fresh pool per dose would confound dose with document selection, because longer
  documents are a different sample of the corpus.

    python experiments/t59_read_dose.py --device cpu
    python experiments/t59_read_dose.py --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))
from provenance import write_result  # noqa: E402

MODEL = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))
K = [1, 2, 5, 10, 20, 50, 100]
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
# OOD_CommonPile is EXCLUDED: measured, it has 0 documents of >=768 tokens and 147 of >=384, so it
# cannot supply a longer read context at all. Declared bias 1 in the pre-registration.
CORPORA = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github",
           "OOD_News_2024", "OOD_arXiv_2023"]
DOSES = [128, 384, 768]
SEEDS = [0, 1, 2]
FIT_TOKENS = 128          # what every operator was fitted on, for the dose-ratio reporting
FIT_N = 200


def lens_path(c: str, s: int) -> str:
    stem = f"lens_{c}_410m_n200_s{s}.pt" if c.startswith("OOD_") else \
           f"lens_INSTREAM_{c}_410m_n200_s{s}.pt"
    return os.path.join(HERE, "..", "results", "e48", stem)


def derangement(band, seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        perm = [band[i] for i in torch.randperm(len(band), generator=g).tolist()]
        if all(p != l for p, l in zip(perm, band)):
            return perm


def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    rx, ry = rk(x), rk(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def shares(M, corp):
    n = len(corp)
    g = statistics.mean(M[(f, q)] for f in corp for q in corp)
    row = {f: statistics.mean(M[(f, q)] for q in corp) for f in corp}
    col = {q: statistics.mean(M[(f, q)] for f in corp) for q in corp}
    SSr = n * sum((row[f] - g) ** 2 for f in corp)
    SSc = n * sum((col[q] - g) ** 2 for q in corp)
    SSt = sum((M[(f, q)] - g) ** 2 for f in corp for q in corp)
    return {"fit_pct": 100 * SSr / SSt, "read_pct": 100 * SSc / SSt,
            "residual_pct": 100 * (SSt - SSr - SSc) / SSt,
            "row_means": row, "col_means": col,
            "fit_span": max(row.values()) - min(row.values()),
            "read_span": max(col.values()) - min(col.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--items-per-chunk", type=int, default=24)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rstrip", action="store_true",
                    help="THE CORRECTED READOUT. Apply anchor_evals' load-bearing .rstrip() to "
                         "the prompt before taking the readout position, matching "
                         "t52_factorial.py --rstrip and t48_crossover.py --rstrip. WITHOUT this "
                         "flag the run is on the LEGACY readout and its numbers are NOT "
                         "comparable to the corrected factorial. See the module docstring.")
    ap.add_argument("--out", default=None,
                    help="default: results/e59_read_dose_410m{_rstrip}.json, chosen by --rstrip "
                         "so a corrected run cannot silently overwrite the legacy artifact")
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(HERE, "..", "results",
                             f"e59_read_dose_410m{'_rstrip' if a.rstrip else ''}.json")
    # A corrected run must not land on the legacy path, and vice versa. The legacy artifact is
    # historical provenance: it is what the stored dose ladder was computed from, and overwriting
    # it would destroy the record of the defect rather than correct it.
    legacy = os.path.abspath(os.path.join(HERE, "..", "results", "e59_read_dose_410m.json"))
    if a.rstrip and os.path.abspath(a.out) == legacy:
        raise SystemExit("REFUSING: --rstrip may not write the legacy path "
                         "results/e59_read_dose_410m.json. Omit --out to get the _rstrip name.")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    d, L = model.d_model, len(BAND)
    HALF = L // 2

    corpora = CORPORA[:3] if a.smoke else CORPORA
    seeds = [0] if a.smoke else SEEDS
    doses = [128, 384] if a.smoke else DOSES
    max_dose = max(doses)

    # ---------------------------------------------------------------- eval items
    items, per_set_n = [], {}
    for name in EVAL_SETS:
        for it in load_eval(name):
            tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
            if not tgt:
                continue
            if a.smoke and per_set_n.get(name, 0) >= 8:
                continue
            per_set_n[name] = per_set_n.get(name, 0) + 1
            # THE READOUT CONVENTION. `.rstrip()` is load-bearing (anchor_evals.py:32-34): the
            # released prompts end in a trailing space that BPE absorbs into the target, so the
            # UNSTRIPPED final token occurs nowhere in the tokenised sequence, on 157 of 551 items.
            # Until 2026-09-01 this script had no --rstrip flag at all and always took the legacy
            # branch, so the stored dose ladder is legacy-scored while the headline factorial it
            # was cited to bound is corrected. That is the defect this flag exists to close.
            p = it["prompt"].rstrip() if a.rstrip else it["prompt"]
            items.append({"set": name, "ids": tok(p, add_special_tokens=True).input_ids,
                          "pos": readout_position(tok, name, p), "tgt": tgt})
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
    print(f"{len(items)} items, {P_n} pairs, doses {doses}, {len(corpora)} corpora", flush=True)

    # ---------------------------------------------------------------- held-out LONG prefix pools
    _fit_cache: dict[str, set] = {}

    def fit_documents(corpus: str) -> set:
        if corpus in _fit_cache:
            return _fit_cache[corpus]
        texts = [json.loads(l)["text"]
                 for l in open(os.path.join(HERE, "..", "corpora", f"{corpus}.jsonl"))]
        b = len(texts) // 3
        used = set()
        for s in SEEDS:
            pool = [t for t in texts[s * b:(s + 1) * b]
                    if len(tok(t).input_ids) >= FIT_TOKENS]
            used.update(pool[:FIT_N])
        _fit_cache[corpus] = used
        return used

    POOLS, POOL_META = {}, {}
    for c in corpora:
        texts = [json.loads(l)["text"]
                 for l in open(os.path.join(HERE, "..", "corpora", f"{c}.jsonl"))]
        banned = fit_documents(c)
        for ps in ([0] if a.smoke else SEEDS):
            g = torch.Generator().manual_seed(9000 + ps)
            out, n_banned_seen = [], 0
            for i in torch.randperm(len(texts), generator=g).tolist():
                t = texts[i]
                if t in banned:
                    n_banned_seen += 1
                    continue
                ids = tok(t).input_ids
                if len(ids) >= max_dose:
                    out.append(ids[:max_dose])
                if len(out) >= len(items):
                    break
            if not out:
                raise SystemExit(f"ABORT: {c} has no held-out document of >= {max_dose} tokens")
            cyc = len(items) / len(out)
            if len(out) < len(items):                    # C4: cycle, and RECORD it
                out = (out * (len(items) // len(out) + 1))[:len(items)]
            POOLS[(c, ps)] = out
            POOL_META[f"{c}|p{ps}"] = {"n_distinct_docs": len(set(map(tuple, out))),
                                       "cycling_factor": cyc,
                                       "n_fitting_docs_excluded_seen": n_banned_seen}
            print(f"  pool {c:20s} p{ps}  {len(set(map(tuple,out))):4d} distinct, "
                  f"cycling {cyc:.2f}x", flush=True)

    # C3: zero prefix documents may come from a fitting pool
    c3_hits = 0
    for (c, ps), pool in POOLS.items():
        banned_ids = {tuple(tok(t).input_ids[:max_dose]) for t in fit_documents(c)
                      if len(tok(t).input_ids) >= max_dose}
        c3_hits += sum(1 for p in pool if tuple(p) in banned_ids)
    print(f"C3 prefix/fit overlap: {c3_hits} documents  "
          f"{'FIRES' if c3_hits == 0 else 'DOES NOT FIRE'}", flush=True)

    # ---------------------------------------------------------------- operators
    missing = [lens_path(c, s) for c in corpora for s in seeds if not os.path.exists(lens_path(c, s))]
    if missing:
        raise SystemExit("ABORT: missing lenses:\n  " + "\n  ".join(missing))
    JF = {(c, s): {l: torch.load(lens_path(c, s), map_location="cpu",
                                 weights_only=True)["J"][l].float() for l in BAND}
          for c in corpora for s in seeds}
    JSHUF = {l: JF[(corpora[0], seeds[0])][q]
             for l, q in zip(BAND, derangement(BAND, 7000))}

    def build(rung, ps, dose):
        pref = [p[:dose] for p in POOLS[(rung, ps)]]
        out = []
        for it, pre in zip(items, pref):
            body = it["ids"][1:] if it["ids"] and it["ids"][0] == tok.bos_token_id else it["ids"]
            out.append(([tok.bos_token_id] + list(pre) + list(body),
                        it["pos"] if it["pos"] < 0 else it["pos"] + len(pre)))
        return out

    def cache(cell):
        A = torch.empty(len(items), L, d)
        with torch.no_grad():
            for ii, (ids, pos) in enumerate(cell):
                t = torch.tensor([ids], device=a.device)
                with ActivationRecorder(model.layers, at=BAND) as rec:
                    model.forward(t)
                    A[ii] = torch.stack([rec.activations[l][0][pos].detach().float() for l in BAND])
        return A

    def score(A, T):
        H = A if T is None else torch.stack([A[:, j, :] @ T[BAND[j]].T for j in range(L)], dim=1)
        flat = H.reshape(-1, d)
        R = torch.empty(P_n, L, dtype=torch.float32)
        with torch.no_grad():
            for i0 in range(0, len(items), a.items_per_chunk):
                i1 = min(i0 + a.items_per_chunk, len(items))
                lg = model.unembed(flat[i0 * L:i1 * L]).float()
                for ii in range(i0, i1):
                    block = lg[(ii - i0) * L:(ii - i0 + 1) * L]
                    for pi in ITEM_PAIRS[ii]:
                        cand = torch.stack([(block > block[:, i:i + 1]).sum(1) + 1
                                            for i in pair_tgt[pi]])
                        R[pi] = cand.min(0).values.float()
        mn = R.min(dim=1).values
        return {"min": torch.stack([(mn <= k).float() for k in K]).mean(0),
                "persist": torch.stack([((R <= k).float().sum(1) >= HALF).float()
                                        for k in K]).mean(0)}

    def adm(v):
        return statistics.mean([float(v[SET_IDX[s]].mean()) for s in ADMITTED])

    # ---------------------------------------------------------------- the sweep
    CELL, t0 = {}, time.time()
    for dose in doses:
        for rung in corpora:
            for ps in ([0] if a.smoke else SEEDS):
                A = cache(build(rung, ps, dose))
                arms = {"logit": score(A, None), "shuf": score(A, JSHUF)}
                for c in corpora:
                    for s in seeds:
                        arms[f"{c}|s{s}"] = score(A, JF[(c, s)])
                CELL[(dose, rung, ps)] = arms
                print(f"  dose={dose:4d} read={rung:20s} p{ps}  "
                      f"logit={adm(arms['logit']['persist']):.5f} "
                      f"shuf={adm(arms['shuf']['persist']):.5f}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)

    # ---------------------------------------------------------------- decomposition per dose
    pseeds = [0] if a.smoke else SEEDS
    by_dose = {}
    for dose in doses:
        by_dose[str(dose)] = {}
        for ag in ("persist", "min"):
            M = {(f, q): statistics.mean(adm(CELL[(dose, q, ps)][f"{f}|s{s}"][ag])
                                         for s in seeds for ps in pseeds)
                 for f in corpora for q in corpora}
            sh = shares(M, corpora)
            sh["matrix"] = {f"{f}|{q}": v for (f, q), v in M.items()}
            sh["logit_arm_mean"] = statistics.mean(
                adm(CELL[(dose, q, ps)]["logit"][ag]) for q in corpora for ps in pseeds)
            sh["logit_arm_by_read_context"] = {
                q: statistics.mean(adm(CELL[(dose, q, ps)]["logit"][ag]) for ps in pseeds)
                for q in corpora}
            sh["shuf_arm_mean"] = statistics.mean(
                adm(CELL[(dose, q, ps)]["shuf"][ag]) for q in corpora for ps in pseeds)
            # per-draw shares, so the dose curve carries a spread
            per_draw = []
            for s in seeds:
                for ps in pseeds:
                    Md = {(f, q): adm(CELL[(dose, q, ps)][f"{f}|s{s}"][ag])
                          for f in corpora for q in corpora}
                    sd = shares(Md, corpora)
                    per_draw.append({"fit_pct": sd["fit_pct"], "read_pct": sd["read_pct"]})
            sh["per_draw"] = per_draw
            sh["fit_pct_draw_range"] = [min(x["fit_pct"] for x in per_draw),
                                        max(x["fit_pct"] for x in per_draw)]
            sh["read_pct_draw_range"] = [min(x["read_pct"] for x in per_draw),
                                         max(x["read_pct"] for x in per_draw)]
            by_dose[str(dose)][ag] = sh

    # ---------------------------------------------------------------- controls + adjudication
    # C1 must reference the factorial scored under THE SAME readout convention. Comparing a
    # corrected dose ladder against the legacy factorial would make the control test the readout
    # rather than the pool, which is the confusion this whole correction exists to remove.
    e52_name = f"e52_factorial_410m{'_rstrip' if a.rstrip else ''}.json"
    e52 = json.load(open(os.path.join(HERE, "..", "results", e52_name)))
    ref_rows = {c: statistics.mean(e52["by_aggregation"]["persist"]["matrix"][f"{c}|{q}"]
                                   for q in corpora) for c in corpora}
    got_rows = by_dose[str(doses[0])]["persist"]["row_means"]
    c1_rho = spearman([ref_rows[c] for c in corpora], [got_rows[c] for c in corpora])
    c1 = {"spearman_row_ordering_vs_e52": c1_rho, "threshold": 0.8, "fires": c1_rho >= 0.8,
          "reference": f"results/{e52_name}",
          "note": "cell VALUES cannot match E52 — the pool is the >=768-token subset — so the "
                  "control is on the row ORDERING. The reference file tracks --rstrip."}
    lg = [by_dose[str(x)]["persist"]["logit_arm_mean"] for x in doses]
    c2 = {"logit_arm_by_dose": dict(zip(map(str, doses), lg)),
          "range": max(lg) - min(lg), "monotone": lg == sorted(lg) or lg == sorted(lg, reverse=True),
          "fires": (max(lg) - min(lg)) > 0.0,
          "note": "the unfitted arm has no operator, so its movement is pure context effect and "
                  "bounds how much the read axis can move at all"}

    rp = [by_dose[str(x)]["persist"]["read_pct"] for x in doses]
    fp = [by_dose[str(x)]["persist"]["fit_pct"] for x in doses]
    delta = rp[-1] - rp[0]
    monotone = all(b >= x for x, b in zip(rp, rp[1:]))
    if not c1["fires"]:
        verdict = ("UNCLEAR — C1 did not fire; the long-document subsample does not preserve the "
                   "fit-axis ordering, so dose is confounded with document selection")
    elif rp[-1] >= 2 * rp[0] and monotone:
        verdict = (f"DOSE-DRIVEN — read share rises from {rp[0]:.1f}% to {rp[-1]:.1f}% over a 6x "
                   f"dose increase, monotonically. The 91/7 headline is substantially a statement "
                   f"about unequal manipulation strength and must be restated as such.")
    elif abs(delta) < 2.0 and all(f > r for f, r in zip(fp, rp)):
        verdict = (f"DOSE-INSENSITIVE — read share moves only {delta:+.1f} points over a 6x dose "
                   f"increase ({rp[0]:.1f}% -> {rp[-1]:.1f}%) and the fit axis dominates at every "
                   f"dose. The asymmetry is not an artifact of dose.")
    else:
        verdict = (f"PARTIAL — read share moves {delta:+.1f} points ({rp[0]:.1f}% -> {rp[-1]:.1f}%), "
                   f"monotone={monotone}. Report the curve; the headline is dose-dependent with "
                   f"this slope.")

    rec = {
        "experiment": "E59 — the read-axis dose-response",
        "prereg": "docs/experiments/preregs/E59_read_dose_response.md, written before this ran",
        "status": "PRE-REGISTERED",
        "question": ("is the 91%/7% fit-vs-read asymmetry a property of the method, or of a 200x "
                     "dose ratio between the two axes?"),
        "model": MODEL, "band": BAND, "K": K, "admitted_sets": ADMITTED,
        "corpora": corpora, "excluded_corpus": {
            "OOD_CommonPile": "0 documents of >=768 tokens, 147 of >=384 — physically cannot "
                              "supply a longer read context. Measured 2026-08-16."},
        "doses": doses, "fit_seeds": seeds, "prefix_seeds": pseeds,
        "n_items": len(items), "n_pairs": P_n, "smoke": a.smoke,
        "rstrip": bool(a.rstrip),
        "readout": ("CORRECTED — item['prompt'].rstrip() before readout_position, matching "
                    "t52_factorial.py --rstrip and t48_crossover.py --rstrip" if a.rstrip else
                    "LEGACY — the UNSTRIPPED prompt. NOT comparable to the corrected factorial; "
                    "see anchor_evals.py:32-34 and the module docstring."),
        "dose_ratio_fit_vs_read": {str(x): (FIT_N * FIT_TOKENS) / x for x in doses},
        "operators": "results/e48/lens_*_410m_n200_s*.pt — identical to E52's; only the read "
                     "axis moves",
        "prefix_pool_construction": ("drawn ONCE per rung from held-out documents of >= "
                                     f"{max_dose} tokens, then TRUNCATED to each dose, so the "
                                     "same documents are read at every dose"),
        "pool_meta": POOL_META,
        "by_dose": by_dose,
        "controls": {"C1_row_ordering_vs_e52": c1, "C2_logit_arm_moves_with_dose": c2,
                     "C3_prefix_fit_overlap": {"n_documents": c3_hits, "fires": c3_hits == 0},
                     "C4_cycling": {k: v["cycling_factor"] for k, v in POOL_META.items()}},
        "VERDICT": verdict,
    }
    write_result(a.out, rec, experiment="E59",
                 inputs=[lens_path(c, s) for c in corpora for s in seeds])

    print("\n" + "=" * 78)
    for ag in ("persist", "min"):
        print(f"  {ag}:")
        for x in doses:
            b = by_dose[str(x)][ag]
            print(f"    dose {x:4d}  fit={b['fit_pct']:6.2f}%  read={b['read_pct']:6.2f}%  "
                  f"resid={b['residual_pct']:5.2f}%  logit={b['logit_arm_mean']:.5f}  "
                  f"(read draw range {b['read_pct_draw_range'][0]:.1f}-{b['read_pct_draw_range'][1]:.1f})")
    print(f"\nC1 row ordering vs E52: rho={c1_rho:+.3f} -> {'FIRES' if c1['fires'] else 'DOES NOT FIRE'}")
    print(f"C2 logit arm range over dose: {c2['range']:.5f} -> {'FIRES' if c2['fires'] else 'DOES NOT FIRE'}")
    print(f"C3 prefix/fit overlap: {c3_hits} -> {'FIRES' if c3_hits == 0 else 'DOES NOT FIRE'}")
    print(f"\nVERDICT: {verdict}\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
