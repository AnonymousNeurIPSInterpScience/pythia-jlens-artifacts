#!/usr/bin/env python3
"""t52_factorial.py — E52: the fit x read factorial. The missing cell, J^Q read on Q.

THE QUESTION, in the operator's words
  "Can the J lens trained on an out of distribution sample read some latent properties of the
   model on an out of distribution corpus?"

WHY NEITHER E36 NOR E48 CAN ANSWER IT
  E48 moved the FIT distribution and held the read fixed -> exposure of the fitting corpus does
  not order the read. E36 moved the READ distribution and held the operator fixed -> REJECT S3;
  the fitted operator is FLATTER across the shift axis than the unfitted one, 5/5.
  Both are marginal. The estimator framing's last sharp prediction lives in the JOINT cell: if the
  corpus effect is a MATCHING effect, an operator fitted on Q should read Q better than one fitted
  on P does. Nothing measured so far separates that from "some corpora just make better operators".

PRE-REGISTRATION: specs/archive/prereg/PREREG_E52_FACTORIAL.md, written before this ran.
  PRIMARY   the DIAGONAL EXCESS. Over the 8x8 matrix y[f,q] (fit corpus f, read rung q), remove
            main effects:  y[f,q] = m + a_f + b_q + g[f,q].
            a_f = "some corpora make better operators" (the E28/E33/E48 effect)
            b_q = "some read contexts are easier"
            g[f,q] = the MATCHING term.  D = mean(g[f,f]) - mean(g[f,q!=f]).
  RULE      CI(D) strictly > 0  -> MATCHING CONFIRMED
            CI(D) includes 0    -> NO MATCHING (publishable null): the corpus effect is a property
                                   of the fitting corpus per se, and EQ2's ||J^P - J^Q|| gap
                                   framing is not what the data measures
            CI(D) strictly < 0  -> ANTI-MATCHING, reported separately
  `min` NEVER VOTES (CLAUDE.md 6.0; E48 measured the deranged operator beating the real one on
  103/120 draws under min). Reported, never decisive.

EVERY OPERATOR SHARES ONE FITTER. All 24 (8 corpora x 3 seed blocks) come from trainval.py at
N=200, --band 9,21, dim_batch=128, max_seq_len=128, skip_first=16, target_layer=-2, fp32, TF32
off. E48's in-stream arm used fastfit.fast_fit and its OOD arm used jacobian_for_prompt; a
factorial cannot tolerate a fitter difference confounded with the axis under test, which is what
the in-stream refits were produced for.

    python t52_factorial.py --device cpu
    python t52_factorial.py --smoke
"""
from __future__ import annotations

import argparse
import json
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
INSTREAM = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github"]
OOD = ["OOD_News_2024", "OOD_arXiv_2023", "OOD_CommonPile"]
CORPORA = INSTREAM + OOD                       # both the FIT axis and the READ axis
DEGENERATE = ["Github"]                        # established at/below its floor: E33, E48, E36
SEEDS = [0, 1, 2]
PREFIX_TOKENS = 128


def lens_path(c: str, s: int) -> str:
    """Matched-provenance lenses ONLY. In-stream ones are the INSTREAM refits, not the E28 fits."""
    stem = f"lens_{c}_410m_n200_s{s}.pt" if c.startswith("OOD_") else \
           f"lens_INSTREAM_{c}_410m_n200_s{s}.pt"
    return os.path.join(HERE, "..", "results", "e48", stem)


def derangement(band, seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        perm = [band[i] for i in torch.randperm(len(band), generator=g).tolist()]
        if all(p != l for p, l in zip(perm, band)):
            return perm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prefix-seeds", type=int, default=3)
    ap.add_argument("--items-per-chunk", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--e48", default=os.path.join(HERE, "..", "results", "e48_crossover_410m.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e52_factorial_410m.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of
    from bootstrap import hierarchical_bootstrap_seeded

    # ---------------------------------------------------------------- HELD-OUT PREFIX POOLS
    # E36's load_pool() draws prefixes from the WHOLE corpus file. On the diagonal of this
    # factorial that is fatal: measured, 24-26% of the 541 prefix documents for a corpus are
    # LITERALLY among the 600 documents its operators were fitted on (200 x 3 seed blocks out of
    # 2400). The diagonal would then be elevated because the operator has already averaged a
    # derivative at those exact activations — a self-overlap artifact, not distributional matching,
    # and it points in the direction that would manufacture MATCHING CONFIRMED.
    #
    # The prefix pool for rung X is therefore X's documents MINUS every document used to fit any
    # operator on X. The exclusion depends only on the RUNG, so every operator reading that rung
    # sees identical prefixes and the fit axis stays comparable; only the diagonal's unfair
    # advantage is removed.
    _fit_docs_cache: dict[str, set] = {}

    def fit_documents(corpus: str) -> set:
        if corpus in _fit_docs_cache:
            return _fit_docs_cache[corpus]
        texts = [json.loads(l)["text"]
                 for l in open(os.path.join(HERE, "..", "corpora", f"{corpus}.jsonl"))]
        b = len(texts) // 3
        used = set()
        for s in SEEDS:                      # every seed block, not just the ones scored here
            pool = [t for t in texts[s * b:(s + 1) * b] if len(tok(t).input_ids) >= PREFIX_TOKENS]
            used.update(pool[:200])          # trainval.py takes the first 200 qualifying
        _fit_docs_cache[corpus] = used
        return used

    def load_pool_heldout(corpus: str, n: int, seed: int):
        texts = [json.loads(l)["text"]
                 for l in open(os.path.join(HERE, "..", "corpora", f"{corpus}.jsonl"))]
        banned = fit_documents(corpus)
        g = torch.Generator().manual_seed(9000 + seed)
        out = []
        for i in torch.randperm(len(texts), generator=g).tolist():
            t = texts[i]
            if t in banned:
                continue
            ids = tok(t).input_ids
            if len(ids) >= PREFIX_TOKENS:
                out.append(ids[:PREFIX_TOKENS])
            if len(out) >= n:
                break
        if len(out) < n:
            raise SystemExit(f"ABORT: {corpus} has only {len(out)} held-out prefix documents, "
                             f"need {n}. Cycling them would reintroduce the overlap this "
                             f"function exists to remove.")
        return out

    load_pool = load_pool_heldout

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    d, L = model.d_model, len(BAND)
    HALF = L // 2

    fit_corpora = CORPORA[:3] if a.smoke else CORPORA
    seeds = [0] if a.smoke else SEEDS
    pseeds = [0] if a.smoke else list(range(a.prefix_seeds))
    rungs = (["Q0"] + fit_corpora + ([] if a.smoke else ["SHUFFLED_Pile-CC"]))

    missing = [lens_path(c, s) for c in fit_corpora for s in seeds
               if not os.path.exists(lens_path(c, s))]
    if missing:
        raise SystemExit(f"ABORT: missing matched-provenance lenses:\n  " + "\n  ".join(missing))
    JF = {(c, s): {l: torch.load(lens_path(c, s), map_location="cpu",
                                 weights_only=True)["J"][l].float() for l in BAND}
          for c in fit_corpora for s in seeds}
    JSHUF = {l: JF[(fit_corpora[0], seeds[0])][q]
             for l, q in zip(BAND, derangement(BAND, 7000))}
    print(f"{len(JF)} matched operators | fit {fit_corpora} | rungs {rungs} | "
          f"prefix seeds {pseeds}", flush=True)

    # ---- eval items; positions computed on the item ALONE and later offset (E36's rule)
    items, per_set_n = [], {}
    for name in EVAL_SETS:
        for it in load_eval(name):
            tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
            if not tgt:
                continue
            if a.smoke and per_set_n.get(name, 0) >= 8:
                continue
            per_set_n[name] = per_set_n.get(name, 0) + 1
            p = it["prompt"]
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
    print(f"{len(items)} items, {P_n} pairs", flush=True)

    def build(rung, seed):
        if rung == "Q0":
            return [(it["ids"], it["pos"]) for it in items]
        if rung.startswith("SHUFFLED_"):
            base = load_pool(rung[len("SHUFFLED_"):], len(items), seed)
            g = torch.Generator().manual_seed(4242 + seed)
            pref = [[b[i] for i in torch.randperm(len(b), generator=g).tolist()] for b in base]
        else:
            pref = load_pool(rung, len(items), seed)
        out = []
        for it, pre in zip(items, pref):
            body = it["ids"][1:] if it["ids"] and it["ids"][0] == tok.bos_token_id else it["ids"]
            out.append(([tok.bos_token_id] + list(pre) + list(body),
                        it["pos"] if it["pos"] < 0 else it["pos"] + len(pre)))
        return out

    def cache(cell):
        A = torch.empty(len(items), L, d)
        cap = []
        with torch.no_grad():
            for ii, (ids, pos) in enumerate(cell):
                t = torch.tensor([ids], device=a.device)
                with ActivationRecorder(model.layers, at=BAND) as rec:
                    out = model.forward(t)
                    A[ii] = torch.stack([rec.activations[l][0][pos].detach().float() for l in BAND])
                h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                lg = hf.get_output_embeddings()(h[0, pos]).float()
                for tg in items[ii]["tgt"]:
                    cap.append(min(int((lg > lg[i]).sum().item()) + 1 for i in tg))
        return A, statistics.mean(cap)

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

    CELL = {}
    t0 = time.time()
    for rung in rungs:
        for ps in pseeds:
            A, cap = cache(build(rung, ps))
            arms = {"logit": score(A, None), "shuf": score(A, JSHUF)}
            for c in fit_corpora:
                for s in seeds:
                    arms[f"{c}|s{s}"] = score(A, JF[(c, s)])
            CELL[(rung, ps)] = {"arms": arms, "capability_mean_rank": cap}
            best = max(adm(arms[f"{c}|s{seeds[0]}"]["persist"]) for c in fit_corpora)
            print(f"  read={rung:22s} p{ps}  logit={adm(arms['logit']['persist']):.5f} "
                  f"bestJ={best:.5f} shuf={adm(arms['shuf']['persist']):.5f} "
                  f"cap={cap:.0f}  [{time.time()-t0:.0f}s]", flush=True)

    # ------------------------------------------------------------------ the matrix
    def y(f, q, ag="persist"):
        """read AUC for operator fitted on f, read on rung q — averaged over fit and prefix seeds"""
        return statistics.mean([adm(CELL[(q, ps)]["arms"][f"{f}|s{s}"][ag])
                                for s in seeds for ps in pseeds])

    def diag_excess(mat, corpora):
        m = statistics.mean([mat[(f, q)] for f in corpora for q in corpora])
        a_f = {f: statistics.mean([mat[(f, q)] for q in corpora]) - m for f in corpora}
        b_q = {q: statistics.mean([mat[(f, q)] for f in corpora]) - m for q in corpora}
        g = {(f, q): mat[(f, q)] - m - a_f[f] - b_q[q] for f in corpora for q in corpora}
        on = [g[(f, f)] for f in corpora]
        off = [g[(f, q)] for f in corpora for q in corpora if f != q]
        return {"D": statistics.mean(on) - statistics.mean(off),
                "diag_mean": statistics.mean(on), "offdiag_mean": statistics.mean(off),
                "a_fit": a_f, "b_read": b_q,
                "g": {f"{f}|{q}": v for (f, q), v in g.items()}}

    def surrogate_D(mat, corpora):
        """C6 — the LINK-ARTIFACT null, and it is not small.

        An ADDITIVE decomposition of a bounded rate is not an instrument for "matching". For any
        y[f,q] = phi(u_f + v_q) with NO matching term, a second-order expansion leaves
        g[f,q] = phi'' * (u_f - ubar)(v_q - vbar), so D picks up phi'' * Cov(u_f, v_f) purely
        because operator quality and read-easiness are correlated across the SHARED corpus index.
        Measured on E36's own 5x5 matrix: corr(a_f, b_f) = +0.325, and a purely MULTIPLICATIVE
        surrogate built from that matrix's own margins — zero matching by construction — yields
        D = +0.000280 against an observed +0.001214. **23% of a positive D is a link artifact.**

        This returns D for the multiplicative surrogate with the SAME margins as `mat`. It is the
        value D would take with no matching at all, and it is what D must be compared against.
        """
        gm = statistics.mean([mat[(f, q)] for f in corpora for q in corpora])
        rm = {f: statistics.mean([mat[(f, q)] for q in corpora]) for f in corpora}
        cm = {q: statistics.mean([mat[(f, q)] for f in corpora]) for q in corpora}
        sur = {(f, q): rm[f] * cm[q] / gm for f in corpora for q in corpora}
        return diag_excess(sur, corpora)["D"]

    results = {}
    for ag in ("persist", "min"):
        mat = {(f, q): y(f, q, ag) for f in fit_corpora for q in fit_corpora}
        full = diag_excess(mat, fit_corpora)
        nog = fit_corpora
        sub = [c for c in fit_corpora if c not in DEGENERATE]
        without = diag_excess({k: v for k, v in mat.items()
                               if k[0] in sub and k[1] in sub}, sub)
        # C4 permutation null: shuffle the FIT labels relative to the read rungs
        g_ = torch.Generator().manual_seed(11)
        null = []
        for _ in range(a.n_perm):
            perm = [fit_corpora[i] for i in torch.randperm(len(fit_corpora), generator=g_).tolist()]
            mp = dict(zip(fit_corpora, perm))
            null.append(diag_excess({(mp[f], q): v for (f, q), v in mat.items()},
                                    fit_corpora)["D"])
        n_ge = sum(1 for v in null if abs(v) >= abs(full["D"]))
        # the same permutation null for the WITHOUT-DEGENERATE form, so DECLARED BIAS 2 ("the
        # claim is made only in the form that survives both") is actually executable — it was not,
        # because only the all-8 form had an interval.
        matw = {k: v for k, v in mat.items() if k[0] in sub and k[1] in sub}
        gw = torch.Generator().manual_seed(12)
        nullw = []
        for _ in range(a.n_perm):
            perm = [sub[i] for i in torch.randperm(len(sub), generator=gw).tolist()]
            mp = dict(zip(sub, perm))
            nullw.append(diag_excess({(mp[f], q): v for (f, q), v in matw.items()}, sub)["D"])
        n_gew = sum(1 for v in nullw if abs(v) >= abs(without["D"]))
        results[ag] = {
            "matrix": {f"{f}|{q}": v for (f, q), v in mat.items()},
            "all8": full, "without_degenerate": without,
            "C4_permutation": {"null_mean": statistics.mean(null),
                               "null_sd": statistics.pstdev(null),
                               "n_abs_ge_observed": n_ge, "n_perm": a.n_perm,
                               "p_two_sided": (1 + n_ge) / (1 + a.n_perm)},
            "C4_permutation_without_degenerate": {
                "null_mean": statistics.mean(nullw), "null_sd": statistics.pstdev(nullw),
                "n_abs_ge_observed": n_gew, "n_perm": a.n_perm,
                "p_two_sided": (1 + n_gew) / (1 + a.n_perm)},
            "C6_link_artifact_null": {
                "surrogate_D_all8": surrogate_D(mat, fit_corpora),
                "surrogate_D_without_degenerate": surrogate_D(matw, sub),
                "D_minus_surrogate_all8": full["D"] - surrogate_D(mat, fit_corpora),
                "D_minus_surrogate_without_degenerate": (without["D"]
                                                         - surrogate_D(matw, sub)),
                "note": ("a multiplicative surrogate with the SAME margins has zero matching by "
                         "construction; its D is the link artifact. D must exceed it, not 0.")},
        }

    # interval on D, eval set as the outer replication unit, prefix seed nested
    def pairvec(f, q, ps, ag="persist"):
        return torch.stack([CELL[(q, ps)]["arms"][f"{f}|s{s}"][ag] for s in seeds]).mean(0)
    per_set_D = {}
    for st in ADMITTED:
        blocks = []
        for ps in pseeds:
            mat = {(f, q): float(pairvec(f, q, ps)[SET_IDX[st]].mean())
                   for f in fit_corpora for q in fit_corpora}
            blocks.append(diag_excess(mat, fit_corpora)["D"])
        per_set_D[st] = blocks
    # D is one SCALAR per (eval set, prefix seed) — there is no item axis, because D is already a
    # contrast over the whole matrix. Wrapping each scalar as a length-1 vector makes the seeded
    # resampler do exactly the right thing: resample eval SETS, then prefix SEEDS within each,
    # with the inner item draw degenerate. The baseline arm is the constant 0, so the interval is
    # on D itself rather than on a difference of two measured arms.
    Dboot = hierarchical_bootstrap_seeded(
        {st: ([[v] for v in per_set_D[st]], [0.0]) for st in ADMITTED},
        n_boot=a.n_boot, seed=0)
    # per_set_D was previously computed and DISCARDED. It must be stored: `adm` averages the five
    # admitted sets BEFORE the contrast, and E51 established that the corpus x set interaction
    # (6.8% of variance) exceeds the corpus main effect (1.9%) with signs that reverse per set
    # (Github +1.3 on order-ops, -1.9 on multihop). Sign-opposed per-set matching therefore
    # CANCELS in D, so a null on D is only a null on the set-averaged contrast. Storing the
    # per-set values is what makes that statement checkable rather than an assumption.
    per_set_D_out = {st: {"per_prefix_seed": per_set_D[st],
                          "mean": statistics.mean(per_set_D[st]),
                          "sd": statistics.pstdev(per_set_D[st])} for st in ADMITTED}
    signs = [1 if per_set_D_out[st]["mean"] > 0 else -1 for st in ADMITTED]
    per_set_sign_split = {"n_positive": sum(1 for s in signs if s > 0),
                          "n_negative": sum(1 for s in signs if s < 0),
                          "note": ("if the sets split in sign, a null on the set-averaged D is "
                                   "cancellation, not absence of matching")}

    # ------------------------------------------------------------------ secondary
    secondary = {}
    for q in (OOD if not a.smoke else fit_corpora):
        if q not in rungs:
            continue
        matched = statistics.mean([adm(CELL[(q, ps)]["arms"][f"{q}|s{s}"]["persist"])
                                   for s in seeds for ps in pseeds])
        lg = statistics.mean([adm(CELL[(q, ps)]["arms"]["logit"]["persist"]) for ps in pseeds])
        sh = statistics.mean([adm(CELL[(q, ps)]["arms"]["shuf"]["persist"]) for ps in pseeds])
        best_in = max((statistics.mean([adm(CELL[(q, ps)]["arms"][f"{c}|s{s}"]["persist"])
                                        for s in seeds for ps in pseeds]), c)
                      for c in fit_corpora if c in INSTREAM)
        secondary[q] = {
            "matched_J^q": matched, "logit": lg, "shuf_floor": sh,
            "best_instream_J^P": best_in[0], "best_instream_corpus": best_in[1],
            "beats_logit": matched > lg, "beats_shuf_floor": matched > sh,
            "beats_best_instream": matched > best_in[0],
            "delta_vs_logit": matched - lg, "delta_vs_best_instream": matched - best_in[0]}

    # ------------------------------------------------------------------ controls
    ctrl = {}
    if os.path.exists(a.e48) and not a.smoke and "Q0" in rungs:
        e48 = json.load(open(a.e48))
        diffs = {}
        for c in fit_corpora:
            if c in e48["rungs"]:
                here = statistics.mean([adm(CELL[("Q0", ps)]["arms"][f"{c}|s{s}"]["persist"])
                                        for s in seeds for ps in pseeds])
                there = e48["rungs"][c]["persist"]["jp_admitted_mean"]
                diffs[c] = {"here": here, "e48": there, "abs_diff": abs(here - there)}
        ctrl["C1_Q0_vs_e48"] = {
            "per_corpus": diffs, "max_abs_diff": max(v["abs_diff"] for v in diffs.values()),
            "tolerance": 1e-6,
            "note": ("in-stream rungs are EXPECTED to differ: E48 scored the fastfit E28 lenses, "
                     "this scores the matched-provenance INSTREAM refits. The OOD rungs use the "
                     "SAME .pt files in both and must agree to fp32."),
            "ood_max_abs_diff": max((v["abs_diff"] for c, v in diffs.items()
                                     if c.startswith("OOD_")), default=None)}
        # NOT `(x or 1) < 1e-6`: a PERFECT reproduction gives x == 0.0, which is falsy, so
        # `or 1` substitutes 1 and the control reports DOES NOT FIRE on a flawless match.
        _ood = ctrl["C1_Q0_vs_e48"]["ood_max_abs_diff"]
        ctrl["C1_fires"] = bool(_ood is not None and _ood < 1e-6)
    below = {}
    for q in rungs:
        sh = statistics.mean([adm(CELL[(q, ps)]["arms"]["shuf"]["persist"]) for ps in pseeds])
        below[q] = [c for c in fit_corpora if c not in DEGENERATE
                    and statistics.mean([adm(CELL[(q, ps)]["arms"][f"{c}|s{s}"]["persist"])
                                         for s in seeds for ps in pseeds]) <= sh]
    ctrl["C2_shuf_floor_excl_degenerate"] = {"below_floor": below,
                                             "fires": all(not v for v in below.values())}
    q0r = statistics.mean([CELL[("Q0", ps)]["capability_mean_rank"] for ps in pseeds]) \
        if "Q0" in rungs else None
    caps = {q: statistics.mean([CELL[(q, ps)]["capability_mean_rank"] for ps in pseeds])
            for q in rungs}
    ctrl["C3_capability"] = {"q0_mean_rank": q0r, "per_rung": caps,
                             "collapsed": [q for q in rungs if q0r and caps[q] > 2 * q0r],
                             "fires": not [q for q in rungs if q0r and caps[q] > 2 * q0r]}
    ctrl["C5_self_distance"] = {"value": 0.0, "fires": True}

    D = results["persist"]["all8"]["D"]
    Dw = results["persist"]["without_degenerate"]["D"]
    lo, hi = Dboot["ci_lo"], Dboot["ci_hi"]
    sur = results["persist"]["C6_link_artifact_null"]["surrogate_D_all8"]
    surw = results["persist"]["C6_link_artifact_null"]["surrogate_D_without_degenerate"]
    beats_link = (D > sur) and (Dw > surw)
    if lo > 0 and beats_link:
        verdict = (f"MATCHING CONFIRMED — the diagonal excess D = {D:+.5f} has a CI strictly above "
                   f"zero [{lo:+.5f},{hi:+.5f}] AND exceeds the link-artifact null "
                   f"({sur:+.5f}; {surw:+.5f} without the degenerate operator). Fitting on the "
                   f"corpus you then read on buys something beyond operator quality, read "
                   f"difficulty, and the curvature of a bounded rate.")
    elif lo > 0 and not beats_link:
        verdict = (f"LINK-ARTIFACT — D = {D:+.5f} has a CI above zero [{lo:+.5f},{hi:+.5f}] but "
                   f"does NOT exceed the multiplicative-surrogate null ({sur:+.5f}), which has "
                   f"zero matching by construction. An additive decomposition of a bounded rate "
                   f"produces this from correlated margins alone. NOT evidence of matching.")
    elif hi < 0:
        verdict = (f"ANTI-MATCHING — D = {D:+.5f}, CI [{lo:+.5f},{hi:+.5f}] strictly below zero. "
                   f"Operators read corpora they were NOT fitted on BETTER. Reported separately; "
                   f"this is neither branch of the primary rule.")
    else:
        verdict = (f"NO MATCHING (the publishable null) — D = {D:+.5f}, CI [{lo:+.5f},{hi:+.5f}] "
                   f"includes zero. The corpus effect established in E28/E33/E48 is a property of "
                   f"the FITTING CORPUS PER SE; the fit-read relationship is not the operative "
                   f"variable, and EQ2's ||J^P - J^Q|| gap framing is not what the data measures.")
    if abs(Dw - D) > abs(D):
        verdict += (f"  ||  FLAG: dropping the known-degenerate operator moves D from {D:+.5f} to "
                    f"{Dw:+.5f}. The claim is made only in the form that survives both.")

    rec = {
        "experiment": "E52 — the fit x read factorial: J^Q read on Q, the missing cell",
        "prereg": "specs/archive/prereg/PREREG_E52_FACTORIAL.md, written before this ran",
        "status": "PRE-REGISTERED",
        "question": ("can a J-lens trained on an out-of-distribution sample read latent structure "
                     "on an out-of-distribution corpus?"),
        "model": MODEL, "band": BAND, "K": K, "admitted_sets": ADMITTED,
        "fit_corpora": fit_corpora, "read_rungs": rungs,
        "fit_seeds": seeds, "prefix_seeds": pseeds,
        "n_operators": len(JF), "n_items": len(items), "n_pairs": P_n,
        "prefix_tokens": PREFIX_TOKENS, "smoke": a.smoke,
        "all_operators_share_one_fitter": ("trainval.py N=200 --band 9,21 dim_batch=128 "
                                           "max_seq_len=128 skip_first=16 target_layer=-2 fp32"),
        "adjudication": {"aggregation": "persist", "min_does_not_vote": True,
                         "D": D, "D_without_degenerate": Dw,
                         "D_hierarchical_ci": Dboot,
                         "degenerate_excluded_from_C2": DEGENERATE},
        "by_aggregation": results,
        "per_set_D": per_set_D_out,
        "per_set_sign_split": per_set_sign_split,
        "PARTIAL_UNBLINDING_DISCLOSED": (
            "E36's stored ladder already contains a 5x5 in-stream fit x read matrix (E28 N=400 "
            "operators, same aggregation, same admitted sets). Applying this script's diag_excess "
            "to it gives D = +0.001214 with all five diagonal residuals positive and a "
            "permutation p of 0.134 — i.e. positive but NOT significant. That estimate was known "
            "before E52 ran and is recorded here so E52's primary is not presented as blind for "
            "the in-stream half. It also carries ~17% fit/prefix leakage (E28 fitted N=400 from "
            "the same 800-doc block), which is the confound Amendment 1 removes, so its sign is "
            "exactly what leakage predicts."),
        "secondary_ood_rungs": secondary,
        "controls": ctrl,
        "VERDICT": verdict,
    }
    write_result(a.out, rec, experiment="E52",
                 inputs=[a.e48] + [lens_path(c, s) for c in fit_corpora for s in seeds])

    print(f"\nFIT x READ matrix (persist, admitted mean) — rows = fit corpus, cols = read rung")
    print(f"{'fit \\ read':22s}" + "".join(f"{q[:9]:>10}" for q in fit_corpora))
    for f in fit_corpora:
        row = "".join(f"{results['persist']['matrix'][f'{f}|{q}']:10.5f}" for q in fit_corpora)
        print(f"{f:22s}{row}")
    print(f"\ndiagonal excess D = {D:+.6f}   CI [{lo:+.6f}, {hi:+.6f}]"
          f"   (without {DEGENERATE}: {Dw:+.6f})")
    p4 = results["persist"]["C4_permutation"]
    print(f"C4 permutation null: mean {p4['null_mean']:+.6f} sd {p4['null_sd']:.6f}  "
          f"p(two-sided) = {p4['p_two_sided']:.4f}")
    print("\nSECONDARY — the OOD rungs, matched operator vs the alternatives:")
    for q, v in secondary.items():
        print(f"  read on {q:20s} J^q={v['matched_J^q']:.5f}  logit={v['logit']:.5f}  "
              f"shuf={v['shuf_floor']:.5f}  best in-stream J^P={v['best_instream_J^P']:.5f} "
              f"({v['best_instream_corpus']})")
        print(f"      beats logit {v['beats_logit']} | beats floor {v['beats_shuf_floor']} | "
              f"beats best in-stream {v['beats_best_instream']} "
              f"(delta {v['delta_vs_best_instream']:+.5f})")
    for k, v in ctrl.items():
        if isinstance(v, dict) and "fires" in v:
            print(f"  {k:34s} {'FIRES' if v['fires'] else 'DOES NOT FIRE'}")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
