#!/usr/bin/env python3
"""da1_derangement_adjudication.py — DA1: what does the layer-derangement control identify?

PRE-REGISTRATION: docs/experiments/preregs/DA1_derangement_adjudication.md, written before this
ran and before any DA1 number existed.

THE PROBLEM THIS ADJUDICATES
  R9 reports that under the published statistic `min`, 0 of 8 operators clear all fifteen of their
  own random layer-derangement draws (median z = -0.71). That was read as evidence that `min` does
  not certify layer correspondence. An external audit objected that the random derangement removes
  TWO things at once:

    (1) absolute layer correspondence -- band layer l no longer receives its own J_l; and
    (2) cross-layer dependence -- a derangement's per-layer readouts agree with each other less
        than the real operator's do.

  `min` is an EXISTENTIAL UNION over the band, and a union grows as its events decorrelate. So a
  null that decorrelates the layers moves `min` through channel (2) whether or not channel (1)
  matters. `results/d1_min_union_diagnostic_410m.json` already reported that mechanism. What has
  never been run is a null that separates the channels.

WHAT DA1 ADDS
  A CYCLIC SHIFT `J_l -> J_{l+k mod |B|}`. After any non-zero shift no band layer holds its own
  Jacobian, so correspondence is gone; but the ORDER of the sequence survives, so adjacent band
  layers still receive adjacent Jacobians except at one wrap point. Twelve shifts are scored, not
  one, so no favourable k can be selected. Cyclic shifting is NOT assumed to preserve the
  dependence structure -- section D measures how much each null moves it.

  Plus: the same 8 x 15 random-derangement grid R9 used, re-scored under two summaries that are
  not unions over the band (`best1L`, `mean`), so the question "is the derangement's advantage
  specific to an existential aggregation?" can be answered on R9's own objects.

THE ONE DESIGN POINT THAT MAKES THIS CHEAP AND EXACT
  Every arm here is a PERMUTATION of which Jacobian meets which band activation slot. There are
  only 13 x 13 = 169 such (slot, Jacobian) combinations per operator. Scoring all 169 once costs
  13 arm-equivalents and makes EVERY permutation -- identity, twelve cyclic shifts, five
  derangements -- a lookup into one tensor. Two arms therefore cannot differ by anything except
  the permutation: there is no separate forward pass, no separate cache and no separate scorer for
  them to differ in.

EVERYTHING ELSE IS TRANSCRIBED, NOT REWRITTEN
  The model, the corrected `.rstrip()` readout, the band, K, the admitted sets, the activation
  cache and the scoring function are taken from `experiments/t48_crossover.py` run with --rstrip,
  which is the run that produced the corrected numbers R9 adjudicates. The gate in section G
  requires this file's `min` and `persist` to reproduce that run's stored per-arm values.

  Runs on CPU from results/*.pt. No GPU, no refit, no new corpus, no new evaluation definition.

    python experiments/da1_derangement_adjudication.py --device cpu
    python experiments/da1_derangement_adjudication.py --smoke
"""
from __future__ import annotations

import argparse
import itertools
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
CORPORA = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github",
           "OOD_News_2024", "OOD_arXiv_2023", "OOD_CommonPile"]
SEEDS = [0, 1, 2]
N_DERANGEMENTS = 5                 # t48_crossover.py's default; 3 seeds x 5 perms = R9's "fifteen"
SHIFTS = list(range(1, 13))        # every non-zero cyclic shift of a 13-layer band
GATE_FILE = "results/e48_crossover_410m_rstrip.json"
GATE_TOL = 1e-6


def lens_path(corpus: str, seed: int) -> str:
    """Transcribed from t48_crossover.py:138-142 so DA1 scores THE SAME operator files."""
    if corpus.startswith("OOD_"):
        return os.path.join(HERE, "..", "results", "e48",
                            f"lens_{corpus}_410m_n200_s{seed}.pt")
    return os.path.join(HERE, "..", "results", "lenses", "e28",
                        f"e28_{corpus}_410m_n200_s{seed}.pt")


def derangement(band, seed):
    """Transcribed VERBATIM from t48_crossover.py so the permutations are the SAME objects."""
    g = torch.Generator().manual_seed(seed)
    while True:
        perm = [band[i] for i in torch.randperm(len(band), generator=g).tolist()]
        if all(p != l for p, l in zip(perm, band)):
            return perm


def cyclic(band, k):
    """J_l -> J_{l+k mod |B|}. Returns the Jacobian layer each band slot receives."""
    n = len(band)
    return [band[(i + k) % n] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--items-per-chunk", type=int, default=24)
    ap.add_argument("--corpora", default=",".join(CORPORA))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--smoke", action="store_true",
                    help="8 items per set, 2 corpora, 1 seed. Proves the path, measures nothing.")
    ap.add_argument("--skip-band-constant", action="store_true",
                    help="omit the exploratory A4 diagnostic (section 2.5 of the prereg)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results",
                                                  "da1_derangement_adjudication_410m.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of

    corpora = (CORPORA[:2] if a.smoke else a.corpora.split(","))
    seeds = ([0] if a.smoke else [int(s) for s in a.seeds.split(",")])

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    d, L = model.d_model, len(BAND)
    HALF = L // 2

    # ------------------------------------------------------------------ activation cache
    # Transcribed from t48_crossover.py:224-246 with the --rstrip branch taken. The `.rstrip()`
    # IS the corrected readout: the released prompts end in a trailing space that BPE absorbs
    # into the target, so the unstripped final token occurs nowhere in the tokenised sequence.
    t0 = time.time()
    acts_rows, pair_set, pair_targets, pair_item, per_set_n = [], [], [], [], {}
    with torch.no_grad():
        for name in EVAL_SETS:
            for it in load_eval(name):
                tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
                if not tgt:
                    continue
                if a.smoke and per_set_n.get(name, 0) >= 8:
                    continue
                per_set_n[name] = per_set_n.get(name, 0) + 1
                p = it["prompt"].rstrip()                      # CORRECTED READOUT
                ids = model.encode(p, max_length=256)
                pos = readout_position(tok, name, p)
                with ActivationRecorder(model.layers, at=BAND) as rec:
                    model.forward(ids)
                    A = torch.stack([rec.activations[l][0][pos].detach().float() for l in BAND])
                idx_item = len(acts_rows)
                acts_rows.append(A)
                for t in tgt:
                    pair_set.append(name)
                    pair_targets.append(t)
                    pair_item.append(idx_item)
    ACTS = torch.stack(acts_rows)                              # [I, L, d]
    n_items, P_n = ACTS.shape[0], len(pair_targets)
    ITEM_PAIRS = {}
    for pi, ii in enumerate(pair_item):
        ITEM_PAIRS.setdefault(ii, []).append(pi)
    SET_IDX = {s: torch.tensor([i for i, p in enumerate(pair_set) if p == s], dtype=torch.long)
               for s in EVAL_SETS}
    ADMITTED_IDX = torch.cat([SET_IDX[s] for s in ADMITTED]).sort().values
    print(f"cached {n_items} items, {P_n} pairs, band={BAND}, corrected readout  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ------------------------------------------------------------------ the ONE scoring function
    def rank_matrix(T):
        """T: {band layer -> [d,d]} or None for the identity. Returns ranks [P, L].

        Transcribed from t48_crossover.py:275-299. 1-indexed strict-greater count over the full
        vocabulary, min over the token ids of a multi-token intermediate.
        """
        if T is None:
            H = ACTS
        else:
            H = torch.stack([ACTS[:, j, :] @ T[BAND[j]].T for j in range(L)], dim=1)
        flat = H.reshape(-1, d)
        R = torch.empty(P_n, L, dtype=torch.float32)
        with torch.no_grad():
            for i0 in range(0, n_items, a.items_per_chunk):
                i1 = min(i0 + a.items_per_chunk, n_items)
                lg = model.unembed(flat[i0 * L:i1 * L]).float()
                for ii in range(i0, i1):
                    block = lg[(ii - i0) * L:(ii - i0 + 1) * L]
                    for pi in ITEM_PAIRS[ii]:
                        cand = torch.stack([(block > block[:, i:i + 1]).sum(1) + 1
                                            for i in pair_targets[pi]])
                        R[pi] = cand.min(0).values.float()
        return R

    def set_mean(v, s):
        return float(v[SET_IDX[s]].mean())

    def admitted_mean(v):
        return statistics.mean([set_mean(v, s) for s in ADMITTED])

    # ------------------------------------------------------------------ the four summaries
    # `min` and `persist` transcribed from t48_crossover.py:301-307.
    # `best1L` and `mean` take their SEMANTICS from t17_reaggregate.py:aggregate() (per-layer
    # quality, then max / mean over layers) and their k-summary and set-averaging from
    # t48_crossover, so `min` reproduces the stored corrected value exactly. Prereg section 2.3
    # records that DA1 computes all four OVER THE BAND, where t17 computed three of them over all
    # layers -- a change of scope, declared before running, because the claim under adjudication
    # is band-scoped.
    def summaries(R):
        mn = R.min(dim=1).values
        v_min = torch.stack([(mn <= k).float() for k in K]).mean(0)
        v_per = torch.stack([torch.stack([(R[:, j] <= k).float() for k in K]).mean(0)
                             for j in range(L)])                       # [L, P]
        v_pers = torch.stack([((R <= k).float().sum(1) >= HALF).float() for k in K]).mean(0)
        per_layer = [admitted_mean(v_per[j]) for j in range(L)]
        return {"min": admitted_mean(v_min),
                "persist": admitted_mean(v_pers),
                "best1L": max(per_layer),
                "best1L_layer": BAND[int(max(range(L), key=lambda j: per_layer[j]))],
                "mean": statistics.mean(per_layer),
                "per_layer": [round(x, 6) for x in per_layer],
                "_v_min": v_min, "_v_pers": v_pers}

    # ------------------------------------------------------------------ dependence (prereg 2.4)
    def dependence(R):
        """Cross-layer dependence OF THE RECOVERY EVENTS the aggregation consumes.

        Not Jacobian similarity: the inferential question is about the correlation of
        1[rank_l <= k] across l, because that is what a union over the band consumes.
        Computed over admitted pairs only, matching the scored population.
        """
        Ra = R[ADMITTED_IDX]
        out = {"per_k": {}, "n_degenerate_pairs": 0}
        rhos, gaps = [], []
        for k in K:
            Hk = (Ra <= k).float()                                     # [Pa, L]
            p_l = Hk.mean(0)                                           # per-layer hit rate
            # mean off-diagonal Pearson correlation over the 78 band layer pairs
            sd = Hk.std(0, unbiased=False)
            good = [j for j in range(L) if float(sd[j]) > 0]
            deg = L * (L - 1) // 2 - len(good) * (len(good) - 1) // 2
            out["n_degenerate_pairs"] += deg
            if len(good) >= 2:
                Hc = Hk[:, good] - Hk[:, good].mean(0)
                C = (Hc.T @ Hc) / Hc.shape[0]
                s = C.diag().sqrt()
                corr = C / (s[:, None] * s[None, :])
                iu = torch.triu_indices(len(good), len(good), offset=1)
                rho = float(corr[iu[0], iu[1]].mean())
            else:
                rho = float("nan")
            union_obs = float(Hk.max(1).values.mean())
            union_ind = float(1.0 - torch.prod(1.0 - p_l))
            out["per_k"][str(k)] = {"rho_bar": rho, "union_observed": union_obs,
                                    "union_independent": union_ind,
                                    "union_gap": union_obs - union_ind,
                                    "mean_per_layer_hit_rate": float(p_l.mean())}
            if not math.isnan(rho):
                rhos.append(rho)
            gaps.append(union_obs - union_ind)
        out["rho_bar_mean_over_k"] = statistics.mean(rhos) if rhos else float("nan")
        out["union_gap_mean_over_k"] = statistics.mean(gaps)
        return out

    # ------------------------------------------------------------------ permutations (prereg 2.2)
    SHUF_PERMS = {f"shuf{dr}": derangement(BAND, 7000 + 97 * dr) for dr in range(N_DERANGEMENTS)}
    CYC_PERMS = {f"cyc{k}": cyclic(BAND, k) for k in SHIFTS}
    ARM_PERMS = {"real": list(BAND), **CYC_PERMS, **SHUF_PERMS}

    # C2 / C4: structural controls on the permutations themselves, each with its failing value.
    controls = {}
    controls["C2_shifts_have_no_fixed_point"] = {
        "requirement": "every non-zero cyclic shift moves every band layer off its own Jacobian",
        "fails_if": "any k in 1..12 leaves some slot receiving its own layer",
        "per_shift_fixed_points": {f"cyc{k}": sum(1 for i, q in enumerate(cyclic(BAND, k))
                                                  if q == BAND[i]) for k in SHIFTS},
    }
    controls["C2_shifts_have_no_fixed_point"]["fires"] = all(
        v == 0 for v in controls["C2_shifts_have_no_fixed_point"]["per_shift_fixed_points"].values())
    controls["C4_derangements_match_e48"] = {
        "requirement": "the five shuf permutations equal t48_crossover.derangement(BAND, 7000+97d)",
        "fails_if": "any element differs, which would mean DA1 is not scoring E48's objects",
        "perms": {n: p for n, p in SHUF_PERMS.items()},
        "fires": all(SHUF_PERMS[f"shuf{dr}"] == derangement(BAND, 7000 + 97 * dr)
                     for dr in range(N_DERANGEMENTS)),
    }

    # ------------------------------------------------------------------ logit arm + gate C3
    print("scoring logit arm...", flush=True)
    R_logit = rank_matrix(None)
    S_logit = summaries(R_logit)
    gate = {"file": GATE_FILE, "tolerance": GATE_TOL, "checked": [], "max_abs_diff": 0.0}
    stored = None
    gp = os.path.join(HERE, "..", GATE_FILE)
    if os.path.exists(gp) and not a.smoke:
        stored = json.load(open(gp))["arms_admitted_mean"]
        for ag in ("min", "persist"):
            diff = abs(S_logit[ag] - stored["logit_I"][ag])
            gate["checked"].append({"arm": "logit_I", "agg": ag, "da1": S_logit[ag],
                                    "stored": stored["logit_I"][ag], "abs_diff": diff})
            gate["max_abs_diff"] = max(gate["max_abs_diff"], diff)
        print(f"  C3 logit vs stored: max|diff| = {gate['max_abs_diff']:.3e}", flush=True)

    # ------------------------------------------------------------------ the sweep
    OPS, DEP, PERLAYER, BANDCONST = {}, {}, {}, {}
    t0 = time.time()
    for c in corpora:
        for s in seeds:
            path = lens_path(c, s)
            if not os.path.exists(path):
                raise SystemExit(f"ABORT: missing operator {os.path.relpath(path, os.path.join(HERE,'..'))}")
            blob = torch.load(path, map_location="cpu", weights_only=True)["J"]
            missing = [l for l in BAND if l not in blob]
            if missing:
                raise SystemExit(f"{path} lacks band layers {missing}")
            J = {l: blob[l].float() for l in BAND}

            # ---- the 13 x 13 (slot, Jacobian) rank tensor: 13 arm-equivalents, all perms free
            Rall = torch.empty(P_n, L, L, dtype=torch.float32)          # [P, slot j, jacobian q]
            for qi, q in enumerate(BAND):
                Rall[:, :, qi] = rank_matrix({BAND[j]: J[q] for j in range(L)})
            qidx = {q: i for i, q in enumerate(BAND)}

            for arm, perm in ARM_PERMS.items():
                R = torch.stack([Rall[:, j, qidx[perm[j]]] for j in range(L)], dim=1)
                S = summaries(R)
                PERLAYER[f"{c}|s{s}|{arm}"] = S.pop("per_layer")
                S.pop("_v_min"); S.pop("_v_pers")
                OPS[f"{c}|s{s}|{arm}"] = S
                DEP[f"{c}|s{s}|{arm}"] = dependence(R)

            if not a.skip_band_constant:
                Jbar = torch.stack([J[l] for l in BAND]).mean(0)
                Rbc = rank_matrix({l: Jbar for l in BAND})
                Sbc = summaries(Rbc)
                PERLAYER[f"{c}|s{s}|bandconst"] = Sbc.pop("per_layer")
                Sbc.pop("_v_min"); Sbc.pop("_v_pers")
                BANDCONST[f"{c}|s{s}"] = Sbc
                DEP[f"{c}|s{s}|bandconst"] = dependence(Rbc)

            print(f"  {c:20s} s{s}  real min={OPS[f'{c}|s{s}|real']['min']:.6f} "
                  f"persist={OPS[f'{c}|s{s}|real']['persist']:.6f}  [{time.time()-t0:.0f}s]",
                  flush=True)

    # ------------------------------------------------------------------ GATE (prereg section 3)
    if stored is not None:
        for c in corpora:
            for s in seeds:
                pairs = [(f"J|{c}|s{s}", f"{c}|s{s}|real")]
                pairs += [(f"shuf|{c}|s{s}|d{dr}", f"{c}|s{s}|shuf{dr}")
                          for dr in range(N_DERANGEMENTS)]
                for stored_arm, mine in pairs:
                    if stored_arm not in stored:
                        continue
                    for ag in ("min", "persist"):
                        diff = abs(OPS[mine][ag] - stored[stored_arm][ag])
                        gate["checked"].append({"arm": stored_arm, "agg": ag,
                                                "da1": OPS[mine][ag],
                                                "stored": stored[stored_arm][ag],
                                                "abs_diff": diff})
                        gate["max_abs_diff"] = max(gate["max_abs_diff"], diff)
    gate["n_checked"] = len(gate["checked"])
    gate["fires"] = bool(gate["checked"]) and gate["max_abs_diff"] < GATE_TOL
    gate["smallest_possible_real_difference"] = 1.0 / (max(len(SET_IDX[s]) for s in EVAL_SETS) * len(K))
    gate["tolerance_justification"] = (
        "one pair flipping at one k moves a set mean by at least 1/(largest set * |K|); the "
        "tolerance sits far below that and above float32 accumulation noise. Transcribed from "
        "t48_crossover.py's C0, not tuned here.")
    controls["C3_gate_reproduces_e48_rstrip"] = {k: gate[k] for k in
                                                 ("file", "tolerance", "n_checked",
                                                  "max_abs_diff", "fires")}
    controls["C1_identity_permutation_is_the_real_arm"] = {
        "requirement": "the identity permutation of the 13x13 tensor IS the real operator arm",
        "note": "structural: `real` is ARM_PERMS['real'] = BAND, so its lookup is Rall[:,j,j]. "
                "C3 tests the resulting numbers against the independently-scored stored arms.",
        "fires": ARM_PERMS["real"] == list(BAND)}

    print(f"\nGATE vs {GATE_FILE}: {gate['n_checked']} arm/aggregation comparisons, "
          f"max|diff| = {gate['max_abs_diff']:.3e}  ->  "
          f"{'FIRES' if gate['fires'] else 'DOES NOT FIRE'}", flush=True)

    # ------------------------------------------------------------------ paired summaries
    def corpus_mean(c, arm, ag):
        return statistics.mean(OPS[f"{c}|s{s}|{arm}"][ag] for s in seeds)

    AGGS = ["min", "persist", "best1L", "mean"]
    by_corpus = {}
    for c in corpora:
        row = {"real": {ag: corpus_mean(c, "real", ag) for ag in AGGS}}
        row["cyclic"] = {
            f"cyc{k}": {ag: corpus_mean(c, f"cyc{k}", ag) for ag in AGGS} for k in SHIFTS}
        row["shuf"] = {
            f"shuf{dr}": {ag: corpus_mean(c, f"shuf{dr}", ag) for ag in AGGS}
            for dr in range(N_DERANGEMENTS)}
        row["cyclic_mean"] = {ag: statistics.mean(row["cyclic"][f"cyc{k}"][ag] for k in SHIFTS)
                              for ag in AGGS}
        row["shuf_mean"] = {ag: statistics.mean(row["shuf"][f"shuf{dr}"][ag]
                                                for dr in range(N_DERANGEMENTS)) for ag in AGGS}
        row["real_minus_cyclic_mean"] = {ag: row["real"][ag] - row["cyclic_mean"][ag]
                                         for ag in AGGS}
        row["real_minus_shuf_mean"] = {ag: row["real"][ag] - row["shuf_mean"][ag] for ag in AGGS}
        row["n_cyclic_shifts_real_beats"] = {
            ag: sum(1 for k in SHIFTS if row["real"][ag] > row["cyclic"][f"cyc{k}"][ag])
            for ag in AGGS}
        row["n_shuf_draws_real_beats"] = {
            ag: sum(1 for s in seeds for dr in range(N_DERANGEMENTS)
                    if OPS[f"{c}|s{s}|real"][ag] > OPS[f"{c}|s{s}|shuf{dr}"][ag])
            for ag in AGGS}
        if BANDCONST:
            row["bandconst"] = {ag: statistics.mean(BANDCONST[f"{c}|s{s}"][ag] for s in seeds)
                                for ag in AGGS}
            row["real_minus_bandconst"] = {ag: row["real"][ag] - row["bandconst"][ag]
                                           for ag in AGGS}
        by_corpus[c] = row

    across = {}
    for ag in AGGS:
        across[ag] = {
            "n_corpora_real_beats_cyclic_mean": sum(
                1 for c in corpora if by_corpus[c]["real_minus_cyclic_mean"][ag] > 0),
            "n_corpora_real_beats_shuf_mean": sum(
                1 for c in corpora if by_corpus[c]["real_minus_shuf_mean"][ag] > 0),
            "median_real_minus_cyclic_mean": statistics.median(
                by_corpus[c]["real_minus_cyclic_mean"][ag] for c in corpora),
            "median_real_minus_shuf_mean": statistics.median(
                by_corpus[c]["real_minus_shuf_mean"][ag] for c in corpora),
            "n_corpora": len(corpora),
            "total_cyclic_cells_real_beats": sum(
                by_corpus[c]["n_cyclic_shifts_real_beats"][ag] for c in corpora),
            "total_cyclic_cells": len(corpora) * len(SHIFTS),
            "total_shuf_draws_real_beats": sum(
                by_corpus[c]["n_shuf_draws_real_beats"][ag] for c in corpora),
            "total_shuf_draws": len(corpora) * len(seeds) * N_DERANGEMENTS,
        }
        if BANDCONST:
            across[ag]["n_corpora_real_beats_bandconst"] = sum(
                1 for c in corpora if by_corpus[c]["real_minus_bandconst"][ag] > 0)
            across[ag]["median_real_minus_bandconst"] = statistics.median(
                by_corpus[c]["real_minus_bandconst"][ag] for c in corpora)

    # per-shift aggregate across corpora, so no single k can be selected
    per_shift = {}
    for k in SHIFTS:
        per_shift[f"cyc{k}"] = {
            ag: {"median_real_minus_shift": statistics.median(
                     by_corpus[c]["real"][ag] - by_corpus[c]["cyclic"][f"cyc{k}"][ag]
                     for c in corpora),
                 "n_corpora_real_beats": sum(
                     1 for c in corpora
                     if by_corpus[c]["real"][ag] > by_corpus[c]["cyclic"][f"cyc{k}"][ag])}
            for ag in AGGS}

    # dependence roll-up (prereg 2.4), the quantity the confound argument is about
    def dep_mean(arms, field):
        vals = [DEP[f"{c}|s{s}|{arm}"][field] for c in corpora for s in seeds for arm in arms
                if not math.isnan(DEP[f"{c}|s{s}|{arm}"][field])]
        return statistics.mean(vals) if vals else float("nan")

    dep_summary = {
        "what_rho_bar_measures": (
            "mean off-diagonal Pearson correlation, over the 78 band layer pairs, of the binary "
            "recovery events 1[rank_l <= k] across admitted pairs. This is the dependence an "
            "existential union over the band consumes. It is NOT Jacobian cosine similarity, "
            "which is a different quantity and cannot answer the question."),
        "what_union_gap_measures": (
            "observed union minus the independent-layer prediction 1 - prod(1 - p_l). A sequence "
            "whose layers are redundant sits below its prediction; a decorrelated one sits near "
            "it. Same construction as D1's H3 clause."),
        "rho_bar": {"real": dep_mean(["real"], "rho_bar_mean_over_k"),
                    "cyclic": dep_mean([f"cyc{k}" for k in SHIFTS], "rho_bar_mean_over_k"),
                    "shuf": dep_mean([f"shuf{d}" for d in range(N_DERANGEMENTS)],
                                     "rho_bar_mean_over_k")},
        "union_gap": {"real": dep_mean(["real"], "union_gap_mean_over_k"),
                      "cyclic": dep_mean([f"cyc{k}" for k in SHIFTS], "union_gap_mean_over_k"),
                      "shuf": dep_mean([f"shuf{d}" for d in range(N_DERANGEMENTS)],
                                       "union_gap_mean_over_k")},
        "per_shift_rho_bar": {f"cyc{k}": dep_mean([f"cyc{k}"], "rho_bar_mean_over_k")
                              for k in SHIFTS},
    }
    if BANDCONST:
        dep_summary["rho_bar"]["bandconst"] = dep_mean(["bandconst"], "rho_bar_mean_over_k")
        dep_summary["union_gap"]["bandconst"] = dep_mean(["bandconst"], "union_gap_mean_over_k")

    rec = {
        "experiment": "DA1 — adjudicating what the layer-derangement control identifies",
        "prereg": "docs/experiments/preregs/DA1_derangement_adjudication.md",
        "status": "PRE-REGISTERED — plan written before any DA1 number existed",
        "adjudicates": ["results/r9_permutation_calibrated_min.json",
                        "results/e48_crossover_410m_rstrip.json",
                        "results/d1_min_union_diagnostic_410m.json"],
        "readout": "CORRECTED — item['prompt'].rstrip(), per anchor_evals.py:32-34",
        "no_refit": True,
        "model": MODEL, "band": BAND, "K": K, "admitted_sets": ADMITTED,
        "corpora": corpora, "seeds": seeds,
        "n_items": n_items, "n_pairs": P_n,
        "shifts": SHIFTS, "n_derangements": N_DERANGEMENTS,
        "derangement_perms": SHUF_PERMS,
        "cyclic_perms": CYC_PERMS,
        "note_on_the_fifteen_draws": (
            "R9's 'fifteen derangements' are 3 seed blocks x 5 permutations, so there are five "
            "DISTINCT permutations, not fifteen. That is a property of the E48 design being "
            "adjudicated; DA1 reproduces it rather than changing it."),
        "aggregation_definitions": {
            "min": "admitted mean of mean_k 1[min over band layers of rank <= k] — t48_crossover",
            "persist": "admitted mean of mean_k 1[#{band layers with rank<=k} >= floor(|B|/2)]",
            "best1L": "max over band layers of the admitted mean of mean_k 1[rank_l <= k]",
            "mean": "mean over band layers of the admitted mean of mean_k 1[rank_l <= k]",
            "semantics_from": "experiments/t17_reaggregate.py:aggregate()",
            "k_summary_and_set_averaging_from": "experiments/t48_crossover.py",
            "declared_deviation": (
                "t17 computes min/best1L/mean over ALL layers and persist over the band, under "
                "KS=(1,2,5,10,25,100) with a log-k trapezoid. DA1 computes all four OVER THE BAND "
                "under K=(1,2,5,10,20,50,100) with a flat mean, because the claim under "
                "adjudication is band-scoped and stated under that k-summary. Declared in the "
                "pre-registration before running."),
            "best1L_caveat": (
                "a maximum over 13 layers is itself a selection; best1L is reported as a "
                "diagnostic of where per-layer quality sits, not as a corrected recovery metric."),
        },
        "logit_arm": {ag: S_logit[ag] for ag in AGGS},
        "gate": gate,
        "controls": controls,
        "by_corpus": by_corpus,
        "across_corpora": across,
        "per_shift": per_shift,
        "dependence": dep_summary,
        "dependence_per_arm": DEP,
        "per_layer_admitted_mean": PERLAYER,
        "band_constant_exploratory": {
            "status": "EXPLORATORY — NOT a matched control. J_bar is a different object with a "
                      "different norm and spectrum, not a permutation of the existing ones, so it "
                      "changes more than the cyclic and deranged arms do. Reported only to ask "
                      "whether layer-specific variation carries anything beyond one average "
                      "transport over the band. Carries no verdict.",
            "by_operator": BANDCONST} if BANDCONST else {"status": "skipped"},
        "arms_per_operator": OPS,
    }
    write_result(a.out, rec, experiment="DA1", script=__file__,
                 inputs=[lens_path(c, s) for c in corpora for s in seeds] +
                        ([os.path.join(HERE, "..", GATE_FILE)] if stored is not None else []))
    if not gate["fires"] and not a.smoke:
        print("\nGATE DID NOT FIRE. DA1's numbers are reported but are NOT adopted over the "
              "stored ones; the disagreement is the result. See CLAUDE.md standing rule.",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
