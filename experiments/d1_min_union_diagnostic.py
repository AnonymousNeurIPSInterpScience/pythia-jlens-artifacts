#!/usr/bin/env python
"""d1_min_union_diagnostic.py — WHY does `min` prefer the layer-deranged operator?

NOT a new experiment. A DIAGNOSTIC RECOMPUTATION from artifacts already on disk, run because
CLAUDE.md §2.4 requires that a surprising result be treated as a bug report until the artifacts
that could produce it have been enumerated and tested. The surprising result is E54's C2:
`min` prefers J^shuf to J^P on 104 of 120 paired draws.

THE HYPOTHESIS UNDER TEST
  `min`-over-layers is an EXISTENTIAL over the band: pass@k iff ANY of the 13 layers puts the
  target in top-k. That is a UNION over 13 events. A union grows with the number of events and
  shrinks with their correlation. A real J^P produces per-layer readouts that AGREE (that is
  what `persist` measures); a layer derangement DESTROYS that agreement. So the derangement
  can lose at every single layer and still win the union.

  If true, three things must hold simultaneously:
    (H1) J^shuf's per-layer pass@k is LOWER than J^P's at (nearly) every layer.
    (H2) J^shuf's layers are LESS correlated in which pairs they recover.
    (H3) J^shuf's union sits closer to the independence prediction 1 - prod(1 - p_j) than
         J^P's does. i.e. J^P is redundant, J^shuf is not.
  H1 + H3 together are the artifact: every part worse, the aggregate better.

  The null: J^shuf genuinely reads better, i.e. it beats J^P at individual layers too. Then
  `min` is right and `persist` is what needs explaining.

GATE (this diagnostic is VOID if it fails)
  Recomputed `min` and `persist` for each J^P and each J^shuf must match the stored values in
  results/e48_crossover_410m.json to < 1e-6. Same lenses, same band, same K, same derangement
  seeds, same scoring function transcribed from t48_crossover.py.

Runs on CPU from results/*.pt. No GPU, no new fit, no new measurement.
"""
import argparse, json, os, statistics, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))

from anchor_evals import load_eval, readout_position, token_ids_of          # noqa: E402
from provenance import write_result                                         # noqa: E402
from jlens.hooks import ActivationRecorder                                  # noqa: E402
from jlens.hf import from_hf                                                # noqa: E402
from t2_fastfit import PYTHIA_LAYOUT_T5                                     # noqa: E402

MODEL = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))
K = [1, 2, 5, 10, 20, 50, 100]
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
EVAL_SETS = ["multihop", "multilingual", "order-ops", "poetry", "typo", "association"]
RES = os.path.join(HERE, "..", "results")


def lens_path(corpus, seed):
    if corpus.startswith("OOD_"):
        return os.path.join(RES, "e48", f"lens_{corpus}_410m_n200_s{seed}.pt")
    return os.path.join(RES, "lenses", "e28", f"e28_{corpus}_410m_n200_s{seed}.pt")


def derangement(band, seed):
    """Transcribed verbatim from t48_crossover.py so the objects are the SAME derangements."""
    g = torch.Generator().manual_seed(seed)
    while True:
        perm = [band[i] for i in torch.randperm(len(band), generator=g).tolist()]
        if all(p != l for p, l in zip(perm, band)):
            return perm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="USPTO_Backgrounds,Pile-CC,Github")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-derangements", type=int, default=3)
    ap.add_argument("--items-per-chunk", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(RES, "d1_min_union_diagnostic_410m.json"))
    a = ap.parse_args()
    corpora = a.corpora.split(",")

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    d, L = model.d_model, len(BAND)
    HALF = L // 2

    # ---------------------------------------------------------- ONE activation cache (t48 clause 5)
    t0 = time.time()
    acts_rows, pair_set, pair_targets, pair_item = [], [], [], []
    for name in EVAL_SETS:
        for it in load_eval(name):
            tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
            if not tgt:
                continue
            p = it["prompt"]
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
    ACTS = torch.stack(acts_rows)
    n_items, P_n = ACTS.shape[0], len(pair_targets)
    ITEM_PAIRS = {}
    for pi, ii in enumerate(pair_item):
        ITEM_PAIRS.setdefault(ii, []).append(pi)
    SET_IDX = {s: torch.tensor([i for i, p in enumerate(pair_set) if p == s], dtype=torch.long)
               for s in EVAL_SETS}
    ADM = torch.cat([SET_IDX[s] for s in ADMITTED])
    print(f"cached {n_items} items, {P_n} pairs, band={BAND} [{time.time()-t0:.0f}s]", flush=True)

    # ---------------------------------------------------------- scoring, transcribed from t48
    def score(T):
        H = ACTS if T is None else torch.stack(
            [ACTS[:, j, :] @ T[BAND[j]].T for j in range(L)], dim=1)
        flat = H.reshape(-1, d)
        R = torch.empty(P_n, L, dtype=torch.float32)
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

    def admitted_mean(v):
        return statistics.mean([float(v[SET_IDX[s]].mean()) for s in ADMITTED])

    def min_auc_on_sub_band(R, w):
        """`min` restricted to the CENTRED sub-band of width w. If min's preference for the
        derangement is a union-size artifact, it must vanish as w shrinks: at w=1 min is a
        single-layer readout and cannot benefit from decorrelation at all."""
        lo = (L - w) // 2
        sub = R[:, lo:lo + w]
        mn = sub.min(dim=1).values
        return admitted_mean(torch.stack([(mn <= k).float() for k in K]).mean(0))

    def summarize(R, label):
        """Everything the hypothesis needs, on the admitted-set pairs only."""
        mn = R.min(dim=1).values
        agg = {"min": admitted_mean(torch.stack([(mn <= k).float() for k in K]).mean(0)),
               "persist": admitted_mean(torch.stack(
                   [((R <= k).float().sum(1) >= HALF).float() for k in K]).mean(0))}
        Ra = R[ADM]                                            # [Padm, L]
        per_layer, union_obs, union_ind, mean_pair_corr = [], [], [], []
        for k in K:
            B = (Ra <= k).float()                              # [Padm, L] recovery indicators
            p = B.mean(0)                                      # per-layer marginal
            per_layer.append(p.tolist())
            union_obs.append(float((B.sum(1) > 0).float().mean()))
            union_ind.append(float(1.0 - torch.prod(1.0 - p)))
            # mean off-diagonal Pearson correlation between layers' recovery indicators
            Bc = B - B.mean(0, keepdim=True)
            sd = Bc.pow(2).mean(0).sqrt().clamp_min(1e-12)
            C = (Bc.T @ Bc) / Bc.shape[0] / (sd[:, None] * sd[None, :])
            off = C[~torch.eye(L, dtype=torch.bool)]
            mean_pair_corr.append(float(off.mean()))
        # AUC-style flat mean over K of the per-layer curves
        pl = torch.tensor(per_layer)                           # [K, L]
        return {
            "label": label,
            "min": agg["min"], "persist": agg["persist"],
            "per_layer_auc": pl.mean(0).tolist(),
            "best_single_layer_auc": float(pl.mean(0).max()),
            "mean_single_layer_auc": float(pl.mean()),
            "union_observed_by_k": union_obs,
            "union_if_layers_independent_by_k": union_ind,
            "redundancy_ratio_by_k": [o / i if i else float("nan")
                                      for o, i in zip(union_obs, union_ind)],
            "mean_pairwise_layer_corr_by_k": mean_pair_corr,
            "mean_layers_recovering_by_k": [float((Ra <= k).float().sum(1).mean()) for k in K],
            "min_auc_by_band_width": {str(w): min_auc_on_sub_band(R, w)
                                      for w in range(1, L + 1)},
        }

    arms, gate = {}, {}
    t = time.time()
    arms["logit_I"] = summarize(score(None), "logit_I")
    print(f"  logit_I  min={arms['logit_I']['min']:.6f} persist={arms['logit_I']['persist']:.6f}"
          f"  [{time.time()-t:.0f}s]", flush=True)

    stored = json.load(open(os.path.join(RES, "e48_crossover_410m.json")))["arms_admitted_mean"]
    gate["logit_I"] = {ag: {"recomputed": arms["logit_I"][ag], "stored": stored["logit_I"][ag],
                            "abs_diff": abs(arms["logit_I"][ag] - stored["logit_I"][ag])}
                       for ag in ("min", "persist")}

    for c in corpora:
        JP = torch.load(lens_path(c, a.seed), map_location="cpu", weights_only=True)["J"]
        JP = {int(l): JP[l].float() if torch.is_tensor(JP[l]) else torch.tensor(JP[l]).float()
              for l in JP}
        name = f"J|{c}|s{a.seed}"
        t = time.time()
        arms[name] = summarize(score(JP), name)
        print(f"  {name:28s} min={arms[name]['min']:.6f} persist={arms[name]['persist']:.6f}"
              f"  [{time.time()-t:.0f}s]", flush=True)
        gate[name] = {ag: {"recomputed": arms[name][ag], "stored": stored[name][ag],
                           "abs_diff": abs(arms[name][ag] - stored[name][ag])}
                      for ag in ("min", "persist")}
        for dr in range(a.n_derangements):
            perm = derangement(BAND, 7000 + 97 * dr)
            sname = f"shuf|{c}|s{a.seed}|d{dr}"
            t = time.time()
            arms[sname] = summarize(score({l: JP[q] for l, q in zip(BAND, perm)}), sname)
            print(f"  {sname:28s} min={arms[sname]['min']:.6f} "
                  f"persist={arms[sname]['persist']:.6f}  [{time.time()-t:.0f}s]", flush=True)
            gate[sname] = {ag: {"recomputed": arms[sname][ag], "stored": stored[sname][ag],
                                "abs_diff": abs(arms[sname][ag] - stored[sname][ag])}
                           for ag in ("min", "persist")}

    gate_ok = all(v[ag]["abs_diff"] < 1e-6 for v in gate.values() for ag in ("min", "persist"))

    # ---------------------------------------------------------- adjudicate H1/H2/H3
    verdict = {}
    for c in corpora:
        jp = arms[f"J|{c}|s{a.seed}"]
        sh = [arms[f"shuf|{c}|s{a.seed}|d{dr}"] for dr in range(a.n_derangements)]
        jp_pl = torch.tensor(jp["per_layer_auc"])
        sh_pl = torch.stack([torch.tensor(s["per_layer_auc"]) for s in sh]).mean(0)
        verdict[c] = {
            "H1_shuf_worse_at_every_layer": {
                "n_layers_where_shuf_lower": int((sh_pl < jp_pl).sum()),
                "n_layers": L,
                "jp_mean_single_layer_auc": float(jp_pl.mean()),
                "shuf_mean_single_layer_auc": float(sh_pl.mean()),
                "jp_best_single_layer_auc": float(jp_pl.max()),
                "shuf_best_single_layer_auc": float(sh_pl.max()),
                "fires": bool((sh_pl < jp_pl).sum() >= L - 1)},
            "H2_shuf_layers_less_correlated": {
                "jp_mean_pairwise_corr": statistics.mean(jp["mean_pairwise_layer_corr_by_k"]),
                "shuf_mean_pairwise_corr": statistics.mean(
                    [statistics.mean(s["mean_pairwise_layer_corr_by_k"]) for s in sh]),
                "fires": bool(statistics.mean(
                    [statistics.mean(s["mean_pairwise_layer_corr_by_k"]) for s in sh])
                    < statistics.mean(jp["mean_pairwise_layer_corr_by_k"]))},
            "H3_shuf_union_closer_to_independence": {
                "jp_redundancy_ratio": statistics.mean(jp["redundancy_ratio_by_k"]),
                "shuf_redundancy_ratio": statistics.mean(
                    [statistics.mean(s["redundancy_ratio_by_k"]) for s in sh]),
                "fires": bool(statistics.mean(
                    [statistics.mean(s["redundancy_ratio_by_k"]) for s in sh])
                    > statistics.mean(jp["redundancy_ratio_by_k"]))},
            "min_gap_jp_minus_shuf": jp["min"] - statistics.mean([s["min"] for s in sh]),
            "persist_gap_jp_minus_shuf": jp["persist"] - statistics.mean([s["persist"] for s in sh]),
            "H4_artifact_scales_with_band_width": {
                "min_gap_jp_minus_shuf_by_band_width": {
                    str(w): jp["min_auc_by_band_width"][str(w)]
                            - statistics.mean([s["min_auc_by_band_width"][str(w)] for s in sh])
                    for w in range(1, L + 1)},
                "narrowest_width_where_jp_still_wins": next(
                    (str(w) for w in range(L, 0, -1)
                     if jp["min_auc_by_band_width"][str(w)]
                     > statistics.mean([s["min_auc_by_band_width"][str(w)] for s in sh])), None),
            },
        }

    fires = {h: sum(1 for c in corpora if verdict[c][h]["fires"])
             for h in ("H1_shuf_worse_at_every_layer", "H2_shuf_layers_less_correlated",
                       "H3_shuf_union_closer_to_independence")}
    if not gate_ok:
        V = "VOID — the recomputation does not reproduce the stored e48 arms; nothing here counts."
    elif all(v == len(corpora) for v in fires.values()):
        V = ("ARTIFACT CONFIRMED — the derangement loses at the individual layers and wins the "
             "union, because scrambling decorrelates the layers. `min` rewards disagreement "
             "across layers, which is the opposite of what a transport readout should reward.")
    else:
        V = f"MIXED — hypotheses firing: {fires}. Do not restate the mechanism as established."

    out = {
        "experiment": "D1 — diagnostic: is `min`'s preference for J^shuf a union-over-layers artifact?",
        "status": "DIAGNOSTIC RECOMPUTATION, not a pre-registered experiment. Recomputes from "
                  "results/*.pt; measures nothing new.",
        "hypothesis": "min is an existential union over the band; a derangement decorrelates the "
                      "layers; a union of decorrelated weaker events can exceed a union of "
                      "correlated stronger ones.",
        "model": MODEL, "band": BAND, "K": K, "admitted_sets": ADMITTED,
        "half_threshold_for_persist": HALF,
        "n_items": n_items, "n_pairs": P_n,
        "corpora": corpora, "seed": a.seed, "n_derangements": a.n_derangements,
        "GATE_reproduces_stored_e48_arms": {"tolerance": 1e-6, "fires": gate_ok, "per_arm": gate},
        "arms": arms,
        "adjudication": verdict,
        "hypotheses_firing_out_of_n_corpora": fires,
        "VERDICT": V,
    }
    write_result(a.out, out, experiment=out["experiment"], script=__file__,
                 inputs=[os.path.join(RES, "e48_crossover_410m.json")]
                        + [lens_path(c, a.seed) for c in corpora])
    print("\n" + V)
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
