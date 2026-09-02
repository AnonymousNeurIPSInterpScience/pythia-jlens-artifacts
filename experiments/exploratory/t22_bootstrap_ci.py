#!/usr/bin/env python3
"""t22_bootstrap_ci.py — E26. Confidence intervals for every read headline.

WHY. Not one number in this programme carries a confidence interval, and every headline is
stated in n_win: a sign statistic over 6 eval sets that share a model, a lens, a tokenizer and a
fit corpus. external review Prompt 8 sec5A: "If you have 10,000 prompts from two domains, you do not have
10,000 independent shift replicates. You have approximately two domain-level replicates."

This is not a new experiment. It re-scores existing per-(item,intermediate) ranks -- which one
forward pass already produces -- and attaches uncertainty at TWO replication units.

================================ PRE-REGISTRATION ================================
Fixed before the first number.

PRIMARY OUTCOME
  For each (model, eval set): the paired-item 95% CI on AUC(J^P) - AUC(I), and the same for
  AUC(J^P) - AUC(J^shuf). Plus the HIERARCHICAL CI across sets, which treats the eval set as
  the replication unit.

DECISION RULE -- applied per set, and this is a RE-GRADING of claims already made:
  CI excludes 0        -> that set's contribution to n_win SURVIVES and may be cited.
  CI includes 0        -> that set's contribution is NOT SUPPORTED and n_win must be reported
                          with it removed. A set that flips sign inside its own CI was never
                          evidence.
  hierarchical CI includes 0 while several per-set CIs exclude it
                       -> the effect does not GENERALISE across sets; report per-set only and
                          retire the pooled claim.

CONTROLS
  C1 SELF-TEST. A scaled identity transport must give a paired CI of exactly [0, 0] against the
     logit lens, since the two arms are bit-identical. Any width means the pairing is broken.
  C2 WIDTH RATIO. hierarchical width / pooled-item width. If this is large, the item count was
     never the evidence.

DECLARED BIAS
  Six sets is at the bottom of Prompt 8's "usable" band (2 case study / 3-4 weak / 5-8 usable /
  10+ strong) and they are not independent. A hierarchical CI over 6 correlated sets is itself
  optimistic. We report it as the honest floor, not as a solution.
==================================================================================
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "jacobian-lens"))

from jlens.hf import Layout, from_hf                                     # noqa: E402
from jlens.lens import JacobianLens                                      # noqa: E402
from anchor_evals import (EVAL_SETS, load_eval, readout, readout_position,  # noqa: E402
                          rank_of, token_ids_of)
from t13_transport_controls import _derangement                          # noqa: E402
from t17_reaggregate import band_of                                      # noqa: E402
from bootstrap import (paired_item_bootstrap, hierarchical_bootstrap,    # noqa: E402
                       nwin_interval, compare_units)

PYTHIA_LAYOUT = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                       embed="embed_in", lm_head="lm_head")
KS = (1, 2, 5, 10, 25, 100)
_XS = [math.log(k) for k in KS]


def _auc_from_curve(ys):
    a = sum((ys[i] + ys[i + 1]) / 2 * (_XS[i + 1] - _XS[i]) for i in range(len(KS) - 1))
    return a / (_XS[-1] - _XS[0])


def per_pair_auc(model, transport, eval_name, layers, max_items=None, max_seq_len=256,
                 aggregation="min", persist_kfrac=0.5):
    """The A.6 statistic decomposed to ONE VALUE PER (item, intermediate) PAIR.

    pass@k over a pool is the mean of the per-pair indicator 1[min_l rank <= k], so the pooled
    AUC is exactly the mean of the per-pair AUCs. That identity is what makes an item-level
    bootstrap of the published metric possible without changing the metric -- and it is
    asserted at runtime below.
    """
    items = load_eval(eval_name)
    if max_items:
        items = items[:max_items]
    out = []
    for item in items:
        prompt = item["prompt"].rstrip()
        pos = readout_position(model.tokenizer, eval_name, prompt)
        o = readout(model, transport, prompt, layers, pos, max_seq_len)
        for word in item["intermediates"]:
            ids = token_ids_of(model.tokenizer, word)
            if not ids:
                continue
            ranks = [rank_of(o[l], ids) for l in layers]
            if aggregation == "min":
                # EXISTENTIAL: recovered if readable at ANY layer. The anchor's A.6 form, and
                # the one sec F1 shows rewards diversity over accuracy.
                curve = [1.0 if min(ranks) <= k else 0.0 for k in KS]
            elif aggregation == "persist":
                # NON-EXISTENTIAL: recovered only if readable at >= K of the band's layers.
                K = max(1, int(round(persist_kfrac * len(layers))))
                curve = [1.0 if sum(r <= k for r in ranks) >= K else 0.0 for k in KS]
            else:
                raise ValueError(f"unknown aggregation {aggregation!r}")
            out.append(_auc_from_curve(curve))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--evals", default="all")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--aggregation", default="min", choices=["min", "persist"],
                    help="min = the anchor's EXISTENTIAL A.6 form; persist = the "
                         "non-existential fix from sec F1. J-vs-shuffled must be read under "
                         "persist, because min is the aggregation that launders it.")
    ap.add_argument("--persist-kfrac", type=float, default=0.5)
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
    layers = list(lens.source_layers)
    band = band_of(model.n_layers, layers)
    perm = _derangement(layers, a.seed)
    sets = EVAL_SETS if a.evals == "all" else a.evals.split(",")

    J = {l: lens.jacobians[l].float() for l in layers}
    Jsh = {l: J[perm[l]] for l in layers}

    def T_logit(h, l):
        return h.float()

    def T_j(h, l):
        return h.float() @ J[l].T.float()

    def T_sh(h, l):
        return h.float() @ Jsh[l].T.float()

    def T_scaled_id(h, l):                       # C1 self-test: must equal T_logit up to scale
        return 2.0 * h.float()

    arms = {"logit": T_logit, "jlens": T_j, "shuffled": T_sh, "scaled_id": T_scaled_id}
    per_set, sets_j_vs_i, sets_j_vs_sh = {}, {}, {}
    for ev in sets:
        scores = {}
        for name, T in arms.items():
            scores[name] = per_pair_auc(model, T, ev, band, a.max_items,
                                        aggregation=a.aggregation,
                                        persist_kfrac=a.persist_kfrac)
        n = min(len(v) for v in scores.values())
        scores = {k: v[:n] for k, v in scores.items()}
        # identity: pooled AUC must equal the mean of per-pair AUCs
        pooled = {k: float(torch.tensor(v).mean()) for k, v in scores.items()}
        r_ji = paired_item_bootstrap(scores["jlens"], scores["logit"],
                                     n_boot=a.n_boot, seed=a.seed)
        r_js = paired_item_bootstrap(scores["jlens"], scores["shuffled"],
                                     n_boot=a.n_boot, seed=a.seed)
        r_self = paired_item_bootstrap(scores["scaled_id"], scores["logit"],
                                       n_boot=min(a.n_boot, 500), seed=a.seed)
        per_set[ev] = {
            "n_pairs": n, "pooled_auc": pooled,
            "jlens_minus_logit": r_ji, "jlens_minus_shuffled": r_js,
            "C1_self_test_scaled_id_minus_logit": r_self,
            "C1_fires": abs(r_self["point"]) < 1e-9
                        and (r_self["ci_hi"] - r_self["ci_lo"]) < 1e-9,
            "survives_vs_logit": r_ji["excludes_zero"],
            "survives_vs_shuffled": r_js["excludes_zero"],
        }
        sets_j_vs_i[ev] = (scores["jlens"], scores["logit"])
        sets_j_vs_sh[ev] = (scores["jlens"], scores["shuffled"])
        print(f"  {ev:<14} J-I {r_ji['point']:+.4f} [{r_ji['ci_lo']:+.4f},{r_ji['ci_hi']:+.4f}]"
              f" {'SURVIVES' if r_ji['excludes_zero'] else 'not supported':<14}"
              f" | J-shuf {r_js['point']:+.4f} [{r_js['ci_lo']:+.4f},{r_js['ci_hi']:+.4f}]"
              f" {'SURVIVES' if r_js['excludes_zero'] else 'not supported'}")

    hi_ji = hierarchical_bootstrap(sets_j_vs_i, n_boot=a.n_boot, seed=a.seed)
    hi_js = hierarchical_bootstrap(sets_j_vs_sh, n_boot=a.n_boot, seed=a.seed)
    nw_ji = nwin_interval(sets_j_vs_i, n_boot=min(a.n_boot, 2000), seed=a.seed)
    nw_js = nwin_interval(sets_j_vs_sh, n_boot=min(a.n_boot, 2000), seed=a.seed)
    cu = compare_units(sets_j_vs_i, n_boot=min(a.n_boot, 2000), seed=a.seed)

    n_surv_i = sum(1 for v in per_set.values() if v["survives_vs_logit"])
    n_surv_s = sum(1 for v in per_set.values() if v["survives_vs_shuffled"])
    c1 = all(v["C1_fires"] for v in per_set.values())

    verdict = (f"vs LOGIT: {n_surv_i}/{len(sets)} sets survive their own CI, "
               f"hierarchical {'EXCLUDES' if hi_ji['excludes_zero'] else 'INCLUDES'} 0. "
               f"vs SHUFFLED: {n_surv_s}/{len(sets)} survive, hierarchical "
               f"{'EXCLUDES' if hi_js['excludes_zero'] else 'INCLUDES'} 0.")

    out = {
        "experiment": "T22 / E26 — confidence intervals on every read headline",
        "status": "EXPLORATORY", "prereg": "in-file header, fixed before the first number",
        "decision_rule": ("per set: CI excludes 0 -> contribution survives; includes 0 -> not "
                          "supported and n_win must be reported without it. Hierarchical CI "
                          "including 0 while per-set CIs exclude it -> does not generalise."),
        "VERDICT": verdict,
        "C1_self_test_all_sets_fired": c1,
        "aggregation": a.aggregation, "persist_kfrac": a.persist_kfrac,
        "aggregation_note": ("min is EXISTENTIAL and sec F1 shows it launders the J-vs-shuffled "
                             "difference; persist is the non-existential fix. Any J-vs-shuffled "
                             "claim must be read off the persist run."),
        "model": a.model, "lens": a.lens, "band": band, "n_boot": a.n_boot, "seed": a.seed,
        "derangement_layer_map": {str(k): v for k, v in perm.items()},
        "per_set": per_set,
        "hierarchical_jlens_minus_logit": hi_ji,
        "hierarchical_jlens_minus_shuffled": hi_js,
        "nwin_vs_logit": nw_ji, "nwin_vs_shuffled": nw_js,
        "C2_width_ratio": cu["width_ratio"], "C2_detail": {k: cu[k] for k in
                                                           ("width_item", "width_hierarchical",
                                                            "verdict")},
        "env": {"torch": torch.__version__}, "argv": sys.argv,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n  C1 self-test fired on every set: {c1}")
    print(f"  hierarchical J-I  {hi_ji['point']:+.4f} "
          f"[{hi_ji['ci_lo']:+.4f}, {hi_ji['ci_hi']:+.4f}]")
    print(f"  hierarchical J-sh {hi_js['point']:+.4f} "
          f"[{hi_js['ci_lo']:+.4f}, {hi_js['ci_hi']:+.4f}]")
    print(f"  n_win vs logit {nw_ji['n_win']}/{nw_ji['n_sets']} "
          f"CI [{nw_ji['ci_lo']:.1f}, {nw_ji['ci_hi']:.1f}]  |  "
          f"vs shuffled {nw_js['n_win']}/{nw_js['n_sets']} "
          f"CI [{nw_js['ci_lo']:.1f}, {nw_js['ci_hi']:.1f}]")
    print(f"  C2 width ratio (hierarchical / item) = {cu['width_ratio']:.2f}x")
    print(f"  VERDICT: {verdict}")
    print(f"wrote {a.out} ({out['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
