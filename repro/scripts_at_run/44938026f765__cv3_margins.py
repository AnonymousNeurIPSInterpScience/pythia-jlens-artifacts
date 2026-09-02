#!/usr/bin/env python3
"""
CV3 — MARGINS.  Is the corpus effect a score effect or a ranking artifact?

Pre-registration: docs/experiments/preregs/CV3_margins.md, committed before this file.
DO NOT reinterpret the decision rule.  It is transcribed verbatim into DECISION_RULE below.

Every number in this programme is a RANK statistic.  Rank is unstable when many tokens score
similarly, so a 13-45 seed-SD corpus effect could be operators genuinely scoring the target
differently, or a crowded neighbourhood amplifying sub-threshold score differences.  This measures
both spaces on identical cached activations and takes their ratio.

  .venv/bin/python experiments/cv3_margins.py
"""
import json, math, os, sys, glob, statistics as st
from collections import defaultdict

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "jacobian-lens"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from anchor_evals import load_eval, token_ids_of, readout_position   # noqa: E402

MODEL = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
KS = (1, 2, 5, 10, 20, 50, 100)
INSTREAM = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github"]
SEEDS = [0, 1, 2]
N_DERANGE = 3

DECISION_RULE = ("ACCEPT (SCORE EFFECT) if PRIMARY >= 0.50 | "
                 "REJECT (RANKING ARTIFACT) if PRIMARY < 0.25 — stop and alert, do not re-cut | "
                 "UNCLEAR if 0.25 <= PRIMARY < 0.50 — report and stop")
C1_TARGET = 0.19810852520167826   # e48_crossover_410m_rstrip.json : arms_admitted_mean.logit_I.min


def ksummary(curve):
    """FLAT MEAN over the 7 k values — the programme's convention, per
    docs/context/CONFIG_MATRIX.md ("k-summary: flat mean, 7 pts") and the paper's appendix
    score = (1/|K|) sum_k 1[...].  NOT a trapezoid AUC over log k: an earlier draft of this
    script used the AUC and control C1 caught it (0.18966 vs the stored 0.19811)."""
    return sum(curve) / len(curve)


def derange(layers, seed):
    """Fixed-point-free permutation of the band, by rejection. Matches the programme's control."""
    g = torch.Generator().manual_seed(seed)
    while True:
        p = torch.randperm(len(layers), generator=g).tolist()
        if all(p[i] != i for i in range(len(layers))):
            return {layers[i]: layers[p[i]] for i in range(len(layers))}


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from jlens.hooks import ActivationRecorder
    from jlens.hf import from_hf
    from t2_fastfit import PYTHIA_LAYOUT_T5

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)

    # ---- cache activations once, at each item's own readout position, no prefix
    print("caching activations ...", flush=True)
    pairs = []          # (eval_name, item_idx, [synonym ids])
    acts = {l: [] for l in BAND}
    row_of_item = []
    with torch.no_grad():
        for name in ADMITTED:
            for it in load_eval(name):
                prompt = it["prompt"].rstrip() if isinstance(it["prompt"], str) else \
                    " ".join(m.get("content", "") for m in it["prompt"]).rstrip()
                pos = readout_position(tok, name, prompt)
                ids = model.encode(prompt, max_length=256)
                with ActivationRecorder(model.layers, at=BAND) as rec:
                    model.forward(ids)
                    for l in BAND:
                        acts[l].append(rec.activations[l][0][pos].detach().float())
                r = len(row_of_item)
                row_of_item.append((name, r))
                for w in it["intermediates"]:
                    sy = token_ids_of(tok, w)
                    if sy:
                        pairs.append((name, r, sy))
    A = {l: torch.stack(acts[l]) for l in BAND}          # [n_items, d]
    n_items = A[BAND[0]].shape[0]
    print(f"  {n_items} items, {len(pairs)} scorable (item, intermediate) pairs", flush=True)

    # ---- arms
    arms = {"logit_I": None}
    for c in INSTREAM:
        for s in SEEDS:
            p = os.path.join(ROOT, "results", "e48", f"lens_INSTREAM_{c}_410m_n200_s{s}.pt")
            if os.path.exists(p):
                arms[f"J|{c}|s{s}"] = ("real", p)
    base = os.path.join(ROOT, "results", "e48", "lens_INSTREAM_Pile-CC_410m_n200_s0.pt")
    for d in range(N_DERANGE):
        arms[f"shuf|Pile-CC|s0|d{d}"] = ("shuf", base, d)
    print(f"  {len(arms)} arms", flush=True)

    # ---- score every arm on the same activations
    stats = {}
    c2_violations = 0
    for arm, spec in arms.items():
        if spec is None:
            Jmap = None
        else:
            J = torch.load(spec[1], map_location="cpu")["J"]
            if spec[0] == "real":
                Jmap = {l: J[l].float() for l in BAND}
            else:
                sig = derange(BAND, spec[2])
                Jmap = {l: J[sig[l]].float() for l in BAND}

        rank = torch.empty(len(pairs), len(BAND), dtype=torch.int32)
        marg = torch.empty(len(pairs), len(BAND))
        zsc = torch.empty(len(pairs), len(BAND))
        with torch.no_grad():
            for li, l in enumerate(BAND):
                h = A[l] if Jmap is None else A[l] @ Jmap[l].T
                lg = model.unembed(h).float()                     # [n_items, V]
                mu, sd = lg.mean(1, keepdim=True), lg.std(1, keepdim=True)
                for pi, (_, r, sy) in enumerate(pairs):
                    row = lg[r]
                    best = row[sy].max()
                    rank[pi, li] = int((row > best).sum().item()) + 1
                    masked = row.clone(); masked[sy] = float("-inf")
                    marg[pi, li] = float(best - masked.max())
                    zsc[pi, li] = float((best - mu[r]) / sd[r])
                del lg
        # C2: margin > 0 iff rank == 1
        c2_violations += int(((marg > 0) != (rank == 1)).sum().item())

        # RANK SPACE — reproduce the existing statistic exactly: min over layers, pass@k, AUC
        minrank = rank.min(1).values
        per_set_rank, per_set_marg, per_set_z = defaultdict(list), defaultdict(list), defaultdict(list)
        for pi, (name, _, _) in enumerate(pairs):
            per_set_rank[name].append(int(minrank[pi]))
            per_set_marg[name].append(float(marg[pi].max()))      # max margin = min-rank analogue
            per_set_z[name].append(float(zsc[pi].max()))
        rank_auc = st.mean([ksummary([sum(1 for r in per_set_rank[n] if r <= k) / len(per_set_rank[n])
                                 for k in KS]) for n in ADMITTED])
        stats[arm] = {
            "rank_min_auc": rank_auc,
            "margin_mean": st.mean([st.mean(per_set_marg[n]) for n in ADMITTED]),
            "z_mean": st.mean([st.mean(per_set_z[n]) for n in ADMITTED]),
        }
        print(f"  {arm:26} rank_auc={rank_auc:.5f}  margin={stats[arm]['margin_mean']:+.4f}  "
              f"z={stats[arm]['z_mean']:+.4f}", flush=True)

    # ---- controls
    c1 = abs(stats["logit_I"]["rank_min_auc"] - C1_TARGET)
    ctrl = {
        "C1_logit_reproduces_stored": {
            "required": f"|logit_I rank_min_auc - {C1_TARGET}| <= 1e-6",
            "observed": stats["logit_I"]["rank_min_auc"], "abs_diff": c1, "fires": c1 <= 1e-6},
        "C2_margin_rank_consistency": {
            "required": "margin > 0 iff rank == 1 on 100% of triples",
            "violations": c2_violations, "fires": c2_violations == 0},
        "C4_seed_sd_nondegenerate": {},
    }

    # ---- PRIMARY
    def spread_and_sd(key):
        per_corpus = {}
        for c in INSTREAM:
            v = [stats[f"J|{c}|s{s}"][key] for s in SEEDS if f"J|{c}|s{s}" in stats]
            per_corpus[c] = (st.mean(v), st.stdev(v) if len(v) > 2 else 0.0)
        sp = max(m for m, _ in per_corpus.values()) - min(m for m, _ in per_corpus.values())
        pooled = math.sqrt(sum(s * s for _, s in per_corpus.values()) / len(per_corpus))
        return sp, pooled, {c: {"mean": m, "seed_sd": s} for c, (m, s) in per_corpus.items()}

    sp_r, sd_r, pc_r = spread_and_sd("rank_min_auc")
    sp_m, sd_m, pc_m = spread_and_sd("margin_mean")
    sp_z, sd_z, pc_z = spread_and_sd("z_mean")
    ctrl["C4_seed_sd_nondegenerate"] = {"required": "pooled seed SD > 0 in both spaces",
                                        "rank": sd_r, "margin": sd_m,
                                        "fires": sd_r > 0 and sd_m > 0}
    R_rank = sp_r / sd_r if sd_r else None
    R_marg = sp_m / sd_m if sd_m else None
    R_z = sp_z / sd_z if sd_z else None
    PRIMARY = (R_marg / R_rank) if (R_rank and R_marg) else None
    verdict = ("ACCEPT — SCORE EFFECT" if PRIMARY is not None and PRIMARY >= 0.50 else
               "REJECT — RANKING ARTIFACT. STOP AND ALERT THE OPERATOR" if PRIMARY is not None and PRIMARY < 0.25
               else "UNCLEAR — report and stop")

    # ---- SECONDARY: the derangement contrast in both spaces
    real = stats["J|Pile-CC|s0"]
    shuf = {k: st.mean([stats[f"shuf|Pile-CC|s0|d{d}"][k] for d in range(N_DERANGE)])
            for k in ("rank_min_auc", "margin_mean", "z_mean")}
    secondary = {
        "real_minus_shuf_rank_min_auc": real["rank_min_auc"] - shuf["rank_min_auc"],
        "real_minus_shuf_margin": real["margin_mean"] - shuf["margin_mean"],
        "real_minus_shuf_z": real["z_mean"] - shuf["z_mean"],
        "note": ("if the derangement's advantage is present in rank space and absent in margin "
                 "space, `min` rewards rank reshuffling rather than better scoring"),
    }

    out = {
        "experiment": "CV3 — margins: score effect or ranking artifact?",
        "prereg": "docs/experiments/preregs/CV3_margins.md",
        "status": "PRE-REGISTERED",
        "decision_rule_verbatim": DECISION_RULE,
        "model": MODEL, "band": BAND, "admitted_sets": ADMITTED, "K": list(KS),
        "readout_convention": "STRIPPED (corrected); no read-context prefix",
        "n_items": n_items, "n_pairs": len(pairs),
        "per_arm": stats,
        "rank_space": {"spread": sp_r, "pooled_seed_sd": sd_r, "spread_over_sd": R_rank,
                       "per_corpus": pc_r},
        "margin_space": {"spread": sp_m, "pooled_seed_sd": sd_m, "spread_over_sd": R_marg,
                         "per_corpus": pc_m},
        "z_space": {"spread": sp_z, "pooled_seed_sd": sd_z, "spread_over_sd": R_z,
                    "per_corpus": pc_z},
        "PRIMARY": PRIMARY,
        "VERDICT": verdict,
        "SECONDARY_derangement": secondary,
        "controls": ctrl,
        "declared_bias": ("margin is measured against the top competitor, which is itself "
                          "operator-dependent; z and the raw logit are recorded so a "
                          "competitor-suppression effect is separable from a target-score effect"),
    }
    dest = os.path.join(ROOT, "results", "cv3_margins_410m.json")
    try:
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from provenance import write_result
        write_result(dest, out, script=__file__, experiment="CV3", inputs=[p for _, p in ((k, v[1]) for k, v in arms.items() if v) ] + [os.path.join(ROOT, "results", "e48_crossover_410m_rstrip.json")])
    except Exception as e:                       # never swallow silently — rule 0b
        print(f"  !! provenance stamp FAILED: {e!r}", file=sys.stderr)
        json.dump(out, open(dest, "w"), indent=1)
    print("\n" + "=" * 66)
    print(f"rank space   spread {sp_r:.5f} / sd {sd_r:.6f} = {R_rank:.2f}")
    print(f"margin space spread {sp_m:.5f} / sd {sd_m:.6f} = {R_marg:.2f}")
    print(f"z space      spread {sp_z:.5f} / sd {sd_z:.6f} = {R_z:.2f}")
    print(f"PRIMARY = {PRIMARY:.4f}   ->  {verdict}")
    print(f"controls: " + ", ".join(f"{k}={v.get('fires')}" for k, v in ctrl.items()))
    print(f"SECONDARY real-shuf: rank {secondary['real_minus_shuf_rank_min_auc']:+.5f}  "
          f"margin {secondary['real_minus_shuf_margin']:+.4f}  z {secondary['real_minus_shuf_z']:+.4f}")
    print("wrote", os.path.relpath(dest, ROOT))


if __name__ == "__main__":
    main()
