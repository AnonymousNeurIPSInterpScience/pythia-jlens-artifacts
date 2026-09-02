#!/usr/bin/env python3
"""
D3 — EXPLORATORY DIAGNOSTIC: decompose CV3's corpus effect by eval family.

STATUS: EXPLORATORY. No pre-registration, no decision rule, and none may be invented after the
fact. This recomputes stored quantities per family that CV3 reported only in aggregate.

WHY. CV4 Phase 1 showed the battery is heterogeneous in capability: at 2.8B, order-ops is 14.5%
top-1, multihop 6.5%, multilingual 0.9% -- and multilingual is 24% of the admitted battery
contributing floor at every scale across 40x of parameters. CV3's 27 sigma corpus effect is a mean
over five families with that composition. Aggregation can hide structure (the same failure the
margins check found at the metric level), so the per-family decomposition is worth having before
any restricted contrast is designed.

THE ORDERING COST, RECORDED. Computing this BEFORE pre-registering a restricted corpus contrast
means such a restriction can no longer be described as chosen on capability grounds alone. The
operator was told and elected to see the decomposition first. Any later restriction must be
disclosed as informed by this file.

  .venv/bin/python experiments/d3_corpus_by_family.py
"""
import json, math, os, sys, statistics as st
from collections import defaultdict

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in ("src", "jacobian-lens", "experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from anchor_evals import load_eval, token_ids_of, readout_position   # noqa: E402

MODEL = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
KS = (1, 2, 5, 10, 20, 50, 100)
INSTREAM = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github"]
SEEDS = [0, 1, 2]


def ksummary(c):
    return sum(c) / len(c)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from jlens.hooks import ActivationRecorder
    from jlens.hf import from_hf
    from t2_fastfit import PYTHIA_LAYOUT_T5

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)

    print("caching activations ...", flush=True)
    pairs, acts = [], {l: [] for l in BAND}
    with torch.no_grad():
        for name in ADMITTED:
            for it in load_eval(name):
                pr = it["prompt"] if isinstance(it["prompt"], str) else \
                    " ".join(m.get("content", "") for m in it["prompt"])
                pr = pr.rstrip()
                pos = readout_position(tok, name, pr)
                ids = model.encode(pr, max_length=256)
                with ActivationRecorder(model.layers, at=BAND) as rec:
                    model.forward(ids)
                    for l in BAND:
                        acts[l].append(rec.activations[l][0][pos].detach().float())
                r = len(acts[BAND[0]]) - 1
                for w in it["intermediates"]:
                    sy = token_ids_of(tok, w)
                    if sy:
                        pairs.append((name, r, sy))
    A = {l: torch.stack(acts[l]) for l in BAND}
    npf = {f: sum(1 for p in pairs if p[0] == f) for f in ADMITTED}
    print(f"  pairs per family: {npf}", flush=True)

    arms = {"logit_I": None}
    for c in INSTREAM:
        for s in SEEDS:
            p = os.path.join(ROOT, "results", "e48", f"lens_INSTREAM_{c}_410m_n200_s{s}.pt")
            if os.path.exists(p):
                arms[f"{c}|s{s}"] = p

    per_arm = {}
    for arm, path in arms.items():
        Jm = None if path is None else {l: torch.load(path, map_location="cpu")["J"][l].float()
                                        for l in BAND}
        rk = torch.empty(len(pairs), len(BAND), dtype=torch.int32)
        zs = torch.empty(len(pairs), len(BAND))
        with torch.no_grad():
            for li, l in enumerate(BAND):
                h = A[l] if Jm is None else A[l] @ Jm[l].T
                lg = model.unembed(h).float()
                mu, sd = lg.mean(1, keepdim=True), lg.std(1, keepdim=True)
                for pi, (_, r, sy) in enumerate(pairs):
                    row = lg[r]; best = row[sy].max()
                    rk[pi, li] = int((row > best).sum().item()) + 1
                    zs[pi, li] = float((best - mu[r]) / sd[r])
                del lg
        mn = rk.min(1).values
        pr_, pz_ = defaultdict(list), defaultdict(list)
        for pi, (f, _, _) in enumerate(pairs):
            pr_[f].append(int(mn[pi])); pz_[f].append(float(zs[pi].max()))
        per_arm[arm] = {
            f: {"rank": ksummary([sum(1 for x in pr_[f] if x <= k) / len(pr_[f]) for k in KS]),
                "z": st.mean(pz_[f])} for f in ADMITTED}
        print(f"  {arm:26} " + " ".join(f"{f[:5]}={per_arm[arm][f]['z']:.3f}" for f in ADMITTED),
              flush=True)

    out = {"experiment": "D3 — EXPLORATORY: CV3's corpus effect decomposed by eval family",
           "status": "EXPLORATORY — no pre-registration, no decision rule, none may be invented",
           "ordering_cost_recorded": (
               "computed BEFORE any restricted-contrast pre-registration; any later restriction "
               "to a family subset must be disclosed as informed by this file, not as chosen on "
               "capability grounds alone"),
           "model": MODEL, "band": BAND, "K": list(KS), "n_pairs_per_family": npf,
           "readout_convention": "STRIPPED (corrected), no prefix, flat-mean-7",
           "per_arm": per_arm, "by_family": {}}

    for f in ADMITTED:
        rows = {}
        for c in INSTREAM:
            v = [(per_arm[f"{c}|s{s}"][f]["rank"], per_arm[f"{c}|s{s}"][f]["z"])
                 for s in SEEDS if f"{c}|s{s}" in per_arm]
            rows[c] = {"rank_mean": st.mean(x for x, _ in v),
                       "rank_sd": st.stdev([x for x, _ in v]) if len(v) > 2 else 0.0,
                       "z_mean": st.mean(y for _, y in v),
                       "z_sd": st.stdev([y for _, y in v]) if len(v) > 2 else 0.0}
        def sp(key, sdk):
            m = [rows[c][key] for c in INSTREAM]
            pooled = math.sqrt(sum(rows[c][sdk] ** 2 for c in INSTREAM) / len(INSTREAM))
            return max(m) - min(m), pooled, (max(m) - min(m)) / pooled if pooled else None
        rs, rp, rr = sp("rank_mean", "rank_sd")
        zs_, zp, zr = sp("z_mean", "z_sd")
        out["by_family"][f] = {
            "n_pairs": npf[f], "logit_rank": per_arm["logit_I"][f]["rank"],
            "logit_z": per_arm["logit_I"][f]["z"], "per_corpus": rows,
            "rank_spread": rs, "rank_pooled_seed_sd": rp, "rank_spread_over_sd": rr,
            "z_spread": zs_, "z_pooled_seed_sd": zp, "z_spread_over_sd": zr,
            "order_by_z": sorted(INSTREAM, key=lambda c: -rows[c]["z_mean"]),
            "order_by_rank": sorted(INSTREAM, key=lambda c: -rows[c]["rank_mean"])}

    dest = os.path.join(ROOT, "results", "d3_corpus_by_family_410m.json")
    try:
        from provenance import write_result
        write_result(dest, out, script=__file__, experiment="D3",
                     inputs=[p for p in arms.values() if p] +
                            [os.path.join(ROOT, "results", "cv3_margins_410m.json")])
    except Exception as e:
        print(f"  !! provenance stamp FAILED: {e!r}", file=sys.stderr)
        json.dump(out, open(dest, "w"), indent=1)

    print("\n" + "=" * 78)
    print(f"{'family':13} {'n':>4} {'z spread':>9} {'z sd':>8} {'z x SD':>7} {'rank x SD':>10}  best->worst (z)")
    for f in ADMITTED:
        b = out["by_family"][f]
        print(f"{f:13} {b['n_pairs']:4} {b['z_spread']:9.4f} {b['z_pooled_seed_sd']:8.5f} "
              f"{b['z_spread_over_sd']:7.1f} {b['rank_spread_over_sd']:10.1f}  "
              + " > ".join(x[:11] for x in b["order_by_z"]))
    print("\nwrote", os.path.relpath(dest, ROOT))


if __name__ == "__main__":
    main()
