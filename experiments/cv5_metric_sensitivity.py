#!/usr/bin/env python3
"""
CV5 — at E66's perturbation magnitude, is rank blind where z is sighted?

Pre-registration: docs/experiments/preregs/CV5_metric_sensitivity.md, committed before this file.
The decision rule is transcribed verbatim into DECISION_RULE and is not reinterpretable.

  .venv/bin/python experiments/cv5_metric_sensitivity.py
"""
import json, os, sys, statistics as st

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in ("src", "jacobian-lens", "experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from anchor_evals import load_eval, token_ids_of, readout_position   # noqa: E402

MODEL = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
KS = (1, 2, 5, 10, 20, 50, 100)
BASE = os.path.join(ROOT, "results", "e48", "lens_INSTREAM_Pile-CC_410m_n200_s0.pt")

# magnitudes, and what each represents (see prereg)
MAGS = [0.0, 1.06e-07, 2.535e-03, 1.0e-02, 5.0e-02]
NOISE_SEEDS = [0, 1, 2]

# pooled seed SDs from CV3's 5-corpus x 3-seed panel — the denominators
SD_RANK, SD_Z = 0.002581, 0.013407
C2_TARGET = 0.19810852520167826
DECISION_RULE = ("at r=2.535e-03: ACCEPT (RANK WAS BLIND) if S_z >= 1.0 and S_rank < 0.5 | "
                 "REJECT (FUNCTIONALLY IDENTICAL) if S_z < 0.5 | UNCLEAR otherwise")


def ksummary(curve):
    """Flat mean over the 7 k values — this programme's convention (CONFIG_MATRIX.md)."""
    return sum(curve) / len(curve)


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
    print(f"  {A[BAND[0]].shape[0]} items, {len(pairs)} pairs", flush=True)

    J0 = torch.load(BASE, map_location="cpu")["J"]
    J0 = {l: J0[l].float() for l in BAND}

    def score(Jmap):
        """-> (rank_min_flatmean7, z_mean). Identical harness to CV3."""
        rk = torch.empty(len(pairs), len(BAND), dtype=torch.int32)
        zs = torch.empty(len(pairs), len(BAND))
        with torch.no_grad():
            for li, l in enumerate(BAND):
                h = A[l] if Jmap is None else A[l] @ Jmap[l].T
                lg = model.unembed(h).float()
                mu, sd = lg.mean(1, keepdim=True), lg.std(1, keepdim=True)
                for pi, (_, r, sy) in enumerate(pairs):
                    row = lg[r]
                    best = row[sy].max()
                    rk[pi, li] = int((row > best).sum().item()) + 1
                    zs[pi, li] = float((best - mu[r]) / sd[r])
                del lg
        mn = rk.min(1).values
        per_r, per_z = {n: [] for n in ADMITTED}, {n: [] for n in ADMITTED}
        for pi, (n, _, _) in enumerate(pairs):
            per_r[n].append(int(mn[pi])); per_z[n].append(float(zs[pi].max()))
        rank = st.mean([ksummary([sum(1 for x in per_r[n] if x <= k) / len(per_r[n]) for k in KS])
                        for n in ADMITTED])
        return rank, st.mean([st.mean(per_z[n]) for n in ADMITTED])

    print("scoring logit anchor ...", flush=True)
    logit_rank, _ = score(None)
    base_rank, base_z = score(J0)
    print(f"  logit_I rank={logit_rank:.8f}   J0 rank={base_rank:.5f} z={base_z:.4f}", flush=True)

    rows = {}
    for r in MAGS:
        if r == 0.0:
            rk, z = score({l: J0[l].clone() for l in BAND})
            rows["0.0"] = {"r": 0.0, "n_seeds": 1,
                           "rank": [rk], "z": [z],
                           "S_rank": abs(rk - base_rank) / SD_RANK,
                           "S_z": abs(z - base_z) / SD_Z}
            print(f"  r=0.0        S_rank={rows['0.0']['S_rank']:.3f} S_z={rows['0.0']['S_z']:.3f}",
                  flush=True)
            continue
        rks, zsv = [], []
        for s in NOISE_SEEDS:
            g = torch.Generator().manual_seed(1000 * s + int(r * 1e9) % 9973)
            Jp = {}
            for l in BAND:
                n = torch.randn(J0[l].shape, generator=g)
                # scale so max|dJ| / max|J| == r
                n = n / n.abs().max() * (J0[l].abs().max() * r)
                Jp[l] = J0[l] + n
            a, b = score(Jp)
            rks.append(a); zsv.append(b)
        rows[f"{r:g}"] = {
            "r": r, "n_seeds": len(NOISE_SEEDS), "rank": rks, "z": zsv,
            "S_rank": abs(st.mean(rks) - base_rank) / SD_RANK,
            "S_z": abs(st.mean(zsv) - base_z) / SD_Z}
        print(f"  r={r:<11g} S_rank={rows[f'{r:g}']['S_rank']:.3f} "
              f"S_z={rows[f'{r:g}']['S_z']:.3f}", flush=True)

    key = rows["0.002535"]
    verdict = ("ACCEPT — RANK WAS BLIND" if key["S_z"] >= 1.0 and key["S_rank"] < 0.5 else
               "REJECT — THE OPERATORS ARE FUNCTIONALLY IDENTICAL" if key["S_z"] < 0.5 else
               "UNCLEAR — report and stop")
    harness = rows["0.05"]
    ctrl = {
        "C1_zero_perturbation_is_noop": {
            "required": "r=0 reproduces the unperturbed read exactly in both spaces",
            "S_rank": rows["0.0"]["S_rank"], "S_z": rows["0.0"]["S_z"],
            "fires": rows["0.0"]["S_rank"] == 0.0 and rows["0.0"]["S_z"] == 0.0},
        "C2_logit_anchor": {
            "required": f"logit_I flat-mean-7 min == {C2_TARGET} within 1e-6",
            "observed": logit_rank, "abs_diff": abs(logit_rank - C2_TARGET),
            "fires": abs(logit_rank - C2_TARGET) <= 1e-6},
        "C3_harness_responds": {
            "required": "at r=5e-2 at least one space must move >= 1 seed SD, else the scoring "
                        "path is not responding to the operator and every number is void",
            "S_rank": harness["S_rank"], "S_z": harness["S_z"],
            "fires": harness["S_rank"] >= 1.0 or harness["S_z"] >= 1.0},
        "C4_monotone_in_r": {
            "required": "S_z non-decreasing across magnitudes",
            "S_z_by_r": [rows[k]["S_z"] for k in ("0.0", "1.06e-07", "0.002535", "0.01", "0.05")],
            "fires": all(rows[a]["S_z"] <= rows[b]["S_z"] + 1e-9 for a, b in
                         zip(("0.0", "1.06e-07", "0.002535", "0.01"),
                             ("1.06e-07", "0.002535", "0.01", "0.05")))},
    }

    out = {
        "experiment": "CV5 — metric sensitivity: is rank blind where z is sighted?",
        "prereg": "docs/experiments/preregs/CV5_metric_sensitivity.md",
        "status": "PRE-REGISTERED",
        "decision_rule_verbatim": DECISION_RULE,
        "model": MODEL, "band": BAND, "admitted_sets": ADMITTED, "K": list(KS),
        "base_operator": os.path.relpath(BASE, ROOT),
        "readout_convention": "STRIPPED (corrected), no prefix; flat-mean-7 k-summary",
        "denominators_from_cv3": {"pooled_seed_sd_rank": SD_RANK, "pooled_seed_sd_z": SD_Z},
        "unperturbed": {"rank": base_rank, "z": base_z, "logit_I_rank": logit_rank},
        "by_magnitude": rows,
        "PRIMARY_at_e66_magnitude": {"r": 2.535e-03, "S_rank": key["S_rank"], "S_z": key["S_z"]},
        "VERDICT": verdict,
        "controls": ctrl,
        "declared_bias": ("Gaussian noise is not a fitter difference. This bounds sensitivity to a "
                          "perturbation of that size; it does not reproduce E66's specific operator "
                          "delta, which is structural (accumulation order, batching, kernels)."),
        "e66_context": ("results/e66_fitter_equivalence_cuda.json: trainval vs stored operator "
                        "max_rel 2.535e-03 with read difference exactly 0.0; its C1 did not fire "
                        "and its verdict flags the stored lens's provenance. The flagged file is "
                        "the base operator used here and in CV3."),
    }
    dest = os.path.join(ROOT, "results", "cv5_metric_sensitivity_410m.json")
    try:
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from provenance import write_result
        write_result(dest, out, script=__file__, experiment="CV5", inputs=[BASE, os.path.join(ROOT, "results", "e66_fitter_equivalence_cuda.json"), os.path.join(ROOT, "results", "cv3_margins_410m.json")])
    except Exception as e:                       # never swallow silently — rule 0b
        print(f"  !! provenance stamp FAILED: {e!r}", file=sys.stderr)
        json.dump(out, open(dest, "w"), indent=1)
    print("\n" + "=" * 64)
    print(f"at E66's magnitude r=2.535e-03:  S_rank={key['S_rank']:.3f}  S_z={key['S_z']:.3f}")
    print(f"VERDICT: {verdict}")
    print("controls: " + ", ".join(f"{k}={v['fires']}" for k, v in ctrl.items()))
    print("wrote", os.path.relpath(dest, ROOT))


if __name__ == "__main__":
    main()
