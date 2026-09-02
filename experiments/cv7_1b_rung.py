#!/usr/bin/env python3
"""
CV7 — the 1B rung. SCORING ONLY: no fit, no GPU.

Pre-registration: docs/experiments/preregs/CV7_1b_rung.md, committed BEFORE this file.
DO NOT reinterpret the decision rule. It is transcribed verbatim into DECISION_RULE below.

The 15 operators already exist -- `results/ladder1b_b613/lens_<corpus>_1b_n200_s<seed>.pt`, fitted
for E62 at N=200 with `source_layers` [6,13]. This caches h_t once for the admitted five families
and scores those 15 plus the unfitted identity arm against it.

EVERY SCORING FUNCTION IS IMPORTED FROM cv6_per_family_ladder, not reimplemented. That is
deliberate: CV6's control C0 proved that scorer reproduces D3 on the stored 410M operators at
EXACTLY 0.000e+00, with a negative control at the legacy readout separating at 1.997. Copying the
code would forfeit that guarantee; importing it inherits it.

THE ONE THING NOT TO DO WITH THE OUTPUT. These lenses are N=200 and CV6's 1.4B/2.8B are N=25. R is
a noise-scaled ratio whose denominator moves with N (measured 2.29x-5.70x between those two N), so
R(1B) is comparable to 410M and NOT to 1.4B. Cross-ladder, compare the z SPREAD.

  .venv/bin/python experiments/cv7_1b_rung.py            # score  (~minutes, CPU)
  .venv/bin/python experiments/cv7_1b_rung.py --adjudicate
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
import statistics as st

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in ("src", "jacobian-lens", "experiments"):
    sys.path.insert(0, os.path.join(ROOT, _p))

from cv6_per_family_ladder import (                                    # noqa: E402
    ADMITTED, INSTREAM, SEEDS, KS, band_for, cache_activations, pair_tensors,
    score_arm, kendall_tau, per_family_table,
)

MODEL = "EleutherAI/pythia-1b-deduped"
SHORT = "1b"
N_FIT = 200                       # the stored operators' N; NOT a choice made here
LENS_DIR = os.path.join(ROOT, "results", "ladder1b_b613")

DECISION_RULE = (
    "Evaluated PER FAMILY; the verdict is the count across the five. Thresholds inherited VERBATIM "
    "from CV6_per_family_ladder.md, which fixed them before any ladder number existed. "
    "REPLICATES AT 1B if R(f,1B) >= 10 on >= 4 of 5 families AND ordering tau >= 0.6 on >= 3 of 5. "
    "ATTENUATES AT 1B if R(f,1B) < 5 on >= 4 of 5 families. "
    "UNCLEAR otherwise, including a split -- report the table and stop. "
    "Do not pool. Do not drop a family. Do not re-cut R across mismatched N."
)

# D3's pair counts. C5: a different battery is a different experiment.
D3_PAIRS = {"multihop": 103, "multilingual": 394, "order-ops": 110, "poetry": 98, "typo": 96}


def lens_path(corpus, seed):
    return os.path.join(LENS_DIR, f"lens_{corpus}_{SHORT}_n{N_FIT}_s{seed}.pt")


def run(a):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from jlens.hf import from_hf
    from t2_fastfit import PYTHIA_LAYOUT_T5

    dev = a.device
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    band = band_for(model.n_layers)
    print(f"{SHORT}  n_layers={model.n_layers}  d_model={model.d_model}  "
          f"band={band} ({len(band)} layers)  device={dev}", flush=True)

    # ---- C2 + C4, BEFORE any scoring: the band and N are read off every artifact
    c2, c4 = {}, {}
    for c in INSTREAM:
        for s in SEEDS:
            p = lens_path(c, s)
            if not os.path.exists(p):
                raise SystemExit(f"ABORT: missing operator {os.path.relpath(p, ROOT)}")
            d = torch.load(p, map_location="cpu", weights_only=False)
            c2[f"{c}|s{s}"] = list(d["source_layers"])
            c4[f"{c}|s{s}"] = int(d["n_prompts"])
    band_ok = all(v == band for v in c2.values())
    n_ok = set(c4.values()) == {N_FIT}
    print(f"  C2 band on all 15 artifacts == rule: {band_ok}", flush=True)
    print(f"  C4 n_prompts on all 15 artifacts: {sorted(set(c4.values()))} -> {n_ok}", flush=True)
    if not (band_ok and n_ok):
        raise SystemExit("ABORT: C2 or C4 failed; the run is void. Do not score.")

    print("caching h_t (one forward per eval item, corrected readout) ...", flush=True)
    t0 = time.time()
    A, pairs, n_items = cache_activations(model, tok, band, dev)
    rows, syn_ids, syn_mask = pair_tensors(pairs, dev)
    npf = {f: sum(1 for p in pairs if p[0] == f) for f in ADMITTED}
    print(f"  {n_items} items, {len(pairs)} pairs {npf}  ({time.time()-t0:.0f}s)", flush=True)
    if npf != D3_PAIRS:
        raise SystemExit(f"ABORT: C5 failed -- pairs {npf} != D3 {D3_PAIRS}")

    out = {
        "experiment": "CV7 — the 1B rung, scoring only",
        "prereg": "docs/experiments/preregs/CV7_1b_rung.md",
        "status": "PRE-REGISTERED",
        "decision_rule_verbatim": DECISION_RULE,
        "model": MODEL, "short": SHORT, "n_layers": model.n_layers, "d_model": model.d_model,
        "band": band, "band_rule": "floor(0.38*L)..floor(0.92*L), layers < L-2",
        "band_read_off_artifacts": c2, "band_matches_rule": band_ok,
        "N": N_FIT, "n_prompts_per_arm": c4,
        "operators_from": "results/ladder1b_b613/ (fitted for E62; NOT refitted here)",
        "device": dev, "dtype": "float32",
        "operator_storage_dtype": "float16",
        "device_caveat": ("operators fitted on CUDA, scored on CPU. D2 measured a CUDA-vs-CPU "
                          "cell-level divergence of 2.774e-04. This is D3's situation, which is "
                          "what makes 1B-vs-410M the matched comparison."),
        "torch": torch.__version__,
        "K": list(KS), "admitted_sets": ADMITTED, "corpora": INSTREAM, "seeds": SEEDS,
        "readout_convention": "STRIPPED (corrected), no prefix, flat-mean-7",
        "pooling_convention": ("flat pool over (item, intermediate) pairs WITHIN a family; rank = "
                               "min over band then flat-mean-7 pass@k; z = max over band then mean"),
        "n_items": n_items, "n_pairs": len(pairs), "n_pairs_per_family": npf,
        "per_arm": {},
    }
    dest = a.out or os.path.join(ROOT, "results", "cv7", f"cv7_{SHORT}_n{N_FIT}.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def flush(final=False):
        if final:
            try:
                from provenance import write_result
                write_result(dest, out, script=__file__, experiment="CV7",
                             inputs=[lens_path(c, s) for c in INSTREAM for s in SEEDS])
                return
            except Exception as e:
                print(f"  !! provenance stamp FAILED: {e!r}", file=sys.stderr)
        json.dump(out, open(dest, "w"), indent=1, default=str)

    # ---- C1: the identity arm before any J arm
    out["per_arm"]["logit_I"] = score_arm(model, A, band, rows, syn_ids, syn_mask, pairs, None)
    pooled = st.mean(out["per_arm"]["logit_I"][f]["rank"] for f in ADMITTED)
    out["logit_I_pooled_rank_flatmean7"] = pooled
    print(f"  logit_I  pooled rank {pooled:.11f}  " +
          " ".join(f"{f[:5]}={out['per_arm']['logit_I'][f]['z']:.3f}" for f in ADMITTED), flush=True)
    flush()

    for c in INSTREAM:
        for s in SEEDS:
            arm = f"J|{c}|s{s}"
            J = torch.load(lens_path(c, s), map_location="cpu", weights_only=False)["J"]
            Jm = {l: J[l].float().to(dev) for l in band}
            t = time.time()
            out["per_arm"][arm] = score_arm(model, A, band, rows, syn_ids, syn_mask, pairs, Jm)
            del J, Jm
            print(f"  {arm:26} {time.time()-t:5.0f}s  " +
                  " ".join(f"{f[:5]}={out['per_arm'][arm][f]['z']:.3f}" for f in ADMITTED),
                  flush=True)
            flush()

    flush(final=True)
    print(f"\nwrote {os.path.relpath(dest, ROOT)}")
    return 0


def adjudicate(a):
    src = os.path.join(ROOT, "results", "cv7", f"cv7_{SHORT}_n{N_FIT}.json")
    if not os.path.exists(src):
        raise SystemExit(f"ABORT: {os.path.relpath(src, ROOT)} missing — score first")
    rec = json.load(open(src))
    d3p = os.path.join(ROOT, "results", "d3_corpus_by_family_410m.json")
    d3 = json.load(open(d3p))
    cv6p = os.path.join(ROOT, "results", "cv6_per_family_ladder.json")
    cv6 = json.load(open(cv6p)) if os.path.exists(cv6p) else None

    fam = per_family_table(rec["per_arm"])
    for f in ADMITTED:
        fam[f]["kendall_tau_vs_410m"] = kendall_tau(fam[f]["order_by_z"],
                                                    d3["by_family"][f]["order_by_z"])

    R = {f: fam[f]["R"] for f in ADMITTED}
    TAU = {f: fam[f]["kendall_tau_vs_410m"] for f in ADMITTED}
    n_ge10 = sum(1 for f in ADMITTED if R[f] is not None and R[f] >= 10)
    n_lt5 = sum(1 for f in ADMITTED if R[f] is not None and R[f] < 5)
    n_tau = sum(1 for f in ADMITTED if TAU[f] is not None and TAU[f] >= 0.6)
    verdict = ("REPLICATES AT 1B" if (n_ge10 >= 4 and n_tau >= 3)
               else "ATTENUATES AT 1B" if n_lt5 >= 4 else "UNCLEAR")

    # the ONLY legitimate cross-ladder comparison: z spread, at matched or stated N
    spread = {"410m_N200": {f: d3["by_family"][f]["z_spread"] for f in ADMITTED},
              "1b_N200": {f: fam[f]["z_spread"] for f in ADMITTED}}
    if cv6:
        for m in ("1.4b", "2.8b"):
            spread[f"{m}_N25"] = {f: cv6["by_model"][m]["by_family"][f]["z_spread"]
                                  for f in ADMITTED}

    out = {
        "experiment": "CV7 — does the per-family corpus effect hold at the 1B rung?",
        "prereg": "docs/experiments/preregs/CV7_1b_rung.md",
        "status": "PRE-REGISTERED",
        "decision_rule_verbatim": DECISION_RULE,
        "model": MODEL, "band": rec["band"], "N": N_FIT,
        "device": rec["device"], "dtype": rec["dtype"],
        "readout_convention": rec["readout_convention"],
        "pooling_convention": rec["pooling_convention"],
        "n_pairs_per_family": rec["n_pairs_per_family"],
        "logit_I": rec["per_arm"]["logit_I"],
        "logit_I_pooled_rank_flatmean7": rec["logit_I_pooled_rank_flatmean7"],
        "by_family": fam,
        "PRIMARY": {"R_per_family_1b": R, "kendall_tau_vs_410m": TAU,
                    "n_families_R_ge_10": n_ge10, "n_families_R_lt_5": n_lt5,
                    "n_families_tau_ge_0.6": n_tau},
        "VERDICT": verdict,
        "SECONDARY_order_ops_tau_sign": {
            "what": ("pre-specified, NO threshold and NO verdict: order-ops carries tau -0.40 at "
                     "1.4B, the only negative tau in CV6. Its sign at 1B says whether that is a "
                     "property of the family or of the 1.4B checkpoint"),
            "tau_1b_order_ops": TAU["order-ops"],
            "tau_1.4b_order_ops": (cv6 and cv6["by_model"]["1.4b"]["by_family"]["order-ops"]
                                   ["kendall_tau_vs_410m"]),
            "tau_2.8b_order_ops": (cv6 and cv6["by_model"]["2.8b"]["by_family"]["order-ops"]
                                   ["kendall_tau_vs_410m"]),
            "reading": ("negative at BOTH 1B and 1.4B -> a property of order-ops; positive at 1B "
                        "-> specific to the 1.4B checkpoint. n=1 rung either way; a sign, not a "
                        "result"),
        },
        "CROSS_LADDER_z_spread": {
            "why_not_R": ("R is a noise-scaled ratio and its denominator moves with N -- measured "
                          "2.29x-5.70x between N=200 and N=25. 1B and 410M are both N=200 and "
                          "their R is comparable; CV6's 1.4B and 2.8B are N=25 and theirs is not "
                          "comparable to either. Rank rungs by SPREAD, and state the N."),
            "z_spread_by_rung": spread,
        },
        "controls": {
            "C1_identity_arm": {
                "required": "logit_I emitted for all five families before any J arm is graded",
                "emitted": sorted(rec["per_arm"]["logit_I"].keys()),
                "pooled_rank_flatmean7": rec["logit_I_pooled_rank_flatmean7"],
                "scorer_validated_by": ("CV6 control C0: 0.000e+00 against D3 on the stored 410M "
                                        "operators; negative control separates at 1.997. The same "
                                        "scorer is IMPORTED here, not reimplemented"),
                "fires": set(rec["per_arm"]["logit_I"].keys()) == set(ADMITTED)},
            "C2_band_read_off_artifacts": {
                "required": "every one of the 15 lenses carries source_layers == band_for(16)",
                "rule": band_for(rec["n_layers"]), "observed_distinct":
                    [list(x) for x in {tuple(v) for v in rec["band_read_off_artifacts"].values()}],
                "fires": rec["band_matches_rule"]},
            "C3_seed_sd_nondegenerate": {
                "required": "pooled_seed_sd_z(f) > 0 for all five families",
                "per_family": {f: fam[f]["z_pooled_seed_sd"] for f in ADMITTED},
                "void_cells": [f for f in ADMITTED if not fam[f]["z_pooled_seed_sd"] > 0],
                "fires": all(fam[f]["z_pooled_seed_sd"] > 0 for f in ADMITTED)},
            "C4_N_identical": {
                "required": f"all 15 lenses record n_prompts == {N_FIT}",
                "observed": sorted(set(rec["n_prompts_per_arm"].values())),
                "fires": set(rec["n_prompts_per_arm"].values()) == {N_FIT}},
            "C5_battery_identical_to_D3": {
                "required": "pair counts per family equal D3's exactly",
                "observed": rec["n_pairs_per_family"], "expected": D3_PAIRS,
                "fires": rec["n_pairs_per_family"] == D3_PAIRS},
        },
        "declared_bias": [
            "ORDERING: written after CV6's 1.4B and 2.8B tables were seen. Thresholds are "
            "inherited verbatim from CV6, but the decision to run 1B is informed by results, so "
            "this is a disclosed follow-up and not a blind extension.",
            "1B is 16 layers at d_model 2048 against 1.4B's 24 at the SAME width -- shallower, not "
            "narrower -- so its band is 8 layers against 13 and 18. min-over-band and max-over-band "
            "are both band-width sensitive (D1's union effect). A 1B-vs-1.4B difference may be a "
            "BAND-WIDTH effect and nothing in this design separates them.",
            f"N={N_FIT} here vs N=25 in CV6. The single easiest error to make with this table.",
            "Operators fitted on CUDA, scored on CPU; D2 measured 2.774e-04 at the cell level.",
            "The 5 corpora are a fixed panel, not a sample. n=5 is the replication unit.",
        ],
    }
    dest = os.path.join(ROOT, "results", "cv7_1b_rung.json")
    try:
        from provenance import write_result
        write_result(dest, out, script=__file__, experiment="CV7",
                     inputs=[src, d3p] + ([cv6p] if cv6 else []))
    except Exception as e:
        print(f"  !! provenance stamp FAILED: {e!r}", file=sys.stderr)
        json.dump(out, open(dest, "w"), indent=1, default=str)

    print("\n" + "=" * 92)
    print(f"1B  band={rec['band'][0]}..{rec['band'][-1]}  N={N_FIT}  {rec['device']}")
    print(f"  {'family':13} {'n':>4} {'z spread':>9} {'z seed SD':>10} {'R':>8} {'tau':>6}  best->worst (z)")
    for f in ADMITTED:
        x = fam[f]
        print(f"  {f:13} {x['n_pairs']:4} {x['z_spread']:9.4f} {x['z_pooled_seed_sd']:10.5f} "
              f"{x['R']:8.2f} {x['kendall_tau_vs_410m']:+6.2f}  "
              + " > ".join(y[:11] for y in x["order_by_z"]))
    print(f"\ncontrols: " + ", ".join(f"{k.split('_')[0]}={v['fires']}"
                                      for k, v in out["controls"].items()))
    print(f"R>=10 on {n_ge10}/5   R<5 on {n_lt5}/5   tau>=0.6 on {n_tau}/5")
    print(f"VERDICT = {verdict}")
    print(f"\nCROSS-LADDER z SPREAD (compare THIS, not R -- N is stated per rung):")
    print(f"  {'family':13} " + " ".join(f"{k:>12}" for k in spread))
    for f in ADMITTED:
        print(f"  {f:13} " + " ".join(f"{spread[k][f]:12.4f}" for k in spread))
    print("\nwrote", os.path.relpath(dest, ROOT))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--adjudicate", action="store_true")
    a = ap.parse_args()
    return adjudicate(a) if a.adjudicate else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
