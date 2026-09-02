#!/usr/bin/env python3
"""t7_bootstrap_audit.py — paired item-level bootstrap for the T7 transport comparison.

WHY THIS EXISTS. `anchor_evals.pass_at_k` returns a point AUC and discards the per-item
fractions it computed. No T7 result in this repo carries an uncertainty of any kind, so the
headline `n_win` (evals where J-lens AUC > logit AUC) has never been checked against sampling
noise over items. RIGOR_SKILL.md axis 6 (power) is scored 0 for every T7 cell as a result, and
the PREREG_PYTHIA_T7 primary outcome is a sign count with no effect-size floor: a +0.0001 win
counts exactly as much as a +0.12 win.

WHAT IT DOES. Re-runs the exact T7 readout (same transports, same layers, same items, same
metric) but retains the per-item pass@k fraction vector for each (eval, transport). Because both
transports are scored on the SAME items, the bootstrap is PAIRED: one resample of item indices
is applied to both arms, so the delta CI reflects only item sampling, not arm-vs-arm variance.

Reports, per model:
  - per-set AUC delta with a 95% percentile CI and the two-sided sign-flip probability
  - the bootstrap distribution of `n_win`, i.e. P(n_win >= 5) under item resampling
  - which set wins are "floor wins" (both arms' pass@k = 0 for all k <= 10)

EXPLORATORY. This is a secondary analysis of data already burned per PREREG_PYTHIA_T7 §0.
It does not touch the pre-registered confirmatory test and cannot rescue or falsify H1.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

import torch

torch.set_num_threads(min(8, os.cpu_count() or 8))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jlens.hf import Layout, from_hf  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402
from anchor_evals import (EVAL_SETS, load_eval, readout, readout_position,  # noqa: E402
                          rank_of, token_ids_of, transport_jlens, transport_logit)

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")
KS = (1, 2, 5, 10, 25, 100)
_LOGK = [math.log(k) for k in KS]
_SPAN = _LOGK[-1] - _LOGK[0]


def auc_from_curve(curve: list[float]) -> float:
    """Normalized AUC over log k — identical to anchor_evals.pass_at_k's trapezoid."""
    a = sum((curve[i] + curve[i + 1]) / 2 * (_LOGK[i + 1] - _LOGK[i]) for i in range(len(KS) - 1))
    return a / _SPAN


@torch.no_grad()
def per_item_fractions(model, transport, eval_name: str, layers) -> list[list[float]]:
    """[n_scored_items][len(KS)] — the per-item pass@k fractions pass_at_k averages away."""
    out = []
    for item in load_eval(eval_name):
        prompt = item["prompt"].rstrip()
        pos = readout_position(model.tokenizer, eval_name, prompt)
        logits = readout(model, transport, prompt, layers, pos)
        best = []
        for word in item["intermediates"]:
            ids = token_ids_of(model.tokenizer, word)
            if not ids:
                continue
            best.append(min(rank_of(logits[l], ids) for l in layers))
        if best:
            out.append([sum(r <= k for r in best) / len(best) for k in KS])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    try:
        model = from_hf(hf, tok)
    except ValueError:
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    lens = JacobianLens.load(a.lens)
    layers = list(lens.source_layers)

    transports = {"logit_lens": transport_logit(),
                  "jlens": transport_jlens(lens, model=model)}

    g = torch.Generator().manual_seed(a.seed)
    per_set, deltas_boot = {}, []
    for ev in EVAL_SETS:
        frac = {t: torch.tensor(per_item_fractions(model, T, ev, layers))
                for t, T in transports.items()}
        n = frac["logit_lens"].shape[0]
        assert frac["jlens"].shape[0] == n, "arms disagree on scored items — not paired"

        point = {t: auc_from_curve(f.mean(0).tolist()) for t, f in frac.items()}
        delta = point["jlens"] - point["logit_lens"]

        # PAIRED bootstrap: one index draw, applied to both arms.
        idx = torch.randint(0, n, (a.n_boot, n), generator=g)
        db = []
        for t in ("jlens", "logit_lens"):
            curves = frac[t][idx].mean(dim=1)                       # [n_boot, len(KS)]
            w = torch.tensor([(_LOGK[i + 1] - _LOGK[i]) for i in range(len(KS) - 1)])
            trap = ((curves[:, :-1] + curves[:, 1:]) / 2 * w).sum(1) / _SPAN
            db.append(trap)
        d = (db[0] - db[1])
        deltas_boot.append(d)

        lo, hi = torch.quantile(d, torch.tensor([0.025, 0.975])).tolist()
        floor = all(frac[t].mean(0)[i].item() == 0.0 for t in frac for i in range(4))  # k<=10
        per_set[ev] = {
            "n_items_scored": n,
            "auc_logit": round(point["logit_lens"], 4),
            "auc_jlens": round(point["jlens"], 4),
            "delta": round(delta, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "p_sign_flip": round(float((d <= 0).float().mean() if delta > 0
                                       else (d >= 0).float().mean()), 4),
            "floor_cell_both_arms_zero_through_k10": floor,
        }
        c = per_set[ev]
        print(f"  {ev:13s} d={c['delta']:+.4f}  CI95=[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]"
              f"  p_flip={c['p_sign_flip']:.3f}  {'FLOOR' if floor else ''}", flush=True)

    nwin_boot = torch.stack(deltas_boot, 1).gt(0).sum(1)             # [n_boot]
    hist = {str(v): int((nwin_boot == v).sum()) for v in range(7)}
    out = {
        "experiment": "T7 paired item-level bootstrap (audit of the n_win headline)",
        "status": "EXPLORATORY / DIRECTIONAL — secondary analysis of PREREG §0 burned cells",
        "model": a.model, "lens": a.lens, "layers": layers,
        "n_boot": a.n_boot, "seed": a.seed,
        "bootstrap": "paired over items; one index draw applied to both transports",
        "metric": "normalized pass@k AUC over log k, ks=(1,2,5,10,25,100)",
        "n_win_point": sum(1 for v in per_set.values() if v["delta"] > 0),
        "n_win_bootstrap_hist": hist,
        "P_n_win_ge_5": round(float((nwin_boot >= 5).float().mean()), 4),
        "P_n_win_le_3": round(float((nwin_boot <= 3).float().mean()), 4),
        "n_sets_ci_excludes_zero": sum(1 for v in per_set.values() if v["ci_excludes_zero"]),
        "per_set": per_set,
        "env": {"transformers": transformers.__version__, "torch": torch.__version__,
                "device": "cpu", "dtype": "float32"},
        "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"  n_win={out['n_win_point']}  P(n_win>=5)={out['P_n_win_ge_5']:.3f}  "
          f"sets with CI excluding 0: {out['n_sets_ci_excludes_zero']}/6")
    print(f"wrote {a.out} ({out['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
