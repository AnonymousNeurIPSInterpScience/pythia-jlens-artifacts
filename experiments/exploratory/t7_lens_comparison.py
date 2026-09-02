#!/usr/bin/env python3
"""t7_lens_comparison.py — the merged 1+3 experiment: one metric, three transports.

Runs the anchor's own §A.6 pass@k on its own released evaluation sets, for each transport:

    logit lens   T = I        (no fitting, no corpus, no distribution dependence)
    J-lens       T = J_l      (our fitted averaged Jacobian)
    tuned lens   T = A_l      (optional; requires --tuned-lens)

The anchor reports J-lens > logit lens > tuned lens on probing, across three Claude variants,
with the margin over the logit lens "modest on multihop and association, but substantial on
multilingual, order-of-operations, poetry, and typo". Nobody has run this below ~2B.

Reference point for sanity-checking the harness (◇ external third-party reimplementation):
multihop normalized pass@k AUC ≈ 0.39 (J-lens) vs ≈ 0.24 (logit lens).

EXPLORATORY unless a committed pre-registration is cited in --prereg.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import torch

torch.set_num_threads(min(8, os.cpu_count() or 8))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jlens.hf import Layout, from_hf  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402
from anchor_evals import EVAL_SETS, pass_at_k, transport_jlens, transport_logit  # noqa: E402

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--evals", default="multihop")
    ap.add_argument("--band-lo", type=int, default=None,
                    help="restrict to layers >= this; default = all fitted layers")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--fold-norm-gain", action="store_true")
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # PREREG_PYTHIA_T7_v2 §3.10 / §7.4 — the assert that would have prevented the 2026-08-11
    # contamination. Scoring a CONFIRMATORY model without citing the prereg is refused outright.
    # This is mechanical on purpose: the discipline existed in prose and did not fire.
    CONFIRMATORY_MODELS = ("pythia-1b-deduped", "pythia-1.4b-deduped", "pythia-2.8b-deduped")
    if any(m in a.model for m in CONFIRMATORY_MODELS) and not a.prereg:
        raise SystemExit(
            f"REFUSED: {a.model} is a CONFIRMATORY model under PREREG_PYTHIA_T7_v2 §2.\n"
            f"  Scoring it without --prereg burns a registered cell.\n"
            f"  If you need a lens scored for a control or a diagnostic, use an already-burned\n"
            f"  model (70m / 160m / 410m) -- they answer method questions identically.\n"
            f"  If this IS the confirmatory run, pass --prereg docs/experiments/preregs/superseded/PREREG_PYTHIA_T7_v2.md")

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
    layers = [l for l in lens.source_layers if a.band_lo is None or l >= a.band_lo]

    transports = {
        "logit_lens": transport_logit(),
        "jlens": transport_jlens(lens, fold_norm_gain=a.fold_norm_gain, model=model),
    }

    names = [n.strip() for n in a.evals.split(",")] if a.evals != "all" else EVAL_SETS
    results: dict = {}
    for ev in names:
        results[ev] = {}
        for tname, T in transports.items():
            r = pass_at_k(model, T, ev, layers, max_items=a.max_items)
            results[ev][tname] = r
            print(f"  {ev:13s} {tname:11s} AUC={r['normalized_auc_over_logk']:.4f}  "
                  f"pass@1={r['pass_at_k']['1']:.3f} pass@10={r['pass_at_k']['10']:.3f} "
                  f"median_rank={r['median_best_rank']:.0f}  "
                  f"(unscorable multi-token intermediates: "
                  f"{r['n_intermediates_unscorable_multitoken']})", flush=True)
        # PRIMARY is the paper's §A.6 metric; the shipped-README form is kept as secondary
        # so every pre-2026-08-10 number in results/ stays comparable.
        jp = results[ev]["jlens"]["normalized_auc_over_logk_paper"]
        lp = results[ev]["logit_lens"]["normalized_auc_over_logk_paper"]
        j = results[ev]["jlens"]["normalized_auc_over_logk"]
        l = results[ev]["logit_lens"]["normalized_auc_over_logk"]
        results[ev]["jlens_minus_logit_auc_paper"] = round(jp - lp, 4)
        results[ev]["jlens_minus_logit_auc"] = round(j - l, 4)
        print(f"  {ev:13s} {'DELTA':11s} PAPER {jp - lp:+.4f}   readme {j - l:+.4f}\n",
              flush=True)

    # A prereg PATH proves nothing — the file could change after the run. Hash it, so the
    # result is bound to the exact criteria text. PREREG v1 §3.7 claimed this happened; it
    # did not. Fixed 2026-08-10.
    prereg_sha = None
    if a.prereg:
        import hashlib
        h = hashlib.sha256()
        with open(a.prereg, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        prereg_sha = h.hexdigest()
        print(f"  prereg {a.prereg}  sha256={prereg_sha[:16]}…", flush=True)

    n_win_paper = sum(1 for ev in results if results[ev]["jlens_minus_logit_auc_paper"] > 0)
    out = {
        "experiment": "T7 lens comparison — anchor §A.6 pass@k, one metric, three transports",
        "status": ("CONFIRMATORY" if a.prereg else
                   "EXPLORATORY / DIRECTIONAL — no pre-registration"),
        "prereg": a.prereg, "prereg_sha256": prereg_sha,
        "primary_metric": "normalized_auc_over_logk_paper (§A.6 per-item binary)",
        "n_win_paper": n_win_paper if len(results) == 6 else None,
        "n_win_note": ("n_win is only defined over all six sets; None means a partial run "
                       "and is NOT evaluable against the pre-registered threshold"),
        "target_layer_effective": max(lens.source_layers) + 1,
        "model": a.model, "lens": a.lens, "lens_n_prompts": lens.n_prompts,
        "layers": layers, "n_layers_used": len(layers),
        "fold_norm_gain": a.fold_norm_gain,
        "eval_data_provenance": ("anthropics/jacobian-lens @581d398, Apache-2.0, "
                                 "data/evaluations/*.json — vendored, not downloaded"),
        "metric": ("pass@k = mean over ITEMS of the fraction of that item's intermediates whose "
                   "min-over-layers rank <= k; AUC normalized over log k (anchor README)"),
        "env": {"transformers": transformers.__version__, "torch": torch.__version__,
                "device": "cpu", "dtype": "float32"},
        "results": results,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}  ({out['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
