#!/usr/bin/env python3
"""t5_write_battery.py — the anchor §2.5 write on Pythia. EXPLORATORY / DIRECTIONAL.

First writes ever run on Pythia in this program. Built to PLAN.md §3's rules:

  §3.4  every cell carries an ORACLE (swap the ANSWER tokens instead of the concept tokens).
        A cell whose oracle fails is VOID, not a null -- it cannot distinguish "the lens
        direction is wrong" from "this model is not writable here".
  §3.5  only cells whose clean answer is already top-1 are admissible (capability screen).
  §3.6  continuous DVs (delta logp, rank, KL) -- flips are reported but never alone.
  §3.7  ||Delta|| logged per layer and per cell, ALWAYS. Controls are matched on the
        MEASURED Delta, not on ||v_t - v_s||.
  no mu. Ever.

ARMS per cell:
  clean            no intervention
  identity         hook plumbing, no edit (null)
  swap@a           anchor §2.5 coordinate swap, source concept -> target concept
  rand_target@a    v_t re-pointed at a random direction, dose matched to swap@a
  orth@a           v_t re-pointed orthogonal to the real payload, dose matched to swap@a
  oracle@a         swap the ANSWER tokens (Paris->Beijing) instead of the concepts
"""
from __future__ import annotations

import argparse, json, os, sys, time

import torch

torch.set_num_threads(min(8, os.cpu_count() or 8))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jlens.hf import Layout, from_hf  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402
from joperator import (IdentityHooks, JLensSwapHooks, control_orth,  # noqa: E402
                       control_rand_target, make_swap_basis, token_direction)
from t5a_capability_screen import FACTS, REL_ORDER, RELATIONS  # noqa: E402

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
    ap.add_argument("--lens", required=True)
    ap.add_argument("--band-lo", type=int, default=9)
    ap.add_argument("--alphas", default="1,2,4")
    ap.add_argument("--max-cells", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    alphas = [float(x) for x in a.alphas.split(",")]
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
    WU = model._lm_head.weight.detach().float()
    blocks = model.model.layers if hasattr(model, "model") else model.layers
    band = [l for l in lens.source_layers if l >= a.band_lo]
    tid = lambda s: tok.encode(s, add_special_tokens=False)[0]

    def logits(prompt, hooks=None):
        ids = model.encode(prompt, max_length=64)
        with torch.no_grad():
            if hooks is None:
                out = model.forward(ids)
                hs = out if torch.is_tensor(out) else out[0]
                return model.unembed(hs[0, -1].float()).float()
            with hooks:
                out = model.forward(ids)
                hs = out if torch.is_tensor(out) else out[0]
                return model.unembed(hs[0, -1].float()).float()

    def summarize(lg, src_id, tgt_id, clean_lp=None):
        lp = torch.log_softmax(lg, -1)
        order = lg.argsort(descending=True)
        d = {"top1": tok.decode([int(order[0])]),
             "rank_tgt": int((order == tgt_id).nonzero()[0]),
             "rank_src": int((order == src_id).nonzero()[0]),
             "logp_tgt": round(float(lp[tgt_id]), 4),
             "logp_src": round(float(lp[src_id]), 4),
             "flip_to_tgt": bool(int(order[0]) == tgt_id)}
        if clean_lp is not None:
            d["kl_vs_clean"] = round(float(torch.nn.functional.kl_div(
                lp, clean_lp, log_target=True, reduction="sum")), 4)
        return d

    # ---- admissible cells: BOTH source and target answers must be clean top-1 ----
    countries = list(FACTS)
    cells = []
    for ri, rel in enumerate(REL_ORDER):
        tmpl = RELATIONS[rel][0]
        for i, src in enumerate(countries):
            tgt = countries[(i + 1) % len(countries)]
            if FACTS[src][ri] == FACTS[tgt][ri]:
                continue
            cells.append((rel, tmpl, src, tgt, FACTS[src][ri], FACTS[tgt][ri]))
    rows, n_void = [], 0
    for (rel, tmpl, src, tgt, ans_s, ans_t) in cells:
        if len(rows) >= a.max_cells:
            break
        prompt = tmpl.format(X=src)
        s_id, t_id = tid(ans_s), tid(ans_t)
        clean = logits(prompt)
        if int(clean.argmax()) != s_id:      # §3.5 capability gate
            continue
        clean_lp = torch.log_softmax(clean, -1)

        vs = {l: token_direction(lens.jacobians[l].float(), WU[tid(" " + src)]) for l in band}
        vt = {l: token_direction(lens.jacobians[l].float(), WU[tid(" " + tgt)]) for l in band}
        va = {l: token_direction(lens.jacobians[l].float(), WU[s_id]) for l in band}
        vb = {l: token_direction(lens.jacobians[l].float(), WU[t_id]) for l in band}

        row = {"relation": rel, "src": src, "tgt": tgt, "prompt": prompt,
               "answer_src": ans_s, "answer_tgt": ans_t, "band": band,
               "clean": summarize(clean, s_id, t_id), "arms": {}}

        row["arms"]["identity"] = summarize(
            logits(prompt, IdentityHooks(blocks, band)), s_id, t_id, clean_lp)

        for al in alphas:
            # --- real swap; capture per-position ||Delta|| for dose matching ---
            bases = {l: make_swap_basis(vs[l], vt[l]) for l in band}
            hk = JLensSwapHooks(blocks, bases, alpha=al, readback=False)
            r = summarize(logits(prompt, hk), s_id, t_id, clean_lp)
            r["delta_norm_band_sum"] = round(hk.record.delta_norm_sum_over_band, 4)
            r["delta_norm_per_layer"] = {l: round(v, 4)
                                         for l, v in hk.record.delta_norm_last.items()}
            r["gram_cond_max"] = round(max(hk.record.cond.values()), 2)
            row["arms"][f"swap@a{al:g}"] = r
            match = {l: v.clone() for l, v in hk.per_position.items()}

            # --- matched-dose controls (PLAN.md §3.7) ---
            for name, ctrl in (("rand_target", control_rand_target), ("orth", control_orth)):
                cb = {l: make_swap_basis(vs[l], ctrl(vs[l], vt[l], a.seed + l)) for l in band}
                ch = JLensSwapHooks(blocks, cb, alpha=al, readback=False,
                                    match_delta_norm=match)
                cr = summarize(logits(prompt, ch), s_id, t_id, clean_lp)
                cr["delta_norm_band_sum"] = round(ch.record.delta_norm_sum_over_band, 4)
                cr["dose_matched_to"] = f"swap@a{al:g}"
                row["arms"][f"{name}@a{al:g}"] = cr

            # --- oracle: swap the ANSWER tokens (§3.4) ---
            ob = {l: make_swap_basis(va[l], vb[l]) for l in band}
            oh = JLensSwapHooks(blocks, ob, alpha=al, readback=False)
            orr = summarize(logits(prompt, oh), s_id, t_id, clean_lp)
            orr["delta_norm_band_sum"] = round(oh.record.delta_norm_sum_over_band, 4)
            row["arms"][f"oracle@a{al:g}"] = orr

        row["oracle_any_flip"] = any(row["arms"][f"oracle@a{al:g}"]["flip_to_tgt"]
                                     for al in alphas)
        row["VOID_no_oracle"] = not row["oracle_any_flip"]
        n_void += int(row["VOID_no_oracle"])
        rows.append(row)
        print(f"  {rel:9s} {src:8s}->{tgt:8s} clean={row['clean']['top1']!r:12s} "
              f"swap@1 rank {row['clean']['rank_tgt']:5d}->{row['arms']['swap@a1']['rank_tgt']:5d} "
              f"oracle={'OK' if row['oracle_any_flip'] else 'VOID'}", flush=True)

    def agg(arm):
        v = [r["arms"][arm] for r in rows if not r["VOID_no_oracle"]]
        if not v:
            return None
        med = lambda k: float(torch.tensor([x[k] for x in v]).float().median())
        return {"n": len(v), "flips": sum(x["flip_to_tgt"] for x in v),
                "median_rank_tgt": med("rank_tgt"),
                "median_dlogp_tgt": round(med("logp_tgt"), 4),
                "median_kl_vs_clean": round(med("kl_vs_clean"), 4),
                "median_delta_norm": round(med("delta_norm_band_sum"), 3)
                if "delta_norm_band_sum" in v[0] else None}

    valid = [r for r in rows if not r["VOID_no_oracle"]]
    out = {
        "experiment": "T5 write battery — anchor §2.5 swap on Pythia",
        "status": "EXPLORATORY / DIRECTIONAL — no pre-registration; not a confirmatory result",
        "prereg": None,
        "model": a.model, "lens": a.lens, "lens_n_prompts": lens.n_prompts,
        "band": band, "band_lo": a.band_lo, "alphas": alphas,
        "positions": "all", "clamp": "band, every forward", "centered": False,
        "n_cells_run": len(rows), "n_cells_void_no_oracle": n_void,
        "n_cells_valid": len(valid),
        "median_clean_rank_tgt": float(torch.tensor(
            [r["clean"]["rank_tgt"] for r in valid]).float().median()) if valid else None,
        "summary": {arm: agg(arm) for arm in
                    (["identity"] + [f"{p}@a{al:g}" for al in alphas
                                     for p in ("swap", "rand_target", "orth", "oracle")])},
        "env": {"transformers": transformers.__version__, "torch": torch.__version__,
                "device": "cpu", "dtype": "float32"},
        "rows": rows,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    json.dump(out, open(a.out, "w"), indent=1)

    print(f"\n=== {a.model}  band L{band[0]}-{band[-1]}  cells={len(rows)} "
          f"void={n_void} valid={len(valid)} ===")
    print(f"{'arm':16s} {'n':>3} {'flips':>6} {'med_rank_tgt':>13} {'med_logp_tgt':>13} "
          f"{'med_KL':>8} {'med_||D||':>10}")
    if valid:
        print(f"{'clean':16s} {len(valid):3d} {'-':>6} "
              f"{out['median_clean_rank_tgt']:13.1f}")
    for arm, s in out["summary"].items():
        if s:
            print(f"{arm:16s} {s['n']:3d} {s['flips']:6d} {s['median_rank_tgt']:13.1f} "
                  f"{s['median_dlogp_tgt']:13.4f} {s['median_kl_vs_clean']:8.3f} "
                  f"{str(s['median_delta_norm']):>10}")
    print(f"\nwrote {a.out}  ({out['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
