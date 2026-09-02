#!/usr/bin/env python3
"""t8_probe_swap.py — the write battery, rebuilt on the anchor's own `probe-swap` set.

REPLACES t5_write_battery.py, whose result was RETRACTED 2026-08-10 for two defects
(oracle gated at ANY alpha rather than the scored alpha; control_rand_target deterministic).

DATA: `jacobian-lens/data/experiments/probe-swap.json`, Apache-2.0, 90 two-hop factual items:
    {prompt, intermediate, answer, swap_to, swap_answer, category}
The bridge entity (`intermediate`) is neither in the prompt nor the answer, which is the
construction Nanda's review of the anchor demands and which our hand-rolled country->capital
battery lacked.

PROTOCOL, from the anchor's data/experiments/README.md, verbatim:
    "Baseline: greedy next-token == `answer`. Swap: replace the `intermediate` representation
     ... with `swap_to` across the band at every prompt token position; score next-token at the
     final position == `swap_answer`."

GATES (PLAN.md §3.4/§3.5/§3.6/§3.7), all enforced per-cell AND PER-DOSE:
  - baseline: greedy next-token must equal `answer`, else the cell is inadmissible;
  - surface-form match: `intermediate`/`swap_to` and `answer`/`swap_answer` must each resolve to
    DISTINCT single tokens under a common surface form, else the swap mixes unrelated directions
    (the ' yuan'/' yen' collision that contaminated the retracted battery);
  - ORACLE PER DOSE: at each alpha, a cell counts only if the answer-token oracle flips AT THAT
    ALPHA. A cell whose oracle passes only at some other alpha is VOID at this alpha.
  - ||Delta|| logged per layer and per cell; controls matched on the MEASURED Delta.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import torch

torch.set_num_threads(min(8, os.cpu_count() or 8))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jlens.hf import Layout, from_hf  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402
from anchor_evals import load_experiment  # noqa: E402
from joperator import (final_logits, IdentityHooks, JLensSwapHooks, control_orth,  # noqa: E402
                       control_rand_target, control_seed, make_swap_basis, token_direction)

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")


def matched_single_tokens(tok, a: str, b: str):
    """Distinct single-token ids for `a`,`b` under a COMMON surface form, else None."""
    for f in (lambda w: " " + w, lambda w: " " + w.lower(), lambda w: w, lambda w: w.lower()):
        ea, eb = tok.encode(f(a), add_special_tokens=False), tok.encode(f(b), add_special_tokens=False)
        if len(ea) == 1 and len(eb) == 1 and ea[0] != eb[0]:
            return int(ea[0]), int(eb[0])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--band-lo", type=int, default=None)
    ap.add_argument("--alphas", default="1,2")
    ap.add_argument("--max-cells", type=int, default=90)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]
    prereg_sha = None
    if a.prereg:                                  # M4: CONFIRMATORY must be earned, not asserted
        if not os.path.exists(a.prereg):
            raise SystemExit(f"--prereg {a.prereg!r} does not exist; refusing CONFIRMATORY status")
        import hashlib
        prereg_sha = hashlib.sha256(open(a.prereg, "rb").read()).hexdigest()

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
    band = [l for l in lens.source_layers if a.band_lo is None or l >= a.band_lo]

    def logits(prompt, hooks=None):
        return final_logits(model, prompt, hooks)      # single final norm; see joperator

    def dose(hk):
        """H4/H5: PLAN §3.7 requires ||Delta|| per layer, per position, unconditionally — and
        the Gram condition, so a cell that silently fell back to pinv on a degenerate basis is
        visible rather than reported as an ordinary arm."""
        r = hk.record
        return {"delta_norm_band_sum": round(r.delta_norm_sum_over_band, 4),
                "delta_norm_lastpos_sum": round(r.delta_norm_lastpos_sum_over_band, 4),
                "delta_norm_per_layer_sum": {str(l): round(v, 4)
                                             for l, v in r.delta_norm_sum_positions.items()},
                "delta_norm_per_layer_max": {str(l): round(v, 4)
                                             for l, v in r.delta_norm_max.items()},
                "gram_cond_max": round(max(r.cond.values()), 3) if r.cond else None,
                "n_degenerate_layers": int(sum(r.degenerate.values()))}

    def summarize(lg, ans_id, swap_id, clean_lp=None):
        lp = torch.log_softmax(lg, -1)
        order = lg.argsort(descending=True)
        d = {"top1": tok.decode([int(order[0])]),
             "rank_swap_answer": int((order == swap_id).nonzero()[0]),
             "rank_answer": int((order == ans_id).nonzero()[0]),
             "logp_swap_answer": round(float(lp[swap_id]), 4),
             "logp_answer": round(float(lp[ans_id]), 4),
             "flip_to_swap_answer": bool(int(order[0]) == swap_id)}
        if clean_lp is not None:
            d["kl_vs_clean"] = round(float(torch.nn.functional.kl_div(
                clean_lp, lp, log_target=True, reduction="sum")), 4)   # KL(intervened||clean)
        return d

    items = load_experiment("probe-swap")["items"][:a.max_cells]
    rows, n_baseline_fail, n_tok_fail = [], 0, 0
    for it in items:
        prompt = it["prompt"].rstrip()
        pair_c = matched_single_tokens(tok, it["intermediate"], it["swap_to"])
        pair_a = matched_single_tokens(tok, it["answer"], it["swap_answer"])
        if pair_c is None or pair_a is None:
            n_tok_fail += 1
            continue
        c_s, c_t = pair_c
        ans_id, swap_id = pair_a
        clean = logits(prompt)
        if int(clean.argmax()) != ans_id:            # baseline gate (anchor protocol)
            n_baseline_fail += 1
            continue
        clean_lp = torch.log_softmax(clean, -1)

        vs = {l: token_direction(lens.jacobians[l].float(), WU[c_s]) for l in band}
        vt = {l: token_direction(lens.jacobians[l].float(), WU[c_t]) for l in band}
        va = {l: token_direction(lens.jacobians[l].float(), WU[ans_id]) for l in band}
        vb = {l: token_direction(lens.jacobians[l].float(), WU[swap_id]) for l in band}

        row = {"name": it["name"], "category": it.get("category"), "prompt": prompt,
               "intermediate": it["intermediate"], "swap_to": it["swap_to"],
               "answer": it["answer"], "swap_answer": it["swap_answer"],
               "clean": summarize(clean, ans_id, swap_id), "arms": {}, "admissible": {}}
        # L1: do NOT enter the context here — final_logits() enters it. Entering at both
        # levels registered two hooks per layer; with a swap that would apply the write TWICE.
        row["arms"]["identity"] = summarize(
            logits(prompt, IdentityHooks(blocks, band)), ans_id, swap_id, clean_lp)

        for al in alphas:
            bases = {l: make_swap_basis(vs[l], vt[l]) for l in band}
            hk = JLensSwapHooks(blocks, bases, alpha=al, readback=(len(rows) < 3),
                                skip_positions=1)
            r = summarize(logits(prompt, hk), ans_id, swap_id, clean_lp)
            r.update(dose(hk))
            if hk.readback:                       # M5: mechanically verify the exchange
                r["exchange_verified"] = hk.exchange_verified(atol=1e-2)
            row["arms"][f"swap@a{al:g}"] = r
            match = {l: v.clone() for l, v in hk.per_position.items()}

            for nm, ctrl in (("rand_target", control_rand_target), ("orth", control_orth)):
                # H1: independent seed per (run, item, layer, arm) — the two controls used to
                # share a draw (cos 0.998) and were identical across every item (cos 1.000000).
                cb = {l: make_swap_basis(
                          vs[l], ctrl(vs[l], vt[l], control_seed(a.seed, it["name"], l, nm)))
                      for l in band}
                ch = JLensSwapHooks(blocks, cb, alpha=al, readback=False,
                                    match_delta_norm=match, skip_positions=1)
                cr = summarize(logits(prompt, ch), ans_id, swap_id, clean_lp)
                cr.update(dose(ch))
                row["arms"][f"{nm}@a{al:g}"] = cr

            # H2: the gate must be evaluated at the dose it gates. An unmatched oracle ran at
            # its own dose (measured 0.47x the swap arm's in one cell), so a cell could be
            # voided because the ORACLE was under-dosed, not because the model is unwritable.
            ob = {l: make_swap_basis(va[l], vb[l]) for l in band}
            oh = JLensSwapHooks(blocks, ob, alpha=al, readback=False,
                                match_delta_norm=match, skip_positions=1)
            orr = summarize(logits(prompt, oh), ans_id, swap_id, clean_lp)
            orr.update(dose(oh))
            oh_native = JLensSwapHooks(blocks, ob, alpha=al, readback=False, skip_positions=1)
            _ = logits(prompt, oh_native)
            orr["native_dose_if_unmatched"] = round(
                oh_native.record.delta_norm_sum_over_band, 4)
            row["arms"][f"oracle@a{al:g}"] = orr
            # PER-DOSE gate: this cell counts at THIS alpha only if the oracle flips at THIS alpha
            row["admissible"][f"a{al:g}"] = bool(orr["flip_to_swap_answer"])

        rows.append(row)
        a0 = f"{alphas[0]:g}"
        print(f"  {it['name'][:26]:26s} {it['intermediate'][:9]:9s}->{it['swap_to'][:9]:9s} "
              f"rank {row['clean']['rank_swap_answer']:5d}->"
              f"{row['arms'][f'swap@a{a0}']['rank_swap_answer']:5d}"
              f"  oracle@{a0}={'OK' if row['admissible'][f'a{a0}'] else 'void'}", flush=True)

    def agg(arm, al):
        v = [r["arms"][arm] for r in rows if r["admissible"][f"a{al:g}"]]
        if not v:
            return {"n": 0}
        med = lambda k: float(torch.tensor([x[k] for x in v]).float().median())
        return {"n": len(v), "flips": sum(x["flip_to_swap_answer"] for x in v),
                "median_rank_swap_answer": med("rank_swap_answer"),
                "median_kl_vs_clean": round(med("kl_vs_clean"), 4),
                "median_delta_norm": round(med("delta_norm_band_sum"), 3)
                if "delta_norm_band_sum" in v[0] else None}

    summary = {}
    for al in alphas:
        adm = [r for r in rows if r["admissible"][f"a{al:g}"]]
        summary[f"alpha={al:g}"] = {
            "n_admissible_this_dose": len(adm),
            "median_clean_rank": float(torch.tensor(
                [r["clean"]["rank_swap_answer"] for r in adm]).float().median()) if adm else None,
            **{arm: agg(arm, al) for arm in
               ("identity", f"swap@a{al:g}", f"rand_target@a{al:g}", f"orth@a{al:g}",
                f"oracle@a{al:g}")},
        }

    out = {
        "experiment": "T8 write battery on the anchor's probe-swap set",
        "status": "CONFIRMATORY" if a.prereg else "EXPLORATORY / DIRECTIONAL — no pre-registration",
        "prereg": a.prereg, "prereg_sha256": prereg_sha,
        "supersedes": "results/t5_write_410m.json (RETRACTED 2026-08-10)",
        "data_provenance": "anthropics/jacobian-lens @581d398, Apache-2.0, data/experiments/probe-swap.json",
        "seed": a.seed, "max_cells": a.max_cells, "band_lo": a.band_lo,
        "control_seed_policy": "sha256(seed|item|layer|arm) — independent per arm and per item",
        "skip_positions": 1,
        "skip_positions_note": "BOS/attention-sink position excluded; its residual norm is 7-9x "
                               "every real token, so an all-positions write is dominated by a "
                               "token the anchor's protocol does not include",
        "argv": sys.argv,
        "model": a.model, "lens": a.lens, "lens_n_prompts": lens.n_prompts,
        "band": band, "alphas": alphas, "positions": "all", "centered": False,
        "n_items_total": len(items), "n_dropped_tokenization": n_tok_fail,
        "n_dropped_baseline": n_baseline_fail, "n_cells_run": len(rows),
        "gate": "PER-DOSE: a cell counts at alpha only if the answer-token oracle flips at that alpha",
        "env": {"transformers": transformers.__version__, "torch": torch.__version__,
                "device": "cpu", "dtype": "float32"},
        "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "summary": summary, "rows": rows,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    json.dump(out, open(a.out, "w"), indent=1)

    print(f"\n=== {a.model} band L{band[0]}-{band[-1]} ===")
    print(f"{len(items)} items -> dropped {n_tok_fail} tokenization, {n_baseline_fail} baseline; "
          f"{len(rows)} run")
    for al in alphas:
        s = summary[f"alpha={al:g}"]
        print(f"\n  alpha={al:g}  admissible (oracle flips at THIS dose): {s['n_admissible_this_dose']}")
        if s["n_admissible_this_dose"]:
            print(f"    {'clean':16s} median rank {s['median_clean_rank']:.0f}")
            for arm in ("identity", f"swap@a{al:g}", f"rand_target@a{al:g}", f"orth@a{al:g}"):
                x = s[arm]
                print(f"    {arm:16s} median rank {x['median_rank_swap_answer']:7.1f}  "
                      f"flips {x['flips']}/{x['n']}  ||D|| {x.get('median_delta_norm')}")
    print(f"\nwrote {a.out}  ({out['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
