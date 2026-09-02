#!/usr/bin/env python3
"""t5a_capability_screen.py — does the model know the fact at all? (PLAN.md §3.5)

THE GATE. Pythia are BASE models. A concept-swap write test needs the model to produce the
source answer in the first place: if "The capital of France is" does not yield " Paris",
there is no answer to swap, no oracle to pass, and a null write result is uninterpretable.

This screen runs BEFORE any write compute is provisioned. Its output defines the sub-ladder
on which Claim 1b is askable at all. If the floor sits at 1B, then the 70M/160M/410M story
is legitimately READ-ONLY -- and that boundary is a result, not a failure.

Reports, per model:
  - clean top-1 accuracy per relation and overall
  - median rank of the correct answer (continuous, so a model that "nearly knows" is visible)
  - the number of cells clearing a pre-declared bar, which is what n>=32 (PLAN.md §3.6) needs

Pre-declared bar (fixed here, before any model is run):
  CELL_OK   = correct answer is top-1 among the full vocabulary
  MODEL_OK  = >= 32 cells pass CELL_OK  ->  write battery is admissible at this scale
"""
from __future__ import annotations

import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jlens.hf import Layout, from_hf  # noqa: E402
from joperator import final_logits  # noqa: E402

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")

MIN_CELLS_FOR_WRITE_BATTERY = 32          # PLAN.md §3.6 power floor

RELATIONS = {
    "capital":   ("The capital of {X} is the city of", "capital"),
    "language":  ("The main language spoken in {X} is", "language"),
    "continent": ("The country {X} is located in the continent of", "continent"),
    "currency":  ("The currency used in {X} is called the", "currency"),
}

FACTS = {
    # country:      capital,        language,     continent,   currency
    "France":      (" Paris",      " French",    " Europe",   " euro"),
    "China":       (" Beijing",    " Chinese",   " Asia",     " yuan"),
    "Japan":       (" Tokyo",      " Japanese",  " Asia",     " yen"),
    "Germany":     (" Berlin",     " German",    " Europe",   " euro"),
    "Italy":       (" Rome",       " Italian",   " Europe",   " euro"),
    "Spain":       (" Madrid",     " Spanish",   " Europe",   " euro"),
    "Russia":      (" Moscow",     " Russian",   " Europe",   " ruble"),
    "Egypt":       (" Cairo",      " Arabic",    " Africa",   " pound"),
    "India":       (" Delhi",      " Hindi",     " Asia",     " rupee"),
    "Brazil":      (" Brasilia",   " Portuguese"," America",  " real"),
    "Canada":      (" Ottawa",     " English",   " America",  " dollar"),
    "Mexico":      (" Mexico",     " Spanish",   " America",  " peso"),
    "Greece":      (" Athens",     " Greek",     " Europe",   " euro"),
    "Poland":      (" Warsaw",     " Polish",    " Europe",   " zloty"),
    "Sweden":      (" Stockholm",  " Swedish",   " Europe",   " krona"),
    "Norway":      (" Oslo",       " Norwegian", " Europe",   " krone"),
    "Turkey":      (" Ankara",     " Turkish",   " Asia",     " lira"),
    "Korea":       (" Seoul",      " Korean",    " Asia",     " won"),
    "Thailand":    (" Bangkok",    " Thai",      " Asia",     " baht"),
    "Kenya":       (" Nairobi",    " Swahili",   " Africa",   " shilling"),
}
REL_ORDER = ["capital", "language", "continent", "currency"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
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

    # H7: 16 of 80 answers are multi-token and were scored on their first BPE piece, which
    # collides ' yuan'/' yen' -> ' y' AND ' krona'/' krone' -> ' k'. M8: some answers are
    # copyable from the prompt (Mexico/capital -> ' Mexico'), so cell_ok is satisfiable by
    # induction rather than recall. Both are now excluded and counted.
    rows, n_multitoken, n_copyable = [], 0, 0
    for country, answers in FACTS.items():
        for rel, ans in zip(REL_ORDER, answers):
            tmpl = RELATIONS[rel][0]
            prompt = tmpl.format(X=country)
            ids = model.encode(prompt, max_length=64)
            lg = final_logits(model, prompt, max_length=64)
            enc = tok.encode(ans, add_special_tokens=False)
            if len(enc) != 1:
                n_multitoken += 1
                continue
            if ans.strip().lower() in prompt.lower():
                n_copyable += 1
                continue
            ans_id = enc[0]
            order = lg.argsort(descending=True)
            rank = int((order == ans_id).nonzero()[0])
            top1 = int(order[0])
            rows.append({
                "country": country, "relation": rel, "prompt": prompt,
                "answer": ans, "answer_token": tok.decode([ans_id]),
                "top1": tok.decode([top1]),
                "rank_of_answer": rank,
                "logp_answer": round(float(torch.log_softmax(lg, -1)[ans_id]), 4),
                "cell_ok": rank == 0,
            })

    n_ok = sum(r["cell_ok"] for r in rows)
    by_rel = {rel: {
        "n": sum(1 for r in rows if r["relation"] == rel),
        "n_ok": sum(r["cell_ok"] for r in rows if r["relation"] == rel),
        "median_rank": int(torch.tensor(
            [r["rank_of_answer"] for r in rows if r["relation"] == rel]).median()),
    } for rel in REL_ORDER}

    out_json = {
        "experiment": "T5a capability screen (gates the write battery, PLAN.md §3.5)",
        "status": "GATE — pre-declared bar fixed in-script before any model was run",
        "model": a.model,
        "n_cells": len(rows),
        "n_excluded_multitoken_answer": n_multitoken,
        "n_excluded_answer_copyable_from_prompt": n_copyable,
        "n_distinct_answer_tokens": len({r["answer_token"] for r in rows}),
        "n_cells_ok": n_ok,
        "accuracy": round(n_ok / len(rows), 4),
        "median_rank_overall": int(torch.tensor([r["rank_of_answer"] for r in rows]).median()),
        "bar_min_cells": MIN_CELLS_FOR_WRITE_BATTERY,
        "WRITE_BATTERY_ADMISSIBLE": n_ok >= MIN_CELLS_FOR_WRITE_BATTERY,
        "by_relation": by_rel,
        "env": {"transformers": transformers.__version__, "torch": torch.__version__,
                "device": "cpu", "dtype": "float32"},
        "rows": rows,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    json.dump(out_json, open(a.out, "w"), indent=1)

    print(f"\n{a.model}")
    print(f"  cells top-1 correct : {n_ok}/{len(rows)}  ({n_ok/len(rows):.1%})")
    print(f"  median rank of answer: {out_json['median_rank_overall']}")
    for rel in REL_ORDER:
        b = by_rel[rel]
        print(f"    {rel:10s} {b['n_ok']:2d}/{b['n']:2d}  median_rank={b['median_rank']}")
    print(f"  WRITE BATTERY ADMISSIBLE (>= {MIN_CELLS_FOR_WRITE_BATTERY} cells): "
          f"{out_json['WRITE_BATTERY_ADMISSIBLE']}")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
