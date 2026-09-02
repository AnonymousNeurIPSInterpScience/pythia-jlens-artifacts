#!/usr/bin/env python3
"""
CV4 PHASE 1 — capability calibration across the Pythia ladder, ALL SIX families.

Supersedes the ad-hoc reuse of cv1_answer_competence.py for this purpose, which covered only the
three families carrying a `target` (255 of 551 items) and reported bare rates with no interval.
The CV4 pre-registration asks for all six families; this delivers that.

TWO MEASURES, because no single one is defined across the battery:

  ANSWER RANK      defined only where `target` is non-null: multihop, order-ops, multilingual
                   (255 items). Rank of the expected answer's first token at the readout position.
                   This is the competence measure that matters for interpreting the S1 scale claim.

  PROMPT SURPRISAL defined for ALL 551 items in ALL six families. Mean per-token negative
                   log-likelihood of the prompt itself under the model. It does not measure task
                   competence; it measures whether the model finds the text natural at all, which
                   separates "cannot do the task" from "cannot even parse the prompt". For
                   poetry / typo / association the annotated intermediate IS the object of
                   interest and there is no answer to produce, so answer competence is UNDEFINED
                   for them -- not zero -- and surprisal is what is available.

WILSON INTERVALS on every rate. At n=55 (order-ops) a 5.5% top-1 rate is three items and its 95%
interval runs to ~15%; reading a ladder off point estimates at these n is not supportable. The
interval is reported so nobody does.

MEASUREMENT ONLY. No registered decision rule -- CV4's rule is adjudicated in Phase 3 on the corpus
contrast, not here. Phase 1 gates which model M* the later phases target.

  .venv/bin/python experiments/cv4_phase1_capability.py --models 70m,160m,410m,1b,1.4b,2.8b
"""
import argparse, json, math, os, sys, statistics as st

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EVAL_DIR = os.path.join(ROOT, "jacobian-lens", "data", "evaluations")

MODELS = {"70m": "EleutherAI/pythia-70m-deduped", "160m": "EleutherAI/pythia-160m-deduped",
          "410m": "EleutherAI/pythia-410m-deduped", "1b": "EleutherAI/pythia-1b-deduped",
          "1.4b": "EleutherAI/pythia-1.4b-deduped", "2.8b": "EleutherAI/pythia-2.8b-deduped"}
FAMILIES = ["multihop", "multilingual", "order-ops", "poetry", "typo", "association"]
WITH_TARGET = {"multihop", "multilingual", "order-ops"}


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion. Correct at small n, unlike normal."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def prompt_text(item):
    p = item.get("prompt")
    if isinstance(p, list):
        return " ".join(m.get("content", "") for m in p)
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="70m,160m,410m,1b,1.4b,2.8b")
    a = ap.parse_args()

    items = {f: json.load(open(os.path.join(EVAL_DIR, f"lens-eval-{f}.json")))["items"]
             for f in FAMILIES}
    n_total = sum(len(v) for v in items.values())

    out = {
        "experiment": "CV4 Phase 1 — capability calibration across the ladder, all six families",
        "prereg": "docs/experiments/preregs/CV4_capability_ladder.md",
        "status": "MEASUREMENT ONLY — Phase 1 gates which model later phases target; "
                  "CV4's registered rule is adjudicated in Phase 3, not here",
        "readout_convention": "STRIPPED (corrected)",
        "n_items_total": n_total,
        "n_items_with_target": sum(len(items[f]) for f in WITH_TARGET),
        "measures": {
            "answer_rank": "rank of the expected answer's first token; defined only where "
                           "`target` is non-null (multihop, order-ops, multilingual)",
            "prompt_surprisal": "mean per-token NLL of the prompt; defined for all six families. "
                                "NOT a task-competence measure — it separates 'cannot do the task' "
                                "from 'cannot parse the prompt'",
        },
        "note_on_undefined": ("poetry / typo / association carry target: null. The annotated "
                              "intermediate is the object of interest and there is no separate "
                              "answer, so ANSWER COMPETENCE IS UNDEFINED for them, not zero."),
        "interval": "95% Wilson score interval on every rate",
        "by_model": {},
    }

    for short in [m.strip() for m in a.models.split(",") if m.strip()]:
        mid = MODELS[short]
        print(f"\n=== {short} ({mid}) ===", flush=True)
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
        rec, pooled_k, pooled_n, pooled_k10 = {}, 0, 0, 0
        with torch.no_grad():
            for f in FAMILIES:
                ranks, sur, top1, top10 = [], [], 0, 0
                for it in items[f]:
                    ids = tok(prompt_text(it).rstrip(), return_tensors="pt")["input_ids"]
                    lg = model(ids).logits[0].float()
                    # prompt surprisal: mean NLL of tokens 1..T-1 given the prefix
                    if ids.shape[1] > 1:
                        lp = torch.log_softmax(lg[:-1], -1)
                        sur.append(float(-lp[torch.arange(ids.shape[1] - 1),
                                             ids[0, 1:]].mean()))
                    if f in WITH_TARGET and it.get("target") is not None:
                        t0 = tok(" " + str(it["target"]).strip())["input_ids"][0]
                        last = lg[-1]
                        r = int((last > last[t0]).sum().item()) + 1
                        ranks.append(r); top1 += (r == 1); top10 += (r <= 10)
                e = {"n_items": len(items[f]),
                     "mean_prompt_surprisal": st.mean(sur) if sur else None,
                     "median_prompt_surprisal": sorted(sur)[len(sur) // 2] if sur else None}
                if ranks:
                    n = len(ranks)
                    lo1, hi1 = wilson(top1, n); lo10, hi10 = wilson(top10, n)
                    e.update({"n_scored_for_answer": n,
                              "top1_count": top1, "top1_rate": top1 / n,
                              "top1_wilson95": [lo1, hi1],
                              "top10_count": top10, "top10_rate": top10 / n,
                              "top10_wilson95": [lo10, hi10],
                              "median_rank": sorted(ranks)[n // 2]})
                    pooled_k += top1; pooled_k10 += top10; pooled_n += n
                else:
                    e["answer_competence"] = "UNDEFINED — target is null for this family"
                rec[f] = e
                if ranks:
                    print(f"  {f:14} n={len(ranks):4} top1={top1/n:6.3f} "
                          f"[{lo1:.3f},{hi1:.3f}]  top10={top10/n:6.3f}  "
                          f"surprisal={e['mean_prompt_surprisal']:.3f}", flush=True)
                else:
                    print(f"  {f:14} n={len(items[f]):4} answer=UNDEFINED            "
                          f"           surprisal={e['mean_prompt_surprisal']:.3f}", flush=True)
        lo1, hi1 = wilson(pooled_k, pooled_n); lo10, hi10 = wilson(pooled_k10, pooled_n)
        rec["_pooled_answer"] = {"n": pooled_n, "top1_count": pooled_k,
                                 "top1_rate": pooled_k / pooled_n, "top1_wilson95": [lo1, hi1],
                                 "top10_rate": pooled_k10 / pooled_n, "top10_wilson95": [lo10, hi10]}
        rec["_pooled_surprisal_all_six"] = st.mean(
            [rec[f]["mean_prompt_surprisal"] for f in FAMILIES])
        print(f"  POOLED top1={pooled_k/pooled_n:.3f} [{lo1:.3f},{hi1:.3f}] "
              f"(n={pooled_n})  surprisal(all 6)={rec['_pooled_surprisal_all_six']:.3f}", flush=True)
        out["by_model"][short] = rec
        del model

    lines = [f"{m}: top1 {r['_pooled_answer']['top1_rate']*100:.1f}% "
             f"[{r['_pooled_answer']['top1_wilson95'][0]*100:.1f},"
             f"{r['_pooled_answer']['top1_wilson95'][1]*100:.1f}]"
             for m, r in out["by_model"].items()]
    out["VERDICT"] = ("Answer competence, 255 items with a target, 95% Wilson: " + " | ".join(lines)
                      + ". Prompt surprisal covers all 551 items in all six families.")

    dest = os.path.join(ROOT, "results", "cv4_phase1_capability.json")
    try:
        from provenance import write_result
        write_result(dest, out, script=__file__, experiment="CV4P1",
                     inputs=[os.path.join(EVAL_DIR, f"lens-eval-{f}.json") for f in FAMILIES])
    except Exception as e:                       # never swallow silently — rule 0b
        print(f"  !! provenance stamp FAILED: {e!r}", file=sys.stderr)
        json.dump(out, open(dest, "w"), indent=1)
    print("\n" + out["VERDICT"])
    print("wrote", os.path.relpath(dest, ROOT))


if __name__ == "__main__":
    main()
