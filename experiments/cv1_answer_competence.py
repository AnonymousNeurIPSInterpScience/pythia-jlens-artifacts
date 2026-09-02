#!/usr/bin/env python3
"""
CV1 — ANSWER COMPETENCE.  Can the model do the task at all?

WHY.  Every number in this programme scores whether an ANNOTATED INTERMEDIATE appears in a
transported readout.  Nothing has ever measured whether the model produces the ANSWER.  The
existing `capability_*` fields measure the rank of the intermediate, not the target, and
experiments/t36_qladder.py:483-489 records that the registered competence gate is degenerate and
"cannot fire" in either direction.

Without this, "the J-lens is degenerate at 70M/160M" is confounded with "the task is impossible at
70M/160M".  That is the S1 claim, and it is the programme's lead claim.

WHAT IS MEASURABLE.  Only three of the six sets carry a `target` (the expected answer): multihop,
order-ops, multilingual.  poetry / typo / association have `target: null` -- for those the
annotated intermediate IS the object of interest and there is no separate answer, so answer
competence is UNDEFINED, not zero.  Reported as such rather than silently dropped.

MEASUREMENT ONLY.  No registered decision rule.  This produces the number that the S1 claim needs
in order to be interpretable; it does not adjudicate S1.

  .venv/bin/python experiments/cv1_answer_competence.py --models 70m,160m,410m,1b
"""
import argparse, json, os, glob, statistics as st

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(ROOT, "jacobian-lens", "data", "evaluations")
MODELS = {"70m": "EleutherAI/pythia-70m-deduped", "160m": "EleutherAI/pythia-160m-deduped",
          "410m": "EleutherAI/pythia-410m-deduped", "1b": "EleutherAI/pythia-1b-deduped",
          "1.4b": "EleutherAI/pythia-1.4b-deduped", "2.8b": "EleutherAI/pythia-2.8b-deduped"}
WITH_TARGET = ["multihop", "order-ops", "multilingual"]


def prompt_text(item):
    p = item.get("prompt")
    if isinstance(p, list):
        return " ".join(m.get("content", "") for m in p)
    return str(p)


def load(name):
    return json.load(open(os.path.join(EVAL_DIR, f"lens-eval-{name}.json")))["items"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="70m,160m,410m")
    ap.add_argument("--limit", type=int, default=0, help="items per set, 0 = all")
    a = ap.parse_args()

    out = {
        "experiment": "CV1 — answer competence: can the model do the task at all?",
        "status": "MEASUREMENT ONLY — no registered decision rule",
        "why": ("existing capability_* fields measure the INTERMEDIATE's rank, not the ANSWER's; "
                "t36_qladder.py:483-489 records the registered competence gate as degenerate and "
                "unable to fire. This supplies the missing number."),
        "readout_convention": "STRIPPED (corrected) — rank of the target's first token at the "
                              "final position of the rstripped prompt",
        "sets_with_target": WITH_TARGET,
        "sets_without_target": ["poetry", "typo", "association"],
        "note_on_undefined": ("poetry/typo/association carry target: null. The annotated "
                              "intermediate is the object of interest and there is no separate "
                              "answer, so answer competence is UNDEFINED for them — not zero."),
        "by_model": {},
    }

    for short in [m.strip() for m in a.models.split(",") if m.strip()]:
        mid = MODELS[short]
        print(f"\n=== {short} ({mid}) ===", flush=True)
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
        rec = {}
        for name in WITH_TARGET:
            items = load(name)
            if a.limit:
                items = items[:a.limit]
            ranks, probs, top1, top10 = [], [], 0, 0
            n_multi = 0
            with torch.no_grad():
                for it in items:
                    tgt = it.get("target")
                    if tgt is None:
                        continue
                    ids = tok(prompt_text(it).rstrip(), return_tensors="pt")["input_ids"]
                    tids = tok(" " + str(tgt).strip())["input_ids"]
                    if len(tids) > 1:
                        n_multi += 1          # score the FIRST token of a multi-token answer
                    t0 = tids[0]
                    lg = model(ids).logits[0, -1].float()
                    r = int((lg > lg[t0]).sum().item()) + 1
                    ranks.append(r)
                    probs.append(float(torch.softmax(lg, -1)[t0]))
                    top1 += (r == 1)
                    top10 += (r <= 10)
            n = len(ranks)
            rec[name] = {
                "n_scored": n,
                "n_multitoken_answers_scored_on_first_token": n_multi,
                "top1_rate": top1 / n, "top10_rate": top10 / n,
                "median_rank": sorted(ranks)[n // 2],
                "mean_rank": st.mean(ranks),
                "median_prob": sorted(probs)[n // 2],
            }
            print(f"  {name:14} n={n:4}  top1={top1/n:6.3f}  top10={top10/n:6.3f}  "
                  f"median_rank={sorted(ranks)[n//2]:6}", flush=True)
        allr = [v["top1_rate"] * v["n_scored"] for v in rec.values()]
        alln = sum(v["n_scored"] for v in rec.values())
        rec["_pooled"] = {"n": alln, "top1_rate": sum(allr) / alln,
                          "top10_rate": sum(v["top10_rate"] * v["n_scored"]
                                            for v in rec.values() if "n_scored" in v) / alln}
        out["by_model"][short] = rec
        del model

    lines = []
    for m, r in out["by_model"].items():
        p = r["_pooled"]
        lines.append(f"{m}: top1 {p['top1_rate']*100:.1f}%, top10 {p['top10_rate']*100:.1f}% "
                     f"(n={p['n']} items with a target)")
    out["VERDICT"] = ("Answer competence on the three sets carrying a target. " + " | ".join(lines))

    dest = os.path.join(ROOT, "results", os.environ.get("CV1_OUT", "cv1_answer_competence.json"))
    try:
        import sys; sys.path.insert(0, os.path.join(ROOT, "src"))
        from provenance import write_result
        write_result(dest, out, script=__file__, experiment="CV1",
                     inputs=[os.path.join(EVAL_DIR, f"lens-eval-{n}.json") for n in WITH_TARGET])
    except Exception as e:                       # never swallow silently — rule 0b
        print(f"  !! provenance stamp FAILED: {e!r}", file=sys.stderr)
        json.dump(out, open(dest, "w"), indent=1)
    print("\n" + out["VERDICT"])
    print("wrote", os.path.relpath(dest, ROOT))


if __name__ == "__main__":
    main()
