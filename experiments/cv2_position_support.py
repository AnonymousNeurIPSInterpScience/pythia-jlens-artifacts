#!/usr/bin/env python3
"""
CV2 — POSITIONAL SUPPORT AUDIT.  Is the operator evaluated inside the positional range it was
averaged over?

WHY.  fast_fit averages the Jacobian over `valid_position_mask(seq_len, skip_first=16)`, which
excludes the first 16 positions (attention-sink behaviour) and the final position (no next-token
target).  With max_seq_len=128 that is positions 16..126 inclusive.

The readout, however, happens at whatever position the eval item's rule selects, inside a sequence
that is [BOS] ++ [128 Q tokens] ++ [item prompt] for a prefixed rung, or the item alone for Q0.
Nothing has ever checked that those positions fall inside 16..126.

If they do not, every read is an extrapolation of a position-averaged operator, and Q0 and the
prefixed rungs may extrapolate in OPPOSITE directions -- which would depress absolute pass@k and
would do so unequally across the ladder.

THIS SCRIPT MEASURES ONLY.  It does not adjudicate; there is no registered rule here.  It emits the
support table that docs/validity/CONSTRUCT_VALIDITY.md section 8 asks for, so that the mismatch can
be described as confirmed or refuted rather than suspected.

  .venv/bin/python experiments/cv2_position_support.py
"""
import json, os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(ROOT, "jacobian-lens", "data", "evaluations")
MODEL = "EleutherAI/pythia-410m-deduped"
SKIP_FIRST = 16
MAX_SEQ_LEN = 128
PREFIX_TOKENS = 128

# readout rule per set, transcribed from jacobian-lens/data/evaluations/README.md
READOUT_RULE = {
    "multihop": "last",        # token immediately preceding `target` == last token of stripped prompt
    "multilingual": "last",
    "order-ops": "last",
    "typo": "last",            # final prompt token (last fragment of the misspelling)
    "association": "last",     # final prompt token (the closing period)
    "poetry": "last_newline",  # end of line 1 of the couplet
}


def prompt_text(item):
    p = item.get("prompt")
    if isinstance(p, list):
        return " ".join(m.get("content", "") for m in p)
    return str(p)


def readout_index(tok, text, rule):
    """Position of the readout token WITHIN the item's own token sequence (0-indexed)."""
    ids = tok(text.rstrip())["input_ids"]      # rstrip: the corrected readout convention
    if rule == "last":
        return len(ids) - 1, len(ids)
    if rule == "last_newline":
        nl = [i for i, t in enumerate(ids) if "\n" in tok.decode([t])]
        return (nl[-1] if nl else len(ids) - 1), len(ids)
    raise ValueError(rule)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    lo, hi = SKIP_FIRST, MAX_SEQ_LEN - 2      # inclusive support of the fitting average

    rows = {}
    agg = {"Q0": [], "prefixed": []}
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "lens-eval-*.json"))):
        name = os.path.basename(path).replace("lens-eval-", "").replace(".json", "")
        rule = READOUT_RULE[name]
        items = json.load(open(path))["items"]
        q0, pre, lens = [], [], []
        for it in items:
            idx, n = readout_index(tok, prompt_text(it), rule)
            lens.append(n)
            # Q0: [BOS] ++ item      -> readout sits at 1 + idx
            q0.append(1 + idx)
            # prefixed: [BOS] ++ 128 Q tokens ++ item -> readout at 1 + PREFIX_TOKENS + idx
            pre.append(1 + PREFIX_TOKENS + idx)
        agg["Q0"] += q0
        agg["prefixed"] += pre
        rows[name] = {
            "n_items": len(items), "readout_rule": rule,
            "median_prompt_tokens": sorted(lens)[len(lens) // 2],
            "Q0": {"median_read_pos": sorted(q0)[len(q0) // 2],
                   "min": min(q0), "max": max(q0),
                   "n_in_support": sum(1 for p in q0 if lo <= p <= hi),
                   "frac_in_support": sum(1 for p in q0 if lo <= p <= hi) / len(q0)},
            "prefixed": {"median_read_pos": sorted(pre)[len(pre) // 2],
                         "min": min(pre), "max": max(pre),
                         "n_in_support": sum(1 for p in pre if lo <= p <= hi),
                         "frac_in_support": sum(1 for p in pre if lo <= p <= hi) / len(pre)},
        }

    out = {
        "experiment": "CV2 — positional support audit",
        "status": "MEASUREMENT ONLY — no registered decision rule; this establishes or refutes "
                  "the mismatch flagged in docs/validity/CONSTRUCT_VALIDITY.md section 4.2",
        "model": MODEL,
        "fitting_support": {
            "source": "src/fastfit.py:116 -> jlens.fitting.valid_position_mask(seq_len, skip_first)",
            "skip_first": SKIP_FIRST, "max_seq_len": MAX_SEQ_LEN,
            "excludes": "first skip_first positions (attention sink) and the final position "
                        "(no next-token target)",
            "inclusive_range": [lo, hi],
        },
        "readout_convention": "STRIPPED (corrected). Position computed on the item alone, then "
                              "shifted by the prefix length — matches t36_qladder.py:27-31",
        "by_set": rows,
        "overall": {
            arm: {"n": len(v),
                  "median_read_pos": sorted(v)[len(v) // 2],
                  "min": min(v), "max": max(v),
                  "n_in_support": sum(1 for p in v if lo <= p <= hi),
                  "frac_in_support": sum(1 for p in v if lo <= p <= hi) / len(v)}
            for arm, v in agg.items()
        },
    }
    o = out["overall"]
    out["VERDICT"] = (
        f"Q0 reads inside the fitting support on {o['Q0']['frac_in_support']*100:.1f}% of items "
        f"(median position {o['Q0']['median_read_pos']}); prefixed rungs on "
        f"{o['prefixed']['frac_in_support']*100:.1f}% (median {o['prefixed']['median_read_pos']}). "
        f"Fitting support is positions {lo}..{hi}.")

    dest = os.path.join(ROOT, "results", "cv2_position_support.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from provenance import write_result
        write_result(dest, out, script=__file__, experiment="CV2",
                     inputs=sorted(glob.glob(os.path.join(EVAL_DIR, "lens-eval-*.json"))))
    except Exception as e:                       # never swallow silently — rule 0b
        print(f"  !! provenance stamp FAILED: {e!r}", file=sys.stderr)
        json.dump(out, open(dest, "w"), indent=1)
    print(json.dumps({"fitting_support": out["fitting_support"]["inclusive_range"],
                      "overall": out["overall"], "VERDICT": out["VERDICT"]}, indent=1))
    print("\nwrote", os.path.relpath(dest, ROOT))


if __name__ == "__main__":
    main()
