#!/usr/bin/env python3
"""pin_fit_corpus.py — write a hash manifest of the EXACT prompts a fit consumed.

WHY. `t2_fastfit.load_prompts` selects its fitting corpus by streaming a HuggingFace dataset
and taking the first `n` rows passing a length/heading filter. No dataset revision is pinned and
the selected prompts are not recorded anywhere, so `lens_*_provenance.json` cannot tell you which
200 documents produced the operator. That is RIGOR_SKILL axis 2 (reproducibility from the
artifact alone) scored 0: if Salesforce re-exports wikitext, or the streaming shard order changes,
the fit corpus silently changes and no stored value would reveal it.

This writes the per-prompt SHA-256s, the concatenation hash, and the resolved dataset revision,
so a future refit can be proved byte-identical (or proved not to be).

Run once per (corpus, n) pair used by any citable lens.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

_SEP = "\n"   # explicit joiner so the concat hash is defined by this file, not by whitespace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="wikitext")
    ap.add_argument("-n", "--n-prompts", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    import datasets
    from t2_fastfit import load_prompts

    prompts = load_prompts(a.n_prompts, min_chars=a.min_chars, corpus=a.corpus)

    revision = None
    if a.corpus == "wikitext":
        try:
            from huggingface_hub import HfApi
            revision = HfApi().dataset_info("Salesforce/wikitext").sha
        except Exception as exc:                                    # offline is not fatal
            revision = f"UNRESOLVED: {type(exc).__name__}"

    lens_ = sorted(len(p) for p in prompts)
    man = {
        "purpose": "byte-level pin of the fitting corpus (RIGOR_SKILL axis 2)",
        "corpus_arg": a.corpus,
        "loader": "t2_fastfit.py:load_prompts",
        "selection_rule": f"first {a.n_prompts} streamed rows with "
                          f"len(text.strip()) >= {a.min_chars} and not startswith('=')",
        "hf_dataset": ("Salesforce/wikitext:wikitext-103-raw-v1:train"
                       if a.corpus == "wikitext" else a.corpus),
        "hf_dataset_revision": revision,
        "datasets_version": datasets.__version__,
        "n_prompts": len(prompts),
        "sha256_concat": hashlib.sha256(_SEP.join(prompts).encode()).hexdigest(),
        "sha256_per_prompt_first12": [hashlib.sha256(p.encode()).hexdigest()[:12]
                                      for p in prompts],
        "chars_min_median_max": [lens_[0], lens_[len(lens_) // 2], lens_[-1]],
        "first_prompt_head": prompts[0][:200],
        "last_prompt_head": prompts[-1][:200],
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(man, open(a.out, "w"), indent=1)
    print(f"n={man['n_prompts']}  concat sha256={man['sha256_concat'][:24]}...")
    print(f"revision={man['hf_dataset_revision']}")
    print(f"chars min/med/max={man['chars_min_median_max']}")
    print(f"first: {man['first_prompt_head'][:120]!r}")
    print(f"wrote {a.out} ({man['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
