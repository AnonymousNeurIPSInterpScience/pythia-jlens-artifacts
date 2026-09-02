#!/usr/bin/env python3
"""readout_exposure.py — ground truth on which results are exposed to the readout defect.

THE DEFECT
  The released eval prompts carry a trailing space so that `prompt + target` concatenates as a
  string. `src/anchor_evals.py:65` records the rule: "last" == final prompt token AFTER RSTRIP,
  and `:228` applies it. `readout_position()` itself returns -1 for the five "last" sets and never
  strips, so the CALLER must strip before tokenizing. Scripts that do not strip read token id 209,
  a bare space, which does not occur in the tokenisation of prompt+target.

  Affected: 157 of 551 released items (multihop 83/93, order-ops 55/55, multilingual 19/107;
  poetry/typo/association 0). Measured cost, persist, admitted-5, 410M ladder:
  order-ops 29.8x, multihop 3.13x, multilingual 1.01x, poetry/typo/association 1.00x exactly.

CLASSIFICATION, assigned mechanically
  IMMUNE      the producing script never loads the eval battery, and neither does anything it
              reads. No readout, so the defect cannot reach it.
  CLEAN       the producing script loads the battery AND strips. Already correct.
  EXPOSED     the producing script loads the battery and does NOT strip.
  INHERITED   the script does not load the battery itself but reads results produced by an
              EXPOSED script. Exposure propagates through provenance.inputs.
  UNKNOWN     no provenance, or the script is missing.

  Exposure is resolved to a fixpoint over the provenance.inputs graph, so a recompute three levels
  downstream of a defective scorer is still marked EXPOSED.

  RESCORED marks the small set for which a corrected-readout measurement exists.

    python tools/readout_exposure.py                  # table to stdout
    python tools/readout_exposure.py --json out.json  # machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")
EXPERIMENTS = os.path.join(REPO, "experiments")

# The corrected-readout measurements that actually exist today.
RESCORED = {
    "ladder410": "15 (corpus, seed) cells re-scored under .rstrip() at read-rung Q0 "
                 "(review/reproducibility/data/rstrip_rescore/rs_*.json)",
    "e33_logit_baseline_410m_v2": "logit constant re-derived under both conventions "
                                  "(review/reproducibility/scripts/logit_baseline.json)",
    "e54_aggregation_audit": "derangement control re-run under both readouts, 5 corpora x "
                             "N in {200,400} x 5 draws (review/reproducibility/scripts/derange.py)",
}


# The results files paper/original_paper.tex actually cites, per paper/PAPER_HANDOFF.md section 5.
# This is the denominator that matters: results/ holds many per-cell inputs, but a defect only
# reaches the submission through one of these.
PAPER_CITED = [
    ("scale floor, W_U concept-row geometry", "results/e38_jgeometry.json"),
    ("160M interval, both aggregations", "results/t22_bootstrap_ci_160m.json"),
    ("160M interval, persist arm", "results/t22_bootstrap_ci_persist_160m.json"),
    ("the 8x8 grid, margins, LOO, leave-two-out", "results/e52_factorial_410m.json"),
    ("aggregation audit, matrix_loo", "results/e54_aggregation_audit.json"),
    ("bootstrap intervals on the variance split", "results/e57_grid_variance_ci.json"),
    ("per-draw cells behind E57", "results/e57_factorial_cells_410m.json"),
    ("independent 5x5 replication at N=400", "results/e55_matrix_robustness.json"),
    ("read-dose sweep", "results/e59_read_dose_410m.json"),
    ("per-corpus asymptotes, flatness in N", "results/e53_ladder_summary.json"),
    ("20 failed predictors", "results/e56_predictor_registry.json"),
    ("derangement, both aggregations", "results/e48_crossover_410m.json"),
    ("min-union mechanism and band-width sweep", "results/d1_min_union_diagnostic_410m.json"),
    ("randomised-network null", "results/e61_randomized_null_410m.json"),
    ("containment index", "results/e48c_exposure_vs_read.json"),
    ("fitter and readout fidelity", "results/e60_fitter_determinism.json"),
    ("context-pool disjointness", "results/e58_algebra_audit.json"),
]


# Files whose CITED FIELDS are safer than the file as a whole. This tool classifies files; two
# results files mix a readout-free computation with a readout one, and the paper cites the safe
# half. Stated explicitly rather than silently upgrading the file.
FIELD_LEVEL_NOTES = {
    "results/e38_jgeometry.json":
        "the file is EXPOSED (t38 loads the battery), but the fields the paper cites are "
        "E39/*/mean_cos_I and eff_rank_I, computed from W_U alone with no prompts. The scale-floor "
        "claim is IMMUNE at field level. E42's gap_by_corpus_410m is not.",
    "results/e58_algebra_audit.json":
        "the file is EXPOSED (t58 loads the battery for the concept-token overlap), but the claim "
        "the paper cites is context-pool disjointness, a corpus-vs-corpus computation with no "
        "readout. IMMUNE at field level.",
}


# Results files that carry no provenance.script. Their producer is inferred from the path, which
# is stated here rather than guessed at read time so the mapping is auditable.
INFERRED_PRODUCER = {
    "results/ladder410/": "experiments/e28_ladder_410m.py",
    "results/ladder1b/": "experiments/e28_ladder_410m.py",
    "results/ladder1b_b613/": "experiments/e28_ladder_410m.py",
    "results/ladder410_cpu_rescore/": "experiments/e28_ladder_410m.py",
    "results/e48/": "experiments/trainval.py",
    "results/e61/": "experiments/trainval.py",
}


# Scripts whose strip status is a fact about the RUN rather than about the source, stated here so
# it is auditable rather than inferred. Same idiom as INFERRED_PRODUCER and FIELD_LEVEL_NOTES.
# Nothing may be added here to make a file look better than it is: each entry names the line.
SCRIPT_NOTES = {
    "experiments/cv6_per_family_ladder.py": (
        "RSTRIP",
        "cache_activations() strips at `pr = pr.rstrip() if strip else pr`, and `strip` DEFAULTS "
        "TO TRUE. The False path is reachable only from --c0-reproduce-d3's negative control, "
        "which exists to prove control C0 can fail and is stored under "
        "`negative_control` in cv6_c0_scorer_equivalence.json, never in a graded arm."),
}

_RSTRIP_DIRECT = re.compile(r"""\[["']prompt["']\]\s*\.rstrip\(\)""")
_PROMPT_BIND = re.compile(r"""^\s*(\w+)\s*=\s*[^\n]*\[["']prompt["']\]""", re.M)


def _flag_conditioned(line: str) -> bool:
    """Is this .rstrip() behind a STRIP FLAG, as opposed to inside some other conditional?

    The previous test was `" if " in line`, which fired on the isinstance ternary that every
    script in this repo uses to accept either a string prompt or a message list:

        prompt = it["prompt"].rstrip() if isinstance(it["prompt"], str) else " ".join(...).rstrip()

    Both branches of that strip. Reading it as "strips only when --rstrip is passed" marked four
    correctly-stripped CV-series files EXPOSED -- including CV3, whose own C1 control fired against
    the STRIPPED constant 0.19810852520167826, which is positive proof it stripped. Look at the
    CONDITION, and call it conditional only when the condition itself is about stripping.
    """
    m = re.search(r"\bif\b(?P<cond>.*?)(?:\belse\b|$)", line)
    if not m:
        return False
    return bool(re.search(r"\b(?:rstrip|strip)\b", m.group("cond")))


def script_class(path: str) -> tuple[str, str]:
    """(class, why) for one experiment script, from its source."""
    if not os.path.exists(path):
        return "MISSING", "script not on disk"
    rel = os.path.relpath(os.path.abspath(path), REPO)
    src = open(path, encoding="utf-8", errors="replace").read()
    if "load_eval" not in src:
        # DELEGATION. A script that imports its battery-caching helper from another experiment
        # reads the battery just as surely as one that calls load_eval itself -- the call is one
        # frame down. Reporting "no readout over the battery" there is a FALSE NEGATIVE, and the
        # dangerous kind: it would say the same thing if the delegate did NOT strip. Follow the
        # import one level and inherit the delegate's class, recording who was followed.
        # Both import spellings, including the PARENTHESISED MULTI-LINE form -- a first cut read
        # only the first physical line, so `from cv6 import (  # noqa\n    cache_activations,` was
        # missed and CV7 came out NO_EVAL, the exact false negative this block exists to remove.
        imports = [(m.group(1), m.group(2)) for m in
                   re.finditer(r"^\s*from\s+(\w+)\s+import\s+\(([^)]*)\)", src, re.M | re.S)]
        imports += [(m.group(1), m.group(2)) for m in
                    re.finditer(r"^\s*from\s+(\w+)\s+import\s+([^(\n]*)$", src, re.M)]
        for mod, names in imports:
            if "cache_activations" not in names and "readout_position" not in names:
                continue
            dele = os.path.join(EXPERIMENTS, f"{mod}.py")
            if not os.path.exists(dele) or os.path.abspath(dele) == os.path.abspath(path):
                continue
            k, w = script_class(dele)
            if k in ("RSTRIP", "COND_RSTRIP", "NO_RSTRIP"):
                return k, f"delegates the battery read to {mod}.py, which {w}"
        return "NO_EVAL", "never calls load_eval; no readout over the battery"
    if rel in SCRIPT_NOTES:
        k, w = SCRIPT_NOTES[rel]
        return k, w
    # (a) the direct form:            it["prompt"].rstrip()
    hits = [(m.start(), m.end()) for m in _RSTRIP_DIRECT.finditer(src)]
    # (b) the two-step form, which is what cv5/d3/cv6 use and what the direct regex missed:
    #         pr = it["prompt"] if isinstance(...) else " ".join(...)
    #         pr = pr.rstrip()
    #     Bind the variable from the prompt, then require a later rstrip ONTO THAT VARIABLE.
    for b in _PROMPT_BIND.finditer(src):
        var = b.group(1)
        for m in re.finditer(rf"^\s*{re.escape(var)}\s*=\s*{re.escape(var)}\s*\.rstrip\(\)",
                             src[b.end():], re.M):
            hits.append((b.end() + m.start(), b.end() + m.end()))
    if not hits:
        return "NO_RSTRIP", "loads the battery and does NOT strip the prompt"
    for a, b in hits:
        line = src[src.rfind("\n", 0, a) + 1: src.find("\n", b)]
        if _flag_conditioned(line):
            return "COND_RSTRIP", "strips only when --rstrip is passed; unstripped by default"
    return "RSTRIP", "loads the battery and applies .rstrip() to the prompt"


_EXP_BY_NUM = None


def producer_from_stem(fn: str) -> str | None:
    """Map a results filename to its experiment script by the shared NN index.

    The repo names results eNN_/dN_/tNN_ and scripts tNN_/dN_, with e->t for the experiment
    series (e33_logit_baseline_410m.json <- experiments/t33_logit_baseline.py). Matching on the
    integer alone is what makes that mechanical rather than a guess.
    """
    global _EXP_BY_NUM
    if _EXP_BY_NUM is None:
        _EXP_BY_NUM = {}
        for f in sorted(os.listdir(EXPERIMENTS)):
            if not f.endswith(".py"):
                continue
            m = re.match(r"^([a-z]+)(\d+)[a-z]*_", f)
            if m:
                _EXP_BY_NUM.setdefault((m.group(1)[0], m.group(2)), f)
    m = re.match(r"^([a-z]+)(\d+)[a-z]*_", fn)
    if not m:
        return None
    series, num = m.group(1)[0], m.group(2)
    # results use e/d/t; scripts use t/d. e-series results come from t-series scripts.
    for s in (series, "t", "d"):
        hit = _EXP_BY_NUM.get((s, num))
        if hit:
            return os.path.join("experiments", hit)
    return None


def load_results() -> dict:
    """{stem: {path, script, inputs}} for every non-sidecar results JSON."""
    out = {}
    for root, _dirs, files in os.walk(RESULTS):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            if fn.endswith("_provenance.json") or fn.endswith("_reads.json"):
                continue
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, REPO)
            try:
                d = json.load(open(p, encoding="utf-8", errors="replace"))
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            prov = d.get("provenance") or {}
            sc = (prov.get("script") or {}).get("path")
            inferred = False
            if not sc:
                for pref, prod in INFERRED_PRODUCER.items():
                    if rel.startswith(pref):
                        sc, inferred = prod, True
                        break
            if not sc:
                sc = producer_from_stem(fn)
                inferred = bool(sc)
            ins = [i.get("path") for i in (prov.get("inputs") or []) if isinstance(i, dict)]
            # a results file that declares its own rstrip arm is authoritative about it
            arm = d.get("rstrip_arm")
            out[rel] = {"script": sc, "inputs": [i for i in ins if i],
                        "inferred": inferred, "rstrip_arm": arm}
    return out


def resolve(res: dict) -> dict:
    """Assign a class to every results file, propagating exposure through inputs to a fixpoint."""
    cls = {}
    why = {}
    for rel, meta in res.items():
        sc = meta["script"]
        if not sc:
            cls[rel], why[rel] = "UNKNOWN", "no provenance.script"
            continue
        k, w = script_class(os.path.join(REPO, sc))
        if k == "COND_RSTRIP":
            # authoritative: the stored file records whether the arm ran
            if meta.get("rstrip_arm") is True:
                k, w = "RSTRIP", w + "; this file records rstrip_arm=true"
            else:
                k, w = "NO_RSTRIP", w + f"; this file records rstrip_arm={meta.get('rstrip_arm')!r}"
        if meta.get("inferred"):
            w = f"[producer inferred from path] {w}"
        if k == "NO_RSTRIP":
            cls[rel], why[rel] = "EXPOSED", f"{os.path.basename(sc)}: {w}"
        elif k == "RSTRIP":
            cls[rel], why[rel] = "CLEAN", f"{os.path.basename(sc)}: {w}"
        elif k == "NO_EVAL":
            cls[rel], why[rel] = "IMMUNE", f"{os.path.basename(sc)}: {w}"
        else:
            cls[rel], why[rel] = "UNKNOWN", f"{sc}: {w}"

    # propagate: an IMMUNE recompute that reads an EXPOSED result is INHERITED
    changed = True
    rounds = 0
    while changed and rounds < 50:
        changed, rounds = False, rounds + 1
        for rel, meta in res.items():
            if cls[rel] not in ("IMMUNE", "UNKNOWN"):
                continue
            for src in meta["inputs"]:
                # inputs may name a directory or a glob-ish path; match by prefix
                hits = [r for r in res if r == src or r.startswith(src.rstrip("/") + "/")]
                if any(cls.get(h) in ("EXPOSED", "INHERITED") for h in hits):
                    cls[rel] = "INHERITED"
                    why[rel] = f"reads {src}, which is exposed"
                    changed = True
                    break
    return cls, why


def rescored_note(rel: str) -> str:
    for k, v in RESCORED.items():
        if k in rel:
            return v
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    res = load_results()
    cls, why = resolve(res)

    order = ["IMMUNE", "CLEAN", "RESCORED", "EXPOSED", "INHERITED", "UNKNOWN"]
    buckets: dict[str, list] = {k: [] for k in order}
    for rel in sorted(res):
        k = cls[rel]
        if rescored_note(rel) and k in ("EXPOSED", "INHERITED"):
            buckets["RESCORED"].append(rel)
        else:
            buckets[k].append(rel)

    tot = len(res)
    print(f"READOUT EXPOSURE — {tot} non-sidecar results files under results/\n")
    for k in order:
        n = len(buckets[k])
        print(f"  {k:<10} {n:>4}  ({100*n/tot:5.1f}%)")
    print()
    safe = len(buckets["IMMUNE"]) + len(buckets["CLEAN"])
    print(f"  readout-safe (IMMUNE + CLEAN)      {safe:>4}  ({100*safe/tot:5.1f}%)")
    exp = len(buckets["EXPOSED"]) + len(buckets["INHERITED"])
    print(f"  needs re-scoring (EXPOSED + INHER) {exp:>4}  ({100*exp/tot:5.1f}%)")
    print()

    for k in order:
        if not buckets[k]:
            continue
        print(f"--- {k} ---")
        for rel in buckets[k]:
            note = rescored_note(rel)
            print(f"  {rel}")
            print(f"      {why[rel]}")
            if note:
                print(f"      RE-SCORED: {note}")
        print()

    print("=" * 78)
    print("THE VIEW THAT DECIDES THE PAPER — the results files original_paper.tex cites")
    print("=" * 78)
    def klass(rel):
        if rel in buckets["RESCORED"]:
            return "RESCORED"
        return cls.get(rel, "NOT-ON-DISK")
    tally = {}
    for what, rel in PAPER_CITED:
        k = klass(rel)
        tally[k] = tally.get(k, 0) + 1
        print(f"  {k:<10} {what}")
        print(f"             {rel}")
    print()
    for k in order + ["NOT-ON-DISK"]:
        if tally.get(k):
            print(f"  {k:<10} {tally[k]:>3} of {len(PAPER_CITED)}")
    print("  FIELD-LEVEL EXCEPTIONS (the file is exposed; the cited fields are not):")
    for rel, note in FIELD_LEVEL_NOTES.items():
        print(f"    {rel}\n      {note}")
    print()
    safe_p = tally.get("IMMUNE", 0) + tally.get("CLEAN", 0)
    print()
    print(f"  paper-cited files that are readout-safe : {safe_p} of {len(PAPER_CITED)}")
    print(f"  paper-cited files needing a re-score    : "
          f"{tally.get('EXPOSED',0)+tally.get('INHERITED',0)} of {len(PAPER_CITED)}")
    print()

    if a.json:
        payload = {
            "n_results": tot,
            "counts": {k: len(buckets[k]) for k in order},
            "by_file": {rel: {"class": ("RESCORED" if rel in buckets["RESCORED"] else cls[rel]),
                              "why": why[rel],
                              "script": res[rel]["script"],
                              "rescored": rescored_note(rel)} for rel in sorted(res)},
        }
        with open(a.json, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
