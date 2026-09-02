#!/usr/bin/env python3
"""build_ood_corpora.py — candidate Q corpora OUTSIDE the Pile, in the E28 window format.

WHY
  Every corpus this programme has used is a Pile component, so Pythia trained on all of them.
  Every "shift" measured so far is WITHIN the model's training distribution: we have measured
  corpus identity, not distribution shift. These are candidate Q's that are plausibly outside it.

  CRITICAL: "plausibly outside" is a HYPOTHESIS, not a property. Nothing here is labelled OOD.
  Each corpus carries a `claim` field stating why it MIGHT be outside and what that is worth, and
  M1 containment (t35_containment.py) then MEASURES overlap against the actual Pythia stream.
  That is the point of building them: to be measured, not assumed.

  Panel design follows the two external review reviews:
    * temporal exclusion is the strongest cheap evidence (the Pile snapshot is ~2020);
    * source exclusion alone is weak;
    * string-level exclusion is the only thing that really settles it, and needs the dataloader;
    * Common-Crawl derivatives are a documented TRAP -- Pile-CC is itself CC-derived, so FineWeb
      may contain the same pages, later versions, or mirrors. We include FineWeb DELIBERATELY as
      a positive control: if M1 does not show it overlapping, M1 is broken.

FORMAT
  Identical to build_corpora.py so every downstream tool works unchanged: {"text": ...} JSONL,
  2400 documents each of >= MIN_TOKENS tokens under the Pythia tokenizer, three disjoint
  800-document seed blocks, manifest with SHA-256.

    python build_ood_corpora.py --n 2400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "corpora")
MIN_TOKENS = 128

# ---------------------------------------------------------------------------------------------
# DATASET REVISION PINNING (added 2026-09-01)
#
# Until now this called load_dataset() with no `revision=`, so the pool it produced depended on
# whatever the upstream dataset happened to be that day. the release procedure finding
# B0 recorded exactly that and concluded the pool "is not regenerable". A from-scratch rebuild on
# 2026-08-31 in fact reproduced 8 of 11 corpora BIT-IDENTICALLY, which is evidence the upstream
# had not moved -- not a guarantee that it never will. These pins make it a guarantee.
#
# The pins live in corpora/dataset_revisions.json, ONE source of truth for both builders and for
# the containment experiment, so a pin cannot drift between them.
def _revisions():
    import json as _j, os as _o
    p = _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "corpora",
                     "dataset_revisions.json")
    if not _o.path.exists(p):
        raise SystemExit(
            "corpora/dataset_revisions.json is missing. It pins the upstream dataset commit for "
            "every corpus and is what makes a rebuild verifiable. Restore it from the repository.")
    return {k: v for k, v in _j.load(open(p)).items() if not k.startswith("_")}


def pinned_load(load_dataset, path, **kw):
    """load_dataset() with the recorded revision, and a hard failure if none is recorded.

    `name` (the second positional of load_dataset) is a CONFIG name, not a locale. Datasets that
    declare no configs must not be given one -- monology/pile-uncopyrighted declares none.
    """
    revs = _revisions()
    if path not in revs:
        raise SystemExit(f"{path} has no pinned revision in corpora/dataset_revisions.json. "
                         "Add one rather than loading an unpinned dataset.")
    rev = revs[path]["sha"]
    print(f"    {path} @ {rev[:12]}", flush=True)
    return load_dataset(path, revision=rev, **kw)


# name -> (loader kwargs, text field, claim about why it may be outside the Pile)
PANEL = {
    "OOD_Wikipedia_2023": (
        dict(path="wikimedia/wikipedia", name="20231101.en", split="train"), "text",
        "TEMPORAL. Same source family as the Pile's Wikipedia component, but a 2023-11 snapshot. "
        "Revisions written after the ~2020 Pile snapshot cannot have been in it. Confounds: three "
        "years of new entities, events and editorial convention move with the exclusion. This is "
        "external review's PRIMARY recommended panel because it holds language and register nearly fixed."),
    "OOD_arXiv_2023": (
        dict(path="neuralwork/arxiver", split="train"), "markdown",
        "TEMPORAL. 2023 arXiv papers; the Pile's arXiv component predates the snapshot. Stable "
        "LaTeX/scientific register. Confounds: new terminology, fields, authors; heavy template "
        "reuse means short-k overlap will be high regardless of exposure."),
    "OOD_News_2024": (
        dict(path="RealTimeData/bbc_news_alltime", name="2024-06", split="train"), "content",
        "TEMPORAL. June 2024 news, four years past the snapshot. Ordinary English, so it tests "
        "transport without immediately confounding on script or competence. Confounds: news "
        "register, named entities, and possible syndication into web crawls."),
    "OOD_CommonPile": (
        dict(path="common-pile/comma_v0.1_training_dataset", split="train"), "text",
        "PROVENANCE. Assembled later from openly-licensed and public-domain sources, covering "
        "domains not deliberately represented in the Pile. Tests whether the effect is about "
        "corpus construction/licensing rather than topic. Confounds: licence filtering changes the "
        "source population; some documents may still be older web material."),
    "TRAP_FineWeb": (
        dict(path="HuggingFaceFW/fineweb", name="sample-10BT", split="train"), "text",
        "DELIBERATE POSITIVE CONTROL, NOT A CLEAN Q. FineWeb is Common-Crawl-derived and Pile-CC "
        "is too, so it can contain the same pages, later versions of them, or mirrors. Both "
        "external review reviews name CC derivatives as a trap. Included so M1 has something it MUST "
        "flag as overlapping: if FineWeb does not show high containment, M1 is broken."),
    "CONTROL_PubMed_2023": (
        dict(path="MedRAG/pubmed", split="train"), "content",
        "AMBIGUOUS BY DESIGN. The Pile contains PubMed Central, so this is source-INCLUDED; only "
        "post-snapshot articles would be temporally excluded and this dump is not date-filtered. "
        "Expected to land BETWEEN the Pile components and the temporal panels, which makes it a "
        "useful mid-scale reference point for whatever divergence metric we fit."),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-scan", type=int, default=400_000)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
    os.makedirs(OUT, exist_ok=True)
    man_path = os.path.join(OUT, "manifest_ood.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}

    for name, (kw, field, claim) in PANEL.items():
        if a.only and name not in a.only:
            continue
        dst = os.path.join(OUT, f"{name}.jsonl")
        if os.path.exists(dst) and man.get(name, {}).get("n") == a.n:
            print(f"  {name}: already built, skipping"); continue
        print(f"  {name}: streaming...", flush=True)
        _kw = dict(kw); _path = _kw.pop("path")
        ds = pinned_load(load_dataset, _path, **_kw, streaming=True)
        kept, scanned = [], 0
        for row in ds:
            scanned += 1
            if scanned > a.max_scan:
                break
            t = row.get(field) or ""
            if not isinstance(t, str) or len(t) < 400:
                continue
            if len(tok(t[:4000]).input_ids) < MIN_TOKENS:
                continue
            kept.append(t)
            if len(kept) >= a.n:
                break
        if len(kept) < a.n:
            print(f"    WARNING only {len(kept)}/{a.n} after scanning {scanned}", flush=True)
        with open(dst, "w") as f:
            for t in kept:
                f.write(json.dumps({"text": t}) + "\n")
        sha = hashlib.sha256(open(dst, "rb").read()).hexdigest()
        b = len(kept) // 3
        man[name] = {"file": f"{name}.jsonl", "n": len(kept), "sha256": sha,
                     "seed_blocks": [[0, b], [b, 2 * b], [2 * b, 3 * b]],
                     "hf": {k: v for k, v in kw.items()}, "text_field": field,
                     "claim": claim, "scanned": scanned,
                     "status": "CANDIDATE — outside-Pile is a HYPOTHESIS to be measured by M1, "
                               "not a property asserted here"}
        json.dump(man, open(man_path, "w"), indent=1)
        print(f"    {len(kept)} docs, sha {sha[:16]}...", flush=True)

    print(f"\nmanifest -> {man_path}")
    for k, v in man.items():
        print(f"  {k:22s} n={v['n']:5d}  {v['claim'][:60]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
