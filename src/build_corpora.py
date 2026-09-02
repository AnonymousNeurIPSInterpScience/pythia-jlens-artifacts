#!/usr/bin/env python3
"""build_corpora.py — build the E28 corpus pool ONCE, ship identically to every box.

WHY ONCE. E28 uses >=3 seeds per (corpus, N) cell. A seed only means something if every box draws
from the SAME pool; if each box streamed the Pile independently, "seed 0" would denote different
prompts on different machines and the seed variance would be uninterpretable.

WHY THESE FIVE. Chosen to span DISPERSION (internal heterogeneity), which is what identifies the
slope in Y = Y_inf + c*D(P)/N -- NOT to maximise separability, which is saturated at AUC 1.0 and
cannot rank. Expectations below are to be MEASURED, not assumed:
    Pile-CC              heterogeneous web prose        expect HIGH dispersion
    StackExchange        semi-structured Q&A            expect mid
    Wikipedia (en)       encyclopedic prose             expect mid
    Github               code, templated                expect LOW
    USPTO Backgrounds    patent boilerplate             expect VERY LOW
All five are >=5.7% of the Pile, so one pass collects them together.
"""
import json, os, sys, hashlib, argparse

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


COMPONENTS = ["Pile-CC", "StackExchange", "Wikipedia (en)", "Github", "USPTO Backgrounds"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=2400, help="prompts per corpus; 3 disjoint "
                    "seed-blocks of pool/3 each, so seeds share NO prompts")
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--max-rows", type=int, default=400000)
    ap.add_argument("--out", default="corpora")
    a = ap.parse_args()
    from datasets import load_dataset
    os.makedirs(a.out, exist_ok=True)
    buf = {c: [] for c in COMPONENTS}
    ds = pinned_load(load_dataset, "monology/pile-uncopyrighted", split="train", streaming=True)
    for i, r in enumerate(ds):
        n = (r.get("meta") or {}).get("pile_set_name")
        if n in buf and len(buf[n]) < a.pool:
            t = r["text"].strip()
            if len(t) >= a.min_chars:
                buf[n].append(t)
        if i % 20000 == 0:
            print(f"  {i:>7} rows | " + " ".join(f"{k.split()[0][:8]}={len(v)}" for k, v in buf.items()),
                  flush=True)
        if all(len(v) >= a.pool for v in buf.values()) or i >= a.max_rows:
            break
    man = {}
    for c, texts in buf.items():
        slug = c.replace(" ", "_").replace("(", "").replace(")", "")
        p = os.path.join(a.out, f"{slug}.jsonl")
        with open(p, "w") as f:
            for t in texts:
                f.write(json.dumps({"text": t}) + "\n")
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        man[c] = {"file": os.path.basename(p), "n": len(texts), "sha256": sha,
                  "seed_blocks": [[0, len(texts)//3], [len(texts)//3, 2*len(texts)//3],
                                  [2*len(texts)//3, len(texts)]]}
        print(f"  wrote {p}  n={len(texts)}  sha={sha[:16]}")
    man["_note"] = ("seed_blocks are DISJOINT index ranges; seed k draws its N prompts from "
                    "block k, so seeds share no prompts and the seed variance is a genuine "
                    "resampling variance rather than an overlap artifact")
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print("manifest written")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
