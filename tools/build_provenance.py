#!/usr/bin/env python3
"""build_provenance.py — stamp a generated PROVENANCE block into every experiment document.

Each `docs/experiments/preregs/*.md` states what was run and why. This adds, at the end of each, a block
naming every results file it cites with that file's size, SHA-256, producing script, and readout
exposure class. The block is regenerated in place between markers, so it cannot drift from the tree.

It also writes the index table `docs/experiments/INDEX.md`, which `docs/context/CONTEXT.md` links to.

    tools/build_provenance.py            # rewrite every block + the index
    tools/build_provenance.py --check    # exit 1 if any block is stale or a file is missing
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXP = os.path.join(REPO, "docs", "experiments", "preregs")

BEGIN = "<!-- BEGIN GENERATED PROVENANCE — tools/build_provenance.py -->"
END = "<!-- END GENERATED PROVENANCE -->"

# readout exposure, reused from the ledger rather than re-derived
# The readout ledger moved to docs/reproducibility/ in the specs/ -> docs/ restructure (d06a39d)
# and this path was not updated. tmp/ is gitignored and absent, load_exposure() swallowed the
# resulting FileNotFoundError, and so EVERY generated provenance block in the repo has been
# printing `UNCLASSIFIED` for the readout column since the restructure -- a column CLAUDE.md 5
# says this tool stamps. The canonical location is checked first; tmp/ stays as a fallback for
# an ad-hoc regeneration.
EXPOSURE = os.path.join(REPO, "docs", "reproducibility", "readout_exposure.json")
EXPOSURE_FALLBACK = os.path.join(REPO, "tmp", "readout_exposure.json")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_exposure() -> dict:
    """The readout class per results file. Absence is REPORTED, not silently defaulted --
    a provenance table whose readout column reads UNCLASSIFIED everywhere looks like a
    measured fact and is actually a missing file."""
    for path in (EXPOSURE, EXPOSURE_FALLBACK):
        try:
            return json.load(open(path))["by_file"]
        except Exception:
            continue
    print(f"WARNING: no readout ledger at {os.path.relpath(EXPOSURE, REPO)} or "
          f"{os.path.relpath(EXPOSURE_FALLBACK, REPO)}; the readout column will read "
          f"UNCLASSIFIED for every file. Regenerate with tools/readout_exposure.py --json",
          file=sys.stderr)
    return {}


def cited_results(text: str) -> list[str]:
    """Results files a document cites, including directory references like results/ladder410/."""
    out, seen = [], set()
    for h in re.findall(r"results/[A-Za-z0-9_/.\-]*\.json", text):
        if h.endswith("_provenance.json") or h in seen:
            continue
        seen.add(h)
        out.append(h)
    # directory citations: results/ladder410/ or results/ladder410/*.json
    for d in re.findall(r"results/([A-Za-z0-9_\-]+)/(?:\*|\s|`|\)|$)", text):
        dp = os.path.join(REPO, "results", d)
        if not os.path.isdir(dp):
            continue
        for f in sorted(glob.glob(os.path.join(dp, "*.json"))):
            rel = os.path.relpath(f, REPO)
            if rel.endswith("_provenance.json") or rel in seen:
                continue
            seen.add(rel)
            out.append(rel)
    return out


def own_result(doc: str, recs: list[dict], text: str = "") -> dict | None:
    """The results file that belongs to THIS document, matched on the shared id.

    Without this the index reports a neighbour's verdict: R9 cites e48's file as an input, and a
    naive first-match prints e48's verdict under R9's row.
    """
    # an explicit "**Output:**" line in the document is authoritative
    decl = re.search(r"\*\*Output:\*\*(.+)", text or "")
    if decl:
        named = re.findall(r"results/[A-Za-z0-9_/.\-]+", decl.group(1))
        for n in named:
            for r in recs:
                if r["path"].startswith(n.rstrip("/")):
                    return r
    m = re.match(r"^([A-Za-z]+\d+[a-z]?)_", doc)
    if not m:
        return None
    stem = m.group(1).lower()
    for r in recs:
        if os.path.basename(r["path"]).lower().startswith(stem + "_"):
            return r
    return None


def describe(rel: str, exposure: dict) -> dict:
    p = os.path.join(REPO, rel)
    if not os.path.exists(p):
        return {"path": rel, "exists": False}
    d = {}
    try:
        d = json.load(open(p))
    except Exception:
        d = {}
    prov = d.get("provenance") or {} if isinstance(d, dict) else {}
    script = (prov.get("script") or {}).get("path")
    payload = prov.get("payload_sha256")
    verdict = ""
    if isinstance(d, dict):
        for k in ("VERDICT", "verdict", "status", "STATUS"):
            if d.get(k):
                verdict = str(d[k])
                break
    return {
        "path": rel,
        "exists": True,
        "bytes": os.path.getsize(p),
        "sha256": sha256_file(p),
        "script": script,
        "payload_sha256": payload,
        "exposure": (exposure.get(rel) or {}).get("class", "UNCLASSIFIED"),
        "verdict": verdict,
    }


def block_for(recs: list[dict]) -> str:
    if not recs:
        return (BEGIN + "\n\n*No results file is cited by this document.*\n\n" + END)
    lines = [BEGIN, "",
             "## PROVENANCE",
             "",
             "Generated by `tools/build_provenance.py`; do not edit by hand. `readout` is the",
             "exposure class from `tools/readout_exposure.py`: **CLEAN** or **IMMUNE** need no",
             "re-score, **EXPOSED** or **INHERITED** were produced at the legacy readout.",
             "",
             "| results file | bytes | sha256 (first 16) | produced by | readout |",
             "|---|---:|---|---|---|"]
    for r in recs:
        if not r["exists"]:
            lines.append(f"| `{r['path']}` | — | **MISSING** | — | — |")
            continue
        sc = os.path.basename(r["script"]) if r.get("script") else "—"
        lines.append(f"| `{r['path']}` | {r['bytes']:,} | `{r['sha256'][:16]}` | `{sc}` | {r['exposure']} |")
    stamped = [r for r in recs if r.get("payload_sha256")]
    if stamped:
        lines += ["", "**Payload checksums** (content only, provenance block excluded):", ""]
        for r in stamped:
            lines.append(f"* `{os.path.basename(r['path'])}` — `{r['payload_sha256'][:32]}`")
    lines += ["", END]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    exposure = load_exposure()
    if not exposure:
        print("NOTE: tmp/readout_exposure.json absent; readout column will read UNCLASSIFIED.",
              file=sys.stderr)

    stale, missing, index = [], [], []
    for md in sorted(glob.glob(os.path.join(EXP, "*.md"))):
        name = os.path.basename(md)
        if name in ("README.md", "INDEX.md"):
            continue
        text = open(md).read()

        body = re.split(re.escape(BEGIN), text)[0].rstrip()
        # the separator this tool appends must not accumulate across runs
        body = re.sub(r"\n-{3,}\s*$", "", body).rstrip()

        # Scan the BODY, never the block this tool wrote last time. Reading the full text made the
        # generator latch: the table it emits names every file it found, so a citation the analysis
        # had already removed from the prose was rediscovered from the table on the next run and
        # went on reporting MISSING forever. CV4 is the case that exposed it — its phases 2-3 were
        # superseded before they ran, so the output they would have written never existed.
        recs = [describe(r, exposure) for r in cited_results(body)]
        missing += [r["path"] for r in recs if not r["exists"]]
        new_block = block_for(recs)
        updated = body + "\n\n---\n\n" + new_block + "\n"
        if updated != text:
            stale.append(name)
            if not a.check:
                open(md, "w").write(updated)

        mine = own_result(name, recs, text)
        index.append({
            "doc": name,
            "n_files": len(recs),
            "readout": sorted({r.get("exposure", "?") for r in recs if r["exists"]}),
            "own": os.path.basename(mine["path"]) if mine else "",
            "verdict": ((mine.get("verdict") or "")[:110] if mine else ""),
        })

    if a.check:
        if missing:
            print("MISSING results files:", *sorted(set(missing)), sep="\n  ")
        if stale:
            print("STALE provenance blocks:", *stale, sep="\n  ")
        print(f"{len(index)} documents; {len(missing)} missing files; {len(stale)} stale blocks")
        return 1 if (missing or stale) else 0

    idx = ["# INDEX — every experiment document, generated",
           "",
           "Written by `tools/build_provenance.py`. One row per document under",
           "`docs/experiments/preregs/`. The narrative account is [`../context/CONTEXT.md`](../context/CONTEXT.md);",
           "every claim with its tier is [`../context/RESULTS_TAXONOMY.md`](../context/RESULTS_TAXONOMY.md).",
           "",
           "The verdict is read from **this document's own** results file, matched on the shared",
           "id, not from the first file it happens to cite. A blank means the document cites no",
           "file of its own, which is itself worth knowing.",
           "",
           "| document | files | its own result | readout | verdict as stored |",
           "|---|---:|---|---|---|"]
    for r in index:
        ro = ", ".join(r["readout"]) or "—"
        v = r["verdict"].replace("|", "/").replace("\n", " ") or "—"
        own = f"`{r['own']}`" if r["own"] else "**none**"
        idx.append(f"| [`{r['doc']}`](preregs/{r['doc']}) | {r['n_files']} | {own} | {ro} | {v} |")
    idx += ["", f"**{len(index)} documents.** Regenerate with `tools/build_provenance.py`."]
    open(os.path.join(EXP, "..", "INDEX.md"), "w").write("\n".join(idx) + "\n")

    print(f"stamped {len(stale)} document(s); wrote docs/experiments/INDEX.md")
    if missing:
        print("WARNING, missing results files:", *sorted(set(missing)), sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
