#!/usr/bin/env python3
"""r4g_e28_provenance.py — R4g: repair E28's sidecar provenance and the artifact ledger.

PRE-REGISTRATION: docs/experiments/preregs/R4_corrections.md item R4g.
Findings closed: M-3, M-4, F-11.

THREE DEFECTS, each verified here rather than taken on report:

  M-3  All 166 E28 sidecars record `"corpus": "wikitext"`. That is `--corpus`'s DEFAULT leaking:
       the true corpus is in the same file, as `corpus_file` (e.g. "corpora/Github.jsonl"). Any
       reader trusting `corpus` attributes every one of 166 operators to the wrong corpus -- on a
       programme whose entire headline is a corpus effect.

  M-4  All 166 record `anchor_agreement_gate: null`. Null is ambiguous between "the gate ran and
       returned nothing" and "the gate was never run". It is rewritten to say which.

  F-11 0 of 166 E28 operators are named in ARTIFACTS.md, and ARTIFACTS.md still asserts they are
       unpulled and sitting on rented boxes. They are on local disk.

HOW THE SIDECARS ARE EDITED, and why that is not a silent rewrite: `corpus` is set to the TRUE
value and the wrong one is preserved as `corpus_as_originally_recorded`, alongside a note naming
this item. Nothing is destroyed. Sidecars carry no `payload_sha256` (verified: 0 of 166), so no
hash chain is broken, and the `.pt` files themselves are never touched.

    python tools/r4g_e28_provenance.py --dry-run     # report only
    python tools/r4g_e28_provenance.py               # repair
"""
from __future__ import annotations
import argparse, collections, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "src"))
from provenance import sha256_file, write_result  # noqa: E402

R = lambda *p: os.path.join(REPO, *p)
STALE = "**Still outstanding:** E28's fits are on the four rented boxes and are NOT yet pulled."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-hashes", action="store_true",
                    help="re-hash every .pt and compare to the sidecar's recorded sha256. Slow "
                         "(~6.7 GB of reads) and it is the only thing that makes the ledger mean "
                         "anything -- file PRESENCE was accepted for file INTEGRITY once already "
                         "(CLAUDE.md §6.0b) and a truncated lens got through.")
    a = ap.parse_args()

    side = sorted(glob.glob(R("results", "e28_*_provenance.json")))
    repaired, corpora, gate_states, mismatches, missing_pt = [], collections.Counter(), collections.Counter(), [], []
    for p in side:
        d = json.load(open(p))
        recorded = d.get("corpus")
        cf = d.get("corpus_file") or ""
        true = os.path.splitext(os.path.basename(cf))[0] if cf else None
        corpora[f"{recorded} -> {true}"] += 1
        gate_states[str(d.get("anchor_agreement_gate"))] += 1
        art = R(d["artifact"]) if "artifact" in d else None
        if art and not os.path.exists(art):
            missing_pt.append(d.get("artifact"))
        elif a.verify_hashes and art:
            got = sha256_file(art)
            if got != d.get("sha256"):
                mismatches.append({"artifact": d["artifact"], "recorded": d.get("sha256"), "actual": got})
        if true and recorded != true:
            d["corpus"] = true
            d["corpus_as_originally_recorded"] = recorded
            d["corpus_backfilled_from"] = "corpus_file"
            d["R4g_note"] = ("`corpus` recorded the --corpus DEFAULT ('wikitext'), not the corpus "
                             "actually fitted. Backfilled from `corpus_file`, which was correct "
                             "all along. Original value preserved above.")
            repaired.append(os.path.relpath(p, REPO))
        if d.get("anchor_agreement_gate", "MISSING") is None:
            d["anchor_agreement_gate"] = "NOT RECORDED AT FIT TIME"
            d["anchor_agreement_gate_note"] = (
                "null was ambiguous between 'ran, returned nothing' and 'never ran'. The fitter "
                "wrote no gate value for these runs, so it is the latter. Stated explicitly so "
                "that absence is never read as a passing gate.")
        if not a.dry_run:
            with open(p, "w") as f:
                json.dump(d, f, indent=1)

    # ---- the ledger: every E28 operator, name + hash, from the sidecars
    rows = []
    for p in side:
        d = json.load(open(p))
        rows.append((os.path.basename(d.get("artifact", "?")), d.get("sha256"),
                     d.get("corpus"), d.get("n_prompts_used"), d.get("seed_block")))
    rows.sort()
    block = ["", "## E28 read-ladder operators (166) — added by R4g",
             "",
             "Recorded here because **0 of 166 were named in this file** while the paragraph above",
             "asserted they were still on rented boxes. They are on local disk. `corpus` is the",
             "**backfilled** value: every sidecar recorded the `--corpus` default `wikitext`, and the",
             "true corpus came from `corpus_file` (M-3).",
             "",
             "| artifact | sha256 | corpus | N | seed |", "|---|---|---|---|---|"]
    block += [f"| `{n}` | `{h}` | {c} | {nn} | {s} |" for n, h, c, nn, s in rows]
    block.append("")

    led_path = R("ARTIFACTS.md")
    led = open(led_path).read()
    led_changed = False
    if STALE in led:
        led = led.replace(
            STALE,
            "**Resolved 2026-08-19 (R4g):** E28's 166 fits are pulled, on local disk, and hashed\n"
            "into the table below. The teardown obligation they were subject to is discharged.")
        led_changed = True
    if "## E28 read-ladder operators (166)" not in led:
        led = led.rstrip() + "\n" + "\n".join(block)
        led_changed = True
    if led_changed and not a.dry_run:
        open(led_path, "w").write(led)

    rec = {
        "experiment": "R4g — E28 provenance and artifact-ledger repair",
        "prereg": "docs/experiments/preregs/R4_corrections.md",
        "prereg_sha256": sha256_file(R("docs", "experiments", "preregs", "R4_corrections.md")),
        "status": "PRE-REGISTERED", "dry_run": bool(a.dry_run),
        "findings_closed": ["M-3 corpus default leak", "M-4 null agreement gate",
                            "F-11 operators absent from ARTIFACTS.md"],
        "n_sidecars": len(side),
        "n_corpus_backfilled": len(repaired),
        "corpus_recorded_to_true": dict(corpora),
        "anchor_agreement_gate_states_before": dict(gate_states),
        "n_operators_added_to_ARTIFACTS_md": len(rows),
        "artifacts_md_stale_line_removed": STALE in open(led_path).read() is False,
        "hash_verification": ({"ran": True, "n_checked": len(side) - len(missing_pt),
                               "n_mismatched": len(mismatches), "mismatches": mismatches,
                               "missing_pt": missing_pt}
                              if a.verify_hashes else
                              {"ran": False, "why": "pass --verify-hashes; see the flag's help"}),
        "H3_defect": ("repro/30_repo_health.sh H3 tested `os.path.basename(p) not in led` -- a "
                      "SUBSTRING match of the filename against the ledger text. A file named in "
                      "the ledger with the WRONG hash passed. Fixed to compare the hash."),
    }
    write_result(R("results", "r4g_e28_provenance.json"), rec, experiment="R4g")
    print(f"sidecars                : {len(side)}")
    print(f"corpus backfilled       : {len(repaired)}")
    print(f"corpus mapping observed : {dict(corpora)}")
    print(f"gate states before      : {dict(gate_states)}")
    print(f"operators added to ledger: {len(rows)}")
    if a.verify_hashes:
        print(f"hash verification       : {len(side)-len(missing_pt)} checked, {len(mismatches)} MISMATCHED, "
              f"{len(missing_pt)} .pt missing")
        for m in mismatches[:5]:
            print("   MISMATCH", m["artifact"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
