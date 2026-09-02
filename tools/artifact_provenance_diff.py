#!/usr/bin/env python3
"""
artifact_provenance_diff.py — can this artifact still be reproduced, without refitting it?

THE IDEA. Refitting 381 operators to check they reproduce is expensive and mostly redundant. A
result records the SHA-256 of every input it consumed, the hash of the script that ran, and the
library versions it ran under. If all three still match what is on disk, then the inputs to the
computation are unchanged and the artifact is PREDICTED reproducible. If any differ, that is a
DELTA, and a delta is exactly where a refit is worth paying for.

WHAT THIS IS AND IS NOT. Provenance identity is NECESSARY, not SUFFICIENT. The programme has
measured both directions:

  * E66c matched a recorded environment (torch 2.11.0+cu128, TF32 off) and the operator refit to
    2.441e-04, inside one fp16 step. Recorded-and-matched provenance did predict reproduction.
  * E66b refit an operator whose provenance recorded NOTHING (argv null, env null: it was fitted
    two minutes before provenance.py existed) and got 1.0986e-03, above one fp16 step.

So this tool PARTITIONS artifacts by how much is knowable, and the partitions get different
treatment. It never claims an artifact reproduces; it says whether anything visible has changed.

  .venv/bin/python tools/artifact_provenance_diff.py
  .venv/bin/python tools/artifact_provenance_diff.py --json out.json
"""
import argparse, glob, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def sha(path):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    try:
        import torch
        cur_torch = torch.__version__
    except Exception:
        cur_torch = None

    rows, in_cache = [], {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "**", "*.json"), recursive=True)):
        try:
            o = json.load(open(p))
        except Exception:
            continue
        if not isinstance(o, dict):
            continue
        pr = o.get("provenance")
        rel = os.path.relpath(p, ROOT)
        if not pr:
            rows.append({"result": rel, "class": "NO_PROVENANCE", "detail":
                         "no provenance block; nothing to diff. Predates the provenance layer "
                         "(1a6f5df, 2026-08-14 20:05 PDT) or was written by a script that did "
                         "not stamp. Reproduction cannot be predicted; it must be tested."})
            continue

        deltas = []

        # 1. the SCRIPT that ran
        sc = pr.get("script") or {}
        spath, ssha = sc.get("path"), sc.get("sha256")
        if spath and ssha:
            live = sha(os.path.join(ROOT, spath))
            if live is None:
                vault = glob.glob(os.path.join(ROOT, "repro", "scripts_at_run",
                                               f"{ssha[:12]}__*"))
                deltas.append("script missing from tree" if not vault
                              else None)
                deltas = [d for d in deltas if d]
            elif live != ssha:
                vault = glob.glob(os.path.join(ROOT, "repro", "scripts_at_run", f"{ssha[:12]}__*"))
                deltas.append(f"script CHANGED since the run ({spath})"
                              + ("" if vault else "; original NOT in scripts_at_run"))

        # 2. the INPUTS it consumed -- this is the load-bearing check for .pt operators
        n_in = n_ok = 0
        for i in (pr.get("inputs") or []):
            ip, isha = i.get("path"), i.get("sha256")
            if not ip or not isha:
                continue
            n_in += 1
            full = os.path.join(ROOT, ip)
            if full not in in_cache:
                in_cache[full] = sha(full)
            live = in_cache[full]
            if live is None:
                deltas.append(f"input MISSING: {ip}")
            elif live != isha:
                deltas.append(f"input CHANGED: {ip}")
            else:
                n_ok += 1

        # 3. the ENVIRONMENT
        env = pr.get("env") or {}
        env_known = bool(env.get("torch"))
        if env_known and cur_torch and env.get("torch") != cur_torch:
            deltas.append(f"torch {env['torch']} -> {cur_torch}")
        if env_known and not env.get("threads"):
            deltas.append("thread count not recorded (added 2026-08-29; e60 showed it matters)")

        cls = ("DELTA" if deltas else
               ("PREDICTED_REPRODUCIBLE" if (n_in or spath) else "STAMP_ONLY"))
        rows.append({"result": rel, "class": cls, "n_inputs": n_in, "n_inputs_matching": n_ok,
                     "env_recorded": env_known, "deltas": deltas})

    from collections import Counter
    cnt = Counter(r["class"] for r in rows)
    print(f"  {len(rows)} results files\n")
    for k in ("PREDICTED_REPRODUCIBLE", "DELTA", "STAMP_ONLY", "NO_PROVENANCE"):
        if cnt.get(k):
            print(f"   {k:24s} {cnt[k]:4d}")

    dl = [r for r in rows if r["class"] == "DELTA"]
    if dl:
        print(f"\n  DELTAS — these are where a refit is worth paying for ({len(dl)}):")
        for r in dl[:25]:
            print(f"     {r['result']}")
            for d in r["deltas"][:3]:
                print(f"        - {d}")
        if len(dl) > 25:
            print(f"     ... and {len(dl)-25} more")

    tot_in = sum(r.get("n_inputs", 0) for r in rows)
    tot_ok = sum(r.get("n_inputs_matching", 0) for r in rows)
    print(f"\n  input-hash check: {tot_ok}/{tot_in} recorded input hashes still match on disk")

    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
        print(f"  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
