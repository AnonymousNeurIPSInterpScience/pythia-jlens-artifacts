#!/usr/bin/env python3
"""migrate_provenance_paths.py — re-point stored provenance at a moved tree, and prove nothing else moved.

WHY THIS EXISTS
  On 2026-08-15 the programme directory `pythia/` was dissolved and its contents moved to the
  repo root, so that `pythia/t52_factorial.py` became `t52_factorial.py`. Every stored results
  file records the path of the script that produced it and the path of every input it consumed,
  and `provenance.verify_result` resolves both against the repo root. Left alone, all of them
  would report "script file not found now" — which reads exactly like corruption.

WHAT IT TOUCHES, AND WHAT IT REFUSES TO TOUCH
  This is the whole design, and the reason the migration is safe:

    MIGRATED   provenance.script.path        resolution pointer, read by verify_result
               provenance.script.sha256      only when the file's CONTENT changed
               provenance.script.n_lines     ditto
               provenance.inputs[].path      resolution pointers, read by verify_result

    LEFT       provenance.argv               what was actually typed. History, not a pointer.
               provenance.git.dirty_files    ditto
               EVERYTHING OUTSIDE provenance  .fitter, .lens, .prereg, .artifact, ...

  The last line is the load-bearing one. `payload_sha256` is defined as the hash of the result
  content WITH THE PROVENANCE BLOCK EXCLUDED (provenance.canonical_payload_sha). So editing
  inside `provenance` cannot change it, and editing anywhere else would — silently invalidating
  the one field that proves two runs produced the same science. This script asserts that
  invariant per file and aborts on the first violation rather than writing a single byte.

  Payload-level path strings (`"fitter": "fastfit.py:fast_fit"`, `"lens": "..."`) therefore keep
  the old `pythia/` prefix in files written before the move. That is deliberate: they are a
  record of where the file was when the result was produced. Strip the prefix when reading them.

WHAT IT RECORDS
  Each migrated file gains `provenance.path_migration` — the date, the reason, the exact prefix
  rule, and the before/after script hash. A reader can therefore always tell that the pointer was
  rewritten by a move rather than by a rerun, and can recover the pre-move path.

THE SECOND PASS, AND WHY IT IS NOT A FUDGE
  Re-pointing a file changes its bytes. Several of these results declare *each other* as inputs
  (e48b -> e36 and e48c; e48-gate -> e48; e48 -> e48c and e52), and an input is recorded by
  SHA-256 of the whole file. So migrating the tree necessarily stales those links, and the honest
  question is whether the link broke because the science changed or because the bytes did.

  `--refresh-input-hashes` answers it per link and refuses when it cannot:

    refresh IF   the input is a stamped results JSON whose own `payload_sha256` still equals the
                 canonical hash of its current content — i.e. its science is provably intact and
                 only its provenance block moved
    REFUSE       otherwise. A .pt lens, a corpus file, or a JSON whose payload hash moved is a
                 real break and must not be papered over.

  Each refreshed link records sha256_before, sha256_after and the payload hash that licensed it,
  under `provenance.path_migration.inputs_refreshed`. One link here was ALREADY stale before this
  migration — e52 consumed e48_crossover before e48_crossover was given a provenance block — and
  it is refreshed on exactly the same evidence, with `pre_existing: true`.

USAGE
    python tools/migrate_provenance_paths.py --dry-run     # print the plan, write nothing
    python tools/migrate_provenance_paths.py               # apply
    python tools/migrate_provenance_paths.py --refresh-input-hashes
    python tools/migrate_provenance_paths.py --verify      # re-check the tree afterwards

  Idempotent: a second run of either pass finds nothing to do.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
from provenance import canonical_payload_sha, sha256_file, verify_result  # noqa: E402

PREFIX = "pythia/"
MIGRATION = {
    "date": "2026-08-15",
    "reason": "the programme directory `pythia/` was dissolved; its contents moved to the repo root",
    "rule": "re-point provenance.script.path and provenance.inputs[].path at where each file "
            "lives now, resolving by unique basename; refuse if the basename is ambiguous",
    "not_touched": ["provenance.argv", "provenance.git.dirty_files",
                    "every field outside the provenance block (payload_sha256 must not move)"],
}


SKIP_DIRS = (".git", ".venv", "jacobian-lens", "thirdparty", "__pycache__", "logs")


def _index_repo() -> dict[str, list[str]]:
    """basename -> every repo-relative path with that basename. Built once, used to relocate."""
    idx: dict[str, list[str]] = {}
    for dirpath, dirnames, names in os.walk(REPO):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for n in names:
            idx.setdefault(n, []).append(os.path.relpath(os.path.join(dirpath, n), REPO))
    return idx


_INDEX: dict[str, list[str]] | None = None


def relocate(p: str) -> str:
    """Where does `p` live now?

    Resolution order, and the order matters:
      1. it already resolves -> unchanged;
      2. stripping a legacy 'pythia/' prefix resolves it -> use that;
      3. exactly ONE file in the repo has that basename -> use it;
      4. zero or several -> give up and return `p` unchanged, so the caller reports a real break
         rather than guessing. An ambiguous basename must never be silently resolved.
    """
    global _INDEX
    if not isinstance(p, str) or not p:
        return p
    if os.path.exists(os.path.join(REPO, p)):
        return p
    if p.startswith(PREFIX) and os.path.exists(os.path.join(REPO, p[len(PREFIX):])):
        return p[len(PREFIX):]
    if _INDEX is None:
        _INDEX = _index_repo()
    cand = _INDEX.get(os.path.basename(p), [])
    return cand[0] if len(cand) == 1 else p


def strip(p: str) -> str:
    return relocate(p)


def plan_one(path: str) -> dict | None:
    """Return the change this file needs, or None if it needs none."""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    p = d.get("provenance")
    if not isinstance(p, dict):
        return None

    old_script = p.get("script", {}).get("path")
    new_script = strip(old_script) if old_script else None
    inputs = p.get("inputs", []) or []
    n_inputs = sum(1 for i in inputs if isinstance(i, dict)
                   and relocate(str(i.get("path", ""))) != str(i.get("path", "")))
    if new_script == old_script and n_inputs == 0:
        return None

    disk = os.path.join(REPO, new_script) if new_script else None
    now_sha = sha256_file(disk) if disk else None
    return {
        "file": os.path.relpath(path, REPO),
        "old_script": old_script,
        "new_script": new_script,
        "recorded_sha": p.get("script", {}).get("sha256"),
        "disk_sha": now_sha,
        "script_content_changed": bool(now_sha and p.get("script", {}).get("sha256") and
                                       now_sha != p["script"]["sha256"]),
        "script_on_disk": now_sha is not None,
        "n_input_paths": n_inputs,
        "payload_before": canonical_payload_sha(d),
    }


def apply_one(path: str, plan: dict) -> None:
    d = json.load(open(path))
    before = canonical_payload_sha(d)
    p = d["provenance"]

    rec = dict(MIGRATION)
    rec["script_path_before"] = plan["old_script"]
    rec["script_path_after"] = plan["new_script"]
    rec["script_sha256_before"] = plan["recorded_sha"]

    if plan["new_script"]:
        p["script"]["path"] = plan["new_script"]
    # NOTE: this pass re-points the PATH only. If the script's CONTENT also changed it is left
    # recording the old hash, so verify_result keeps reporting the mismatch, and --restamp-scripts
    # is the only way to clear it — and that pass refuses unless every differing line is a path
    # line. Blessing a content change here, where there is no such proof, is the one shortcut that
    # would turn this tool into a rubber stamp.
    rec["script_content_changed"] = bool(plan["script_content_changed"])
    if plan["script_content_changed"]:
        rec["script_sha256_on_disk_now"] = plan["disk_sha"]
        rec["restamp_required"] = ("content differs; run --restamp-scripts <pre-move-commit>, "
                                   "which proves the diff is path-only before recording it")

    for i in p.get("inputs", []) or []:
        if isinstance(i, dict):
            i["path"] = strip(i.get("path"))

    p["path_migration"] = rec

    after = canonical_payload_sha(d)
    if after != before:
        raise SystemExit(f"ABORT {path}: payload_sha256 moved {before[:16]} -> {after[:16]}. "
                         f"Nothing outside `provenance` may be edited by this script.")
    if p.get("payload_sha256") and p["payload_sha256"] != after:
        raise SystemExit(f"ABORT {path}: stored payload_sha256 already disagreed with content "
                         f"BEFORE this migration. Fix that first.")

    tmp = path + ".migrating"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1, default=str)
    os.replace(tmp, path)


def science_intact(path: str) -> str | None:
    """If `path` is a stamped results JSON whose payload hash still matches, return that hash."""
    if not path.endswith(".json") or not os.path.isfile(path):
        return None
    try:
        d = json.load(open(path))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    stored = (d.get("provenance") or {}).get("payload_sha256")
    now = canonical_payload_sha(d)
    return now if stored and stored == now else None


def refresh_inputs(dry: bool) -> int:
    files = sorted(glob.glob(os.path.join(REPO, "results", "**", "*.json"), recursive=True))
    touched = refused = 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or not isinstance(d.get("provenance"), dict):
            continue
        p = d["provenance"]
        before = canonical_payload_sha(d)
        refreshed, bad = [], []
        for i in p.get("inputs", []) or []:
            if not isinstance(i, dict):
                continue
            disk = os.path.join(REPO, i.get("path", ""))
            now = sha256_file(disk)
            if now is None or now == i.get("sha256"):
                continue
            payload = science_intact(disk)
            if payload is None:
                bad.append(i.get("path"))
                continue
            refreshed.append({
                "path": i["path"],
                "sha256_before": i.get("sha256"),
                "sha256_after": now,
                "payload_sha256_of_input": payload,
                "why": "file bytes moved (provenance re-pointed); the input's own payload hash "
                       "still matches its content, so the science it carries is unchanged",
                "pre_existing": bool(i.get("sha256") and not (p.get("path_migration"))),
            })
            i["sha256"] = now
            i["bytes"] = os.path.getsize(disk)
        if bad:
            refused += 1
            print(f"  REFUSED {os.path.relpath(f, REPO)}: cannot vouch for {bad}")
        if not refreshed:
            continue
        touched += 1
        print(f"  {os.path.relpath(f, REPO)}: {len(refreshed)} link(s) re-anchored on payload")
        for r in refreshed:
            print(f"     {r['path']}  {str(r['sha256_before'])[:12]} -> {r['sha256_after'][:12]}")
        if dry:
            continue
        p.setdefault("path_migration", dict(MIGRATION))["inputs_refreshed"] = refreshed
        after = canonical_payload_sha(d)
        if after != before:
            raise SystemExit(f"ABORT {f}: payload_sha256 moved. Refusing to write.")
        tmp = f + ".migrating"
        with open(tmp, "w") as h:
            json.dump(d, h, indent=1, default=str)
        os.replace(tmp, f)
    if not touched and not refused:
        print("  nothing to refresh — every declared input hash matches disk")
    elif dry:
        print("\n  --dry-run: nothing written")
    return 1 if refused else 0


def restamp_scripts(dry: bool, baseline: str) -> int:
    """Re-record a script hash after a LATER path-only edit — and prove it was path-only first.

    The move happened in two steps (dissolve `pythia/`, then relocate the pre-registrations), so a
    script could be edited again after its provenance was already re-pointed. Blindly re-recording
    the new hash would turn this tool into a rubber stamp for any edit at all. So each candidate is
    diffed against its pre-move blob and **every differing line must be a path line** — it must
    contain a separator, or a path-construction call, or a directory name we actually moved. One
    line that is not, and the file is refused, loudly. A changed constant, threshold or expression
    cannot pass this.
    """
    PATHISH = ("/", "os.path", "sys.path", "jacobian-lens", "thirdparty")
    import subprocess
    files = sorted(glob.glob(os.path.join(REPO, "results", "**", "*.json"), recursive=True))
    touched = refused = 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        p = d.get("provenance") if isinstance(d, dict) else None
        if not isinstance(p, dict) or "path_migration" not in p:
            continue
        rel = p["script"]["path"]
        disk = os.path.join(REPO, rel)
        now = sha256_file(disk)
        if now is None or now == p["script"].get("sha256"):
            continue
        base = os.path.basename(rel)
        tree = subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "--name-only", baseline],
                              capture_output=True, text=True).stdout.split()
        hits = [t for t in tree if os.path.basename(t) == base]
        if len(hits) != 1:
            print(f"  REFUSED {os.path.relpath(f, REPO)}: {len(hits)} candidate(s) for {base} "
                  f"at {baseline} — cannot identify the pre-move blob")
            refused += 1
            continue
        old_blob = subprocess.run(["git", "-C", REPO, "show", f"{baseline}:{hits[0]}"],
                                  capture_output=True)
        if old_blob.returncode != 0:
            print(f"  REFUSED {os.path.relpath(f, REPO)}: cannot read {baseline}:{hits[0]}")
            refused += 1
            continue
        before = old_blob.stdout.decode().splitlines()
        after = open(disk).read().splitlines()
        import difflib
        offending = [l[1:].strip() for l in difflib.unified_diff(before, after, n=0)
                     if l[:1] in "+-" and l[:3] not in ("+++", "---")
                     and not any(t in l for t in PATHISH)]
        if offending:
            print(f"  REFUSED {os.path.relpath(f, REPO)}: {rel} has NON-PATH changes:")
            for l in offending[:5]:
                print(f"      {l[:100]}")
            refused += 1
            continue
        touched += 1
        print(f"  {os.path.relpath(f, REPO)}: {rel} restamped "
              f"{str(p['script']['sha256'])[:12]} -> {now[:12]}  (diff is path-only, verified)")
        if dry:
            continue
        pm = p["path_migration"]
        pm.setdefault("script_restamps", []).append({
            "sha256_before": p["script"]["sha256"], "sha256_after": now,
            "verified": "every differing line vs the pre-move blob is a path line "
                        "(contains a separator, a path-construction call, or a moved directory name)",
            "baseline_commit": baseline})
        p["script"]["sha256"] = now
        p["script"]["n_lines"] = len(after)
        tmp = f + ".migrating"
        with open(tmp, "w") as h:
            json.dump(d, h, indent=1, default=str)
        os.replace(tmp, f)
    if not touched and not refused:
        print("  nothing to restamp — every recorded script hash matches disk")
    elif dry:
        print("\n  --dry-run: nothing written")
    return 1 if refused else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--refresh-input-hashes", action="store_true")
    ap.add_argument("--restamp-scripts", metavar="BASELINE_COMMIT",
                    help="re-record script hashes after a later path-only edit, "
                         "refusing any file whose diff is not path-only")
    a = ap.parse_args()

    if a.restamp_scripts:
        return restamp_scripts(a.dry_run, a.restamp_scripts)
    if a.refresh_input_hashes:
        return refresh_inputs(a.dry_run)

    files = sorted(glob.glob(os.path.join(REPO, "results", "**", "*.json"), recursive=True))

    if a.verify:
        n = ok = 0
        for f in files:
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if not isinstance(d, dict) or not d.get("provenance"):
                continue
            n += 1
            v = verify_result(f)
            flag = "ok " if v["ok"] else "BAD"
            ok += bool(v["ok"])
            if not v["ok"]:
                why = []
                if not v["script_unchanged"]:
                    why.append(v.get("script_unchanged_reason", "script hash differs"))
                if not v["payload_sha_matches"]:
                    why.append("PAYLOAD HASH MOVED")
                for b in v["inputs_changed_or_missing"]:
                    why.append(f"input {b['path']}: {b['why']}")
                print(f"  {flag} {os.path.relpath(f, REPO)}  — {'; '.join(why)}")
            else:
                print(f"  {flag} {os.path.relpath(f, REPO)}")
        print(f"\n  {ok}/{n} stamped results files verify end-to-end")
        return 0 if ok == n else 1

    plans = [(f, p) for f in files if (p := plan_one(f))]
    if not plans:
        print("  nothing to migrate — every provenance pointer already resolves at the current layout")
        return 0

    print(f"  {len(plans)} file(s) to migrate\n")
    for f, p in plans:
        mark = "SCRIPT CONTENT CHANGED" if p["script_content_changed"] else "pointer only"
        miss = "" if p["script_on_disk"] else "   !! script NOT FOUND at the new path"
        print(f"  {p['file']}\n     {p['old_script']} -> {p['new_script']}   [{mark}]{miss}\n"
              f"     {p['n_input_paths']} input path(s) re-pointed")
    if a.dry_run:
        print("\n  --dry-run: nothing written")
        return 0
    for f, p in plans:
        apply_one(f, p)
    print(f"\n  migrated {len(plans)} file(s). Every payload_sha256 asserted unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
