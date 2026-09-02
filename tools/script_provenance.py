#!/usr/bin/env python3
"""
script_provenance.py - close the "script hash differs" gap, and make the closure survive the flatten.

THE PROBLEM THIS EXISTS FOR
    `tools/migrate_provenance_paths.py --verify` reports that 48 of 73 stamped results files record
    a `provenance.script.sha256` that no longer matches the script on disk. That is not corruption:
    the code evolved after the result was produced, which is normal. But it means a reviewer holding
    the results file cannot obtain the code that produced it, and Tier A's own definition is
    "recomputes from stored artifacts".

    47 of those 48 ARE recoverable today, because the exact blob is still in git history. The 48th
    was produced by uncommitted code and is gone.

THE FLATTEN DESTROYS THIS
    the release procedure section 5 publishes a SINGLE SQUASHED COMMIT with no history.
    After that step `git show <commit>:<path>` returns nothing and all 47 become unrecoverable.
    So this tool must run BEFORE the flatten. It extracts every historical blob into
    repro/scripts_at_run/, which is a tracked directory and therefore rides into the single commit.

WHAT IT PRODUCES
    repro/scripts_at_run/<sha12>__<basename>            the exact bytes that produced each result
    docs/reproducibility/SCRIPT_PROVENANCE.md           results file -> script -> sha -> where to find it

USAGE
    .venv/bin/python tools/script_provenance.py            # resolve, extract, write the table
    .venv/bin/python tools/script_provenance.py --verify   # exit 1 if any recorded hash is unobtainable
    .venv/bin/python tools/script_provenance.py --report   # print, write nothing

    --verify is the gate. It does NOT consult git, so it keeps working after the flatten: it asks
    only "is every recorded script hash obtainable from the published tree?"

WHAT IT DOES NOT PROMISE
    That re-running the extracted script reproduces the stored payload. It promises that the code,
    the argv and the input hashes are all available. Bit-reproducibility is a separate claim and
    this programme does not make it (repro/exp/README.md).
"""
import argparse, hashlib, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
VAULT = os.path.join(ROOT, "repro", "scripts_at_run")
EXCEPT = os.path.join(VAULT, "EXCEPTIONS.json")
TABLE = os.path.join(ROOT, "docs", "reproducibility", "SCRIPT_PROVENANCE.md")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def declared_exceptions():
    """(results_file, sha12) pairs whose producing code is admitted to be gone, with a reason."""
    if not os.path.exists(EXCEPT):
        return {}
    d = json.load(open(EXCEPT))
    return {(e["results_file"], e["sha256"][:12]): e for e in d.get("exceptions", [])}


def stamped():
    """Every results file carrying provenance.script.{path,sha256}."""
    out = []
    for dp, _, fn in os.walk(RES):
        for f in sorted(fn):
            if not f.endswith(".json") or f.endswith("_provenance.json"):
                continue
            p = os.path.join(dp, f)
            try:
                o = json.load(open(p))
            except Exception:
                continue
            if not isinstance(o, dict):
                continue
            sc = (o.get("provenance") or {}).get("script") or {}
            if sc.get("path") and sc.get("sha256"):
                out.append((os.path.relpath(p, ROOT), sc["path"], sc["sha256"]))
    return sorted(out)


def vault_name(path, h):
    return f"{h[:12]}__{os.path.basename(path)}"


def from_git(path, h):
    """Return the blob bytes whose sha256 is h, searching this file's git history."""
    log = subprocess.run(["git", "-C", ROOT, "log", "--format=%H", "--follow", "--", path],
                         capture_output=True, text=True)
    if log.returncode != 0:
        return None, None
    for c in log.stdout.split():
        b = subprocess.run(["git", "-C", ROOT, "show", f"{c}:{path}"], capture_output=True)
        if b.returncode == 0 and sha(b.stdout) == h:
            return b.stdout, c[:12]
    return None, None


def resolve(path, h, use_git):
    """(bytes, source) for the recorded hash, or (None, None). Never consults git under --verify."""
    disk = os.path.join(ROOT, path)
    if os.path.exists(disk):
        b = open(disk, "rb").read()
        if sha(b) == h:
            return b, "on disk, unchanged"
    v = os.path.join(VAULT, vault_name(path, h))
    if os.path.exists(v):
        b = open(v, "rb").read()
        if sha(b) == h:
            return b, "repro/scripts_at_run/" + vault_name(path, h)
    if use_git:
        b, c = from_git(path, h)
        if b is not None:
            return b, "git " + c
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="gate mode: no git, no writes; exit 1 if any hash is unobtainable")
    ap.add_argument("--report", action="store_true", help="print only, write nothing")
    a = ap.parse_args()
    use_git = not a.verify
    write = not (a.verify or a.report)

    rows, extracted, lost = [], 0, []
    for res, path, h in stamped():
        b, src = resolve(path, h, use_git)
        if b is None:
            lost.append((res, path, h))
            rows.append((res, path, h, "UNOBTAINABLE"))
            continue
        if src.startswith("git ") and write:
            os.makedirs(VAULT, exist_ok=True)
            open(os.path.join(VAULT, vault_name(path, h)), "wb").write(b)
            extracted += 1
            src = f"repro/scripts_at_run/{vault_name(path, h)} (extracted from {src})"
        rows.append((res, path, h, src))

    exc = declared_exceptions()
    undeclared = [(r, p_, h) for r, p_, h in lost if (r, h[:12]) not in exc]
    unchanged = sum(1 for r in rows if r[3] == "on disk, unchanged")
    print(f"  {len(rows)} stamped results files")
    print(f"  {unchanged} whose script is unchanged on disk")
    print(f"  {len(rows) - unchanged - len(lost)} resolved to a historical blob")
    if write:
        print(f"  {extracted} blob(s) extracted into repro/scripts_at_run/")
    print(f"  {len(lost)} unobtainable "
          f"({len(lost) - len(undeclared)} declared in EXCEPTIONS.json, {len(undeclared)} UNDECLARED)")
    for res, path, h in lost:
        tag = "declared" if (res, h[:12]) in exc else "UNDECLARED"
        print(f"      [{tag}] {res}  needs {path} @ {h[:12]}")

    if write:
        os.makedirs(os.path.dirname(TABLE), exist_ok=True)
        with open(TABLE, "w") as fh:
            fh.write("# SCRIPT_PROVENANCE — the code that produced each results file\n\n")
            fh.write("**Generated by `tools/script_provenance.py`. Do not hand-edit.**\n\n")
            fh.write(
                "A results file records the SHA-256 of the script that wrote it. Code evolves, so\n"
                "that hash often no longer matches the script on disk — which is honest, and useless\n"
                "to a reviewer on its own. This table resolves every recorded hash to bytes that are\n"
                "present in this repository, so the producing code is obtainable without git history.\n"
                "That matters because the released tree is a single squashed commit and has none.\n\n"
                "Verify with `.venv/bin/python tools/script_provenance.py --verify` (no git, no network).\n\n")
            fh.write(f"- stamped results files: **{len(rows)}**\n")
            fh.write(f"- script unchanged on disk: **{unchanged}**\n")
            fh.write(f"- resolved to `repro/scripts_at_run/`: **{len(rows) - unchanged - len(lost)}**\n")
            fh.write(f"- unobtainable: **{len(lost)}**\n\n")
            if lost:
                fh.write("## Unobtainable — produced by code that was never committed\n\n")
                fh.write("| results file | script | recorded sha256 | what stands in its place |\n|---|---|---|---|\n")
                for res, path, h in lost:
                    e = exc.get((res, h[:12]))
                    note = e["what_stands_in_its_place"] if e else "**UNDECLARED**"
                    fh.write(f"| `{res}` | `{path}` | `{h[:16]}` | {note} |\n")
                fh.write("\nThese ran with `provenance.git.dirty = true` and the working-tree diff was\n"
                         "never committed. The result stands as a stored number; its exact code does not.\n"
                         "Declared in `repro/scripts_at_run/EXCEPTIONS.json`, which the gate reads.\n\n")
            fh.write("## Every stamped result\n\n")
            fh.write("| results file | script | recorded sha256 | obtain from |\n|---|---|---|---|\n")
            for res, path, h, src in rows:
                fh.write(f"| `{res}` | `{path}` | `{h[:16]}` | {src} |\n")
        print(f"  wrote {os.path.relpath(TABLE, ROOT)}")

    if a.verify and undeclared:
        print("\n  FAIL — a recorded script hash is neither obtainable nor declared in "
              "repro/scripts_at_run/EXCEPTIONS.json.")
        return 1
    if a.verify:
        print("\n  PASS — every recorded script hash is obtainable from this tree, or declared lost with a reason.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
