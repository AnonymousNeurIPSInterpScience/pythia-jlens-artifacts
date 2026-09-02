#!/usr/bin/env python3
"""
registration_ledger.py - freeze the "registered before it ran" evidence before the flatten eats it.

THE PROBLEM
    CLAUDE.md section 5: "The pre-registration is committed in its own commit, BEFORE the code that
    computes the result." The paper leans on that ordering ("registered before it ran" is part of
    the Tier A definition). The evidence for it is git commit order and nothing else.

    the release procedure section 5 publishes ONE squashed commit with no history.
    Section 5.1 already names the cost: "Flattening destroys the git-timestamp evidence for
    pre-registration." This tool pays that cost by materialising the evidence into a tracked file
    while the history is still there, exactly as tools/script_provenance.py does for code.

WHAT IT RECORDS, per pre-registration
    - the commit that ADDED the prereg, and its author date
    - the commit that last touched the SCRIPT the prereg's results name, and its date
    - ORDER: REGISTERED-FIRST / SAME-COMMIT / CODE-FIRST / UNVERIFIABLE
    - the prereg's SHA-256, so the frozen text is pinned

    CODE-FIRST is not automatically a violation: a prereg may name a script that already existed
    for another experiment. The ledger reports the fact; adjudication stays with the operator.

USAGE
    .venv/bin/python tools/registration_ledger.py            # write docs/experiments/REGISTRATION_LEDGER.md
    .venv/bin/python tools/registration_ledger.py --report   # print only
"""
import argparse, hashlib, json, os, subprocess, sys, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREGS = os.path.join(ROOT, "docs", "experiments", "preregs")
OUT = os.path.join(ROOT, "docs", "experiments", "REGISTRATION_LEDGER.md")


def git(*a):
    r = subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def added(path):
    """(commit, iso date) of the commit that introduced this path, following renames."""
    out = git("log", "--diff-filter=A", "--follow", "--format=%h\t%aI", "--", path)
    lines = [l for l in out.split("\n") if l.strip()]
    if not lines:
        out = git("log", "--follow", "--format=%h\t%aI", "--", path)
        lines = [l for l in out.split("\n") if l.strip()]
    if not lines:
        return None, None
    c, d = lines[-1].split("\t")
    return c, d


def last_touch(path):
    out = git("log", "-1", "--format=%h\t%aI", "--", path)
    if not out:
        return None, None
    c, d = out.split("\t")
    return c, d


def results_for(prereg_rel):
    """Results files whose `prereg` field names this document (by basename, paths having drifted)."""
    base = os.path.basename(prereg_rel)
    hits = []
    for p in glob.glob(os.path.join(ROOT, "results", "**", "*.json"), recursive=True):
        if "_provenance" in p:
            continue
        try:
            o = json.load(open(p))
        except Exception:
            continue
        if not isinstance(o, dict):
            continue
        pr = str(o.get("prereg", ""))
        names = [pr] + [str(x.get("path", "")) for x in (o.get("prereg_files") or [])
                        if isinstance(x, dict)]
        if any(base in n for n in names):
            sc = (o.get("provenance") or {}).get("script") or {}
            hits.append((os.path.relpath(p, ROOT), sc.get("path")))
    return hits


_BLOBS = None


def md_blobs():
    """sha256(content) -> (git blob id, last path seen) over EVERY markdown blob in history.

    Resolving by PATH fails here: the 2026-08-22 repath renamed specs/ -> docs/experiments/ and
    edited the files in the same commit, which defeats git's rename detection and made ten
    pre-registrations look unregistered. Content addressing does not have that failure mode.
    """
    global _BLOBS
    if _BLOBS is not None:
        return _BLOBS
    _BLOBS = {}
    out = git("rev-list", "--objects", "--all")
    for line in out.split("\n"):
        parts = line.split(" ", 1)
        if len(parts) != 2 or not parts[1].endswith(".md"):
            continue
        blob = subprocess.run(["git", "-C", ROOT, "cat-file", "-p", parts[0]], capture_output=True)
        if blob.returncode == 0:
            _BLOBS.setdefault(hashlib.sha256(blob.stdout).hexdigest(), (parts[0], parts[1]))
    return _BLOBS


def commit_introducing(path):
    out = git("log", "--diff-filter=A", "--format=%h\t%aI", "--all", "--", path)
    lines = [l for l in out.split("\n") if l.strip()]
    return lines[-1].split("\t") if lines else (None, None)


def freeze_as_registered(write=True):
    """Extract the exact registered bytes of every prereg a results file hash-pins.

    Same argument as tools/script_provenance.py: after the flatten there is no history, so the
    frozen text has to be IN the published tree or the freeze claim is unverifiable.
    """
    vault = os.path.join(ROOT, "docs", "experiments", "preregs", "as_registered")
    blobs = md_blobs()
    rows, missing = [], []
    for raw, sha, n in recorded_prereg_hashes():
        base = os.path.basename(raw)
        live = os.path.join("docs", "experiments", "preregs", base)
        cur = None
        if os.path.exists(os.path.join(ROOT, live)):
            cur = hashlib.sha256(open(os.path.join(ROOT, live), "rb").read()).hexdigest()
        if cur == sha:
            rows.append((raw, base, sha, "unchanged on disk", None, None)); continue
        vf = os.path.join(vault, f"{sha[:12]}__{base}")
        if os.path.exists(vf) and hashlib.sha256(open(vf, "rb").read()).hexdigest() == sha:
            c, d = commit_introducing(raw)
            rows.append((raw, base, sha, f"docs/experiments/preregs/as_registered/{sha[:12]}__{base}", c, d))
            continue
        hit = blobs.get(sha)
        if not hit:
            missing.append((raw, sha)); rows.append((raw, base, sha, "UNOBTAINABLE", None, None)); continue
        body = subprocess.run(["git", "-C", ROOT, "cat-file", "-p", hit[0]], capture_output=True).stdout
        if write:
            os.makedirs(vault, exist_ok=True)
            open(vf, "wb").write(body)
        c, d = commit_introducing(hit[1])
        rows.append((raw, base, sha, f"docs/experiments/preregs/as_registered/{sha[:12]}__{base}", c, d))
    return rows, missing


def recorded_prereg_hashes():
    """(recorded path, sha256, n_results) for every prereg a results file hash-pins."""
    seen = {}
    for p in glob.glob(os.path.join(ROOT, "results", "**", "*.json"), recursive=True):
        if "_provenance" in p:
            continue
        try:
            o = json.load(open(p))
        except Exception:
            continue
        if not isinstance(o, dict):
            continue
        pairs = []
        if isinstance(o.get("prereg"), str) and o.get("prereg_sha256"):
            pairs.append((o["prereg"].split(",")[0].split()[0], o["prereg_sha256"]))
        for pf in (o.get("prereg_files") or []):
            if isinstance(pf, dict) and pf.get("path") and pf.get("sha256"):
                pairs.append((pf["path"], pf["sha256"]))
        for raw, sha in pairs:
            k = (raw, sha)
            seen[k] = seen.get(k, 0) + 1
    return [(r, s, n) for (r, s), n in sorted(seen.items())]


def live_preregs():
    """basename (lowercased) -> repo-relative path, over every prereg location that exists."""
    out = {}
    for d in ("docs/experiments/preregs",):
        for f in glob.glob(os.path.join(ROOT, d, "**", "*.md"), recursive=True):
            out.setdefault(os.path.basename(f).lower(), os.path.relpath(f, ROOT))
    return out


def prereg_resolution():
    """Every prereg string a results file records, resolved against the tree as it stands.

    The `prereg`, `prereg_sha256` and `prereg_files` fields are PAYLOAD keys, not provenance, so
    they cannot be repathed without moving `payload_sha256` and breaking the integrity chain.
    The resolution therefore lives here, as a table, and the results files are left untouched.
    """
    live = live_preregs()
    rows = {}
    for p in glob.glob(os.path.join(ROOT, "results", "**", "*.json"), recursive=True):
        if "_provenance" in p:
            continue
        try:
            o = json.load(open(p))
        except Exception:
            continue
        if not isinstance(o, dict):
            continue
        recorded = []
        if isinstance(o.get("prereg"), str):
            recorded.append((o["prereg"], o.get("prereg_sha256")))
        for pf in (o.get("prereg_files") or []):
            if isinstance(pf, dict) and pf.get("path"):
                recorded.append((pf["path"], pf.get("sha256")))
        for raw, sha in recorded:
            # the recorded string is often prose: "specs/.../X.md, written before this ran"
            cand = None
            for tok in raw.replace(",", " ").split():
                b = os.path.basename(tok).lower()
                if b.endswith(".md") and b in live:
                    cand = live[b]; break
            if cand is None and not raw.lower().endswith(".md"):
                pass
            e = rows.setdefault(raw, {"raw": raw, "resolves_to": cand, "sha": sha,
                                      "n": 0, "sha_state": "—"})
            e["n"] += 1
            if cand and sha:
                cur = hashlib.sha256(open(os.path.join(ROOT, cand), "rb").read()).hexdigest()
                e["sha_state"] = "matches" if cur == sha else "**MOVED since the run**"
    return sorted(rows.values(), key=lambda r: (-r["n"], r["raw"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    rows = []
    files = sorted(glob.glob(os.path.join(PREREGS, "**", "*.md"), recursive=True))
    for f in files:
        rel = os.path.relpath(f, ROOT)
        if os.path.basename(f) == "README.md":
            continue
        sha = hashlib.sha256(open(f, "rb").read()).hexdigest()
        pc, pd = added(rel)
        res = results_for(rel)
        scripts = sorted({s for _, s in res if s})
        sc, sd = (None, None)
        for s in scripts:
            c, d = added(s)
            if d and (sd is None or d < sd):
                sc, sd = c, d
        if not pd:
            order = "UNVERIFIABLE (prereg not in history)"
        elif not sd:
            order = "NO SCRIPT LINKED" if not scripts else "UNVERIFIABLE (script not in history)"
        elif pd < sd:
            order = "REGISTERED-FIRST"
        elif pc == sc:
            order = "SAME-COMMIT"
        else:
            order = "CODE-FIRST"
        rows.append(dict(prereg=rel, sha=sha, prereg_commit=pc, prereg_date=(pd or "")[:10],
                         scripts=scripts, script_commit=sc, script_date=(sd or "")[:10],
                         order=order, n_results=len(res)))

    tally = {}
    for r in rows:
        tally[r["order"]] = tally.get(r["order"], 0) + 1
    print(f"  {len(rows)} pre-registrations")
    for k in sorted(tally):
        print(f"    {k:38s} {tally[k]}")

    if a.report:
        return 0
    with open(OUT, "w") as fh:
        fh.write("# REGISTRATION_LEDGER — the pre-registration ordering evidence, frozen\n\n")
        fh.write("**Generated by `tools/registration_ledger.py`. Do not hand-edit.**\n\n")
        fh.write(
            "`CLAUDE.md` section 5 requires each pre-registration to be committed in its own commit,\n"
            "*before* the code that computes the result, and the paper's Tier A definition depends on\n"
            "that ordering. The only evidence for it is git commit order.\n\n"
            "The released tree is a single squashed commit with no history\n"
            "(`the release procedure` section 5), which destroys that evidence. This file is the\n"
            "evidence, computed while the history existed, so the claim remains checkable afterwards.\n\n"
            "`CODE-FIRST` is reported, not judged: a pre-registration may legitimately name a script\n"
            "that already existed for an earlier experiment. Adjudication stays with the operator.\n\n")
        for k in sorted(tally):
            fh.write(f"- **{k}**: {tally[k]}\n")
        res = prereg_resolution()
        unres = [r for r in res if not r["resolves_to"] and r["raw"].lower().rstrip().endswith(".md") or
                 (not r["resolves_to"] and ".md" in r["raw"])]
        moved = [r for r in res if "MOVED" in r["sha_state"]]
        fh.write(f"\n## Pre-registration pointers recorded in results files\n\n")
        fh.write(
            "`prereg`, `prereg_sha256` and `prereg_files` are **payload** keys, so repathing them\n"
            "would move `payload_sha256` and break the integrity chain every other check depends on.\n"
            "The results files are therefore left exactly as written, and the drift is resolved here.\n\n"
            f"- distinct recorded pointers: **{len(res)}**\n"
            f"- resolve to a live pre-registration: **{sum(1 for r in res if r['resolves_to'])}**\n"
            f"- do not resolve: **{sum(1 for r in res if not r['resolves_to'])}** "
            "(mostly `in-file docstring` and `PLAN.tex section 6`, which name no separate document)\n"
            f"- recorded SHA-256 no longer matches the live file: **{len(moved)}**\n\n")
        fh.write("| recorded in the results file | resolves to | recorded sha | uses |\n|---|---|---|---:|\n")
        for r in res:
            fh.write(f"| `{r['raw'][:78]}` | {('`'+r['resolves_to']+'`') if r['resolves_to'] else '— none —'} "
                     f"| {r['sha_state']} | {r['n']} |\n")
        fh.write("\n| pre-registration | sha256 | registered | producing script first added | order | results |\n")
        fh.write("|---|---|---|---|---|---:|\n")
        for r in rows:
            s = "<br>".join(f"`{x}`" for x in r["scripts"]) or "—"
            fh.write(f"| `{r['prereg']}` | `{r['sha'][:12]}` | `{r['prereg_commit'] or '—'}` "
                     f"{r['prereg_date']} | {s} `{r['script_commit'] or '—'}` {r['script_date']} "
                     f"| **{r['order']}** | {r['n_results']} |\n")
    frozen, missing = freeze_as_registered(write=not a.report)
    n_ok = sum(1 for r in frozen if r[3] != "UNOBTAINABLE")
    print(f"  {len(frozen)} hash-pinned pre-registrations, {n_ok} obtainable, {len(missing)} not")
    with open(OUT, "a") as fh:
        fh.write("\n## The registered text, frozen\n\n")
        fh.write(
            "A results file records `prereg_sha256`: the hash of the pre-registration **as it stood\n"
            "when the experiment ran**. Ten of those no longer match the file on disk, because the\n"
            "2026-08-22 repath renamed `specs/` to `docs/experiments/` and rewrote the paths inside.\n"
            "The registered text is not lost - it is in history, findable by content hash rather than\n"
            "by path - and it is extracted here so it survives the single-commit release.\n\n"
            f"- hash-pinned pre-registrations: **{len(frozen)}**\n"
            f"- obtainable: **{n_ok}** · unobtainable: **{len(missing)}**\n\n")
        fh.write("| recorded path | recorded sha256 | the registered bytes are at | registering commit |\n|---|---|---|---|\n")
        for raw, base, sha, where, c, d in frozen:
            w = where if where in ("unchanged on disk", "UNOBTAINABLE") else "`" + where + "`"
            fh.write(f"| `{raw}` | `{sha[:12]}` | {w} | `{c or '—'}` {(d or '')[:10]} |\n")
    print(f"  wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
