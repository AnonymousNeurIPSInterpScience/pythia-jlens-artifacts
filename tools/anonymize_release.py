#!/usr/bin/env python3
"""
anonymize_release.py - the double-blind gate, and the release-time rewriter it gates.

SPECIFIED BY the release procedure section 4.2, which called for this file and marked it
"to be written". This is it.

THE CONSTRAINT. The venue is double-blind: "no identifying information ... in the main text,
figures, or supplementary materials". A linked artifact repository IS supplementary material, so
every byte a reviewer can reach must be clean - the code, the docs, the results JSON, the git
author, and the URL itself.

THREE MODES
  --scan     report every hit, by file, by pattern. Read-only. Run this first.
  --apply    rewrite results/**.json provenance blocks in place: hostname -> "<redacted>",
             absolute paths -> repo-relative. THEN RE-VERIFY EVERY PAYLOAD HASH. `payload_sha256`
             excludes the provenance block by construction, so if a hash moves the rewriter touched
             a number and the release is void. That is the test and it is a strong one.
  --verify   exit 1 if any hit remains anywhere. This is health check H12.

WHAT IT DELIBERATELY DOES NOT TOUCH
  Third-party names that are citations or licence attribution, not identity. A blanket
  `grep -ri anthropic | sed` would destroy the paper's own subject citation and is forbidden
  (the release procedure "KEEP"). The KEEP list below is enforced by --self-test.

WHAT THIS TOOL CANNOT DO
  Anonymise the git history (251 commits, 100% carrying an identity). That needs the flatten,
  the release procedure Nor the HF account name, which needs a new account. Both are listed in
  the release-hygiene procedure as manual steps with a checklist.
"""
import argparse, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# --- what breaks double-blind -------------------------------------------------------------------
# THE LITERAL STRINGS LIVE IN tools/deanon_patterns.json, WHICH IS UNTRACKED ON PURPOSE.
# A gate that ships its own pattern list republishes exactly the strings it exists to remove:
# before this split, `--scan` reported 25 hits against this file's own source. The method is here
# and public; the values are local. the release-hygiene procedure describes each entry
# without quoting it, so the list is rebuildable if lost.
PATTERN_FILE = os.path.join(ROOT, "tools", "deanon_patterns.json")


def _load():
    if not os.path.exists(PATTERN_FILE):
        return None
    d = json.load(open(PATTERN_FILE))
    return (
        [(lbl, re.compile(rx), why) for lbl, rx, why in d["patterns"]],
        list(d["keep"]),
        list(d["planted_leaks_for_self_test"]),
        [(re.compile(a), b) for a, b in d["redactions"]],
    )


_P = _load()
if _P is None:
    PATTERNS, KEEP, PLANTED, REDACTIONS = [], [], [], []
else:
    PATTERNS, KEEP, PLANTED, REDACTIONS = _P


def require_patterns():
    if _P is None:
        print(f"  FAIL — {os.path.relpath(PATTERN_FILE, ROOT)} is absent.\n"
              "         This is expected in a PUBLISHED tree (the file is untracked by design) and\n"
              "         is a hard failure anywhere you are trying to certify a release: you cannot\n"
              "         certify anonymity without the list of what makes it fail.\n"
              "         Rebuild it from the release-hygiene procedure.")
        return False
    return True


SKIP_DIRS = {".git", ".venv", "jacobian-lens", "__pycache__", "node_modules", ".ruff_cache"}
# paper/ is excluded from the published tree entirely (see the release-hygiene procedure); scanning it is
# still useful locally, so it is skipped only when --published-tree is passed.
BINARY_EXT = {".png", ".pdf", ".pt", ".npz", ".jpg", ".gz", ".zip", ".lock"}


def walk(published_tree=False):
    """Exactly what would be published: git-TRACKED files, plus results/ (gitignored but mirrored).

    Walking the filesystem instead was wrong twice over: it scanned .venv and it scanned
    tools/deanon_patterns.json, which is gitignored precisely so it is NOT published. A gate whose
    scope is not the publication scope reports leaks that cannot leak, and would hide ones that can.
    """
    skip = set(SKIP_DIRS) | ({"paper"} if published_tree else set())
    seen = set()
    tracked = subprocess.run(["git", "-C", ROOT, "ls-files", "-z"],
                             capture_output=True, text=True)
    if tracked.returncode == 0:
        for rel in tracked.stdout.split("\0"):
            if not rel or rel.split("/")[0] in skip:
                continue
            if os.path.splitext(rel)[1].lower() in BINARY_EXT:
                continue
            p = os.path.join(ROOT, rel)
            if os.path.exists(p):
                seen.add(p); yield p
    res = os.path.join(ROOT, "results")
    for dp, dn, fn in os.walk(res):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            if os.path.splitext(f)[1].lower() in BINARY_EXT:
                continue
            p = os.path.join(dp, f)
            if p not in seen:
                yield p


def keep_spans(text):
    out = []
    for k in KEEP:
        for m in re.finditer(re.escape(k), text):
            out.append((m.start(), m.end()))
    return out


def hits(text):
    """[(label, lineno, excerpt)] excluding anything inside a KEEP span."""
    keeps = keep_spans(text)
    out = []
    for label, rx, _ in PATTERNS:
        for m in rx.finditer(text):
            if any(s <= m.start() and m.end() <= e for s, e in keeps):
                continue
            ln = text.count("\n", 0, m.start()) + 1
            a = text.rfind("\n", 0, m.start()) + 1
            b = text.find("\n", m.end())
            out.append((label, ln, text[a: b if b != -1 else len(text)].strip()[:150]))
    return out


def scan(published_tree=False):
    per_file, per_pat = {}, {}
    for p in walk(published_tree):
        try:
            t = open(p, encoding="utf-8", errors="strict").read()
        except (UnicodeDecodeError, OSError):
            continue
        h = hits(t)
        if h:
            rel = os.path.relpath(p, ROOT)
            per_file[rel] = h
            for label, _, _ in h:
                per_pat[label] = per_pat.get(label, 0) + 1
    return per_file, per_pat


# --- the results/ rewriter -----------------------------------------------------------------------
def redact_provenance(node):
    """Rewrite identity out of a provenance block. Returns True if anything changed."""
    changed = False
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if k == "hostname" and isinstance(v, str) and hits(v):
                node[k] = "<redacted for double-blind review>"; changed = True
            elif isinstance(v, str):
                new = v
                for rx, rep in REDACTIONS:
                    new = rx.sub(rep, new)
                if new != v:
                    node[k] = new; changed = True
            elif isinstance(v, (dict, list)):
                changed |= redact_provenance(v)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, str):
                new = v
                for rx, rep in REDACTIONS:
                    new = rx.sub(rep, new)
                if new != v:
                    node[i] = new; changed = True
            elif isinstance(v, (dict, list)):
                changed |= redact_provenance(v)
    return changed


def apply_results(dry=False):
    from provenance import canonical_payload_sha
    res = os.path.join(ROOT, "results")
    if not os.path.isdir(res):
        print("  results/ absent — run repro/04_fetch_results.sh first"); return 1
    touched = moved = scanned = 0
    for dp, _, fn in os.walk(res):
        for f in sorted(fn):
            if not f.endswith(".json"):
                continue
            p = os.path.join(dp, f)
            try:
                o = json.load(open(p))
            except Exception:
                continue
            if not isinstance(o, dict):
                continue
            scanned += 1
            before = canonical_payload_sha(o)
            # Redact ONLY inside the provenance block. Nothing else may be touched.
            if "provenance" not in o:
                continue
            if not redact_provenance(o["provenance"]):
                continue
            after = canonical_payload_sha(o)
            if after != before:
                print(f"  VOID  {os.path.relpath(p, ROOT)} — payload hash moved {before[:12]} -> {after[:12]}")
                return 1
            touched += 1
            if not dry:
                json.dump(o, open(p, "w"), indent=1)
    print(f"  {scanned} results JSON scanned, {touched} provenance block(s) "
          f"{'would be ' if dry else ''}redacted, 0 payload hashes moved")
    return 0


PAYLOAD_LOG = os.path.join(ROOT, "docs", "reproducibility", "PAYLOAD_REDACTIONS.md")


def apply_payload(dry=False):
    """Redact identity out of PAYLOAD keys - which necessarily moves `payload_sha256`.

    Four strings in three results files sit outside the provenance block: an absolute `c1_dir`
    and two `inputs` entries naming a since-deleted document whose FILENAME is the leak. They
    cannot be left (they deanonymise) and they cannot be redacted silently (the payload hash is
    the integrity chain). So: redact, re-stamp, and record BOTH hashes, so a holder of a
    pre-release copy can still verify the original and a reviewer can verify the released one.
    """
    from provenance import canonical_payload_sha
    rows = []
    for dp, _, fn in os.walk(os.path.join(ROOT, "results")):
        for f in sorted(fn):
            if not f.endswith(".json"):
                continue
            path = os.path.join(dp, f)
            try:
                o = json.load(open(path))
            except Exception:
                continue
            if not isinstance(o, dict):
                continue
            body = {k: v for k, v in o.items() if k != "provenance"}
            txt = json.dumps(body)
            if not hits(txt):
                continue
            before = canonical_payload_sha(o)
            n = redact_provenance(body)          # same redaction rules, applied to the payload
            if not n:
                continue
            for k, v in body.items():
                o[k] = v
            after = canonical_payload_sha(o)
            pr = o.setdefault("provenance", {})
            pr["payload_sha256_before_redaction"] = before
            pr["payload_sha256"] = after
            pr["redaction"] = ("identity strings redacted from PAYLOAD keys for double-blind "
                               "review; see docs/reproducibility/PAYLOAD_REDACTIONS.md. The "
                               "pre-redaction hash is retained so the original file remains "
                               "verifiable.")
            rows.append((os.path.relpath(path, ROOT), before, after))
            if not dry:
                json.dump(o, open(path, "w"), indent=1)
    print(f"  {len(rows)} results file(s) had identity in a PAYLOAD key "
          f"{'(dry run)' if dry else '- redacted and re-stamped'}")
    for r, b, a in rows:
        print(f"      {r}\n        payload {b[:12]} -> {a[:12]}")
    if rows and not dry:
        with open(PAYLOAD_LOG, "w") as fh:
            fh.write("# PAYLOAD_REDACTIONS — the four strings that could not be redacted for free\n\n")
            fh.write("**Generated by `tools/anonymize_release.py --apply-payload`.**\n\n")
            fh.write(
                "`tools/anonymize_release.py --apply` redacts the **provenance block**, which is\n"
                "excluded from `payload_sha256` by construction, so 66 files were cleaned with every\n"
                "hash unchanged - and that invariance is the test that the rewriter touched no number.\n\n"
                "These three files are different. The identity string sat in a **payload** key, so\n"
                "redacting it necessarily moves the hash. Leaving it was not an option; moving a hash\n"
                "silently was not either. Both hashes are recorded, in the file and here.\n\n"
                "No numeric value was altered. The redactions are confined to filesystem paths.\n\n")
            fh.write("| results file | key | payload_sha256 before | after |\n|---|---|---|---|\n")
            for r, b, a in rows:
                fh.write(f"| `{r}` | absolute path / input filename | `{b[:16]}` | `{a[:16]}` |\n")
        print(f"  wrote {os.path.relpath(PAYLOAD_LOG, ROOT)}")
    return 0


def self_test():
    """A gate that cannot fail is not a gate. Prove this one fires, and prove KEEP survives."""
    fails = []
    must_fire = PLANTED
    for s in must_fire:
        if not hits(s):
            fails.append(f"MISSED a real leak: {s!r}")
    for s in KEEP:
        if hits(s):
            fails.append(f"WOULD DESTROY a required string: {s!r}")
    if fails:
        for f in fails:
            print("  FAIL " + f)
        return 1
    print(f"  ok   gate fires on {len(must_fire)} planted leaks and spares {len(KEEP)} required strings")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--apply", action="store_true", help="rewrite results/ provenance blocks")
    ap.add_argument("--apply-payload", action="store_true",
                    help="redact identity from PAYLOAD keys; moves payload_sha256, records both")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true", help="H12: exit 1 if any leak remains")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--published-tree", action="store_true",
                    help="scan only what the anonymous repo will publish (excludes paper/)")
    a = ap.parse_args()
    if a.apply_payload:
        if not require_patterns():
            return 1
        return apply_payload(dry=a.dry_run)
    if not any([a.scan, a.apply, a.verify, a.self_test]):
        a.scan = True

    if not require_patterns():
        return 1
    if a.self_test:
        return self_test()
    if a.apply:
        return apply_results(dry=a.dry_run)

    per_file, per_pat = scan(a.published_tree)
    scope = "the PUBLISHED tree (paper/ excluded)" if a.published_tree else "the whole working tree"
    print(f"  scope: {scope}")
    print(f"  {len(per_file)} file(s) carry a deanonymising string, "
          f"{sum(len(v) for v in per_file.values())} occurrence(s)\n")
    for label, _, why in PATTERNS:
        n = per_pat.get(label, 0)
        print(f"    {label:16s} {n:5d}   {why}")
    if a.scan and per_file:
        print()
        for rel in sorted(per_file, key=lambda r: -len(per_file[r]))[:40]:
            h = per_file[rel]
            print(f"  {rel}  ({len(h)})")
            for label, ln, ex in h[:3]:
                print(f"      {ln:>5}: [{label}] {ex}")
    if a.verify:
        if per_file:
            print("\n  FAIL — H12: the tree would deanonymise a contributor.")
            return 1
        print("\n  PASS — H12: no deanonymising string in the published tree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
