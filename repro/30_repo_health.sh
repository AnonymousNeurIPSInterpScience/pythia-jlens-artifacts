#!/usr/bin/env bash
# repro/30_repo_health.sh — the standing reproducibility audit. Free, read-only, ~20 s.
#
#   bash repro/30_repo_health.sh
#
# Ten checks, each one the residue of something that actually went wrong here. Run it
# before a handoff, before a submission, and after anything that moves files.

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
cd "$REPO_ROOT"
hdr "H1. The vendored anchor — the single point of failure"
info "jlens is imported 57 times. Every fit, read and write goes through it."
if [ -d "$VENDOR_DIR/.git" ]; then
  got="$(git -C "$VENDOR_DIR" rev-parse HEAD)"
  [ "$got" = "$VENDOR_PIN" ] && ok "pinned at ${VENDOR_PIN:0:7}" || bad "at ${got:0:7}, expected ${VENDOR_PIN:0:7}"
  [ -z "$(git -C "$VENDOR_DIR" status --porcelain)" ] && ok "unmodified" || bad "MODIFIED IN PLACE — forbidden"
else bad "absent — run repro/01_setup_local.sh"; fi

hdr "H2. No large binaries in git"
n_pt="$(git ls-files '*.pt' '*.pth' '*.safetensors' | wc -l | tr -d ' ' || true)"
[ "$n_pt" = "0" ] && ok "0 model binaries tracked" || bad "$n_pt tracked — they belong in the HF mirror"
big="$(git ls-files -z | xargs -0 du -k 2>/dev/null | awk '$1>5000{print $2" ("int($1/1024)"MB)"}' | head -5 || true)"
[ -z "$big" ] && ok "no tracked file over 5 MB" || { warn "large tracked files:"; echo "$big" | sed 's/^/      /'; }

hdr "H3. Every artifact hashed"
"$PY" - <<'EOF'
import glob, hashlib, json, os, re
# R4g. This test used to be `basename(p) not in ARTIFACTS.md` -- a SUBSTRING match of the
# filename against the ledger TEXT. A file named in the ledger with the WRONG hash passed, which
# is file PRESENCE accepted as file INTEGRITY, the exact substitution CLAUDE.md §6.0b forbids and
# the one that let a truncated lens through once. It now compares the HASH.
# RECURSIVE (fixed 2026-09-01). Operators live under results/lenses/**, results/e48/**,
# results/r6/** and more; the old non-recursive results/*.pt matched ZERO files, so this whole
# check silently passed on an empty list -- a control that cannot fail (CLAUDE.md rule 10).
pts = sorted(glob.glob("results/**/*.pt", recursive=True))
# The ledger moved to docs/reproducibility/ during the 2026-08-22 restructure. Reading a
# non-existent "ARTIFACTS.md" left `led` empty, so `ledger` was empty, so every operator counted
# as "not in the ledger" -- or, with pts empty too, nothing was checked at all.
_led_paths = ["docs/reproducibility/ARTIFACTS.md", "ARTIFACTS.md"]
_led = next((x for x in _led_paths if os.path.exists(x)), None)
if _led is None:
    print("  FAIL no ARTIFACTS.md ledger found — cannot verify operator hashes")
led = open(_led).read() if _led else ""
ledger = {}                       # basename -> sha256, parsed from the ledger's own tables
for m in re.finditer(r"`([A-Za-z0-9_.\-]+\.pt)`[^`\n]*`([0-9a-f]{64})`", led):
    ledger[m.group(1)] = m.group(2)
for m in re.finditer(r"([A-Za-z0-9_.\-]+\.pt)\s*[|:\s]\s*([0-9a-f]{64})", led):
    ledger.setdefault(m.group(1), m.group(2))

def sha(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b: break
            h.update(b)
    return h.hexdigest()

def recorded_sha(p):
    """prefer the ledger; fall back to the .pt's own sidecar, which is what E28 has."""
    b = os.path.basename(p)
    if b in ledger: return ledger[b], "ARTIFACTS.md"
    sc = p.replace(".pt", "_provenance.json")
    if os.path.exists(sc):
        try: return json.load(open(sc)).get("sha256"), "sidecar"
        except Exception: return None, "sidecar-unreadable"
    return None, None

unrec, bad, prov = [], [], []
for p in pts:
    want, src = recorded_sha(p)
    if want is None:
        unrec.append(p)
    elif sha(p) != want:
        bad.append((os.path.basename(p), src))
    if not os.path.exists(p.replace(".pt", "_provenance.json")):
        prov.append(p)
print(f"  {len(pts)} .pt on disk, {len(ledger)} hashes parsed from ARTIFACTS.md")
print(f"  {'ok  ' if not unrec else 'warn'} {len(unrec)} with NO recorded hash anywhere"
      + ("" if not unrec else ": " + ", ".join(os.path.basename(p) for p in unrec[:4])))
print(f"  {'ok  ' if not bad else 'FAIL'} {len(bad)} whose bytes DISAGREE with the recorded hash"
      + ("" if not bad else ": " + ", ".join(f"{b} (vs {s})" for b, s in bad[:4])))
print(f"  {'ok  ' if not prov else 'warn'} {len(prov)} without a _provenance.json"
      + ("" if not prov else ": " + ", ".join(os.path.basename(p) for p in prov[:4])))
if bad:
    raise SystemExit(1)
EOF

hdr "H4. Every number traceable to a file"
info "The rule: if a figure is not in a results JSON, it does not exist."
n_json="$(ls results/*.json 2>/dev/null | wc -l | tr -d ' ')"
ok "$n_json results JSON files"
# A claim document must name the results file behind it. The .tex documents did that with \src{};
# the markdown ones name it as a backticked *.json. Accept either — what is checked is that the
# document points at a file, not which markup it uses.
for doc in docs/context/RESULTS_TAXONOMY.md docs/experiments/preregs/E*.md paper/paper.tex; do
  [ -f "$doc" ] || continue
  n_src="$(grep -oE '\\src\{|`[A-Za-z0-9_./*-]+\.json`' "$doc" 2>/dev/null | wc -l | tr -d ' ' || true)"
  [ "$n_src" -gt 0 ] && ok "$(printf '%-34s' "$(basename $doc)") $n_src source-file citation(s)" \
                     || warn "$(basename $doc): names no results file"
done

hdr "H5. Test suite"
pass=0; fail=0
for t in tests/test_*.py; do
  if "$PY" "$t" >/dev/null 2>&1; then pass=$((pass+1)); else fail=$((fail+1)); bad "$(basename "$t")"; fi
done
[ "$fail" = "0" ] && ok "$pass/$((pass+fail)) test files green" || bad "$fail test file(s) failing"

hdr "H6. Load-bearing modules without a test"
"$PY" - <<'EOF'
import ast, glob, pathlib, collections
local = {p.stem for p in list(pathlib.Path("src").glob("*.py")) + list(pathlib.Path("experiments").glob("*.py"))}
rev = collections.Counter(); tested = set()
for p in (list(pathlib.Path("src").glob("*.py")) + list(pathlib.Path("experiments").glob("*.py"))
          + list(pathlib.Path("tests").glob("*.py"))):
    try: tree = ast.parse(p.read_text())
    except SyntaxError: continue
    for n in ast.walk(tree):
        mods = ([a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import)
                else [n.module.split(".")[0]] if isinstance(n, ast.ImportFrom) and n.module else [])
        for m in mods:
            if m in local:
                rev[m] += 1
                if "tests/" in str(p): tested.add(m)
gaps = [(m, c) for m, c in rev.most_common() if c >= 4 and m not in tested]
if gaps:
    for m, c in gaps: print(f"  warn {m}.py — {c} importers, 0 tests")
    print("       a module this many things depend on should have one")
else:
    print("  ok   every module with >=4 importers has a test")
EOF

hdr "H7. Nothing is still billing"
if command -v vastai >/dev/null 2>&1 && vastai show user --raw >/dev/null 2>&1; then
  vastai show instances --raw 2>/dev/null | "$PY" -c "
import json,sys
d=json.load(sys.stdin)
if not d: print('  ok   no instances running')
else:
    tot=sum(i.get('dph_total',0) for i in d)
    print(f'  warn {len(d)} instance(s) live, \${tot:.2f}/h = \${tot*24:.0f}/day')
    for i in d: print(f\"       {i['id']}  {i.get('gpu_name')}  {i.get('actual_status')}\")
    print('       pull + verify, then: bash repro/15_vast_teardown.sh')
" 2>/dev/null
else info "vastai unavailable — cannot check"; fi

hdr "H8. Uncommitted work"
n="$(git status --porcelain | wc -l | tr -d ' ' || true)"
[ "$n" = "0" ] && ok "clean tree" || { warn "$n uncommitted path(s)"; git status --short | head -8 | sed 's/^/      /'; }

hdr "H9. Data discipline"
info "Licensed corpus plaintext ships as builder + manifest, never as text."
n_c="$(git ls-files 'corpora/*.jsonl' | wc -l | tr -d ' ' || true)"
if [ "$n_c" = "0" ]; then ok "no corpus plaintext tracked"
else warn "$n_c corpus .jsonl tracked ($(git ls-files 'corpora/*.jsonl' | xargs du -ch 2>/dev/null | tail -1 | cut -f1)) — reproducible from src/build_corpora.py + manifest.json"; fi
[ -f corpora/manifest.json ] && ok "manifest.json present (SHA + seed blocks)" || bad "manifest.json missing"

hdr "H11. No disproved claim is stated as a finding"
info "The layer-shuffled 5/6 is an artifact of min-over-layers scoring, not a result."
info "It was reported as a finding, stood for weeks, and was re-raised after retraction."
info "Remembering is not a control. This is."
info "The gate carries a per-occurrence exemption for pattern hits that are a DIFFERENT quantity."
info "An exemption is how a control turns into a no-op, so its self-test runs here too."
if "$PY" "$REPRO_DIR/lib/banned_claims.py" --self-test >/dev/null 2>&1; then
  ok "claim gate still fails on what it must (self-test)"
else bad "the claim gate's self-test FAILED — the gate may no longer catch the claim"; fi
if "$PY" "$REPRO_DIR/lib/banned_claims.py"; then ok "no banned claims restated"
else bad "a disproved claim is stated as a finding — fix it before anything else"; fi

hdr "H12. No deanonymizing string in the published tree"
info "The venue is double-blind and a linked artifact repository IS supplementary material."
info "Scope is the PUBLICATION surface - git-tracked files plus results/ - not the filesystem:"
info "a gate scoped to the working tree reports leaks that cannot leak and hides ones that can."
info "The pattern list is untracked on purpose; a gate that ships it republishes what it removes."
if "$PY" "$REPO_ROOT/tools/anonymize_release.py" --self-test >/dev/null 2>&1; then
  ok "the anonymization gate still fires on planted leaks, and spares required strings"
else bad "the anonymization gate's self-test FAILED - it may no longer catch a real leak"; fi
if "$PY" "$REPO_ROOT/tools/anonymize_release.py" --verify --published-tree >/dev/null 2>&1; then
  ok "no deanonymizing string in the published tree"
else bad "the published tree would deanonymize a contributor - run: tools/anonymize_release.py --scan"; fi

hdr "H13. Every recorded script hash is obtainable"
info "48 of 73 results files record a script SHA-256 that no longer matches the script on disk."
info "That is normal - code evolves - but it leaves a reviewer unable to get the producing code."
info "47 were recoverable from git history. The release is ONE SQUASHED COMMIT with no history,"
info "so they are extracted into repro/scripts_at_run/ and this check runs WITHOUT consulting git."
if "$PY" "$REPO_ROOT/tools/script_provenance.py" --verify >/dev/null 2>&1; then
  ok "every recorded script hash is obtainable from the tree, or declared lost with a reason"
else bad "a results file names code that cannot be obtained - run: tools/script_provenance.py"; fi

hdr "H10. Docs point at files that exist"
"$PY" - <<'EOF'
import re, os, glob
# A path in a doc may be written relative to the REPO ROOT (the common case) or relative to the
# DOC ITSELF (`../paper/CONTEXT.md` from docs/README.md, which is what a markdown link has to be
# to work for a human reader). Resolving only against the root reported two false dangling refs
# and would have pushed someone to break working links to satisfy the checker. Accept either.
bad = []
for doc in ["CLAUDE.md", "README.md"] + glob.glob("docs/**/*.md", recursive=True) + glob.glob("paper/*.md"):
    if not os.path.exists(doc): continue
    here = os.path.dirname(doc)
    for m in re.findall(r'`([a-zA-Z0-9_./-]+\.(?:py|md|tex|json|sh))`', open(doc).read()):
        if "/" not in m: continue
        if not (os.path.exists(m) or os.path.exists(os.path.normpath(os.path.join(here, m)))):
            bad.append((doc, m))
if bad:
    for d, m in bad[:8]: print(f"  warn {d} -> {m} (not found)")
    print(f"       {len(bad)} dangling reference(s)")
else:
    print("  ok   every referenced path resolves")
EOF

summary
