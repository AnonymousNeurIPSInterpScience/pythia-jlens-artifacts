#!/usr/bin/env bash
# repro/04_fetch_results.sh — pull the results tree from the artifact mirror.
#
#   bash repro/04_fetch_results.sh              # every results JSON (~8 MB)   <-- START HERE
#   bash repro/04_fetch_results.sh --bulk       # + the per-experiment .json/.npz subdirs
#   bash repro/04_fetch_results.sh --check      # compare local against the mirror, change nothing
#
# THIS IS THE FIRST COMMAND A FRESH CLONE MUST RUN, BEFORE ANY TIER.
#   `results/` is gitignored in full (`.gitignore`, "results/ CUT-OVER"), so a fresh clone has an
#   EMPTY results/ and EVERY T0 module fails with FileNotFoundError until this has run. The
#   cut-over fired on 2026-08-22 (commit f9ed601, "Untrack results/ (now mirrored to HF)"); the
#   header that stood here until 2026-09-01 still described it as pending and told the reader this
#   script was "a no-op against a clone, because the JSON is still in git". It is not a no-op and
#   the JSON is not in git.
#
# THE CANONICAL MIRROR IS PUBLIC. No account and no login are needed:
#
#     https://huggingface.co/AnonymousInterpScience/pythia-jlens-artifacts
#
# Override with HF_REPO=<other/repo> only if you are working against a different mirror.
#
# WHY THIS IS SEPARATE FROM 03_fetch_artifacts.sh
#   03 fetches the things git CANNOT carry: 16 GB of fitted .pt operators. This fetches the ~8 MB
#   of results JSON — the record of record, and everything T0 needs. Splitting them keeps the
#   reason for each exclusion legible: one is a size constraint, the other is a decision about
#   where the record lives.
#
# THE MIRROR LAYOUT MATCHES THIS REPO ONE-FOR-ONE:  results/ -> results/,  corpora/ -> corpora/.
#
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
need_py
cd "$REPO_ROOT"

export HF_REPO="${HF_REPO:-AnonymousInterpScience/pythia-jlens-artifacts}"
MODE="${1:---json}"
export FETCH_MODE="$MODE"

hdr "Mirror: $HF_REPO"
"$PY" - <<'EOF' || die "cannot reach the mirror. It is public; check your network, or set HF_REPO."
from huggingface_hub import HfApi
import os
api = HfApi(); R = os.environ["HF_REPO"]
info = api.repo_info(R)
f = api.list_repo_files(R)
print(f"  authenticated as {api.whoami().get('name','?')}; private={info.private}")
print(f"  {sum(1 for x in f if x.startswith('results/') and x.endswith('.json'))} results JSON")
print(f"  {sum(1 for x in f if x.startswith('results/') and x.endswith('.pt'))} lenses")
EOF

if [ "$MODE" = "--check" ]; then
  hdr "Local vs mirror — reporting only, nothing written"
  "$PY" - <<'EOF'
from huggingface_hub import HfApi
import glob, os
api = HfApi(); R = os.environ["HF_REPO"]
remote = set(api.list_repo_files(R))
local = sorted(glob.glob("results/**/*.json", recursive=True))
miss = [p for p in local if p not in remote]
print(f"  {len(local)} local results JSON, {len(local)-len(miss)} present on the mirror")
if miss:
    print(f"  {len(miss)} local results file(s) are NOT on the mirror:")
    for p in miss[:20]: print("     ", p)
    if len(miss) > 20: print(f"      ... and {len(miss)-20} more")
    raise SystemExit(1)
print("  every local results file is present on the mirror")
EOF
  exit $?
fi

hdr "Fetching"
"$PY" - <<'EOF'
from huggingface_hub import snapshot_download
import os, shutil
R, MODE = os.environ["HF_REPO"], os.environ["FETCH_MODE"]
pat = ["results/*.json"] if MODE == "--json" else ["results/*.json", "results/*/*.json",
                                                   "results/*/*.npz"]
print(f"  snapshot: {pat}")
snap = snapshot_download(R, allow_patterns=pat)
n = 0
src_root = os.path.join(snap, "results")
for dirpath, _, names in os.walk(src_root):
    rel = os.path.relpath(dirpath, src_root)
    dst = os.path.join("results", rel) if rel != "." else "results"
    os.makedirs(dst, exist_ok=True)
    for f in sorted(names):
        shutil.copy(os.path.join(dirpath, f), os.path.join(dst, f)); n += 1
print(f"  {n} file(s) placed under results/")
EOF

hdr "Verify: does every provenance-stamped file still check out?"
"$PY" tools/migrate_provenance_paths.py --verify || warn "some stamped files did not verify — read the lines above before trusting any number"

hdr "Post-fetch anonymity check — a fetch is a WRITE"
# A fetch overwrites local files with the mirror's copies. If the mirror holds pre-scrub JSON, a
# fetch silently re-introduces every identity string the scrub removed. This is measured, not
# predicted: the release-hygiene procedure records a background 03_fetch_artifacts.sh
# overwriting 58 already-redacted files half an hour after a clean H12, with nothing warning.
# Hostname stamping was removed from src/provenance.py on 2026-08-31, so newly WRITTEN files are
# clean by construction -- but a FETCHED file was written by whatever produced it, whenever.
if [ -f tools/deanon_patterns.json ]; then
  "$PY" tools/anonymize_release.py --verify --published-tree >/dev/null 2>&1 \
    && ok "no identity string in the fetched tree" \
    || { warn "THE FETCH RE-INTRODUCED AN IDENTITY STRING."
         warn "The mirror holds pre-scrub copies. Re-scrub before publishing anything:"
         warn "  .venv/bin/python tools/anonymize_release.py --apply --apply-payload"
         "$PY" tools/anonymize_release.py --scan --published-tree 2>&1 | head -20; }
else
  warn "tools/deanon_patterns.json absent - cannot check the fetched tree for identity strings"
fi

hdr "Next"
echo "  ./lab verify      the Tier 0 gate"
