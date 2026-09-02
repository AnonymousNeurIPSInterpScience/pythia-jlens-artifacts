#!/usr/bin/env bash
# repro/03_fetch_artifacts.sh — pull the fitted operators that are deliberately not in git.
#
#   bash repro/03_fetch_artifacts.sh            # the one lens the Tier-0 gate needs (~27 MB)
#   bash repro/03_fetch_artifacts.sh --all      # every operator, nested dirs included (~16 GB)
#   bash repro/03_fetch_artifacts.sh --check    # list what is missing locally, download nothing
#
# THE CANONICAL ARTIFACT MIRROR IS PUBLIC. No account and no login are needed:
#
#     https://huggingface.co/AnonymousInterpScience/pythia-jlens-artifacts
#
# Override with HF_REPO=<other/repo> only if you are working against a different mirror.
#
# THE MIRROR LAYOUT MATCHES THIS REPO ONE-FOR-ONE:  results/ -> results/,  corpora/ -> corpora/.
# Operators live in NESTED directories (results/lenses/{e28,plad,ladder,misc}/, results/e48/,
# results/r6/, results/r7/, ...). Until 2026-09-01 this script copied only the TOP level of
# results/ and silently dropped every nested operator, which is most of them.
#
# THIS STEP IS NOT OPTIONAL FOR T1. tests/test_anchor_fidelity.py -- the assertion that our read
# path is bit-identical to the anchor's -- loads a real lens. A clone without it fails with a
# FileNotFoundError, which is a missing artifact, not a broken repo.
#
# THE CORPUS POOL IS NOT HERE, BY DESIGN. The mirror is CC BY 4.0 and cannot carry third-party
# corpus plaintext (two files are CC BY-SA, which is share-alike). Only the manifests ship.
# Rebuild the pool at its pinned dataset revisions with:   bash repro/06_data.sh --build

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
need_py
cd "$REPO_ROOT"

export HF_REPO="${HF_REPO:-AnonymousInterpScience/pythia-jlens-artifacts}"
MODE="${1:---minimal}"
export FETCH_MODE="$MODE"

hdr "Artifact mirror: $HF_REPO"
"$PY" - <<'EOF' || die "cannot reach the mirror. It is public; check your network, or set HF_REPO."
from huggingface_hub import HfApi
import os
api = HfApi(); R = os.environ["HF_REPO"]
info = api.repo_info(R)
f = api.list_repo_files(R)
pts = [x for x in f if x.endswith(".pt")]
print(f"  {R}  private={info.private}")
print(f"  {len(pts)} operators, {len([x for x in f if x.endswith('.json')])} JSON, "
      f"{len(f)} files total")
EOF

if [ "$MODE" = "--check" ]; then
  hdr "What is missing locally — nothing will be downloaded"
  "$PY" - <<'EOF'
from huggingface_hub import HfApi
import os
api = HfApi(); R = os.environ["HF_REPO"]
remote = [x for x in api.list_repo_files(R) if x.startswith("results/") and x.endswith(".pt")]
miss = [p for p in remote if not os.path.exists(p)]
print(f"  {len(remote)} operators on the mirror, {len(remote)-len(miss)} present locally")
if miss:
    print(f"  {len(miss)} missing:")
    for p in miss[:15]: print("     ", p)
    if len(miss) > 15: print(f"      ... and {len(miss)-15} more")
    print("  fix:  bash repro/03_fetch_artifacts.sh --all")
else:
    print("  every operator on the mirror is present locally")
EOF
  exit 0
fi

hdr "Fetching"
"$PY" - <<'EOF'
from huggingface_hub import snapshot_download
import os
R, MODE = os.environ["HF_REPO"], os.environ["FETCH_MODE"]

if MODE == "--minimal":
    pat = ["results/lenses/ladder/lens_70m_n200_db128.pt",
           "results/lenses/ladder/lens_70m_n200_db128_provenance.json"]
    print("  the one operator the Tier-0 gate needs. Use --all for the full set (~16 GB).")
elif MODE == "--all":
    pat = ["results/**"]
    print("  every operator and record, nested directories included (~16 GB).")
elif MODE == "--corpora":
    raise SystemExit(
        "  --corpora is retired: the mirror carries NO corpus plaintext (licence, see the header).\n"
        "  Rebuild the pool at its pinned revisions instead:  bash repro/06_data.sh --build")
else:
    raise SystemExit(f"unknown mode '{MODE}' — use --minimal, --all or --check")

# local_dir places files at ./<repo path> directly, so nested directories are preserved and
# nothing is copied twice. The previous implementation walked only the top level of results/.
snapshot_download(R, allow_patterns=pat, local_dir=".")
n = sum(1 for _, _, fs in os.walk("results") for _ in fs) if os.path.isdir("results") else 0
print(f"  results/ now holds {n} file(s)")
EOF

hdr "Verify what arrived against the recorded hashes"
"$PY" - <<'EOF'
import glob, hashlib, json, os
# RECURSIVE. Operators live under results/lenses/**, results/e48/**, results/r6/** and more;
# a non-recursive results/*.pt glob matched ZERO files and reported "0 verified / 0 mismatch",
# which reads as success.
pts = sorted(glob.glob("results/**/*.pt", recursive=True))
ok = bad = nosha = 0
for pt in pts:
    pj = pt.replace(".pt", "_provenance.json")
    if not os.path.exists(pj):
        nosha += 1; continue
    d = json.load(open(pj))
    sha = d.get("sha256")
    if not isinstance(sha, str) or len(sha) < 16:
        nosha += 1; continue
    h = hashlib.sha256(open(pt, "rb").read()).hexdigest()
    if h == sha: ok += 1
    else:
        bad += 1; print(f"  MISMATCH {pt}")
print(f"  operators: {len(pts)} on disk / {ok} hash-verified / {bad} MISMATCH / "
      f"{nosha} with no recorded sha")
if bad: raise SystemExit(1)
EOF

hdr "Post-fetch anonymity check — a fetch is a WRITE"
# A fetch overwrites local files with the mirror's copies. If the mirror holds pre-scrub JSON, a
# fetch silently re-introduces every identity string the scrub removed. Measured, not predicted:
# the release-hygiene procedure records a background fetch overwriting 58 already-redacted
# files half an hour after a clean H12, with nothing warning.
if [ -f tools/deanon_patterns.json ]; then
  "$PY" tools/anonymize_release.py --verify --published-tree >/dev/null 2>&1 \
    && ok "no identity string in the fetched tree" \
    || { warn "THE FETCH RE-INTRODUCED AN IDENTITY STRING."
         "$PY" tools/anonymize_release.py --scan --published-tree 2>&1 | head -20; }
else
  info "tools/deanon_patterns.json absent (it is gitignored and never published) — skipping"
fi

hdr "Next"
echo "  bash repro/RUN_ALL.sh --tier T1     re-score the stored operators on CPU"
