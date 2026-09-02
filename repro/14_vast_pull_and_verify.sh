#!/usr/bin/env bash
# repro/14_vast_pull_and_verify.sh — pull artifacts off a box and prove they arrived intact.
#
#   bash repro/14_vast_pull_and_verify.sh [alias] [remote_glob]
#   bash repro/14_vast_pull_and_verify.sh vast-box '/workspace/results/*'
#
# Runs BEFORE any teardown, always. Two lenses were lost by destroying a box first, which
# is why 15_vast_teardown.sh reads the receipt this script writes and refuses without it.
#
# "Verify" means: hash on the box, hash locally, compare. Not "the file exists".

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
need_py

ALIAS="${1:-$SSH_ALIAS}"
GLOB="${2:-/workspace/results/*}"
DEST="$REPO_ROOT/results"
RECEIPT="$REPO_ROOT/.pull_receipt_${ALIAS}.json"

hdr "Remote inventory on '$ALIAS'"
ssh -o ConnectTimeout=15 "$ALIAS" "ls -1 $GLOB 2>/dev/null | wc -l" >/dev/null 2>&1 \
  || die "cannot reach '$ALIAS'. Check ~/.ssh/config, or the box is already gone."
n_remote="$(ssh "$ALIAS" "ls -1 $GLOB 2>/dev/null | wc -l" | tr -d '[:space:]')"
info "$n_remote file(s) matching $GLOB"
[ "$n_remote" -gt 0 ] || die "nothing to pull — check the glob"

hdr "Hashing on the box (before transfer)"
# `sha256sum results/*` exits NON-ZERO whenever the glob contains a directory -- it is a
# "read error" per file -- and with stderr suppressed the caller saw only "remote hashing
# failed". E62 writes into results/ladder1b_b613/, so this aborted a pull whose artifacts were
# all present and intact. Since 15_vast_teardown.sh refuses to destroy without this receipt,
# the failure mode was: complete run, no receipt, and a box you cannot tear down cleanly.
# Hash FILES ONLY, recursively, so subdirectories are covered rather than fatal.
ssh "$ALIAS" "cd /workspace && find $GLOB -type f -print0 2>/dev/null | xargs -0 sha256sum" \
  > "${RECEIPT}.remote" || die "remote hashing failed"
n_hashes="$(wc -l < "${RECEIPT}.remote" | tr -d ' ')"
[ "$n_hashes" -gt 0 ] || die "remote hashing produced no hashes"
ok "$n_hashes hashes recorded (files only, recursive)"

hdr "Transfer"
mkdir -p "$DEST"
rsync -av --progress "$ALIAS:$GLOB" "$DEST/" | tail -5
ok "rsync complete"

hdr "Verify: remote hash == local hash, file by file"
"$PY" - "${RECEIPT}.remote" "$DEST" "$RECEIPT" "$ALIAS" <<'PYEOF'
import hashlib, json, os, sys
remote_file, dest, receipt, alias = sys.argv[1:5]
ok = bad = missing = 0
rows = []
for line in open(remote_file):
    line = line.strip()
    if not line: continue
    sha, path = line.split(None, 1)
    # Preserve the SUBDIRECTORY. os.path.basename() flattened results/ladder1b_b613/tv_X.json
    # to tv_X.json and then looked for it at results/tv_X.json, so every nested artifact was
    # reported MISSING and `verified` came out False on a pull where all 433 files were in fact
    # byte-identical. Since 15_vast_teardown.sh reads `verified`, that turned a clean pull into
    # an un-tearable box.
    rel = path.strip()
    for pre in ("/workspace/results/", "results/"):
        if rel.startswith(pre):
            rel = rel[len(pre):]; break
    rel = os.path.basename(rel) if os.path.isabs(rel) else rel
    local = os.path.join(dest, rel)
    if not os.path.exists(local):
        print(f"  MISSING  {rel}"); missing += 1; continue
    h = hashlib.sha256(open(local, "rb").read()).hexdigest()
    if h == sha: ok += 1; rows.append({"file": rel, "sha256": h, "match": True})
    else:
        print(f"  MISMATCH {rel}\n    box   {sha}\n    local {h}")
        bad += 1; rows.append({"file": rel, "sha256": h, "match": False})
print(f"\n  {ok} match / {bad} mismatch / {missing} missing")
json.dump({"alias": alias, "verified": bad == 0 and missing == 0,
           "n_match": ok, "n_mismatch": bad, "n_missing": missing, "files": rows},
          open(receipt, "w"), indent=1)
sys.exit(1 if (bad or missing) else 0)
PYEOF
rc=$?
rm -f "${RECEIPT}.remote"

if [ "$rc" -ne 0 ]; then
  bad "verification FAILED — do NOT destroy this box. Re-run the pull."
  exit 1
fi
ok "every file verified; receipt at $(basename "$RECEIPT")"

hdr "Next"
cat <<EOF
  Record the hashes in ARTIFACTS.md, mirror to Hugging Face from THIS machine
  (never with a write token on the box), and only then:
      bash repro/15_vast_teardown.sh $ALIAS
EOF
