#!/usr/bin/env bash
# repro/15_vast_teardown.sh — destroy a box, but only after its artifacts are provably safe.
#
#   bash repro/15_vast_teardown.sh [alias] [instance_id]
#   FORCE=1 ... to skip the prompt (the receipt check still applies)
#   ALLOW_UNVERIFIED=1 ... deliberate override; it makes you type the reason
#
# THE ORDER IS NOT NEGOTIABLE: pull -> hash-verify -> mirror -> destroy. Two lenses were
# lost with their box because someone reordered it. This script enforces the order by
# reading the receipt that 14_vast_pull_and_verify.sh writes.

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
need_py
need_cmd vastai "pip install --user vastai"

ALIAS="${1:-$SSH_ALIAS}"
RECEIPT="$REPO_ROOT/.pull_receipt_${ALIAS}.json"

if [ -n "${2:-}" ]; then INSTANCE_ID="$2"; else need_instance; fi

hdr "Instance $INSTANCE_ID (alias '$ALIAS')"
vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | "$PY" -c "
import json,sys
d=json.load(sys.stdin)
dph=d.get('dph_total',0);
print(f\"  {d.get('gpu_name')}  \${dph:.3f}/h  status={d.get('actual_status')}\")
" 2>/dev/null || warn "could not read instance (it may already be gone)"

hdr "Artifact safety check"
if [ -f "$RECEIPT" ]; then
  "$PY" - "$RECEIPT" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  receipt: {d['n_match']} verified / {d['n_mismatch']} mismatch / {d['n_missing']} missing")
sys.exit(0 if d.get("verified") else 1)
PYEOF
  if [ $? -eq 0 ]; then ok "artifacts pulled and hash-verified"; VERIFIED=1
  else bad "receipt exists but records a mismatch"; VERIFIED=0; fi
else
  bad "NO PULL RECEIPT for '$ALIAS'"
  info "Run this first:  bash repro/14_vast_pull_and_verify.sh $ALIAS"
  VERIFIED=0
fi

if [ "$VERIFIED" -ne 1 ]; then
  if [ "${ALLOW_UNVERIFIED:-0}" = "1" ]; then
    warn "ALLOW_UNVERIFIED=1 — overriding the guard"
    printf "  %sState, in one line, why this box holds nothing worth keeping:%s " "$Y" "$N"
    read -r reason
    [ -n "$reason" ] || die "no reason given"
    echo "$(date -u +%FT%TZ)  destroy $INSTANCE_ID unverified: $reason" >> "$REPO_ROOT/logs/teardown.log"
    warn "recorded in logs/teardown.log"
  else
    die "refusing to destroy an unverified box. Pull first, or set ALLOW_UNVERIFIED=1 deliberately."
  fi
fi

hdr "Mirror reminder"
cat <<'EOF'
  A local copy is one disk failure from zero. If these artifacts are not yet on
  Hugging Face, mirror them BEFORE destroying — from this machine, never with a
  write-scoped token on the box.
EOF

confirm "Destroy instance $INSTANCE_ID? This is irreversible and the box's disk goes with it."

# vastai has its OWN y/N prompt. Without stdin it silently declines and the box keeps
# billing while the script reports "destroy issued". Feed it explicitly.
yes | vastai destroy instance "$INSTANCE_ID" 2>&1 | tail -1 | sed 's/^/  /' 
ok "destroy issued"
sleep 3
still="$(vastai show instances --raw 2>/dev/null | "$PY" -c \
  "import json,sys; print(sum(1 for i in json.load(sys.stdin) if str(i['id'])=='$INSTANCE_ID'))" 2>/dev/null || echo '?')"
if [ "$still" = "0" ]; then ok "confirmed gone"
else bad "STILL RUNNING and still billing — destroy did NOT take effect"
     info "run: yes | vastai destroy instance $INSTANCE_ID"; fi

[ -f "$INSTANCE_FILE" ] && [ "$(tr -d '[:space:]' < "$INSTANCE_FILE")" = "$INSTANCE_ID" ] && rm -f "$INSTANCE_FILE" && info "cleared .instance_id"
mv -f "$RECEIPT" "$REPO_ROOT/logs/pull_receipt_${ALIAS}_$(date -u +%Y%m%dT%H%M%SZ).json" 2>/dev/null && info "receipt archived to logs/"

hdr "Remaining instances (anything here is still billing)"
vastai show instances 2>/dev/null | head -10 | sed 's/^/  /'
