#!/usr/bin/env bash
# repro/10_vast_auth.sh — get vast.ai working, end to end. Walks you through each step
# and verifies it, rather than assuming. Costs nothing.
#
#   bash repro/10_vast_auth.sh

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

hdr "1. The vastai CLI"
if command -v vastai >/dev/null 2>&1; then
  ok "installed: $(vastai --version 2>&1 | head -1)"
else
  warn "not installed"
  cat <<'EOF'
    Install it, then re-run this script:
        pip install --user vastai
    Make sure ~/.local/bin is on your PATH.
EOF
  exit 1
fi

hdr "2. API key"
info "Get it from https://cloud.vast.ai/account/  (the 'API Key' box)."
info "It is stored at ~/.vast_api_key by the CLI. It is a SECRET: never commit it,"
info "never paste it into a script, never put it on a rented box."
if vastai show user --raw >/dev/null 2>&1; then
  "$PY" - <<'EOF' 2>/dev/null || true
import json, subprocess
d = json.loads(subprocess.run(["vastai","show","user","--raw"],capture_output=True,text=True).stdout)
print(f"   authenticated as {d.get('username','?')}  <{d.get('email','?')}>")
print(f"   credit: ${d.get('credit',0):.2f}")
print(f"   NOTE: a run that outlives your credit is killed mid-fit. Top up before a ladder.")
EOF
  ok "authenticated"
else
  warn "not authenticated"
  cat <<'EOF'
    Run this yourself (it prompts for the key, so it must not live in a script):
        vastai set api-key <YOUR_KEY>
    Then re-run this script.
EOF
  exit 1
fi

hdr "3. SSH key — how you actually reach the box"
KEY=""
for k in "$HOME/.ssh/id_ed25519.pub" "$HOME/.ssh/id_rsa.pub"; do [ -f "$k" ] && KEY="$k" && break; done
if [ -n "$KEY" ]; then
  ok "local public key: $KEY"
else
  warn "no SSH key found"
  info "FIX: ssh-keygen -t ed25519 -C \"\$(whoami)@repro\"   (accept the defaults)"
  exit 1
fi
n_reg="$(vastai show ssh-keys --raw 2>/dev/null | "$PY" -c 'import json,sys; print(len(json.load(sys.stdin)))' 2>/dev/null || echo 0)"
if [ "$n_reg" -gt 0 ]; then
  ok "$n_reg key(s) registered with vast.ai"
else
  warn "no key registered with vast.ai — you will provision a box you cannot log into"
  info "FIX: vastai create ssh-key \"\$(cat $KEY)\""
fi

hdr "4. SSH aliases"
info "The repo's scripts talk to boxes by alias, never by raw host:port, so a rebuild"
info "does not scatter IP addresses through your shell history."
if grep -q "^Host ${SSH_ALIAS}\b" "$HOME/.ssh/config" 2>/dev/null; then
  ok "alias '${SSH_ALIAS}' present in ~/.ssh/config"
else
  warn "alias '${SSH_ALIAS}' not in ~/.ssh/config"
  info "repro/12_vast_provision.sh writes it for you after provisioning."
fi

hdr "5. Ground rules that cost money if you skip them"
cat <<'EOF'
  - Control plane via the vastai CLI only. NEVER `vastai execute` — use ssh.
  - Budget gate: pause and report before any job projected above $40 or 6 hours.
    repro/20_cost_estimate.sh tells you which side of the line you are on.
  - Nothing is destroyed until its artifacts are pulled AND hash-verified.
    repro/15_vast_teardown.sh refuses otherwise. Two lenses were lost learning this.
  - A rented box gets a READ-ONLY HF token, or none. Mirror from your laptop.
EOF

hdr "Next"
echo "  bash repro/11_vast_find_offers.sh L40S      # what is available and what it costs"
