#!/usr/bin/env bash
# repro/00_check_config.sh — WHAT YOU MUST CONFIGURE. Run this first; it changes nothing.
#
# Prints, for every external thing this repo needs: whether it is present, what it is for,
# and the exact command to fix it if it is missing. Read-only and free.
#
#   bash repro/00_check_config.sh

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

echo "Repository: $REPO_ROOT"
echo "This check is READ-ONLY. Nothing is installed, uploaded or destroyed."

# ============================================================ 1. local toolchain
hdr "1. Local toolchain (needed for everything, including the free CPU tier)"

if command -v python3 >/dev/null 2>&1; then
  ok "python3 $(python3 -V 2>&1 | awk '{print $2}')"
else
  bad "python3 missing — install Python >= 3.10"
fi

if [ -x "$PY" ]; then
  ok "venv at .venv ($("$PY" -V 2>&1 | awk '{print $2}'))"
else
  bad "no .venv"; info "FIX: bash repro/01_setup_local.sh"
fi

if command -v uv >/dev/null 2>&1; then ok "uv $(uv --version 2>&1 | awk '{print $2}')"
else warn "uv missing (optional; 01_setup_local.sh falls back to pip)"
     info "FIX: curl -LsSf https://astral.sh/uv/install.sh | sh"; fi

if command -v git >/dev/null 2>&1; then ok "git $(git --version | awk '{print $3}')"
else bad "git missing"; fi

# ============================================================ 2. the vendored anchor
hdr "2. The vendored anchor (jlens) — THE single point of failure"
info "Imported 57 times. Every fit, read and write routes through it. Nothing runs without it."

if [ -d "$VENDOR_DIR/.git" ]; then
  got="$(git -C "$VENDOR_DIR" rev-parse HEAD 2>/dev/null || echo none)"
  if [ "$got" = "$VENDOR_PIN" ]; then ok "jacobian-lens at pinned commit ${VENDOR_PIN:0:7}"
  else bad "jacobian-lens at ${got:0:7}, expected ${VENDOR_PIN:0:7}"
       info "FIX: git -C jacobian-lens checkout $VENDOR_PIN"; fi
  if [ -z "$(git -C "$VENDOR_DIR" status --porcelain 2>/dev/null)" ]; then
    ok "vendored tree is unmodified (it must never be edited in place)"
  else bad "vendored tree has local modifications — discipline violation"; fi
else
  bad "jacobian-lens/ absent"; info "FIX: bash repro/01_setup_local.sh"
fi

if [ -x "$PY" ]; then
  if "$PY" -c "import jlens" 2>/dev/null; then ok "jlens importable"
  else bad "jlens not importable"
       info "FIX: bash repro/01_setup_local.sh   (a uv-created venv has no pip, so it"
       info "     uses 'uv pip install -e jacobian-lens --no-deps' rather than python -m pip)"; fi
fi

# ============================================================ 3. Hugging Face
hdr "3. Hugging Face"
info "READ access downloads Pythia weights (public, no token strictly required, but"
info "rate limits are far kinder with one). WRITE access is only for mirroring artifacts."

if [ -x "$PY" ] && "$PY" -c "import huggingface_hub" 2>/dev/null; then
  who="$("$PY" -c "from huggingface_hub import HfApi; print(HfApi().whoami().get('name',''))" 2>/dev/null || true)"
  if [ -n "$who" ]; then ok "logged in as '$who'"
  else warn "not logged in (downloads still work; mirroring will not)"
       info "FIX: $PY -m huggingface_hub.commands.huggingface_cli login"; fi
else
  warn "huggingface_hub unavailable — run repro/01_setup_local.sh"
fi
info "NEVER put a write-scoped HF token on a rented box. Pull, verify locally, mirror from your laptop."

# ============================================================ 4. GPU rental
hdr "4. GPU rental (vast.ai) — only needed for the paid tiers"

if command -v vastai >/dev/null 2>&1; then
  ok "vastai CLI $(vastai --version 2>&1 | head -1)"
  if vastai show user --raw >/dev/null 2>&1; then
    bal="$(vastai show user --raw 2>/dev/null | "$PY" -c "import json,sys; print(json.load(sys.stdin).get('credit','?'))" 2>/dev/null || echo '?')"
    ok "vast.ai authenticated (credit: \$$bal)"
  else bad "vastai not authenticated"; info "FIX: bash repro/10_vast_auth.sh"; fi
else
  warn "vastai CLI missing (only needed for GPU tiers)"
  info "FIX: pip install vastai   then   bash repro/10_vast_auth.sh"
fi

if [ -f "$HOME/.ssh/id_ed25519.pub" ] || [ -f "$HOME/.ssh/id_rsa.pub" ]; then
  ok "SSH public key present"
else
  warn "no SSH public key — you cannot reach a rented box without one"
  info "FIX: ssh-keygen -t ed25519 -C \"\$(whoami)@repro\""
fi

if [ -f "$INSTANCE_FILE" ]; then
  info "current .instance_id -> $(tr -d '[:space:]' < "$INSTANCE_FILE")"
else
  info "no .instance_id (expected when nothing is rented)"
fi

# ============================================================ 5. data
hdr "5. Data that is NOT in the repo and must be regenerated"
info "Corpus plaintext is deliberately not shipped. Builders + manifests are."

for f in corpora/manifest.json src/build_corpora.py; do
  [ -e "$REPO_ROOT/$f" ] && ok "$f" || bad "$f missing"
done
if [ -d "$VENDOR_DIR/data/evaluations" ]; then
  ok "anchor eval sets at jacobian-lens/data/evaluations"
else
  bad "anchor eval sets missing — they ship inside the vendored repo"
fi

# ============================================================ 6. what it costs
hdr "6. What each tier costs, before you spend anything"
cat <<'EOF'
  TIER 0  local CPU, FREE, ~2 min      repro/02_verify_local.sh
          Full test suite + bit-identity to the anchor. Proves the read path is correct.
          If this fails, no GPU spend can be trusted. NEVER skip it.

  TIER 1  one GPU-hour, ~$0.50         repro/12_vast_provision.sh + 13_vast_preflight.sh
          Provision, install the toolchain, re-run the anchor gate ON the box.

  TIER 2  a single experiment          repro/20_cost_estimate.sh tells you the price FIRST
          e.g. one 410M lens at N=200 on an L40S: 0.17 GPU-h, ~$0.14

  TIER 3  a full ladder                budget-gated: pause and report above $40 or 6 h
          e.g. E28 (180 fits, 4 boxes): ~21 GPU-h, ~5-6 h wall, ~$17

EOF

summary
