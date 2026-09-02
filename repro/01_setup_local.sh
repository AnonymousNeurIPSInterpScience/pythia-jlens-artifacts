#!/usr/bin/env bash
# repro/01_setup_local.sh — build the local CPU environment from nothing.
#
# Idempotent: safe to re-run. Installs into ./.venv and vendors the anchor repo.
# Costs nothing. Takes ~3 minutes on a warm pip cache.
#
#   bash repro/01_setup_local.sh

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
cd "$REPO_ROOT"

hdr "1. Python environment"
if [ -x "$PY" ]; then
  ok "reusing existing .venv ($("$PY" -V 2>&1 | awk '{print $2}'))"
elif command -v uv >/dev/null 2>&1; then
  info "uv sync (uses uv.lock, so this is the reproducible path)"
  uv sync
  ok "created .venv via uv"
else
  warn "uv not installed — falling back to pip, which does NOT use uv.lock"
  info "the resulting environment may differ from the recorded one"
  python3 -m venv .venv
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q torch "transformers>=5.5" numpy scipy scikit-learn \
        huggingface_hub datasets accelerate safetensors
  ok "created .venv via pip"
fi

hdr "2. Vendor the anchor (github.com/anthropics/jacobian-lens @ ${VENDOR_PIN:0:7}, Apache-2.0)"
info "Gitignored on purpose: third-party code is re-fetched at a pin, never committed here,"
info "and never modified in place."
if [ -d "$VENDOR_DIR/.git" ]; then
  ok "already present"
else
  git clone --quiet "$VENDOR_URL" "$VENDOR_DIR"
  ok "cloned"
fi
git -C "$VENDOR_DIR" fetch --quiet --all
git -C "$VENDOR_DIR" checkout --quiet "$VENDOR_PIN"
got="$(git -C "$VENDOR_DIR" rev-parse HEAD)"
[ "$got" = "$VENDOR_PIN" ] || die "checkout landed on ${got:0:7}, expected ${VENDOR_PIN:0:7}"
ok "pinned at ${VENDOR_PIN:0:7}"

hdr "3. Install jlens (deps already satisfied, so --no-deps is safe)"
# A uv-created venv has NO pip in it, so `$PY -m pip` fails on a fresh clone.
# Use whichever installer this venv actually has.
if command -v uv >/dev/null 2>&1; then
  VIRTUAL_ENV="$REPO_ROOT/.venv" uv pip install -q -e "$VENDOR_DIR" --no-deps
  info "installed with uv pip"
elif "$PY" -m pip --version >/dev/null 2>&1; then
  "$PY" -m pip install -q -e "$VENDOR_DIR" --no-deps
  info "installed with python -m pip"
else
  die "neither uv nor pip is available in $PY — cannot install jlens"
fi
"$PY" -c "import jlens; print('   jlens from', jlens.__file__)" \
  || die "jlens installed but not importable"
ok "jlens importable"

hdr "4. Sanity"
"$PY" - <<'EOF'
import torch, transformers
print(f"   torch {torch.__version__}   cuda={torch.cuda.is_available()}")
print(f"   transformers {transformers.__version__}")
EOF
info "CUDA False is CORRECT on a laptop. The local tier is CPU-only by design."

hdr "Next"
cat <<'EOF'
  bash repro/02_verify_local.sh     free, ~2 min, and mandatory before any GPU spend
EOF
