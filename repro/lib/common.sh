#!/usr/bin/env bash
# repro/lib/common.sh — shared helpers. Sourced by every repro/*.sh; not run directly.
#
# Everything here exists so the scripts can be read top-to-bottom by someone who has
# never seen this repo. No cleverness, no hidden state.

set -euo pipefail

# ---------------------------------------------------------------- paths
REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$REPRO_DIR/.." && pwd)"
PY="${REPO_ROOT}/.venv/bin/python"
VENDOR_DIR="${REPO_ROOT}/jacobian-lens"
VENDOR_PIN="581d398613e5602a5af361e1c34d3a92ea82ba8e"
VENDOR_URL="https://github.com/anthropics/jacobian-lens"
INSTANCE_FILE="${REPO_ROOT}/.instance_id"
SSH_ALIAS="${SSH_ALIAS:-vast-box}"
REMOTE_PY="/opt/conda/bin/python"     # the 2.4.0 pytorch image; NOT /venv/main/bin/python
REMOTE_CWD="/workspace"

# ---------------------------------------------------------------- output
if [ -t 1 ]; then
  R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; B=$'\033[34m'; DIM=$'\033[2m'; N=$'\033[0m'
else
  R=""; G=""; Y=""; B=""; DIM=""; N=""
fi
_fails=0; _warns=0
hdr()  { printf "\n%s== %s%s\n" "$B" "$*" "$N"; }
ok()   { printf "  %sok  %s%s\n" "$G" "$*" "$N"; }
warn() { printf "  %swarn%s %s\n" "$Y" "$N" "$*"; _warns=$((_warns+1)); }
bad()  { printf "  %sFAIL%s %s\n" "$R" "$N" "$*"; _fails=$((_fails+1)); }
info() { printf "  %s%s%s\n" "$DIM" "$*" "$N"; }
die()  { printf "\n%sABORT:%s %s\n" "$R" "$N" "$*" >&2; exit 1; }
summary() {
  if [ "$_fails" -gt 0 ]; then
    printf "\n%s%d FAILED%s, %d warning(s). A FAIL blocks; fix it before spending.\n" \
      "$R" "$_fails" "$N" "$_warns"; return 1
  elif [ "$_warns" -gt 0 ]; then
    printf "\n%s0 failed, %d warning(s)%s — nothing is broken, but each warning is a\n" \
      "$Y" "$_warns" "$N"
    printf "known gap someone will trip over. See the notes above.\n"; return 0
  else printf "\n%sALL CHECKS PASSED%s\n" "$G" "$N"; return 0; fi
}

# ---------------------------------------------------------------- guards
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found. $2"; }
need_py()  { [ -x "$PY" ] || die "no venv at $PY — run repro/01_setup_local.sh first"; }
need_instance() {
  [ -f "$INSTANCE_FILE" ] || die "no $INSTANCE_FILE — run repro/12_vast_provision.sh, or write the id yourself"
  INSTANCE_ID="$(tr -d '[:space:]' < "$INSTANCE_FILE")"
  [ -n "$INSTANCE_ID" ] || die "$INSTANCE_FILE is empty"
}

# Confirm before anything outward-facing or destructive. FORCE=1 skips it (for CI).
confirm() {
  [ "${FORCE:-0}" = "1" ] && return 0
  printf "%s%s%s [type 'yes' to proceed] " "$Y" "$1" "$N"
  read -r a; [ "$a" = "yes" ] || die "cancelled"
}

# ---------------------------------------------------------------- the hardware model
# s/prompt = K_GPU * n_layers * d_model^2 / 1e6, with K measured, not assumed.
#
# MEASURED anchors (each traceable):
#   410m on L40S  3.04-3.11 s/prompt over N in {50,100,200,400}   E28 run log, 2026-08-12
#   1b   on H100  8.375 s/prompt                                  CLAUDE.md §7
#   1b   on A100  21.44 s/prompt                                  CLAUDE.md §7
# giving K = s/prompt / (n_layers*d_model^2/1e6):
#   L40S 0.122   H100 0.125   A100 0.320
# L40S and H100 land within 2% of each other despite very different fp32 peaks, so this
# workload is NOT purely fp32-limited on those two; A100 is 2.6x worse, consistent with
# its 19.5 TFLOPS fp32. Anything not listed above is EXTRAPOLATION — labelled as such.
gpu_k() {
  case "$(echo "$1" | tr 'a-z' 'A-Z')" in
    L40S)      echo "0.122 measured" ;;
    H100*)     echo "0.125 measured" ;;
    A100*)     echo "0.320 measured" ;;
    RTX4090|4090) echo "0.135 extrapolated" ;;   # fp32 82.6 TF, between L40S and H100
    *)         echo "0.150 unknown-gpu-guess" ;;
  esac
}

# n_layers d_model  per Pythia size
model_shape() {
  case "$1" in
    70m)  echo "6 512"    ;; 160m) echo "12 768"   ;; 410m) echo "24 1024"  ;;
    1b)   echo "16 2048"  ;; 1.4b) echo "24 2048"  ;; 2.8b) echo "32 2560"  ;;
    6.9b) echo "32 4096"  ;; 12b)  echo "36 5120"  ;;
    *) die "unknown model '$1' — use one of 70m 160m 410m 1b 1.4b 2.8b 6.9b 12b" ;;
  esac
}

# VRAM floor for an fp32 fit, GB. Model weights x4 bytes plus the dim_batch working set.
model_vram_gb() {
  case "$1" in
    70m) echo 3 ;; 160m) echo 4 ;; 410m) echo 8 ;; 1b) echo 14 ;;
    1.4b) echo 18 ;; 2.8b) echo 32 ;; 6.9b) echo 64 ;; 12b) echo 96 ;;
  esac
}
