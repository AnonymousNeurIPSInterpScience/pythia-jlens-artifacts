#!/usr/bin/env bash
# repro/exp/_lib.sh — the shared harness every experiment module sources.
#
# WHY THESE MODULES EXIST
#   Until now an experiment was "a .py someone ran once with flags they remembered". The flags
#   are load-bearing -- `--band 9,21` vs the 0.35-0.85 default is [9,21] vs [8,20], and E49's 410M
#   cells are on the wrong band because of exactly that -- and they lived in shell history. Each
#   module here is the ONE canonical invocation of one experiment: the flags that produced the
#   stored result, in a file, in git, next to the reason they are those flags.
#
# WHAT EVERY MODULE GUARANTEES
#   1. It declares its OUTPUTS and its INPUTS up front, and refuses to start if an input is missing.
#   2. It checks the real exit status of the python process. `run_py` uses PIPESTATUS, because a
#      grep filter in a pipeline masks a SystemExit -- that has bitten this programme
#      (`SystemExit("pool ... yielded 796/800")` was missed by a grep for "Error").
#   3. It asserts the output file EXISTS AND CARRIES PROVENANCE afterwards. File presence is not
#      file integrity; a truncated lens once passed a presence check.
#   4. It prints cost and, above the budget gate, refuses without --yes.
#
# CONVENTIONS
#   MODULE_ID     short id, matches the results filename family (e48, e36, ...)
#   MODULE_WHY    one line: what question this settles
#   MODULE_COST   free | cpu-hours | gpu:<$>  -- printed before anything runs
#   MODULE_TIER   T0 | T1 | T2               -- what a reviewer needs in order to run it
#   OUTPUTS       array of paths, relative to repo root, that must exist afterwards
#   INPUTS        array of paths that must exist before
#
# THE TIERS, and why they are separate from MODULE_COST. Cost answers "what will this spend";
# tier answers "can this be run at all", which is the question a reviewer asks first.
#
#   T0  pure recomputation over the stored results JSON. No model, no operator, no GPU, no
#       download beyond the repository itself. Seconds. EVERY STATISTIC IN THE PAPER IS T0.
#   T1  scores stored .pt operators against the battery on CPU. Needs the artifact mirror.
#       Minutes to hours.
#   T2  fits operators from scratch. GPU, fp32, TF32 forbidden. Hours, and real money.
#
# `bash repro/RUN_ALL.sh --tier T0` is the front door and it is free. A reviewer who gets a green
# result in ninety seconds reads the rest of the repository generously.

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

_c()  { printf '\033[%sm%s\033[0m\n' "$1" "$2"; }
hdr() { echo; _c "1;36" "── $* ─────────────────────────────────────────"; }
info(){ _c "0;37" "   $*"; }
ok()  { _c "0;32" "   ✓ $*"; }
warn(){ _c "0;33" "   ! $*"; }
die() { _c "0;31" "   ✗ $*"; exit 1; }

ASSUME_YES=0
DRY_RUN=0
for _a in "$@"; do
  case "$_a" in
    --yes|-y) ASSUME_YES=1 ;;
    --dry-run|-n) DRY_RUN=1 ;;
  esac
done

# ---------------------------------------------------------------- preamble
# Print the module contract BEFORE doing anything, so `--dry-run` is a readable spec.
module_banner() {
  hdr "$MODULE_ID — ${MODULE_TITLE:-}"
  info "why    : ${MODULE_WHY:-（unstated）}"
  info "tier   : ${MODULE_TIER:-UNCLASSIFIED}"
  info "cost   : ${MODULE_COST:-unknown}"
  [ "${#INPUTS[@]}"  -gt 0 ] && info "inputs : ${INPUTS[*]}"
  [ "${#OUTPUTS[@]}" -gt 0 ] && info "outputs: ${OUTPUTS[*]}"
  return 0
}

require_inputs() {
  local missing=0
  for f in "${INPUTS[@]:-}"; do
    [ -z "$f" ] && continue
    if [ ! -e "$f" ]; then warn "MISSING INPUT: $f"; missing=1; fi
  done
  [ "$missing" -eq 0 ] || die "refusing to run with missing inputs — see above"
}

# Above the CLAUDE.md budget gate ($40 / 6 h) a module must be explicitly confirmed.
#
# Three things this got wrong until 2026-08-15, all of which a reviewer hits on their first run:
#   1. only the literal string "free" counted as free, so every module declaring
#      `free (~25 min CPU)` prompted for spend confirmation it did not need;
#   2. --dry-run prompted too, though a dry run spends nothing by construction;
#   3. with no TTY the `read` hit EOF, left $r empty, and the module died "aborted" —
#      a confirmation prompt reported as a failure, which is trap 4 in a new costume.
confirm_cost() {
  [ "$DRY_RUN" -eq 1 ] && return 0
  case "${MODULE_COST:-}" in
    free|free\ *|*"\$0"*|"") return 0 ;;
  esac
  [ "$ASSUME_YES" -eq 1 ] && return 0
  [ -t 0 ] || die "this module costs ${MODULE_COST} and stdin is not a terminal — pass --yes to confirm"
  read -r -p "   this module costs ${MODULE_COST}. proceed? [y/N] " r
  [[ "$r" =~ ^[Yy]$ ]] || die "aborted"
}

# ---------------------------------------------------------------- the run
# run_py <script.py> [args...]
# Never pipe this into a filter without capturing PIPESTATUS -- that is what this wrapper is for.
run_py() {
  local script="$1"; shift
  [ -f "$script" ] || die "no such script: $script"
  info "run: $PY $script $*"
  if [ "$DRY_RUN" -eq 1 ]; then warn "--dry-run: not executing"; return 0; fi
  set +e
  "$PY" "$script" "$@" 2>&1 | grep -Ev '^(Loading weights|.*it/s\])'
  local rc=${PIPESTATUS[0]}
  set -e
  [ "$rc" -eq 0 ] || die "$script exited $rc"
  ok "$script exited 0"
}

# ---------------------------------------------------------------- the receipt
# Presence is not integrity. Every declared output must exist AND carry a provenance block whose
# recorded script hash matches the file on disk right now.
verify_outputs() {
  [ "$DRY_RUN" -eq 1 ] && { warn "--dry-run: skipping output verification"; return 0; }
  local bad=0
  for f in "${OUTPUTS[@]:-}"; do
    [ -z "$f" ] && continue
    [ -s "$f" ] || { warn "MISSING OR EMPTY OUTPUT: $f"; bad=1; continue; }
    case "$f" in
      *.json)
        if ! "$PY" - "$f" <<'EOF'
import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from provenance import verify_result
r = verify_result(sys.argv[1])
if not r.get("has_provenance"):
    print(f"   ! {sys.argv[1]}: NO PROVENANCE BLOCK (module predates provenance.py?)")
    raise SystemExit(2)
if not r["script_unchanged"]:
    print(f"   ! {sys.argv[1]}: script changed since write ({r.get('script_unchanged_reason','hash differs')})")
    raise SystemExit(3)
if not r["payload_sha_matches"]:
    print(f"   ! {sys.argv[1]}: payload hash does not match its own record — file edited by hand?")
    raise SystemExit(4)
print(f"   provenance ok: commit {r['commit']}{' DIRTY' if r['dirty_when_written'] else ''}, "
      f"{r['inputs_checked']} input(s) hashed")
EOF
        then bad=1; fi ;;
      *) info "output present: $f ($(du -h "$f" | cut -f1))" ;;
    esac
  done
  [ "$bad" -eq 0 ] || die "output verification FAILED — the result is not trustworthy"
  ok "all outputs present and provenance-verified"
}

# The standard module body: banner, gate, run the caller's `main`, verify.
module_main() {
  module_banner
  require_inputs
  confirm_cost
  main "$@"
  verify_outputs
  hdr "$MODULE_ID done"
}
