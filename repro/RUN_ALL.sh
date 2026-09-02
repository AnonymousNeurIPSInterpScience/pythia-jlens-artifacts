#!/usr/bin/env bash
# repro/RUN_ALL.sh — the front door. One command, no reading required.
#
#   bash repro/RUN_ALL.sh --tier T0     free, ~1 min. Recomputes every statistic from stored JSON.
#   bash repro/RUN_ALL.sh --tier T1     CPU. Needs the .pt operators. Re-scores every read.
#   bash repro/RUN_ALL.sh --tier T2     GPU. Refits operators. Prints a cost estimate, needs --yes.
#
#   --list        show what would run and stop
#   --keep-going  do not stop at the first failure (default IS to stop)
#   --yes         confirm spend for T2
#
# WHY A TIER AND NOT JUST "RUN EVERYTHING". A reviewer's first question is not "what does this
# cost", it is "can this be run at all". T0 needs nothing but the repository: no model download, no
# 7.7 GB artifact mirror, no GPU. Every statistic in the paper is T0. That is the promise this
# script exists to make good on, and it is the only promise that gets a reviewer to a green result
# before they lose patience.
#
# WHY IT REFUSES INSTEAD OF HALF-RUNNING. `repro/00_check_config.sh` already diagnoses a broken
# environment precisely; this script's job is to consult it and stop, with the exact fix printed,
# rather than run eleven modules and fail on the twelfth for a reason stated at the top.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
EXP_DIR="repro/exp"

TIER=""; LIST=0; KEEP=0; YES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER="${2:-}"; shift 2 ;;
    --tier=*) TIER="${1#*=}"; shift ;;
    --list) LIST=1; shift ;;
    --keep-going) KEEP=1; shift ;;
    --yes|-y) YES="--yes"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

_c(){ printf '\033[%sm%s\033[0m\n' "$1" "$2"; }
hdr(){ echo; _c "1;36" "══ $* ══════════════════════════════════════"; }
ok(){ _c "0;32" "   ✓ $*"; }
bad(){ _c "0;31" "   ✗ $*"; }
note(){ _c "0;37" "   $*"; }

case "$TIER" in
  T0|T1|T2) ;;
  "") bad "no tier given."; note "try: bash repro/RUN_ALL.sh --tier T0    (free, ~1 min)"; exit 2 ;;
  *)  bad "unknown tier '$TIER' — expected T0, T1 or T2"; exit 2 ;;
esac

# ---------------------------------------------------------------- collect
# A module opts in by declaring MODULE_TIER. Anything without one is skipped and NAMED, because a
# silently skipped module is how coverage rots: repro/exp/cv7_1b.sh is a detached launcher, not a
# module, and it must not be counted as either run or missing.
mods=(); untiered=()
for f in "$EXP_DIR"/*.sh; do
  b="$(basename "$f")"
  [ "$b" = "_lib.sh" ] && continue
  t="$(sed -n 's/^MODULE_TIER="\([^"]*\)".*/\1/p' "$f" | head -1)"
  if [ -z "$t" ]; then untiered+=("$b"); continue; fi
  [ "$t" = "$TIER" ] && mods+=("$f")
done

hdr "$TIER — ${#mods[@]} module(s)"
case "$TIER" in
  T0) note "pure recomputation over stored results JSON. No model, no operator, no GPU." ;;
  T1) note "scores stored .pt operators on CPU. Needs the artifact mirror (repro/03_fetch_artifacts.sh)." ;;
  T2) note "refits operators on GPU. fp32, TF32 forbidden. This costs real money." ;;
esac
for f in "${mods[@]}"; do
  b="$(basename "$f" .sh)"
  w="$(sed -n 's/^MODULE_WHY="\([^"]*\)".*/\1/p' "$f" | head -1)"
  printf "   %-22s %s\n" "$b" "${w:0:80}"
done
if [ "${#untiered[@]}" -gt 0 ]; then
  note ""
  note "not a tiered module, skipped: ${untiered[*]}"
fi

if [ "$LIST" -eq 1 ]; then exit 0; fi
if [ "${#mods[@]}" -eq 0 ]; then bad "nothing to run at $TIER"; exit 1; fi

# ---------------------------------------------------------------- preflight
hdr "preflight"
if [ ! -x .venv/bin/python ]; then
  bad "no .venv — run:  uv sync"
  exit 1
fi
ok ".venv present"
if ! .venv/bin/python -c "import jlens" 2>/dev/null; then
  bad "the vendored anchor library is not importable."
  note "fix:  bash repro/01_setup_local.sh"
  note "      (clones github.com/anthropics/jacobian-lens @ 581d398 and installs jlens)"
  note "This is the step most often missed: jacobian-lens/ is gitignored, so a fresh clone"
  note "does not have it and five of the eight gates fail with ModuleNotFoundError."
  exit 1
fi
ok "jlens importable"

# EVERY tier needs the results tree. `results/` is gitignored in full, so a fresh clone has an
# EMPTY results/ and every module fails with FileNotFoundError. Until 2026-09-01 nothing checked
# this and the first module simply crashed, which reads as a broken repo rather than a missing
# fetch. Assert on observed state, never on "the clone succeeded" (CLAUDE.md rule 9).
if [ ! -d results ] || [ -z "$(find results -name '*.json' -print -quit 2>/dev/null)" ]; then
  bad "results/ has no JSON. A fresh clone does not carry it — it is gitignored in full."
  note "fix:  bash repro/04_fetch_results.sh"
  note "      (public mirror, ~8 MB, no account needed:"
  note "       https://huggingface.co/AnonymousInterpScience/pythia-jlens-artifacts)"
  exit 1
fi
ok "results tree present ($(find results -name '*.json' | wc -l | tr -d ' ') JSON)"

if [ "$TIER" != "T0" ]; then
  if [ -z "$(find results -name '*.pt' -print -quit 2>/dev/null)" ]; then
    bad "$TIER needs the fitted operators and results/ has no .pt files."
    note "fix:  bash repro/03_fetch_artifacts.sh --all   (see docs/reproducibility/ARTIFACTS.md)"
    note "Or run the free tier instead:  bash repro/RUN_ALL.sh --tier T0"
    exit 1
  fi
  ok "operators present ($(find results -name '*.pt' | wc -l | tr -d ' ') .pt)"
fi

if [ "$TIER" = "T2" ]; then
  hdr "cost"
  bash repro/20_cost_estimate.sh 2>/dev/null || note "(cost estimator unavailable)"
  if [ -z "$YES" ]; then
    bad "T2 refits operators on rented GPUs and will spend money. Re-run with --yes to confirm."
    exit 1
  fi
fi

# ---------------------------------------------------------------- run
pass=0; fail=0; failed=()
for f in "${mods[@]}"; do
  b="$(basename "$f" .sh)"
  hdr "run $b"
  if bash "$f" $YES; then
    pass=$((pass+1)); ok "$b"
  else
    fail=$((fail+1)); failed+=("$b"); bad "$b FAILED"
    if [ "$KEEP" -eq 0 ]; then
      hdr "stopped at the first failure"
      note "re-run with --keep-going to see every failure in one pass"
      break
    fi
  fi
done

hdr "$TIER result"
ok "$pass passed"
if [ "$fail" -gt 0 ]; then
  bad "$fail failed: ${failed[*]}"
  exit 1
fi
_c "1;32" "   every $TIER module reproduced."
