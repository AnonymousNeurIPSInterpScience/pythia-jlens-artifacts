#!/usr/bin/env bash
#
# THIS IS A LAUNCHER, NOT A REPRO MODULE. It backgrounds the CV7 run so a laptop can close its
# lid without killing it. It does not source _lib.sh, so it makes none of the harness guarantees
# (declared inputs, real exit status, provenance-verified outputs). The canonical, verifiable
# invocation is repro/exp/cv7_rung.sh; it carries no MODULE_TIER on purpose, so RUN_ALL.sh
# does not pick it up twice.
# repro/exp/cv7_1b.sh — CV7, the 1B rung. LOCAL CPU. Detached, survives a closed lid.
#
#   bash repro/exp/cv7_1b.sh          launch detached, print how to watch it
#   bash repro/exp/cv7_1b.sh --fg     run in the foreground instead
#   bash repro/exp/cv7_1b.sh --status where it is now
#
# WHY THIS EXISTS RATHER THAN JUST RUNNING THE PYTHON. Three things kill a long local job on a
# laptop and none of them is the job's fault:
#   1. closing the terminal sends SIGHUP          -> nohup + setsid
#   2. macOS idle-sleeps the machine on lid close -> caffeinate -i (idle sleep only; the DISPLAY
#      is still allowed to sleep, so this does not hold the screen on)
#   3. an ssh/mosh session dropping                -> setsid detaches from the controlling tty
# `caffeinate -i` holds the assertion only for the lifetime of the wrapped command, so the machine
# returns to normal power behaviour the moment the run ends. There is nothing to clean up.
#
# NO GPU. NO PROVISIONING. NO SPEND. The 15 operators were fitted for E62 and are on disk at
# results/ladder1b_b613/; this only caches h_t and scores. ~6 TFLOP, minutes.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"
LOG="$REPO_ROOT/logs/cv7_1b.log"
PIDF="$REPO_ROOT/logs/cv7_1b.pid"
mkdir -p "$REPO_ROOT/logs"

status() {
  if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "  RUNNING  pid $(cat "$PIDF")"
  else
    echo "  not running"
  fi
  [ -f "$LOG" ] && { echo "  --- last 12 log lines ---"; tail -12 "$LOG"; } || echo "  no log yet"
  [ -f "$REPO_ROOT/results/cv7_1b_rung.json" ] \
    && echo "  ADJUDICATED -> results/cv7_1b_rung.json" \
    || echo "  not yet adjudicated"
}

case "${1:-}" in
  --status) status; exit 0 ;;
esac

[ -x "$PY" ] || { echo "no venv at $PY — run ./lab setup"; exit 1; }
n_lens=$(ls "$REPO_ROOT"/results/ladder1b_b613/lens_*_1b_n200_s*.pt 2>/dev/null | wc -l | tr -d ' ')
[ "$n_lens" = "15" ] || { echo "expected 15 operators in results/ladder1b_b613/, found $n_lens"; exit 1; }
echo "  15 operators present; no fitting, no GPU, no spend"

# score, then adjudicate, in one chain so a partial run cannot be mistaken for a verdict
CMD="$PY $REPO_ROOT/experiments/cv7_1b_rung.py --device cpu \
  && $PY $REPO_ROOT/experiments/cv7_1b_rung.py --adjudicate"

if [ "${1:-}" = "--fg" ]; then
  cd "$REPO_ROOT" && exec caffeinate -i bash -c "$CMD"
fi

cd "$REPO_ROOT"
: > "$LOG"
# setsid detaches from the tty; nohup ignores SIGHUP; caffeinate blocks idle sleep for the
# lifetime of the command only. `command -v setsid` because macOS ships it only with util-linux.
if command -v setsid >/dev/null 2>&1; then
  setsid nohup caffeinate -i bash -c "$CMD" >> "$LOG" 2>&1 &
else
  nohup caffeinate -i bash -c "$CMD" >> "$LOG" 2>&1 &
fi
echo $! > "$PIDF"
disown 2>/dev/null || true
echo "  launched detached, pid $(cat "$PIDF")"
echo "  log    : $LOG"
echo "  watch  : tail -f logs/cv7_1b.log"
echo "  status : bash repro/exp/cv7_1b.sh --status"
echo "  it survives closing the terminal, closing the lid, and an ssh drop"
