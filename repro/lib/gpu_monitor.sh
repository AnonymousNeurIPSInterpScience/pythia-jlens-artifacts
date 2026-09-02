#!/usr/bin/env bash
# gpu_monitor.sh — sample GPU utilisation on every provisioned box and judge it against
# what the roofline says we should expect. Runs whenever instances exist.
#
#   bash repro/lib/gpu_monitor.sh                 one sample across the fleet
#   bash repro/lib/gpu_monitor.sh --watch 300     sample every 300 s until no boxes remain
#
# WHY A THRESHOLD AT ALL. "The box is on" is not evidence it is working. E28's evaluation
# ran 25 minutes at 0% GPU because one runner had no --device flag and the other crashed on
# a device mismatch, and neither failure was visible from the outside. A number with a stated
# expectation is the difference between watching and checking.
#
# WHAT TO EXPECT, from the arithmetic intensity of this workload:
#
#   Jacobian fit:  AI = 482 FLOP/byte, measured. An A100's roofline ridge point is 9.6, an
#   L40S's is ~106 (91.6 TFLOP/s fp32 / 864 GB/s). We sit 4.5x to the right of the L40S ridge
#   and 50x to the right of the A100's, so the job is COMPUTE-BOUND on both:
#       expect SM utilisation  >= 95%      (measured: 99-100% on 4x L40S during E28 fits)
#       expect HBM utilisation <= 40%      (bandwidth is NOT the constraint; high HBM with
#                                           low SM means we are memory-stalled and something
#                                           is wrong -- e.g. dim_batch too small)
#       expect power at cap                (E28: 343-348 W against a 350 W TDP, throttle
#                                           reason SwPowerCap -- correct for a compute-bound
#                                           fp32 job, NOT a fault)
#
#   Read/write eval: many small forward passes, LATENCY-bound rather than throughput-bound.
#   Utilisation is legitimately lower, but must not be near zero:
#       expect SM utilisation  >= 40%
#       floor for ANY job      >= 80% is the flag line the operator set; below that this
#                              script reports FLAG and dumps a diagnosis.
#
# EXIT CODE: 1 if any box is below the floor, so a caller can gate on it.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

FLOOR="${FLOOR:-80}"            # operator-set flag line, % SM utilisation
HBM_CEIL="${HBM_CEIL:-40}"      # above this WITH low SM means memory-stalled
WATCH=0; PERIOD=300
[ "${1:-}" = "--watch" ] && { WATCH=1; PERIOD="${2:-300}"; }

sample() {
  local aliases; aliases=($(cut -f1 "$REPO_ROOT/.fleet" 2>/dev/null || true))
  if [ "${#aliases[@]}" -eq 0 ]; then echo "  no fleet — nothing provisioned"; return 0; fi
  local flagged=0
  printf "  %-8s %6s %6s %8s %7s %7s  %s\n" alias SM% HBM% mem power temp verdict
  for a in "${aliases[@]}"; do
    [ -n "$a" ] || continue
    local r
    r="$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$a" \
        'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader,nounits' \
        2>/dev/null | head -1)"
    if [ -z "$r" ]; then printf "  %-8s %s\n" "$a" "UNREACHABLE"; flagged=1; continue; fi
    local sm hbm mem pw tmp thr
    sm=$(echo "$r"  | cut -d, -f1 | tr -d ' ')
    hbm=$(echo "$r" | cut -d, -f2 | tr -d ' ')
    mem=$(echo "$r" | cut -d, -f3 | tr -d ' ')
    pw=$(echo "$r"  | cut -d, -f4 | tr -d ' ')
    tmp=$(echo "$r" | cut -d, -f5 | tr -d ' ')
    thr=$(echo "$r" | cut -d, -f6 | tr -d ' ')
    local verdict
    if   [ "${sm:-0}" -ge "$FLOOR" ]; then verdict="ok"
    elif [ "${sm:-0}" -lt 5 ];        then verdict="FLAG idle"; flagged=1
    elif [ "${hbm:-0}" -gt "$HBM_CEIL" ]; then verdict="FLAG memory-stalled"; flagged=1
    else verdict="FLAG under-utilised"; flagged=1; fi
    printf "  %-8s %5s%% %5s%% %7sM %6sW %6sC  %s\n" "$a" "$sm" "$hbm" "$mem" "$pw" "$tmp" "$verdict"
    [ "$verdict" != "ok" ] && diagnose "$a" "$sm" "$hbm" "$mem" "$thr"
  done
  return $flagged
}

diagnose() {
  local a="$1" sm="$2" hbm="$3" mem="$4" thr="$5"
  echo "      diagnosis for $a:"
  [ "${mem:-0}" -lt 100 ] && echo "        - GPU memory is ~empty: no model is resident. The job is on CPU or dead."
  local procs
  procs="$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$a" 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -3' 2>/dev/null)"
  [ -z "$procs" ] && echo "        - no CUDA process attached to the GPU at all" \
                  || echo "        - CUDA procs: $procs"
  local py
  py="$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$a" 'ps -eo args | grep "[p]ython" | head -2 | cut -c1-100' 2>/dev/null)"
  [ -n "$py" ] && echo "        - python running: $py" \
               || echo "        - no python process: the job exited"
  [ "${hbm:-0}" -gt "$HBM_CEIL" ] && [ "${sm:-0}" -lt "$FLOOR" ] && \
    echo "        - HBM ${hbm}% with SM ${sm}%: MEMORY-STALLED. AI says this job should be"
  [ "${hbm:-0}" -gt "$HBM_CEIL" ] && [ "${sm:-0}" -lt "$FLOOR" ] && \
    echo "          compute-bound at 482 FLOP/byte; suspect dim_batch too small or a CPU-side bottleneck."
  [ "$thr" != "0x0000000000000000" ] && [ -n "$thr" ] && \
    echo "        - throttle reasons active: $thr  (0x4 = SwPowerCap, normal at TDP)"
  echo "        - checklist: does the runner take --device? is the model .to(device)? are the"
  echo "          lens/weight tensors on the same device? did a CUDA error get swallowed by a grep?"
}

if [ "$WATCH" = "1" ]; then
  while :; do
    echo "[$(date -u +%H:%M:%SZ)] GPU utilisation"
    sample || true
    [ -s "$REPO_ROOT/.fleet" ] || { echo "  fleet empty — stopping"; break; }
    sleep "$PERIOD"
  done
else
  echo "[$(date -u +%H:%M:%SZ)] GPU utilisation  (floor ${FLOOR}% SM, HBM ceiling ${HBM_CEIL}%)"
  sample
fi
