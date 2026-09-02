#!/usr/bin/env bash
# e28_eval_410m.sh — score the 83 410M E28 lenses. Embarrassingly parallel; shard by index.
#
#   bash e28_eval_410m.sh <shard_index> <n_shards>     e.g. 0 4  on the first of four boxes
#   bash e28_eval_410m.sh --smoke                      one lens, proves the GPU path
#
# 410M FIRST, deliberately. It is the only scale with a documented read success, its band is
# 13 layers so `persist` is a genuine non-existential statistic, and its seed SD is ~0.002-0.006
# against effects of ~0.03-0.07 -- so the metric can actually resolve a difference there.
# At 70M the band is TWO layers, which makes persist identical to the disqualified min
# statistic, and the whole scale is below the rank-1 unembedding transition.
#
# READS ONLY. W3 is a separate pass: it is ~50x the cost per lens and its CUDA path was fixed
# but never verified on hardware.
#
# The read metric, for the record: pass@k = mean over ITEMS of the fraction of INTERMEDIATE
# concept tokens whose lens rank is <= k, read at a per-set position (final prompt token, or
# the last newline for poetry). It scores latent concepts, NOT the emitted token.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PY:-/opt/conda/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
export NVIDIA_TF32_OVERRIDE=0
DEV="${DEV:-cuda}"
MODEL="EleutherAI/pythia-410m-deduped"
LOG="e28_eval410_${1:-x}.log"

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }

ALL=( $(ls -1 results/lenses/e28/e28_*_410m_n*_s*.pt 2>/dev/null | grep -v _ckpt || true) )
[ "${#ALL[@]}" -gt 0 ] || { say "no 410m lenses in results/"; exit 1; }

if [ "${1:-}" = "--smoke" ]; then
  say "SMOKE on $DEV: 1 lens, proving the GPU read path before any fleet run"
  L="${ALL[0]}"
  t0=$(date +%s)
  $PY t17_reaggregate.py --model "$MODEL" --lens "$L" --device "$DEV" \
      --out /tmp/smoke410_reads.json 2>&1 | tail -3 | tee -a "$LOG"
  say "smoke took $(( $(date +%s) - t0 ))s"
  $PY - <<'EOF'
import json
d=json.load(open("/tmp/smoke410_reads.json"))
r=d["results"]["multihop"]["jlens"]
print(f"  band={len(d['band'])} layers  persist_K={r['persist_K']}  "
      f"multihop jlens persist={r['persist']:.4f}")
print("  aggregations present:", d["aggregations"])
EOF
  exit 0
fi

IDX="${1:?usage: e28_eval_410m.sh <shard_index> <n_shards>}"
NSH="${2:?usage: e28_eval_410m.sh <shard_index> <n_shards>}"

MINE=(); i=0
for L in "${ALL[@]}"; do [ $((i % NSH)) -eq "$IDX" ] && MINE+=("$L"); i=$((i+1)); done
say "shard $IDX/$NSH: ${#MINE[@]} of ${#ALL[@]} lenses on $DEV"

ok=0; skip=0; fail=0; t_start=$(date +%s)
for L in "${MINE[@]}"; do
  B="$(basename "$L" .pt)"; OUT="results/${B}_reads.json"
  [ -f "$OUT" ] && { skip=$((skip+1)); continue; }
  t0=$(date +%s)
  if $PY t17_reaggregate.py --model "$MODEL" --lens "$L" --device "$DEV" --out "$OUT" >>"$LOG" 2>&1; then
    ok=$((ok+1)); el=$(( $(date +%s) - t0 ))
    left=$(( ${#MINE[@]} - ok - skip ))
    say "ok $B (${el}s)  $((ok+skip))/${#MINE[@]}  eta ~$(( left * el / 60 ))m"
  else
    fail=$((fail+1)); say "FAIL $B"; tail -3 "$LOG" | sed 's/^/      /'
  fi
done
say "SHARD $IDX DONE: $ok scored, $skip already present, $fail failed, $(( $(date +%s) - t_start ))s"
