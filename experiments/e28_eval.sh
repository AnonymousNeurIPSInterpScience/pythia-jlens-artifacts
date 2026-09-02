#!/usr/bin/env bash
# e28_eval.sh — score every E28 lens. Runs ON a box, after its fits finish.
#
#   bash e28_eval.sh              score every e28_*.pt found, resumable
#   bash e28_eval.sh --smoke      one lens, 5 items/set — proves the path before spending
#   bash e28_eval.sh --wait       block until SHARD COMPLETE, then score
#
# WHY THIS EXISTS. E28's shard runner fits lenses and records dispersion. That is the
# x-axis of the two-parameter fit Y = Y_inf,P + c*disp(P)/N. The PRE-REGISTERED primary
# outcome (SPINE.tex sec E28) is per-metric slopes and per-corpus asymptotes of Y --- a
# PERFORMANCE number. Without this pass, E28 cannot decide its own ACCEPT/REJECT rule.
#
# WHAT IT RUNS, per lens:
#   t17_reaggregate.py   read AUC under ALL FOUR aggregations (min, best1L, mean, persist)
#                        from ONE forward pass per prompt
#   t20_ablation_kl.py   W3 ablation KL, arms: jlens / logit / shuffled / rand_match,
#                        and W4 dose (||delta h|| per layer and per position)
#
# WHAT IT CANNOT RUN, declared rather than silently skipped:
#   W1 swap      never produced a clean number (E5 retracted; E8 gave 1 of 8 cells)
#   W2 steering  never run at any scale
# The prereg says "ALL read and ALL write metrics are RUN". This pass meets that for
# reads and for W3+W4, and does NOT meet it for W1/W2. compliance.json records this.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PY:-/opt/conda/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
export NVIDIA_TF32_OVERRIDE=0          # 10-bit mantissa breaks the anchor gate's 2e-5 tolerance
DEV="${DEV:-cuda}"
OUT="results"
EVALS="multihop,multilingual,order-ops,poetry,typo,association"
LOG="e28_eval.log"

SMOKE=0; WAIT=0
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    --wait)  WAIT=1 ;;
    *) echo "unknown flag $a"; exit 2 ;;
  esac
done

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }

if [ "$WAIT" = "1" ]; then
  say "waiting for SHARD COMPLETE in ../e28.log"
  while ! grep -qa "SHARD COMPLETE" ../e28.log 2>/dev/null; do sleep 60; done
  say "fits finished; starting evaluation"
fi

LENSES=( $(ls -1 "$OUT"/e28_*.pt 2>/dev/null | grep -v '_ckpt\.pt$' || true) )
if [ "${#LENSES[@]}" -eq 0 ]; then say "no e28_*.pt in $OUT — nothing to score"; exit 1; fi

if [ "$SMOKE" = "1" ]; then
  LENSES=( "${LENSES[0]}" ); MAXITEMS="--max-items 5"
  say "SMOKE: 1 lens, 5 items/set — proving the path, not measuring"
else
  MAXITEMS=""
  say "scoring ${#LENSES[@]} lens(es) on $DEV"
fi

# e28_<corpus>_<model>_n<N>_s<seed>.pt  ->  model id
model_of() {
  case "$1" in
    *_70m_*)  echo "EleutherAI/pythia-70m-deduped" ;;
    *_160m_*) echo "EleutherAI/pythia-160m-deduped" ;;
    *_410m_*) echo "EleutherAI/pythia-410m-deduped" ;;
    *_1b_*)   echo "EleutherAI/pythia-1b-deduped" ;;
    *_1.4b_*) echo "EleutherAI/pythia-1.4b-deduped" ;;
    *_2.8b_*) echo "EleutherAI/pythia-2.8b-deduped" ;;
    *) echo "" ;;
  esac
}

ok=0; skip=0; fail=0; t_start=$(date +%s)
for L in "${LENSES[@]}"; do
  B="$(basename "$L" .pt)"
  M="$(model_of "$B")"
  if [ -z "$M" ]; then say "FAIL $B — cannot infer model from filename"; fail=$((fail+1)); continue; fi

  R_OUT="$OUT/${B}_reads.json"
  W_OUT="$OUT/${B}_w3.json"
  if [ "$SMOKE" = "0" ] && [ -f "$R_OUT" ] && [ -f "$W_OUT" ]; then
    skip=$((skip+1)); continue                       # resumable: already scored
  fi
  [ "$SMOKE" = "1" ] && { R_OUT="/tmp/smoke_reads.json"; W_OUT="/tmp/smoke_w3.json"; }

  t0=$(date +%s)
  # --- reads: all four aggregations from one pass
  if ! $PY t17_reaggregate.py --model "$M" --lens "$L" --out "$R_OUT" \
        >>"$LOG" 2>&1; then
    say "FAIL $B reads"; fail=$((fail+1)); continue
  fi
  # --- W3 ablation KL + W4 dose, with both controls
  if ! $PY t20_ablation_kl.py --model "$M" --lens "$L" --evals "$EVALS" \
        --device "$DEV" $MAXITEMS --out "$W_OUT" >>"$LOG" 2>&1; then
    say "FAIL $B w3"; fail=$((fail+1)); continue
  fi
  ok=$((ok+1))
  el=$(( $(date +%s) - t0 ))
  done_n=$((ok+skip)); left=$(( ${#LENSES[@]} - done_n ))
  eta=$(( left * el / 60 ))
  say "ok $B  (${el}s)   $done_n/${#LENSES[@]}  eta ~${eta}m"
done

say "reads+W3 done: $ok scored, $skip already present, $fail failed, $(( $(date +%s) - t_start ))s"

# ---------------------------------------------------------------- compliance record
if [ "$SMOKE" = "1" ]; then
  say "SMOKE: outputs went to /tmp and are NOT counted; skipping the compliance record."
  say "Verify them with: python tests/test_e28_eval_compliance.py /tmp/smoke_reads.json /tmp/smoke_w3.json"
  say "COMPLETE (smoke)"; exit 0
fi
$PY - "$OUT" <<'PYEOF' | tee -a "$LOG"
import glob, json, os, sys
out = sys.argv[1]
lenses = [p for p in glob.glob(os.path.join(out, "e28_*.pt")) if not p.endswith("_ckpt.pt")]
reads  = glob.glob(os.path.join(out, "e28_*_reads.json"))
w3     = glob.glob(os.path.join(out, "e28_*_w3.json"))
rec = {
  "prereg": "SPINE.tex sec E28",
  "prereg_requires": "ALL read and ALL write metrics are RUN",
  "n_lenses": len(lenses), "n_reads_scored": len(reads), "n_w3_scored": len(w3),
  "metrics_run": {
    "read_AUC_min":      "RUN (barred from voting: existential statistic)",
    "read_AUC_best1L":   "RUN",
    "read_AUC_mean":     "RUN",
    "read_AUC_persist":  "RUN (trusted prior standing)",
    "W3_ablation_KL":    "RUN (trusted prior standing), arms jlens/logit/shuffled/rand_match",
    "W4_dose_delta_h":   "RUN as a covariate; never votes on the slope",
  },
  "metrics_NOT_run": {
    "W1_swap":     "E5 retracted (three measurement defects); E8 gave 1 admissible cell of 8. "
                   "No clean number exists at any scale.",
    "W2_steering": "never run at any scale.",
  },
  "compliance": "PARTIAL — reads and W3/W4 meet the prereg; W1 and W2 do not. "
                "Recorded rather than silently skipped.",
  "controls_available_per_cell": ["shuffled (J^shuf)", "rand_match (norm-matched random)"],
}
p = os.path.join(out, "e28_compliance.json")
json.dump(rec, open(p, "w"), indent=1)
print(f"  compliance: {len(reads)} reads / {len(w3)} W3 of {len(lenses)} lenses -> {p}")
print(f"  {rec['compliance']}")
PYEOF

say "COMPLETE"
