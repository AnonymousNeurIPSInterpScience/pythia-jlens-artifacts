#!/usr/bin/env bash
# repro/02_verify_local.sh — TIER 0. Free, CPU-only, ~2 minutes. Mandatory.
#
# This is the gate that decides whether any number this repo produces can be believed.
# It proves two different things:
#
#   (a) the machinery is internally consistent   — 196 unit assertions
#   (b) our read path is BIT-IDENTICAL to the anchor's own JacobianLens.apply
#       (max|diff| = 0.00e+00), and the anchor's readout matches hf(...).logits to 2.3e-3
#
# (b) is the one that matters to an auditor. Anyone can write a lens; this shows ours is
# the same operator the original authors published, not a lookalike.
#
# The tiny-first discipline: a failed $2 job is fine, a silently-wrong $50 job is not.
# So this runs the exact code path the GPU jobs use, on CPU-sized input, first.
#
#   bash repro/02_verify_local.sh

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
need_py
cd "$REPO_ROOT"

SUITE=(test_joperator test_anchor_evals test_dispersion test_hooks_and_audit
       test_anchor_fidelity test_write_families test_metrics_shift test_corpora
       test_bootstrap test_write_geometry)

hdr "Test suite (tests/)"
total=0
for t in "${SUITE[@]}"; do
  f="tests/$t.py"
  if [ ! -f "$f" ]; then bad "$t — file missing"; continue; fi
  out="$("$PY" "$f" 2>&1)" && rc=0 || rc=$?
  line="$(echo "$out" | grep -oE '[0-9]+/[0-9]+ PASSED' | tail -1 || true)"
  if [ "$rc" -eq 0 ] && [ -n "$line" ]; then
    ok "$(printf '%-24s %s' "$t" "$line")"
    total=$((total + $(echo "$line" | cut -d/ -f1)))
  elif echo "$out" | grep -q "FileNotFoundError.*results/.*\\.pt"; then
    bad "$t — needs a lens that is not in git"
    info "     Lenses live on the HF mirror by design. Fetch them, then re-run:"
    info "         ./lab fetch          # ~50 MB, the minimum this gate needs"
  else
    bad "$t"
    echo "$out" | tail -12 | sed 's/^/        /'
  fi
done
info "$total assertions passed"

hdr "What the anchor-fidelity test actually asserts"
cat <<'EOF'
  - our transported activation equals jlens.lens.JacobianLens.apply to max|diff| 0.00e+00
  - the anchor's own readout reproduces hf(...).logits to 2.3e-3
  - fitting honours skip_first (the released code hardcodes 16 at jlens/fitting.py:42,
    while the paper text says 0 — we use 16 and disclose it)
  - no centering anywhere: mu never enters a lens vector or a write
EOF

hdr "Environment recorded for this run"
"$PY" - <<'EOF'
import torch, transformers, platform, sys
print(f"   python       {sys.version.split()[0]}  ({platform.platform()})")
print(f"   torch        {torch.__version__}")
print(f"   transformers {transformers.__version__}")
print(f"   TF32 allowed {torch.backends.cuda.matmul.allow_tf32}  (must be False on GPU: "
      f"a 10-bit mantissa breaks the 2e-5 anchor tolerance)")
EOF

summary && {
  hdr "Next"
  cat <<'EOF'
  repro/20_cost_estimate.sh <model> <n> [gpu]   price a run BEFORE renting anything
  repro/10_vast_auth.sh                         set up GPU rental
  repro/30_repo_health.sh                       directory + provenance audit
EOF
}
