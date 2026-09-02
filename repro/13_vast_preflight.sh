#!/usr/bin/env bash
# repro/13_vast_preflight.sh — the gate between "a box exists" and "spend money on it".
#
#   bash repro/13_vast_preflight.sh [alias]
#
# Costs a few minutes of rental. Skipping it once cost 180 fits: a pip had failed into
# /dev/null, torch stayed at 2.4.0 on all four boxes, and the shard runner's grep swallowed
# the traceback so every job "completed" with no output.
#
# It checks the four things that have actually gone wrong, on the box, not on your laptop.

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
ALIAS="${1:-$SSH_ALIAS}"

ssh -o ConnectTimeout=15 "$ALIAS" true 2>/dev/null || die "cannot reach '$ALIAS'"

hdr "1. Hardware"
ssh "$ALIAS" 'nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader' | sed 's/^/  /'

hdr "2. Toolchain — torch must NOT be the image's 2.4.0, and CUDA must be live"
ssh "$ALIAS" "$REMOTE_PY - <<'EOF'
import torch, transformers, sys
print(f'  torch {torch.__version__}   cuda_runtime {torch.version.cuda}')
print(f'  transformers {transformers.__version__}')
print(f'  cuda available: {torch.cuda.is_available()}')
if torch.__version__.startswith('2.4.'):
    sys.exit('  FAIL: torch is still the image default 2.4.0 — the pip install did not take')
if not torch.cuda.is_available():
    sys.exit('  FAIL: CUDA not available')
print('  ok')
EOF" || { bad "toolchain"; exit 1; }
ok "toolchain"

hdr "3. TF32 must be OFF"
info "A 10-bit mantissa breaks the anchor gate's 2e-5 tolerance. Silent, and it corrupts"
info "every number the run produces."
# The flag read ALONE is a proxy and it lied twice. torch 2.11 defaults
# `cudnn.allow_tf32` to True while `matmul.allow_tf32` is already False, so a bare interpreter
# reports "TF32 enabled" on a box where every matmul this workload issues is full fp32 -- and
# GPT-NeoX has no convolutions, so cudnn is never reached. Conversely a driver-level override
# could force TF32 on in a way no python flag reveals. So: set the flags the way every experiment
# here sets them (trainval.py / cv6_per_family_ladder.py do it at import), assert they took, and
# then MEASURE -- a 1024x1024 fp32 matmul against a float64 reference. TF32's 10-bit mantissa
# shows up at ~1e-3 relative; true fp32 lands near 1e-7. That number cannot be faked by a flag.
ssh "$ALIAS" "NVIDIA_TF32_OVERRIDE=0 $REMOTE_PY -c \"
import torch,sys
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
    sys.exit('  FAIL: TF32 flags refused to turn off')
if torch.get_float32_matmul_precision() != 'highest':
    sys.exit('  FAIL: float32_matmul_precision=' + torch.get_float32_matmul_precision())
g = torch.Generator(device='cuda').manual_seed(0)
A = torch.randn(1024,1024, device='cuda', dtype=torch.float32, generator=g)
B = torch.randn(1024,1024, device='cuda', dtype=torch.float32, generator=g)
ref = A.double() @ B.double()
err = float(((A@B).double()-ref).abs().max() / ref.abs().max())
print('  matmul.allow_tf32=False cudnn.allow_tf32=False precision=highest')
print('  measured fp32 matmul rel err %.3e (TF32 would be ~1e-3)' % err)
if err > 1e-5:
    sys.exit('  FAIL: matmul is not full fp32 -- rel err %.3e' % err)\"" || { bad "TF32"; exit 1; }
ok "TF32 off, and measured off"

hdr "4. Anchor fidelity ON THE BOX"
info "The same bit-identity assertion the local tier runs. Different hardware, different"
info "kernels — it has to hold here too, or the fits are not the anchor's operator."
if ssh "$ALIAS" "test -f $REMOTE_CWD/tests/test_anchor_fidelity.py" 2>/dev/null; then
  ssh "$ALIAS" "cd $REMOTE_CWD && NVIDIA_TF32_OVERRIDE=0 $REMOTE_PY tests/test_anchor_fidelity.py" \
    | tail -8 | sed 's/^/  /' || { bad "anchor fidelity FAILED — do not spend"; exit 1; }
  ok "anchor fidelity"
else
  warn "code not on the box yet"
  info "FIX: ./lab push $ALIAS      (rsync the code, excluding artifacts and corpora)"
fi

hdr "5. Disk headroom"
ssh "$ALIAS" "df -h /workspace | tail -1" | sed 's/^/  /'
info "A 410M lens is 46 MB; a full E28 shard writes ~1 GB. Running out mid-fit loses the fit."

summary && {
  hdr "Cleared to spend"
  echo "  ./lab run $ALIAS '<command>'    launch inside a durable remote tmux session"
  echo "  ./lab watch                     live dashboard across every box"
}
