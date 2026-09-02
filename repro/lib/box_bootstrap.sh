#!/usr/bin/env bash
# repro/lib/box_bootstrap.sh — runs ON a rented box, piped in over ssh. Not run locally.
#
# Every step here exists because its absence cost something:
#  - the trio pin: the 2.4.0 image's torchvision/torchaudio break the GPTNeoX import
#  - cu128 not cu124: nvidia-cudnn-cu12==9.1.0.70 was pulled from the index (2026-08-12)
#  - the explicit verify: a pip that failed into /dev/null once left torch at 2.4.0 on
#    four boxes and all 180 fits failed silently. NEVER report success unconditionally.
#  - NVIDIA_TF32_OVERRIDE=0: a 10-bit mantissa breaks the anchor gate's 2e-5 tolerance

set -euo pipefail
PY=/opt/conda/bin/python          # this image; NOT /venv/main/bin/python (that is vast's template)
cd /workspace

echo "== driver"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "== torch trio (cu128)"
$PY -m pip install -q --force-reinstall torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3

echo "== support libs"
$PY -m pip install -q "transformers>=5.5" datasets accelerate safetensors huggingface_hub 2>&1 | tail -2

echo "== vendored anchor"
if [ ! -d /workspace/jacobian-lens/.git ]; then
  git clone --quiet https://github.com/anthropics/jacobian-lens /workspace/jacobian-lens
fi
git -C /workspace/jacobian-lens checkout --quiet 581d398613e5602a5af361e1c34d3a92ea82ba8e
$PY -m pip install -q -e /workspace/jacobian-lens --no-deps

echo "== VERIFY (this must not be a no-op echo)"
$PY - <<'EOF'
import sys, torch, transformers
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE — do not spend on this box"
assert not torch.backends.cuda.matmul.allow_tf32, "TF32 is ON — forbidden, breaks the anchor gate"
import jlens
print(f"   torch {torch.__version__}  cuda {torch.version.cuda}  {torch.cuda.get_device_name(0)}")
print(f"   transformers {transformers.__version__}")
print(f"   jlens {jlens.__file__}")
print("   TF32 off, CUDA live, jlens importable")
EOF

echo 'export NVIDIA_TF32_OVERRIDE=0' >> ~/.bashrc
echo "BOOTSTRAP OK"
