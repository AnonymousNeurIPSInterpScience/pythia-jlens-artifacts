#!/usr/bin/env bash
# repro/12_vast_provision.sh — rent a box, wait for it, name it, install the toolchain.
#
#   bash repro/12_vast_provision.sh <offer_id> [alias] [disk_gb]
#   bash repro/12_vast_provision.sh 38575672 vast-box 60
#
# THIS SPENDS MONEY. It asks first (FORCE=1 skips the prompt, for automation).
#
# After it returns you have: a running instance, an ~/.ssh/config alias, the id recorded
# in .instance_id, and a verified torch+CUDA toolchain. It does NOT run any experiment.

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
need_py
need_cmd vastai "pip install --user vastai"

OFFER="${1:?usage: 12_vast_provision.sh <offer_id> [alias] [disk_gb]  (find one with 11_vast_find_offers.sh)}"
ALIAS="${2:-$SSH_ALIAS}"
DISK="${3:-60}"
IMAGE="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"

hdr "Offer $OFFER"
# This lookup is ADVISORY ONLY. `vastai search offers` returns a rotating sample, so an ask that
# was live in the search that produced $OFFER is frequently absent from the very next call --
# observed 2026-08-13, five consecutive false "no longer available" aborts on ids that then
# rented fine. The authoritative test of whether an offer exists is `create instance` below.
# Aborting on the advisory pre-check is CLAUDE.md 0b: accepting a proxy for the thing we care
# about. Print what we can, warn when we cannot, and let `create` be the gate.
vastai search offers "id=$OFFER" --raw 2>/dev/null | "$PY" -c "
import json,sys
o=json.load(sys.stdin)
if not o:
    print('  offer not in the current search sample — proceeding; create is the real gate')
    sys.exit(0)
o=o[0]
print(f\"  {o.get('gpu_name')} x{o.get('num_gpus')}   \${o.get('dph_total'):.3f}/h\")
print(f\"  VRAM {int((o.get('gpu_ram') or 0)/1024)} GB   disk {int(o.get('disk_space') or 0)} GB\")
print(f\"  reliability {o.get('reliability2',0):.3f}   CUDA {o.get('cuda_max_good')}   {o.get('geolocation')}\")
if (o.get('cuda_max_good') or 0) < 12.4: print('  ** WARNING: CUDA < 12.4, the cu128 wheels may not load **')
" || warn "offer lookup unavailable — create will decide"

confirm "Rent this box? It bills from the moment it starts."

hdr "Creating instance"
OUT="$(vastai create instance "$OFFER" --image "$IMAGE" --disk "$DISK" \
        --ssh --direct --raw 2>&1)" || die "create failed: $OUT"
ID="$(echo "$OUT" | "$PY" -c "import json,sys; print(json.load(sys.stdin).get('new_contract',''))" 2>/dev/null || true)"
[ -n "$ID" ] || die "could not parse instance id from: $OUT"
echo "$ID" > "$INSTANCE_FILE"
ok "instance $ID created; id written to .instance_id"

hdr "Waiting for it to come up (this is usually 1-3 minutes)"
for i in $(seq 1 60); do
  st="$(vastai show instance "$ID" --raw 2>/dev/null | "$PY" -c \
       "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','?'))" 2>/dev/null || echo '?')"
  printf "\r  [%02d/60] status=%-12s" "$i" "$st"
  [ "$st" = "running" ] && { echo; ok "running"; break; }
  sleep 10
done
[ "$st" = "running" ] || die "instance did not reach 'running'. Check: vastai show instance $ID"

hdr "SSH alias '$ALIAS'"
INST_JSON="$(mktemp)"; trap 'rm -f "$INST_JSON"' EXIT
vastai show instance "$ID" --raw 2>/dev/null > "$INST_JSON"
"$PY" - "$ALIAS" "$INST_JSON" <<'PYEOF'
import json, sys, os, re
alias = sys.argv[1]
d = json.load(open(sys.argv[2]))
host = d.get("ssh_host"); port = d.get("ssh_port")
if not host or not port: sys.exit("  no ssh_host/ssh_port yet — wait a moment and re-run")
cfg = os.path.expanduser("~/.ssh/config")
# IdentityFile is REQUIRED: vast authenticates with a dedicated key, and without this
# line ssh offers nothing and every box reads as "unreachable" despite TCP connecting.
block = (f"Host {alias}\n  HostName {host}\n  Port {port}\n  User root\n"
         f"  IdentityFile ~/.ssh/vast_ed25519\n  IdentitiesOnly yes\n"
         f"  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n"
         f"  ServerAliveInterval 30\n")
old = open(cfg).read() if os.path.exists(cfg) else ""
old = re.sub(rf"(?ms)^Host {re.escape(alias)}\b.*?(?=^Host |\Z)", "", old)
open(cfg, "w").write(old.rstrip() + "\n\n" + block)
print(f"  {alias} -> {host}:{port}")
PYEOF
ok "~/.ssh/config updated"

hdr "Toolchain (torch + CUDA)"
info "The 2.4.0 image's torchvision and torchaudio both break the GPTNeoX import, so all"
info "three are reinstalled together from the cu128 channel. cu124 is DEAD:"
info "nvidia-cudnn-cu12==9.1.0.70 was pulled from the index and torch 2.6.0 pins it exactly."
for i in $(seq 1 12); do ssh -o ConnectTimeout=10 "$ALIAS" true 2>/dev/null && break || sleep 10; done
ssh "$ALIAS" 'bash -s' < "$REPRO_DIR/lib/box_bootstrap.sh" || die "toolchain install failed"

hdr "Verify on the box"
ssh "$ALIAS" "$REMOTE_PY -c \"import torch;print('  torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\"" \
  || die "torch verification failed — do NOT spend on this box"
ok "toolchain live"

hdr "Next"
cat <<EOF
  bash repro/13_vast_preflight.sh $ALIAS    # anchor fidelity ON the box. Do not skip.
  ./lab watch                               # tmux dashboard
  bash repro/15_vast_teardown.sh            # when done — it refuses until artifacts are pulled
EOF
