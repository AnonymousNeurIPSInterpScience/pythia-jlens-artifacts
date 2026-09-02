#!/usr/bin/env bash
# repro/11_vast_find_offers.sh — offer hunting, ranked by what actually matters here.
#
#   bash repro/11_vast_find_offers.sh [GPU] [n_gpus] [max_$/h]
#   bash repro/11_vast_find_offers.sh                # the default sweep across all viable GPUs
#   bash repro/11_vast_find_offers.sh L40S 1 1.00
#
# WHY THE RANKING IS NOT "CHEAPEST". This workload is compute-bound — arithmetic
# intensity 482 FLOP/byte against an A100 ridge point of 9.6, measured at ~100% of fp32
# peak. So the figure of merit is fp32 throughput per dollar-hour, not $/h and not VRAM.
# VRAM only decides whether a model fits at all. A cheap A100 is a bad deal here: it has
# 19.5 fp32 TFLOPS against an L40S's 91.6.
#
# Reliability is not cosmetic. A box that dies at hour 4 of a 5-hour ladder costs the
# whole run, so anything below 0.95 is filtered out by default.

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
need_py
need_cmd vastai "install with: pip install --user vastai"

GPU="${1:-ALL}"; NGPU="${2:-1}"; MAXDPH="${3:-2.50}"
MIN_REL="${MIN_REL:-0.95}"

if [ "$GPU" = "ALL" ]; then QUERY_GPUS=(L40S RTX_4090 H100_PCIE A100_PCIE); else QUERY_GPUS=("$GPU"); fi

RAW="$(mktemp)"; trap 'rm -f "$RAW"' EXIT
echo "[]" > "$RAW"
for g in "${QUERY_GPUS[@]}"; do
  info "querying $g ..."
  vastai search offers "gpu_name=$g num_gpus=$NGPU rentable=true dph_total<$MAXDPH" \
    -o 'dph' --raw 2>/dev/null > "${RAW}.$g" || echo "[]" > "${RAW}.$g"
done

"$PY" - "$RAW" "$MIN_REL" "${QUERY_GPUS[@]}" <<'PYEOF'
import json, sys, os
raw, min_rel = sys.argv[1], float(sys.argv[2])
gpus = sys.argv[3:]

# fp32 TFLOPS. The one number that predicts wall-clock on this workload.
TF32 = {"L40S":91.6, "RTX 4090":82.6, "RTX4090":82.6, "H100 PCIE":67.0, "H100 SXM":67.0,
        "H100 NVL":67.0, "A100 PCIE":19.5, "A100 SXM4":19.5, "A100X":19.5}
def tflops(name):
    for k, v in TF32.items():
        if k.lower().replace(" ","") in (name or "").lower().replace(" ",""): return v
    return None

rows = []
for g in gpus:
    p = f"{raw}.{g}"
    if not os.path.exists(p): continue
    try: offers = json.load(open(p))
    except Exception: continue
    for o in offers:
        rel = o.get("reliability2", 0) or 0
        if rel < min_rel: continue
        dph = o.get("dph_total", 0) or 0
        if dph <= 0: continue
        name = o.get("gpu_name", "?")
        tf = tflops(name)
        rows.append(dict(id=o.get("id"), gpu=name, n=o.get("num_gpus",1), dph=dph,
                         vram=int((o.get("gpu_ram") or 0)/1024), rel=rel,
                         cuda=o.get("cuda_max_good"), geo=o.get("geolocation") or "?",
                         down=o.get("inet_down") or 0, disk=int(o.get("disk_space") or 0),
                         tf=tf, tfpd=(tf/dph if tf else 0)))

if not rows:
    print("\n  no offers matched. Loosen the filters: MIN_REL=0.90 or a higher max $/h.")
    raise SystemExit(0)

rows.sort(key=lambda r: -r["tfpd"])
print(f"\n  {len(rows)} offer(s), reliability >= {min_rel}, ranked by fp32 TFLOPS per dollar-hour")
print(f"  {'':>2} {'id':>10} {'gpu':<14}{'VRAM':>5}{'$/h':>7}{'fp32':>7}{'TF/$h':>8}"
      f"{'rel':>7}{'cuda':>6}{'disk':>6}  location")
print("  " + "-"*94)
for i, r in enumerate(rows[:15]):
    star = "*" if i == 0 else " "
    tf = f"{r['tf']:.0f}" if r["tf"] else "?"
    tfpd = f"{r['tfpd']:.0f}" if r["tfpd"] else "?"
    print(f"  {star} {r['id']:>10} {r['gpu'][:14]:<14}{r['vram']:>4}G{r['dph']:>7.3f}"
          f"{tf:>7}{tfpd:>8}{r['rel']:>7.3f}{str(r['cuda']):>6}{r['disk']:>5}G  {r['geo']}")

b = rows[0]
print(f"""
  BEST VALUE: id={b['id']}  {b['gpu']}  ${b['dph']:.3f}/h  {b['vram']} GB  rel {b['rel']:.3f}

  Sanity before you rent:
    - CUDA >= 12.4. Current boxes ship driver 575.x (CUDA 12.9); the toolchain installer
      uses the cu128 wheel channel. The old cu124 recipe is DEAD -- nvidia-cudnn-cu12
      ==9.1.0.70 was pulled from the index and torch 2.6.0 pins it exactly.
    - disk >= 40 GB. Lenses are large: a 410M lens is 46 MB, a 2.8B lens far more.
    - VRAM must clear the model: 410m needs 8 GB, 1b 14 GB, 2.8b 32 GB, 6.9b 64 GB.
      repro/20_cost_estimate.sh checks this for you.

  Provision it:
      bash repro/12_vast_provision.sh {b['id']}
      ./lab up 4 L40S          # or let the orchestrator pick the top N offers
""")
PYEOF
rm -f "${RAW}".* 2>/dev/null || true
