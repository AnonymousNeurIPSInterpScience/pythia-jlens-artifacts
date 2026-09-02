#!/usr/bin/env bash
# repro/20_cost_estimate.sh — price a run BEFORE renting anything. Free, instant, offline.
#
#   bash repro/20_cost_estimate.sh <model> <n_prompts> [gpu] [n_boxes]
#   bash repro/20_cost_estimate.sh 410m 200            # one lens
#   bash repro/20_cost_estimate.sh 410m 1575 L40S 4    # one E28 shard on four boxes
#   bash repro/20_cost_estimate.sh --table             # every model at N=200, every GPU
#   bash repro/20_cost_estimate.sh --experiment e28    # a named experiment, fully costed
#
# THE MODEL, stated so you can disagree with it:
#   s/prompt = K_gpu * n_layers * d_model^2 / 1e6
# K is MEASURED per GPU, not derived from spec sheets. See repro/lib/common.sh for the
# three anchors and COMPUTE.md for why fp32 TFLOPS is the right axis and TF32 is banned.
#
# Fit time is linear in N to within 2% (E28: 3.04-3.11 s/prompt across N=50..400), so
# these numbers are interpolation, not hope.

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
need_py

BUDGET_USD=40      # repo budget gate: pause and report above these
BUDGET_HOURS=6

case "${1:---table}" in
  --table)      MODE=table ;;
  --experiment) MODE=experiment; EXP="${2:-e28}" ;;
  *)            MODE=single ;;
esac

"$PY" - "$MODE" "${1:-}" "${2:-}" "${3:-L40S}" "${4:-1}" "${EXP:-}" "$BUDGET_USD" "$BUDGET_HOURS" <<'PYEOF'
import sys
mode, a1, a2, gpu, nbox, exp, budget_usd, budget_h = sys.argv[1:9]
nbox = int(nbox); budget_usd = float(budget_usd); budget_h = float(budget_h)

SHAPE = {"70m":(6,512), "160m":(12,768), "410m":(24,1024), "1b":(16,2048),
         "1.4b":(24,2048), "2.8b":(32,2560), "6.9b":(32,4096), "12b":(36,5120)}
VRAM  = {"70m":3,"160m":4,"410m":8,"1b":14,"1.4b":18,"2.8b":32,"6.9b":64,"12b":96}
# K (s per prompt per million layer*d^2 units), and typical $/h seen on vast.ai
GPU = {  # name        K      provenance      $/h    VRAM  fp32 TFLOPS
  "L40S":     (0.122, "measured",     0.80,  48,  91.6),
  "H100":     (0.125, "measured",     2.40,  80,  67.0),
  "A100":     (0.320, "measured",     1.10,  80,  19.5),
  "RTX4090":  (0.135, "extrapolated", 0.40,  24,  82.6),
}

def cost(model, n, gpu, nbox=1):
    L, d = SHAPE[model]; k, prov, dph, vram, tf = GPU[gpu]
    sec = k * L * d * d / 1e6 * n
    gpu_h = sec / 3600
    wall_h = gpu_h / nbox
    return dict(model=model, n=n, gpu=gpu, nbox=nbox, sper=k*L*d*d/1e6,
                gpu_h=gpu_h, wall_h=wall_h, usd=gpu_h*dph, dph=dph,
                fits=vram >= VRAM[model], vram_need=VRAM[model], vram_have=vram,
                prov=prov, tf=tf)

def show(c):
    print(f"\n  model {c['model']}   N={c['n']}   {c['gpu']} x{c['nbox']}")
    print(f"  {'-'*66}")
    print(f"  per-prompt        {c['sper']:.3f} s     ({c['prov']} constant for this GPU)")
    print(f"  GPU-hours         {c['gpu_h']:.2f}")
    print(f"  wall-clock        {c['wall_h']:.2f} h    (across {c['nbox']} box"
          f"{'es' if c['nbox']>1 else ''})")
    print(f"  cost              ${c['usd']:.2f}   at ${c['dph']:.2f}/h/box")
    print(f"  VRAM              needs {c['vram_need']} GB, {c['gpu']} has {c['vram_have']} GB"
          f"   -> {'FITS' if c['fits'] else 'DOES NOT FIT'}")
    flags = []
    if c['usd']  > budget_usd: flags.append(f"cost ${c['usd']:.2f} EXCEEDS the ${budget_usd:.0f} gate")
    if c['wall_h'] > budget_h: flags.append(f"wall {c['wall_h']:.1f} h EXCEEDS the {budget_h:.0f} h gate")
    if not c['fits']:          flags.append("model does not fit in VRAM")
    if flags:
        print("\n  ** BUDGET GATE **  pause and report before running:")
        for f in flags: print(f"     - {f}")
    else:
        print(f"  budget gate       clear (under ${budget_usd:.0f} and {budget_h:.0f} h)")

if mode == "single":
    model, n = a1, int(a2)
    if model not in SHAPE: sys.exit(f"unknown model {model}; use {' '.join(SHAPE)}")
    if gpu not in GPU:     sys.exit(f"unknown gpu {gpu}; use {' '.join(GPU)}")
    show(cost(model, n, gpu, nbox))
    print("\n  same run on other hardware:")
    for g in GPU:
        c = cost(model, n, g, nbox)
        mark = "" if c['fits'] else "   (does not fit)"
        print(f"    {g:<9} {c['gpu_h']:6.2f} GPU-h   ${c['usd']:7.2f}   "
              f"{GPU[g][1]:<13}{mark}")

elif mode == "table":
    print("\n  ONE LENS AT N=200 — the standard fit. GPU-hours / dollars.\n")
    print(f"  {'model':<7}{'layers':>7}{'d_model':>8}{'VRAM':>6}   " +
          "".join(f"{g:>16}" for g in GPU))
    print("  " + "-"*(28 + 16*len(GPU)))
    for m in SHAPE:
        L, d = SHAPE[m]
        row = f"  {m:<7}{L:>7}{d:>8}{VRAM[m]:>5}G   "
        for g in GPU:
            c = cost(m, 200, g)
            row += f"{c['gpu_h']:>7.2f}h ${c['usd']:>6.2f}" if c['fits'] else f"{'--- no fit':>16}"
        print(row)
    print("\n  fp32 TFLOPS (the axis that matters — this workload is compute-bound,")
    print("  arithmetic intensity 482 FLOP/byte vs an A100 ridge point of 9.6):")
    for g,(k,p,dph,v,tf) in GPU.items():
        print(f"    {g:<9} {tf:>6.1f} TF   ${dph:.2f}/h   {tf/dph:>6.1f} TFLOPS per dollar-hour   [{p}]")

elif mode == "experiment":
    EXPS = {
      "e28":  dict(desc="two-parameter fit: 5 corpora x 6 N x 3 seeds x 2 models",
                   jobs=[("410m", 1575*15), ("70m", 1575*15)], gpu="L40S", nbox=4),
      "ladder": dict(desc="one lens per Pythia size at N=200",
                   jobs=[(m,200) for m in SHAPE], gpu="L40S", nbox=1),
      "nstar": dict(desc="every lens refit at its law-prescribed N* (eps=5%)",
                   jobs=[("70m",92),("160m",257),("410m",946),("1b",695),
                         ("1.4b",1159),("2.8b",1768)], gpu="L40S", nbox=4),
    }
    e = EXPS.get(exp)
    if not e: sys.exit(f"unknown experiment '{exp}'; use {' '.join(EXPS)}")
    print(f"\n  EXPERIMENT: {exp}  —  {e['desc']}")
    print(f"  {'-'*66}")
    tot_h = tot_usd = 0
    for m, n in e["jobs"]:
        c = cost(m, n, e["gpu"])
        tot_h += c["gpu_h"]; tot_usd += c["usd"]
        print(f"    {m:<6} N={n:<6} {c['gpu_h']:7.2f} GPU-h   ${c['usd']:7.2f}"
              + ("" if c["fits"] else "   ** DOES NOT FIT **"))
    wall = tot_h / e["nbox"]
    print(f"  {'-'*66}")
    print(f"    TOTAL       {tot_h:7.2f} GPU-h   ${tot_usd:7.2f}")
    print(f"    on {e['nbox']} x {e['gpu']}: {wall:.2f} h wall")
    if tot_usd > budget_usd or wall > budget_h:
        print(f"\n  ** BUDGET GATE ** — above ${budget_usd:.0f} or {budget_h:.0f} h. "
              "Pause and report before launching.")
    else:
        print(f"    budget gate clear")
PYEOF

cat <<'EOF'

  Prices are the $/h typically seen on vast.ai, not a quote. For live offers:
      bash repro/11_vast_find_offers.sh L40S
EOF
