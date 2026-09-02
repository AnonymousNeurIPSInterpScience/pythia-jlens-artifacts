#!/usr/bin/env python3
"""t14_dispersion_ladder.py — is the dispersion profile scale-invariant under relative depth?

RESEARCH_NOTES.tex calls Jacobian dispersion the spine of the paper. Until 2026-08-10 it had never been
measured on a single ladder lens (the accumulator postdated every fit). It now exists for the
penultimate-target lenses, and the raw layer-0 values grow with model size:

    pythia-1b   (14 source layers)   disp(0) = 1.736
    pythia-1.4b (22 source layers)   disp(0) = 2.897
    pythia-2.8b (30 source layers)   disp(0) = 4.418

That growth is confounded: layer 0 of a 30-layer stack is simply *further from the target* than
layer 0 of a 14-layer stack. The question this answers is whether the profile collapses onto one
curve when indexed by RELATIVE depth `l / (n_source - 1)`, which is the same normalization the
anchor uses for the workspace band (L38-L92 on a 0-100 scale).

Reports, per model:
  - dispersion at fixed relative depths (linear interpolation between fitted layers)
  - the fraction of per-prompt magnitude that CANCELS in the average, 1 - 1/sqrt(1+disp)
  - heavy-tail ratio at the same relative depths, so magnitude-outlier effects are visible
  - dispersion inside the anchor's normalized workspace band

Reads only `lens_*_provenance.json`. No model, no GPU, no forward pass.

EXPLORATORY.
"""
from __future__ import annotations

import argparse, glob, json, os


def interp(vals: list[float], rel: float) -> float:
    """Value at relative position `rel` in [0,1] along `vals`, linearly interpolated."""
    if len(vals) == 1:
        return vals[0]
    x = rel * (len(vals) - 1)
    lo = int(x)
    if lo >= len(vals) - 1:
        return vals[-1]
    frac = x - lo
    return vals[lo] * (1 - frac) + vals[lo + 1] * frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/lenses/ladder/lens_*_db128_pen_provenance.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    models = []
    for p in sorted(glob.glob(a.glob)):
        d = json.load(open(p))
        disp = d.get("jacobian_dispersion")
        if not disp:
            continue
        keys = sorted(disp, key=lambda k: int(k))
        models.append({
            "file": os.path.basename(p), "model": d["model"],
            "n_layers": d["n_layers"], "d_model": d["d_model"],
            "n_source": len(keys),
            "target_penultimate": bool(d.get("target_layer_is_penultimate")),
            "dispersion": [disp[k]["dispersion"] for k in keys],
            "heavy_tail": [disp[k]["heavy_tail_ratio"] for k in keys],
        })
    if not models:
        raise SystemExit(f"no provenance files with a dispersion block matched {a.glob!r}")
    models.sort(key=lambda m: m["n_layers"])

    RELS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    print(f"{'model':26s} {'nsrc':>4s} " + " ".join(f"{r:>7.2f}" for r in RELS))
    print("--- dispersion at relative depth ---")
    for m in models:
        vals = [interp(m["dispersion"], r) for r in RELS]
        m["dispersion_at_rel"] = {str(r): round(v, 4) for r, v in zip(RELS, vals)}
        print(f"{m['model'].split('/')[-1]:26s} {m['n_source']:4d} "
              + " ".join(f"{v:7.3f}" for v in vals))

    print("--- fraction of per-prompt magnitude CANCELLED by averaging ---")
    for m in models:
        frac = [1 - (1 + v) ** -0.5 for v in
                [m["dispersion_at_rel"][str(r)] for r in RELS]]
        m["cancelled_fraction_at_rel"] = {str(r): round(f, 4) for r, f in zip(RELS, frac)}
        print(f"{m['model'].split('/')[-1]:26s} {m['n_source']:4d} "
              + " ".join(f"{f*100:6.1f}%" for f in frac))

    print("--- heavy-tail ratio at relative depth (magnitude outliers) ---")
    for m in models:
        vals = [interp(m["heavy_tail"], r) for r in RELS]
        m["heavy_tail_at_rel"] = {str(r): round(v, 4) for r, v in zip(RELS, vals)}
        print(f"{m['model'].split('/')[-1]:26s} {m['n_source']:4d} "
              + " ".join(f"{v:7.3f}" for v in vals))

    # spread across models at each relative depth: does the profile collapse?
    collapse = {}
    for r in RELS:
        vs = [m["dispersion_at_rel"][str(r)] for m in models]
        collapse[str(r)] = {"min": round(min(vs), 4), "max": round(max(vs), 4),
                            "max_over_min": round(max(vs) / min(vs), 3) if min(vs) > 0 else None}

    # dispersion inside the anchor's normalized workspace band (L38-L92 of 0-100)
    for m in models:
        # floor, not round: the declared rule is floor(0.38L)..floor(0.92L) (tests/test_band_rule.py).
        # round() disagrees at 12 and 32 layers and put this file's 160M band at [5,9] where the
        # rule gives [4,9]. Stored results produced before 2026-08-24 carry the old band.
        lo, hi = int(0.38 * m["n_layers"]), int(0.92 * m["n_layers"])
        hi = min(hi, m["n_source"] - 1)
        band = m["dispersion"][lo:hi + 1] if lo <= hi else []
        m["band"] = [lo, hi]
        m["dispersion_in_band"] = ([round(min(band), 4), round(max(band), 4)] if band else None)

    out = {
        "experiment": "T14 — dispersion ladder under relative-depth normalization",
        "status": "EXPLORATORY",
        "source": "lens_*_provenance.json jacobian_dispersion blocks; no model or GPU needed",
        "definition": "disp(l) = E||J_l(x) - Jbar_l||_F^2 / ||Jbar_l||_F^2",
        "cancelled_fraction": "1 - 1/sqrt(1+disp) = fraction of rms per-prompt magnitude lost to averaging",
        "relative_depths": RELS,
        "models": models,
        "collapse_across_models": collapse,
        "note": ("If the profile collapses under relative depth, dispersion is a property of "
                 "FRACTIONAL depth rather than of absolute layer index or model size, and the "
                 "raw growth of disp(layer 0) with model size is a distance-to-target artifact."),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("\n--- collapse check: max/min across models at each relative depth ---")
    for r in RELS:
        c = collapse[str(r)]
        print(f"  rel={r:4.2f}  min={c['min']:7.3f}  max={c['max']:7.3f}  ratio={c['max_over_min']}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
