#!/usr/bin/env python3
"""build_config_matrix.py — emit the configuration matrix, read from the results files themselves.

WHY
  EXPERIMENT_VALIDITY_AUDIT v2 §7.1: "the single structural weakness this audit found is that no
  document anywhere tabulates which lens, fitter, band, N, k-summary, aggregation and eval-set list
  produced each number." In active use right now: two fitters (fastfit, trainval), two N conventions
  (fixed N=200 vs an asymptote over N>=75), two eval-set lists (5 admitted vs E31's 6), several
  410M bands, and — until E62 — a 1B band that violated the paper's own declared rule.

  A number is only comparable to another number if these agree. This reads each field out of the
  file that carries it, so the matrix cannot drift from the data the way a hand-maintained table
  would.

WHAT IT EMITS
  docs/context/CONFIG_MATRIX.md   the table, plus a DIVERGENCES section listing every field on which two
                           paper-cited files disagree.

  Anything it cannot find is printed as `—` rather than guessed. An unknown configuration is a
  finding, not a blank to fill in.

    python tools/build_config_matrix.py
"""
from __future__ import annotations
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results")

# Files the paper cites, in the order it cites them. Anything here must be comparable.
PAPER_CITED = [
    "e54_aggregation_audit.json", "e52_factorial_410m.json", "e57_grid_variance_ci.json",
    "e57_factorial_cells_410m.json", "e59_read_dose_410m.json", "e61_randomized_null_410m.json",
    "e62_band_adjudication.json", "e55_matrix_robustness.json", "e53_ladder_summary.json",
    "e56_predictor_registry.json", "e58_algebra_audit.json", "e60_fitter_determinism.json",
    "e51_interaction_variance.json", "e48_crossover_410m.json", "e48c_exposure_vs_read.json",
    "e48b_exposure_growth.json", "e36_qladder_410m.json", "e33_logit_baseline_410m_v2.json",
    "e38_jgeometry.json", "e31_local_bakeoff_410m.json", "e48_competence_gate_410m.json",
    "t22_bootstrap_ci_410m.json", "t22_bootstrap_ci_persist_410m.json",
    "t22_bootstrap_ci_160m.json", "t22_bootstrap_ci_persist_160m.json",
]

K_FLAT = [1, 2, 5, 10, 20, 50, 100]
K_ANCHOR = [1, 2, 5, 10, 25, 100]


def band_str(b):
    if not b:
        return "—"
    if isinstance(b, dict):
        return "per-scale"
    try:
        return f"[{min(b)},{max(b)}]"
    except Exception:
        return str(b)[:20]


def dig(d, *keys):
    """First present key, searching one level into obvious containers."""
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] not in (None, [], {}):
            return d[k]
    return None


def describe(path):
    try:
        d = json.load(open(path))
    except Exception as e:
        return {"file": os.path.basename(path), "error": str(e)[:40]}
    if not isinstance(d, dict):
        return {"file": os.path.basename(path), "error": "not an object"}

    K = dig(d, "K")
    ksum = "—"
    if K == K_FLAT:
        ksum = "flat mean, 7 pts"
    elif K == K_ANCHOR:
        ksum = "anchor log-k, 6 pts"
    elif K:
        ksum = f"{len(K)} pts"

    sets = dig(d, "admitted_sets", "eval_sets")
    setstr = "—"
    if sets:
        setstr = f"{len(sets)} admitted" if len(sets) == 5 else f"{len(sets)} sets"
    if "outcome_definition" in d and "SIX" in str(d["outcome_definition"]).upper():
        setstr = "6 sets (!)"

    agg = dig(d, "aggregation")
    if agg is None and "by_aggregation" in d:
        agg = "both"
    if agg is None and isinstance(dig(d, "adjudication"), dict):
        agg = d["adjudication"].get("aggregation")

    return {
        "file": os.path.basename(path),
        "model": str(dig(d, "model") or "—").replace("EleutherAI/pythia-", "").replace("-deduped", ""),
        "band": band_str(dig(d, "band")),
        "N": str(dig(d, "n_max", "N", "asymptote_n_min") or
                 ("200" if "n200" in str(dig(d, "operators", "all_operators_share_one_fitter") or "") else "—")),
        "ksum": ksum,
        "agg": str(agg or "—"),
        "sets": setstr,
        "items": str(dig(d, "n_items") or "—"),
        "prov": "yes" if "provenance" in d else "NO",
        "recompute": "recompute" if d.get("recomputes_not_remeasures") else "measure",
    }


def main() -> int:
    rows = [describe(os.path.join(RES, f)) for f in PAPER_CITED
            if os.path.exists(os.path.join(RES, f))]
    missing = [f for f in PAPER_CITED if not os.path.exists(os.path.join(RES, f))]

    # divergences: fields where paper-cited MEASUREMENT files disagree
    # Band legitimately differs BY MODEL (410m is [9,21], 160m is [5,9]); a divergence is only a
    # finding if two files at the SAME model disagree. Everything else is compared globally.
    div = {}
    for field in ("band", "ksum", "sets", "items"):
        groups = {}
        for r in rows:
            if r.get("error") or r["recompute"] == "recompute":
                continue
            v = r.get(field, "—")
            if v == "—":
                continue
            key = r["model"] if field == "band" else "all"
            groups.setdefault(key, {}).setdefault(v, []).append(r["file"])
        for key, vals in groups.items():
            if len(vals) > 1:
                div[f"{field} @ {key}" if key != "all" else field] = vals

    out = [
        "# CONFIGURATION MATRIX — what produced each number the paper cites",
        "",
        "**Generated by `tools/build_config_matrix.py`. Do not hand-edit.**",
        "",
        "Every field is read out of the results file that carries it. `—` means the file does not",
        "record it, which is itself a finding: a number whose configuration is unrecorded cannot be",
        "compared to one whose configuration is.",
        "",
        "A number is comparable to another number only if these columns agree. `EXPERIMENT_VALIDITY_",
        "AUDIT.md` §7.1 exists because nothing tabulated this and four distinct 160M bands, several",
        "410M bands, two fitters, two N conventions and two eval-set lists were all in simultaneous",
        "use.",
        "",
        "| file | model | band | N | k-summary | aggregation | eval sets | items | provenance | kind |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("error"):
            out.append(f"| `{r['file']}` | — | — | — | — | — | — | — | — | ERROR: {r['error']} |")
            continue
        out.append("| `{file}` | {model} | {band} | {N} | {ksum} | {agg} | {sets} | {items} | "
                   "{prov} | {recompute} |".format(**r))

    out += ["", "## Divergences among paper-cited measurement files", ""]
    if not div:
        out.append("None on the recorded fields.")
    for field, vals in div.items():
        out.append(f"**{field}** — {len(vals)} distinct values in simultaneous use:")
        out.append("")
        for v, fs in sorted(vals.items(), key=lambda kv: -len(kv[1])):
            out.append(f"- `{v}` — {', '.join('`'+x+'`' for x in fs)}")
        out.append("")

    if missing:
        out += ["## Cited but absent from `results/`", ""]
        out += [f"- `{m}`" for m in missing]
        out.append("")

    out += [
        "## Known, deliberate divergences",
        "",
        "- **`t22_bootstrap_ci_410m.json` carries no `aggregation` key** — it is the `min` run; the",
        "  `persist` run is the separate `_persist_` file. The paper cites the pair via a glob.",
        "- **`e31_local_bakeoff_410m.json` scores SIX eval sets**, while every headline asymptote is",
        "  over the five admitted ones. Figure 2's y-axis is therefore not the quantity §grid",
        "  tabulates, and the paper says so.",
        "- **E36's operators are `e28_*_n400_s0.pt`** — N=400, one seed block per corpus — so its",
        "  intervals contain no operator-estimation variance. Its three seeds are prefix draws.",
        "- **The 1B ladder was fitted on [5,13]** where the declared rule gives [6,13]. Corrected by",
        "  E62 (`ladder1b_b613/`); the original `ladder1b/` is retained for the C1 comparison.",
        "",
    ]
    dest = os.path.join(ROOT, "docs", "context", "CONFIG_MATRIX.md")
    open(dest, "w").write("\n".join(out) + "\n")
    print(f"{len(rows)} files tabulated, {len(missing)} cited-but-absent, "
          f"{len(div)} divergent field(s): {list(div)}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
