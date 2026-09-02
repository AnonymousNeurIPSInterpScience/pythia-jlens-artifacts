#!/usr/bin/env python3
"""t62_adjudicate_band.py — E62: does the 1B replication survive the band the paper declares?

PRE-REGISTRATION: docs/experiments/preregs/E62_ladder1b_corrected_band.md, written before the run.
  PRIMARY   the 1B corpus x concept-set interaction share (E51's statistic) on band [6,13],
            against the stored 9.158% on [5,13].
  RULE      CONFIRMS  |delta| < 2 pp AND the interaction still exceeds the corpus main effect
            WEAKENS   still exceeds the main effect but moves by >= 2 pp
            REVERSES  the interaction no longer exceeds the corpus main effect at 1B
            UNCLEAR   C1 or C2 does not fire
  C1  per-layer dispersion for the SHARED layers 6-13 must reproduce the stored ladder1b to
      <= 1e-3. J is fitted per layer and cannot depend on whether layer 5 was also fitted, so a
      failure means something OTHER than the band changed and nothing is comparable.
  C2  every output must record band == [6..13].  C3  n_used per cell must match the stored run.

    python experiments/t62_adjudicate_band.py
"""
from __future__ import annotations
import argparse, glob, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from provenance import write_result  # noqa: E402

RES = os.path.join(HERE, "..", "results")
CORPORA = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
SHARED = list(range(6, 14))
N_MIN = 75
STORED_INTERACTION = 9.158325214755475
STORED_CORPUS_MAIN = 2.2117202251311614


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default=os.path.join(RES, "ladder1b_b613"))
    ap.add_argument("--old", default=os.path.join(RES, "ladder1b"))
    ap.add_argument("--e62-interaction", default=os.path.join(RES, "e62_interaction_b613.json"),
                    help="output of t51_interaction_variance.py --ladder1b-dir <new>")
    ap.add_argument("--out", default=os.path.join(RES, "e62_band_adjudication.json"))
    a = ap.parse_args()

    def load(d):
        out = {}
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            r = json.load(open(f))
            out[(r["corpus"], r["seed"])] = r
        return out

    new, old = load(a.new), load(a.old)
    rec = {"experiment": "E62 — the 1B ladder on the declared band [6,13]",
           "prereg": "docs/experiments/preregs/E62_ladder1b_corrected_band.md",
           "status": "PRE-REGISTERED",
           "n_cells_new": len(new), "n_cells_old": len(old)}

    # ---- C2 band
    bands = {f"{c}|s{s}": v["band"] for (c, s), v in new.items()}
    c2_ok = all(b == SHARED for b in bands.values())
    rec["C2_band"] = {"expected": SHARED, "n_cells": len(bands),
                      "n_matching": sum(1 for b in bands.values() if b == SHARED),
                      "fires": c2_ok}

    # ---- C3 prompt pool
    c3 = {}
    for k in new:
        if k in old:
            c3[f"{k[0]}|s{k[1]}"] = {"new": new[k].get("n_used"), "old": old[k].get("n_used"),
                                     "same": new[k].get("n_used") == old[k].get("n_used")}
    rec["C3_prompt_pool"] = {"per_cell": c3,
                             "fires": all(v["same"] for v in c3.values()) if c3 else False}

    # ---- C1 shared-layer dispersion
    worst, worst_where, n_cmp = 0.0, None, 0
    for k in sorted(new):
        if k not in old:
            continue
        for n in new[k]["by_N"]:
            if n not in old[k]["by_N"]:
                continue
            dn = new[k]["by_N"][n].get("dispersion_per_layer") or {}
            do = old[k]["by_N"][n].get("dispersion_per_layer") or {}
            for l in SHARED:
                ln = str(l)
                if ln in dn and ln in do:
                    d = abs(float(dn[ln]) - float(do[ln])); n_cmp += 1
                    if d > worst:
                        worst, worst_where = d, f"{k[0]}|s{k[1]}|N={n}|L{l}"
    rec["C1_shared_layer_dispersion"] = {
        "layers": SHARED, "n_comparisons": n_cmp, "max_abs_diff": worst,
        "worst_at": worst_where, "tolerance": 1e-3, "fires": n_cmp > 0 and worst <= 1e-3,
        "why": ("J is fitted per layer, so layers 6-13 cannot depend on whether layer 5 was also "
                "fitted. A failure means something other than the band changed."),
    }

    # ---- asymptotes, new vs old
    def asym(store):
        out = {}
        for c in CORPORA:
            per = []
            for s in (0, 1, 2):
                r = store.get((c, s))
                if not r:
                    continue
                vals = []
                for n, blob in r["by_N"].items():
                    if int(n) < N_MIN:
                        continue
                    node = blob["reads"] if "reads" in blob else blob
                    vals.append(st.mean(node[k]["persist"] for k in ADMITTED))
                if vals:
                    per.append(st.mean(vals))
            if per:
                out[c] = {"mean": st.mean(per), "seed_sd": st.stdev(per) if len(per) > 1 else 0.0}
        return out

    an, ao = asym(new), asym(old)
    rec["asymptotes"] = {
        "band_6_13": an, "band_5_13_stored": ao,
        "delta": {c: an[c]["mean"] - ao[c]["mean"] for c in an if c in ao},
        "order_6_13": sorted(an, key=lambda c: -an[c]["mean"]),
        "order_5_13": sorted(ao, key=lambda c: -ao[c]["mean"]),
    }
    rec["asymptotes"]["ordering_unchanged"] = (
        rec["asymptotes"]["order_6_13"] == rec["asymptotes"]["order_5_13"])

    # ---- PRIMARY: the interaction on the corrected band
    prim = None
    if os.path.exists(a.e62_interaction):
        e = json.load(open(a.e62_interaction))
        s1b = e["scales"].get("1b", {}).get("admitted5", {})
        # UNITS. t51 stores these as FRACTIONS (0.0916); e54 and the paper quote the same
        # quantity as a PERCENTAGE (9.158). Comparing the two directly reports a spurious
        # 9-point collapse. Everything below is in percentage points.
        def pct(v):
            return None if v is None else 100.0 * v
        prim = {"frac_interaction": pct(s1b.get("frac_interaction")),
                "frac_corpus_main": pct(s1b.get("frac_corpus_main")),
                "frac_set_main": pct(s1b.get("frac_set_main")),
                "interaction_over_corpus_main": s1b.get("interaction_over_corpus_main"),
                "units": "percent, converted from t51's fractions"}
    rec["PRIMARY"] = {"band_6_13": prim,
                      "band_5_13_stored": {"frac_interaction": STORED_INTERACTION,
                                           "frac_corpus_main": STORED_CORPUS_MAIN}}

    if not (rec["C1_shared_layer_dispersion"]["fires"] and c2_ok):
        verdict = ("UNCLEAR — a control did not fire; the corrected-band ladder is not comparable "
                   "to the stored one and nothing here is interpretable")
    elif prim is None or prim.get("frac_interaction") is None:
        verdict = ("INCOMPLETE — run t51_interaction_variance.py --ladder1b-dir "
                   "results/ladder1b_b613 first")
    else:
        d = prim["frac_interaction"] - STORED_INTERACTION
        rec["PRIMARY"]["delta_pp"] = d
        exceeds = prim["frac_interaction"] > prim["frac_corpus_main"]
        if not exceeds:
            verdict = (f"REVERSES — on the declared band the 1B interaction "
                       f"({prim['frac_interaction']:.3f}%) no longer exceeds the corpus main "
                       f"effect ({prim['frac_corpus_main']:.3f}%). The paper's second scale does "
                       f"not replicate and Limitation 6 must be rewritten.")
        elif abs(d) < 2.0:
            verdict = (f"CONFIRMS — interaction {prim['frac_interaction']:.3f}% on [6,13] vs "
                       f"{STORED_INTERACTION:.3f}% on [5,13] ({d:+.3f} pp), still above the "
                       f"{prim['frac_corpus_main']:.3f}% corpus main effect. The band defect was "
                       f"immaterial; report the corrected number and retire the deviation.")
        else:
            verdict = (f"WEAKENS — interaction moves {d:+.3f} pp to "
                       f"{prim['frac_interaction']:.3f}%, still above the corpus main effect. "
                       f"The 1B replication is band-sensitive and must be reported as such.")
    rec["VERDICT"] = verdict

    write_result(a.out, rec, experiment="E62",
                 inputs=sorted(glob.glob(os.path.join(a.new, "*.json"))))
    print(f"C1 dispersion  : {n_cmp} comparisons, max|diff|={worst:.3e} -> "
          f"{'FIRES' if rec['C1_shared_layer_dispersion']['fires'] else 'DOES NOT FIRE'}")
    print(f"C2 band        : {rec['C2_band']['n_matching']}/{rec['C2_band']['n_cells']} -> "
          f"{'FIRES' if c2_ok else 'DOES NOT FIRE'}")
    print(f"C3 prompt pool : {'FIRES' if rec['C3_prompt_pool']['fires'] else 'DOES NOT FIRE'}")
    print(f"ordering unchanged: {rec['asymptotes']['ordering_unchanged']}")
    print(f"  [6,13] {rec['asymptotes']['order_6_13']}")
    print(f"  [5,13] {rec['asymptotes']['order_5_13']}")
    print(f"\nVERDICT: {verdict}\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
