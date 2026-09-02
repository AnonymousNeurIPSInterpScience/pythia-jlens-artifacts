#!/usr/bin/env python3
"""r8_ladder_flatness.py — R8: does S2 ("reads are FLAT in N") survive the corrected readout?

PRE-REGISTRATION: docs/experiments/preregs/R8_ladder_rstrip_S2.md, committed before the flag existed.
Promoted to Tier A by R1's QUALIFIED verdict, per POSTREVIEW_EXPERIMENTS.md §5's own rule.

THE ESTIMANDS, defined so the denominators cannot drift:
  admitted read for (corpus, seed, N) = mean over the five admitted sets of by_N[N][set][agg]
  asymptote(corpus, seed)             = mean over N >= 75 of that            (e53's convention)
  seed_sd(corpus)                     = population SD over seeds of the asymptote
  range_over_N(corpus)                = max - min over N of the SEED-MEAN curve
  between_corpus_spread               = max - min over corpora of the asymptote mean
  pooled_seed_sd                      = MEAN of per-corpus seed_sd  (e53's convention, verified
                                        against its stored 35.52x at 1B)

    python tools/r8_ladder_flatness.py
"""
from __future__ import annotations
import argparse, glob, json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "src"))
from provenance import sha256_file, write_result  # noqa: E402

R = lambda *p: os.path.join(REPO, *p)

# Paths that go INTO the payload must be repo-relative. An absolute path carries the
# operator's home directory, and unlike the provenance block the payload is covered by
# `payload_sha256`, so redacting it after the fact MOVES a hash that is supposed to be the
# integrity check. This was the last identity leak in results/ after hostname stamping was
# removed on 2026-08-31: one key, `c1_dir`, in this one file.
rel = lambda p: os.path.relpath(os.path.abspath(p), os.path.abspath(REPO))
CORPORA = ["Pile-CC", "StackExchange", "Wikipedia_en", "Github", "USPTO_Backgrounds"]
SEEDS = [0, 1, 2]
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
N_MIN = 75


def load_dir(d):
    out = {}
    for c in CORPORA:
        for s in SEEDS:
            p = os.path.join(d, f"ladder_{c}_s{s}.json")
            if os.path.exists(p):
                out[(c, s)] = json.load(open(p))
    return out


def analyse(cells, ag):
    adm = lambda cell, N: st.mean(cell["by_N"][N][x][ag] for x in ADMITTED)
    per_corpus, curves = {}, {}
    for c in CORPORA:
        seeds = [s for s in SEEDS if (c, s) in cells]
        if not seeds:
            continue
        Ns = sorted(set.intersection(*[set(cells[(c, s)]["by_N"]) for s in seeds]), key=int)
        curve = {int(N): st.mean(adm(cells[(c, s)], N) for s in seeds) for N in Ns}
        asym = [st.mean(adm(cells[(c, s)], N) for N in Ns if int(N) >= N_MIN) for s in seeds]
        curves[c] = curve
        per_corpus[c] = {
            "n_values": sorted(curve), "seed_mean_curve": curve,
            "asymptote_mean": st.mean(asym), "asymptote_per_seed": asym,
            # t53's convention, verbatim (t53_ladder_summary.py:138): SAMPLE sd over 3 seed
            # blocks, i.e. pstdev * sqrt(3/2). Using the population sd instead inflates every
            # ratio below by 1.2247x, which is exactly the discrepancy that first showed up
            # against e53's stored 58.38x.
            "seed_sd": st.pstdev(asym) * math.sqrt(len(asym) / (len(asym) - 1)) if len(asym) > 1
                       else 0.0,
            "range_over_N": max(curve.values()) - min(curve.values()),
            "argmax_N": max(curve, key=curve.get), "argmin_N": min(curve, key=curve.get)}
        sd = per_corpus[c]["seed_sd"]
        per_corpus[c]["range_over_N_in_seed_sd"] = (per_corpus[c]["range_over_N"] / sd) if sd else None
    means = {c: v["asymptote_mean"] for c, v in per_corpus.items()}
    pooled = st.mean([v["seed_sd"] for v in per_corpus.values()])
    spread = max(means.values()) - min(means.values())
    ranges = [v["range_over_N_in_seed_sd"] for v in per_corpus.values() if v["range_over_N_in_seed_sd"]]
    return {"per_corpus": per_corpus, "pooled_seed_sd": pooled,
            "between_corpus_spread": spread,
            "between_corpus_spread_in_seed_sd": (spread / pooled) if pooled else None,
            "range_over_N_in_seed_sd_min": min(ranges) if ranges else None,
            "range_over_N_in_seed_sd_max": max(ranges) if ranges else None,
            "largest_range_over_N_abs": max(v["range_over_N"] for v in per_corpus.values()),
            "N_axis_as_fraction_of_corpus_axis":
                (max(v["range_over_N"] for v in per_corpus.values()) / spread) if spread else None,
            "grand_mean_admitted_read": st.mean(means.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unstripped", default=R("results", "ladder410_cpu"))
    ap.add_argument("--stripped", default=R("results", "ladder410_cpu_rstrip"))
    ap.add_argument("--c1-dir", default=R("results", "r8_c1_unstripped"),
                    help="cells re-scored WITHOUT --rstrip, for C1's exact-equality check")
    ap.add_argument("--out", default=R("results", "r8_ladder_flatness.json"))
    a = ap.parse_args()
    U, S = load_dir(a.unstripped), load_dir(a.stripped)
    if not S:
        raise SystemExit(f"ABORT: no stripped cells in {a.stripped}")
    if len(S) != len(U):
        print(f"  NOTE: {len(U)} unstripped cells vs {len(S)} stripped", flush=True)

    res = {ag: {"unstripped": analyse(U, ag), "stripped": analyse(S, ag)} for ag in ("persist", "min")}

    ctrl = {}
    # C4 — the N grid must be the SAME for every seed of a corpus, or "seed SD" is not a seed
    # quantity. This is checked here because nothing else in the tree checks it: t53 stores
    # `asymptote_grids_identical_across_corpora` (across CORPORA) and never compares across SEEDS.
    ragged = {}
    for c in CORPORA:
        grids = {s: tuple(sorted(int(x) for x in S[(c, s)]["by_N"])) for s in SEEDS if (c, s) in S}
        if len({v for v in grids.values()}) > 1:
            ragged[c] = {s: len(g) for s, g in grids.items()}
    ctrl["C4_N_grid_identical_across_seeds"] = {
        "required": "every seed of a corpus reconstructs the SAME N grid",
        "ragged_corpora": ragged, "fires": not ragged,
        "why_it_matters": (
            "t53_ladder_summary.py:136 averages each seed over ITS OWN N grid. Where the grids "
            "differ, the resulting 'seed SD' conflates seed variation with N-grid variation, and it "
            "is the denominator of the programme's 58.4x headline. Measured: Wikipedia_en seed 2 is "
            "missing its N=800 block (results/lenses/e28/e28_Wikipedia_en_410m_n800_s2.pt does not exist), and "
            "it is the ONLY corpus affected. Under `min` that corpus's seed SD is 1.121e-04 on the "
            "ragged grid against 5.820e-04 on the intersected one -- a 5.2x difference in one of the "
            "five terms of the pooled denominator, which moves spread_over_seed_sd from 55.5 to "
            "49.4. This tool uses the INTERSECTED grid, which is the clean estimand."),
        "convention_used_here": "intersected N grid across seeds"}
    # C1 was a hardcoded literal until 2026-08-20: it asserted max_abs_diff = 0.0 and fires = True
    # without opening a file. R8's pre-registration says "C1 can fail (it is an exact-equality check
    # on ~2000 values)", so a literal is not a control (CLAUDE.md 3, and 6.0b: never accept a proxy
    # for the thing you care about). This now re-scores the cell WITHOUT --rstrip into --c1-dir and
    # compares every (N, eval set, aggregation) value against the stored unstripped ladder.
    C1 = load_dir(a.c1_dir)
    if not C1:
        ctrl["C1_recovery_without_the_flag"] = {
            "required": "the same code without --rstrip reproduces results/ladder410_cpu "
                        "at max_abs_diff = 0.0 over every (N, eval set, aggregation) value",
            "checked": False, "c1_dir": rel(a.c1_dir),
            "fires": False,
            "why_not": (f"no re-scored cells found in {rel(a.c1_dir)}. Produce them with: "
                        f"experiments/e28_ladder_410m.py --device cpu --cells 'Github:0' "
                        f"--out {rel(a.c1_dir)}  (no --rstrip). A control that did not run does not fire.")}
    else:
        worst, worst_at, n_vals, cells = 0.0, None, 0, []
        for key in sorted(C1):
            if key not in U:
                continue
            cells.append(f"{key[0]}|s{key[1]}")
            bn_a, bn_b = C1[key]["by_N"], U[key]["by_N"]
            for nn in sorted(set(bn_a) & set(bn_b), key=int):
                for st_ in sorted(set(bn_a[nn]) & set(bn_b[nn])):
                    for ag in sorted(set(bn_a[nn][st_]) & set(bn_b[nn][st_])):
                        d = abs(bn_a[nn][st_][ag] - bn_b[nn][st_][ag])
                        n_vals += 1
                        if d > worst:
                            worst, worst_at = d, f"{key[0]}|s{key[1]}|N={nn}|{st_}|{ag}"
        ctrl["C1_recovery_without_the_flag"] = {
            "required": "the same code without --rstrip reproduces results/ladder410_cpu "
                        "at max_abs_diff = 0.0 over every (N, eval set, aggregation) value",
            "checked": True, "c1_dir": rel(a.c1_dir), "cells_compared": cells,
            "n_values_compared": n_vals,
            "max_abs_diff": worst, "worst_at": worst_at,
            "fires": n_vals > 0 and worst == 0.0}
    moved = {ag: res[ag]["stripped"]["grand_mean_admitted_read"]
                 / res[ag]["unstripped"]["grand_mean_admitted_read"] for ag in res}
    ctrl["C2_the_convention_bites_here_too"] = {
        "required": "the stripped ladder's admitted mean must be materially above the unstripped one",
        "grand_mean_ratio_stripped_over_unstripped": moved,
        "grid_moved_by_for_comparison": 2.55,
        "fires": all(v > 1.5 for v in moved.values())}
    sds = {ag: res[ag]["stripped"]["pooled_seed_sd"] for ag in res}
    ctrl["C3_seed_sd_does_not_collapse"] = {
        "required": "pooled seed SD non-zero and within an order of magnitude of the unstripped 3.7e-04",
        "stripped_pooled_seed_sd": sds,
        "unstripped_pooled_seed_sd": {ag: res[ag]["unstripped"]["pooled_seed_sd"] for ag in res},
        "fires": all(v > 0 for v in sds.values())}

    frac = {ag: res[ag]["stripped"]["N_axis_as_fraction_of_corpus_axis"] for ag in res}
    both_below_25 = all(v is not None and v < 0.25 for v in frac.values())
    any_at_or_above_1 = any(v is not None and v >= 1.0 for v in frac.values())
    if any_at_or_above_1:
        verdict = (f"OVERTURNED — the largest range over N is at or above the between-corpus spread "
                   f"({frac}). N matters as much as corpus identity and S2 is false. STOP AND ALERT.")
    elif both_below_25:
        verdict = (f"FLAT STANDS — the largest per-corpus range over N is "
                   f"{100*frac['persist']:.1f}% of the between-corpus spread under persist and "
                   f"{100*frac['min']:.1f}% under min, both below the pre-registered 25%. S2 holds "
                   f"at the corrected readout, on this repository's own measurement.")
    else:
        verdict = (f"UNCLEAR — the largest range over N is {frac} of the between-corpus spread, "
                   f"between 25% and 100% under at least one aggregation. Report and stop; do not "
                   f"re-cut (CLAUDE.md §2.9).")

    prereg = "docs/experiments/preregs/R8_ladder_rstrip_S2.md"
    rec = {"experiment": "R8 — the 410M ladder at the corrected readout: does S2 survive?",
           "prereg": prereg, "prereg_sha256": sha256_file(R(prereg)),
           "status": "PRE-REGISTERED",
           "promoted_from": ("POSTREVIEW_EXPERIMENTS.md §5 Tier C item S-1, which promotes itself: "
                             "'If R1 returns QUALIFIED or REJECT this becomes Tier A immediately.' "
                             "R1 returned QUALIFIED."),
           "decision_rule_verbatim": (
               "FLAT STANDS — the largest per-corpus range over N is below 25% of the between-corpus "
               "spread, under both aggregations. | OVERTURNED — the largest range over N is at or "
               "above the between-corpus spread. STOP and alert. | UNCLEAR — anything between 25% "
               "and 100%. Report and stop; do not re-cut."),
           "DISAGREEMENT_the_published_range_does_not_reproduce": (
               "the programme's stated range over N -- 1.2-3.7 seed SD under persist, 1.2-2.3 under "
               "min -- does NOT reproduce under the estimands defined above, on either the full "
               "reconstructed N grid or the fitted-N-only subset, and no script in the tree computes "
               "it. Recomputed on the SAME cells with e53's own seed-SD convention: persist "
               "3.75-7.54 over all reconstructed N and 2.49-6.27 over fitted N only; min 1.53-9.87 "
               "and 1.27-7.02. Reported as a disagreement, NOT adopted (R4's standing rule). It does "
               "not contaminate this experiment's verdict: the registered rule is a RELATIVE "
               "comparison between the N axis and the corpus axis computed with identical estimands "
               "on both conventions, so the stripped-vs-unstripped contrast is valid whether or not "
               "the published absolute number can be located."),
           "note_on_the_strict_reading": (
               "the strict reading of 'flat' -- that N moves the read no more than seed noise -- "
               "ALREADY FAILS at the unstripped readout (published 1.2-3.9 seed SD, all above 1). "
               "'Flat' has always meant small RELATIVE TO THE CORPUS AXIS, and the rule is written "
               "on that comparison. Both readings are reported."),
           "device": "cpu (R3 established the CPU numbers are the ones to print)",
           "n_cells": {"unstripped": len(U), "stripped": len(S)},
           "admitted_sets": ADMITTED, "asymptote_n_min": N_MIN,
           "by_aggregation": res, "controls": ctrl, "VERDICT": verdict}
    write_result(a.out, rec, experiment="R8",
                 inputs=sorted(glob.glob(os.path.join(a.stripped, "*.json"))))

    for ag in ("persist", "min"):
        for arm in ("unstripped", "stripped"):
            r = res[ag][arm]
            print(f"[{ag}/{arm:10s}] range over N = "
                  f"{r['range_over_N_in_seed_sd_min']:.1f}-{r['range_over_N_in_seed_sd_max']:.1f} seed SD | "
                  f"between-corpus spread = {r['between_corpus_spread_in_seed_sd']:.1f} seed SD | "
                  f"N/corpus = {100*r['N_axis_as_fraction_of_corpus_axis']:.1f}%")
    for k, v in ctrl.items():
        print(f"  {k:36s} {v['fires']}")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
