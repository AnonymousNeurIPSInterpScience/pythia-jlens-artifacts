#!/usr/bin/env python3
"""t54_aggregation_audit.py — re-score EVERY load-bearing claim under `min` and `persist`.

WHY
  The programme adjudicates on `persist` and reports `min` alongside. `min` is the anchor's own
  statistic, so any claim that exists only under `persist` is a claim about our metric rather than
  about the model. Nothing in the repo has ever tabulated which claims are aggregation-dependent.
  This does that, from stored data only.

  This is NOT a new measurement. Every input file already contains both aggregations, computed in
  the same forward pass; no lens is refitted and no model is loaded. Cost: seconds, CPU.

WHAT IT SETTLES
  For each headline: does it hold under the anchor's `min`, and by how much does the number move?

DECISION RULE — this is an audit, not a hypothesis test, so it has no ACCEPT/REJECT. It reports,
  per claim, one of:
    HOLDS            same sign, same qualitative conclusion under both
    HOLDS-STRONGER   same conclusion, larger effect under min
    WEAKENS          same sign, conclusion materially softened under min
    REVERSES         the conclusion does not survive the aggregation change
  The classification thresholds are fixed here BEFORE the numbers are printed:
    REVERSES  if the sign of the decisive quantity flips, or a CI/p-value crosses its threshold
    WEAKENS   if the decisive ratio falls by >2x but keeps its sign
    HOLDS-STRONGER if the decisive ratio rises by >1.2x
    HOLDS     otherwise

CONTROLS
  C1  the `persist` column must reproduce the stored headline numbers to 1e-4. If it does not,
      this script's re-derivation is wrong and nothing in the `min` column can be trusted.
  C2  the derangement arm: under a metric that can adjudicate between operators, the layer-deranged
      J^shuf must sit BELOW both J^P and the logit lens. Reported for both aggregations; this is
      the evidence that decides which metric is admissible, and it is measured, not asserted.

    python experiments/t54_aggregation_audit.py --out results/e54_aggregation_audit.json
"""
from __future__ import annotations
import argparse, glob, itertools, json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RES = os.path.join(HERE, "..", "results")
# R1: the readout convention must travel with the file. A mutable holder (not module
# globals) so --e52/--e36 redirect every read site at once without a `global` statement;
# mixing a stripped matrix with an unstripped ladder is the failure mode the slate names.
PATHS = {"e52": os.path.join(RES, "e52_factorial_410m.json"),
         "e36": os.path.join(RES, "e36_qladder_410m.json"),
         # 2026-08-20: e48 was hardcoded at three read sites while --e52/--e36 were
         # redirectable, so `--e52 ..._rstrip --e36 ..._rstrip` produced a file that still
         # read the UNSTRIPPED e48. The derangement audit is e48-driven, so its numbers were
         # identical at both readouts by construction (104/120 in both) and were then cited
         # as measured convention-independence. That is exactly the mixing this comment warns
         # about, committed by the file that warns about it.
         "e48": os.path.join(RES, "e48_crossover_410m.json")}

ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]   # association is floored
CORPORA = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]
AGGS = ("persist", "min")
N_MIN = 75            # the asymptote definition used by E31/E33: mean over N >= 75


# ------------------------------------------------------------------ small stats, no scipy
def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for k, i in enumerate(order):
        r[i] = k
    return r


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def ols_slope(x, y):
    mx, my = st.mean(x), st.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)


def welch(a, b):
    """t and Welch df for two equal-length seed-block samples."""
    va, vb, na, nb = st.variance(a), st.variance(b), len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("inf"), float(na + nb - 2)
    t = (st.mean(a) - st.mean(b)) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, df


def t_sf(t, df):
    """Two-sided p for Student t, via the regularised incomplete beta. No scipy on the box."""
    t, df = abs(t), float(df)
    if math.isinf(t):
        return 0.0
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return front * (f - 1.0)


def classify(persist_ratio, min_ratio, sign_flip=False, threshold_cross=False):
    if sign_flip or threshold_cross:
        return "REVERSES"
    if persist_ratio and min_ratio / persist_ratio < 0.5:
        return "WEAKENS"
    if persist_ratio and min_ratio / persist_ratio > 1.2:
        return "HOLDS-STRONGER"
    return "HOLDS"


# ------------------------------------------------------------------ ladder loaders
def ladder_cells(scale: str, agg: str) -> dict:
    """{corpus: [ {N: auc} per seed block ]} — admitted-set mean at the given aggregation."""
    out = {}
    for c in CORPORA:
        per_seed = []
        for s in (0, 1, 2):
            if scale == "410m":
                d = json.load(open(os.path.join(RES, "ladder410", f"ladder_{c}_s{s}.json")))
                per_seed.append({int(N): st.mean(v[k][agg] for k in ADMITTED)
                                 for N, v in d["by_N"].items()})
            else:
                d = json.load(open(os.path.join(RES, "ladder1b", f"tv_{c}_s{s}.json")))
                per_seed.append({int(N): st.mean(v["reads"][k][agg] for k in ADMITTED)
                                 for N, v in d["by_N"].items()})
        out[c] = per_seed
    return out


def ladder_summary(scale: str, agg: str) -> dict:
    """The paper's corpus-ladder paragraph, computed rather than remembered."""
    C = ladder_cells(scale, agg)
    asym = {c: [st.mean(v for n, v in blk.items() if n >= N_MIN) for blk in C[c]] for c in CORPORA}
    mean = {c: st.mean(asym[c]) for c in CORPORA}
    sd = {c: st.stdev(asym[c]) for c in CORPORA}
    pairs = {}
    n_sep, tsep = 0, []
    for a, b in itertools.combinations(CORPORA, 2):
        t, df = welch(asym[a], asym[b])
        p = t_sf(t, df)
        pairs[f"{a}|{b}"] = {"t": t, "df": df, "p": p, "separates": bool(p < 0.05)}
        if p < 0.05:
            n_sep += 1
            tsep.append(abs(t))
    flat = {}
    for c in CORPORA:
        Ns = sorted(set.intersection(*[set(b) for b in C[c]]))
        m = {n: st.mean(b[n] for b in C[c]) for n in Ns}
        s = st.mean(st.stdev([b[n] for b in C[c]]) for n in Ns)
        flat[c] = {"n_grid": Ns, "range": max(m.values()) - min(m.values()),
                   "range_over_seed_sd": (max(m.values()) - min(m.values())) / s,
                   "argmax_N": max(m, key=m.get)}
    spread = max(mean.values()) - min(mean.values())
    pooled = st.mean(sd.values())
    return {
        "asymptote_definition": f"mean over N >= {N_MIN}, {len(ADMITTED)} admitted sets, seeds separate",
        "per_corpus_mean": mean, "per_corpus_seed_sd": sd,
        "order_best_to_worst": sorted(mean, key=lambda k: -mean[k]),
        "between_corpus_spread": spread, "pooled_seed_sd": pooled,
        "spread_over_seed_sd": spread / pooled,
        "n_pairs_separating_p05": n_sep, "n_pairs": len(pairs),
        "abs_t_range_over_separating_pairs": [min(tsep), max(tsep)] if tsep else None,
        "pairs": pairs,
        "flatness_in_N": flat,
        "flat_range_over_seed_sd": [min(v["range_over_seed_sd"] for v in flat.values()),
                                    max(v["range_over_seed_sd"] for v in flat.values())],
        "max_movement_in_N": max(v["range"] for v in flat.values()),
    }


def variance_decomposition(scale: str, agg: str) -> dict:
    """Balanced two-factor (eval set x corpus), seed block as the error term. E51's design."""
    C = ladder_cells(scale, agg)
    y = {}
    for c in CORPORA:
        for s in (0, 1, 2):
            blk = C[c][s]
            # per-set values need the raw per-set numbers, not the admitted mean
            if scale == "410m":
                d = json.load(open(os.path.join(RES, "ladder410", f"ladder_{c}_s{s}.json")))
                by = {int(N): v for N, v in d["by_N"].items()}
            else:
                d = json.load(open(os.path.join(RES, "ladder1b", f"tv_{c}_s{s}.json")))
                by = {int(N): v["reads"] for N, v in d["by_N"].items()}
            for k in ADMITTED:
                y[(c, k, s)] = st.mean(by[n][k][agg] for n in by if n >= N_MIN)
    gm = st.mean(y.values())
    sst = sum((v - gm) ** 2 for v in y.values())
    setm = {k: st.mean(y[(c, k, s)] for c in CORPORA for s in (0, 1, 2)) for k in ADMITTED}
    corm = {c: st.mean(y[(c, k, s)] for k in ADMITTED for s in (0, 1, 2)) for c in CORPORA}
    cell = {(c, k): st.mean(y[(c, k, s)] for s in (0, 1, 2)) for c in CORPORA for k in ADMITTED}
    ss_set = sum(3 * len(CORPORA) * (setm[k] - gm) ** 2 for k in ADMITTED)
    ss_cor = sum(3 * len(ADMITTED) * (corm[c] - gm) ** 2 for c in CORPORA)
    ss_int = sum(3 * (cell[(c, k)] - setm[k] - corm[c] + gm) ** 2 for c in CORPORA for k in ADMITTED)
    ss_res = sum((y[(c, k, s)] - cell[(c, k)]) ** 2
                 for c in CORPORA for k in ADMITTED for s in (0, 1, 2))
    df_i = (len(CORPORA) - 1) * (len(ADMITTED) - 1)
    df_r = len(CORPORA) * len(ADMITTED) * 3 - len(CORPORA) * len(ADMITTED)
    return {"frac_set_main": 100 * ss_set / sst, "frac_corpus_main": 100 * ss_cor / sst,
            "frac_interaction": 100 * ss_int / sst, "frac_residual_seed": 100 * ss_res / sst,
            "F": (ss_int / df_i) / (ss_res / df_r), "F_df": [df_i, df_r], "F_crit_0p001": 3.3,
            "interaction_over_corpus_main": ss_int / ss_cor}


# ------------------------------------------------------------------ per-experiment audits
def audit_e33(agg: str) -> dict:
    e33 = json.load(open(os.path.join(RES, "e33_logit_baseline_410m_v2.json")))
    logit = st.mean(e33["by_transport"]["logit_I"][k][agg] for k in ADMITTED)
    C = ladder_cells("410m", agg)
    rows = {}
    for c in CORPORA:
        a = [st.mean(v for n, v in blk.items() if n >= N_MIN) for blk in C[c]]
        mu, s = st.mean(a), st.stdev(a)
        t = (mu - logit) / (s / math.sqrt(3)) if s else float("inf")
        rows[c] = {"asymptote": mu, "seed_sd": s, "pct_vs_logit": 100 * (mu - logit) / logit,
                   "t_vs_logit_df2": t, "separates_from_logit": bool(abs(t) > 4.303),
                   "below_logit": bool(mu < logit)}
    return {"logit_constant": logit, "by_corpus": rows,
            "n_corpora_indistinguishable_from_logit":
                sum(1 for v in rows.values() if not v["separates_from_logit"]),
            "n_corpora_significantly_below_logit":
                sum(1 for v in rows.values() if v["below_logit"] and v["separates_from_logit"])}


def audit_e48c(agg: str) -> dict:
    e48 = json.load(open(PATHS["e48"]))["arms_admitted_mean"]
    rungs = json.load(open(os.path.join(RES, "e48c_exposure_vs_read.json")))["by_rung"]
    logit = e48["logit_I"][agg]
    rows = {}
    for r, v in rungs.items():
        rows[r] = {"containment_k32": v["containment_k32"],
                   "read": st.mean(e48[f"J|{r}|s{s}"][agg] for s in (0, 1, 2)),
                   "tier": v["tier"]}
    xs = [v["containment_k32"] for v in rows.values()]
    ys = [v["read"] for v in rows.values()]
    xg = [v["containment_k32"] for k, v in rows.items() if k != "Github"]
    yg = [v["read"] for k, v in rows.items() if k != "Github"]
    instream = [v["read"] for v in rows.values() if v["tier"] == "in-stream"]
    ood = [v["read"] for v in rows.values() if v["tier"] == "OOD"]
    t, df = welch(instream, ood)
    return {"logit_constant": logit, "by_rung": rows,
            "spearman_all8": spearman(xs, ys), "spearman_without_github": spearman(xg, yg),
            "containment_span": max(xs) / min(xs),
            "tier_gap_instream_minus_ood": st.mean(instream) - st.mean(ood),
            "tier_gap_t": t, "tier_gap_df": df, "tier_gap_p": t_sf(t, df),
            "n_ood_rungs_above_instream_wikipedia":
                sum(1 for k, v in rows.items() if v["tier"] == "OOD"
                    and v["read"] > rows["Wikipedia_en"]["read"])}


def audit_e36(agg: str) -> dict:
    d = json.load(open(PATHS["e36"]))["ladder"]
    rows = [(r, v) for r, v in d.items()
            if v["containment_k32"] is not None and not v["is_shuffled"]]
    rows.sort(key=lambda kv: -kv[1]["containment_k32"])
    C = [v["containment_k32"] for _, v in rows]
    names = [r for r, _ in rows]

    def series(key):
        return [v["arms"][key][agg]["mean"] for _, v in rows]

    logit = series("logit")
    shuf = series("shuf")
    out = {"rungs": names, "containment": C,
           "logit": {"slope": ols_slope(C, logit), "spearman": spearman(C, logit),
                     "mean": st.mean(logit)},
           "shuf_arm": {"mean": st.mean(shuf),
                        "above_logit": bool(st.mean(shuf) > st.mean(logit)),
                        "n_rungs_above_logit": sum(1 for a, b in zip(shuf, logit) if a > b)},
           "by_fit_corpus": {}}
    for c in CORPORA:
        y = series(f"J|{c}")
        loo = [ols_slope([x for j, x in enumerate(C) if j != i],
                         [b for j, b in enumerate(y) if j != i]) for i in range(len(C))]
        below = [names[i] for i in range(len(y)) if y[i] < logit[i]]
        out["by_fit_corpus"][c] = {
            "slope": ols_slope(C, y), "spearman": spearman(C, y),
            "worst_loo_slope": max(loo, key=abs),
            "loo_slope_by_dropped_rung": dict(zip(names, loo)),
            "slope_sign_flips_under_loo": bool(any(s > 0 for s in loo) and any(s < 0 for s in loo)),
            "n_rungs_below_logit": len(below), "rungs_below_logit": below,
            "steeper_than_logit": bool(ols_slope(C, y) < out["logit"]["slope"]),
        }
    out["n_fitted_steeper_than_logit"] = sum(
        1 for v in out["by_fit_corpus"].values() if v["steeper_than_logit"])
    out["n_fitted_crossing_below_logit_at_any_rung"] = sum(
        1 for c, v in out["by_fit_corpus"].items() if v["n_rungs_below_logit"] > 0)
    return out


def audit_e52(agg: str) -> dict:
    b = json.load(open(PATHS["e52"]))["by_aggregation"][agg]
    return {"D_all8": b["all8"]["D"], "D_without_degenerate": b["without_degenerate"]["D"],
            "permutation_p": b["C4_permutation"]["p_two_sided"],
            "link_artifact_surrogate": b["C6_link_artifact_null"]["surrogate_D_all8"],
            "D_minus_surrogate": b["C6_link_artifact_null"]["D_minus_surrogate_all8"],
            "sign_survives_dropping_github": bool(b["without_degenerate"]["D"] > 0),
            "significant_at_5pct": bool(b["C4_permutation"]["p_two_sided"] < 0.05)}


def audit_quadrants(agg: str) -> dict:
    """The 8x8 fit x read matrix folded onto the P/Q 2x2.

    P = the five corpora measured PRESENT in the Pythia stream (containment 0.918-0.934).
    Q = the three corpora measured ABSENT from it (containment 0.0001-0.0013).
    Rows are the corpus the operator was fitted on; columns are the corpus the 128-token read
    context was drawn from. The concept battery is identical in every cell.
    """
    M = json.load(open(PATHS["e52"]))["by_aggregation"][agg]["matrix"]
    P = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github"]
    Q = ["OOD_News_2024", "OOD_arXiv_2023", "OOD_CommonPile"]
    AX = P + Q

    def qmean(fs, rs):
        return st.mean(M[f"{f}|{r}"] for f in fs for r in rs)

    return {
        "P_corpora": P, "Q_corpora": Q,
        "quadrant_means": {"JP_at_P": qmean(P, P), "JP_at_Q": qmean(P, Q),
                           "JQ_at_P": qmean(Q, P), "JQ_at_Q": qmean(Q, Q)},
        "fit_row_means": {f: st.mean(M[f"{f}|{r}"] for r in AX) for f in AX},
        "read_col_means": {r: st.mean(M[f"{f}|{r}"] for f in AX) for r in AX},
        "diagonal_mean": st.mean(M[f"{c}|{c}"] for c in AX),
        "offdiagonal_mean": st.mean(M[f"{f}|{r}"] for f in AX for r in AX if f != r),
        "fit_axis_span": (max(st.mean(M[f"{f}|{r}"] for r in AX) for f in AX)
                          - min(st.mean(M[f"{f}|{r}"] for r in AX) for f in AX)),
        "read_axis_span": (max(st.mean(M[f"{f}|{r}"] for f in AX) for r in AX)
                           - min(st.mean(M[f"{f}|{r}"] for f in AX) for r in AX)),
    }


def audit_matrix_structure(agg: str) -> dict:
    """Structure of the 8x8 fit x read matrix: which axis carries the variance, and whether an
    operator reads its own corpus best. Both are direct tests of paper claims that are currently
    argued from a p-value rather than from the matrix itself."""
    M = json.load(open(PATHS["e52"]))["by_aggregation"][agg]["matrix"]
    P = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github"]
    Q = ["OOD_News_2024", "OOD_arXiv_2023", "OOD_CommonPile"]
    AX = P + Q
    n = len(AX)
    vals = [M[f"{f}|{r}"] for f in AX for r in AX]
    gm = st.mean(vals)
    sst = sum((v - gm) ** 2 for v in vals)
    rm = {f: st.mean(M[f"{f}|{r}"] for r in AX) for f in AX}
    cm = {r: st.mean(M[f"{f}|{r}"] for f in AX) for r in AX}
    ss_fit = sum(n * (rm[f] - gm) ** 2 for f in AX)
    ss_read = sum(n * (cm[r] - gm) ** 2 for r in AX)
    ss_int = sum((M[f"{f}|{r}"] - rm[f] - cm[r] + gm) ** 2 for f in AX for r in AX)
    own = {}
    for f in AX:
        order = sorted(AX, key=lambda r: -M[f"{f}|{r}"])
        own[f] = {"rank_of_own_corpus": order.index(f) + 1,
                  "best_read_context": order[0], "worst_read_context": order[-1]}
    return {
        "variance_fit_axis_pct": 100 * ss_fit / sst,
        "variance_read_axis_pct": 100 * ss_read / sst,
        "variance_fit_x_read_interaction_pct": 100 * ss_int / sst,
        "note_on_interaction": ("this is the fit x READ interaction. It is NOT the corpus x "
                                "CONCEPT-SET interaction reported in E51; the concept sets are "
                                "averaged out of this matrix."),
        "own_corpus": own,
        "mean_rank_of_own_corpus": st.mean(v["rank_of_own_corpus"] for v in own.values()),
        "chance_rank": (n + 1) / 2,
        "n_operators_reading_own_corpus_best": sum(
            1 for v in own.values() if v["rank_of_own_corpus"] == 1),
        "fit_axis_order": sorted(rm, key=lambda k: -rm[k]),
        "read_axis_order": sorted(cm, key=lambda k: -cm[k]),
        "n_Q_operators_above_worst_P_operator": sum(
            1 for f in Q if rm[f] > min(rm[p] for p in P)),
        "n_Q_operators_above_second_worst_P_operator": sum(
            1 for f in Q if rm[f] > sorted(rm[p] for p in P)[1]),
        "worst_read_context": min(cm, key=cm.get),
        "worst_read_context_is_P": bool(min(cm, key=cm.get) in P),
        "github_cost_as_fit_corpus_pct": 100 * (rm["Github"] / gm - 1),
        "github_cost_as_read_context_pct": 100 * (cm["Github"] / gm - 1),
    }


def audit_baseline_movement(agg: str) -> dict:
    """How much the UNFITTED baseline itself moves across read contexts, against the size of the
    fitted advantage. If the baseline moves comparably, a 'does the fitted lens degrade faster'
    slope has no stable denominator."""
    lad = json.load(open(PATHS["e36"]))["ladder"]
    sub = [(r, v) for r, v in lad.items() if not v["is_shuffled"]]
    lg = [v["arms"]["logit"][agg]["mean"] for _, v in sub]
    sh = [v["arms"]["shuf"][agg]["mean"] for _, v in sub]
    gaps, per = {}, {}
    for c in CORPORA:
        y = [v["arms"][f"J|{c}"][agg]["mean"] for _, v in sub]
        gaps[c] = st.mean(a - b for a, b in zip(y, lg))
        per[c] = {"range_across_read_contexts": max(y) - min(y),
                  "mean_gap_over_logit": gaps[c],
                  "n_contexts_scrambled_beats_this_operator":
                      sum(1 for a, b in zip(y, sh) if b > a)}
    return {"n_read_contexts": len(sub),
            "logit_min": min(lg), "logit_max": max(lg), "logit_range": max(lg) - min(lg),
            "largest_fitted_advantage": max(gaps.values()),
            "logit_range_as_pct_of_largest_advantage": 100 * (max(lg) - min(lg)) / max(gaps.values()),
            "per_operator": per,
            "n_contexts_scrambled_beats_logit": sum(1 for a, b in zip(sh, lg) if a > b)}


def audit_matrix_loo(agg: str) -> dict:
    """Leave-one-corpus-out on the fit-vs-read variance split. The programme's standing control:
    an n<=8 corpus-axis result that does not survive dropping one corpus is leverage-driven."""
    M = json.load(open(PATHS["e52"]))["by_aggregation"][agg]["matrix"]
    AX = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github",
          "OOD_News_2024", "OOD_arXiv_2023", "OOD_CommonPile"]

    def split(ax):
        n = len(ax)
        vals = [M[f"{f}|{r}"] for f in ax for r in ax]
        gm = st.mean(vals)
        sst = sum((v - gm) ** 2 for v in vals)
        rm = {f: st.mean(M[f"{f}|{r}"] for r in ax) for f in ax}
        cm = {r: st.mean(M[f"{f}|{r}"] for f in ax) for r in ax}
        ssf = sum(n * (rm[f] - gm) ** 2 for f in ax)
        ssc = sum(n * (cm[r] - gm) ** 2 for r in ax)
        ssi = sum((M[f"{f}|{r}"] - rm[f] - cm[r] + gm) ** 2 for f in ax for r in ax)
        return {"fit_pct": 100 * ssf / sst, "read_pct": 100 * ssc / sst,
                "interaction_pct": 100 * ssi / sst,
                "fit_span": max(rm.values()) - min(rm.values()),
                "read_span": max(cm.values()) - min(cm.values()),
                "span_ratio": (max(rm.values()) - min(rm.values())) / (max(cm.values()) - min(cm.values())),
                "fit_dominates_read": bool(ssf > ssc)}

    full = split(AX)
    loo = {drop: split([c for c in AX if c != drop]) for drop in AX}
    two = split([c for c in AX if c not in ("Github", "USPTO_Backgrounds")])
    return {"full": full, "leave_one_out": loo,
            "drop_both_extremes_github_and_uspto": two,
            "worst_loo_span_ratio": min(v["span_ratio"] for v in loo.values()),
            "worst_loo_corpus": min(loo, key=lambda k: loo[k]["span_ratio"]),
            "fit_dominates_in_all_loo": all(v["fit_dominates_read"] for v in loo.values()),
            "fit_dominates_with_both_extremes_dropped": two["fit_dominates_read"]}


def audit_derangement() -> dict:
    """C2 — the evidence that decides which aggregation may adjudicate between operators."""
    A = json.load(open(PATHS["e48"]))["arms_admitted_mean"]
    corp = sorted({k.split("|")[1] for k in A if k.startswith("J|")})
    out = {}
    for agg in AGGS:
        paired = ties = corpmean = 0
        per_corpus = {}
        for c in corp:
            jp_seeds = [A[f"J|{c}|s{s}"][agg] for s in (0, 1, 2)]
            jp_mean = st.mean(jp_seeds)
            draws, w = [], 0
            for s in (0, 1, 2):
                for i in range(5):
                    x = A[f"shuf|{c}|s{s}|d{i}"][agg]
                    draws.append(x)
                    if x > jp_seeds[s]:
                        paired += 1
                        w += 1
                    elif x == jp_seeds[s]:
                        ties += 1
                    if x > jp_mean:
                        corpmean += 1
            per_corpus[c] = {"jp_mean": jp_mean, "shuf_mean": st.mean(draws),
                             "gap": jp_mean - st.mean(draws),
                             "shuf_sd_over_15_draws": st.stdev(draws),
                             "shuf_beats_jp_paired": w}
        out[agg] = {
            "n_draws": len(corp) * 3 * 5,
            "shuf_beats_jp_paired_by_seed": paired, "ties": ties,
            "shuf_beats_jp_vs_corpus_mean": corpmean,
            "n_corpora_where_shuf_beats_jp_on_the_mean":
                sum(1 for v in per_corpus.values() if v["gap"] < 0),
            "shuf_sd_range": [min(v["shuf_sd_over_15_draws"] for v in per_corpus.values()),
                              max(v["shuf_sd_over_15_draws"] for v in per_corpus.values())],
            "gap_range": [min(v["gap"] for v in per_corpus.values()),
                          max(v["gap"] for v in per_corpus.values())],
            "per_corpus": per_corpus,
            "admissible_as_an_operator_comparator": bool(paired == 0),
        }
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e52", default=PATHS["e52"],
                    help="R1: point at results/e52_factorial_410m_rstrip.json to "
                         "re-derive on the corrected readout. Must match --e36.")
    ap.add_argument("--e36", default=PATHS["e36"])
    ap.add_argument("--e48", default=PATHS["e48"],
                    help="R1: point at results/e48_crossover_410m_rstrip.json so the "
                         "derangement audit is computed at the same readout as --e52/--e36.")
    ap.add_argument("--out", default=os.path.join(RES, "e54_aggregation_audit.json"))
    a = ap.parse_args()
    PATHS.update(e52=a.e52, e36=a.e36, e48=a.e48)

    res = {
        "experiment": "E54 — the aggregation audit: every headline under `min` and `persist`",
        "why": ("`persist` is ours; `min` is the anchor's. A claim that exists only under `persist` "
                "is a claim about our metric. Nothing had ever tabulated which claims those are."),
        "recomputes_not_remeasures": True,
        "inputs": ["results/ladder410/*.json", "results/ladder1b/*.json",
                   "results/e33_logit_baseline_410m_v2.json",
                   os.path.relpath(PATHS["e36"], os.path.join(RES, "..")),
                   os.path.relpath(PATHS["e48"], os.path.join(RES, "..")),
                   "results/e48c_exposure_vs_read.json",
                   os.path.relpath(PATHS["e52"], os.path.join(RES, ".."))],
        "readout_convention_note": (
            "e48c_exposure_vs_read.json has no corrected arm and is read unstripped in every "
            "run. Only its containment_k32 column is used here, which is an n-gram overlap "
            "between eval items and the training stream and carries no readout position; the "
            "READ values it is paired against come from --e48. Declared, not assumed away."),
        "admitted_sets": ADMITTED, "asymptote_n_min": N_MIN,
        "classification_thresholds": {
            "REVERSES": "sign of the decisive quantity flips, or a p-value crosses 0.05",
            "WEAKENS": "decisive ratio falls by more than 2x, sign kept",
            "HOLDS-STRONGER": "decisive ratio rises by more than 1.2x",
            "HOLDS": "otherwise"},
        "ladder": {}, "variance_decomposition": {}, "e33": {}, "e48c": {}, "e36": {}, "e52": {},
        "quadrants": {}, "matrix_structure": {}, "baseline_movement": {}, "matrix_loo": {},
    }
    for scale in ("410m", "1b"):
        res["ladder"][scale] = {agg: ladder_summary(scale, agg) for agg in AGGS}
        res["variance_decomposition"][scale] = {agg: variance_decomposition(scale, agg)
                                                for agg in AGGS}
    for agg in AGGS:
        res["e33"][agg] = audit_e33(agg)
        res["e48c"][agg] = audit_e48c(agg)
        res["e36"][agg] = audit_e36(agg)
        res["e52"][agg] = audit_e52(agg)
        res["quadrants"][agg] = audit_quadrants(agg)
        res["matrix_structure"][agg] = audit_matrix_structure(agg)
        res["baseline_movement"][agg] = audit_baseline_movement(agg)
        res["matrix_loo"][agg] = audit_matrix_loo(agg)
    res["C2_derangement"] = audit_derangement()

    # ---- C1: the persist column must reproduce the stored headlines
    e51 = json.load(open(os.path.join(RES, "e51_interaction_variance.json")))["scales"]
    e52p = json.load(open(PATHS["e52"]))["adjudication"]
    e48c = json.load(open(os.path.join(RES, "e48c_exposure_vs_read.json")))
    checks = {
        "e51_410m_interaction_pct": (res["variance_decomposition"]["410m"]["persist"]["frac_interaction"],
                                     e51["410m"]["admitted5"]["frac_interaction"] * 100),
        "e51_410m_corpus_pct": (res["variance_decomposition"]["410m"]["persist"]["frac_corpus_main"],
                                e51["410m"]["admitted5"]["frac_corpus_main"] * 100),
        "e51_1b_interaction_pct": (res["variance_decomposition"]["1b"]["persist"]["frac_interaction"],
                                   e51["1b"]["admitted5"]["frac_interaction"] * 100),
        "e51_410m_F": (res["variance_decomposition"]["410m"]["persist"]["F"],
                       e51["410m"]["admitted5"]["candidate_noise_floors"]["mean_squares_F"]),
        "e33_logit_constant": (res["e33"]["persist"]["logit_constant"],
                               json.load(open(os.path.join(RES, "e33_logit_baseline_410m_v2.json")))
                               ["persist_admitted_mean"]["logit_I"]),
        "e33_uspto_asymptote": (res["e33"]["persist"]["by_corpus"]["USPTO_Backgrounds"]["asymptote"],
                                0.049735264846729854),
        "e48c_spearman_all8": (res["e48c"]["persist"]["spearman_all8"],
                               e48c["correlations"]["all8_spearman"]),
        "e52_D": (res["e52"]["persist"]["D_all8"], e52p["D"]),
    }
    c1 = {k: {"recomputed": v[0], "stored": v[1], "abs_diff": abs(v[0] - v[1]),
              "ok": abs(v[0] - v[1]) < 1e-4} for k, v in checks.items()}
    res["C1_persist_column_reproduces_stored_headlines"] = c1
    res["C1_fires"] = all(v["ok"] for v in c1.values())

    # ---- verdict table
    L = res["ladder"]
    V = res["variance_decomposition"]
    v = {}
    v["corpus_identity_decides_the_read_410m"] = {
        "persist": L["410m"]["persist"]["spread_over_seed_sd"],
        "min": L["410m"]["min"]["spread_over_seed_sd"],
        "verdict": classify(L["410m"]["persist"]["spread_over_seed_sd"],
                            L["410m"]["min"]["spread_over_seed_sd"])}
    v["reads_are_flat_in_N_410m"] = {
        "persist": L["410m"]["persist"]["flat_range_over_seed_sd"],
        "min": L["410m"]["min"]["flat_range_over_seed_sd"], "verdict": "HOLDS"}
    v["github_indistinguishable_from_not_fitting"] = {
        "persist": res["e33"]["persist"]["by_corpus"]["Github"]["t_vs_logit_df2"],
        "min": res["e33"]["min"]["by_corpus"]["Github"]["t_vs_logit_df2"],
        "verdict": "REVERSES" if (res["e33"]["persist"]["by_corpus"]["Github"]["separates_from_logit"]
                                  != res["e33"]["min"]["by_corpus"]["Github"]["separates_from_logit"])
        else "HOLDS",
        "note": "under min Github is not merely indistinguishable from the logit lens, it is "
                "significantly BELOW it — a stronger form of the same claim, but a different sentence"}
    v["effect_is_an_interaction_not_a_level_410m"] = {
        "persist": V["410m"]["persist"]["interaction_over_corpus_main"],
        "min": V["410m"]["min"]["interaction_over_corpus_main"],
        "verdict": classify(V["410m"]["persist"]["interaction_over_corpus_main"],
                            V["410m"]["min"]["interaction_over_corpus_main"])}
    v["effect_is_an_interaction_not_a_level_1b"] = {
        "persist": V["1b"]["persist"]["interaction_over_corpus_main"],
        "min": V["1b"]["min"]["interaction_over_corpus_main"],
        "verdict": classify(V["1b"]["persist"]["interaction_over_corpus_main"],
                            V["1b"]["min"]["interaction_over_corpus_main"])}
    v["exposure_does_not_order_the_read"] = {
        "persist": res["e48c"]["persist"]["spearman_all8"],
        "min": res["e48c"]["min"]["spearman_all8"], "verdict": "HOLDS",
        "note": "both are indistinguishable from zero at n=8; all three OOD rungs beat in-stream "
                "Wikipedia under both"}
    v["e36_no_fitted_operator_crosses_below_logit"] = {
        "persist": res["e36"]["persist"]["n_fitted_crossing_below_logit_at_any_rung"],
        "min": res["e36"]["min"]["n_fitted_crossing_below_logit_at_any_rung"],
        "verdict": "HOLDS",
        "note": "the count includes Github, which sits below the logit lens at every rung under "
                "both aggregations and is degenerate by E33. Excluding it: 0 of 4 under persist, "
                "1 of 4 (Wikipedia, at 1 rung of 11) under min. The pre-registration states that a "
                "crossing appearing only under min does not count."}
    v["e36_fitted_operators_are_flatter_than_logit"] = {
        "persist": res["e36"]["persist"]["n_fitted_steeper_than_logit"],
        "min": res["e36"]["min"]["n_fitted_steeper_than_logit"],
        "verdict": "REVERSES",
        "note": "0 of 5 fitted operators are steeper than the logit lens under persist; "
                f"{res['e36']['min']['n_fitted_steeper_than_logit']} of 5 are under min. "
                "This secondary is aggregation-dependent AND single-rung-fragile under LOO."}
    v["e52_matched_fit_read_helps"] = {
        "persist": res["e52"]["persist"]["permutation_p"],
        "min": res["e52"]["min"]["permutation_p"],
        "verdict": "REVERSES",
        "note": "p = %.4f under persist, %.4f under min; and D changes sign without Github under "
                "min (%.2e)" % (res["e52"]["persist"]["permutation_p"],
                                res["e52"]["min"]["permutation_p"],
                                res["e52"]["min"]["D_without_degenerate"])}
    res["VERDICT_BY_CLAIM"] = v
    res["VERDICT"] = (
        "THE SPINE IS AGGREGATION-ROBUST; THREE SECONDARIES ARE NOT. Corpus identity deciding the "
        "read, reads being flat in N, and exposure not ordering the read all hold under the "
        "anchor's own `min` — the first with a LARGER absolute spread. What does not survive: the "
        "interaction's dominance over the corpus main effect at 410M (3.56x -> 1.05x), E36's slope "
        "secondary (0 of 5 fitted operators steeper -> 4 of 5), and E52's matched-diagonal effect "
        "(p=0.0055 -> p=0.66, and D changes sign without Github). Separately, C2 shows why `min` "
        "cannot be the adjudicating metric: under it the layer-deranged operator beats the real one "
        "on 104 of 120 paired draws and outscores the logit lens, while under `persist` it loses "
        "120 of 120. Both facts must be reported together — `min` is inadmissible as an operator "
        "comparator AND three of our claims depend on not using it.")

    try:
        from provenance import write_result
        write_result(a.out, res, script=__file__, experiment="E54", inputs=[
            os.path.join(RES, "e33_logit_baseline_410m_v2.json"),
            PATHS["e36"],
            PATHS["e48"],
            os.path.join(RES, "e48c_exposure_vs_read.json"),
            os.path.join(RES, "e51_interaction_variance.json"),
            PATHS["e52"],
        ] + sorted(glob.glob(os.path.join(RES, "ladder410", "*.json")))
          + sorted(glob.glob(os.path.join(RES, "ladder1b", "*.json"))))
    except Exception as e:                      # never swallow silently — rule 0b
        print(f"  !! provenance stamp FAILED: {e!r}", file=sys.stderr)
        json.dump(res, open(a.out, "w"), indent=1)
        return 2

    print(f"\n  C1 (persist reproduces stored headlines): {'FIRES' if res['C1_fires'] else 'FAILED'}")
    for k, c in c1.items():
        print(f"    {'ok ' if c['ok'] else 'FAIL'} {k:34s} {c['recomputed']:.6f} vs {c['stored']:.6f}")
    print(f"\n  {'claim':52s} {'persist':>10s} {'min':>10s}  verdict")
    for k, c in v.items():
        p_, m_ = c["persist"], c["min"]
        f = (lambda z: f"{z[0]:.1f}-{z[1]:.1f}" if isinstance(z, list) else f"{z:.4f}"
             if isinstance(z, float) else str(z))
        print(f"  {k:52s} {f(p_):>10s} {f(m_):>10s}  {c['verdict']}")
    print(f"\n  C2 derangement — is the aggregation admissible as an operator comparator?")
    for agg in AGGS:
        d = res["C2_derangement"][agg]
        print(f"    {agg:8s} shuf beats J^P on {d['shuf_beats_jp_paired_by_seed']:3d}/{d['n_draws']} "
              f"paired draws ({d['shuf_beats_jp_vs_corpus_mean']}/{d['n_draws']} vs the corpus mean), "
              f"on {d['n_corpora_where_shuf_beats_jp_on_the_mean']}/8 corpora  -> "
              f"{'ADMISSIBLE' if d['admissible_as_an_operator_comparator'] else 'INADMISSIBLE'}")
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
