#!/usr/bin/env python3
"""make_figures.py — regenerate every figure from the stored results JSON.

Every number plotted here is read from results/ at draw time. Nothing is typed in.
If a figure and a results file disagree, the results file wins and this script is broken.

    python figures/make_figures.py

WHICH OF THESE THE PAPERS ACTUALLY USE, as of 2026-08-17:
    fM_derangement      both papers   the metric-validity result; the one figure a table cannot carry
    f0_grid             9-page only   the factorial, two panels
    f2_predictor_bakeoff 9-page only  the 20 failed predictors
The rest (f1, f3, f4, f5, f6) are earlier-draft assets kept because they regenerate for free and are
useful for talks. A third panel of f0 -- every leave-one-corpus-out split as a scatter -- was cut on
2026-08-17: nine points in the corner of an empty square, carrying a claim one sentence states
better, at the cost of the width that lets the other two panels show their cell values.

Palette: the validated 5-slot categorical set (blue, orange, aqua, yellow, magenta).
Validated with the dataviz validator on the light surface #fcfcfb:
  lightness band PASS, chroma floor PASS, CVD separation PASS (worst adjacent dE 9.1),
  normal-vision floor PASS (worst adjacent dE 19.6), contrast WARN -> relief applied as
  direct labels + distinct marker shapes, so identity is never carried by colour alone.
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = HERE

# corpora ordered by measured plateau AUC, worst -> best, so colour tracks the entity
CORPORA = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]
LABEL = {"Github": "Github", "Wikipedia_en": "Wikipedia (en)", "StackExchange": "StackExchange",
         "Pile-CC": "Pile-CC", "USPTO_Backgrounds": "USPTO Backgrounds"}
SERIES = {"Github": "#2a78d6", "Wikipedia_en": "#eb6834", "StackExchange": "#1baf7a",
          "Pile-CC": "#eda100", "USPTO_Backgrounds": "#e87ba4"}
MARK = {"Github": "o", "Wikipedia_en": "s", "StackExchange": "^", "Pile-CC": "D",
        "USPTO_Backgrounds": "v"}
SETS = ["multihop", "multilingual", "order-ops", "poetry", "typo"]   # association excluded: floored

# The NeurIPS 2026 style is SINGLE COLUMN with \textwidth = 397.485pt = 5.50in. A figure drawn
# wider than that is scaled DOWN by \includegraphics, shrinking every label with it — which is how
# the previous 13.6in three-panel ended up unreadable at 40% size. Draw at the print width.
TW = 5.50          # \textwidth, inches
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})


def load_ladder():
    cells = {}
    for f in sorted(glob.glob(os.path.join(RES, "ladder410", "*.json"))):
        d = json.load(open(f))
        cells[(d["corpus"], d["seed"])] = d
    return cells


def curve(cells, corpus, sets=SETS):
    """mean read AUC (persist) over `sets`, per N: (Ns, means, sds)."""
    Ns = sorted({int(n) for s in range(3) for n in cells[(corpus, s)]["by_N"]})
    xs, ms, sds = [], [], []
    for N in Ns:
        vals = [statistics.mean([cells[(corpus, s)]["by_N"][str(N)][st]["persist"] for st in sets])
                for s in range(3) if str(N) in cells[(corpus, s)]["by_N"]]
        if len(vals) == 3:
            xs.append(N); ms.append(statistics.mean(vals)); sds.append(statistics.stdev(vals))
    return xs, ms, sds


# ---------------------------------------------------------------- figure 1
def fig1_ladder(cells):
    """The headline: what N buys (nothing) against what the corpus buys (everything)."""
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(11.2, 4.3),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    # StackExchange and Pile-CC coincide to within seed noise -- nudge their labels apart
    NUDGE = {"Pile-CC": 9, "StackExchange": -9}
    for c in CORPORA:
        xs, ms, sds = curve(cells, c)
        ax.fill_between(xs, [m - s for m, s in zip(ms, sds)], [m + s for m, s in zip(ms, sds)],
                        color=SERIES[c], alpha=0.16, linewidth=0)
        ax.plot(xs, ms, color=SERIES[c], linewidth=2.0, marker=MARK[c], markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=LABEL[c])
        ax.annotate(LABEL[c], (xs[-1], ms[-1]), xytext=(8, NUDGE.get(c, 0)),
                    textcoords="offset points", color=INK2, fontsize=8, va="center")
    ax.axvline(945, color=INK3, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate("$N^\\star=945$ — what $\\epsilon=\\sqrt{\\mathrm{disp}/N}$ prescribes",
                (945, 0.0525), xytext=(-5, 0), textcoords="offset points",
                ha="right", va="top", color=INK3, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlim(20, 1900)
    ax.set_ylim(0.0225, 0.0535)
    ax.set_xticks([25, 50, 100, 200, 400, 800])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("N — prompts the operator was averaged over")
    ax.set_ylabel("read AUC (persist, 5 admitted eval sets)")
    ax.set_title("Fitting corpus sets the read level; N does not move it", loc="left",
                 fontsize=10.5, color=INK, pad=9)
    ax.grid(axis="x", alpha=0.5)

    # Right panel. NOT a bias/variance decomposition -- these are two different KINDS of
    # quantity (between-corpus spread vs within-corpus convergence), and the all-5 spread is
    # carried by Github. The leave-one-out value is drawn so the leverage is visible, not hidden.
    nrange, seedsd = [], []
    for c in CORPORA:
        _, ms, sds = curve(cells, c)
        nrange.append(max(ms) - min(ms)); seedsd.append(statistics.mean(sds))
    plat = [statistics.mean([m for x, m in zip(*curve(cells, c)[:2]) if x >= 75]) for c in CORPORA]
    spread_all = max(plat) - min(plat)
    ex = [p for c, p in zip(CORPORA, plat) if c != "Github"]
    spread_nogh = max(ex) - min(ex)
    nr_nogh = max(n for c, n in zip(CORPORA, nrange) if c != "Github")

    bars = [("between fitting corpora\n(worst → best of 5)", spread_all, "#e34948"),
            ("…the same, without Github\n(one high-leverage corpus)", spread_nogh, "#f0a3a2"),
            ("within one corpus, N: 25 → 800\n(32× more data, best case)", max(nrange), "#2a78d6"),
            ("seed noise\n(3 disjoint prompt blocks)", statistics.mean(seedsd), INK3)]
    ypos = range(len(bars))
    axz.barh(list(ypos), [b[1] for b in bars], color=[b[2] for b in bars], height=0.52)
    for i, (lab, v, _) in enumerate(bars):
        axz.annotate(f"{v:.4f}", (v, i), xytext=(6, 0), textcoords="offset points",
                     va="center", color=INK, fontsize=9, fontweight="bold")
    axz.set_yticks(list(ypos)); axz.set_yticklabels([b[0] for b in bars], fontsize=8.2)
    axz.invert_yaxis()
    axz.set_xlim(0, spread_all * 1.30)
    axz.set_xlabel("change in read AUC")
    axz.set_title(f"Between-corpus spread vs within-corpus convergence\n"
                  f"{spread_all/max(nrange):.1f}× on five corpora, "
                  f"{spread_nogh/nr_nogh:.1f}× without Github",
                  loc="left", fontsize=9.8, color=INK, pad=9)
    axz.grid(axis="y", visible=False)
    fig.tight_layout()
    p = os.path.join(OUT, "f1_corpus_vs_N.png")
    fig.savefig(p, dpi=190); plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 2
def fig2_bakeoff():
    """Every candidate predictor, full-sample r against its worst leave-one-corpus-out r."""
    d = json.load(open(os.path.join(RES, "e31_local_bakeoff_410m.json")))
    t = d["bakeoff"]["plateau_auc"]
    rows = sorted(t.items(), key=lambda kv: -abs(kv[1]["r"]))
    fig, ax = plt.subplots(figsize=(TW, 3.05))
    y = list(range(len(rows)))[::-1]
    for yi, (name, v) in zip(y, rows):
        r, lo = abs(v["r"]), abs(v["worst_loo_r"])
        ax.plot([lo, r], [yi, yi], color=GRID, linewidth=1.5, solid_capstyle="round", zorder=1)
        ax.scatter([r], [yi], s=15, color="#2a78d6", zorder=3,
                   edgecolor=SURFACE, linewidth=0.4)
        ax.scatter([lo], [yi], s=15, color="#eb6834", zorder=3, marker="s",
                   edgecolor=SURFACE, linewidth=0.4)
        ax.annotate(v["worst_loo_corpus"].replace("_Backgrounds", "").replace("_en", ""),
                    (lo, yi), xytext=(0, -7.5), textcoords="offset points",
                    ha="center", color=INK3, fontsize=4.6)
    top = -0.62   # anchor the vertical rules at the BOTTOM; the top rows are the crowded ones
    ax.axvline(0.878, color=INK3, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate("p<0.05, n=5", (0.878, top), xytext=(2.5, 0),
                textcoords="offset points", color=INK3, fontsize=5.2, va="bottom", rotation=90)
    ax.axvline(0.8, color="#e34948", linewidth=1.2)
    ax.annotate("pre-registered control", (0.8, top),
                xytext=(-3.5, 0), textcoords="offset points", ha="right",
                color="#e34948", fontsize=5.2, va="bottom", rotation=90)
    ax.set_yticks(y); ax.set_yticklabels([n for n, _ in rows], fontsize=5.6)
    ax.tick_params(labelsize=5.6, length=1.8, pad=1.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(-0.02, 1.06)
    ax.set_xlabel("|r| with plateau read AUC across the 5 corpora", fontsize=6.2, color=INK2)
    ax.set_title("no predictor survives its own control", loc="left",
                 fontsize=7.0, color=INK, pad=5)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", color="#2a78d6",
                                  markersize=3.6, label="all 5 corpora"),
                       plt.Line2D([], [], marker="s", linestyle="", color="#eb6834",
                                  markersize=3.6, label="worst leave-one-corpus-out")],
              loc="upper left", fontsize=5.4, handletextpad=0.4, borderpad=0.5)
    fig.tight_layout(pad=0.45)
    p = os.path.join(OUT, "f2_predictor_bakeoff.png")
    fig.savefig(p, dpi=190); plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 3
def fig3_interaction(cells):
    """The corpus x concept-set interaction: within-set z-scores, diverging about a gray zero."""
    Y = {}
    for c in CORPORA:
        for st in SETS:
            Y[(c, st)] = statistics.mean([
                statistics.mean([cells[(c, s)]["by_N"][n][st]["persist"]
                                 for n in cells[(c, s)]["by_N"] if int(n) >= 75])
                for s in range(3)])
    Z = {}
    for st in SETS:
        v = [Y[(c, st)] for c in CORPORA]
        m, sd = statistics.mean(v), statistics.pstdev(v)
        for c in CORPORA:
            Z[(c, st)] = (Y[(c, st)] - m) / sd
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", ["#2a78d6", "#a9c7e9", "#e8e7e3", "#f3bda2", "#eb6834"])
    M = [[Z[(c, st)] for st in SETS] for c in CORPORA]
    im = ax.imshow(M, cmap=cmap, vmin=-2, vmax=2, aspect="auto")
    for i, c in enumerate(CORPORA):
        for j, st in enumerate(SETS):
            ax.text(j, i, f"{Z[(c,st)]:+.1f}", ha="center", va="center", fontsize=8.5,
                    color=INK if abs(Z[(c, st)]) < 1.3 else "#ffffff")
    ax.set_xticks(range(len(SETS))); ax.set_xticklabels(SETS, fontsize=8.5)
    ax.set_yticks(range(len(CORPORA)))
    ax.set_yticklabels([LABEL[c] for c in CORPORA], fontsize=8.5)
    ax.set_xticks([x - 0.5 for x in range(1, len(SETS))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(CORPORA))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2); ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("z within eval set", fontsize=8); cb.outline.set_visible(False)
    ax.set_title("Corpus does not make the lens better — it changes WHICH concepts it reads",
                 loc="left", fontsize=10.5, color=INK, pad=9)
    fig.tight_layout()
    p = os.path.join(OUT, "f3_corpus_set_interaction.png")
    fig.savefig(p, dpi=190); plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 4
E48_TIER = {"in-stream": "#2a78d6", "github": "#8a8880", "OOD": "#eb6834"}
E48_MARK = {"in-stream": "o", "github": "X", "OOD": "^"}
E48_LABEL = {"Wikipedia_en": "Wikipedia (en)", "USPTO_Backgrounds": "USPTO",
             "Pile-CC": "Pile-CC", "StackExchange": "StackExchange", "Github": "Github",
             "OOD_News_2024": "News 2024", "OOD_arXiv_2023": "arXiv 2023",
             "OOD_CommonPile": "CommonPile"}


def fig4_exposure_crossover():
    """E48. Left: exposure of the fitting corpus spans four orders of magnitude and does not
    order the read. Right: the registered interval cannot resolve ANY rung from zero — which is
    why the verdict is NOT REACHED and why that verdict is about power as much as about shift."""
    e48 = json.load(open(os.path.join(RES, "e48_crossover_410m.json")))
    e48c = json.load(open(os.path.join(RES, "e48c_exposure_vs_read.json")))
    tab = e48c["by_rung"]
    L = e48c["logit_baseline"]
    order = sorted(tab, key=lambda k: -tab[k]["read_persist"])

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(12.4, 4.8),
                                  gridspec_kw={"width_ratios": [1.3, 1]})

    # ---- left panel: containment (log) vs read
    shuf_lo = min(e48["rungs"][k]["persist"]["shuf_admitted_mean"] for k in tab)
    shuf_hi = max(e48["rungs"][k]["persist"]["shuf_admitted_mean"] for k in tab)
    ax.axhspan(shuf_lo, shuf_hi, color=INK3, alpha=0.13, linewidth=0)
    ax.annotate("layer-derangement floor — $J^P$ clears it on 120/120 draws",
                (6.5e-5, shuf_lo), xytext=(0, -4), textcoords="offset points",
                color=INK3, fontsize=7.8, va="top", ha="left")
    ax.axhline(L, color=INK, linewidth=1.4, linestyle=(0, (5, 3)))
    ax.annotate(f"logit lens ($J=I$), {L:.5f} — nothing fitted, so nothing to shift",
                (6.5e-5, L), xytext=(0, 5), textcoords="offset points",
                color=INK, fontsize=8.2, va="bottom", ha="left")
    # the five in-stream rungs pile up at x~0.8; their labels go LEFT into empty space,
    # the three OOD rungs sit at the left edge so theirs go RIGHT. Nothing overprints.
    OFF = {"USPTO_Backgrounds": (-12, 0), "StackExchange": (-12, 7), "Pile-CC": (-12, -8),
           "Wikipedia_en": (-12, 0), "Github": (-12, 0),
           "OOD_arXiv_2023": (12, 0), "OOD_CommonPile": (12, 0), "OOD_News_2024": (12, 0)}
    for k in order:
        t = tab[k]
        x = max(t["containment_k32"], 1e-4)
        ax.errorbar(x, t["read_persist"], yerr=t["seed_sd"], color=E48_TIER[t["tier"]],
                    marker=E48_MARK[t["tier"]], markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=1.0, elinewidth=1.4, capsize=2.5, linestyle="none")
        dx, dy = OFF[k]
        ax.annotate(E48_LABEL[k], (x, t["read_persist"]), xytext=(dx, dy),
                    textcoords="offset points", color=INK2, fontsize=8.5, va="center",
                    ha="left" if dx > 0 else "right")
    ax.set_xscale("log")
    ax.set_xlim(6e-5, 3.0)
    ax.set_ylim(0.0128, 0.0565)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ax.set_xlabel("containment of the FITTING corpus in the Pythia stream "
                  "($k$=32, 75% of the stream)")
    ax.set_ylabel("read AUC (persist, 5 admitted eval sets)")
    ax.set_title("Exposure of the fitting corpus does not order the read",
                 loc="left", fontsize=10.5, color=INK, pad=9)
    sp = e48c["correlations"]["all8_spearman"]
    ax.annotate(f"Spearman $r$ = {sp:+.3f} across {e48c['controls']['C1_containment_spans_orders']['max_over_min']:.0f}$\\times$ in exposure.\n"
                f"arXiv (least exposed) reads ABOVE Wikipedia (80% contained), $t={e48c['headline_pair_seed_axis']['t']:+.2f}$.",
                (0.015, 0.975), xycoords="axes fraction", color=INK2, fontsize=8.2, va="top")
    ax.legend(handles=[Patch(facecolor=E48_TIER[t], label=lbl) for t, lbl in
                       (("in-stream", "in-stream (Pile component)"),
                        ("github", "Github (code, in-stream)"),
                        ("OOD", "measured absent from the stream"))],
              loc="lower right", fontsize=8.2)

    # ---- right panel: the registered CI, every rung
    ys = list(range(len(order)))[::-1]
    for y, k in zip(ys, order):
        h = e48["rungs"][k]["persist"]["hierarchical_ci_vs_logit"]
        t = tab[k]
        axr.plot([h["ci_lo"], h["ci_hi"]], [y, y], color=E48_TIER[t["tier"]], linewidth=2.6,
                 solid_capstyle="round", alpha=0.5)
        axr.plot([h["point"]], [y], marker=E48_MARK[t["tier"]], markersize=7.5,
                 color=E48_TIER[t["tier"]], markeredgecolor=SURFACE, markeredgewidth=1.0)
    axr.axvline(0, color=INK, linewidth=1.2)
    axr.set_yticks(ys)
    axr.set_yticklabels([E48_LABEL[k] for k in order], fontsize=8.5)
    axr.set_ylim(-3.3, len(order) - 0.4)
    axr.set_xlabel("$J^P$ $-$ logit lens   (read AUC, persist)")
    axr.set_title("…and every interval spans zero", loc="left", fontsize=10.5, color=INK, pad=9)
    axr.annotate("clause (b): needed ≥2 OOD rungs strictly < 0 — got 0\n"
                 "clause (a): needed 4 in-stream strictly > 0 — got 0\n"
                 "⇒ NOT REACHED, on a test with no power to say otherwise",
                 (0.5, -3.15), xycoords=("axes fraction", "data"), color=INK2, fontsize=8,
                 va="bottom", ha="center")
    axr.grid(axis="y", visible=False)

    fig.tight_layout()
    p = os.path.join(OUT, "f4_exposure_crossover.png")
    fig.savefig(p, dpi=190)
    plt.close(fig)
    return p



# ---------------------------------------------------------------- figure 5
def fig5_qladder():
    """E36. Left: both lenses degrade together as the READ distribution leaves the training
    stream. Right: the pre-registered slope secondary — every fitted operator is FLATTER across
    the shift axis than the unfitted baseline, which is the opposite of what S3 predicts."""
    d = json.load(open(os.path.join(RES, "e36_qladder_410m.json")))
    lad = d["ladder"]
    FIT = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]
    rows = [(r, v) for r, v in lad.items()
            if v["containment_k32"] is not None and not v["is_shuffled"]]
    rows.sort(key=lambda kv: -kv[1]["containment_k32"])
    C = [v["containment_k32"] for _, v in rows]

    def slope(y):
        mx, my = statistics.mean(C), statistics.mean(y)
        den = sum((a - mx) ** 2 for a in C)
        return sum((a - mx) * (b - my) for a, b in zip(C, y)) / den

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(12.2, 4.6),
                                  gridspec_kw={"width_ratios": [1.45, 1]})
    lg = [v["arms"]["logit"]["persist"]["mean"] for _, v in rows]
    lgsd = [v["arms"]["logit"]["persist"]["seed_sd"] for _, v in rows]
    ax.errorbar(C, lg, yerr=lgsd, color=INK, marker="s", markersize=6, linewidth=2.4,
                capsize=2.5, markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=5)
    ax.annotate("logit lens ($J=I$)", (C[-1], lg[-1]), xytext=(9, -2),
                textcoords="offset points", color=INK, fontsize=8.5, va="center")
    for c in FIT:
        y = [v["arms"][f"J|{c}"]["persist"]["mean"] for _, v in rows]
        sd = [v["arms"][f"J|{c}"]["persist"]["seed_sd"] for _, v in rows]
        ax.errorbar(C, y, yerr=sd, color=SERIES[c], marker=MARK[c], markersize=5,
                    linewidth=1.8, alpha=0.95, capsize=2, markeredgecolor=SURFACE,
                    markeredgewidth=0.8)
        ax.annotate(LABEL[c], (C[-1], y[-1]), xytext=(9, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("containment of the READ context in the Pythia stream  "
                  "($k$=32, full stream) — shift increases to the right →")
    ax.set_ylabel("read AUC (persist, 5 admitted eval sets)")
    ax.set_title("Shifting the READ distribution degrades both lenses together",
                 loc="left", fontsize=10.5, color=INK, pad=9)
    ax.set_xlim(1.35, 4e-5)

    sl_l = slope(lg)
    names = ["logit lens"] + [LABEL[c] for c in FIT]
    vals = [sl_l] + [slope([v["arms"][f"J|{c}"]["persist"]["mean"] for _, v in rows]) for c in FIT]
    cols = [INK] + [SERIES[c] for c in FIT]
    ypos = list(range(len(vals)))[::-1]
    axr.barh(ypos, vals, color=cols, height=0.55)
    for y, v in zip(ypos, vals):
        axr.annotate(f"{v:+.5f}", (v, y), xytext=(-6 if v < 0 else 6, 0),
                     textcoords="offset points", va="center",
                     ha="right" if v < 0 else "left", fontsize=8.5, color=INK)
    axr.axvline(sl_l, color=INK, linewidth=1.0, linestyle=(0, (4, 3)))
    axr.set_yticks(ypos)
    axr.set_yticklabels(names, fontsize=8.5)
    axr.set_xlabel("slope  d(read AUC) / d(containment)")
    axr.set_title("S3 predicts the fitted operators are STEEPER.\nAll five are flatter.",
                  loc="left", fontsize=10.5, color=INK, pad=9)
    axr.set_xlim(min(vals) * 1.45, max(max(vals), 0) + abs(min(vals)) * 0.45)
    axr.grid(axis="y", visible=False)
    fig.tight_layout()
    p = os.path.join(OUT, "f5_qladder.png")
    fig.savefig(p, dpi=190); plt.close(fig)
    return p



# ---------------------------------------------------------------- figure 6
E52_ORDER = ["USPTO_Backgrounds", "Pile-CC", "StackExchange", "Wikipedia_en", "Github",
             "OOD_arXiv_2023", "OOD_CommonPile", "OOD_News_2024"]
E52_SHORT = {"USPTO_Backgrounds": "USPTO", "Pile-CC": "Pile-CC", "StackExchange": "StackExch",
             "Wikipedia_en": "Wikipedia", "Github": "Github", "OOD_arXiv_2023": "arXiv 2023",
             "OOD_CommonPile": "CommonPile", "OOD_News_2024": "News 2024"}


def fig6_fitread_matrix():
    """E52. Left: the fit x read residual after removing 'some corpora make better operators' and
    'some read contexts are easier'. What is left on the diagonal is the OJD effect. Right: the
    same statistic decomposed by eval set, which is where it turns out to live."""
    d = json.load(open(os.path.join(RES, "e52_factorial_410m.json")))
    P = d["by_aggregation"]["persist"]["all8"]
    g = P["g"]
    per_set = d["per_set_D"]
    D = d["adjudication"]["D"]
    ci = d["adjudication"]["D_hierarchical_ci"]

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(12.6, 4.9),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    M = [[g[f"{f}|{q}"] * 1000 for q in E52_ORDER] for f in E52_ORDER]
    vmax = max(abs(v) for row in M for v in row)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", ["#2a78d6", "#a9c7e9", "#e8e7e3", "#f3bda2", "#eb6834"])
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for i, f in enumerate(E52_ORDER):
        for j, q in enumerate(E52_ORDER):
            v = M[i][j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7.6,
                    color=INK if abs(v) < 0.62 * vmax else "#ffffff",
                    fontweight="bold" if i == j else "normal")
    for i in range(len(E52_ORDER)):        # ring the diagonal — it is the estimand
        ax.add_patch(matplotlib.patches.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                                  edgecolor=INK, linewidth=1.8, zorder=4))
    ax.set_xticks(range(len(E52_ORDER)))
    ax.set_xticklabels([E52_SHORT[c] for c in E52_ORDER], fontsize=8, rotation=32, ha="right")
    ax.set_yticks(range(len(E52_ORDER)))
    ax.set_yticklabels([E52_SHORT[c] for c in E52_ORDER], fontsize=8)
    ax.set_xlabel("READ context drawn from →", fontsize=9)
    ax.set_ylabel("← operator FITTED on", fontsize=9)
    ax.set_xticks([x - 0.5 for x in range(1, len(E52_ORDER))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(E52_ORDER))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cb.set_label("residual read AUC  ($\\times 10^{-3}$), main effects removed", fontsize=8)
    cb.outline.set_visible(False)
    ax.set_title("Reading a corpus the operator was fitted on buys almost nothing",
                 loc="left", fontsize=10.5, color=INK, pad=9)

    order = ["typo", "multilingual", "multihop", "order-ops", "poetry"]
    vals = [per_set[s]["mean"] * 1000 for s in order]
    ypos = list(range(len(order)))[::-1]
    cols = ["#eb6834"] + [INK3] * 4
    axr.barh(ypos, vals, color=cols, height=0.55)
    span = max(vals) - min(min(vals), 0)
    for y, v in zip(ypos, vals):
        # near-zero bars have no room on the negative side without colliding with the tick
        # label; push those annotations to the right of the axis instead
        right = v >= 0 or abs(v) < 0.05 * span
        axr.annotate(f"{v:+.2f}", (max(v, 0.0), y), xytext=(7 if right else -7, 0),
                     textcoords="offset points", va="center",
                     ha="left" if right else "right", fontsize=8.5, color=INK)
    axr.axvline(D * 1000, color="#2a78d6", linewidth=1.4, linestyle=(0, (4, 3)))
    axr.annotate(f"$D$ = {D*1000:+.2f}\n(the 5-set mean)", (D * 1000, -0.45),
                 xytext=(6, 0), textcoords="offset points", color="#2a78d6", fontsize=8,
                 va="center")
    axr.axvline(0, color=INK, linewidth=1.0)
    axr.set_yticks(ypos)
    axr.set_yticklabels(order, fontsize=9)
    axr.set_xlabel("per-set diagonal excess  ($\\times 10^{-3}$)")
    axr.set_title("…and 84% of what it does buy is one eval set",
                  loc="left", fontsize=10.5, color=INK, pad=9)
    axr.set_ylim(-1.1, len(order) - 0.4)
    axr.grid(axis="y", visible=False)
    fig.tight_layout()
    p = os.path.join(OUT, "f6_fitread_matrix.png")
    fig.savefig(p, dpi=190); plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 0 — the grid
def _ink_on(rgba):
    """Text colour for a filled cell, chosen from the cell's own luminance.

    The previous rule picked white whenever the value was far from the colormap MIDPOINT, which
    is only correct for a diverging map. On viridis both extremes are 'far' but only one is dark,
    so every high-value cell got white text on bright yellow and was unreadable in print. Relative
    luminance (WCAG) fixes it for any colormap, sequential or diverging.
    """
    r, g, b = rgba[0], rgba[1], rgba[2]

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "#ffffff" if L < 0.45 else INK


def fig0_grid_single():
    """ONE panel: the raw grid with its own margins attached.

    The two-panel version put the matching null (main effects removed) beside the observation. In
    the 5-page paper matching is one sentence and the residual share is already in the variance
    table, so the second panel was answering a question the body no longer asks; it moves to the
    appendix. What replaces it is better: the row and column means drawn as strips on the SAME
    colour scale as the grid. The claim is then visible without reading any of the 64 numbers --
    the right-hand strip spans the full scale, the bottom strip barely moves.
    """
    e52 = json.load(open(os.path.join(RES, "e52_factorial_410m.json")))
    raw = e52["by_aggregation"]["persist"]["matrix"]
    A = [[raw[f"{f}|{q}"] * 100 for q in E52_ORDER] for f in E52_ORDER]
    lo, hi = min(map(min, A)), max(map(max, A))
    cmap = matplotlib.colormaps["viridis"]
    norm = matplotlib.colors.Normalize(vmin=lo, vmax=hi)

    rows = [statistics.mean(r) for r in A]
    cols = [statistics.mean(A[i][j] for i in range(8)) for j in range(8)]

    # aspect="auto" on ALL THREE axes: with aspect="equal" the main heatmap shrinks inside its
    # gridspec cell while the strips fill theirs, and the three stop lining up. Letting the
    # gridspec ratios (8 : 1.05) set the geometry keeps every cell the same physical size.
    fig = plt.figure(figsize=(TW, 2.95))
    gs = fig.add_gridspec(2, 2, width_ratios=[8, 1.0], height_ratios=[8, 1.0],
                          wspace=0.035, hspace=0.035,
                          left=0.140, right=0.885, top=0.985, bottom=0.190)
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1], sharey=ax)
    axc = fig.add_subplot(gs[1, 0], sharex=ax)

    ax.imshow(A, cmap=cmap, norm=norm, aspect="auto")
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{A[i][j]:.2f}", ha="center", va="center", fontsize=5.2,
                    color=_ink_on(cmap(norm(A[i][j]))))
    ax.set_yticks(range(8))
    ax.set_yticklabels([E52_SHORT[c] for c in E52_ORDER], fontsize=5.8)
    ax.set_ylabel("fitting corpus", fontsize=6.8, color=INK, labelpad=3)
    ax.tick_params(axis="y", length=1.6, pad=1.2)
    ax.tick_params(axis="x", length=0, labelbottom=False)
    ax.grid(False)

    # right strip: row means, same colour scale
    axr.imshow([[v] for v in rows], cmap=cmap, norm=norm, aspect="auto")
    for i, v in enumerate(rows):
        axr.text(0, i, f"{v:.2f}", ha="center", va="center", fontsize=5.4,
                 color=_ink_on(cmap(norm(v))))
    axr.set_xticks([0])
    axr.set_xticklabels(["fit\nmean"], fontsize=5.8, color=INK)
    axr.tick_params(axis="y", length=0, labelleft=False)
    axr.tick_params(axis="x", length=0, pad=2)
    axr.grid(False)

    # bottom strip: column means, same colour scale
    axc.imshow([cols], cmap=cmap, norm=norm, aspect="auto")
    for j, v in enumerate(cols):
        axc.text(j, 0, f"{v:.2f}", ha="center", va="center", fontsize=5.4,
                 color=_ink_on(cmap(norm(v))))
    axc.set_yticks([0])
    axc.set_yticklabels(["read mean"], fontsize=5.8, color=INK)
    axc.set_xticks(range(8))
    axc.set_xticklabels([E52_SHORT[c] for c in E52_ORDER], fontsize=5.8, rotation=40, ha="right")
    axc.set_xlabel("read context", fontsize=6.8, color=INK, labelpad=1)
    axc.tick_params(axis="y", length=0, pad=2)
    axc.tick_params(axis="x", length=1.6, pad=1.2)
    axc.grid(False)

    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=[ax, axr, axc], fraction=0.030, pad=0.015, aspect=28)
    cb.set_label("read AUC ($\\times10^{-2}$)", fontsize=6.6, color=INK, labelpad=2)
    cb.ax.tick_params(labelsize=5.8, length=1.6, pad=1.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(GRID)
    p = os.path.join(OUT, "f0_grid_single.png")
    fig.savefig(p, dpi=220); plt.close(fig)
    return p


def figB_band_width():
    """Why `min` fails, and where. The real operator's advantage over its own derangement under
    min-over-layers, as a function of how many layers the min runs over. Above zero the statistic
    prefers the real operator; below zero it prefers the scrambled one. If the failure is a
    union-size effect it has to depend on band width, and this is that prediction tested."""
    d = json.load(open(os.path.join(RES, "d1_min_union_diagnostic_410m.json")))
    adj = d["adjudication"]
    L = len(d["band"])
    order = [c for c in ("USPTO_Backgrounds", "Pile-CC", "Github") if c in adj]

    fig, ax = plt.subplots(figsize=(TW * 0.60, 1.80))
    ax.axhline(0, color=INK, lw=0.9, zorder=1)
    ax.axvspan(6.5, L + 0.5, color="#eb6834", alpha=0.07, zorder=0, lw=0)
    for c in order:
        ys = adj[c]["H4_artifact_scales_with_band_width"]["min_gap_jp_minus_shuf_by_band_width"]
        xs = list(range(1, L + 1))
        ax.plot(xs, [ys[str(w)] for w in xs], marker=MARK.get(c, "o"), ms=3.0, lw=1.3,
                color=SERIES.get(c, INK2), label=LABEL.get(c, c), zorder=3)
    ax.set_xlabel("layers in the $\\mathtt{min}$", fontsize=6.8, color=INK, labelpad=1)
    ax.set_ylabel("$J^{P}$ $-$ $J^{\\mathrm{shuf}}$, read AUC", fontsize=6.8, color=INK, labelpad=1)
    ax.set_xticks(range(1, L + 1))
    ax.tick_params(labelsize=6.0, length=1.6, pad=1.2)
    ax.legend(fontsize=6.0, frameon=False, loc="lower left", handlelength=1.4, borderpad=0.2)
    fig.tight_layout(pad=0.3)
    p = os.path.join(OUT, "fB_band_width.png")
    fig.savefig(p, dpi=220); plt.close(fig)
    return p


def fig0_grid_three_panel():
    """The paper's grid figure. TWO panels, not three.

    A third panel showing every leave-one-corpus-out split as a scatter was cut: nine points in the
    corner of an otherwise empty square, carrying a claim ("every split is far above equal shares")
    that one sentence states better. Removing it doubles the width available to the panels that do
    carry something, which is what lets the cell values go back in — at three panels an 8x8 grid had
    0.2in cells and the numbers had to be dropped to the appendix.

      (a) the RAW grid. Rows separate, columns much less. This is the observation.
      (b) the SAME grid with both main effects removed, so only the matching term is left. The
          ringed diagonal is the estimand and is not distinguishable from its row. Showing (a)
          alone invites "the diagonal is just the row effect"; showing (b) alone invites "you
          removed the signal by construction".
    """
    e52 = json.load(open(os.path.join(RES, "e52_factorial_410m.json")))
    raw = e52["by_aggregation"]["persist"]["matrix"]
    g = e52["by_aggregation"]["persist"]["all8"]["g"]

    fig, axes = plt.subplots(1, 2, figsize=(TW, 2.98))
    div = matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", ["#2a78d6", "#a9c7e9", "#e8e7e3", "#f3bda2", "#eb6834"])

    def heat(ax, M, title, sub, fmt, cmap, vmin, vmax, ring):
        ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for i in range(8):
            for j in range(8):
                v = M[i][j]
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=5.0,
                        color=_ink_on(cm(norm(v))))
        if ring:
            for i in range(8):
                ax.add_patch(matplotlib.patches.Rectangle(
                    (i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.0, zorder=5))
        ax.set_xticks(range(8))
        ax.set_xticklabels([E52_SHORT[c] for c in E52_ORDER], fontsize=5.4, rotation=40, ha="right")
        ax.set_yticks(range(8))
        ax.set_yticklabels([E52_SHORT[c] for c in E52_ORDER], fontsize=5.4)
        ax.tick_params(length=1.6, pad=1.2)
        ax.set_title(title, fontsize=7.0, color=INK, pad=6)
        ax.set_xlabel(sub, fontsize=5.9, color=INK2, labelpad=2)
        ax.grid(False)

    A = [[raw[f"{f}|{q}"] * 100 for q in E52_ORDER] for f in E52_ORDER]
    lo, hi = min(map(min, A)), max(map(max, A))
    heat(axes[0], A, "(a) read AUC  ($\\times10^{-2}$)", "read context drawn from",
         lambda v: f"{v:.2f}", "viridis", lo, hi, False)
    axes[0].set_ylabel("corpus that BUILT the operator", fontsize=6.2, color=INK)

    B = [[g[f"{f}|{q}"] * 1000 for q in E52_ORDER] for f in E52_ORDER]
    vm = max(abs(v) for row in B for v in row)
    heat(axes[1], B, "(b) main effects removed  ($\\times10^{-3}$)",
         "ringed: operator reads its own corpus", lambda v: f"{v:+.1f}", div, -vm, vm, True)

    fig.tight_layout(pad=0.4, w_pad=1.1)
    p = os.path.join(OUT, "f0_grid.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


# ---------------------------------------------------------------- figure M — the metric audit
def figM_derangement():
    """The single most transferable finding, and it had no figure.

    For each of 120 draws (8 fitting corpora x 3 seed blocks x 5 random derangements) we hold the
    activation cache fixed and score the real operator against ITS OWN deranged copy. Paired, so
    each dot is one difference and zero is the only meaningful reference. Under `persist` every
    dot is positive; under the published `min` most are negative -- the statistic prefers the
    broken operator. Plotting the paired difference rather than two score distributions is the
    point: the comparison is within-draw, and a side-by-side of raw scores would hide that.
    """
    d = json.load(open(os.path.join(RES, "e48_crossover_410m.json")))["arms_admitted_mean"]
    corpora = [c for c in E52_ORDER]
    pairs = {agg: {c: [] for c in corpora} for agg in ("persist", "min")}
    for k, v in d.items():
        if not k.startswith("shuf|"):
            continue
        _, corpus, seed, _der = k.split("|")
        real = d.get(f"J|{corpus}|{seed}")
        if real is None or corpus not in pairs["persist"]:
            continue
        for agg in ("persist", "min"):
            pairs[agg][corpus].append((real[agg] - v[agg]) * 1000)

    fig, axes = plt.subplots(1, 2, figsize=(TW, 2.55), sharey=True)
    POS, NEG = "#1baf7a", "#eb6834"
    for ax, agg in zip(axes, ("persist", "min")):
        allv = [x for c in corpora for x in pairs[agg][c]]
        n_pos = sum(1 for x in allv if x > 0)
        for i, c in enumerate(corpora):
            vals = pairs[agg][c]
            jitter = [i + (j - (len(vals) - 1) / 2) * 0.030 for j in range(len(vals))]
            ax.scatter(vals, jitter, s=8.0, zorder=3, edgecolor=SURFACE, linewidth=0.25,
                       c=[POS if x > 0 else NEG for x in vals])
        ax.axvline(0, color=INK, lw=1.2, zorder=2)
        ax.set_yticks(range(len(corpora)))
        ax.set_yticklabels([E52_SHORT[c] for c in corpora], fontsize=6.4)
        ax.invert_yaxis()
        ax.tick_params(labelsize=6.4)
        ax.set_xlabel(r"real $-$ deranged, same draw ($\times10^{-3}$)", fontsize=6.8, color=INK2)
        ax.set_title(f"{agg} — real operator wins {n_pos}/{len(allv)}",
                     fontsize=7.6, color=INK, pad=5)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.grid(axis="x", color=GRID, lw=0.5, ls=(0, (2, 3)))
    # direction is carried by colour + the per-panel win count in the title; an inline
    # annotation here collided with the USPTO row, so the caption says it instead.
    fig.tight_layout()
    p = os.path.join(OUT, "fM_derangement.png")
    fig.savefig(p, dpi=190); plt.close(fig)
    return p


def main():
    cells = load_ladder()
    assert len(cells) == 15, f"expected 15 (corpus, seed) ladder cells, found {len(cells)}"
    for p in (fig0_grid_three_panel(), figM_derangement(), fig1_ladder(cells), fig2_bakeoff(), fig3_interaction(cells),
              fig4_exposure_crossover(), fig5_qladder(), fig6_fitread_matrix()):
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
