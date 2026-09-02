#!/usr/bin/env python3
"""make_audit_figures.py - figures for the measurement-validity paper.

Every number plotted is read from results/*.json at draw time. Nothing is typed in. If a figure
and a results file disagree, the results file wins and this script is broken.

    .venv/bin/python figures/make_audit_figures.py

FIGURES
    fA_effect_and_null.png   (a) between-corpus z spread at four ladder rungs        [d3, cv6, cv7]
                             (b) each operator against its own 15 layer-derangements [r9]
    fC_matching.png          the 8x8 fit x read factorial, residual after main effects  [e52]
    fB_validity.png          (a) where the operator is read vs where it was fitted  [cv2]
                             (b) answer competence and prompt surprisal up the ladder [cv4]
    fM_membership.png        32-gram containment against the read it produces     [e48c, e52]
    fU_union.png             real minus deranged AUC under min, by band width     [d1]

LAYOUT. An earlier version direct-labelled the series inside the axes to save vertical space. At
print size those labels sat on top of the data and it was not clear which mark carried which
weight. Every legend is now OUTSIDE the axes, below its own panel, and the panels are tall enough
that marks and text are not competing. Figures cost page budget; unreadable figures cost more.

Palette and surface follow figures/make_figures.py so the two sets sit together.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = HERE

# The NeurIPS 2026 style is single column with \textwidth = 5.50in. Draw at print width so
# \includegraphics never rescales the labels.
TW = 5.50
FIGH = 2.86          # tall enough that rotated tick labels and the legends do not collide
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e4e3df"
SURFACE = "#ffffff"   # match the page; an off-white panel reads as a box
BLUE, ORANGE, PURPLE, YELLOW, PINK = "#2a78d6", "#eb6834", "#7b52ab", "#eda100", "#e87ba4"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})

CORPUS_LABEL = {"Wikipedia_en": "Wikipedia (en)", "USPTO_Backgrounds": "USPTO",
                "Pile-CC": "Pile-CC", "StackExchange": "StackExchange", "Github": "Github",
                "OOD_News_2024": "News 2024", "OOD_arXiv_2023": "arXiv 2023",
                "OOD_CommonPile": "CommonPile"}


def _load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def _below(ax, handles, labels, ncol, y=-0.30):
    """Legend under the panel, centred on that panel, never over the data and never over the
    neighbouring panel's legend. Keep ncol low enough that the row fits the panel's own width."""
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
              ncol=ncol, fontsize=7, handlelength=1.5, columnspacing=1.1,
              handletextpad=0.5, borderaxespad=0.0)


def fig_effect_and_null():
    """(a) the four-rung ladder, (b) each operator against its own derangement null.

    Panel (a) plots the between-corpus z SPREAD, not R. R = spread / pooled seed SD, and the seed SD
    is 2.29x to 5.70x larger at N=25 than at N=200, so R is not comparable across rungs with
    different N. Ranking rungs by R would show a fall that is entirely a denominator effect. The
    spread is the quantity that is comparable, and N is annotated on every rung.
    """
    d3 = _load("d3_corpus_by_family_410m.json")
    cv7 = _load("cv7_1b_rung.json")
    r9 = _load("r9_permutation_calibrated_min.json")

    fams = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
    rungs = [("410m_N200", "410M", 200, BLUE), ("1b_N200", "1B", 200, PURPLE),
             ("1.4b_N25", "1.4B", 25, YELLOW), ("2.8b_N25", "2.8B", 25, ORANGE)]
    spread = cv7["CROSS_LADDER_z_spread"]["z_spread_by_rung"]
    # The 410M column must agree with D3, which is where it came from.
    for f in fams:
        assert abs(spread["410m_N200"][f] - d3["by_family"][f]["z_spread"]) < 1e-12, f

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(TW, FIGH),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    x = list(range(len(fams)))
    w = 0.20
    for i, (key, label, n, colour) in enumerate(rungs):
        off = (i - 1.5) * w
        axL.bar([v + off for v in x], [spread[key][f] for f in fams], width=w, color=colour)
    axL.set_xticks(x)
    axL.set_xticklabels(fams, fontsize=6.8, rotation=38, ha="right",
                        rotation_mode="anchor")
    axL.set_ylim(0, 0.86)
    axL.set_ylabel("between-corpus $z$ spread", fontsize=7.2, labelpad=2)
    axL.grid(axis="x", visible=False)
    axL.set_title("(a)", fontsize=8.5, loc="left", color=INK, pad=4)
    _below(axL, [Patch(color=c) for _, _, _, c in rungs],
           [f"{lab} (N={n})" for _, lab, n, _ in rungs], 4, y=-0.52)

    zall = r9["by_aggregation"]["persist"]["z_by_corpus"]
    order = sorted(zall, key=lambda c: -zall[c])
    zmin = [r9["by_aggregation"]["min"]["z_by_corpus"][c] for c in order]
    zper = [zall[c] for c in order]
    yy = list(range(len(order)))
    axR.axvline(0.0, color=INK3, lw=0.9, zorder=1)
    for i, (a, b) in enumerate(zip(zmin, zper)):
        axR.plot([a, b], [i, i], color=GRID, lw=1.2, zorder=2)
    axR.scatter(zmin, yy, s=26, color=ORANGE, zorder=3)
    axR.scatter(zper, yy, s=26, color=PURPLE, marker="D", zorder=3)
    axR.set_yticks(yy)
    axR.set_yticklabels([CORPUS_LABEL[c] for c in order], fontsize=8)
    axR.set_ylim(len(order) - 0.5, -0.5)
    axR.set_xlim(-4, 20)
    axR.set_xlabel("$z$ within its own derangement null", fontsize=7.2)
    axR.grid(axis="y", visible=False)
    axR.set_title("(b)", fontsize=8.5, loc="left", color=INK, pad=4)
    _below(axR,
           [Line2D([], [], marker="o", ls="", color=ORANGE, ms=5),
            Line2D([], [], marker="D", ls="", color=PURPLE, ms=5)],
           ["min (any layer)", "persist (majority)"], 2, y=-0.52)

    fig.tight_layout(pad=0.5)
    p = os.path.join(OUT, "fA_effect_and_null.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return p



def fig_matching():
    """fC: the fit x read factorial. Does reading a corpus the operator was fitted on help?

    Plots `g`, the interaction term the experiment itself stores: the cell with the grand mean and
    BOTH main effects already removed. So the diagonal asks only whether the fit-read PAIRING
    matters, with "some corpora read better" and "some operators score better" taken out. Using the
    stored `g` rather than re-deriving it is deliberate -- a re-derivation is a second chance to get
    the contrast wrong, and the first draft of this figure did exactly that by dropping `b_read`.

    Regenerated from results/e52_factorial_410m_rstrip.json at build time. A hand-made ancestor of
    this figure survived in figures/ carrying D = +1.03e-3 against the stored +0.85e-3; it predated
    a convention change and nothing regenerated it. This function exists so that cannot recur.
    """
    import numpy as np
    o = _load("e52_factorial_410m_rstrip.json")
    a8 = o["by_aggregation"]["min"]["all8"]
    g = a8["g"]
    fits = o["fit_corpora"]
    reads = fits                      # the square panel; Q0 and SHUFFLED_ are controls, not corpora

    M = np.full((len(fits), len(reads)), np.nan)
    for i, f in enumerate(fits):
        for j, r in enumerate(reads):
            if f"{f}|{r}" in g:
                M[i, j] = g[f"{f}|{r}"] * 1e3

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(TW, 2.35), gridspec_kw={"width_ratios": [1.58, 1.0], "wspace": 0.58})

    lim = float(np.nanmax(np.abs(M))) or 1.0
    im = axL.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    axL.set_xticks(range(len(reads)))
    axL.set_xticklabels([CORPUS_LABEL.get(r, r) for r in reads], rotation=40, ha="right", fontsize=6.0)
    axL.set_yticks(range(len(fits)))
    axL.set_yticklabels([CORPUS_LABEL.get(f, f) for f in fits], fontsize=6.0)
    axL.set_xlabel("READ context drawn from", fontsize=7.0)
    axL.set_ylabel("operator FITTED on", fontsize=7.0)
    axL.grid(False)
    for i in range(len(fits)):
        for j in range(len(reads)):
            if np.isnan(M[i, j]):
                continue
            diag = (i == j)
            axL.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center",
                     fontsize=5.4, fontweight="bold" if diag else "normal",
                     color=INK if abs(M[i, j]) < 0.60 * lim else "white")
            if diag:
                axL.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                            edgecolor=INK, linewidth=1.6))
    cb = fig.colorbar(im, ax=axL, fraction=0.032, pad=0.03)
    cb.ax.tick_params(labelsize=5.6)
    cb.set_label("interaction, $\\times10^{-3}$", fontsize=6.0)
    axL.set_title("Reading the corpus the operator was fitted on\nbuys nothing",
                  fontsize=7.6, pad=4)

    per = o["per_set_D"]
    sets = sorted(per, key=lambda s: per[s]["mean"])
    xs = [per[s]["mean"] * 1e3 for s in sets]
    D = o["adjudication"]["D"] * 1e3
    top = max(xs)
    axR.barh(range(len(sets)), xs,
             color=[ORANGE if x == top else INK3 for x in xs], height=0.60)
    axR.axvline(0, color=INK, linewidth=0.9)
    axR.axvline(D, color=BLUE, linewidth=1.1, linestyle="--")
    axR.set_yticks(range(len(sets)))
    axR.set_yticklabels(sets, fontsize=6.4)
    axR.set_xlabel("per-set diagonal excess, $\\times10^{-3}$", fontsize=7.0)
    axR.set_xlim(min(xs) * 1.30 if min(xs) < 0 else -0.35 * top, top * 1.34)
    for i, x in enumerate(xs):
        off = 0.05 * top
        if x >= 0:                       # outside the bar, to its right
            axR.text(x + off, i, f"{x:+.2f}", va="center", ha="left",
                     fontsize=6.0, color=INK2)
        else:                            # INSIDE the bar: a left-hand label collides with the
            axR.text(x + off, i, f"{x:+.2f}", va="center", ha="left",   # y-tick text
                     fontsize=6.0, color="white", fontweight="bold")
    axR.set_title(f"and one evaluation set carries it\n$D={D:+.2f}$, CI includes zero",
                  fontsize=7.6, pad=4, color=INK)
    axR.grid(axis="y", visible=False)

    q = os.path.join(OUT, "fC_matching.png")
    fig.savefig(q, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(q)



def fig_grid():
    """fD: the 8x8 fit x read grid of raw read AUC, with both margins.

    DELIBERATELY NUMBERLESS IN THE MATRIX. The finding is a ROW pattern: Github's operator reads
    worst against every context and StackExchange's best, whatever is read. Sixty-four four-digit
    numbers hide that pattern, because the eye reads digits instead of rows, and at any width that
    fits a single column they are too small to read anyway. Colour carries the pattern; the two
    margin strips carry the numbers a reader actually compares; the 64 cell values are tabulated in
    the appendix for anyone who wants them.

    Drawn at full text width and placed at \\linewidth, so the scale factor is 1.

    Under `min`, the source's own statistic and this paper's primary. Regenerated at build time
    from results/e52_factorial_410m_rstrip.json.
    """
    import numpy as np
    o = _load("e52_factorial_410m_rstrip.json")
    cells = o["by_aggregation"]["min"]["matrix"]
    fits = o["fit_corpora"]
    reads = fits

    M = np.full((len(fits), len(reads)), np.nan)
    for i, f in enumerate(fits):
        for j, r in enumerate(reads):
            if f"{f}|{r}" in cells:
                M[i, j] = cells[f"{f}|{r}"] * 1e2
    row_mean = np.nanmean(M, axis=1)
    col_mean = np.nanmean(M, axis=0)

    # The fit-axis variance share, read from the same decomposition the text cites.
    share = _load("e57_grid_variance_ci_rstrip.json")["by_aggregation"]["min"]["point"]["fit_pct"]

    n = len(fits)
    fig = plt.figure(figsize=(TW, 2.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[n, 1.05], height_ratios=[n, 1.05],
                          wspace=0.045, hspace=0.045, left=0.145, right=0.855, top=0.97,
                          bottom=0.30)
    axM = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, 0])

    # The unfitted logit lens is the floor of the scale, not an absent reference. Read from
    # results at build time.
    FREE = _load("e48_crossover_410m_rstrip.json")["arms_admitted_mean"]["logit_I"]["min"] * 1e2
    vmin, vmax = min(float(np.nanmin(M)), FREE), float(np.nanmax(M))
    kw = dict(cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")

    def cells_of(ax, A, fs=None):
        im = ax.imshow(A, **kw)
        if fs is not None:
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if np.isnan(A[i, j]):
                        continue
                    rel = (A[i, j] - vmin) / max(vmax - vmin, 1e-12)
                    ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=fs,
                            color="white" if rel < 0.55 else "#101010")
        ax.set_xticks(range(A.shape[1])); ax.set_yticks(range(A.shape[0]))
        ax.grid(False)
        return im

    # At full text width nine columns give ~0.55in each, so 5-character values fit at 6pt. The
    # numbers were unreadable at 0.58 textwidth, which is why they came out; the fix was the
    # width, not the digits.
    im = cells_of(axM, M, fs=6.0)
    axM.set_xticklabels([])
    axM.set_yticklabels([CORPUS_LABEL.get(f, f) for f in fits], fontsize=7.4)
    axM.set_ylabel("operator FITTED on", fontsize=8.2)

    cells_of(axR, row_mean.reshape(-1, 1), fs=6.4)
    axR.set_xticklabels(["fit\nmean"], fontsize=7.2); axR.set_yticklabels([])

    cells_of(axB, col_mean.reshape(1, -1), fs=6.4)
    axB.set_yticklabels(["read mean"], fontsize=7.2)
    axB.set_xticklabels([CORPUS_LABEL.get(r, r) for r in reads], rotation=34, ha="right",
                        fontsize=7.4)
    axB.set_xlabel("READ context drawn from", fontsize=8.2)

    axM.set_title(f"row structure dominates: the fitting corpus takes {share:.1f}% of grid variance",
                  fontsize=7.8, color=INK, loc="left", pad=4)

    cax = fig.add_axes([0.875, 0.36, 0.016, 0.58])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6.8)
    # The bar's floor IS the unfitted logit lens, so naming it in the label needs no annotation
    # and cannot collide with anything.
    cb.set_label(f"read AUC ($\\times10^{{-2}}$); scale floor is the\nunfitted logit lens, {FREE:.1f}",
                 fontsize=7.2, labelpad=6)

    q = os.path.join(OUT, "fD_grid.png")
    fig.savefig(q, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(q)


def fig_validity():
    cv2 = _load("cv2_position_support.json")
    cv4 = _load("cv4_phase1_capability.json")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(TW, FIGH),
                                   gridspec_kw={"width_ratios": [1.08, 1.0]})

    lo, hi = cv2["fitting_support"]["inclusive_range"]
    sets = list(cv2["by_set"].keys())
    yy = list(range(len(sets)))
    axL.axvspan(lo, hi, color=PURPLE, alpha=0.13, lw=0)
    for i, s in enumerate(sets):
        q0, pf = cv2["by_set"][s]["Q0"], cv2["by_set"][s]["prefixed"]
        axL.plot([q0["min"], q0["max"]], [i - 0.17, i - 0.17], color=YELLOW, lw=3.6,
                 solid_capstyle="butt")
        axL.plot([q0["median_read_pos"]], [i - 0.17], marker="|", color=INK, ms=6.5, mew=1.1)
        axL.plot([pf["min"], pf["max"]], [i + 0.17, i + 0.17], color=BLUE, lw=3.6,
                 solid_capstyle="butt")
        axL.plot([pf["median_read_pos"]], [i + 0.17], marker="|", color=INK, ms=6.5, mew=1.1)
    axL.set_yticks(yy)
    axL.set_yticklabels(sets, fontsize=8)
    axL.set_ylim(len(sets) - 0.45, -0.80)
    axL.set_xlim(0, 190)
    axL.set_xlabel("token position at which the lens is read", fontsize=7.5)
    axL.grid(axis="y", visible=False)
    axL.set_title("(a)", fontsize=8.5, loc="left", color=INK, pad=4)
    axL.text((lo + hi) / 2, -0.46, "fitting support", ha="center", va="center",
             fontsize=7.2, color="#2f6b53")
    _below(axL, [Patch(color=YELLOW), Patch(color=BLUE)],
           ["no prefix", "128-token prefix"], 2)

    models = ["70m", "160m", "410m", "1b", "1.4b", "2.8b"]
    mlabel = ["70M", "160M", "410M", "1B", "1.4B", "2.8B"]
    x = list(range(len(models)))
    pool = [cv4["by_model"][m]["_pooled_answer"] for m in models]
    t1 = [p["top1_rate"] * 100 for p in pool]
    t1err = [[t1[i] - p["top1_wilson95"][0] * 100 for i, p in enumerate(pool)],
             [p["top1_wilson95"][1] * 100 - t1[i] for i, p in enumerate(pool)]]
    t10 = [p["top10_rate"] * 100 for p in pool]
    t10err = [[t10[i] - p["top10_wilson95"][0] * 100 for i, p in enumerate(pool)],
              [p["top10_wilson95"][1] * 100 - t10[i] for i, p in enumerate(pool)]]
    sur = [cv4["by_model"][m]["_pooled_surprisal_all_six"] for m in models]

    axR.errorbar(x, t10, yerr=t10err, color=BLUE, marker="s", ms=4, lw=1.4, capsize=2.5)
    axR.errorbar(x, t1, yerr=t1err, color=ORANGE, marker="o", ms=4, lw=1.4, capsize=2.5)
    axR.set_xticks(x)
    axR.set_xticklabels(mlabel, fontsize=7.2)
    axR.set_xlim(-0.35, 5.35)
    axR.set_ylim(0, 42)
    axR.set_ylabel("percent of answerable items", fontsize=7.5)
    axR.set_xlabel("Pythia deduped", fontsize=7.5)
    axR.set_title("(b)", fontsize=8.5, loc="left", color=INK, pad=4)

    ax2 = axR.twinx()
    ax2.plot(x, sur, color=PINK, marker="^", ms=4, lw=1.4, ls="--")
    ax2.set_ylabel("mean prompt NLL (nats)", fontsize=7.5, color=PINK)
    ax2.tick_params(axis="y", colors=PINK, labelsize=7.5)
    ax2.grid(visible=False)
    ax2.set_ylim(3.7, 5.5)

    _below(axR,
           [Line2D([], [], marker="s", color=BLUE, ms=4, lw=1.4),
            Line2D([], [], marker="o", color=ORANGE, ms=4, lw=1.4),
            Line2D([], [], marker="^", color=PINK, ms=4, lw=1.4, ls="--")],
           ["answer in top 10", "answer at top 1", "prompt surprisal"], 2)

    fig.tight_layout(pad=0.5)
    p = os.path.join(OUT, "fB_validity.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return p


def fig_membership():
    """fM: 32-gram containment against the read it produces, eight fitting corpora.

    The x axis is measured exposure to the model's own published training stream; the y is the
    corrected read the corpus's operator produces, averaged over the eight read contexts of the
    factorial. If membership were the mechanism the points would climb left to right. They do not.

    Two rank correlations exist and they are not the same quantity. Table 2 reports 0.00 on the
    PRE-CORRECTION persist read that e48c itself computed. This panel plots the CORRECTED min read,
    the paper's primary, on which the same correlation is -0.10. Both are reported; a caption
    carrying Table 2's number over this panel's axis would be false.

    Sources: results/e48c_exposure_vs_read.json (containment, tier), and the fit-row means of
    results/e52_factorial_410m_rstrip.json (the read).
    """
    import numpy as np
    e48c = _load("e48c_exposure_vs_read.json")
    e52 = _load("e52_factorial_410m_rstrip.json")
    cv3 = _load("cv3_margins_410m.json")

    cells = e52["by_aggregation"]["min"]["matrix"]
    fits = e52["fit_corpora"]
    read = {f: 1e2 * float(np.mean([cells[f"{f}|{r}"] for r in fits if f"{f}|{r}" in cells]))
            for f in fits}
    cont = {c: v["containment_k32"] for c, v in e48c["by_rung"].items()}
    tier = {c: v["tier"] for c, v in e48c["by_rung"].items()}

    def spearman(xs, ys):
        def rank(a):
            s = sorted(range(len(a)), key=lambda i: a[i])
            r = [0.0] * len(a)
            i = 0
            while i < len(a):
                j = i
                while j + 1 < len(a) and a[s[j + 1]] == a[s[i]]:
                    j += 1
                for k in range(i, j + 1):
                    r[s[k]] = (i + j) / 2 + 1
                i = j + 1
            return r
        rx, ry = rank(xs), rank(ys)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
        return num / den

    order = [c for c in fits if c in cont]
    rho = spearman([cont[c] for c in order], [read[c] for c in order])
    sd = 1e2 * cv3["rank_space"]["pooled_seed_sd"]

    fig, ax = plt.subplots(figsize=(3.96, 2.45))
    ins = [c for c in order if tier[c] == "in-stream"]
    oos = [c for c in order if tier[c] == "OOD"]
    gh = [c for c in order if tier[c] not in ("in-stream", "OOD")]

    # The band is the spread of every corpus that IS in the stream, Github included: it is
    # in-stream at 0.934 and it is the worst reader, which is half the finding. e48c tiers it
    # apart for a different reason (it is the leverage point of seven prior analyses).
    instream_all = ins + gh
    lo = min(read[c] for c in instream_all)
    hi = max(read[c] for c in instream_all)
    ax.axhspan(lo, hi, color=BLUE, alpha=0.07, zorder=0)
    ax.axhline(lo, color=BLUE, lw=0.6, alpha=0.45, zorder=0)
    ax.axhline(hi, color=BLUE, lw=0.6, alpha=0.45, zorder=0)

    for group, colour, marker, label in ((ins, BLUE, "o", "in stream"),
                                         (gh, ORANGE, "D", "in stream (Github)"),
                                         (oos, PURPLE, "s", "out of stream")):
        if not group:
            continue
        ax.errorbar([cont[c] for c in group], [read[c] for c in group], yerr=sd,
                    fmt=marker, ms=5, color=colour, ecolor=colour, elinewidth=1.0,
                    capsize=2, lw=0, zorder=3, label=label)
    # The four in-stream corpora sit within 0.016 of each other on a log axis, so a centred
    # label above every mark is four labels in one place. Offset by rank within the cluster.
    for c in order:
        near = sorted((d for d in order
                       if abs(np.log10(cont[d]) - np.log10(cont[c])) < 0.4),
                      key=lambda d: read[d])
        i = near.index(c)
        dx, ha = ((-8, "right") if i % 2 else (8, "left")) if len(near) > 1 else (0, "center")
        ax.annotate(CORPUS_LABEL.get(c, c), (cont[c], read[c]), textcoords="offset points",
                    xytext=(dx, 9 if len(near) == 1 else -2), ha=ha, va="center",
                    fontsize=6.6, color=INK2)

    ax.set_xscale("log")
    ax.set_xlim(4e-5, 3.0)
    ax.set_xlabel("32-gram containment against the published training stream", fontsize=8.2)
    ax.set_ylabel("read AUC ($\\times10^{-2}$)", fontsize=8.2)
    ax.tick_params(labelsize=7.8)
    ax.set_title(f"Spearman {rho:+.2f} over eight corpora", fontsize=8.2, color=INK2, pad=3)
    # the x label and the legend collided at y=-0.34 once the panel narrowed
    _below(ax, *[[h for h in ax.get_legend_handles_labels()[i]] for i in (0, 1)], 3, y=-0.30)

    fig.tight_layout(pad=0.5)
    p = os.path.join(OUT, "fM_membership.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return p


def fig_union():
    """fU: real minus layer-deranged AUC under `min`, swept over band width 1 to 13.

    `min` is a union over the band. A union's probability grows with the number of events and
    shrinks with their correlation, and deranging the layers decorrelates them. So the wider the
    band, the more the derangement is paid for being decorrelated rather than for being right.
    The crossing is where that payment exceeds the operator's own layer-specific signal.

    Regenerated from results/d1_min_union_diagnostic_410m.json at build time.
    """
    d1 = _load("d1_min_union_diagnostic_410m.json")
    fig, ax = plt.subplots(figsize=(TW * 0.62, 2.30))
    colours = {"Pile-CC": BLUE, "USPTO_Backgrounds": PURPLE, "Github": ORANGE}
    marks = {"Pile-CC": "o", "USPTO_Backgrounds": "s", "Github": "D"}

    ax.axhline(0.0, color=INK3, lw=1.0, zorder=1)
    handles, labels = [], []
    for corpus, adj in d1["adjudication"].items():
        gaps = adj["H4_artifact_scales_with_band_width"]["min_gap_jp_minus_shuf_by_band_width"]
        w = sorted(int(k) for k in gaps)
        y = [gaps[str(k)] for k in w]
        col, mk = colours.get(corpus, INK2), marks.get(corpus, "o")
        ax.plot(w, y, color=col, marker=mk, ms=3.4, lw=1.3, zorder=3)
        # first width at which the real operator stops leading
        cross = next((k for k, v in zip(w, y) if v <= 0), None)
        if cross is not None and cross > 1:
            ax.axvline(cross, color=col, lw=0.7, ls=(0, (2, 1.6)), alpha=0.7, zorder=2)
            ax.annotate(f"{cross}", (cross, max(y)), textcoords="offset points",
                        xytext=(2, -2), fontsize=6.0, color=col)
        handles.append(Line2D([], [], color=col, marker=mk, ms=3.4, lw=1.3))
        labels.append(CORPUS_LABEL.get(corpus, corpus))

    ax.set_xticks(range(1, 14))
    ax.set_xlabel("band width (layers)", fontsize=7.5)
    ax.set_ylabel("real $-$ deranged AUC, min", fontsize=7.5)
    ax.tick_params(labelsize=7.0)
    ax.set_title("above 0: the operator leads its own derangement", fontsize=7.4,
                 color=INK2, pad=3)
    _below(ax, handles, labels, 3, y=-0.34)

    fig.tight_layout(pad=0.5)
    p = os.path.join(OUT, "fU_union.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return p


if __name__ == "__main__":
    print(fig_effect_and_null())
    fig_matching()
    fig_grid()
    print(fig_validity())
    print(fig_membership())
    print(fig_union())
