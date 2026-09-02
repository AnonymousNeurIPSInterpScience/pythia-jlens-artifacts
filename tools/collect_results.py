#!/usr/bin/env python3
"""collect_results.py — build the per-model results tables in RESEARCH_NOTES from results/*.json.

WHY THIS EXISTS. The operator's complaint was "random ass numbers": a table cell that cannot be
traced to a file, and whose metric is not labelled, is not a result. So this script is the ONLY
producer of the per-model tables. Every cell it emits carries the file it came from. A cell with
no file is emitted as an explicit empty box, never as a blank or a dash.

It emits LaTeX to stdout (or --out). RESEARCH_NOTES \input{}s the generated file, so the document
can never drift from results/.

METRIC REGISTRY. Every metric the programme has DEFINED appears as a row for EVERY model, whether
or not it is measured. That is the point: the shape of what is missing is the research plan.
"""
import argparse, json, os, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # the repo root
RES = os.path.join(ROOT, "results")

MODELS = [
    ("70m",   6,  512,  (2, 6)),
    ("160m",  12, 768,  (5, 11)),
    ("410m",  24, 1024, (9, 22)),
    ("1b",    16, 2048, (6, 15)),
    ("1.4b",  24, 2048, (9, 22)),
    ("2.8b",  32, 2560, (12, 29)),
    ("6.9b",  32, 4096, (12, 29)),
    ("12b",   36, 5120, (14, 33)),
]
EVALS = ["multihop", "multilingual", "order-ops", "poetry", "typo", "association"]
# The six transports t13 actually scores. scaled_id is the SELF-TEST: it must equal logit
# exactly, and if it does not the harness is broken. tuned (A) is the one arm we never built.
TRANSPORTS = [("logit", "$I$"), ("jlens", "$J^P$"), ("shuffled", "$J^{\\mathrm{shuf}}$"),
              ("spec_random", "spec-rand"), ("rank1", "rank-1"),
              ("scaled_id", "$cI$ (self-test)"), ("tuned", "$A$ (tuned)")]

# Models that PREREG_PYTHIA_T7_v2 sec2 registers as confirmatory. Never scored without --prereg.
CONFIRMATORY = {"1b", "1.4b", "2.8b"}


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None, None
    try:
        with open(p) as f:
            return json.load(f), os.path.basename(p)
    except Exception:
        return None, None


def first(*names):
    """Return (data, filename) for the first name that exists."""
    for n in names:
        d, f = load(n)
        if d is not None:
            return d, f
    return None, None


def tex(s):
    return (str(s).replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


EMPTY = r"\cell"          # defined in RESEARCH_NOTES preamble as a grey open box
def fmt(v, nd=4):
    if v is None:
        return EMPTY
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return tex(v)


def src(f):
    return r"{\tiny\texttt{" + tex(f) + "}}" if f else r"{\tiny\textit{---}}"


# ----------------------------------------------------------------------------- metric extractors
def m_read_auc(m):
    """R2: normalised pass@k AUC per eval set per transport. The core read table."""
    d, f = first(f"t13_transport_controls_{m}.json", f"t7_lens_comparison_{m}.json")
    if d is None:
        return None, None
    out = {}
    for ev, cell in (d.get("results") or {}).items():
        row = {}
        for key, _ in TRANSPORTS:
            arm = cell.get(key) or cell.get(key + "_lens")
            if isinstance(arm, dict):
                # auc_paper is the A.6 per-item binary metric and is PRIMARY (t13);
                # older t7 files store it under normalized_auc_over_logk_paper.
                row[key] = (arm.get("auc_paper")
                            or arm.get("normalized_auc_over_logk_paper")
                            or arm.get("normalized_auc_over_logk"))
        out[ev] = row
    return out, f


def m_nwin(m):
    """R3: n_win vs the logit lens under each of the four aggregations."""
    d, f = load(f"t17_reaggregate_{m}.json")
    if d is None:
        return None, None
    return d.get("n_win_vs_logit_by_aggregation"), f


def m_disp(m):
    """O1: Jacobian dispersion at L0 and at the last source layer."""
    d, f = load("t14b_dispersion_scaling.json")
    if d is None:
        return None, None
    for r in d.get("rows", []):
        if str(r.get("model", "")) == m:
            return r, f
    return None, f


def m_operr(m):
    """O2: relative operator error at the n we actually fit with."""
    d, f = load("en1_operator_error_at_n200.json")
    if d is None:
        return None, None
    pm = d.get("per_model") or {}
    for k, v in pm.items():
        if k.endswith(m) or k == m:
            return v, f
    return None, f


def m_nlaw(m):
    """O3/O4: the sqrt(disp/n) slope, and the reference-free disagreements."""
    d, f = load(f"t18_n_sweep_{m}.json")
    if d is None:
        return None, None
    return d, f


def m_robust(m):
    """O5: cos(coordinatewise median, mean) -- the E2 estimator-fragility control."""
    d, f = load(f"t16_robust_aggregation_{m}.json")
    if d is None:
        return None, None
    return d.get("summary"), f


def m_skipfirst(m):
    d, f = load(f"t19_skipfirst_{m}.json")
    return (d, f) if d else (None, None)


def m_ablation_kl(m):
    """W3: the ablation-KL ratio per eval set. The first write metric in the programme."""
    d, f = load(f"t20_ablation_kl_{m}.json")
    if d is None:
        return None, None
    return d, f


def m_capability(m):
    """C1: t5a battery accuracy, with its denominator (the mixed-denominator defect)."""
    d, f = first(f"t5a_capability_{m}_v2.json", f"t5a_capability_{m}.json")
    if d is None:
        return None, None
    return d, f


def m_selfread(m):
    """R1: self-read. Kept because its control was near-vacuous and that must stay visible."""
    d, f = first(f"t3_{m}_FIXED.json", f"t3_{m}_db128.json", f"t3_{m}_n200.json")
    return (d, f) if d else (None, None)


def m_lens(m, tgt):
    d, f = load(f"lens_{m}_n200_db128_{tgt}_provenance.json")
    if d is None and tgt == "fin":
        d, f = load(f"lens_{m}_n200_db128_provenance.json")
    return (d, f) if d else (None, None)


def _rank(m):
    d, f = first(f"t13_transport_controls_{m}.json")
    if d is None:
        return None
    cell = ((d.get("results") or {}).get("multihop") or {}).get("jlens") or {}
    return cell.get("median_best_rank")


# ----------------------------------------------------------------------------------------- render
def render_model(m, nl, dm, band, w):
    lo, hi = band
    w(r"\subsection*{Pythia-" + m.upper().replace("M", "M").replace("B", "B") +
      r" \quad\small($n_{\mathrm{layers}}=" + str(nl) + r"$, $d_{\mathrm{model}}=" + str(dm) +
      r"$, band L" + str(lo) + "--L" + str(hi) + ")}")
    if m in CONFIRMATORY:
        w(r"\nopagebreak\par{\small\textbf{Confirmatory model} (PREREG \S2). "
          r"Nothing may be scored here without \texttt{--prereg}; the runner refuses.}\par")

    # ---------- lens inventory
    rows = []
    for tgt, lbl in (("pen", "penultimate"), ("fin", "final")):
        d, f = m_lens(m, tgt)
        rows.append((lbl, ("fitted" if d else None), f))
    w(r"\vspace{2pt}\noindent{\small\textbf{Lenses.}}\par\nopagebreak")
    w(r"\begin{center}\scriptsize\begin{tabular}{@{}lll@{}}\toprule")
    w(r"target & status & provenance file \\ \midrule")
    for lbl, st, f in rows:
        w(f"{lbl} & {fmt(st)} & {src(f)} " + r"\\")
    w(r"\bottomrule\end{tabular}\end{center}")

    # ---------- READ: the full per-eval-set, per-transport grid
    auc, faux = m_read_auc(m)
    w(r"\vspace{-4pt}\noindent{\small\textbf{R2 --- read: normalised pass@$k$ AUC over $\log k$.} "
      r"One number per eval set per transport. Higher is better. "
      r"\textit{Final-target lens; per-layer max (existential).}}\par\nopagebreak")
    w(r"\begin{center}\scriptsize\begin{tabular}{@{}l" + "r" * len(TRANSPORTS) + r"@{}}\toprule")
    w("eval set & " + " & ".join(lbl for _, lbl in TRANSPORTS) + r" \\ \midrule")
    for ev in EVALS:
        cells = []
        for key, _ in TRANSPORTS:
            v = (auc or {}).get(ev, {}).get(key) if auc else None
            cells.append(fmt(v))
        w(tex(ev) + " & " + " & ".join(cells) + r" \\")
    w(r"\bottomrule\end{tabular}\par" + src(faux) + r"\end{center}")

    # ---------- everything else: one row per DEFINED metric, populated or not
    nw, fnw = m_nwin(m)
    dsp, fdsp = m_disp(m)
    oe, foe = m_operr(m)
    nl_, fnl = m_nlaw(m)
    rb, frb = m_robust(m)
    sf, fsf = m_skipfirst(m)
    kl, fkl = m_ablation_kl(m)
    cp, fcp = m_capability(m)
    sr, fsr = m_selfread(m)

    R = []  # (family, id, metric, value, source)
    R.append(("READ", "R1", "self-read $R1$ (band max)",
              fmt(sr.get("any_layer_beats_all_nulls")) if sr else EMPTY, fsr))
    for agg, lbl in (("min", "min-over-layers \\textit{(existential; discredited by E14)}"),
                     ("best1L", "best-single-layer"),
                     ("mean", "mean-over-layers"),
                     ("persist", "persistence-at-$K$ \\textit{(non-existential)}")):
        v = (nw or {}).get(agg)
        R.append(("READ", "R3", f"$n_{{\\mathrm{{win}}}}$ vs $I$, {lbl}",
                  (f"{v}/6" if v is not None else EMPTY), fnw))
    mbr, fmbr = m_read_auc(m)
    R.append(("READ", "R4", "median best rank, $J^P$ on multihop",
              fmt(_rank(m)) if _rank(m) is not None else EMPTY, fmbr))

    # WRITE
    if kl:
        for ev in EVALS:
            cell = (kl.get("results") or {}).get(ev) or {}
            r = cell.get("ratio_jlens_over_logit")
            R.append(("WRITE", "W3", f"ablation KL ratio $J^P\\!/\\!I$, {tex(ev)}",
                      fmt(r), fkl))
        R.append(("WRITE", "W3c", "ablation KL, $J^{\\mathrm{shuf}}$ control fired",
                  fmt("yes" if kl.get("results") else None), fkl))
    else:
        for ev in EVALS:
            R.append(("WRITE", "W3", f"ablation KL ratio $J^P\\!/\\!I$, {tex(ev)}", EMPTY, None))
        R.append(("WRITE", "W3c", "ablation KL, $J^{\\mathrm{shuf}}$ control fired", EMPTY, None))
    R.append(("WRITE", "W1", "swap success (coordinate write)", EMPTY, None))
    R.append(("WRITE", "W2", "steering success", EMPTY, None))
    R.append(("WRITE", "W4", "$\\lVert\\Delta\\rVert$ per layer \\& per position",
              fmt("recorded" if kl else None), fkl))

    # OPERATOR / ESTIMATOR
    R.append(("OPER", "O1", "dispersion $\\mathrm{disp}(L_0)$",
              fmt(dsp.get("disp_layer0")) if dsp else EMPTY, fdsp))
    R.append(("OPER", "O1", "\\# source layers the operator spans",
              fmt(dsp.get("n_source")) if dsp else EMPTY, fdsp))
    R.append(("OPER", "O2", "operator error $\\epsilon$ at $n{=}200$, $L_0$",
              fmt(oe.get("eps_L0_at_200")) if oe else EMPTY, foe))
    R.append(("OPER", "O2", "operator error $\\epsilon$ at $n{=}200$, last layer",
              fmt(oe.get("eps_last_at_200")) if oe else EMPTY, foe))
    R.append(("OPER", "O2", "$n$ required for $\\epsilon\\le5\\%$ at $L_0$",
              fmt(oe.get("n_for_5pct_at_L0")) if oe else EMPTY, foe))
    R.append(("OPER", "O3", "log--log slope of $\\epsilon$ vs $n$ (law: $-0.5$)",
              fmt(nl_.get("slope_mean")) if nl_ else EMPTY, fnl))
    R.append(("OPER", "O4", "split-half disagreement (reference-free)",
              fmt(nl_["reference_free"][str(nl_["layers"][0])]["split_half_disagreement"])
              if nl_ else EMPTY, fnl))
    R.append(("OPER", "O4", "odd--even disagreement (reference-free)",
              fmt(nl_["reference_free"][str(nl_["layers"][0])]["odd_even_disagreement"])
              if nl_ else EMPTY, fnl))
    R.append(("OPER", "O5", "$\\cos(\\mathrm{median},\\mathrm{mean})$, min over layers",
              fmt(rb.get("min_cos_median_vs_mean")) if rb else EMPTY, frb))
    R.append(("OPER", "O6", "\\texttt{skip\\_first} $n_{\\mathrm{win}}$ spread over $\\{0,4,16\\}$",
              fmt(sf.get("spread")) if sf else EMPTY, fsf))

    # DISTRIBUTION -- the whole point of the programme, and entirely unmeasured
    for mid, lbl in (("M1", "hard containment (membership in the token stream)"),
                     ("M2", "$\\mathrm{CE}_M-\\mathrm{CE}_R$ (reference-calibrated loss)"),
                     ("M3", "layerwise domain-classifier AUC (cross-fitted)"),
                     ("M4", "action-on-data $\\mathbb{E}_Q\\lVert(J_P-J_Q)h\\rVert^2$")):
        R.append(("DIST", mid, lbl, EMPTY, None))

    # CAPABILITY
    if cp:
        R.append(("CAP", "C1", f"\\texttt{{t5a}} accuracy (denominator {cp.get('n_cells')})",
                  fmt(cp.get("accuracy")), fcp))
        R.append(("CAP", "C1", "write battery admissible ($\\ge32$ cells)",
                  fmt(str(cp.get("WRITE_BATTERY_ADMISSIBLE"))), fcp))
    else:
        R.append(("CAP", "C1", "\\texttt{t5a} accuracy", EMPTY, None))
        R.append(("CAP", "C1", "write battery admissible ($\\ge32$ cells)", EMPTY, None))
    R.append(("CAP", "C2", "IRT $\\theta_m$ (hierarchical 2PL, cross-fitted)", EMPTY, None))
    R.append(("CAP", "C2", "IRT $b_i$ spread over items", EMPTY, None))

    w(r"\vspace{-4pt}\noindent{\small\textbf{All defined metrics.} An open box is a metric this "
      r"programme has DEFINED and not measured at this scale.}\par\nopagebreak")
    w(r"\begin{center}\scriptsize\begin{tabular}{@{}ll>{\raggedright\arraybackslash}p{7.0cm}rl@{}}\toprule")
    w(r"fam & id & metric & value & source \\ \midrule")
    lastfam = None
    for fam, mid, lbl, val, f in R:
        famcell = fam if fam != lastfam else ""
        if fam != lastfam and lastfam is not None:
            w(r"\addlinespace[2pt]")
        lastfam = fam
        w(f"{famcell} & {mid} & {lbl} & {val} & {src(f)} " + r"\\")
    w(r"\bottomrule\end{tabular}\end{center}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    buf = []
    w = buf.append
    w(r"% GENERATED by tools/collect_results.py -- DO NOT EDIT BY HAND.")
    w(r"% Every cell traces to a file in results/. Rerun the script to refresh.")
    for m, nl, dm, band in MODELS:
        render_model(m, nl, dm, band, w)
    out = "\n".join(buf) + "\n"
    if a.out:
        with open(a.out, "w") as f:
            f.write(out)
        print(f"wrote {a.out} ({len(buf)} lines)", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
