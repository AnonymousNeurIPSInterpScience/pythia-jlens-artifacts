#!/usr/bin/env python3
"""
Corpus-clustered paired t on the derangement gap, at both readouts.

STATUS: RECOMPUTATION. Measures nothing new. It reads the per-corpus gaps E54 already stored and
computes the statistic at the replication unit the programme's own discipline demands, because
the number the notes carry (t(7) = -3.3) was computed at the LEGACY readout and no results file
holds the corrected-readout value.

WHY THIS EXISTS. E54's C2_derangement reports 84 of 120 paired draws and 7 of 8 corpora at the
corrected readout, but five derangements of one operator share an operator and a cache, so draws
are not independent replicates. The corpus is the replication unit. The paired t over the eight
per-corpus gaps is therefore the inferential statement and the draw count is descriptive.

WHAT IT FINDS, and it matters for the paper: at the CORRECTED readout the clustered statistic is
t(7) = -1.92, p = 0.096, which does NOT clear 5%. At the legacy readout it is t(7) = -3.28,
p = 0.013. The readout correction moved this from significant to not, and `docs/context/CONTEXT.md`
section 5 plus `RESULTS_TAXONOMY.md` section 1.2 still print the legacy -3.3 as though it were the
corrected-readout figure.

This does NOT weaken the metric-audit result. R9 is the load-bearing statement and it does not
require the derangement to win: under `min` no operator out of eight beats all fifteen
derangements of itself. This file only fixes which number may be quoted as the inferential test
of "min actively prefers the derangement".

  .venv/bin/python tools/paper_clustered_derangement_t.py
"""
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

ARMS = {
    "corrected": "results/e54_aggregation_audit_rstrip_v2.json",
    "legacy": "results/e54_aggregation_audit.json",
}
AGGS = ("min", "persist")


def _find(obj, key):
    """Locate a key anywhere in the nested record. E54 nests C2 under a controls block whose exact
    path has moved between versions of the file, so we search rather than hardcode a path."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                return v
            hit = _find(v, key)
            if hit is not None:
                return hit
    return None


def _student_t_sf(t, df):
    """Two-sided p for Student's t, via the regularised incomplete beta. scipy is not a dependency
    of this tree, so the identity p = I_{df/(df+t^2)}(df/2, 1/2) is evaluated directly."""
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a,b) by continued fraction (Lentz), as in Numerical Recipes."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log(1.0 - x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def _betacf(a, b, x, itmax=200, eps=3e-16, fpmin=1e-300):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def main():
    out = {
        "experiment": "PAPER — corpus-clustered paired t on the derangement gap",
        "status": "RECOMPUTATION from stored per-corpus gaps; measures nothing new",
        "why": "the draw count (84 of 120) is descriptive because five derangements of one operator "
               "share an operator and a cache; the corpus is the replication unit",
        "statistic": "one-sample paired t over the 8 per-corpus values of (jp_mean - shuf_mean), "
                     "df = 7, two-sided p from Student's t",
        "source_key": "C2_derangement.<agg>.per_corpus.<corpus>.gap",
        "by_readout": {},
    }
    inputs = []
    for readout, rel in ARMS.items():
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            raise SystemExit(f"missing input: {rel}")
        inputs.append(path)
        c2 = _find(json.load(open(path)), "C2_derangement")
        if c2 is None:
            raise SystemExit(f"{rel} has no C2_derangement block")
        arm = {"file": rel}
        for agg in AGGS:
            per = c2[agg]["per_corpus"]
            gaps = {k: v["gap"] for k, v in per.items()}
            g = list(gaps.values())
            n = len(g)
            if n < 2:
                raise SystemExit(f"{rel}/{agg}: need at least 2 corpora, got {n}")
            mean = st.mean(g)
            sd = st.stdev(g)
            if sd == 0.0:
                raise SystemExit(f"{rel}/{agg}: zero between-corpus SD, t is undefined")
            t = mean / (sd / math.sqrt(n))
            p = _student_t_sf(t, n - 1)
            arm[agg] = {
                "n_corpora": n,
                "per_corpus_gap": gaps,
                "mean_gap": mean,
                "sd_gap": sd,
                "t": t,
                "df": n - 1,
                "p_two_sided": p,
                "clears_05": p < 0.05,
                "n_draws_shuf_beats_jp": c2[agg]["shuf_beats_jp_paired_by_seed"],
                "n_draws": c2[agg]["n_draws"],
                "n_corpora_shuf_beats_on_mean": c2[agg]["n_corpora_where_shuf_beats_jp_on_the_mean"],
            }
        out["by_readout"][readout] = arm

    cm = out["by_readout"]["corrected"]["min"]
    lm = out["by_readout"]["legacy"]["min"]
    cp = out["by_readout"]["corrected"]["persist"]
    out["VERDICT"] = (
        f"At the CORRECTED readout the corpus-clustered paired t on `min` is "
        f"t({cm['df']}) = {cm['t']:.2f}, p = {cm['p_two_sided']:.3f}, which does NOT clear 5% — "
        f"the descriptive counts are {cm['n_draws_shuf_beats_jp']} of {cm['n_draws']} draws and "
        f"{cm['n_corpora_shuf_beats_on_mean']} of {cm['n_corpora']} corpora. At the LEGACY readout "
        f"it is t({lm['df']}) = {lm['t']:.2f}, p = {lm['p_two_sided']:.3f}. The readout correction "
        f"moved this from significant to not, so the -3.3 carried in CONTEXT.md section 5 and "
        f"RESULTS_TAXONOMY.md section 1.2 is a LEGACY-readout number. `persist` is unambiguous at "
        f"both readouts: t({cp['df']}) = {cp['t']:.2f}, p = {cp['p_two_sided']:.2e} corrected."
    )
    out["scope"] = (
        "A derangement is a null for layer-to-derivative CORRESPONDENCE, not for the claim that the "
        "Jacobians carry layer-specific information. A t that does not clear 5% means the evidence "
        "that `min` actively PREFERS the derangement is weaker than the draw count suggests; it does "
        "not rehabilitate `min`, because R9 shows separately that no operator out of eight beats all "
        "fifteen derangements of itself under `min`."
    )

    dest = os.path.join(ROOT, "results", "paper_clustered_derangement_t.json")
    try:
        from provenance import write_result
        write_result(dest, out, script=__file__,
                     experiment="PAPER_CLUSTERED_T", inputs=inputs)
    except Exception as e:
        print(f"  !! provenance stamp FAILED: {e!r}", file=sys.stderr)
        json.dump(out, open(dest, "w"), indent=1)

    print("\n" + "=" * 78)
    for readout in ARMS:
        for agg in AGGS:
            a = out["by_readout"][readout][agg]
            print(f"{readout:10} {agg:8} t({a['df']}) = {a['t']:+.3f}  p = {a['p_two_sided']:.4f}  "
                  f"draws {a['n_draws_shuf_beats_jp']}/{a['n_draws']}  "
                  f"corpora {a['n_corpora_shuf_beats_on_mean']}/{a['n_corpora']}")
    print("\n" + out["VERDICT"])


if __name__ == "__main__":
    main()
