#!/usr/bin/env python3
"""r4_recomputes.py — R4a/b/c/h/i/j/k/l: the corrections that are pure recomputation.

PRE-REGISTRATION: docs/experiments/preregs/R4_corrections.md, committed before this script existed.
Source slate: POSTREVIEW_EXPERIMENTS.md §3 Tier A, item R4.

WHAT THIS IS AND IS NOT. Every number below is recomputed from a file already in results/. Nothing
here scores a model, so nothing here is exposed to the readout defect that R1 corrects EXCEPT where
its input was -- and that is declared per item rather than assumed.

THE STANDING RULE, from R4_corrections.md: each item carries the value the slate says closes it, and
**a recompute that does not reproduce that value is a disagreement to REPORT, not a number to
adopt.** Two items below disagree with the slate and say so in the file.

    python tools/r4_recomputes.py
"""
from __future__ import annotations
import itertools, json, math, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "src"))
from provenance import sha256_file, write_result  # noqa: E402

R = lambda *p: os.path.join(REPO, *p)
load = lambda p: json.load(open(R(p)))
INPUTS: list[str] = []


def _in(p):
    if p not in INPUTS and os.path.exists(R(p)):
        INPUTS.append(p)
    return p


# ------------------------------------------------------------------ R4a
def r4a():
    """Containment at the coverage the text claims, keyed by shard count so the columns cannot mix."""
    d = load(_in("results/e48b_exposure_growth.json"))
    cb, ks = d["containment_by_coverage"], d["k_values"]
    n_shards = d["n_shards_total_in_stream"]
    corpora = sorted(cb[str(ks[0])])
    table = {str(k): {c: {f"n_shards_{i+1}": v for i, v in enumerate(cb[str(k)][c])} for c in corpora}
             for k in ks}
    full = {c: cb["32"][c][n_shards - 1] for c in corpora}
    instream = ["Github", "Pile-CC", "StackExchange", "USPTO_Backgrounds", "Wikipedia_en"]
    ood = ["OOD_CommonPile", "OOD_News_2024", "OOD_arXiv_2023"]
    req = {"OOD_Wikipedia_2023": 0.26958, "CONTROL_PubMed_2023": 0.79273}
    obs = {k: round(full[k], 5) for k in req}
    return {
        "what": ("the k=32 containment row at FULL 20/20 shard coverage for every corpus, plus the "
                 "whole coverage curve, in one table keyed by shard count"),
        "finding": "F-8 — the text quoted a k=32 row computed at partial coverage next to one at full",
        "k_values": ks, "n_shards_total_in_stream": n_shards, "primary_k": d["primary_k"],
        "k32_at_full_coverage": full,
        "in_stream_range_at_k32_full": [min(full[c] for c in instream),
                                        max(full[c] for c in instream)],
        "ood_range_at_k32_full": [min(full[c] for c in ood), max(full[c] for c in ood)],
        "slate_required": req, "observed": obs,
        "agrees_with_slate": all(abs(obs[k] - v) < 1e-5 for k, v in req.items()),
        "full_table_by_k_and_shard_count": table,
        "why_keyed": ("stored keyed by BOTH k and shard count so a value can never again be quoted "
                      "at one coverage beside a value at another"),
        "formatting_floor": d["formatting_sensitivity_measured"],
        "reading": d["reading"],
    }


# ------------------------------------------------------------------ R4b (table)
def r4b_table():
    """The four hierarchical intervals in one place. F-5. This is UNFAVOURABLE and it is published."""
    arms = {}
    for scale in ("410m", "160m"):
        for agg, stem in (("min", f"results/t22_bootstrap_ci_{scale}.json"),
                          ("persist", f"results/t22_bootstrap_ci_persist_{scale}.json")):
            d = load(_in(stem))
            vl, vs = d["hierarchical_jlens_minus_logit"], d["hierarchical_jlens_minus_shuffled"]
            arms[f"{scale}|{agg}"] = {
                "file": stem,
                "J_minus_logit": {k: vl[k] for k in ("point", "ci_lo", "ci_hi", "excludes_zero")},
                "J_minus_shuffled": {k: vs[k] for k in ("point", "ci_lo", "ci_hi", "excludes_zero")},
                "beats_free_baseline": bool(vl["excludes_zero"] and vl["point"] > 0),
                "clears_derangement_control": bool(vs["excludes_zero"] and vs["point"] > 0),
                "hurts_vs_free_baseline": bool(vl["excludes_zero"] and vl["point"] < 0)}
    pattern = {k: (v["beats_free_baseline"], v["clears_derangement_control"]) for k, v in arms.items()}
    exclusive = all(not (b and c) for b, c in pattern.values())
    return {
        "what": "all four hierarchical intervals, 410M/160M x min/persist, J-minus-logit and J-minus-shuffled",
        "finding": "F-5 — each file reported only itself, so the pattern across them was invisible",
        "arms": arms,
        "THE_PATTERN": ("no arm both beats the free baseline AND clears the derangement control: "
                        "at 410M `min` beats the logit lens (+0.0557 [+0.0065,+0.1154]) and FAILS "
                        "the derangement control (+0.0188 [-0.0239,+0.0563]), while `persist` "
                        "clears the derangement control (+0.0417 [+0.0134,+0.0732]) and does NOT "
                        "beat the logit lens (+0.0216 [-0.0022,+0.0598]). The aggregation that "
                        "wins the comparison the paper cares about is the one that fails the "
                        "control the paper relies on."),
        "no_arm_does_both": exclusive,
        "agrees_with_slate": bool(exclusive and arms["160m|min"]["hurts_vs_free_baseline"]),
        "SLATE_CORRECTION_CONFIRMED": ("the slate's own §2 note is right and the original F-5 "
                                       "framing is wrong at 160M: NEITHER aggregation beats the "
                                       "logit lens there. Under `min` the interval excludes zero "
                                       "but is NEGATIVE (-0.0207 [-0.0418,-0.0024]) -- the lens "
                                       "measurably HURTS -- and under `persist` it includes zero."),
        "160m_min_is_negative": arms["160m|min"]["hurts_vs_free_baseline"],
    }


# ------------------------------------------------------------------ R4c
def r4c():
    """All four aggregation arms of the corpus x eval-set decomposition, not just the argmax."""
    out, missing = {}, []
    for ag in ("persist", "min", "mean", "best1L"):
        p = f"results/r4c/e51_interaction_{ag}.json"
        if not os.path.exists(R(p)):
            missing.append(p); continue
        d = load(_in(p))
        for scale in d["scales"]:
            s = d["scales"][scale].get("admitted5")
            if not s: continue
            out.setdefault(scale, {})[ag] = {
                "frac_set_main_pct": s["frac_set_main"] * 100,
                "frac_corpus_main_pct": s["frac_corpus_main"] * 100,
                "frac_interaction_pct": s["frac_interaction"] * 100,
                "frac_residual_seed_pct": s["frac_residual_seed"] * 100,
                "interaction_over_corpus_main": s["frac_interaction"] / s["frac_corpus_main"]}
    res = {"what": "the corpus x eval-set variance decomposition under all four aggregations",
           "finding": "F-5 second half — best1L and mean exist in every ladder cell and are never reported",
           "by_scale": out, "missing_inputs": missing}
    if "410m" in out:
        a = out["410m"]
        arg = max(a, key=lambda k: a[k]["frac_interaction_pct"])
        res["THE_REPORTED_SHARE_IS_THE_ARGMAX"] = {
            "reported_aggregation": "persist",
            "argmax_aggregation": arg,
            "reported_is_argmax": arg == "persist",
            "interaction_pct_by_agg": {k: round(a[k]["frac_interaction_pct"], 3) for k in a},
            "range_pp": round(max(a[k]["frac_interaction_pct"] for k in a)
                              - min(a[k]["frac_interaction_pct"] for k in a), 3),
            "what_survives": ("the ORDERING survives all four -- the interaction exceeds the "
                              "corpus main effect under every aggregation (ratio 1.05-3.56) -- "
                              "but the MAGNITUDE does not: 6.78% under `persist` against 2.00% "
                              "under `min`, a 3.4x range, and the reported value is the largest "
                              "of the four."),
            "ordering_holds_under_all_four": all(a[k]["interaction_over_corpus_main"] > 1 for k in a)}
    return res


# ------------------------------------------------------------------ R4h
def r4h():
    """The small numeric corrections, each recomputed rather than restated."""
    items = {}

    g = load(_in("results/e65_ckpt_geometry_410m.json"))["by_checkpoint"]
    ranks = sorted(((c, v["eff_rank"]) for c, v in g.items() if "eff_rank" in v), key=lambda x: x[1])
    items["M-7_effective_rank_minimum"] = {
        "claim_as_written": "effective rank never falls below 726",
        "observed_minimum": ranks[0][1], "at_checkpoint": ranks[0][0],
        "n_checkpoints": len(ranks), "n_below_726": sum(1 for _, v in ranks if v < 726),
        "slate_required": 725.888, "agrees_with_slate": abs(ranks[0][1] - 725.888) < 1e-3,
        "verdict": "the claim is FALSE; one checkpoint of 19 falls below, at 725.888"}

    e47 = load(_in("results/e47_ablation_pairs.json"))["summary"]
    top, rnd, bot = e47["selectivity_top"], e47["selectivity_random"], e47["selectivity_bottom"]
    rb = [k for k in top if rnd[k] > top[k]]
    items["M-20_random_beats_top"] = {
        "claim_as_written": "random better on one pair",
        "observed_n_pairs": len(rb), "of": len(top), "which_pairs": rb,
        "all_are_github_pairs": all(k.startswith("Github") for k in rb),
        "selectivity_top": top, "selectivity_random": rnd,
        "slate_required": "2 of 5, both Github pairs",
        "agrees_with_slate": len(rb) == 2 and all(k.startswith("Github") for k in rb),
        "verdict": "the claim UNDERSTATES: random beats top-k on 2 of 5 pairs, and both are Github"}

    e53 = load(_in("results/e53_ladder_summary.json"))
    mx = e53["e36_containment_slopes"]["persist"]["_summary"]["max_abs_spearman"]
    items["M-21_max_abs_rho"] = {
        "claim_as_written": "|rho| <= 0.28",
        "observed_max_abs_spearman": mx, "slate_required": 0.2818,
        "agrees_with_slate": abs(mx - 0.2818) < 1e-3,
        "verdict": ("the claim is FALSE as written by 0.0018: the maximum is 0.28182 "
                    "(StackExchange), which exceeds 0.28. Write <= 0.29, or print the value")}

    e52 = load(_in("results/e52_factorial_410m.json"))["per_set_D"]
    zeros = {s: e52[s]["mean"] for s in e52 if abs(e52[s]["mean"]) < 1e-4}
    items["M-22_two_sets_exactly_zero"] = {
        "claim_as_written": "two eval sets give exactly zero",
        "observed": zeros,
        "slate_required": {"order-ops": -1.29e-5, "poetry": -2.89e-6},
        "agrees_with_slate": (abs(zeros.get("order-ops", 0) + 1.2883919615498084e-05) < 1e-9
                              and abs(zeros.get("poetry", 0) + 2.892319614710725e-06) < 1e-9),
        "verdict": ("neither is exactly zero; both are small and NEGATIVE. 'Exactly zero' asserts a "
                    "structural fact the numbers do not carry")}

    e58 = load(_in("results/e58_algebra_audit.json"))["C_e36_diagonal_prefix_fit_overlap"]
    items["M-22b_fit_prefix_overlap"] = {
        "claim_as_written": "the repo previously asserted ~25%",
        "observed_mean_over_corpora": e58["mean_over_corpora"],
        "per_corpus": {c: v["mean_overlap_fraction"] for c, v in e58["by_corpus"].items()},
        "slate_required": 0.1704, "agrees_with_slate": abs(e58["mean_over_corpora"] - 0.1704) < 5e-4,
        "verdict": ("the measurement is 0.17043, so a ~17% guess was VINDICATED to 0.04 pp rather "
                    "than corrected. The superseded ~25% appears in two live places, not one: "
                    "paper/CONTEXT.md and docs/context/RESULTS_TAXONOMY.md -- and RESULTS_TAXONOMY.md:66 "
                    "already states the correction in place, which the reviewer's note missed"),
        "note_on_the_reviewers_claim": ("checked directly: docs/context/RESULTS_TAXONOMY.md:66 reads "
                                        "'measured at 0.1704, not the ~25% previously asserted', "
                                        "i.e. it is a corrected restatement, not an uncorrected one")}

    items["M-16_E52_diagonal_excess_D"] = {
        "claim_as_written": "D is mislabelled",
        "verdict": ("PARTLY REFUTED, as the slate says. docs/experiments/preregs/superseded/PREREG_E52_FACTORIAL.md:69 "
                    "defines D verbatim as implemented, so it is not an implementation-vs-prereg "
                    "break. It is a redundant estimand -- D = 8/7 x the diagonal-residual mean "
                    "identically -- plus a misleading English gloss, and it inflates the reported "
                    "figure by 14.3%. The C4 permutation p is computed on D itself, so the 8/7 "
                    "factor cancels and the INFERENCE is untouched. Fix the gloss, not the number"),
        "action": "gloss only; the stored D and its p-value stand"}
    return items


# ------------------------------------------------------------------ R4i
def r4i():
    """The 1B replication re-derived on the band the paper's own rule produces."""
    b = load(_in("results/e62_band_adjudication.json"))
    e53 = load(_in("results/e53_ladder_summary.json"))
    bands = e53["bands_declared_vs_used"]
    ref = e53["by_scale"]["1b"]["persist"]

    def derive(A):
        corp = list(A)
        means = [A[c]["mean"] for c in corp]
        pooled = statistics.mean([A[c]["seed_sd"] for c in corp])   # e53's own convention
        spread = max(means) - min(means)
        sep = sum(1 for x, y in itertools.combinations(corp, 2)
                  if abs(A[x]["mean"] - A[y]["mean"])
                  / math.sqrt(A[x]["seed_sd"] ** 2 / 3 + A[y]["seed_sd"] ** 2 / 3) > 2.776)
        return {"spread": spread, "pooled_seed_sd": pooled, "spread_over_seed_sd": spread / pooled,
                "n_pairs_separating": sep, "n_pairs": 10}
    used, declared = derive(b["asymptotes"]["band_5_13_stored"]), derive(b["asymptotes"]["band_6_13"])
    c1 = abs(used["spread_over_seed_sd"] - ref["spread_over_seed_sd"]) < 1e-9
    return {
        "what": "the 1B replication, re-derived on the declared band [6,13] instead of the used [5,13]",
        "finding": "R4i, new 2026-08-19 — the 1B arm ran one layer below the paper's own band rule",
        "bands_declared_vs_used": bands,
        "C1_reproduces_e53_on_the_stored_band": {
            "required": "the stored-band re-derivation must reproduce e53's 1B spread_over_seed_sd exactly",
            "e53_stored": ref["spread_over_seed_sd"], "recomputed": used["spread_over_seed_sd"],
            "fires": c1},
        "on_the_used_band_5_13": used, "on_the_declared_band_6_13": declared,
        "interaction_share": b["PRIMARY"],
        "VERDICT": (f"the 1B claim SURVIVES the band correction. Pair separation is "
                    f"{declared['n_pairs_separating']} of 10 on the declared band and "
                    f"{used['n_pairs_separating']} of 10 on the used band -- unchanged, and the same "
                    f"pair (Github vs Wikipedia_en) is the one that does not separate in both. The "
                    f"multiplier moves {used['spread_over_seed_sd']:.2f}x -> "
                    f"{declared['spread_over_seed_sd']:.2f}x and the interaction share moves "
                    f"9.158% -> 9.192%. Report the corrected band's numbers; the deviation is "
                    f"immaterial but must not stay undisclosed"),
        "convention_note": ("pooled_seed_sd is the ARITHMETIC MEAN of per-corpus seed SDs, which is "
                            "e53's convention, verified by reproducing its stored 35.52x exactly. "
                            "An RMS pooling gives 29.7x / 27.4x -- the multiplier is convention-"
                            "dependent and the convention must travel with it")}


# ------------------------------------------------------------------ R4j
def r4j():
    """The single leave-two-out failure, with the magnitude the paper does not print."""
    loo = load(_in("results/e54_aggregation_audit.json"))["matrix_loo"]
    k = "drop_both_extremes_github_and_uspto"
    out = {ag: loo[ag][k] for ag in loo if k in loo[ag]}
    m = out["min"]
    return {
        "what": "the drop-both-extremes leave-two-out split, both aggregations, with magnitudes",
        "finding": "R4j, new — the paper says '27/28, the exception being...' without saying by how much",
        "drop_both_extremes_github_and_uspto": out,
        "under_min_the_read_axis_WINS_by_pp": m["read_pct"] - m["fit_pct"],
        "slate_required": {"fit_pct": 42.011, "read_pct": 47.613, "residual": 10.375,
                           "fit_dominates_read": False},
        "agrees_with_slate": (abs(m["fit_pct"] - 42.011) < 1e-2 and abs(m["read_pct"] - 47.613) < 1e-2
                              and m["fit_dominates_read"] is False),
        "VERDICT": (f"under `min`, dropping both extremes REVERSES the headline: read "
                    f"{m['read_pct']:.3f}% beats fit {m['fit_pct']:.3f}% by "
                    f"{m['read_pct'] - m['fit_pct']:.3f} percentage points. Under `persist` the "
                    f"ordering holds ({out['persist']['fit_pct']:.3f}% vs "
                    f"{out['persist']['read_pct']:.3f}%). The abstract's '28/28 leave-two-out "
                    f"splits' carries no aggregation qualifier and is `persist`-only")}


# ------------------------------------------------------------------ R4k
def r4k():
    """What 541 counts, and what the admitted five actually hold."""
    sys.path.insert(0, os.path.join(REPO, "jacobian-lens"))
    from transformers import AutoTokenizer
    from anchor_evals import EVAL_SETS, load_eval, token_ids_of
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
    tok.add_bos_token = True
    ADM = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
    per = {}
    for name in EVAL_SETS:
        rel = load_eval(name)
        sc = pr = 0
        for it in rel:
            t = [x for x in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if x]
            if t: sc += 1; pr += len(t)
        per[name] = {"released": len(rel), "scorable": sc, "pairs": pr, "admitted": name in ADM}
    tot = lambda f, sel: sum(v[f] for k, v in per.items() if sel(k))
    return {
        "what": "the item and pair counts, per set, released vs scorable vs admitted",
        "finding": "R4k, new — '541 items' is the SIX-set scorable count; only five sets are scored",
        "per_set": per,
        "all_six": {"released": tot("released", lambda k: True),
                    "scorable": tot("scorable", lambda k: True),
                    "pairs": tot("pairs", lambda k: True)},
        "admitted_five": {"released": tot("released", lambda k: k in ADM),
                          "scorable": tot("scorable", lambda k: k in ADM),
                          "pairs": tot("pairs", lambda k: k in ADM)},
        "slate_required": {"admitted_released": 449, "admitted_pairs": 801, "all_pairs": 893},
        "agrees_with_slate": (tot("released", lambda k: k in ADM) == 449
                              and tot("pairs", lambda k: k in ADM) == 801
                              and tot("pairs", lambda k: True) == 893),
        "VERDICT": ("t52_factorial.py builds `items` by looping EVAL_SETS, not ADMITTED, so "
                    "n_items = 541 is 551 released minus the 10 association items with no scorable "
                    "intermediate. The scored quantity is the admitted five, which hold 449 "
                    "released items and 801 of the 893 pairs. Calling 541 'the readout task' "
                    "overstates coverage by one whole eval set")}


# ------------------------------------------------------------------ R4l
def r4l():
    """What 'twenty PRE-REGISTERED candidate predictors' is entitled to say."""
    reg = load(_in("results/e56_predictor_registry.json"))["registry"]
    ev = {}
    for f, cls, where in [
        ("e31_local_bakeoff_410m.json", "criterion-only",
         "the file's `prereg` field is null; docs/experiments/preregs/E31_predictor_bakeoff.md records a "
         "pre-registered CRITERION (|r|>=0.8 leave-one-corpus-out) applied uniformly, but no "
         "registration of the individual candidates"),
        ("e38_jgeometry.json", "independent-file",
         "docs/experiments/preregs/superseded/PREREG_E38_JGEOMETRY.md — an independent timestamped document"),
        ("e34a_dact_vs_floor_410m.json", "inherited-rule", "PLAN.tex section 6 (E34), quoted verbatim"),
        ("e34b_jeval_dact_410m.json", "inherited-rule", "PLAN.tex section 6 (E34), rule unchanged"),
        ("e44_dread_410m.json", "in-file-docstring", "`prereg`: 'in-file docstring, fixed before running'; no provenance block"),
        ("e45_disagreement_geometry.json", "in-file-docstring", "same; no provenance block"),
        ("e50_concept_energy.json", "in-file-docstring", "same; no provenance block"),
    ]:
        ev[f] = {"class": cls, "evidence": where}
    cand = [r for r in reg if r["kind"] == "candidate"]
    by = {}
    for r in cand:
        src = str(r["source_file"]).split(" ")[0]
        cls = next((v["class"] for k, v in ev.items() if k in src), "unclassified")
        by.setdefault(cls, []).append(r["predictor"])
    return {
        "what": "the registration evidence behind each of the twenty candidate predictors",
        "finding": "C-1 — 6 of 45 registration claims are UNVERIFIABLE against git",
        "n_candidates": len(cand),
        "evidence_by_source_file": ev,
        "candidates_by_evidence_class": {k: sorted(v) for k, v in by.items()},
        "counts_by_evidence_class": {k: len(v) for k, v in by.items()},
        "slate_required": "three rows of tab:predictors have no locatable registering text (E45, E47, E50 family)",
        "agrees_with_slate": False,
        "DISAGREEMENT_WITH_THE_SLATE": {
            "slate_says": 3,
            "recomputed": len(by.get("in-file-docstring", [])),
            "why": ("counting only rows whose registration is an in-file docstring with no "
                    "independent timestamp and no provenance block gives FIVE candidates, not "
                    "three: 2 from E44 (dread_vs_I, dread_vs_Jeval), 2 from E45, 1 from E50. "
                    "E47 contributes no candidate row to the registry at all -- its rows are the "
                    "ablation pairs, not predictors -- so the slate's 'E45, E47, E50 family' "
                    "mislabels which files are implicated. Reported, not adopted"),
            "and_it_is_larger_still": ("11 further candidates come from E31, whose results file "
                                       "carries `prereg: null`. Their CRITERION is pre-registered "
                                       "and uniformly applied, which is the defensible claim; the "
                                       "individual candidates are not individually registered")},
        "VERDICT": ("'Twenty pre-registered candidate predictors' is not supportable as written. "
                    "What IS supportable, and is the stronger sentence: twenty candidates were "
                    "tested against a criterion fixed in advance and applied uniformly -- the "
                    "|r| >= 0.8 leave-one-corpus-out rule that killed the programme's own former "
                    "headline F19 -- and none passed. The conclusion is unchanged either way")}


def main() -> int:
    parts = {"R4a_containment_by_coverage": r4a(), "R4b_table_four_arm_intervals": r4b_table(),
             "R4c_all_four_aggregations": r4c(), "R4h_numeric_corrections": r4h(),
             "R4i_1b_corrected_band": r4i(), "R4j_leave_two_out_reversal": r4j(),
             "R4k_item_counts": r4k(), "R4l_registration_evidence": r4l()}
    agree = {k: v.get("agrees_with_slate") for k, v in parts.items() if "agrees_with_slate" in v}
    for k, v in parts["R4h_numeric_corrections"].items():
        if "agrees_with_slate" in v:
            agree[f"R4h.{k}"] = v["agrees_with_slate"]
    prereg = "docs/experiments/preregs/R4_corrections.md"
    rec = {
        "experiment": "R4 — the post-review corrections that are pure recomputation",
        "prereg": prereg, "prereg_sha256": sha256_file(R(prereg)) if os.path.exists(R(prereg)) else None,
        "status": "PRE-REGISTERED",
        "decision_rule_verbatim": (
            "Grouped because each is minutes of work, each closes a named finding, and none has a "
            "losing branch. Each still gets a results file and a line in RESULTS_TAXONOMY.md; none "
            "may be reported from memory (§6.2)."),
        "standing_rule": ("a recompute that does not reproduce the slate's stated value is a "
                          "disagreement to REPORT, not a number to adopt"),
        "recomputes_not_remeasures": True,
        "agreement_with_the_slate": agree,
        "n_items_agreeing": sum(1 for v in agree.values() if v),
        "n_items_disagreeing": sum(1 for v in agree.values() if v is False),
        "DISAGREEMENTS": ["R4l — five candidates rest on in-file docstrings, not three, and E47 "
                          "contributes no predictor row at all (see R4l.DISAGREEMENT_WITH_THE_SLATE)"],
        **parts,
    }
    out = "results/r4_corrections.json"
    write_result(R(out), rec, experiment="R4", inputs=[R(p) for p in INPUTS])
    print(f"wrote {out}")
    print(f"  items agreeing with the slate : {rec['n_items_agreeing']}")
    print(f"  items disagreeing             : {rec['n_items_disagreeing']}")
    for k, v in agree.items():
        print(f"    {'OK ' if v else 'DISAGREES'}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
