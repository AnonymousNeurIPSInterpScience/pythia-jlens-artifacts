#!/usr/bin/env python3
"""test_corpora.py — the P-ladder and factorial-Q builders."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from corpora import (Document, QItem, build_P, build_Q_factorial, mixture_quota,  # noqa: E402
                     neyman_quota, standardised_mean_difference, balance_report,
                     PILE_WEIGHTS)

_p = _f = 0
def check(name, cond, detail=""):
    global _p, _f
    if cond: _p += 1; print(f"  ok  {name}   [{detail}]")
    else:    _f += 1; print(f"  FAIL {name}   [{detail}]")


class FakeTok:
    """Whitespace tokenizer -- enough to exercise the windowing logic on CPU."""
    def __call__(self, text, return_tensors=None):
        return {"input_ids": text.split()}
    def decode(self, ids):
        return " ".join(ids)


tok = FakeTok()
docs = [Document(doc_id=f"d{i}", text=" ".join(f"w{i}_{j}" for j in range(500)),
                 component=("Github" if i % 3 == 0 else "Pile-CC"))
        for i in range(40)]

# ---- P builder: recorded seed, document-level sampling, manifest completeness
texts, man = build_P(docs, tok, kind="P_component", name="test", seed=7,
                     n_windows=20, window_tokens=32)
check("P_builder_hits_the_window_budget", len(texts) == 20, f"{len(texts)} windows")
check("P_builder_records_the_seed", man.seed == 7 and man.shuffled,
      "D16/F7: every future fit shuffles with a RECORDED seed")
check("P_builder_counts_unique_documents",
      man.n_unique_documents == man.n_documents == 20,
      f"{man.n_unique_documents} unique docs for {man.n_windows} windows — "
      "one window per doc maximises independent samples")
check("P_manifest_has_component_proportions",
      abs(sum(man.component_proportions.values()) - 1.0) < 1e-6,
      f"{man.component_proportions}")
check("P_manifest_carries_provenance_hashes", len(man.doc_sha256_head) == 8,
      "spot-checkable document identity")

# ---- the seed actually determines the draw (else "recorded seed" is decorative)
t7a, _ = build_P(docs, tok, kind="P_wiki", name="a", seed=7, n_windows=10, window_tokens=32)
t7b, _ = build_P(docs, tok, kind="P_wiki", name="b", seed=7, n_windows=10, window_tokens=32)
t9, _ = build_P(docs, tok, kind="P_wiki", name="c", seed=9, n_windows=10, window_tokens=32)
check("P_builder_is_seed_reproducible", t7a == t7b, "same seed -> identical corpus")
check("P_builder_seed_changes_the_draw", t7a != t9, "different seed -> different corpus")

# ---- non-overlapping windows: no token appears in two windows of the same doc
one = [Document(doc_id="x", text=" ".join(f"t{j}" for j in range(200)))]
w, _ = build_P(one, tok, kind="P_wiki", name="w", seed=0, n_windows=5,
               window_tokens=20, windows_per_document=5)
toks = [t for win in w for t in win.split()]
check("windows_are_non_overlapping", len(toks) == len(set(toks)),
      f"{len(toks)} tokens, {len(set(toks))} unique — overlapping windows would inflate n")

# ---- mixture quota reproduces the Pile weights
q = mixture_quota(1000)
check("mixture_quota_sums_to_budget", sum(q.values()) == 1000, f"{sum(q.values())}")
check("mixture_quota_tracks_pile_weights",
      abs(q["Pile-CC"] / 1000 - PILE_WEIGHTS["Pile-CC"] / sum(PILE_WEIGHTS.values())) < 0.01,
      f"Pile-CC gets {q['Pile-CC']}/1000 vs weight {PILE_WEIGHTS['Pile-CC']:.4f}")
check("neyman_oversamples_high_dispersion_components",
      neyman_quota(1000, {"Github": 5.0, "Pile-CC": 1.0})["Github"]
      > mixture_quota(1000)["Github"],
      "dispersion varies by corpus by >=2x AT MATCHED WINDOW LENGTH (the 'narrow corpora have "
      "2-5x' reading was RETRACTED 2026-08-12 -- length confound), so variance-optimal "
      "allocation oversamples the high-dispersion components whatever orders them")

# ---- factorial Q: the empty-cell report is the point
items = []
for m in ("member", "nonmember"):
    for p in ("near", "intermediate", "far"):
        for d in ("easy", "medium", "hard"):
            items.append(QItem(item_id=f"{m}-{p}-{d}", text="x",
                               membership=("member-exact" if m == "member"
                                           else "provably-non-member"),
                               nonmember_tier=(None if m == "member" else "audited"),
                               proximity_score=0.0, proximity_stratum=p,
                               difficulty=0.5, difficulty_stratum=d))
full = build_Q_factorial(items)
check("full_factorial_is_detected", full["full_factorial"] and full["n_empty"] == 0,
      f"{full['n_items']} items, {len(full['cells'])} cells, 0 empty")
check("three_way_interaction_declared_estimable", full["three_way_interaction_estimable"],
      "Prompt 7 §6: only claimable when every cell is populated")

# THE case that matters: drop member+far and confirm it is reported, not silently absorbed
partial = build_Q_factorial([i for i in items if not i.item_id.startswith("member-far")])
check("empty_cells_are_reported_not_hidden", partial["n_empty"] == 3
      and all(c.startswith("member|far") for c in partial["empty_cells"]),
      f"empty: {partial['empty_cells']}")
check("interaction_declared_NOT_estimable_when_a_cell_is_empty",
      not partial["three_way_interaction_estimable"],
      "this is the guard against claiming an interaction we cannot identify")
check("A3_note_is_carried", "proximity is measured against P" in partial["note"],
      "the member+far cell is populable — the old 'empty by construction' was a "
      "definition artifact")
check("nonmember_tiers_are_counted", partial["nonmember_tiers"].get("audited", 0) > 0,
      f"{partial['nonmember_tiers']} — natural/audited/constructed are never collapsed")

# ---- balance
smd = standardised_mean_difference([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
check("SMD_is_zero_for_identical_covariates", abs(smd) < 1e-9, f"SMD = {smd:.6f}")
rep = balance_report({"length": [1, 2, 3, 4, 5], "digits": [0, 0, 0, 0, 0]},
                     {"length": [10, 11, 12, 13, 14], "digits": [0, 0, 0, 0, 0]})
check("balance_report_flags_an_unbalanced_covariate",
      not rep["balanced"]["length"] and rep["n_unbalanced"] == 1,
      f"length SMD = {rep['smd']['length']:.2f}, |SMD| >= 0.1 fails")

print(f"\n=== {_p}/{_p+_f} PASSED ===")
raise SystemExit(1 if _f else 0)
