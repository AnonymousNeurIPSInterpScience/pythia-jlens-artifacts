#!/usr/bin/env python3
"""corpora.py — the P-ladder and the factorial Q, per external review Prompt 7.

TWO OBJECTS.

**The P-ladder** (E27). Nested fitting conditions that separate three effects we currently
conflate in one number:
    P_wiki       -> a deliberately narrow baseline. What all 14 existing lenses are.
    P_component  -> one Pile component. Isolates NARROWNESS (different support).
    P_mixture    -> mixture-matched from the actual training stream. Isolates
                    MIXTURE-REPRESENTATIVENESS (right support, right weights).
Prompt 7's decomposition is the reason this is a ladder and not a choice:
    Jhat_P - J^mix = (Jhat_P - J_P) + (J_P - J^mix)
                      sampling error     distributional bias
More data shrinks the first and CANNOT touch the second.

**The factorial Q** (E28). NOT a binary ID/OOD split. Three axes, per Prompt 7 §3:
    membership  member / nonmember          -- auditable against the token stream (M1)
    proximity   near / intermediate / far   -- CONTINUOUS, measured against *P*
    difficulty  easy / medium / hard        -- a measured covariate, never a pre-stratum
with THREE nonmember tiers that are never collapsed: natural, audited, constructed.

THE UNBLOCKING POINT. We recorded for weeks that P subset Pile forces IJD subset ID and leaves
the "member + far" cell (arm A_3) EMPTY BY CONSTRUCTION. That was wrong. It is empty only if
proximity is defined against the TRAINING distribution. Defined against *P* -- which is what
``metrics_shift.proximity_to_P`` computes -- any Pile component unlike WikiText is a training
member that is far from P. ``build_Q_factorial`` fills that cell by construction.

SAMPLING DISCIPLINE (Prompt 7 §2.3, and E18/F7). Do NOT treat token windows as independent
samples. Sample DOCUMENTS, take non-overlapping windows within a document, and record the
unique-document count alongside the window count. Every builder here shuffles with a RECORDED
seed, because E18 found our existing fit corpus is ordered (split-half disagreement runs 2-5x
odd-even) and that ordering is currently the largest term in the estimator's error budget.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, asdict, field
from typing import Optional, Sequence

# The Pile mixture weights that matter for P_mixture (Gao et al. 2101.00027, Table 1).
# Used to set per-component document quotas so the fitting corpus matches the training mixture
# rather than an arbitrary uniform draw over components.
PILE_WEIGHTS = {
    "Pile-CC": 0.1810, "PubMed Central": 0.1440, "Books3": 0.1207, "OpenWebText2": 0.1001,
    "ArXiv": 0.0896, "Github": 0.0724, "FreeLaw": 0.0621, "StackExchange": 0.0530,
    "USPTO Backgrounds": 0.0353, "PubMed Abstracts": 0.0307, "Gutenberg (PG-19)": 0.0217,
    "OpenSubtitles": 0.0155, "Wikipedia (en)": 0.0153, "DM Mathematics": 0.0124,
    "Ubuntu IRC": 0.0088, "BookCorpus2": 0.0075, "EuroParl": 0.0073, "HackerNews": 0.0062,
    "YoutubeSubtitles": 0.0060, "PhilPapers": 0.0038, "NIH ExPorter": 0.0030,
    "Enron Emails": 0.0014,
}


@dataclass
class Document:
    """One source document. The unit of resampling -- NOT the window."""
    doc_id: str
    text: str
    component: Optional[str] = None
    date: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


@dataclass
class CorpusManifest:
    """Prompt 7 §7's corpus-level reporting block. Written beside every fitted lens."""
    name: str
    kind: str                       # "P_wiki" | "P_component" | "P_mixture" | "Q"
    n_documents: int
    n_windows: int
    n_unique_documents: int
    component_proportions: dict
    seed: int
    shuffled: bool
    window_tokens: int
    windows_per_document: int
    doc_sha256_head: list           # first 8 document hashes, for spot provenance
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)


def _windows(text: str, tokenizer, window_tokens: int, per_doc: int) -> list[str]:
    """Non-overlapping windows from ONE document (Prompt 7 §2.3 step 2).

    Overlapping windows inflate the apparent sample size without adding information, which is
    exactly the effective-sample-size failure the sample-size law F-8 assumes away.
    """
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    out = []
    for i in range(0, len(ids) - window_tokens + 1, window_tokens):
        out.append(tokenizer.decode(ids[i:i + window_tokens]))
        if len(out) >= per_doc:
            break
    return out


def build_P(docs: Sequence[Document], tokenizer, *, kind: str, name: str, seed: int = 0,
            n_windows: int = 200, window_tokens: int = 128,
            windows_per_document: int = 1) -> tuple[list[str], CorpusManifest]:
    """Build one rung of the P-ladder with document-level sampling and a RECORDED seed.

    ``windows_per_document=1`` by default: one window per document maximises the number of
    independent documents at fixed window budget, which is what the i.i.d. assumption in F-8
    actually needs.
    """
    rng = random.Random(seed)
    pool = list(docs)
    rng.shuffle(pool)                      # RECORDED-seed shuffle -- see D16 / F7
    texts, used = [], []
    for d in pool:
        ws = _windows(d.text, tokenizer, window_tokens, windows_per_document)
        if not ws:
            continue
        texts.extend(ws); used.append(d)
        if len(texts) >= n_windows:
            break
    texts = texts[:n_windows]
    comps: dict = {}
    for d in used:
        comps[d.component or "unknown"] = comps.get(d.component or "unknown", 0) + 1
    tot = max(sum(comps.values()), 1)
    man = CorpusManifest(
        name=name, kind=kind, n_documents=len(used), n_windows=len(texts),
        n_unique_documents=len({d.doc_id for d in used}),
        component_proportions={k: round(v / tot, 4) for k, v in sorted(comps.items())},
        seed=seed, shuffled=True, window_tokens=window_tokens,
        windows_per_document=windows_per_document,
        doc_sha256_head=[d.sha256[:16] for d in used[:8]],
    )
    return texts, man


def mixture_quota(n_windows: int, weights: dict = PILE_WEIGHTS) -> dict:
    """Per-component window counts matching the Pile mixture -- PROPORTIONATE allocation.

    Prompt 7 §2.4 also describes NEYMAN allocation, n_c proportional to w_c * sigma_c, which
    minimises estimator variance at fixed budget. Use proportionate for a faithful
    sample-cost comparison and Neyman for minimum variance; report both. Neyman needs a pilot
    estimate of per-component Jacobian dispersion, which is ``neyman_quota`` below.
    """
    tot = sum(weights.values())
    raw = {k: n_windows * w / tot for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True):
        if sum(out.values()) >= n_windows:
            break
        out[k] += 1
    return {k: v for k, v in out.items() if v > 0}


def neyman_quota(n_windows: int, sigma_by_component: dict,
                 weights: dict = PILE_WEIGHTS) -> dict:
    """n_c proportional to w_c * sigma_c (Prompt 7 §2.4). Needs a pilot dispersion per component.

    RATIONALE CORRECTED 2026-08-12. This previously read "narrow corpora had 2-5x the dispersion
    of wikitext", which is NOT supported: that ordering is confounded with prompt length
    (Spearman(median chars, ratio) = -1.000 over the three task corpora), and pile -- the
    BROADEST corpus, length-matched -- also carries ~2x wikitext's dispersion (0.252 vs 0.126).
    The defensible statement is narrower and still motivates Neyman: DISPERSION VARIES BY CORPUS
    BY AT LEAST 2x AT MATCHED WINDOW LENGTH, and WikiText is the low-dispersion outlier. Whatever
    orders it, a variance-optimal fit should oversample the high-dispersion components rather
    than matching the mixture exactly.
    """
    sc = {k: weights.get(k, 0.0) * sigma_by_component.get(k, 0.0) for k in weights}
    tot = sum(sc.values())
    if tot <= 0:
        return mixture_quota(n_windows, weights)
    out = {k: int(n_windows * v / tot) for k, v in sc.items()}
    return {k: v for k, v in out.items() if v > 0}


@dataclass
class QItem:
    """One factorial-Q item, carrying the three axes plus its membership EVIDENCE."""
    item_id: str
    text: str
    membership: str           # member-exact | member-near-duplicate | member-partial-span |
                              # provably-non-member
    nonmember_tier: Optional[str]   # natural | audited | constructed  (None if member)
    proximity_score: float          # continuous, R_i = -min_p d(x_i, p), measured against P
    proximity_stratum: str          # near | intermediate | far
    difficulty: Optional[float] = None
    difficulty_stratum: Optional[str] = None
    evidence: dict = field(default_factory=dict)

    @property
    def cell(self) -> str:
        m = "member" if self.membership.startswith("member") else "nonmember"
        return f"{m}|{self.proximity_stratum}|{self.difficulty_stratum or 'unbinned'}"

    def to_dict(self) -> dict:
        d = asdict(self); d["cell"] = self.cell; return d


def build_Q_factorial(items: Sequence[QItem]) -> dict:
    """Assemble the 2 x 3 x 3 design and REPORT WHICH CELLS ARE EMPTY.

    Prompt 7 §6: use a fractional factorial only as a fallback, and explicitly report which
    interactions are not estimable. "Do not claim to identify a three-way interaction if one
    cell is absent or nearly deterministic." This function makes that impossible to skip --
    it returns the empty cells as data.
    """
    cells: dict = {}
    for it in items:
        cells.setdefault(it.cell, []).append(it.item_id)
    want = [f"{m}|{p}|{d}"
            for m in ("member", "nonmember")
            for p in ("near", "intermediate", "far")
            for d in ("easy", "medium", "hard")]
    empty = [c for c in want if c not in cells]
    tiers: dict = {}
    for it in items:
        if it.nonmember_tier:
            tiers[it.nonmember_tier] = tiers.get(it.nonmember_tier, 0) + 1
    return {
        "n_items": len(items),
        "cells": {k: len(v) for k, v in sorted(cells.items())},
        "empty_cells": empty,
        "n_empty": len(empty),
        "full_factorial": not empty,
        "nonmember_tiers": tiers,
        "three_way_interaction_estimable": not empty,
        "note": ("A_3 (member + far) is populable when proximity is measured against P rather "
                 "than against the training distribution; see metrics_shift.proximity_to_P."),
    }


def standardised_mean_difference(a: Sequence[float], b: Sequence[float]) -> float:
    """SMD = (mean_a - mean_b) / sqrt((s_a^2 + s_b^2)/2), Prompt 7 §7.

    Report for EVERY matching covariate, not just topic and length. |SMD| < 0.1 is the
    conventional balance threshold.
    """
    import statistics as st
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = st.variance(a), st.variance(b)
    den = ((va + vb) / 2) ** 0.5
    return (st.mean(a) - st.mean(b)) / den if den > 0 else float("nan")


def balance_report(cov_A: dict, cov_B: dict) -> dict:
    """SMD per covariate + a pass/fail on the 0.1 threshold. Freeze this BEFORE outcomes."""
    out = {k: standardised_mean_difference(cov_A[k], cov_B.get(k, []))
           for k in cov_A}
    return {"smd": out,
            "balanced": {k: (abs(v) < 0.1 if v == v else False) for k, v in out.items()},
            "n_unbalanced": sum(1 for v in out.values() if v == v and abs(v) >= 0.1)}
