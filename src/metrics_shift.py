#!/usr/bin/env python3
"""metrics_shift.py — M1..M4, the four distribution instruments S3 needs.

None of these existed before. The DIST family was empty at every scale in every results table,
which is why S3 had no measurement of any kind.

  M1  hard containment      is this window IN the published Pythia token stream?
  M2  reference-calibrated  CE_M(Q) - CE_R(Q); a reference model cancels H(Q)
  M3  domain-classifier AUC layerwise, CROSS-FITTED, with a permuted-label null
  M4  D_act                 E_{x~Q} ||(J_P - J_Q) h(x)||^2, CROSS-FITTED. The novel one.
  R   proximity             continuous, R_i = -min_{p in P} d(x_i, p)

THREE RULES BAKED IN, each from an external review finding, each of which invalidates a result if
skipped:

1. **M3 and M4 must be cross-fitted** (Prompt 8 §2B). Both are computed from hidden states.
   If the same items supply the x-axis and the outcome, the relationship is tautological:
   "a domain classifier trained on hidden states and then correlated with a hidden-state
   distance is not independent evidence". Every estimator here takes a fold assignment and
   refuses to score an item it was fitted on.
2. **M3 needs a permuted-label null and a lexical baseline** (Prompt 8 §3.4). A classifier
   reaches AUC 1.0 on an artifact of how you sampled -- a dateline, a length difference.
3. **M1 does not prove OOD.** Non-membership is not "outside the learned support". The only
   defensible phrase is "provably non-member", and M1 returns EVIDENCE, not a Boolean
   (Prompt 7 §7): match kind, similarity, and the n-gram order at which it was found.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional, Sequence

import torch

# ============================================================ M1 — containment / membership


@dataclass
class MembershipEvidence:
    """Prompt 7 §7: report membership as MORE than a Boolean."""
    exact_match: bool
    max_ngram_overlap: float          # Jaccard over n-grams against the best-matching document
    longest_common_ngram: int         # the largest n at which any n-gram is shared
    n_exposures: int                  # how many stream documents matched at threshold
    method: str
    threshold: float
    max_ngram_order: int = 13         # the index's n; lcn == this means a full-order verbatim hit

    @property
    def verdict(self) -> str:
        """FOUR tiers, not two. Prompt 7 §5 is explicit that natural / audited / constructed
        nonmembers must not be collapsed, and the same applies on the member side: a document
        sharing a long verbatim span with the stream is not a nonmember just because its overall
        Jaccard is low. Carlini et al. (2012.07805) show memorisation is context-dependent and a
        span may be extractable with the right prefix."""
        if self.exact_match:
            return "member-exact"
        if self.max_ngram_overlap >= self.threshold:
            return "member-near-duplicate"
        if self.longest_common_ngram >= self.max_ngram_order:
            return "member-partial-span"          # a full-order n-gram is shared verbatim
        if self.method != "exhaustive":
            return "no-match-found"
        return "provably-non-member"

    def to_dict(self) -> dict:
        d = asdict(self); d["verdict"] = self.verdict; return d


def _shingles(tokens: Sequence[int], n: int) -> set:
    if len(tokens) < n:
        return set()
    return {hash(tuple(tokens[i:i + n])) for i in range(len(tokens) - n + 1)}


class ContainmentIndex:
    """M1. An n-gram containment index over a token stream.

    The Pythia stream is the intended source, but the index is source-agnostic: hand it any
    iterable of token-id sequences. Exact match is by SHA-256 of the token tuple; near-duplicate
    is n-gram Jaccard, which is the MinHash estimand computed exactly (our corpora are small
    enough that exact Jaccard is cheaper than sketching, and it removes a hyperparameter).
    """

    def __init__(self, n: int = 13, threshold: float = 0.8):
        self.n, self.threshold = n, threshold
        self._exact: set[str] = set()
        self._docs: list[set] = []
        self._raw: list[tuple] = []
        self._n_docs = 0

    @staticmethod
    def _key(tokens: Sequence[int]) -> str:
        return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()

    def add(self, tokens: Sequence[int]) -> None:
        self._exact.add(self._key(tokens))
        self._docs.append(_shingles(tokens, self.n))
        self._raw.append(tuple(tokens))
        self._n_docs += 1

    def add_many(self, docs: Iterable[Sequence[int]]) -> None:
        for d in docs:
            self.add(d)

    def _longest_common_ngram(self, tokens: Sequence[int]) -> int:
        """Largest n <= self.n such that some n-gram of `tokens` occurs in some stored doc.

        A sharper membership signal than Jaccard: a short document can share a long verbatim
        span with the stream while having low overall overlap, and that span is what
        memorisation work (Carlini et al. 2012.07805) shows is extractable.
        """
        for n in range(min(self.n, len(tokens)), 2, -1):
            q = _shingles(tokens, n)
            if not q:
                continue
            for raw in self._raw:
                if q & _shingles(raw, n):
                    return n
        return 0

    def query(self, tokens: Sequence[int]) -> MembershipEvidence:
        exact = self._key(tokens) in self._exact
        q = _shingles(tokens, self.n)
        best, hits = 0.0, 0
        for s_doc in self._docs:
            inter = len(q & s_doc)
            if inter == 0:
                continue
            j = inter / max(len(q | s_doc), 1)
            best = max(best, j)
            if j >= self.threshold:
                hits += 1
        return MembershipEvidence(exact_match=exact, max_ngram_overlap=best,
                                  longest_common_ngram=self._longest_common_ngram(tokens),
                                  n_exposures=hits, method="exhaustive",
                                  threshold=self.threshold, max_ngram_order=self.n)

    def __len__(self) -> int:
        return self._n_docs


# ============================================================ M2 — reference-calibrated loss


@torch.no_grad()
def cross_entropy_per_doc(model, tokenizer, texts: Sequence[str], *, device="cpu",
                          max_length: int = 128) -> torch.Tensor:
    """Mean token CE per document, [N]. Used by M2 for both the model and the reference."""
    out = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=max_length).input_ids.to(device)
        if ids.shape[1] < 2:
            out.append(float("nan")); continue
        logits = model(ids).logits[0, :-1].float()
        tgt = ids[0, 1:]
        out.append(float(torch.nn.functional.cross_entropy(logits, tgt)))
    return torch.tensor(out)


def m2_reference_calibrated(ce_model: torch.Tensor, ce_reference: torch.Tensor) -> dict:
    """M2 = CE_M(Q) - CE_R(Q). The reference cancels H(Q), which is why raw perplexity is not
    usable: a hard document has high CE under every model.

    **This is NOT a distance metric.** external review B2 places it in the MEMBERSHIP-INFERENCE family
    (perplexity / reference-calibrated loss / membership inference / distribution distance /
    downstream degradation are five different things). Claim it as such.
    """
    d = (ce_model - ce_reference)
    ok = ~torch.isnan(d)
    return {"delta_mean": float(d[ok].mean()), "delta_median": float(d[ok].median()),
            "delta_per_doc": d.tolist(), "n": int(ok.sum()),
            "family": "membership-inference (NOT a distance)"}


# ============================================================ M3 — layerwise domain AUC


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUC; ties get average rank."""
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    npos = float((labels == 1).sum()); nneg = float((labels == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _fit_logreg(X: torch.Tensor, y: torch.Tensor, *, epochs=200, lr=0.1, l2=1e-3) -> torch.Tensor:
    X = torch.cat([X, torch.ones(len(X), 1)], 1)
    w = torch.zeros(X.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=epochs, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w, y.float()) \
            + l2 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach()


def m3_domain_auc(H_P: torch.Tensor, H_Q: torch.Tensor, *, folds: int = 5,
                  seed: int = 0, permuted_null: bool = True) -> dict:
    """M3 at one layer. H_P, H_Q: [N, d] pooled activations.

    CROSS-FITTED: the probe is fitted on k-1 folds and scored on the held-out fold, so no item
    contributes to both the estimator and the outcome. Prompt 8 §1C: probe accuracy on the same
    examples used to fit it "is not a shift measurement; it is partly a measure of estimator
    capacity and overfitting".

    Returns held-out AUC, the A-distance d_A = 2(1-2*eps), and a PERMUTED-LABEL null. If the
    null is not near 0.5 the pipeline is leaking and the number is void.
    """
    X = torch.cat([H_P.float(), H_Q.float()], 0)
    y = torch.cat([torch.zeros(len(H_P)), torch.ones(len(H_Q))])
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    X, y = X[perm], y[perm]
    fold = torch.arange(len(X)) % folds

    def run(labels):
        s = torch.zeros(len(X))
        for k in range(folds):
            tr, te = fold != k, fold == k
            if labels[tr].unique().numel() < 2:
                continue
            w = _fit_logreg(X[tr], labels[tr])
            s[te] = torch.cat([X[te], torch.ones(int(te.sum()), 1)], 1) @ w
        return _auc(s, labels.long())

    auc = run(y)
    out = {"auc_heldout": auc, "a_distance": 2 * (2 * abs(auc - 0.5)),
           "n_P": len(H_P), "n_Q": len(H_Q), "folds": folds, "cross_fitted": True}
    if permuted_null:
        yp = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(seed + 1))]
        out["auc_permuted_null"] = run(yp)
        out["null_ok"] = abs(out["auc_permuted_null"] - 0.5) < 0.15
    return out


def m3_lexical_baseline(texts_P: Sequence[str], texts_Q: Sequence[str]) -> dict:
    """Prompt 8 §6.2: a classifier may be using length, punctuation or boilerplate. If a
    *lexical* baseline already separates the corpora, M3's hidden-state AUC is not evidence
    about representations."""
    def feats(ts):
        return torch.tensor([[len(t), t.count(" "), t.count("."), t.count(","),
                              sum(c.isdigit() for c in t), sum(c.isupper() for c in t)]
                             for t in ts], dtype=torch.float)
    return m3_domain_auc(feats(texts_P), feats(texts_Q), permuted_null=False)


# ============================================================ M4 — action-on-data divergence


def m4_action_divergence(J_P: torch.Tensor, J_Q: torch.Tensor, H: torch.Tensor,
                         *, normalise: bool = True) -> dict:
    """M4 at one layer.  D_act = E_{x~Q} ||(J_P - J_Q) h(x)||^2.

    J_P, J_Q: [d, d]. H: [N, d] activations drawn from Q.

    WHY THIS FORM and not ||J_P - J_Q||_F: Frobenius counts disagreement in all d directions,
    including the overwhelming majority no real activation ever visits; a large difference in an
    unvisited direction costs nothing behaviourally. D_act weights the operator difference by the
    data it will act on. Frobenius and spectral distance are returned as SECONDARIES.

    ``normalise`` divides by E||J_P h||^2 so the quantity is dimensionless and comparable across
    layers and models -- without it, D_act tracks activation norm, which grows with depth.
    """
    D = (J_P.float() - J_Q.float())
    H = H.float()
    act = (H @ D.T)                                   # [N, d]
    d_act = float((act ** 2).sum(-1).mean())
    base = float(((H @ J_P.float().T) ** 2).sum(-1).mean())
    return {"d_act": d_act,
            "d_act_normalised": (d_act / base) if (normalise and base > 0) else None,
            "frobenius_secondary": float(D.norm()),
            "spectral_secondary": float(torch.linalg.matrix_norm(D, 2)),
            "baseline_energy": base, "n": len(H)}


def m4_cross_fitted(J_P: torch.Tensor, J_Q_folds: Sequence[torch.Tensor],
                    H_folds: Sequence[torch.Tensor]) -> dict:
    """M4, CROSS-FITTED by item -- the only defensible form (F-10 scope, Prompt 8 §2B).

    ``J_Q_folds[k]`` must be fitted WITHOUT the items in ``H_folds[k]``. Each fold's D_act is
    then computed on data the operator never saw, so the x-axis and the outcome are independent.
    A run that skips this produces a number we cannot defend.
    """
    if len(J_Q_folds) != len(H_folds):
        raise ValueError("one J_Q per held-out fold is required")
    per = [m4_action_divergence(J_P, jq, h) for jq, h in zip(J_Q_folds, H_folds)]
    vals = [p["d_act"] for p in per]
    nrm = [p["d_act_normalised"] for p in per if p["d_act_normalised"] is not None]
    t = torch.tensor(vals)
    return {"d_act_mean": float(t.mean()), "d_act_std": float(t.std(unbiased=len(t) > 1)),
            "d_act_normalised_mean": (float(torch.tensor(nrm).mean()) if nrm else None),
            "per_fold": per, "folds": len(per), "cross_fitted": True}


# ============================================================ proximity (continuous)


def proximity_to_P(E_Q: torch.Tensor, E_P: torch.Tensor, *, metric: str = "cosine") -> dict:
    """R_i = -min_{p in P} d(x_i, p), Prompt 7 §6.

    **This is what unblocks arm A_3.** We had recorded that P subset Pile forces IJD subset ID
    and leaves the "member + far" cell empty by construction. That is only true if proximity is
    defined against the TRAINING distribution. Defined against *P*, any Pile component unlike
    WikiText is a training member that is far from P, and the cell fills.

    Returns a continuous score per Q item, so shift is a smooth covariate rather than a label.
    """
    A = torch.nn.functional.normalize(E_Q.float(), dim=-1) if metric == "cosine" else E_Q.float()
    B = torch.nn.functional.normalize(E_P.float(), dim=-1) if metric == "cosine" else E_P.float()
    if metric == "cosine":
        d = 1.0 - (A @ B.T)                       # [NQ, NP]
    else:
        d = torch.cdist(A, B)
    dmin = d.min(dim=1).values
    return {"proximity": (-dmin).tolist(), "nn_distance": dmin.tolist(),
            "metric": metric, "n_Q": len(E_Q), "n_P": len(E_P)}


def stratify(scores: Sequence[float], n_bins: int = 3,
             labels: Sequence[str] = ("far", "intermediate", "near")) -> list[str]:
    """Equal-count bins over a continuous score -> the ordered strata Prompt 7 §3 asks for."""
    t = torch.tensor(list(scores), dtype=torch.float)
    qs = torch.quantile(t, torch.linspace(0, 1, n_bins + 1)[1:-1]) if n_bins > 1 \
        else torch.tensor([])
    idx = torch.bucketize(t, qs)
    return [labels[min(int(i), len(labels) - 1)] for i in idx]
