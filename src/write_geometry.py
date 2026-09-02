#!/usr/bin/env python3
"""write_geometry.py — the conditioning-confound controls for cross-lens writes.

WHY THIS EXISTS. external review Prompt 9 (B9) establishes that our coordinate-swap write is
**structurally biased toward whichever lens defines the basis**. The swap uses c = V†h, and V†
depends on (VᵀV)⁻¹, so the edit magnitude is governed by the Gram geometry of the two concept
directions -- not only by their semantic quality. Since the J-lens induces the metric JJᵀ on
output-space directions, it can produce systematically better-conditioned coordinate frames and
therefore win the benchmark **by construction**.

Prompt 9's verdict, implemented here verbatim:
  - unit normalisation is MANDATORY but NOT SUFFICIENT;
  - **equal achieved dose is the primary cross-lens comparison**;
  - orthogonalisation is a CONTROL, not a neutral normalisation -- it changes the semantics,
    so the discarded component must be reported;
  - rank-1 normalised contrast is fairer for directional efficacy, weaker for coordinate
    separability -- run both;
  - lower |rho| for the J-lens is BOTH a finding about the operator AND a confound for the
    native benchmark. Report raw, then matched. Do not erase it; do not let it stand unadjusted.

DISCIPLINE NOTE (repo rule 1, "no centering, ever"). mu never enters a lens vector or a write,
and nothing here changes that. Sigma is used ONLY to define a *norm on Delta h* for dose
reporting. Delta h is already a difference, so centering cannot affect the edit itself. We
default to the UNCENTERED second moment E[hhᵀ] so that no mean is ever formed; pass
``center=True`` to use the ordinary covariance instead, which is what LEACE assumes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import torch

EPS = 1e-12


# --------------------------------------------------------------------------- basic geometry
def unit(v: torch.Tensor) -> torch.Tensor:
    """v / ||v||. Mandatory preprocessing (Prompt 9 §2A): without it the comparison is
    contaminated by an arbitrary scale, since v -> a·v changes the coordinates c = V†h even
    though the span is unchanged."""
    v = v.float()
    return v / v.norm().clamp_min(EPS)


def gram_stats(v_s: torch.Tensor, v_t: torch.Tensor) -> tuple[float, float, float]:
    """Return (rho, kappa_V, kappa_G) for the UNIT-NORMALISED pair.

    rho      = <v_s, v_t>, the cosine similarity, after unit normalisation.
    kappa_V  = sqrt((1+|rho|)/(1-|rho|))   -- spectral condition number of V
    kappa_G  = (1+|rho|)/(1-|rho|)         -- of the Gram matrix VᵀV

    These are Prompt 9 §1's closed forms and they are exact for unit columns.
    """
    a, b = unit(v_s), unit(v_t)
    rho = float(a @ b)
    r = min(abs(rho), 1.0 - 1e-9)
    kappa_G = (1.0 + r) / (1.0 - r)
    return rho, kappa_G ** 0.5, kappa_G


# --------------------------------------------------------------------------- the dose metric
class DoseMetric:
    """A norm on Delta h, used to equalise intervention magnitude across lenses.

    ``kind='euclidean'``  ||d||_2                       -- always available
    ``kind='whitened'``   sqrt(dᵀ S⁺ d), S the second moment (or covariance if center=True)

    Prompt 9 §2C recommends reporting BOTH: the whitened norm is more meaningful when
    activation dimensions have very different variances, which is exactly the residual-stream
    situation.
    """

    def __init__(self, kind: str = "euclidean", S_inv: Optional[torch.Tensor] = None,
                 *, center: bool = False, shrinkage: float = 0.0):
        if kind not in ("euclidean", "whitened"):
            raise ValueError(f"unknown dose metric {kind!r}")
        if kind == "whitened" and S_inv is None:
            raise ValueError("kind='whitened' requires S_inv from fit_second_moment")
        self.kind, self.S_inv, self.center, self.shrinkage = kind, S_inv, center, shrinkage

    def __call__(self, d: torch.Tensor) -> torch.Tensor:
        """||d|| under this metric. d: [..., dim] -> [...]"""
        d = d.float()
        if self.kind == "euclidean":
            return d.norm(dim=-1)
        q = torch.einsum("...i,ij,...j->...", d, self.S_inv.to(d.device).float(), d)
        return q.clamp_min(0.0).sqrt()

    def describe(self) -> dict:
        return {"kind": self.kind, "center": self.center, "shrinkage": self.shrinkage}


def fit_second_moment(H: torch.Tensor, *, center: bool = False,
                      shrinkage: float = 1e-3) -> torch.Tensor:
    """S⁺ for the whitened dose metric, from stacked activations H [N, d].

    Returns the (shrunk) inverse. Shrinkage is REQUIRED in practice: the residual stream is
    low-rank relative to d at realistic N, so the raw inverse is numerically meaningless.
    S_shrunk = (1-a)S + a·(tr(S)/d)·I, the standard Ledoit-Wolf-style target.

    center=False (default) uses E[hhᵀ] and forms NO mean anywhere -- see the module docstring.
    """
    H = H.float().reshape(-1, H.shape[-1])
    if center:
        H = H - H.mean(dim=0, keepdim=True)
    d = H.shape[1]
    S = (H.T @ H) / max(H.shape[0], 1)
    if shrinkage > 0:
        S = (1 - shrinkage) * S + shrinkage * (torch.diagonal(S).mean()) * torch.eye(d)
    return torch.linalg.pinv(S)


def equal_dose(delta: torch.Tensor, target: float, metric: DoseMetric) -> torch.Tensor:
    """Rescale so that ||delta||_M == target, per Prompt 9 §2C.

    This is THE primary cross-lens comparison. It changes the question from "how effective is
    the full lens-defined coordinate operation" to "how effective is the semantic direction once
    intervention magnitude is controlled", and that distinction must be stated in any caption.
    """
    n = metric(delta)
    scale = target / n.clamp_min(EPS)
    return delta * scale.unsqueeze(-1)


# --------------------------------------------------------------------------- the three writes
@dataclass
class GeometryRecord:
    """Every quantity Prompt 9's recommended table asks for, for one (lens, pair, layer)."""
    lens: str
    norm_v_s: float
    norm_v_t: float
    rho: float                    # cosine similarity AFTER unit normalisation
    kappa_V: float
    kappa_G: float
    discarded_frac: float         # ||<v_t,v_s>v_s|| / ||v_t|| == |rho|, the Gram-Schmidt loss
    dose_euclidean: float
    dose_whitened: Optional[float]
    degenerate: bool

    def to_dict(self) -> dict:
        return asdict(self)


def swap_delta_normalized(h: torch.Tensor, v_s: torch.Tensor, v_t: torch.Tensor,
                          alpha: float = 1.0) -> tuple[torch.Tensor, dict]:
    """ARM 1 -- native two-coordinate swap on UNIT-normalised directions.

    Δ = α·V(σ(c) − c) with c = V†h. This is the anchor's operator with the mandatory
    normalisation applied, and it is the arm that carries the conditioning confound.
    """
    a, b = unit(v_s), unit(v_t)
    V = torch.stack([a, b], dim=1)                        # [d, 2]
    G = V.T @ V
    ev = torch.linalg.eigvalsh(G)
    lo, hi = float(ev.min().clamp_min(0.0)), float(ev.max())
    degenerate = lo <= 1e-8
    Vpinv = torch.linalg.pinv(V) if degenerate else torch.linalg.inv(G) @ V.T
    c = h.float() @ Vpinv.T                               # [..., 2]
    c_sw = torch.stack([c[..., 1], c[..., 0]], dim=-1)
    delta = alpha * ((c_sw - c) @ V.T)
    rho = float(a @ b)
    return delta, {"rho": rho, "kappa_G": (hi / lo) if lo > 0 else float("inf"),
                   "degenerate": degenerate}


def swap_delta_orthogonalized(h: torch.Tensor, v_s: torch.Tensor, v_t: torch.Tensor,
                              alpha: float = 1.0) -> tuple[torch.Tensor, dict]:
    """ARM 2 (CONTROL) -- Gram-Schmidt the basis first, then swap.

    Prompt 9 §2B: this is NOT a neutral normalisation. After orthogonalisation the second
    direction is no longer "the target concept" but "the component of the target not already
    represented by the source". We therefore report the DISCARDED component ||<v_t,v_s>v_s|| /
    ||v_t|| = |rho| alongside, and never present this arm as the headline.
    """
    a = unit(v_s)
    rho = float(a @ unit(v_t))
    b_raw = unit(v_t) - rho * a
    nb = float(b_raw.norm())
    if nb < 1e-8:                       # fully collinear: the orthogonal complement is empty
        return torch.zeros_like(h.float()), {"rho": rho, "discarded_frac": abs(rho),
                                             "degenerate": True}
    b = b_raw / nb
    V = torch.stack([a, b], dim=1)
    c = h.float() @ V                   # orthonormal => V† == Vᵀ
    c_sw = torch.stack([c[..., 1], c[..., 0]], dim=-1)
    delta = alpha * ((c_sw - c) @ V.T)
    return delta, {"rho": rho, "discarded_frac": abs(rho), "degenerate": False}


def rank1_contrast_delta(h: torch.Tensor, v_s: torch.Tensor, v_t: torch.Tensor,
                         alpha: float = 1.0,
                         metric: Optional[DoseMetric] = None) -> tuple[torch.Tensor, dict]:
    """ARM 3 (CONTROL) -- normalised rank-1 contrast, h -> h + α·u, u = (v_t−v_s)/||v_t−v_s||_M.

    Prompt 9 §3: this needs NO pseudoinverse, so it is free of the conditioning confound. It is
    a *different causal question* -- "does moving along this contrast produce the intended
    change" rather than "are the two coordinates separably recoverable" -- and it is the
    decisive diagnostic:

        J-lens wins the swap but NOT the normalised rank-1 arm
            => the advantage is coordinate conditioning, not semantics.

    Note the delta does not depend on h; it is returned broadcast so every arm has one shape.
    """
    u_raw = unit(v_t) - unit(v_s)
    m = metric or DoseMetric("euclidean")
    n = float(m(u_raw))
    if n < 1e-8:
        return torch.zeros_like(h.float()), {"degenerate": True, "contrast_norm": n}
    u = u_raw / n
    delta = (alpha * u).expand_as(h.float()).clone()
    return delta, {"degenerate": False, "contrast_norm": n}


# --------------------------------------------------------------------------- ablation geometry
def ablation_energy(v: torch.Tensor, S: torch.Tensor) -> dict:
    """Prompt 9 §5 -- ablation-KL is fairer than the swap but is NOT lens-agnostic.

    For h ~ (0, S) and the Euclidean projection ablation h - (vᵀh/vᵀv)v:

        removed energy       E||h - h_abl||²  = vᵀSv / (vᵀv)        <- SCALE-INVARIANT
        coordinate variance  Var(vᵀh/vᵀv)     = vᵀSv / (vᵀv)²       <- scales as 1/a²

    **CORRECTION TO external review Prompt 9 §5, verified numerically.** Prompt 9 gives the removed
    energy as vᵀS²v/(vᵀv)². That is wrong. Since h - h_abl = (vᵀh/vᵀv)v,

        ||h - h_abl||² = (vᵀh)²/(vᵀv)² · ||v||² = (vᵀh)²/(vᵀv),

    so the expectation is vᵀSv/(vᵀv). Monte Carlo at d=32, N=4e5 (see the test): empirical
    0.68905 / 3.81529 / 0.64689 for a random, top-eigenvector and rescaled direction, against
    our formula 0.68892 / 3.81515 / 0.64676 and Prompt 9's 0.03192 / 14.55556 / 0.00162.
    Prompt 9's *coordinate variance* formula is correct and is kept verbatim.
    The projection ablation is invariant to ||v||, so the removed energy must be too --
    Prompt 9's version is not, which is the tell.

    **Prompt 9's substantive conclusion is unaffected and is in fact sharper under the correct
    formula:** two UNIT directions still remove very different amounts of activation energy
    (top-eigenvector 3.82 vs random 0.69, a 5.5x gap), so a lens whose directions align with
    high-variance activation modes gets a larger ablation KL for free. Report these alongside
    every ablation-KL and compare KL at MATCHED removed energy.
    """
    v = v.float()
    S = S.float().to(v.device)
    vtv = float(v @ v)
    Sv = S @ v
    return {"removed_energy": float(v @ Sv) / vtv,             # scale-invariant, corrected
            "removed_variance": float(v @ Sv) / (vtv ** 2),    # Prompt 9 verbatim, correct
            "removed_energy_prompt9_erratum": float(Sv @ Sv) / (vtv ** 2),
            "v_norm": vtv ** 0.5}


def geometry_record(lens: str, v_s: torch.Tensor, v_t: torch.Tensor, delta: torch.Tensor,
                    metric_euc: DoseMetric,
                    metric_whit: Optional[DoseMetric] = None) -> GeometryRecord:
    """Assemble Prompt 9's recommended per-(lens, pair) reporting row."""
    rho, kV, kG = gram_stats(v_s, v_t)
    de = float(metric_euc(delta).mean())
    dw = float(metric_whit(delta).mean()) if metric_whit is not None else None
    return GeometryRecord(
        lens=lens,
        norm_v_s=float(v_s.float().norm()), norm_v_t=float(v_t.float().norm()),
        rho=rho, kappa_V=kV, kappa_G=kG, discarded_frac=abs(rho),
        dose_euclidean=de, dose_whitened=dw,
        degenerate=not (kG < 1e8),
    )


ARMS = {
    "swap_native": swap_delta_normalized,
    "swap_orth": swap_delta_orthogonalized,
    "rank1": rank1_contrast_delta,
}
