#!/usr/bin/env python3
"""joperator.py — the Gurnee et al. §2.5 J-lens write, implemented exactly, with dose accounting.

THE RULE (transformer-circuits.pub/2026/workspace, §2.5, verbatim):

    "given a source token s and target token t, we form V = [v_s  v_t], read the lens
     coordinates c = V†h (where V† is the pseudoinverse of V), and set
     h_patched = h + V(σ(c) − c), where σ swaps the two entries of c (optionally scaled
     by a factor α). The component of h orthogonal to span{v_s, v_t} is unchanged."

σ is the SWAP PERMUTATION. It is not a softmax. Expanding at α = 1:

    σ(c) − c        = (c_t − c_s, c_s − c_t)
    V(σ(c) − c)     = (c_t − c_s)·v_s + (c_s − c_t)·v_t
                    = (c_s − c_t)·(v_t − v_s)                                        (RANK ONE)

so the write is a rank-1 edit along (v_t − v_s) whose scalar is the *contrastive, deconfounded*
coordinate difference. For unit vectors with ρ = ⟨v_s,v_t⟩ this is (a − b)/(1 − ρ) where
a = ⟨v_s,h⟩, b = ⟨v_t,h⟩ — the 1/(1−ρ) factor is an automatic gain that grows as the two concept
directions become collinear. A raw-dot-product operator (h + α⟨v_s,h⟩(v_t − v_s)) writes the SAME
direction with no such gain; that is the entire difference between the two operator families.

α scales the WHOLE update: Δ(α) = α·(c_s − c_t)(v_t − v_s), exactly rank-1 at every α, and ‖Δ‖
linear in α FOR A SINGLE APPLICATION.

⚠ CORRECTION 2026-08-10 (second adversarial review). An earlier revision of this docstring
claimed the α-explosion under a band clamp "was a bug, not a property of the operator." **That
was wrong.** In the (s,t) coordinates the write maps the gap c_s − c_t → (1 − 2α)(c_s − c_t).
Clamping at every band layer ITERATES that map, so the injected dose is geometric in
|1 − 2α|^depth. Over the 14-layer band L9–22:

    α=0.5  |1−2α|=0  → the gap collapses to 0 (an ablation, not a partial dose)
    α=1.0  |1−2α|=1  → norm-preserving: **α=1 is the unique fixed point**
    α=2.0  |1−2α|=3  → 3^14 ≈ 5e6
    α=4.0  |1−2α|=7  → 7^14 ≈ 7e11

Measured in `t8_probe_swap_410m.json`: 22.84 → 555,072 from α=1 to α=2 (24,299×). No placement
of α fixes this; it is inherent to iterating a coordinate swap at α≠1. **Sweep the MEASURED ‖Δ‖
via `match_delta_norm`, never α**, and treat any α≠1 arm as off-distribution unless a coherence
ceiling passes it.

DOSE ACCOUNTING IS MANDATORY (PLAN.md §3.7). Every hook records ‖Δ‖ per layer and per position.
An arm without a recorded ‖Δ‖ is not analysable. Controls are matched on the MEASURED ‖Δ‖, not on
‖v_t − v_s‖ — because for this operator the coefficient depends on V†, so re-pointing v_t changes
the dose as well as the direction.

NO CENTERING. μ never appears. The anchor does not centre, and neither does any public
implementation (docs/audit/CENTERING_PROVENANCE.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Gram matrices with a 2-norm condition number above this are flagged: v_s and v_t are so
# collinear that the coordinate solve is numerically unreliable.
COND_WARN = 1e6


def token_direction(J: torch.Tensor, WU_row: torch.Tensor) -> torch.Tensor:
    """v_t = J_lᵀ W_U[t], un-normalised, un-centred, fp32.

    J: [d, d] for one source layer. WU_row: [d]. Returns [d].
    """
    return J.T.float() @ WU_row.float()


@dataclass
class SwapBasis:
    """V = [v_s v_t] and its pseudoinverse, for one layer and one (s, t) pair."""

    V: torch.Tensor          # [d, 2] — columns are v_s, v_t
    Vpinv: torch.Tensor      # [2, d] — V† = (VᵀV)⁻¹Vᵀ
    cond: float              # 2-norm condition number of the Gram matrix VᵀV
    payload: torch.Tensor    # [d] — (v_t − v_s), the rank-1 write direction at α=1
    degenerate: bool         # Gram is numerically singular

    def to(self, device: torch.device) -> SwapBasis:
        if self.V.device == device:
            return self
        return SwapBasis(self.V.to(device), self.Vpinv.to(device), self.cond,
                         self.payload.to(device), self.degenerate)


def make_swap_basis(v_s: torch.Tensor, v_t: torch.Tensor) -> SwapBasis:
    """Build V and V† for a source/target direction pair.

    Uses the explicit 2x2 Gram inverse (V† = (VᵀV)⁻¹Vᵀ), which is exactly
    ``torch.linalg.pinv(V)`` for full-column-rank V and much cheaper. Falls back to
    ``pinv`` when the Gram is numerically singular.
    """
    v_s, v_t = v_s.float(), v_t.float()
    V = torch.stack([v_s, v_t], dim=1)                       # [d, 2]
    G = V.T @ V                                              # [2, 2]
    ev = torch.linalg.eigvalsh(G)
    lo = float(ev.min().clamp_min(0.0))
    hi = float(ev.max())
    cond = float("inf") if lo <= 0 else hi / lo
    degenerate = not (cond < COND_WARN)
    if degenerate:
        Vpinv = torch.linalg.pinv(V)
    else:
        a, b, c, d_ = G[0, 0], G[0, 1], G[1, 0], G[1, 1]
        det = a * d_ - b * c
        Ginv = torch.stack([torch.stack([d_, -b]), torch.stack([-c, a])]) / det
        Vpinv = Ginv @ V.T                                   # [2, d]
    return SwapBasis(V, Vpinv, cond, v_t - v_s, degenerate)


def coords(h: torch.Tensor, basis: SwapBasis) -> torch.Tensor:
    """c = V†h — the deconfounded (least-squares) coordinates. h:[..., d] -> [..., 2]."""
    return h.float() @ basis.Vpinv.T


def swap_delta(h: torch.Tensor, basis: SwapBasis, alpha: float = 1.0) -> torch.Tensor:
    """Δ = α · V(σ(c) − c), the anchor §2.5 update. h:[..., d] -> [..., d] (fp32).

    BUGFIX 2026-08-10 (adversarial review). α previously multiplied the swapped coordinates
    *inside* σ, giving Δ(α) = (α c_t − c_s)v_s + (α c_s − c_t)v_t, which is NOT α·Δ(1) and is
    not rank-1 for α≠1. The anchor glosses α=2 as "doubling the strength with which we subtract
    the source lens vector and add in the target" — i.e. α scales the whole update. With α
    outside, Δ(α) = α(c_s − c_t)(v_t − v_s) stays exactly rank-1 at every α, and ‖Δ‖ is linear
    in α rather than exploding.
    """
    c = coords(h, basis)                                     # [..., 2]
    c_swapped = torch.stack([c[..., 1], c[..., 0]], dim=-1)  # σ: a pure permutation
    return alpha * ((c_swapped - c) @ basis.V.T)             # [..., d]


def patch_swap(
    h: torch.Tensor,
    basis: SwapBasis,
    alpha: float = 1.0,
    *,
    match_delta_norm: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the anchor write. Returns ``(h_patched, delta_norm_per_position)``.

    Args:
        h: residual stream, [..., d].
        basis: the (s, t) basis for this layer.
        alpha: optional scale on the swapped coordinates.
        match_delta_norm: if given, [...] of target ‖Δ‖ per position. Δ is rescaled
            per position to exactly this norm. This is how matched-dose CONTROLS are
            built: re-pointing v_t changes both direction and coefficient, so the only
            honest match is on the measured Δ. Positions whose native ‖Δ‖ is 0 are left
            untouched.
    """
    delta = swap_delta(h, basis, alpha)                      # [..., d]
    native = delta.norm(dim=-1)                              # [...]
    if match_delta_norm is not None:
        if tuple(match_delta_norm.shape) != tuple(native.shape):
            raise ValueError(
                f"match_delta_norm shape {tuple(match_delta_norm.shape)} != per-position "
                f"delta shape {tuple(native.shape)}; broadcasting would silently reshape the "
                f"hidden state")
        scale = torch.where(native > 0, match_delta_norm / native.clamp_min(1e-12),
                            torch.zeros_like(native))
        delta = delta * scale.unsqueeze(-1)
    return (h.float() + delta).to(h.dtype), delta.norm(dim=-1)


def final_logits(model, prompt, hooks=None, max_length=128):
    """Next-token logits at the final position — WITHOUT double-applying the final norm.

    BUGFIX 2026-08-10 (CRITICAL, adversarial review). `model.forward()` calls the bare text
    decoder, whose forward ALREADY ends with `final_layer_norm`. `model.unembed()` is documented
    as "final norm + LM head" and applies it AGAIN, so `unembed(forward(ids)[0])` computes
    lm_head(LN(LN(h))). Measured on pythia-70m: KL(true || double-normed) = 2.395 nats and
    50,234 of 50,304 ranks change. `_lm_head` alone matches hf(...).logits to 2.3e-3 (fp32 noise).

    NOTE this affects ONLY code paths that read `model.forward()`'s return value. Paths that read
    block outputs via ActivationRecorder (t3/t4/t7/anchor_evals) capture the RAW residual and must
    keep using `model.unembed()`.
    """
    import torch as _t
    ids = model.encode(prompt, max_length=max_length)
    with _t.no_grad():
        if hooks is None:
            out = model.forward(ids)
            hs = out if _t.is_tensor(out) else out[0]
            return model._lm_head(hs[0, -1].float()).float()
        with hooks:
            out = model.forward(ids)
            hs = out if _t.is_tensor(out) else out[0]
            return model._lm_head(hs[0, -1].float()).float()


def control_seed(base: int, item: str, layer: int, arm: str) -> int:
    """Independent seed per (run, item, layer, control arm).

    BUGFIX 2026-08-10 (H1). Callers previously passed `base + layer` to BOTH control
    constructors, so `control_rand_target` and `control_orth` drew the SAME random vector
    (measured cosine between their payloads: 0.998-0.9997) — two arms, one control. And because
    the seed depended on layer only, the control direction was IDENTICAL across every item
    (cosine 1.000000), giving the null zero across-item variance. `base + layer` also aliases:
    (base=1234, layer=5) == (base=1235, layer=4).
    """
    import hashlib
    h = hashlib.sha256(f"{base}|{item}|{layer}|{arm}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# --------------------------------------------------------------------------- controls

def control_rand_target(v_s: torch.Tensor, v_t: torch.Tensor, seed: int) -> torch.Tensor:
    """Re-point the target at a random direction, preserving ‖v_t − v_s‖.

    Keeps v_s exactly, so the source coordinate is unchanged; only the payload DIRECTION
    is randomised. Dose is then matched exactly downstream via ``match_delta_norm``.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    v_s, v_t = v_s.float(), v_t.float()
    # BUGFIX 2026-08-10 (CRITICAL, adversarial review). The previous construction was
    #     step = r - v_s ; v_r = v_s + step * ||v_t-v_s|| / ||step||
    # with r a UNIT vector. Real J-lens rows have ||v_s|| ~ 50, so r - v_s ~ -v_s and the
    # payload collapsed to the same deterministic direction -v_s_hat for EVERY seed
    # (measured: pairwise cosine 0.9996 between seeds; cosine with -v_s_hat 0.9998).
    # The seed was decorative and the arm was a source-ablation control, not a random one.
    # Correct construction: keep v_s exactly, and offset by a random direction of the same
    # norm as the real payload.
    r = torch.randn(v_s.shape[0], generator=g, dtype=torch.float32)
    r = r / r.norm()
    return v_s + r * (v_t - v_s).norm()


def control_orth(v_s: torch.Tensor, v_t: torch.Tensor, seed: int) -> torch.Tensor:
    """Re-point the target so the payload is ORTHOGONAL to the real payload, same norm.

    The harsher of the two controls: zero component along the real write direction.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    v_s, d_real = v_s.float(), (v_t.float() - v_s.float())
    r = torch.randn(v_s.shape[0], generator=g, dtype=torch.float32)
    # Project r out of the WHOLE span{v_s, d_real} at once. Sequential Gram-Schmidt is
    # wrong here: removing the v_s component after the d_real component reintroduces a
    # d_real component whenever v_s and d_real are not themselves orthogonal.
    Q, _ = torch.linalg.qr(torch.stack([d_real, v_s], dim=1))   # [d, 2] orthonormal
    r = r - Q @ (Q.T @ r)
    r = r / r.norm() * d_real.norm()
    return v_s + r


# --------------------------------------------------------------------------- hooks

@dataclass
class SwapRecord:
    """Per-layer dose and coordinate accounting for one forward pass."""

    delta_norm_mean: dict[int, float] = field(default_factory=dict)
    delta_norm_max: dict[int, float] = field(default_factory=dict)
    delta_norm_sum_positions: dict[int, float] = field(default_factory=dict)
    delta_norm_last: dict[int, float] = field(default_factory=dict)
    delta_norm_sum_over_band: float = 0.0          # summed over layers AND positions
    delta_norm_lastpos_sum_over_band: float = 0.0  # legacy: last position only
    pre_coords: dict[int, list[float]] = field(default_factory=dict)
    post_coords: dict[int, list[float]] = field(default_factory=dict)
    cond: dict[int, float] = field(default_factory=dict)
    degenerate: dict[int, bool] = field(default_factory=dict)


class JLensSwapHooks:
    """Anchor §2.5 swap as forward hooks: all positions, band-clamped, every forward.

    Args:
        blocks: the model's residual blocks.
        bases: {layer: SwapBasis}.
        alpha: scale on the swapped coordinates.
        readback: also record pre/post coordinates (verifies the exchange mechanically).
        match_delta_norm: {layer: Tensor[...] } target ‖Δ‖ per position, for matched-dose
            control arms. Built from a prior real-arm pass's ``per_position`` record.
    """

    def __init__(self, blocks, bases: dict[int, SwapBasis], alpha: float = 1.0,
                 *, readback: bool = True,
                 match_delta_norm: dict[int, torch.Tensor] | None = None,
                 skip_positions: int = 0) -> None:
        """skip_positions: leading positions left UNPATCHED. Set to 1 to exclude the BOS /
        attention-sink token, whose residual norm is 7-9x every real token on Pythia, so an
        all-positions write is dominated by a position the anchor's protocol ("at every prompt
        token position") does not include."""
        self.blocks, self.bases, self.alpha = blocks, bases, alpha
        self.readback = readback
        self.skip_positions = skip_positions
        if match_delta_norm:
            missing = set(bases) - set(match_delta_norm)
            if missing:
                raise ValueError(
                    f"match_delta_norm is missing layers {sorted(missing)}; a matched-dose "
                    f"control would silently run at its NATIVE dose on those layers")
        self.match = match_delta_norm or {}
        self.record = SwapRecord()
        self.per_position: dict[int, torch.Tensor] = {}
        self._handles: list = []
        for l, b in bases.items():
            self.record.cond[l] = b.cond
            self.record.degenerate[l] = b.degenerate

    def _mk(self, layer: int):
        cached: dict[str, SwapBasis] = {}

        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            b = cached.get("b")
            if b is None or b.V.device != t.device:
                b = self.bases[layer].to(t.device)
                cached["b"] = b
            if self.readback:
                cpre = coords(t.float(), b).reshape(-1, 2).mean(0)
                self.record.pre_coords[layer] = cpre.tolist()
            tgt = self.match.get(layer)
            if tgt is not None:
                tgt = tgt.to(t.device)
            k = self.skip_positions
            if k:
                head, tail = t[..., :k, :], t[..., k:, :]
                # `tgt` is stored FULL-LENGTH (zeros in the skipped head), so slice it to match
                tgt_tail = None if tgt is None else tgt[..., k:]
                new_tail, dn_tail = patch_swap(tail, b, self.alpha, match_delta_norm=tgt_tail)
                new = torch.cat([head, new_tail], dim=-2)
                # record dose at FULL length with zeros for the skipped positions, so that a
                # control re-using this as match_delta_norm sees the same shape as its own
                # hidden state and the shape guard in patch_swap stays meaningful.
                dn = torch.cat([dn_tail.new_zeros(*dn_tail.shape[:-1], k), dn_tail], dim=-1)
            else:
                new, dn = patch_swap(t, b, self.alpha, match_delta_norm=tgt)
            self.per_position[layer] = dn.detach()
            self.record.delta_norm_mean[layer] = float(dn.mean())
            self.record.delta_norm_max[layer] = float(dn.max())
            self.record.delta_norm_sum_positions[layer] = float(dn.sum())
            self.record.delta_norm_last[layer] = float(dn.reshape(-1)[-1])
            if self.readback:
                cpost = coords(new.float(), b).reshape(-1, 2).mean(0)
                self.record.post_coords[layer] = cpost.tolist()
            return new if torch.is_tensor(output) else (new, *output[1:])

        return hook

    def __enter__(self) -> JLensSwapHooks:
        if self._handles:
            raise RuntimeError("JLensSwapHooks is already entered; re-entering would register "
                               "a second hook per layer and apply the write TWICE")
        self.per_position.clear()
        for d in (self.record.delta_norm_mean, self.record.delta_norm_max,
                  self.record.delta_norm_sum_positions, self.record.delta_norm_last,
                  self.record.pre_coords, self.record.post_coords):
            d.clear()
        self.record.delta_norm_sum_over_band = 0.0
        self.record.delta_norm_lastpos_sum_over_band = 0.0
        try:
            for l in self.bases:
                self._handles.append(self.blocks[l].register_forward_hook(self._mk(l)))
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        # ALL positions, not just the last (H5). The old scalar summed only the final token's
        # dose while the hook writes at every position.
        self.record.delta_norm_sum_over_band = sum(
            self.record.delta_norm_sum_positions.values())
        self.record.delta_norm_lastpos_sum_over_band = sum(
            self.record.delta_norm_last.values())

    def exchange_verified(self, atol: float = 1e-3) -> dict[int, bool]:
        """Mechanical check: post-swap coordinates equal the swapped pre-swap ones.

        Only meaningful when no dose matching is applied (matching rescales Δ and so
        deliberately breaks the exact exchange).
        """
        if self.match:
            return {}
        ok = {}
        for l in self.record.pre_coords:
            cs, ct = self.record.pre_coords[l]
            ps, pt = self.record.post_coords[l]
            ok[l] = (abs(ps - self.alpha * ct) < atol and abs(pt - self.alpha * cs) < atol)
        return ok


class IdentityHooks:
    """Identity null: same hook plumbing, dtype round-trip, no edit.

    NOT alpha=0 on the swap — that ablates the pair-span rather than acting as identity.
    """

    def __init__(self, blocks, layers) -> None:
        self.blocks, self.layers = blocks, list(layers)
        self._handles: list = []

    def __enter__(self) -> IdentityHooks:
        if self._handles:
            raise RuntimeError("IdentityHooks is already entered")

        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            new = t.float().to(t.dtype)
            return new if torch.is_tensor(output) else (new, *output[1:])

        for l in self.layers:
            self._handles.append(self.blocks[l].register_forward_hook(hook))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

# ---------------------------------------------------------------- E8: the other two write families
#
# The anchor defines THREE write families (RIGOROUS_ANTHROPIC sec5). Until 2026-08-11 this module
# implemented only the swap, and that experiment was retracted. These are the other two.
#
# NOTE ON CROSS-LENS USE (external review B5/G6): the logit lens and tuned lens have NO lens-native
# coordinate write. Their canonical interventions are activation patching, mean/resampling
# ablation, activation addition (2308.10248) and weight edits (ROME 2202.05262). So when these
# operators are applied with v_c supplied by a different lens, they are a SHARED hidden-state
# intervention, not that lens's own write. Say so wherever they are reported.


def steer_delta(h: torch.Tensor, v: torch.Tensor, alpha: float = 1.0,
                normalize: bool = True) -> torch.Tensor:
    """Anchor sec5.1: h -> h + alpha * v.  Returns Delta only, so dose accounting stays external.

    `normalize=True` reproduces the anchor's verbal-introspection convention: the unit-normalised
    direction scaled by the layer's mean residual norm times a strength scalar. Raw v has
    ||v|| ~ 50 on our lenses, so an unnormalised alpha is not comparable across layers or models.
    """
    v = v.float()
    d = v / v.norm().clamp_min(1e-12) if normalize else v
    if normalize:
        d = d * h.float().norm(dim=-1, keepdim=True).mean()
    return alpha * d.expand_as(h) if d.dim() == 1 else alpha * d


def ablate_delta(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Anchor sec5.2, single direction: h -> h - proj_v(h). Returns Delta = -proj_v(h).

    Exactly removes the component of h along v. Idempotent by construction: applying it twice
    changes nothing, because after the first application <v, h'> = 0.
    """
    v = v.float()
    vhat = v / v.norm().clamp_min(1e-12)
    return -(h.float() @ vhat).unsqueeze(-1) * vhat


def ablate_span_delta(h: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Anchor sec5.2, span form: h -> h - V (V^T V)^dagger V^T h.  V is [d, k].

    Removes the projection onto span(V) in one step. This is NOT the same as ablating each column
    in turn when the columns are non-orthogonal -- sequential removal reintroduces components,
    which is the same error the `control_orth` construction had to fix.
    """
    V = V.float()
    G = V.T @ V
    P = V @ torch.linalg.pinv(G) @ V.T                      # [d, d] projector onto span(V)
    return -(h.float() @ P.T)


def ablation_kl(logits_clean: torch.Tensor, logits_ablated: torch.Tensor) -> float:
    """Anchor Figure 53's DV: D_KL(p_clean || p_ablated) over the full vocabulary, in nats.

    The anchor reports J-lens ablation inducing ~2x the KL of logit- or tuned-lens ablation on
    multihop. This is the cheapest write result in the paper: no oracle, no alpha, no band
    compounding.
    """
    lp_c = torch.log_softmax(logits_clean.float(), dim=-1)
    lp_a = torch.log_softmax(logits_ablated.float(), dim=-1)
    return float((lp_c.exp() * (lp_c - lp_a)).sum())
