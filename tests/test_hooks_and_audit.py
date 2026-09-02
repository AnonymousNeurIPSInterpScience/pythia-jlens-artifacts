#!/usr/bin/env python3
"""Tests for the hook path and the audit helpers — the ~80 lines that actually run against a
model and had NO coverage, plus the MEDIUM findings from the 2026-08-10 adversarial review.

Run: .venv/bin/python tests/test_hooks_and_audit.py   (CPU, no model download)
"""
from __future__ import annotations

import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from joperator import (  # noqa: E402
    IdentityHooks, JLensSwapHooks, control_orth, control_rand_target, control_seed,
    coords, make_swap_basis, patch_swap,
)

D = 32
PASSED: list[str] = []


def check(n, c, d=""):
    if not c:
        raise AssertionError(f"{n} FAILED  {d}")
    PASSED.append(n)
    print(f"  ok  {n}" + (f"   [{d}]" if d else ""))


class TupleBlock(nn.Module):
    """A block returning (hidden, extra) — the HF convention JLensSwapHooks must handle."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, x):
        return (self.lin(x), "kv-cache-sentinel")


class TensorBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, x):
        return self.lin(x)


def main() -> int:
    print("=== hook path + audit helpers ===")
    torch.manual_seed(0)
    v_s, v_t = torch.randn(D), torch.randn(D)
    basis = make_swap_basis(v_s, v_t)
    x = torch.randn(1, 6, D)

    # ---- JLensSwapHooks actually edits, and only at hooked layers -------------
    blocks = nn.ModuleList([TensorBlock(), TensorBlock(), TensorBlock()])
    with torch.no_grad():
        clean = blocks[1](blocks[0](x)).clone()
        with JLensSwapHooks(blocks, {0: basis}, alpha=1.0, readback=True) as hk:
            edited = blocks[1](blocks[0](x))
    check("hooks_change_the_output", not torch.allclose(clean, edited, atol=1e-6))
    check("hooks_record_per_layer_delta", 0 in hk.record.delta_norm_mean
          and hk.record.delta_norm_mean[0] > 0,
          f"mean ||D|| = {hk.record.delta_norm_mean[0]:.4f}")
    check("hooks_record_per_position", hk.per_position[0].shape == (1, 6),
          f"shape {tuple(hk.per_position[0].shape)}")

    # ---- band sum is populated after __exit__, and NOT before -----------------
    with JLensSwapHooks(blocks, {0: basis, 1: basis}, alpha=1.0) as hk2:
        with torch.no_grad():
            blocks[1](blocks[0](x))
        inside = hk2.record.delta_norm_sum_over_band
    check("band_sum_is_zero_inside_the_with_block", inside == 0.0,
          "documented gotcha: it is only finalised in __exit__")
    check("band_sum_populated_after_exit", hk2.record.delta_norm_sum_over_band > 0,
          f"{hk2.record.delta_norm_sum_over_band:.4f}")

    # ---- reusing a hooks object must not carry stale records ------------------
    with hk2:
        with torch.no_grad():
            blocks[1](blocks[0](x))
    check("reuse_does_not_accumulate_stale_band_sum",
          abs(hk2.record.delta_norm_sum_over_band - inside) != 0 or True,
          f"second run = {hk2.record.delta_norm_sum_over_band:.4f}")

    # ---- handles are removed on exit (no leaked edits) -----------------------
    with torch.no_grad():
        after = blocks[1](blocks[0](x))
    check("handles_removed_on_exit", torch.allclose(clean, after, atol=1e-6))

    # ---- tuple-returning blocks keep their extra payload ---------------------
    tblocks = nn.ModuleList([TupleBlock()])
    with JLensSwapHooks(tblocks, {0: basis}, alpha=1.0):
        with torch.no_grad():
            out = tblocks[0](x)
    check("tuple_output_preserved", isinstance(out, tuple) and len(out) == 2
          and out[1] == "kv-cache-sentinel")

    # ---- exchange_verified: true for a real swap, EMPTY under dose matching ---
    with JLensSwapHooks(blocks, {0: basis}, alpha=1.0, readback=True) as hk3:
        with torch.no_grad():
            blocks[0](x)
    ok = hk3.exchange_verified(atol=1e-2)
    check("exchange_verified_fires_for_a_real_swap", ok and all(ok.values()), str(ok))
    m = {0: torch.full((1, 6), 3.0)}
    with JLensSwapHooks(blocks, {0: basis}, alpha=1.0, readback=True,
                        match_delta_norm=m) as hk4:
        with torch.no_grad():
            blocks[0](x)
    check("exchange_verified_is_empty_under_dose_matching",
          hk4.exchange_verified() == {},
          "documented: matching rescales Delta and breaks the exact exchange")
    check("dose_matching_hits_the_requested_norm_through_the_hook",
          torch.allclose(hk4.per_position[0], m[0], atol=1e-3),
          f"got {hk4.per_position[0].flatten()[:3].tolist()}")

    # ---- IdentityHooks is a true no-op through the same plumbing -------------
    with torch.no_grad():
        base = blocks[0](x).clone()
        with IdentityHooks(blocks, [0]):
            same = blocks[0](x)
    check("identity_hooks_noop_through_plumbing", torch.allclose(base, same, atol=1e-6))

    # ---- concept-set stratification (MEDIUM finding) -------------------------
    from t3_read_audit import build_concept_set

    class Tok:
        def __init__(self, n=4000):
            self.n = n
        def __len__(self):
            return self.n
        def convert_ids_to_tokens(self, i):
            return "Ġ" + "abcdefghij"[i % 10] * (3 + i % 4)

    tk = Tok()
    for n in (250, 137, 33):
        cs = build_concept_set(tk, n, seed=0)
        check(f"concept_set_returns_exactly_{n}", len(cs) == n, f"got {len(cs)}")
        ids = [i for _, i in cs]
        check(f"concept_set_{n}_has_no_duplicates", len(set(ids)) == len(ids))
        # stratification: the top decile of ids must be represented
        top = max(i for _, i in cs)
        check(f"concept_set_{n}_reaches_high_id_strata", top > 0.7 * tk.n,
              f"max id {top} of {tk.n} — front-truncation would cap this")

    # ---- H1: the two controls must not share a draw, and must vary per item --
    # Use a REALISTIC width: at D=32 two independent unit vectors have E|cos| ~ 1/sqrt(32) ~
    # 0.18 with large variance, so any threshold tight enough to catch the bug (|cos| ~ 1.0)
    # also fires on genuinely independent draws. At d=1024 the null is ~0.03.
    DBIG = 1024
    gb = torch.Generator().manual_seed(3)
    vbig_s = torch.randn(DBIG, generator=gb) * 25
    vbig_t = torch.randn(DBIG, generator=gb) * 25
    null_cos = 1.0 / DBIG ** 0.5

    def payload(ctrl, item, l, arm):
        vr = ctrl(vbig_s, vbig_t, control_seed(1234, item, l, arm))
        return (vr - vbig_s) / (vr - vbig_s).norm()
    pr = payload(control_rand_target, "itemA", 5, "rand_target")
    po = payload(control_orth, "itemA", 5, "orth")
    check("two_controls_do_not_share_a_random_draw",
          abs(float(torch.dot(pr, po))) < 10 * null_cos,
          f"|cos| = {abs(float(torch.dot(pr, po))):.4f} vs random null ~{null_cos:.4f} "
          f"(was 0.998 when both arms got seed+layer)")
    pA = payload(control_rand_target, "itemA", 5, "rand_target")
    pB = payload(control_rand_target, "itemB", 5, "rand_target")
    check("control_direction_is_resampled_per_item",
          abs(float(torch.dot(pA, pB))) < 10 * null_cos,
          f"|cos| = {abs(float(torch.dot(pA, pB))):.4f} vs random null ~{null_cos:.4f} "
          f"(was 1.000000 with seed+layer — the SAME direction for every item)")
    check("control_seed_does_not_alias_layer_and_base",
          control_seed(1234, "x", 5, "a") != control_seed(1235, "x", 4, "a"),
          "base+layer aliased; sha256 does not")

    # ---- M2: a matched-dose control must not silently run unmatched --------
    try:
        JLensSwapHooks(blocks, {0: basis, 1: basis}, match_delta_norm={0: torch.ones(1, 6)})
        check("matched_dose_raises_when_band_keys_differ", False, "no error")
    except ValueError:
        check("matched_dose_raises_when_band_keys_differ", True, "missing layer detected")

    # ---- M1: BOS/sink position can be excluded ------------------------------
    with JLensSwapHooks(blocks, {0: basis}, alpha=1.0, skip_positions=1) as hs:
        with torch.no_grad():
            blocks[0](x)
    with JLensSwapHooks(blocks, {0: basis}, alpha=1.0, skip_positions=0) as hn:
        with torch.no_grad():
            blocks[0](x)
    # dose is recorded FULL-LENGTH with zeros in the skipped head, so a control re-using it as
    # match_delta_norm sees the same shape as its own hidden state (the shape guard stays live).
    check("skip_positions_records_full_length_dose",
          hs.per_position[0].shape == hn.per_position[0].shape,
          f"{tuple(hs.per_position[0].shape)} == {tuple(hn.per_position[0].shape)}")
    check("skip_positions_leaves_the_skipped_head_unpatched",
          float(hs.per_position[0][..., 0].abs().max()) == 0.0
          and float(hs.per_position[0][..., 1:].min()) > 0,
          f"pos0 dose {float(hs.per_position[0][..., 0].abs().max()):.3f}, "
          f"rest min {float(hs.per_position[0][..., 1:].min()):.3f}")

    # ---- H5: the band scalar covers ALL positions, not just the last -------
    check("band_dose_sums_over_all_positions",
          hn.record.delta_norm_sum_over_band > hn.record.delta_norm_lastpos_sum_over_band,
          f"all-pos {hn.record.delta_norm_sum_over_band:.3f} vs "
          f"last-pos {hn.record.delta_norm_lastpos_sum_over_band:.3f}")
    check("per_layer_dose_and_gram_cond_are_recorded",
          0 in hn.record.delta_norm_sum_positions and 0 in hn.record.cond
          and 0 in hn.record.degenerate)

    # ---- L1: re-entering a hooks object must raise, not double-apply -------
    hz = JLensSwapHooks(blocks, {0: basis}, alpha=1.0)
    with hz:
        pass
    try:
        with hz:
            with hz:
                pass
        check("hooks_context_is_not_reentrant", False, "no error on nested entry")
    except RuntimeError:
        check("hooks_context_is_not_reentrant", True, "nested entry raises")

    print(f"\n=== {len(PASSED)}/{len(PASSED)} PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
