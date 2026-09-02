#!/usr/bin/env python3
"""test_write_families.py — E8 acceptance tests for steering and ablation.

RESEARCH_NOTES sec13 E8: "Acceptance is a test asserting <v, ablate(h,v)> = 0 and
steer(h,v,0) = h exactly." Plus the properties that distinguish span-ablation from
sequential ablation, which is the error class that produced the vacuous `control_orth`.
"""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "jacobian-lens"))
from joperator import steer_delta, ablate_delta, ablate_span_delta, ablation_kl  # noqa: E402

_p = _f = 0
def check(name, cond, detail=""):
    global _p, _f
    if cond: _p += 1; print(f"  ok  {name}   [{detail}]")
    else:    _f += 1; print(f"  FAIL {name}   [{detail}]")

g = torch.Generator().manual_seed(0)
d = 64
h = torch.randn(d, generator=g)
v = torch.randn(d, generator=g)

# --- steering
check("steer_alpha0_is_identity", float(steer_delta(h, v, 0.0).abs().max()) == 0.0,
      "Delta(0) == 0 exactly")
s1, s2 = steer_delta(h, v, 1.0), steer_delta(h, v, 2.0)
check("steer_is_linear_in_alpha", torch.allclose(s2, 2 * s1, atol=1e-6),
      f"max|d(2)-2d(1)| = {float((s2-2*s1).abs().max()):.2e}")
check("steer_direction_is_v", abs(abs(float(torch.nn.functional.cosine_similarity(
      s1, v, dim=0))) - 1.0) < 1e-5, "|cos(Delta, v)| = 1")

# --- single-direction ablation
ha = h + ablate_delta(h, v)
check("ablate_removes_the_component", abs(float(ha @ (v / v.norm()))) < 1e-5,
      f"<vhat, h'> = {float(ha @ (v/v.norm())):.2e}")
check("ablate_is_idempotent", float(ablate_delta(ha, v).abs().max()) < 1e-5,
      "second application is a no-op")
perp = h - (h @ (v / v.norm())) * (v / v.norm())
check("ablate_preserves_orthogonal_part", torch.allclose(ha, perp, atol=1e-5),
      f"max|diff| = {float((ha-perp).abs().max()):.2e}")

# --- span ablation, and why sequential is wrong
V = torch.stack([v, v + 0.6 * torch.randn(d, generator=g)], dim=1)   # deliberately correlated
hs = h + ablate_span_delta(h, V)
check("span_ablate_kills_both_directions",
      max(abs(float(hs @ (V[:, i] / V[:, i].norm()))) for i in (0, 1)) < 1e-4,
      "both columns orthogonal to the result")
seq = h + ablate_delta(h, V[:, 0]); seq = seq + ablate_delta(seq, V[:, 1])
resid = abs(float(seq @ (V[:, 0] / V[:, 0].norm())))
check("sequential_ablation_is_NOT_equivalent", resid > 1e-3,
      f"sequential leaves <v0, h> = {resid:.4f} -- span form is required for correlated dirs")

# --- ablation KL
lg = torch.randn(1000, generator=g)
check("ablation_kl_is_zero_for_identical", abs(ablation_kl(lg, lg)) < 1e-6,
      f"KL(p||p) = {ablation_kl(lg, lg):.2e}")
check("ablation_kl_is_positive", ablation_kl(lg, torch.randn(1000, generator=g)) > 0,
      "KL > 0 for different distributions")

# --- E20: AblationHooks PLUMBING, not just the algebra.
# ablate_delta's math is tested above; the hook is where the double-final-norm class of bug
# lives (KL 2.395 nats, 50,234/50,304 ranks changed, survived every review). So assert on the
# tensor the hook actually emits, against an independently computed reference.
from torch import nn                                                       # noqa: E402
from t20_ablation_kl import AblationHooks                                  # noqa: E402


class _Block(nn.Module):
    def forward(self, x):
        return x * 1.0


blocks = nn.ModuleList([_Block(), _Block()])
x = torch.randn(1, 5, d, generator=g)
vv = torch.randn(d, generator=g)

hk = AblationHooks(blocks, 0, vv, skip_positions=0)
with hk:
    out = blocks[1](blocks[0](x))
vhat = vv / vv.norm()
check("hook_removes_component_at_every_position",
      float((out.float() @ vhat).abs().max()) < 1e-5,
      f"max_pos |<vhat, h'>| = {float((out.float() @ vhat).abs().max()):.2e}")
check("hook_matches_ablate_delta_reference",
      torch.allclose(out.float(), x + ablate_delta(x, vv), atol=1e-6),
      "hook output == independently computed h + ablate_delta(h, v)")
check("hook_dose_per_position_is_recorded",
      hk.per_position is not None and hk.per_position.numel() == 5,
      f"||Delta|| recorded at {hk.per_position.numel()} positions (D12/W4 requires per-position)")
check("hook_dose_matches_delta_norm",
      abs(hk.delta_norm_sum - float((x.float() @ vhat).abs().sum())) < 1e-4,
      "recorded dose == sum_pos |<vhat, h>|, the exact projection magnitude")

hk2 = AblationHooks(blocks, 0, vv, skip_positions=1)
with hk2:
    out2 = blocks[1](blocks[0](x))
check("skip_positions_leaves_the_head_untouched",
      torch.allclose(out2[:, :1].float(), x[:, :1], atol=0) and float(hk2.per_position[0]) == 0.0,
      "position 0 is bit-identical and its recorded dose is exactly 0")
check("skip_positions_still_ablates_the_tail",
      float((out2[:, 1:].float() @ vhat).abs().max()) < 1e-5,
      "positions >= 1 are still fully ablated")

check("hooks_are_removed_on_exit",
      torch.allclose(blocks[1](blocks[0](x)), x, atol=0),
      "after __exit__ the forward is unmodified -- a leaked hook would silently ablate "
      "every later arm in the sweep")

print(f"\n=== {_p}/{_p+_f} PASSED ===")
raise SystemExit(1 if _f else 0)
