#!/usr/bin/env python3
"""test_bootstrap.py — the resamplers that finally put intervals on the headlines."""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from bootstrap import (paired_item_bootstrap, hierarchical_bootstrap, nwin_interval,  # noqa: E402
                       compare_units, hierarchical_bootstrap_seeded)

_p = _f = 0
def check(name, cond, detail=""):
    global _p, _f
    if cond: _p += 1; print(f"  ok  {name}   [{detail}]")
    else:    _f += 1; print(f"  FAIL {name}   [{detail}]")

g = torch.Generator().manual_seed(0)

# ---- paired bootstrap recovers a known effect and its absence
a = (torch.randn(200, generator=g) + 0.5).tolist()
b = torch.randn(200, generator=g).tolist()
r = paired_item_bootstrap(a, b, n_boot=2000, seed=0)
check("paired_recovers_a_real_effect", r["excludes_zero"] and r["point"] > 0,
      f"diff = {r['point']:.4f} CI [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
z = torch.randn(200, generator=g).tolist()
r0 = paired_item_bootstrap(z, z, n_boot=2000, seed=0)
check("paired_gives_zero_for_identical_arms",
      abs(r0["point"]) < 1e-9 and r0["ci_hi"] - r0["ci_lo"] < 1e-9,
      "a perfectly paired null has zero width — the pairing is preserved")
try:
    paired_item_bootstrap([1, 2, 3], [1, 2]); _ok = False
except ValueError:
    _ok = True
check("paired_rejects_unaligned_arms", _ok,
      "length mismatch raises — resampling unpaired arms would inflate the interval")

# ---- hierarchical is WIDER than pooled-item: the whole point
sets = {}
for i, off in enumerate([0.5, 0.45, 0.55, 0.5, 0.48, 0.52]):
    sets[f"set{i}"] = ((torch.randn(100, generator=g) + off).tolist(),
                       torch.randn(100, generator=g).tolist())
cu = compare_units(sets, n_boot=800, seed=0)
check("hierarchical_is_wider_than_item", cu["width_ratio"] > 1.0,
      f"item width {cu['width_item']:.4f} vs hierarchical {cu['width_hierarchical']:.4f} "
      f"= {cu['width_ratio']:.1f}x")
check("hierarchical_declares_its_unit",
      cu["hierarchical"]["unit"] == "eval set (hierarchical)", "replication unit is named")

# heterogeneous sets: hierarchical must blow up, item must not notice
het = {}
for i, off in enumerate([2.0, -1.5, 1.8, -1.2, 0.1, 0.3]):
    het[f"set{i}"] = ((torch.randn(100, generator=g) + off).tolist(),
                      torch.randn(100, generator=g).tolist())
cuh = compare_units(het, n_boot=800, seed=0)
check("heterogeneity_inflates_only_the_hierarchical_interval",
      cuh["width_ratio"] > cu["width_ratio"],
      f"ratio {cuh['width_ratio']:.1f}x with heterogeneous sets vs {cu['width_ratio']:.1f}x "
      "with homogeneous — between-set variance is invisible to the item bootstrap")

# ---- n_win gets an interval
nw = nwin_interval(sets, n_boot=500, seed=0)
check("nwin_reports_a_count_and_an_interval",
      nw["n_win"] == 6 and nw["ci_lo"] <= 6 and "win_rate" in nw,
      f"n_win = {nw['n_win']}/6, CI [{nw['ci_lo']:.1f}, {nw['ci_hi']:.1f}], "
      f"win rate {nw['win_rate']:.2f}")
nw_mixed = nwin_interval({k: v for k, v in list(het.items())}, n_boot=500, seed=0)
check("nwin_interval_is_wide_when_sets_disagree",
      nw_mixed["ci_hi"] - nw_mixed["ci_lo"] >= 1.0,
      f"CI [{nw_mixed['ci_lo']:.1f}, {nw_mixed['ci_hi']:.1f}] on n_win = {nw_mixed['n_win']}/6 "
      "— a bare '4/6' hides this entirely")

# ---- the THREE-level sibling added for E48: eval set > seed block > item pair
# identical arms across every block => a perfectly paired null, zero point and zero width
zz = torch.randn(120, generator=g).tolist()
null3 = hierarchical_bootstrap_seeded({f"set{i}": ([zz, zz, zz], zz) for i in range(5)},
                                      n_boot=400, seed=0)
check("seeded_gives_zero_for_identical_arms",
      abs(null3["point"]) < 1e-9 and null3["ci_hi"] - null3["ci_lo"] < 1e-9,
      "pairing survives BOTH the seed-block and the item resample")

# B=1 must reproduce the two-level POINT exactly: with one block there is no seed variance
one_block = {k: ([v[0]], v[1]) for k, v in sets.items()}
p3 = hierarchical_bootstrap_seeded(one_block, n_boot=200, seed=0)["point"]
p2 = hierarchical_bootstrap(sets, n_boot=200, seed=0)["point"]
check("seeded_reduces_to_two_level_point_at_B1", abs(p3 - p2) < 1e-9,
      f"three-level {p3:.9f} vs two-level {p2:.9f}")

# disagreeing seed blocks must WIDEN the interval versus blocks that agree
agree = {f"set{i}": ([(torch.randn(120, generator=g) + 0.5).tolist() for _ in range(3)],
                     torch.randn(120, generator=g).tolist()) for i in range(5)}
disagree = {f"set{i}": ([(torch.randn(120, generator=g) + o).tolist() for o in (2.0, 0.5, -1.0)],
                        torch.randn(120, generator=g).tolist()) for i in range(5)}
wa = hierarchical_bootstrap_seeded(agree, n_boot=600, seed=0)
wd = hierarchical_bootstrap_seeded(disagree, n_boot=600, seed=0)
check("seed_block_disagreement_widens_the_interval",
      (wd["ci_hi"] - wd["ci_lo"]) > (wa["ci_hi"] - wa["ci_lo"]),
      f"width {wd['ci_hi']-wd['ci_lo']:.4f} when blocks disagree vs "
      f"{wa['ci_hi']-wa['ci_lo']:.4f} when they agree — the component the two-level "
      "resampler cannot see")

try:
    hierarchical_bootstrap_seeded({"s": ([[1, 2, 3], [1, 2]], [1, 2, 3])}); _ok3 = False
except ValueError:
    _ok3 = True
check("seeded_rejects_unaligned_blocks", _ok3,
      "a block of the wrong length raises rather than silently misaligning the pairing")

print(f"\n=== {_p}/{_p+_f} PASSED ===")
raise SystemExit(1 if _f else 0)
