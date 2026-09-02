#!/usr/bin/env python3
"""bootstrap.py — hierarchical and paired resampling for every headline in the programme.

WHY. Not one number in this programme carries a confidence interval, and every headline is
stated in n_win: a sign statistic over 6 eval sets that share a model, a lens, a tokenizer and a
fit corpus. external review Prompt 8 §5A is blunt about what that is worth:

    "If you have 10,000 prompts from two domains, you do not have 10,000 independent shift
     replicates. You have approximately two domain-level replicates."

So the replication unit is the DOMAIN (here: the eval set), not the item. Both resamplers are
provided and they answer different questions:

  paired_item_bootstrap    within one eval set: is THIS set's lens difference real?
                           Resamples the ITEM PAIR, never the two arms independently -- the arms
                           share items, so independent resampling destroys the pairing and
                           inflates the interval.
  hierarchical_bootstrap   across eval sets: does the effect GENERALISE?
                           Resamples sets first, then items within the selected sets. This is
                           the interval that should appear next to any cross-set claim.

n_win itself gets `nwin_interval`, which is the honest way to report a sign statistic: as a
count with an interval, not as a bare fraction.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch


def _pct(t: torch.Tensor, lo=2.5, hi=97.5) -> tuple[float, float]:
    q = torch.quantile(t, torch.tensor([lo / 100, hi / 100], dtype=t.dtype))
    return float(q[0]), float(q[1])


def paired_item_bootstrap(scores_a: Sequence[float], scores_b: Sequence[float], *,
                          n_boot: int = 10000, seed: int = 0,
                          statistic: Optional[Callable] = None) -> dict:
    """CI on the PAIRED difference mean(a) - mean(b) within one eval set.

    ``scores_a[i]`` and ``scores_b[i]`` MUST be the same item under two arms. The resample draws
    item indices once and applies them to both arms, which preserves the pairing and is the
    reason the interval is tighter (and correct) versus resampling each arm separately.
    """
    a = torch.tensor(list(scores_a), dtype=torch.float)
    b = torch.tensor(list(scores_b), dtype=torch.float)
    if len(a) != len(b):
        raise ValueError("paired bootstrap requires equal-length, item-aligned arms")
    n = len(a)
    stat = statistic or (lambda x, y: float((x - y).mean()))
    point = stat(a, b)
    g = torch.Generator().manual_seed(seed)
    draws = torch.randint(0, n, (n_boot, n), generator=g)
    vals = torch.tensor([stat(a[idx], b[idx]) for idx in draws], dtype=torch.float)
    lo, hi = _pct(vals)
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n_items": n, "n_boot": n_boot,
            "excludes_zero": (lo > 0) or (hi < 0),
            "p_two_sided": float(2 * min((vals <= 0).float().mean(),
                                         (vals >= 0).float().mean())),
            "unit": "item (paired)"}


def hierarchical_bootstrap(sets: dict, *, n_boot: int = 10000, seed: int = 0) -> dict:
    """CI that treats the EVAL SET as the replication unit (Prompt 8 §5A, §5C).

    ``sets`` maps set-name -> (scores_a, scores_b), item-aligned within each set.

    Procedure, in Prompt 8's order: resample SETS with replacement; within each selected set
    resample ITEMS with replacement; recompute the pooled statistic. The resulting interval
    covers between-set heterogeneity, which the item bootstrap cannot see and which is the
    dominant source of uncertainty when there are only six sets.

    Expect this interval to be MUCH wider than the item interval. That gap is the finding:
    six shared-model eval sets carry far less evidence than their item count suggests.
    """
    names = list(sets)
    arrs = {k: (torch.tensor(list(v[0]), dtype=torch.float),
                torch.tensor(list(v[1]), dtype=torch.float)) for k, v in sets.items()}
    for k, (a, b) in arrs.items():
        if len(a) != len(b):
            raise ValueError(f"set {k!r} is not item-aligned")
    point = float(torch.tensor([float((a - b).mean()) for a, b in arrs.values()]).mean())
    g = torch.Generator().manual_seed(seed)
    S = len(names)
    vals = []
    set_draws = torch.randint(0, S, (n_boot, S), generator=g)
    for row in set_draws:
        acc = []
        for si in row:
            a, b = arrs[names[int(si)]]
            idx = torch.randint(0, len(a), (len(a),), generator=g)
            acc.append(float((a[idx] - b[idx]).mean()))
        vals.append(sum(acc) / len(acc))
    v = torch.tensor(vals, dtype=torch.float)
    lo, hi = _pct(v)
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n_sets": S, "n_boot": n_boot,
            "excludes_zero": (lo > 0) or (hi < 0),
            "unit": "eval set (hierarchical)",
            "note": "the eval set is the replication unit; six shared-model sets are closer to "
                    "one replicate than six (external review Prompt 8 §5A)"}


def nwin_interval(sets: dict, *, n_boot: int = 10000, seed: int = 0) -> dict:
    """An interval on n_win itself, by resampling SETS (and items within them).

    n_win has been the programme's headline throughout and has never carried uncertainty. This
    reports it as a count with an interval and, more usefully, the probability that the lens
    beats the baseline on a randomly drawn set -- which is what n_win is actually estimating and
    which does not depend on there being exactly six sets.
    """
    names = list(sets)
    arrs = {k: (torch.tensor(list(v[0]), dtype=torch.float),
                torch.tensor(list(v[1]), dtype=torch.float)) for k, v in sets.items()}
    point = sum(1 for a, b in arrs.values() if float(a.mean()) > float(b.mean()))
    g = torch.Generator().manual_seed(seed)
    S = len(names)
    counts = []
    for row in torch.randint(0, S, (n_boot, S), generator=g):
        c = 0
        for si in row:
            a, b = arrs[names[int(si)]]
            idx = torch.randint(0, len(a), (len(a),), generator=g)
            c += int(float(a[idx].mean()) > float(b[idx].mean()))
        counts.append(c)
    t = torch.tensor(counts, dtype=torch.float)
    lo, hi = _pct(t)
    return {"n_win": point, "n_sets": S, "ci_lo": lo, "ci_hi": hi,
            "win_rate": point / S, "win_rate_ci": (lo / S, hi / S),
            "n_boot": n_boot, "unit": "eval set",
            "note": "n_win is a sign statistic over non-independent sets; the interval is wide "
                    "by construction and that is the honest reading"}


def hierarchical_bootstrap_seeded(sets: dict, *, n_boot: int = 10000, seed: int = 0) -> dict:
    """THREE-level sibling of ``hierarchical_bootstrap``: eval set > SEED BLOCK > item pair.

    Added for E48, whose pre-registration asks for a "hierarchical CI (eval-set = replication
    unit, 3 disjoint seed blocks)". ``hierarchical_bootstrap`` has no seed level at all — it
    resamples sets then items — so averaging the seed blocks first would silently discard the
    seed variance component the pre-registration names. This does not redefine the two-level
    function; both are reported and the E48 record states which one adjudicated.

    ``sets`` maps set-name -> (seed_arms, baseline) where ``seed_arms`` is a SEQUENCE of
    per-pair score vectors, one per seed block, all item-aligned with each other and with
    ``baseline``. The baseline (the logit lens) has no seed axis: it is unfitted, which is the
    entire point of the comparison.

    Procedure per replicate: resample SETS with replacement; within each drawn set resample
    SEED BLOCKS with replacement; draw ONE item-pair index vector per drawn set and apply it to
    every drawn block AND to the baseline, which is what preserves the pairing.

    HONEST LIMITATION, stated because it cannot be fixed by design: with B = 3 blocks the seed
    variance component is estimated from three clusters, and a replicate draws the same block
    three times with probability 1/9. The interval is honest about direction; it is NOT
    well calibrated at the seed level. Say that rather than implying otherwise.
    """
    names = list(sets)
    arrs = {}
    for k, (seed_arms, base) in sets.items():
        b = torch.tensor(list(base), dtype=torch.float)
        A = [torch.tensor(list(x), dtype=torch.float) for x in seed_arms]
        for x in A:
            if len(x) != len(b):
                raise ValueError(f"set {k!r} is not item-aligned across seed blocks/baseline")
        arrs[k] = (A, b)
    point = float(torch.tensor(
        [float(torch.tensor([float((x - b).mean()) for x in A]).mean())
         for A, b in arrs.values()]).mean())
    g = torch.Generator().manual_seed(seed)
    S = len(names)
    vals = []
    set_draws = torch.randint(0, S, (n_boot, S), generator=g)
    for row in set_draws:
        acc = []
        for si in row:
            A, b = arrs[names[int(si)]]
            B = len(A)
            blocks = torch.randint(0, B, (B,), generator=g)
            idx = torch.randint(0, len(b), (len(b),), generator=g)
            bb = b[idx]
            acc.append(float(torch.tensor([float((A[int(j)][idx] - bb).mean())
                                           for j in blocks]).mean()))
        vals.append(sum(acc) / len(acc))
    v = torch.tensor(vals, dtype=torch.float)
    lo, hi = _pct(v)
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n_sets": S,
            "n_seed_blocks": len(next(iter(arrs.values()))[0]), "n_boot": n_boot,
            "excludes_zero": (lo > 0) or (hi < 0),
            "unit": "eval set > seed block > item pair (three-level)",
            "note": "seed variance is estimated from 3 clusters; honest about direction, not "
                    "calibrated at the seed level"}


def compare_units(sets: dict, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Run both resamplers and report the WIDTH RATIO.

    This is the diagnostic that makes the power problem concrete: if the hierarchical interval
    is several times wider than the pooled item interval, the item count is not the evidence.
    """
    pooled_a = [x for k in sets for x in sets[k][0]]
    pooled_b = [x for k in sets for x in sets[k][1]]
    item = paired_item_bootstrap(pooled_a, pooled_b, n_boot=n_boot, seed=seed)
    hier = hierarchical_bootstrap(sets, n_boot=n_boot, seed=seed)
    wi = item["ci_hi"] - item["ci_lo"]
    wh = hier["ci_hi"] - hier["ci_lo"]
    return {"item_pooled": item, "hierarchical": hier,
            "width_item": wi, "width_hierarchical": wh,
            "width_ratio": (wh / wi) if wi > 0 else float("inf"),
            "verdict": ("the item interval UNDERSTATES uncertainty by "
                        f"{(wh / wi if wi > 0 else float('inf')):.1f}x")}
