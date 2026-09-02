#!/usr/bin/env python3
"""test_e28_eval_compliance.py — does an E28 eval output satisfy the pre-registration?

The E28 fitting run produced dispersion and no performance number, which is half of what
SPINE.tex sec E28 pre-registers. e28_eval.sh closes that gap. This asserts the gap is
actually closed, against the prereg text rather than against whatever the runner happens
to emit.

    python tests/test_e28_eval_compliance.py <reads.json> <w3.json>
    python tests/test_e28_eval_compliance.py            # uses /tmp smoke outputs

Checks, each traceable to a requirement:
  SPINE E28 "metrics: ALL read metrics"     -> four aggregations present, all numeric
  SPINE E28 admission rule crit. 2          -> both named controls present per arm
  CLAUDE.md discipline 5                    -> ||delta|| per LAYER *and* per POSITION
  SPINE E28 "non-degenerate"                -> read AUCs not all zero / not all identical
  RIGOR axis 4                              -> rand_match is norm-matched, so a control
                                               cannot win on dose alone
"""
from __future__ import annotations
import json, sys

AGGS  = {"min", "best1L", "mean", "persist"}
ARMS  = {"jlens", "logit", "shuffled", "rand_match"}
DOSE_PER_POS   = "dose_mean_per_position"
DOSE_SUM_POS   = "dose_sum_over_positions"

_p = _f = 0
def check(name, cond, detail=""):
    global _p, _f
    if cond: _p += 1; print(f"  ok  {name}   [{detail}]")
    else:    _f += 1; print(f"  FAIL {name}   [{detail}]")

def keys_anywhere(o, out=None):
    out = set() if out is None else out
    if isinstance(o, dict):
        for k, v in o.items(): out.add(k); keys_anywhere(v, out)
    elif isinstance(o, list):
        for v in o: keys_anywhere(v, out)
    return out

def numbers_under(o, key):
    """Every numeric value stored under `key` anywhere in the tree."""
    vals = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == key and isinstance(v, (int, float)): vals.append(float(v))
                else: walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(o); return vals

def main(reads_p, w3_p):
    reads = json.load(open(reads_p))
    w3    = json.load(open(w3_p))

    print("== reads: SPINE E28 'ALL read metrics are RUN'")
    rk = keys_anywhere(reads)
    check("all four aggregations present", AGGS <= rk,
          f"found {sorted(AGGS & rk)}; missing {sorted(AGGS - rk)}")
    for a in sorted(AGGS):
        v = numbers_under(reads, a)
        check(f"{a} is populated and numeric", len(v) > 0, f"{len(v)} value(s)")
    allv = [x for a in AGGS for x in numbers_under(reads, a)]
    check("read AUCs are non-degenerate", len(set(allv)) > 1 and any(x > 0 for x in allv),
          f"{len(set(allv))} distinct values, max {max(allv) if allv else 0:.4f}")

    print("\n== W3: arms and both named controls")
    wk = keys_anywhere(w3)
    check("all four arms present", ARMS <= wk,
          f"found {sorted(ARMS & wk)}; missing {sorted(ARMS - wk)}")
    check("shuffled control present (the decisive one)", "shuffled" in wk, "J^shuf")
    check("rand_match control present", "rand_match" in wk, "norm-matched random direction")

    print("\n== W4 dose: CLAUDE.md discipline 5 — per LAYER and per POSITION")
    check("per-position dose recorded", DOSE_PER_POS in wk, DOSE_PER_POS)
    check("summed-over-position dose recorded", DOSE_SUM_POS in wk, DOSE_SUM_POS)
    # per-layer means the dose keys sit under integer-like layer keys
    def layered(o, key):
        if isinstance(o, dict):
            if any(k.lstrip("-").isdigit() and isinstance(v, dict) and key in v
                   for k, v in o.items() if isinstance(k, str)):
                return True
            return any(layered(v, key) for v in o.values())
        return False
    check("dose is keyed PER LAYER", layered(w3, DOSE_PER_POS),
          "an intervention whose dose is not per-layer is not analysable")

    print("\n== RIGOR axis 4: a control must not win on dose alone")
    dose = {}
    def collect(o, arm=None):
        if isinstance(o, dict):
            for k, v in o.items():
                a = k if k in ARMS else arm
                if k == DOSE_PER_POS and a and isinstance(v, (int, float)):
                    dose.setdefault(a, []).append(float(v))
                else: collect(v, a)
    collect(w3)
    if {"jlens", "rand_match"} <= set(dose):
        j = sum(dose["jlens"]) / len(dose["jlens"])
        r = sum(dose["rand_match"]) / len(dose["rand_match"])
        ratio = max(j, r) / min(j, r) if min(j, r) > 0 else float("inf")
        check("rand_match dose is matched to jlens (ratio < 2x)", ratio < 2.0,
              f"jlens {j:.4f} vs rand_match {r:.4f}  ratio {ratio:.2f}x")
    else:
        check("per-arm dose recoverable", False, f"got arms {sorted(dose)}")

    print(f"\n=== {_p}/{_p+_f} PASSED ===")
    return 1 if _f else 0

if __name__ == "__main__":
    a = sys.argv[1:] or ["/tmp/smoke_reads.json", "/tmp/smoke_w3.json"]
    raise SystemExit(main(a[0], a[1]))
