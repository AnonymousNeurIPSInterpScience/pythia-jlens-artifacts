#!/usr/bin/env python3
"""os1c_selfdistance_adjudication.py — OS1c: is OS1's C1 a failed control, or an unfireable one?

PRE-REGISTRATION: docs/experiments/preregs/OS1c_selfdistance_adjudication.md, committed at 84fc6fc
with an AMENDMENT at 9122728, both BEFORE this file existed. Nothing here was written knowing the
answer for the 23 operators and the second metric it turns on.

WHAT THIS DOES. OS1 is recorded UNCLEAR because its C1 demanded a zero self-distance and `theta`
measured 3.735905e-04. `theta` is `mean(arccos(svdvals(Ua.T @ Ub)))`; at `Ua == Ub` every singular
value is 1 and arccos has infinite derivative there, so the float32 floor is ~sqrt(2*eps) =
4.883e-04. OS1b measured that on ONE operator and ONE metric and explicitly adjudicates nothing.
This measures all 24 operators on BOTH metrics, in float32 and float64, against a norm-matched
random matrix, and applies the registered rule.

PRIMARY, per metric m in {theta_32, d_rel}:
    FLOOR_m = max over the 24 operators of m(A,A) in float32
    F64_m   = max over the 24 operators of m(A,A) in float64
    RAND_m  = m(R,R) in float32
    PRED    = sqrt(2 * eps_float32) = 4.883e-04   (PRED_theta; the rule uses it for BOTH metrics)

RULE   RETIRED  if for BOTH metrics: FLOOR_m <= 10*PRED and F64_m <= FLOOR_m/100 and RAND_m within
                a factor of 10 of FLOOR_m.
       STANDS   if any operator's m(A,A) exceeds 10*PRED on either metric.
       UNCLEAR  in any other configuration. Report and stop. Do not re-cut.

CONTROLS  K1 random matrix reproduces the floor.  K2 float64 collapses it 100x.  K3 THE ONE THAT
          CAN FAIL: distinct-operator theta must sit 100x above the floor, or the metric is blind
          and no verdict issues.  K4 scope is OS1's 24.  K5 band is the asserted rule.

WHAT THIS DOES NOT COVER. It adjudicates C1 and nothing else. It cannot and does not change
SEP_d_rel, SEP_theta, their intervals, or OS1's other three controls, and it says nothing about
whether the corpus displacement OS1 measured is real, large, or meaningful for any readout.

    .venv/bin/python experiments/os1c_selfdistance_adjudication.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics as st
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

RES = os.path.join(ROOT, "results")
OS1_JSON = os.path.join(RES, "os1_operator_space_410m.json")

# Band and k transcribed from experiments/os1_operator_space.py:50-51, not re-chosen.
BAND = list(range(9, 22))
K_SUB = 32
N_LAYERS = 24                       # pythia-410m-deduped

# OS1's operator panel, transcribed from experiments/os1_operator_space.py:52-55. K4 asserts this
# set against the corpora the stored OS1 results file itself names, before anything is measured.
INSTREAM = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github"]
OOD = ["OOD_News_2024", "OOD_arXiv_2023", "OOD_CommonPile"]
SEEDS = [0, 1, 2]

# What OS1 stored, quoted so the adjudication is against the recorded numbers and not a re-run.
OS1_C1_THETA = 3.735905e-04
OS1_C1_D_REL = 0.0


def op_path(corpus: str, seed: int) -> str:
    stem = f"INSTREAM_{corpus}" if corpus in INSTREAM else corpus
    return os.path.join(RES, "e48", f"lens_{stem}_410m_n200_s{seed}.pt")


def panel() -> list[tuple[str, int, str]]:
    return [(c, s, op_path(c, s)) for c in INSTREAM + OOD for s in SEEDS]


# --------------------------------------------------------------------------- OS1's two metrics
# Transcribed verbatim from experiments/os1_operator_space.py:66-79. Not re-implemented: a
# different-but-equivalent expression would make a floor measurement uninterpretable, because the
# floor IS the expression.
def d_rel(A: torch.Tensor, B: torch.Tensor) -> float:
    na, nb = A.norm().item(), B.norm().item()
    return float((A - B).norm().item() / (na * nb) ** 0.5)


def top_subspace(A: torch.Tensor, k: int = K_SUB) -> torch.Tensor:
    U, _, _ = torch.linalg.svd(A, full_matrices=False)
    return U[:, :k]


def theta(Ua: torch.Tensor, Ub: torch.Tensor) -> float:
    """Mean principal angle, radians. Scale-free: depends only on the subspaces."""
    s = torch.linalg.svdvals(Ua.T @ Ub).clamp(-1.0, 1.0)
    return float(torch.arccos(s).mean().item())


def band_mean(vals) -> float:
    """OS1 averages each metric over the 13 band layers (pair_stats, os1_operator_space.py:82-85)."""
    return float(st.mean(vals))


def within_factor(a: float, b: float, f: float = 10.0):
    """Is `a` within a factor of `f` of `b`? Returns (verdict, ratio) with the degenerate case
    reported rather than silently resolved: if either side is exactly 0 the ratio does not exist,
    and the registration's UNCLEAR branch is what covers that."""
    if a == 0.0 and b == 0.0:
        return None, None                       # both exactly zero — no ratio exists
    if a == 0.0 or b == 0.0:
        return False, float("inf")
    r = a / b
    return bool(1.0 / f <= r <= f), float(r)


def main() -> int:
    import provenance as prov

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RES, "os1c_selfdistance_adjudication.json"))
    a = ap.parse_args()

    torch.manual_seed(0)
    eps32 = float(torch.finfo(torch.float32).eps)
    eps64 = float(torch.finfo(torch.float64).eps)
    PRED_THETA = math.sqrt(2 * eps32)                 # 4.8828125e-04
    PRED_D_REL = 0.0
    BAR = 10.0 * PRED_THETA                           # 4.8828125e-03, the rule's threshold for BOTH

    ops = panel()
    paths = [p for _, _, p in ops]

    # ------------------------------------------------------------------ K4 and K5, BEFORE measuring
    os1 = json.load(open(OS1_JSON))
    # The registration says to read the 24 from OS1's own `scope`. That field is a PROSE string
    # ("Displacement of the estimated operator..."), not a path list, and OS1's provenance.inputs is
    # empty -- so the paths cannot be read from it literally. They are transcribed from OS1's own
    # source constants instead, and cross-checked here against the panel size the stored results
    # file reports: 8 corpora choose 2, times 3 seeds, is 84 BETWEEN pairs, and 8 times 3-choose-2
    # is 24 WITHIN-seed pairs. Both pin the panel at 8 corpora x 3 seeds = 24 operators.
    n_between_expected = len(list(itertools.combinations(INSTREAM + OOD, 2))) * len(SEEDS)
    n_within_expected = len(INSTREAM + OOD) * len(list(itertools.combinations(SEEDS, 2)))
    k4_counts = (os1["contrasts"]["BETWEEN_corpus"]["n"] == n_between_expected
                 and os1["contrasts"]["WITHIN_seed_block"]["n"] == n_within_expected)
    k4_exist = [p for p in paths if not os.path.exists(p)]
    k4 = {
        "required": "the 24 operator paths are OS1's panel, asserted before measuring",
        "n_operators": len(ops),
        "os1_scope_field_is_prose_not_a_path_list": isinstance(os1.get("scope"), str),
        "os1_provenance_inputs_recorded": len(os1["provenance"].get("inputs", [])),
        "panel_source": "experiments/os1_operator_space.py:52-55 (INSTREAM + OOD, SEEDS)",
        "cross_check": {
            "os1_n_BETWEEN_corpus": os1["contrasts"]["BETWEEN_corpus"]["n"],
            "expected_from_panel": n_between_expected,
            "os1_n_WITHIN_seed_block": os1["contrasts"]["WITHIN_seed_block"]["n"],
            "expected_within": n_within_expected},
        "missing_paths": k4_exist,
        "fires": bool(len(ops) == 24 and k4_counts and not k4_exist)}
    if not k4["fires"]:
        raise SystemExit(f"ABORT: K4 did not fire before measuring: {json.dumps(k4)}")

    rule_band = [l for l in range(math.floor(0.38 * N_LAYERS), math.floor(0.92 * N_LAYERS) + 1)
                 if l < N_LAYERS - 2]
    literal_band = list(range(int(0.38 * N_LAYERS), int(0.92 * N_LAYERS) + 1))
    k5 = {
        "required": "band == floor(0.38L)..floor(0.92L) intersected with layers strictly below the "
                    "penultimate target layer, at L=24. AMENDED: the registration wrote this rule "
                    "without its second clause; see the 2026-08-30 amendment.",
        "band": BAND, "rule_band": rule_band,
        "literal_registered_arithmetic": literal_band,
        "literal_reading_is_unrealisable": ("layer 22 is absent from every stored operator; "
                                            "os1_operator_space.py:63 asserts source_layers == [9..21]"),
        "target_layer": N_LAYERS - 2,
        "fires": BAND == rule_band}

    print(f"  K4 fires={k4['fires']}  K5 fires={k5['fires']}  band={BAND[0]}..{BAND[-1]}", flush=True)
    print(f"  PRED_theta = sqrt(2*eps32) = {PRED_THETA:.6e}   rule threshold 10x = {BAR:.6e}",
          flush=True)

    # ------------------------------------------------------------------ measure the 24
    per_op: dict[str, dict] = {}
    bases32: dict[str, dict] = {}
    ref_layer_norms: dict[int, float] = {}
    for i, (c, s, p) in enumerate(ops):
        key = f"{c}|s{s}"
        d = torch.load(p, map_location="cpu", weights_only=True)
        J = d["J"]
        # OS1 asserts the same thing on load (os1_operator_space.py:63). K5's literal reading is
        # unrealisable precisely because this assert holds on every operator.
        assert d["source_layers"] == BAND, \
            f"{p}: source_layers {d['source_layers'][0]}..{d['source_layers'][-1]}, expected {BAND[0]}..{BAND[-1]}"
        th32, th64, dr32, dr64 = [], [], [], []
        b32 = {}
        for l in BAND:
            A32 = J[l].float()
            A64 = J[l].double()
            U32 = top_subspace(A32)
            U64 = top_subspace(A64)
            b32[l] = U32
            th32.append(theta(U32, U32))
            th64.append(theta(U64, U64))
            dr32.append(d_rel(A32, A32))
            dr64.append(d_rel(A64, A64))
            if key == "Pile-CC|s0":
                ref_layer_norms[l] = float(A32.norm().item())
        bases32[key] = b32
        per_op[key] = {
            "path": os.path.relpath(p, ROOT),
            "theta_32": {"float32": band_mean(th32), "float64": band_mean(th64),
                         "float32_max_layer": max(th32), "float32_min_layer": min(th32)},
            "d_rel": {"float32": band_mean(dr32), "float64": band_mean(dr64),
                      "float32_max_layer": max(dr32)}}
        print(f"    [{i+1:2d}/24] {key:32s} theta32={band_mean(th32):.6e} "
              f"theta64={band_mean(th64):.3e} d_rel32={band_mean(dr32):.6e}", flush=True)
        del J, d

    FLOOR = {m: max(per_op[k][m]["float32"] for k in per_op) for m in ("theta_32", "d_rel")}
    F64 = {m: max(per_op[k][m]["float64"] for k in per_op) for m in ("theta_32", "d_rel")}
    ARGMAX = {m: max(per_op, key=lambda k: per_op[k][m]["float32"]) for m in ("theta_32", "d_rel")}

    # ------------------------------------------------------------------ the random matrix
    # Norm-matched per band layer to Pile-CC|s0, the construction os1_operator_space.py:196-201
    # uses for its own C2, so the comparison is against the same kind of object OS1 built.
    g = torch.Generator().manual_seed(0)
    rth32, rth64, rdr32, rdr64 = [], [], [], []
    for l in BAND:
        R = torch.randn(1024, 1024, generator=g)
        R = R * (ref_layer_norms[l] / R.norm())
        Ur32, Ur64 = top_subspace(R.float()), top_subspace(R.double())
        rth32.append(theta(Ur32, Ur32))
        rth64.append(theta(Ur64, Ur64))
        rdr32.append(d_rel(R.float(), R.float()))
        rdr64.append(d_rel(R.double(), R.double()))
    RAND = {"theta_32": band_mean(rth32), "d_rel": band_mean(rdr32)}
    RAND64 = {"theta_32": band_mean(rth64), "d_rel": band_mean(rdr64)}
    print(f"  RAND theta32={RAND['theta_32']:.6e}  d_rel32={RAND['d_rel']:.6e}", flush=True)

    # ------------------------------------------------------------------ K3 — the one that can fail
    keys = list(per_op)
    pair_theta = []
    for k1, k2 in itertools.combinations(keys, 2):
        pair_theta.append((f"{k1} vs {k2}",
                           band_mean([theta(bases32[k1][l], bases32[k2][l]) for l in BAND])))
    min_pair_name, min_pair = min(pair_theta, key=lambda t: t[1])
    k3_bar = 100.0 * FLOOR["theta_32"]
    k3 = {
        "required": "min over distinct operator pairs of theta_32(A,B) >= 100 * FLOOR_theta. IF "
                    "THIS FAILS THE METRIC IS BLIND AND NO VERDICT ISSUES.",
        "n_pairs": len(pair_theta), "min_distinct_pair_theta": min_pair,
        "min_pair": min_pair_name, "threshold_100x_floor": k3_bar,
        "ratio_min_pair_over_floor": min_pair / FLOOR["theta_32"] if FLOOR["theta_32"] else None,
        "fires": bool(min_pair >= k3_bar)}
    print(f"  K3 min distinct-pair theta = {min_pair:.6e} ({min_pair_name}) vs bar {k3_bar:.6e} "
          f"-> fires={k3['fires']}", flush=True)

    k1_ok, k1_ratio = within_factor(RAND["theta_32"], FLOOR["theta_32"])
    k1 = {"required": "RAND_theta within 10x of FLOOR_theta — the floor is arithmetic, not these "
                      "operators",
          "RAND_theta": RAND["theta_32"], "FLOOR_theta": FLOOR["theta_32"],
          "ratio": k1_ratio, "fires": bool(k1_ok) if k1_ok is not None else False,
          "os1b_reference": {"theta_self": OS1_C1_THETA, "random": 2.859e-04}}
    k2 = {"required": "F64_theta <= FLOOR_theta / 100 — the floor tracks precision, not structure",
          "F64_theta": F64["theta_32"], "FLOOR_theta_over_100": FLOOR["theta_32"] / 100.0,
          "ratio_f32_over_f64": (FLOOR["theta_32"] / F64["theta_32"]) if F64["theta_32"] else float("inf"),
          "fires": bool(F64["theta_32"] <= FLOOR["theta_32"] / 100.0)}

    # ------------------------------------------------------------------ the registered rule
    conds = {}
    for m in ("theta_32", "d_rel"):
        rand_ok, rand_ratio = within_factor(RAND[m], FLOOR[m])
        conds[m] = {
            "FLOOR_le_10xPRED": bool(FLOOR[m] <= BAR),
            "F64_le_FLOOR_over_100": bool(F64[m] <= FLOOR[m] / 100.0),
            "RAND_within_10x_of_FLOOR": rand_ok,        # None == degenerate, both exactly zero
            "RAND_over_FLOOR_ratio": rand_ratio,
            "all_three_hold": bool(FLOOR[m] <= BAR and F64[m] <= FLOOR[m] / 100.0 and rand_ok is True)}

    any_exceeds = {m: [k for k in per_op if per_op[k][m]["float32"] > BAR] for m in conds}
    exceeds = any_exceeds["theta_32"] + any_exceeds["d_rel"]

    # d_rel(A,A) is an exact zero in exact arithmetic and has no arccos singularity. DECLARED BIAS
    # (3) of the registration states, before any number was computed, what follows if it returns
    # exactly 0.0: "C1's d_rel half was fireable and only its theta half was not -- and the rule
    # above then reports UNCLEAR, because the registered C1 required both." That is the
    # registration's own reading of its own rule for this configuration, applied here, not a
    # re-cut of it.
    d_rel_half_was_fireable = FLOOR["d_rel"] == 0.0
    theta_half_was_fireable = FLOOR["theta_32"] == 0.0

    if not k3["fires"]:
        verdict = (f"NO VERDICT — K3 did not fire: the minimum distinct-operator theta is "
                   f"{min_pair:.3e}, below 100x the floor ({k3_bar:.3e}). The metric cannot "
                   f"distinguish identical from different at this scale, so OS1c issues no "
                   f"adjudication. OS1's UNCLEAR stands untouched.")
        outcome = "NO_VERDICT_K3"
    elif not (k4["fires"] and k5["fires"]):
        verdict = ("NO VERDICT — a scope or band control did not fire: "
                   + ", ".join(n for n, c in (("K4", k4), ("K5", k5)) if not c["fires"]))
        outcome = "NO_VERDICT_SCOPE"
    elif exceeds:
        verdict = (f"OS1's UNCLEAR STANDS — {len(set(exceeds))} operator(s) exceed 10*PRED_theta "
                   f"({BAR:.3e}) on a metric: {sorted(set(exceeds))}. The self-distance is larger "
                   f"than arithmetic explains and C1 caught something real.")
        outcome = "STANDS"
    elif conds["theta_32"]["all_three_hold"] and conds["d_rel"]["all_three_hold"]:
        verdict = (f"C1 IS MIS-SPECIFIED, OS1's UNCLEAR IS RETIRED — for both metrics FLOOR <= "
                   f"{BAR:.3e}, float64 collapses it by >=100x, and the norm-matched random matrix "
                   f"reproduces the floor within 10x. OS1's C1 is recorded MIS-SPECIFIED: it "
                   f"demanded exact zero from a float32 quantity whose floor it did not price. "
                   f"OS1's PRIMARY and its other three controls stand exactly as recorded.")
        outcome = "RETIRED"
    elif d_rel_half_was_fireable:
        verdict = (f"UNCLEAR — d_rel(A,A) is exactly 0.0 on all 24 operators in float32, so C1's "
                   f"d_rel half WAS fireable and did fire; only its theta half could not "
                   f"(FLOOR_theta = {FLOOR['theta_32']:.6e} against a predicted floor of "
                   f"{PRED_THETA:.6e}). The registered rule requires the retirement conditions to "
                   f"hold for BOTH metrics, and a metric that returns an exact zero has no floor "
                   f"for a random matrix to reproduce: the RAND-within-10x condition is degenerate "
                   f"(0 against 0), not satisfied. DECLARED BIAS (3) of the registration states "
                   f"this outcome in advance. Report and stop. Do not re-cut. OS1's UNCLEAR stands, "
                   f"and C1 is NOT retired.")
        outcome = "UNCLEAR"
    else:
        failed = [f"{m}:{c}" for m, cc in conds.items() for c, v in cc.items()
                  if c.startswith(("FLOOR", "F64", "RAND_within")) and v is not True]
        verdict = (f"UNCLEAR — the registered retirement conditions do not all hold and no "
                   f"operator exceeds 10*PRED_theta. Failing conditions: {failed}. Report and "
                   f"stop. Do not re-cut.")
        outcome = "UNCLEAR"

    rec = {
        "experiment": "OS1c — is OS1's C1 a failed control, or an unfireable one?",
        "prereg": "docs/experiments/preregs/OS1c_selfdistance_adjudication.md",
        "status": "PRE-REGISTERED",
        "adjudicates": "results/os1_operator_space_410m.json:C1_self_distance — and nothing else",
        "recomputes_not_remeasures": True,
        "model": "EleutherAI/pythia-410m-deduped", "band": BAND, "k_subspace": K_SUB,
        "N": 200, "device": "cpu", "n_operators": len(ops),
        "torch_num_threads": torch.get_num_threads(),
        "float32_eps": eps32, "float64_eps": eps64,
        "PRED_theta_sqrt_2eps32": PRED_THETA, "PRED_d_rel": PRED_D_REL,
        "rule_threshold_10x_PRED_theta": BAR,

        "os1_as_stored": {
            "C1_registered_requirement": "exactly 0 on both metrics",
            "C1_implemented_test": "self_dr == 0.0 and self_th < 1e-6  "
                                   "(os1_operator_space.py:194)",
            "C1_implemented_scope": "ONE operator, Pile-CC|s0 — not the 24 the registration "
                                    "describes; recorded in the 2026-08-30 prereg amendment",
            "C1_theta_measured": OS1_C1_THETA, "C1_d_rel_measured": OS1_C1_D_REL,
            "C1_fires": os1["C1_self_distance"]["fires"],
            "VERDICT": os1["VERDICT"],
            "PRIMARY_untouched_by_this_file": {
                "SEP_d_rel": os1["PRIMARY"]["SEP_d_rel"], "SEP_theta": os1["PRIMARY"]["SEP_theta"]}},

        "PRIMARY": {
            "FLOOR": FLOOR, "FLOOR_argmax_operator": ARGMAX, "F64": F64,
            "RAND": RAND, "RAND_float64": RAND64,
            "conditions_per_metric": conds,
            "operators_exceeding_10xPRED": any_exceeds,
            "d_rel_half_of_C1_was_fireable": d_rel_half_was_fireable,
            "theta_half_of_C1_was_fireable": theta_half_was_fireable},

        "per_operator": per_op,
        "controls": {"K1_random_reproduces_floor": k1, "K2_float64_collapses_it": k2,
                     "K3_metric_can_still_fail": k3, "K4_scope_is_os1s": k4, "K5_band_rule": k5},
        "controls_fired": {"K1": k1["fires"], "K2": k2["fires"], "K3": k3["fires"],
                           "K4": k4["fires"], "K5": k5["fires"]},
        "OUTCOME": outcome,
        "VERDICT": verdict,
        "scope": ("Adjudicates OS1's C1 only. Says nothing about whether the corpus displacement "
                  "OS1 measured is real or matters for any readout, and changes no OS1 number."),
    }

    print("\n" + verdict, flush=True)
    for k, v in rec["controls_fired"].items():
        print(f"  control {k}: fires={v}", flush=True)

    prov.write_result(a.out, rec, script=__file__, experiment="OS1c", inputs=paths + [OS1_JSON])
    print("wrote", os.path.relpath(a.out, ROOT), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
