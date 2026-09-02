#!/usr/bin/env python3
"""e66d_s_quantization_limited.py — is S quantization-limited, or did the hardware just match?

PRE-REGISTRATION: docs/experiments/preregs/E66d_s_quantization_limited.md, committed at e2d7976
BEFORE this file existed. The decision rule lives there and is not restated here in a form that
could drift from it.

WHAT THIS SETTLES. E66_C_COMPLETION_SPEC section 3 declares S kernel-dependent and warns against
treating it as a reproduction target. The 2026-08-30 L40S re-run returned S bit-identical to the
2026-08-14 value, while F' -- also device-dependent -- did move. This asks whether fp16 rounding
explains that: S compares fp16 against fp16, one fp16 step in [1,2) is 9.765625e-04, and the
measured kernel difference is ~1.07e-06, three orders smaller.

PRIMARY   S_pert(s) = max|fp16(D1 + F' * sign_s) - stored| for five fixed seeds, against S.
RULE      QUANTIZATION-LIMITED    S_pert == S for all five seeds AND Q2 fires
          NOT QUANTIZATION-LIMITED  S_pert != S for any seed
          UNCLEAR                 anything else, in particular Q2 not firing -> no verdict

WHAT THIS CANNOT DO. It can show that quantization SUFFICES to explain S's stability. It cannot
exclude that the 2026-08-14 box was also an L40S: that run never recorded its GPU model, which is
E66b's own finding. The verdict carries that limit.

    .venv/bin/python experiments/e66d_s_quantization_limited.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

RES = os.path.join(ROOT, "results")
D1_PATH = os.path.join(RES, "e66_D1_refit_410m_pilecc_s0.pt")
STORED = os.path.join(RES, "e48", "lens_INSTREAM_Pile-CC_410m_n200_s0.pt")
E66B = os.path.join(RES, "e66b_determinism_floor.json")

BAND = list(range(9, 22))
SEEDS = [0, 1, 2, 3, 4]

# Fixed in the registration. Not recomputed from the run.
FP16_ULP = 9.765625e-04                # one fp16 step in the [1,2) binade
FP16_HALF_ULP = 4.8828125e-04
S_EXPECTED = 0.0010986328125           # what the L40S run reported
D1_SHA = "00eebd6ddffff67532bb954e8d51674f0cd8d5021c0919d7139a942c607fff21"
STORED_SHA = "4f00cc3d7450c3646b78f002"     # first 24 hex, as verified box-to-local
EPS_GRID = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, FP16_HALF_ULP, FP16_ULP]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def s_of(A: dict, B: dict) -> float:
    """S = max over band of |fp16(A) - B|, with A fp32 and B already fp16."""
    return max(float((A[l].half().float() - B[l].float()).abs().max()) for l in BAND)


def s_perturbed(A: dict, B: dict, eps: float, seed: int) -> float:
    """S recomputed after perturbing A by exactly eps with random signs."""
    g = torch.Generator().manual_seed(seed)
    out = 0.0
    for l in BAND:
        sign = (torch.randint(0, 2, A[l].shape, generator=g, dtype=torch.int8).float() * 2.0) - 1.0
        pert = (A[l] + eps * sign).half().float()
        out = max(out, float((pert - B[l].float()).abs().max()))
    return out


def frac_rounding_changed(A: dict, eps: float) -> tuple[float, int, int]:
    """Fraction of entries whose fp16 image changes under +/- eps. This is the mechanism:
    an entry can only move under a perturbation of size eps if it sits within eps of a rounding
    boundary."""
    changed = 0
    total = 0
    for l in BAND:
        a = A[l]
        h = a.half()
        up = (a + eps).half()
        dn = (a - eps).half()
        changed += int(((up != h) | (dn != h)).sum())
        total += a.numel()
    return changed / total, changed, total


def main() -> int:
    import provenance as prov

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RES, "e66d_s_quantization_limited.json"))
    a = ap.parse_args()

    # ------------------------------------------------------------------ Q4/Q5, BEFORE measuring
    d1_sha, st_sha = sha256(D1_PATH), sha256(STORED)
    d1_raw = torch.load(D1_PATH, map_location="cpu", weights_only=True)
    st_raw = torch.load(STORED, map_location="cpu", weights_only=True)
    A = {l: d1_raw["J"][l].float() for l in BAND}
    B = {l: st_raw["J"][l] for l in BAND}
    assert all(B[l].dtype == torch.float16 for l in BAND), "stored operator is not fp16"

    Q4 = {"required": "the inputs are the recorded artifacts, asserted before measuring",
          "D1_sha256": d1_sha, "D1_sha256_expected": D1_SHA,
          "stored_sha256_first24": st_sha[:24], "stored_sha256_expected_first24": STORED_SHA,
          "D1_source_layers": d1_raw["source_layers"], "stored_source_layers": st_raw["source_layers"],
          "D1_dtype": "float32", "stored_dtype": "float16",
          "fires": bool(d1_sha == D1_SHA and st_sha[:24] == STORED_SHA
                        and d1_raw["source_layers"] == BAND and st_raw["source_layers"] == BAND)}
    if not Q4["fires"]:
        raise SystemExit(f"ABORT: Q4 did not fire before measuring: {json.dumps(Q4)}")

    e66b = json.load(open(E66B))
    Fp = e66b["F_prime_changed_reduction_order"]
    Q5 = {"required": "F' is the measured value read from the results file, not a chosen one, and "
                      "it must sit below the fp16 half-ULP",
          "F_prime": Fp, "F_prime_display": f"{Fp:.6e}",
          "source": "results/e66b_determinism_floor.json:F_prime_changed_reduction_order",
          "fp16_half_ulp": FP16_HALF_ULP,
          "F_prime_below_half_ulp": bool(Fp < FP16_HALF_ULP),
          "ratio_half_ulp_over_F_prime": FP16_HALF_ULP / Fp,
          "fires": bool(f"{Fp:.6e}" == "1.072884e-06" and Fp < FP16_HALF_ULP)}
    print(f"  Q4 fires={Q4['fires']}  Q5 fires={Q5['fires']}  F'={Fp:.6e}  "
          f"half-ULP/F' = {FP16_HALF_ULP/Fp:.1f}x", flush=True)

    # ------------------------------------------------------------------ Q3 and Q1
    absmax = max(float(B[l].float().abs().max()) for l in BAND)
    Q3 = {"required": "max|stored| in [1,2) so one fp16 step is 9.765625e-04",
          "abs_max": absmax, "ulp": FP16_ULP, "half_ulp": FP16_HALF_ULP,
          "fires": bool(1.0 <= absmax < 2.0)}

    S = s_of(A, B)
    Q1 = {"required": "S recomputes from the persisted D1 to exactly the value the L40S run "
                      "reported. If it does not, the object on disk is not what produced S.",
          "S_recomputed": S, "S_reported_by_the_run": S_EXPECTED,
          "exact": bool(S == S_EXPECTED), "fires": bool(S == S_EXPECTED)}
    print(f"  Q1 S={S:.10e} expected={S_EXPECTED:.10e} exact={Q1['exact']}", flush=True)
    print(f"  Q3 max|stored|={absmax:.6f} fires={Q3['fires']}", flush=True)

    # ------------------------------------------------------------------ PRIMARY
    primary = {}
    for s in SEEDS:
        sp = s_perturbed(A, B, Fp, s)
        primary[str(s)] = {"S_pert": sp, "unchanged": bool(sp == S), "delta": sp - S}
        print(f"    seed {s}: S_pert={sp:.10e}  unchanged={sp == S}", flush=True)
    all_unchanged = all(v["unchanged"] for v in primary.values())

    # ------------------------------------------------------------------ Q2 — the one that can fail
    q2 = {}
    for s in SEEDS:
        sp = s_perturbed(A, B, FP16_ULP, s)
        q2[str(s)] = {"S_pert": sp, "changed": bool(sp != S)}
    q2_fires = any(v["changed"] for v in q2.values())
    Q2 = {"required": "at eps = one full fp16 ULP (9.765625e-04) the recomputed S MUST change for "
                      "at least one seed. A perturbation that never moves the answer cannot "
                      "distinguish 'invariant' from 'inert'. IF Q2 DOES NOT FIRE, NO VERDICT ISSUES.",
          "eps": FP16_ULP, "by_seed": q2, "n_seeds_changed": sum(v["changed"] for v in q2.values()),
          "fires": bool(q2_fires)}
    print(f"  Q2 full-ULP perturbation changed S on {Q2['n_seeds_changed']}/5 seeds -> "
          f"fires={Q2['fires']}", flush=True)

    # ------------------------------------------------------------------ mechanism, not adjudicated
    grid = [Fp] + EPS_GRID
    mech = {}
    for eps in sorted(set(grid)):
        f, ch, tot = frac_rounding_changed(A, eps)
        mech[f"{eps:.6e}"] = {"fraction_changed": f, "n_changed": ch, "n_total": tot,
                              "is_F_prime": bool(eps == Fp)}
        print(f"    eps={eps:.3e}  frac={f:.6e}  n_changed={ch}/{tot}", flush=True)

    # ------------------------------------------------------------------ the registered rule
    if not Q2["fires"]:
        verdict = ("NO VERDICT — Q2 did not fire: a full fp16 ULP perturbation left S unchanged on "
                   "every seed, so the perturbation machinery is inert and cannot distinguish an "
                   "invariant S from a broken test. Report and stop.")
        outcome = "NO_VERDICT_Q2"
    elif all_unchanged:
        verdict = (
            f"QUANTIZATION-LIMITED — perturbing D1 by exactly the measured device difference "
            f"F' = {Fp:.6e}, with random signs on all {len(SEEDS)} seeds, leaves S bit-identical at "
            f"{S:.10e}. One fp16 step is {FP16_ULP:.6e}, {FP16_HALF_ULP/Fp:.0f}x the half-step above "
            f"F', so fp16 rounding absorbs the kernel difference before S is computed. S's "
            f"reproduction across the two runs is explained by rounding, not by the cards matching, "
            f"and the completion spec's section 3 should be restated as bounded rather than "
            f"absolute: S is comparable across devices for as long as the fitter's run-to-run "
            f"difference stays below the fp16 rounding margin. THIS DOES NOT EXCLUDE that the "
            f"2026-08-14 box was also an L40S — that run recorded no GPU model and the question is "
            f"unrecoverable. Sufficiency is established; exclusivity is not.")
        outcome = "QUANTIZATION_LIMITED"
    else:
        moved = [s for s, v in primary.items() if not v["unchanged"]]
        verdict = (
            f"NOT QUANTIZATION-LIMITED — a perturbation of size F' = {Fp:.6e} moved S on seed(s) "
            f"{moved}. Rounding does not absorb the device difference, so S's exact reproduction is "
            f"not explained by quantization and hardware coincidence becomes the leading "
            f"explanation. Section 3 of the completion spec stands exactly as written.")
        outcome = "NOT_QUANTIZATION_LIMITED"

    rec = {
        "experiment": "E66d — is S quantization-limited, or did the hardware just happen to match?",
        "prereg": "docs/experiments/preregs/E66d_s_quantization_limited.md",
        "status": "PRE-REGISTERED",
        "adjudicates": "the EXPLANATION of S's stability only — not S, F, F', the floor, E66b's "
                       "rule, its CONFIRMED DEFECT verdict, or any of C1-C5",
        "recomputes_not_remeasures": True,
        "model": "EleutherAI/pythia-410m-deduped", "band": BAND, "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "n_entries": 1024 * 1024 * len(BAND),
        "inputs": {"D1": os.path.relpath(D1_PATH, ROOT), "stored": os.path.relpath(STORED, ROOT)},
        "fp16_ulp": FP16_ULP, "fp16_half_ulp": FP16_HALF_ULP,
        "F_prime_measured": Fp,
        "S": S,
        "PRIMARY": {
            "statistic": "S recomputed after perturbing D1 by exactly F' with random signs",
            "by_seed": primary, "all_seeds_unchanged": bool(all_unchanged),
            "seeds": SEEDS},
        "MECHANISM_reported_not_adjudicated": {
            "definition": "fraction of the 13,631,488 band entries whose fp16 image changes under "
                          "+/- eps; an entry can only move if it lies within eps of a rounding "
                          "boundary",
            "by_eps": mech},
        "controls": {"Q1_S_recomputes": Q1, "Q2_perturbation_can_move_S": Q2,
                     "Q3_ulp_constant": Q3, "Q4_inputs_are_recorded": Q4,
                     "Q5_F_prime_is_measured": Q5},
        "controls_fired": {"Q1": Q1["fires"], "Q2": Q2["fires"], "Q3": Q3["fires"],
                           "Q4": Q4["fires"], "Q5": Q5["fires"]},
        "OUTCOME": outcome,
        "VERDICT": verdict,
        "scope": ("Establishes whether fp16 quantization SUFFICES to explain S's stability. It "
                  "cannot exclude hardware coincidence: the 2026-08-14 run recorded no GPU model, "
                  "which is E66b's own finding, and that question is unrecoverable."),
    }

    print("\n" + verdict, flush=True)
    for k, v in rec["controls_fired"].items():
        print(f"  control {k}: fires={v}", flush=True)

    prov.write_result(a.out, rec, script=__file__, experiment="E66d",
                      inputs=[D1_PATH, STORED, E66B])
    print("wrote", os.path.relpath(a.out, ROOT), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
