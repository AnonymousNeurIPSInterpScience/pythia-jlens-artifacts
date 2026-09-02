#!/usr/bin/env python3
"""t48_crossover.py — E48: does the unfitted logit lens OVERTAKE the fitted J-lens under shift?

THIS SCRIPT ADJUDICATES A PRE-REGISTERED RULE IT DOES NOT GET TO CHANGE.
The rule is `docs/experiments/preregs/superseded/PREREG_E48_CROSSOVER.md` + Amendment 1, committed at `408fa5c` /
`5869b63`, BEFORE any OOD lens existed. It is reproduced verbatim in RULE below and the code
that evaluates it is a direct transcription. If a branch fires that the rule does not name, this
script prints UNCLEAR and stops. It never re-thresholds.

WHAT THE PROGRAMME NEEDS FROM THIS
  Every prior corpus in this programme is a Pile component, so the model was IN-DISTRIBUTION
  throughout and the design *could not* produce a crossover. E33 found the logit lens sitting
  INSIDE the corpus spread — a crossover on the CORPUS axis — but the SHIFT axis has never been
  measured. This is the paper's oldest [UNMEASURED].

  The J-lens is an ESTIMATOR: J = E_p[dh_final/dh_l] carries a dependence on the corpus p. The
  logit lens is the J=I special case — no fitted parameters, so it cannot degrade under shift.
  S3 predicts a shift magnitude at which the free baseline overtakes the fitted one.

DESIGN
  model     pythia-410m-deduped, fp32, TF32 off, penultimate target, skip_first=16
  band      [9..21] — E28/E33's band, and exactly the band the 9 OOD lenses carry
  N         200 per fit, 3 disjoint seed blocks per corpus  (MATCHED across every rung)
  window    every fit used max_seq_len=128 with a >=128-token pool filter, so all arms are
            hard length-matched at 128 tokens. (`mean_prompt_tokens` in the tv jsons is the
            UNTRUNCATED document length and is not what was fitted.)
  rungs     in-stream : Wikipedia_en, USPTO_Backgrounds, Pile-CC, StackExchange   [clause a]
            register-far / exposure-near : Github                                 [clause d]
            OOD       : OOD_News_2024, OOD_arXiv_2023, OOD_CommonPile             [clause b]
  arms      logit (J=I) · J^P for all 8 corpora x 3 seeds · J^shuf (n random derangements per
            corpus x seed) · norm-matched random · identity (self-test)
  metric    read AUC over K=[1,2,5,10,20,50,100]; `persist` PRIMARY (as registered), `min`
            reported alongside (the anchor's own metric)
  CI        THREE-level hierarchical bootstrap — eval set > seed block > item pair
            (`bootstrap.hierarchical_bootstrap_seeded`). This is chosen as PRIMARY, and the
            choice is recorded here BEFORE any OOD read was scored, because the pre-registration
            asks for "eval-set = replication unit, 3 disjoint seed blocks" and the repo's
            existing two-level `hierarchical_bootstrap` has NO seed level — averaging the blocks
            first would silently discard the variance component the rule names. The two-level
            interval on the seed-averaged difference is computed and reported ALONGSIDE. If the
            two disagree on any clause that is FLAGGED, not resolved.
            Association is excluded (E28 admission criterion 3).

TWO SCORING-PATH FORKS, BOTH DISCLOSED AND BOTH RUN — declared before any result
  rstrip    `anchor_evals.py:32-34` states that `.rstrip()` on the prompt is LOAD-BEARING (the
            released prompts end in a trailing space, which otherwise becomes the readout token).
            t22/t17 rstrip; trainval.py and t33_logit_baseline.py DO NOT. Measured here: the
            readout token differs on 157/551 items (multihop 83/93, order-ops 55/55,
            multilingual 19/107). So E28's and E33's entire number set — including the 0.02808
            and 0.02844 that clause (d) leans on — was read unstripped.
            PRIMARY is the UNSTRIPPED path, because clause (d) and control C0 are defined
            against E33's numbers and only the unstripped path can reproduce them. The stripped
            path is run as a disclosed sensitivity arm (`--rstrip`). It is CPU and free. If the
            verdict flips between them, that is reported, not resolved.
  pass@k    the anchor's README defines pass@k as a mean over ITEMS of the per-item fraction of
            intermediates recovered; neither this programme's scorers nor t22 implement that —
            both flat-pool over (item, intermediate) pairs. Inherited, unchanged here so that C0
            can fire, and disclosed. Under the set-unweighted hierarchical aggregation the
            pooling is contained, but any pair-pooled statistic is multilingual-dominated
            (394 of ~893 pairs come from one set).

CLAUSE 5 IS BINDING: eval activations are cached ONCE and every arm — logit, each J^P, each
J^shuf, the norm-matched random — is scored against that identical cache by one function.

CONTROLS, each with the number it must produce
  C0 REPRODUCTION   the logit arm and the `e28_Github_410m_n400_s0.pt` arm must reproduce
                    e33_logit_baseline_410m_v2.json to <1e-9 per eval set, under BOTH
                    aggregations. A bit-level proof that this scoring path is the one every
                    prior number in the programme came from. If C0 fails the run is VOID.
  C1 SELF-TEST      identity transport minus logit lens = exactly 0.0 on every pair. Any width
                    means the pairing between arms is broken.
  C2 SHUF FLOOR     J^P must beat its own J^shuf. Reported as a DISTRIBUTION over draws, not one
                    number — E49 measured n_win swinging over [1,3] at 160M across draws, so a
                    single derangement is a sample, not a control.
  C3 RANDOM FLOOR   norm-matched random transport reads ~0 under persist (E33 measured exactly
                    0.0). A nonzero value means the readout leaks.

DECLARED BIAS
  * Github's half of clause (d) is NOT blind: e33 v2 already measured it at 0.02808 vs the logit
    lens's 0.02844 (a CI including zero, which is what (d) requires). Amendment 1 discloses this.
    The FORBIDDING half — "if Github goes strictly negative, REGISTER-CONFOUNDED" — retains full
    force, because that outcome has not been measured under these conditions.
  * Five eval sets is at the bottom of the usable band for a hierarchical CI, and they share a
    model, tokenizer and battery. The interval is the honest floor, not a solution.
  * CE competence is a SEPARATE prerequisite (t48_competence_gate.py). This script REFUSES to
    adjudicate unless that gate's verdict is on disk, because a negative read on a corpus the
    model cannot model is model-confusion, not distributional bias, and the two are not
    separable after the fact.

    python t48_crossover.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))
from provenance import write_result  # noqa: E402

MODEL = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))                       # [9..21] — E28/E33, and the OOD lens band
K = [1, 2, 5, 10, 20, 50, 100]
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]   # association floored

INSTREAM_A = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange"]   # clause (a)
GITHUB = "Github"                                                                # clause (d)
OOD = ["OOD_News_2024", "OOD_arXiv_2023", "OOD_CommonPile"]                      # clause (b)
NEWS = "OOD_News_2024"                                                           # clause (d)
SEEDS = [0, 1, 2]
CORPORA = INSTREAM_A + [GITHUB] + OOD

RULE = """CONFIRMED iff all three hold:
  (a) (J^P - logit) CI strictly > 0 for the in-stream English rungs
      (Wikipedia, USPTO, Pile-CC, StackExchange); AND
  (b) (J^P - logit) CI strictly < 0 for >=2 of 3 genuinely-OOD rungs
      (News_2024, arXiv_2023, CommonPile); AND
  (c) those OOD operators read strictly ABOVE the J_shuf floor
      (structured operator, wrong distribution - not a collapsed one).
NOT REACHED iff (J^P - logit) CI includes or exceeds 0 for all OOD rungs.
  -> positive-framed negative; publishable as registered; NOT to be re-described as a
     weaker crossover, and no hunting for a rung that dips.
DEGENERACY (excluded either way) iff OOD operators read BELOW J_shuf.
AMENDMENT 1, clause (d), required for CONFIRMED:
  OOD_News_2024 must be among the rungs going strictly < 0, AND Github must sit AT the
  crossover (CI includes 0) or above - NOT strictly below. If Github goes strictly
  negative -> REGISTER-CONFOUNDED, not CONFIRMED."""


def lens_path(corpus: str, seed: int) -> str:
    """The n=200, seed-block-`seed` lens for `corpus`. OOD lenses live in results/e48/."""
    if corpus.startswith("OOD_"):
        return os.path.join(HERE, "..", "results", "e48", f"lens_{corpus}_410m_n200_s{seed}.pt")
    return os.path.join(HERE, "..", "results", "lenses", "e28", f"e28_{corpus}_410m_n200_s{seed}.pt")


def derangement(band, seed):
    """A RANDOM derangement. Not a cyclic shift: adjacent-layer Jacobians are the most similar
    pair in the band, so a shift-by-one is the weakest member of this control family, and that
    is exactly what made E33 v1's control fail at 4/6."""
    g = torch.Generator().manual_seed(seed)
    while True:
        perm = [band[i] for i in torch.randperm(len(band), generator=g).tolist()]
        if all(p != l for p, l in zip(perm, band)):
            return perm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-derangements", type=int, default=5,
                    help="draws per (corpus,seed); E49 showed one draw is a sample, not a control")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-boot-item", type=int, default=2000)
    ap.add_argument("--items-per-chunk", type=int, default=24)
    ap.add_argument("--gate", default=os.path.join(HERE, "..", "results",
                                                   "e48_competence_gate_410m.json"))
    ap.add_argument("--allow-failed-gate", action="store_true",
                    help="score anyway and stamp the verdict GATE-FAILED. The pre-registration "
                         "says clause (b) must be RE-SPECIFIED BY THE OPERATOR before scoring a "
                         "failed rung; this flag exists only so the operator can ask for the "
                         "numbers explicitly.")
    ap.add_argument("--rstrip", action="store_true",
                    help="SENSITIVITY ARM. Apply anchor_evals' load-bearing .rstrip() to the "
                         "prompt before encoding. Changes the readout token on 157/551 items. "
                         "PRIMARY is unstripped, because that is the only path that reproduces "
                         "E33 (control C0) and clause (d) is defined against E33's numbers.")
    ap.add_argument("--smoke", action="store_true",
                    help="one corpus, one seed, one derangement, 8 items/set — proves the path")
    ap.add_argument("--controls-only", action="store_true",
                    help="score ONLY the logit arm, C0 and C1, then exit. Scores no rung, so it "
                         "reads no guarded outcome and does not require the competence gate. "
                         "Run this BEFORE the full pass: if C0 does not fire, nothing the full "
                         "pass produces is comparable to E28/E33 and the run is wasted.")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e48_crossover_410m.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of
    from bootstrap import (hierarchical_bootstrap, hierarchical_bootstrap_seeded,
                           paired_item_bootstrap)

    corpora = [INSTREAM_A[0]] if a.smoke else CORPORA
    seeds = [0] if a.smoke else SEEDS
    n_der = 1 if a.smoke else a.n_derangements

    # ---------------------------------------------------------------- the competence gate
    gate = json.load(open(a.gate)) if os.path.exists(a.gate) else None
    if not a.smoke and not a.controls_only and (gate is None or "VERDICT" not in gate):
        raise SystemExit(
            f"REFUSING TO SCORE: the model-competence gate has not been run.\n"
            f"  expected {a.gate}\n"
            f"  run: python t48_competence_gate.py --model 410m --device cpu\n"
            f"A negative read on a corpus the model cannot model is model-confusion, not "
            f"distributional bias, and the two are not separable after the fact.")
    gate_failed = [c for c, g in (gate or {}).get("gate", {}).items() if not g["passes"]]
    if gate_failed and not a.allow_failed_gate and not a.smoke and not a.controls_only:
        raise SystemExit(
            f"STOP — the competence gate FAILED for {gate_failed}.\n{gate['VERDICT']}\n"
            f"The E48 handoff requires clause (b) to be RE-SPECIFIED IN THE PRE-REGISTRATION, by "
            f"the operator, BEFORE a single read is scored. Selecting rungs after seeing a read "
            f"is a researcher degree of freedom. Not resolvable by this script.")

    # ---------------------------------------------------------------- model + ONE activation cache
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    assert len(model.layers) == 24, len(model.layers)
    d = model.d_model
    L = len(BAND)

    t0 = time.time()
    acts_rows, pair_set, pair_targets, pair_item, per_set_n = [], [], [], [], {}
    with torch.no_grad():
        for name in EVAL_SETS:
            for it in load_eval(name):
                tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
                if not tgt:
                    continue
                if a.smoke and per_set_n.get(name, 0) >= 8:
                    continue        # smoke keeps a few items PER SET so no admitted set is empty
                per_set_n[name] = per_set_n.get(name, 0) + 1
                p = it["prompt"].rstrip() if a.rstrip else it["prompt"]
                ids = model.encode(p, max_length=256)
                pos = readout_position(tok, name, p)
                with ActivationRecorder(model.layers, at=BAND) as rec:
                    model.forward(ids)
                    A = torch.stack([rec.activations[l][0][pos].detach().float() for l in BAND])
                idx_item = len(acts_rows)
                acts_rows.append(A)
                for t in tgt:
                    pair_set.append(name)
                    pair_targets.append(t)
                    pair_item.append(idx_item)
    ACTS = torch.stack(acts_rows)                        # [I, L, d]
    n_items, P_n = ACTS.shape[0], len(pair_targets)
    ITEM_PAIRS = {}
    for pi, ii in enumerate(pair_item):
        ITEM_PAIRS.setdefault(ii, []).append(pi)
    SET_IDX = {s: torch.tensor([i for i, p in enumerate(pair_set) if p == s], dtype=torch.long)
               for s in EVAL_SETS}
    print(f"cached {n_items} eval items, {P_n} (item,intermediate) pairs, band={BAND}  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    HALF = L // 2

    # ---------------------------------------------------------------- THE one scoring function
    def score(T):
        """T: {layer: [d,d]} or None for the identity. Returns per-pair rank matrix [P, L].

        Ranks are exactly trainval.py / t33's: 1-indexed strict-greater count over the full
        vocabulary, min over the token ids of a multi-token intermediate. The only difference is
        that items are unembedded in batches so 170 arms are affordable. C0 proves the batching
        changes nothing.
        """
        if T is None:
            H = ACTS
        else:
            H = torch.stack([ACTS[:, j, :] @ T[BAND[j]].T for j in range(L)], dim=1)
        flat = H.reshape(-1, d)                                      # [I*L, d]
        R = torch.empty(P_n, L, dtype=torch.float32)
        with torch.no_grad():
            for i0 in range(0, n_items, a.items_per_chunk):
                i1 = min(i0 + a.items_per_chunk, n_items)
                lg = model.unembed(flat[i0 * L:i1 * L]).float()       # [(i1-i0)*L, V]
                for ii in range(i0, i1):
                    block = lg[(ii - i0) * L:(ii - i0 + 1) * L]       # [L, V]
                    for pi in ITEM_PAIRS[ii]:
                        # rank of each candidate id at every layer at once, then min over ids
                        cand = torch.stack([(block > block[:, i:i + 1]).sum(1) + 1
                                            for i in pair_targets[pi]])   # [n_ids, L]
                        R[pi] = cand.min(0).values.float()
        return R

    def aggregate(R):
        """Per-pair scores under both aggregations. mean over pairs == trainval.py's set AUC."""
        mn = R.min(dim=1).values
        return {"min": torch.stack([(mn <= k).float() for k in K]).mean(0),
                "persist": torch.stack([((R <= k).float().sum(1) >= HALF).float()
                                        for k in K]).mean(0)}

    def set_mean(v, s):
        return float(v[SET_IDX[s]].mean())

    def admitted_mean(v):
        return statistics.mean([set_mean(v, s) for s in ADMITTED])

    def load_J(path):
        blob = torch.load(path, map_location="cpu", weights_only=True)["J"]
        missing = [l for l in BAND if l not in blob]
        if missing:
            raise SystemExit(f"{path} lacks band layers {missing} — arms would not be comparable")
        return {l: blob[l].float() for l in BAND}

    ARMS = {}

    def run(name, T):
        t = time.time()
        ARMS[name] = aggregate(score(T))
        print(f"  {name:40s} persist={admitted_mean(ARMS[name]['persist']):.6f} "
              f"min={admitted_mean(ARMS[name]['min']):.6f}  [{time.time()-t:.0f}s]", flush=True)

    print("scoring arms (ONE cached activation set, ONE scoring function — clause 5)...", flush=True)
    run("logit_I", None)

    # ---- C0 REPRODUCTION against e33 v2, bit-for-bit
    run("C0_ref_Github_n400_s0", load_J(os.path.join(HERE, "..", "results", "lenses", "e28", "e28_Github_410m_n400_s0.pt")))
    c0 = {"target": "e33_logit_baseline_410m_v2.json", "max_abs_diff": 0.0, "per_set": {},
          "skipped": bool(a.smoke or a.rstrip),
          "why_skipped": ("the rstrip sensitivity arm reads a DIFFERENT token on 157/551 items, "
                          "so it cannot reproduce E33 by construction; C0 is a control on the "
                          "PRIMARY (unstripped) path only" if a.rstrip else
                          "smoke run" if a.smoke else None)}
    if not a.smoke and not a.rstrip:
        e33 = json.load(open(os.path.join(HERE, "..", "results", "e33_logit_baseline_410m_v2.json")))
        for here_arm, there_arm in (("logit_I", "logit_I"), ("C0_ref_Github_n400_s0", "jlens_P")):
            for ag in ("min", "persist"):
                for s in EVAL_SETS:
                    mine = set_mean(ARMS[here_arm][ag], s)
                    theirs = e33["by_transport"][there_arm][s][ag]
                    c0["per_set"][f"{here_arm}|{ag}|{s}"] = {"this_run": mine, "e33_v2": theirs,
                                                             "abs_diff": abs(mine - theirs)}
                    c0["max_abs_diff"] = max(c0["max_abs_diff"], abs(mine - theirs))
        # TOLERANCE, derived rather than tuned. A set mean is (1/|K|) * mean over pairs of a
        # 0/1 indicator, so the SMALLEST change any genuine scoring difference can produce is
        # one pair flipping at one k: 1/(P_max * |K|) = 1/(394*7) = 3.6e-4, using the largest
        # set (multilingual, 394 pairs — a smaller set moves MORE, not less). Anything below
        # that cannot be a changed rank; it can only be float32 accumulation order, which
        # differs here because this scorer unembeds items in batches while t33 unembedded one
        # vector at a time. 1e-6 sits 2.6 orders below the smallest real difference and 1.7
        # orders above the observed float noise, so it distinguishes the two cleanly.
        # This tolerance is a property of THIS control, which is mine; it is not, and must not
        # be confused with, the pre-registered decision rule, which is untouched.
        smallest_real_diff = 1.0 / (max(len(SET_IDX[s]) for s in EVAL_SETS) * len(K))
        c0["fp32_tolerance"] = 1e-6
        c0["smallest_possible_real_difference"] = smallest_real_diff
        c0["tolerance_justification"] = (
            f"one pair flipping at one k moves a set mean by >= 1/({max(len(SET_IDX[s]) for s in EVAL_SETS)}*{len(K)}) "
            f"= {smallest_real_diff:.2e}; the tolerance is {smallest_real_diff/1e-6:.0f}x below that")
        assert smallest_real_diff > 100 * c0["fp32_tolerance"], smallest_real_diff
        c0["fires"] = c0["max_abs_diff"] < c0["fp32_tolerance"]
        print(f"\nC0 REPRODUCTION vs {c0['target']}: max|diff| = {c0['max_abs_diff']:.3e}  "
              f"(fp32 tolerance {c0['fp32_tolerance']:.0e}; smallest possible REAL difference "
              f"{smallest_real_diff:.2e})  ->  {'FIRES' if c0['fires'] else 'DOES NOT FIRE'}",
              flush=True)
        if not c0["fires"]:
            raise SystemExit(
                "C0 FAILED. This scoring path does not reproduce E33, so nothing it produces is "
                "comparable to E28/E33 and the pre-registered rule cannot be adjudicated on it. "
                "Fix the scorer; do not reinterpret the threshold.")

    # ---- C1 SELF-TEST
    run("C1_identity", {l: torch.eye(d) for l in BAND})
    c1 = float((ARMS["C1_identity"]["persist"] - ARMS["logit_I"]["persist"]).abs().max())
    print(f"C1 SELF-TEST identity-vs-logit max|per-pair diff| = {c1:.3e}  ->  "
          f"{'FIRES' if c1 == 0.0 else 'DOES NOT FIRE'}\n", flush=True)

    if a.controls_only:
        print("--controls-only: no rung was scored, no guarded outcome was read. "
              f"C0 {'FIRES' if c0.get('fires') else 'SKIPPED/FAILED'}, "
              f"C1 {'FIRES' if c1 == 0.0 else 'DOES NOT FIRE'}.")
        return 0 if (c0.get("fires") and c1 == 0.0) else 1

    # ---- every rung, every seed; then the derangement floor; then norm-matched random
    for c in corpora:
        for s in seeds:
            JP = load_J(lens_path(c, s))
            run(f"J|{c}|s{s}", JP)
            for dr in range(n_der):
                perm = derangement(BAND, 7000 + 97 * dr)
                run(f"shuf|{c}|s{s}|d{dr}", {l: JP[q] for l, q in zip(BAND, perm)})
        JP0 = load_J(lens_path(c, seeds[0]))
        g = torch.Generator().manual_seed(0)
        rnd = {}
        for l in BAND:
            M = torch.randn(JP0[l].shape, generator=g)
            rnd[l] = M * (JP0[l].norm() / M.norm())
        run(f"rand|{c}", rnd)

    # ---------------------------------------------------------------- adjudication
    print("\nbootstrapping...", flush=True)
    rungs = {}
    for c in corpora:
        rungs[c] = {}
        for ag in ("persist", "min"):
            jp = torch.stack([ARMS[f"J|{c}|s{s}"][ag] for s in seeds]).mean(0)
            sh = torch.stack([ARMS[f"shuf|{c}|s{s}|d{dr}"][ag]
                              for s in seeds for dr in range(n_der)]).mean(0)
            lg = ARMS["logit_I"][ag]
            # PRIMARY: three-level — eval set > seed block > item pair. The logit arm has no
            # seed axis (it is unfitted; that is the whole comparison), so it is the baseline.
            vs_logit = hierarchical_bootstrap_seeded(
                {s: ([ARMS[f"J|{c}|s{k}"][ag][SET_IDX[s]].tolist() for k in seeds],
                     lg[SET_IDX[s]].tolist()) for s in ADMITTED},
                n_boot=a.n_boot, seed=0)
            vs_shuf = hierarchical_bootstrap_seeded(
                {s: ([ARMS[f"J|{c}|s{k}"][ag][SET_IDX[s]].tolist() for k in seeds],
                     sh[SET_IDX[s]].tolist()) for s in ADMITTED},
                n_boot=a.n_boot, seed=0)
            # SECONDARY, reported alongside: the repo's existing two-level resampler on the
            # seed-AVERAGED difference. Discloses how much of the width is the seed component.
            vs_logit_2 = hierarchical_bootstrap(
                {s: (jp[SET_IDX[s]].tolist(), lg[SET_IDX[s]].tolist()) for s in ADMITTED},
                n_boot=a.n_boot, seed=0)
            rungs[c][ag] = {
                "hierarchical_ci_vs_logit_2level_seedavg": vs_logit_2,
                "ci_2level_strictly_positive": bool(vs_logit_2["ci_lo"] > 0),
                "ci_2level_strictly_negative": bool(vs_logit_2["ci_hi"] < 0),
                "jp_admitted_mean": admitted_mean(jp),
                "logit_admitted_mean": admitted_mean(lg),
                "shuf_admitted_mean": admitted_mean(sh),
                "rand_admitted_mean": admitted_mean(ARMS[f"rand|{c}"][ag]),
                "hierarchical_ci_vs_logit": vs_logit,
                "hierarchical_ci_vs_shuf": vs_shuf,
                "per_seed_admitted_mean": {f"s{s}": admitted_mean(ARMS[f"J|{c}|s{s}"][ag])
                                           for s in seeds},
                "per_derangement_admitted_mean": {
                    f"s{s}|d{dr}": admitted_mean(ARMS[f"shuf|{c}|s{s}|d{dr}"][ag])
                    for s in seeds for dr in range(n_der)},
                "per_set": {s: {"jp": set_mean(jp, s), "logit": set_mean(lg, s),
                                "shuf": set_mean(sh, s),
                                "paired_item_ci": paired_item_bootstrap(
                                    jp[SET_IDX[s]].tolist(), lg[SET_IDX[s]].tolist(),
                                    n_boot=a.n_boot_item, seed=0)}
                            for s in EVAL_SETS},
                "ci_strictly_positive": bool(vs_logit["ci_lo"] > 0),
                "ci_strictly_negative": bool(vs_logit["ci_hi"] < 0),
                "above_shuf_floor": bool(vs_shuf["ci_lo"] > 0),
                "above_shuf_point": bool(admitted_mean(jp) > admitted_mean(sh)),
            }
            print(f"  {c:22s} {ag:8s} d={vs_logit['point']:+.5f} "
                  f"[{vs_logit['ci_lo']:+.5f},{vs_logit['ci_hi']:+.5f}]  "
                  f"vs shuf {vs_shuf['point']:+.5f} "
                  f"[{vs_shuf['ci_lo']:+.5f},{vs_shuf['ci_hi']:+.5f}]", flush=True)

    PA = "persist"                                   # the registered adjudication aggregation
    have = set(corpora)
    clause_a = {c: rungs[c][PA]["ci_strictly_positive"] for c in INSTREAM_A if c in have}
    clause_b_neg = [c for c in OOD if c in have and rungs[c][PA]["ci_strictly_negative"]]
    clause_c = {c: rungs[c][PA]["above_shuf_floor"] for c in OOD if c in have}
    degenerate = [c for c in OOD if c in have and not rungs[c][PA]["above_shuf_point"]]
    gh_neg = rungs[GITHUB][PA]["ci_strictly_negative"] if GITHUB in have else None

    a_ok = bool(clause_a) and all(clause_a.values())
    b_ok = len(clause_b_neg) >= 2
    c_ok = bool(clause_b_neg) and all(clause_c[c] for c in clause_b_neg)
    d_ok = (NEWS in clause_b_neg) and (gh_neg is False)
    all_ood_nonneg = (not clause_b_neg) and all(c in have for c in OOD)

    if b_ok and gh_neg:
        verdict = ("REGISTER-CONFOUNDED — >=2 OOD rungs go strictly negative BUT Github "
                   "(register-far, exposure-NEAR) also goes strictly negative. Register distance "
                   "alone reaches the crossover, so the exposure attribution is not identified. "
                   "Amendment 1, clause (d).")
    elif a_ok and b_ok and c_ok and d_ok:
        verdict = ("CONFIRMED — the unfitted logit lens OVERTAKES the fitted J-lens under measured "
                   "distribution shift, on rungs whose operators are structured (above the "
                   "derangement floor) and whose model competence passed the gate. S3's founding "
                   "prediction is measured.")
    elif all_ood_nonneg:
        verdict = ("NOT REACHED — (J^P - logit) includes or exceeds 0 on ALL three OOD rungs. "
                   "Reported straight as a positive-framed negative: the bias term's practical "
                   "magnitude is below the estimator worst case; the fitted operator tolerates "
                   "shift the theory permits it to fail on. Publishable exactly as pre-registered "
                   "and MUST NOT be re-described as a weaker crossover.")
    else:
        verdict = ("UNCLEAR — the outcome falls in a gap the pre-registered rule does not name "
                   f"(clause-a all-positive={a_ok} {clause_a}; OOD strictly-negative="
                   f"{clause_b_neg}; above-shuf={clause_c}; News-negative={NEWS in clause_b_neg}; "
                   f"Github-strictly-negative={gh_neg}). STOP. The rule is not reinterpretable "
                   "after a result — operator decision required.")
    # ---- PRE-DECLARED SENSITIVITY, written here BEFORE any OOD read existed.
    # OOD_CommonPile's eligibility as an OOD rung is CONTESTED on evidence gathered while
    # preparing to score (see e48b_exposure_growth.json and this run's `contested_rungs`):
    # all 2400 documents are arXiv abstracts (2-space indent, 79-col wrap, LaTeX escapes,
    # 98.4% scientific vocabulary), and arXiv is a Pile component at ~9% weight, so the
    # manifest's claim that it covers "domains not deliberately represented in the Pile" is
    # false as sampled. Its containment IS below the formatting floor, so absence is not
    # refuted — but it is not established either, and two of the three "register-diverse"
    # OOD rungs are in fact both arXiv science.
    # This does NOT change the registered rule. The registered clause (b) over all three rungs
    # is the primary and is reported first. This is a disclosed secondary computed over
    # {News, arXiv} only, so that the operator can see whether the verdict is even sensitive to
    # CommonPile's status. Declaring it blind is what keeps it from being a degree of freedom.
    OOD_UNCONTESTED = [c for c in OOD if c != "OOD_CommonPile"]
    b_neg_unc = [c for c in OOD_UNCONTESTED if c in clause_b_neg]
    sensitivity = {
        "why": ("OOD_CommonPile is 100% arXiv abstracts and arXiv is a Pile component; its OOD "
                "status is contested. Declared BEFORE any read was scored."),
        "ood_set_used": OOD_UNCONTESTED,
        "b_ok_needs_both": len(b_neg_unc) >= 2,
        "b_strictly_negative": b_neg_unc,
        "verdict_would_change": (len(clause_b_neg) >= 2) != (len(b_neg_unc) >= 2),
    }

    # do the PRIMARY (3-level) and SECONDARY (2-level, seed-averaged) intervals agree per clause?
    ci_disagree = [c for c in corpora
                   if (rungs[c][PA]["ci_strictly_positive"]
                       != rungs[c][PA]["ci_2level_strictly_positive"])
                   or (rungs[c][PA]["ci_strictly_negative"]
                       != rungs[c][PA]["ci_2level_strictly_negative"])]
    if degenerate:
        verdict += (f"  ||  DEGENERACY: {degenerate} read AT OR BELOW their own derangement floor. "
                    f"Excluded from the crossover claim in either direction — operator collapse, "
                    f"not distributional bias.")
    if ci_disagree:
        verdict += (f"  ||  FLAG: the three-level CI (primary, as pre-committed) and the "
                    f"two-level seed-averaged CI reach DIFFERENT clause outcomes on "
                    f"{ci_disagree}. Not resolved here — both are in the record; operator "
                    f"decision on which the paper reports.")
    if gate_failed:
        verdict = f"GATE-FAILED ({gate_failed}) — " + verdict

    rec = {
        "experiment": "E48 — the S3 crossover: does the unfitted logit lens overtake J^P under shift?",
        "prereg": "docs/experiments/preregs/superseded/PREREG_E48_CROSSOVER.md + Amendment 1, commits 408fa5c / 5869b63",
        "status": "PRE-REGISTERED — rule fixed before any OOD lens existed",
        "rule_verbatim": RULE,
        "model": MODEL, "band": BAND, "device": a.device, "K": K,
        "n_items": n_items, "n_pairs": P_n, "smoke": a.smoke,
        "admitted_sets": ADMITTED,
        "excluded_sets": {"association": "floored — 196/205 ladder cells exactly zero (E28 crit 3)"},
        "N_per_fit": 200, "seeds": seeds, "n_derangements_per_cell": n_der,
        "adjudication_aggregation": PA,
        "ci_primary": ("three-level: eval set > seed block > item pair "
                       "(bootstrap.hierarchical_bootstrap_seeded). Chosen and written into this "
                       "script BEFORE any OOD read was scored, because the pre-registration "
                       "names both units and the repo's two-level resampler has no seed level."),
        "ci_secondary": ("two-level on the seed-averaged difference "
                         "(bootstrap.hierarchical_bootstrap) — the repo's existing resampler, "
                         "reported so the seed variance component is visible"),
        "ci_clause_disagreement": ci_disagree,
        "sensitivity_drop_contested_rung": sensitivity,
        "contested_rungs": {
            "OOD_CommonPile": ("100% arXiv abstracts (2400/2400 start with two spaces, 79-col "
                               "hard wrap, LaTeX escapes, 98.4% scientific vocabulary). arXiv is "
                               "a Pile component at ~9% weight, so the manifest's 'domains not "
                               "deliberately represented in the Pile' is false as sampled. k=32 "
                               "containment 0.0002 IS below the 0.0014 single-line-reflow floor, "
                               "so absence is not refuted — but the rung is not register-"
                               "independent of OOD_arXiv_2023, which is what clause (b)'s "
                               "'>=2 of 3' was meant to guarantee."),
            "OOD_News_2024": ("33.4% exact duplicates (2400 -> 1598 distinct), seed-asymmetric: "
                              "the fitted 200 contain 199/166/161 distinct documents at seeds "
                              "0/1/2. This is the clause-(d) load-bearing rung. A deduplicated "
                              "refit is a separate disclosed robustness arm."),
        },
        "rstrip_arm": bool(a.rstrip),
        "scoring_path_forks_disclosed": {
            "rstrip": ("PRIMARY is UNSTRIPPED (matches trainval/t33/E28/E33, and is the only "
                       "path that can reproduce C0). anchor_evals.py:32-34 calls .rstrip() "
                       "load-bearing; it changes the readout token on 157/551 items. The "
                       "stripped path is run separately via --rstrip."),
            "passk_pooling": ("both this scorer and t22 flat-pool over (item, intermediate) "
                              "pairs; the anchor README defines a mean over items of the "
                              "per-item fraction. Inherited unchanged so C0 can fire."),
        },
        "competence_gate": ({"file": os.path.basename(a.gate),
                             "verdict": gate.get("VERDICT", "INCOMPLETE"),
                             "failed_rungs": gate_failed,
                             "ce_by_corpus": {c: v["ce_mean"]
                                              for c, v in gate.get("by_corpus", {}).items()}}
                            if gate else None),
        "controls": {
            "C0_reproduces_e33_v2": c0,
            "C1_identity_equals_logit": {"max_abs_per_pair_diff": c1, "fires": c1 == 0.0},
            "C2_shuf_floor": {c: {"jp": rungs[c][PA]["jp_admitted_mean"],
                                  "shuf_mean_over_draws": rungs[c][PA]["shuf_admitted_mean"],
                                  "ci_vs_shuf": rungs[c][PA]["hierarchical_ci_vs_shuf"],
                                  "fires": rungs[c][PA]["above_shuf_floor"]} for c in corpora},
            "C3_norm_matched_random": {c: rungs[c][PA]["rand_admitted_mean"] for c in corpora},
        },
        "clauses": {"a_instream_ci_positive": clause_a, "a_ok": a_ok,
                    "b_ood_ci_strictly_negative": clause_b_neg, "b_ok": b_ok,
                    "c_above_shuf_floor": clause_c, "c_ok": c_ok,
                    "d_news_among_negatives": NEWS in clause_b_neg,
                    "d_github_strictly_negative": gh_neg, "d_ok": d_ok},
        "rungs": rungs,
        "arms_admitted_mean": {k: {ag: admitted_mean(v[ag]) for ag in ("min", "persist")}
                               for k, v in ARMS.items()},
        "VERDICT": verdict,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write_result(a.out, rec, experiment="E48",
                 inputs=[a.gate] + [lens_path(c, s) for c in corpora for s in seeds]
                        + [os.path.join(HERE, "..", "results", "e33_logit_baseline_410m_v2.json")])

    print(f"\n{'rung':24s} {'tier':10s} {'J^P':>9} {'logit':>9} {'delta':>10} "
          f"{'CI lo':>10} {'CI hi':>10} {'shuf':>9}")
    for c in corpora:
        r = rungs[c][PA]
        tier = ("in-stream" if c in INSTREAM_A else "github" if c == GITHUB else "OOD")
        h = r["hierarchical_ci_vs_logit"]
        print(f"{c:24s} {tier:10s} {r['jp_admitted_mean']:9.5f} {r['logit_admitted_mean']:9.5f} "
              f"{h['point']:+10.5f} {h['ci_lo']:+10.5f} {h['ci_hi']:+10.5f} "
              f"{r['shuf_admitted_mean']:9.5f}")
    print(f"\nclause (a) in-stream all strictly > 0 : {a_ok}   {clause_a}")
    print(f"clause (b) OOD strictly < 0           : {clause_b_neg}  (need >=2)")
    print(f"clause (c) OOD above shuf floor       : {clause_c}")
    print(f"clause (d) News negative={NEWS in clause_b_neg}  Github strictly negative={gh_neg}")
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
