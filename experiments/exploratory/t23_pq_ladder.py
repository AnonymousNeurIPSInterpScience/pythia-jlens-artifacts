#!/usr/bin/env python3
"""t23_pq_ladder.py — E27. Is the distributional BIAS term larger than the sampling floor?

WHY THIS IS THE EXPERIMENT S3 HAS BEEN BLOCKED ON. The programme rests on the decomposition

    ||Jhat_P - J_Q||  <=  ||Jhat_P - J_P||   +   ||J_P - J_Q||
                          sampling error         distributional bias
                          shrinks as 1/sqrt(n)   DOES NOT shrink with n

E18 measured the FIRST term and validated its 1/sqrt(n) law. **Nothing has ever measured the
second.** If the bias term is at or below the sampling floor, the J-lens is corpus-INVARIANT and
the paper becomes the mechanistic explanation of why an averaged derivative is robust. If it
dominates, the J-lens is corpus-VARIANT and the crossover with the unfitted baseline exists.
Both outcomes are publishable; we do not chase either.

THE DESIGN, and why it is fair. Fitting on two corpora and comparing gives a difference that
CONFOUNDS bias with sampling noise. So every operator here is fitted at the SAME n on a HALF of
its corpus, and we compare:

    WITHIN-corpus   D_act(J_halfA(c),  J_halfB(c))    <- pure sampling floor, no bias
    BETWEEN-corpus  D_act(J_halfA(c1), J_halfA(c2))   <- sampling + bias, SAME n

The ratio between-over-within is then a bias-to-noise ratio with the sampling contribution
matched by construction. This is the equal-n comparison Prompt 7 sec2.4 asks for.

D_act is used rather than Frobenius because Frobenius counts disagreement in directions no
activation ever visits (metrics_shift.m4_action_divergence; a synthetic check gives D_act = 0
where Frobenius = 565.7). Activations come from the EVAL prompts, i.e. the data the lens will
actually act on.

================================ PRE-REGISTRATION ================================
Fixed before the first number.

PRIMARY OUTCOME
  R = median over band layers of  D_act(between-corpus) / D_act(within-corpus, matched n).

DECISION RULE
  R >= 3.0        -> CORPUS-VARIANT. The bias term dominates the sampling floor; S3's premise
                     holds and the crossover experiment (E30) is worth its compute.
  1.5 <= R < 3.0  -> WEAKLY VARIANT. Bias is present but comparable to noise; E30 needs a
                     larger n to separate them, and that cost must be re-priced before running.
  R < 1.5         -> CORPUS-INVARIANT. Report as a NULL RESULT WITH A MECHANISM: the averaged
                     Jacobian is robust to the fitting corpus, and sec(iii) below reports WHY
                     (which spectral component absorbs the corpus change).

CONTROLS, each with the number it must produce to count as firing
  C1 SELF-DISTANCE. D_act(J, J) must be exactly 0. Any drift means the estimator is wrong.
  C2 SHUFFLE FLOOR. D_act(J_halfA, J^shuf_halfA) -- the distance to a deliberately broken
     operator on the SAME corpus. Between-corpus distance should be far BELOW this, else
     "different corpus" is as damaging as "destroyed operator" and the scale is uninterpretable.
  C3 NORMALISED FORM. D_act / E||J_P h||^2 is reported alongside the raw value, because the raw
     value tracks activation norm and therefore depth.

DECLARED BIAS
  Small n per half. 70m first, where the E18 law says n=92 gives ~5% operator error at eps=5%;
  the halves are therefore ABOVE that error and the within-corpus floor is CONSERVATIVE (too
  large), which biases R DOWNWARD -- i.e. against finding variance. Stated so a null is not
  over-read.
  70m is also the model where sec F10 shows the unembedding is nearly rank-1; a corpus effect on
  the OPERATOR is still well-defined there, but its behavioural consequence may not be.
==================================================================================
"""
from __future__ import annotations

import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "jacobian-lens"))

from jlens.hf import Layout, from_hf                                      # noqa: E402
from jlens.lens import JacobianLens                                       # noqa: E402
from anchor_evals import EVAL_SETS, load_eval                             # noqa: E402
from t13_transport_controls import _derangement                           # noqa: E402
from t17_reaggregate import band_of                                       # noqa: E402
from metrics_shift import m4_action_divergence                            # noqa: E402

PYTHIA_LAYOUT = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                       embed="embed_in", lm_head="lm_head")


@torch.no_grad()
def eval_activations(model, layers, n_prompts=60, max_len=128):
    """h_l(x) for x drawn from the EVAL sets — the data D_act weights the operator difference by.

    Deliberately NOT drawn from any fitting corpus: D_act must be evaluated on the distribution
    the lens is used on, not on either corpus being compared.
    """
    prompts = []
    for ev in EVAL_SETS:
        for it in load_eval(ev):
            prompts.append(it["prompt"].rstrip())
    prompts = prompts[::max(1, len(prompts) // n_prompts)][:n_prompts]
    from jlens.hooks import ActivationRecorder
    acc = {l: [] for l in layers}
    for p in prompts:
        ids = model.encode(p, max_length=max_len)
        with ActivationRecorder(model.layers, at=list(layers)) as rec:
            model.forward(ids)
            for l in layers:
                acc[l].append(rec.activations[l][0].detach()[-1].float().cpu())
    return {l: torch.stack(v) for l, v in acc.items()}, len(prompts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens-dir", default="results")
    ap.add_argument("--corpora", default="wikitext,pile,code")
    ap.add_argument("--tag", required=True, help="filename tag, e.g. 70m")
    ap.add_argument("--n-eval-prompts", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    try:
        model = from_hf(hf, tok)
    except ValueError:
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT)

    corpora = a.corpora.split(",")
    lenses = {}
    for c in corpora:
        for half in ("A", "B"):
            p = os.path.join(a.lens_dir, f"plad_{a.tag}_{c}_{half}.pt")
            if not os.path.exists(p):
                print(f"MISSING {p} — fit it first"); return 2
            lenses[(c, half)] = JacobianLens.load(p)

    layers = list(lenses[(corpora[0], "A")].source_layers)
    band = band_of(model.n_layers, layers)
    H, n_eval = eval_activations(model, band, a.n_eval_prompts)
    print(f"model={a.model}  band={band}  eval activations from {n_eval} prompts")

    def J(c, half, l):
        return lenses[(c, half)].jacobians[l].float()

    # ---- C1 self-distance
    c1 = max(m4_action_divergence(J(corpora[0], "A", l), J(corpora[0], "A", l), H[l])["d_act"]
             for l in band)

    within, between, shuffle_floor = {}, {}, {}
    perm = _derangement(layers, a.seed)
    for c in corpora:
        w = [m4_action_divergence(J(c, "A", l), J(c, "B", l), H[l]) for l in band]
        within[c] = {"d_act": [x["d_act"] for x in w],
                     "d_act_norm": [x["d_act_normalised"] for x in w],
                     "median": float(torch.tensor([x["d_act"] for x in w]).median())}
        # C2: distance to a DERANGED operator on the same corpus, same half
        sf = [m4_action_divergence(J(c, "A", l), J(c, "A", perm[l]), H[l])["d_act"] for l in band]
        shuffle_floor[c] = float(torch.tensor(sf).median())

    for i, c1n in enumerate(corpora):
        for c2n in corpora[i + 1:]:
            b = [m4_action_divergence(J(c1n, "A", l), J(c2n, "A", l), H[l]) for l in band]
            between[f"{c1n}|{c2n}"] = {
                "d_act": [x["d_act"] for x in b],
                "d_act_norm": [x["d_act_normalised"] for x in b],
                "median": float(torch.tensor([x["d_act"] for x in b]).median())}

    # PRIMARY: median over band layers of between/within, pooled over corpus pairs
    ratios = []
    for key, bv in between.items():
        c1n, c2n = key.split("|")
        floor = [(within[c1n]["d_act"][i] + within[c2n]["d_act"][i]) / 2
                 for i in range(len(band))]
        ratios.append([bv["d_act"][i] / max(floor[i], 1e-30) for i in range(len(band))])
    flat = torch.tensor([r for row in ratios for r in row])
    R = float(flat.median())

    if R >= 3.0:
        verdict = (f"CORPUS-VARIANT — R={R:.2f} >= 3.0; the distributional bias term dominates "
                   "the sampling floor, S3's premise holds")
    elif R >= 1.5:
        verdict = (f"WEAKLY VARIANT — R={R:.2f} in [1.5,3.0); bias present but comparable to "
                   "noise, E30 needs a larger n and must be re-priced")
    else:
        verdict = (f"CORPUS-INVARIANT — R={R:.2f} < 1.5; the averaged Jacobian is robust to the "
                   "fitting corpus. Report as a NULL WITH A MECHANISM")

    out = {
        "experiment": "T23 / E27 — P-ladder: distributional bias vs the sampling floor",
        "status": "EXPLORATORY", "prereg": "in-file header, fixed before the first number",
        "decision_rule": ("R = median over band layers of D_act(between)/D_act(within, matched n); "
                          ">=3.0 variant / [1.5,3.0) weakly variant / <1.5 invariant"),
        "primary_outcome": "R", "VERDICT": verdict, "R": R,
        "R_per_pair": {k: float(torch.tensor(ratios[i]).median())
                       for i, k in enumerate(between)},
        "C1_self_distance_max": c1, "C1_fires": c1 < 1e-10,
        "C2_shuffle_floor_median": shuffle_floor,
        "C2_between_over_shuffle": {k: between[k]["median"] /
                                    max(min(shuffle_floor.values()), 1e-30) for k in between},
        "within_corpus": within, "between_corpus": between,
        "model": a.model, "corpora": corpora, "band": band, "n_eval_prompts": n_eval,
        "seed": a.seed, "derangement_layer_map": {str(k): v for k, v in perm.items()},
        "env": {"torch": torch.__version__}, "argv": sys.argv,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"  C1 self-distance max = {c1:.3e}  ({'FIRES' if out['C1_fires'] else 'FAILED'})")
    for c in corpora:
        print(f"  within  {c:<10} median D_act = {within[c]['median']:.5e}   "
              f"C2 shuffle floor = {shuffle_floor[c]:.5e}")
    for k, v in between.items():
        print(f"  between {k:<18} median D_act = {v['median']:.5e}   "
              f"ratio to shuffle = {out['C2_between_over_shuffle'][k]:.4f}")
    print(f"\n  R (between/within, median over band) = {R:.3f}")
    print(f"  VERDICT: {verdict}")
    print(f"wrote {a.out} ({out['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
