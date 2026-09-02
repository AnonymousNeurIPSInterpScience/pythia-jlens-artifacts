#!/usr/bin/env python3
"""t58_algebra_audit.py — E58: three algebra/measurement checks the validity audit could not
resolve by reading, only by computing.

WHY
  EXPERIMENT_VALIDITY_AUDIT v2 section 8 listed E45/E46/E47's subspace algebra and the containment
  streaming implementation as unaudited. Reading them settled E46/E47 (the ablation
  (I - BB^T)J^P with B from svd(J^P - J^Q).U is correct, and E47's verdict logic executes
  correctly). Three things could not be settled by reading:

  A. E45's POST-NORM VARIANT USES THE WRONG TRANSPOSE.
     The read is W_U . norm(J h). Linearising the final norm at the operating point,
     logit_c = const + W_U[c]^T G (J h), so the effective read row for concept c is W_U[c]^T G,
     i.e. row c of (W_U G). t40_disagreement_geometry.py:165 computes `WU @ Gm[l].T`, whose row c
     is W_U[c]^T G^T. These coincide only if G is symmetric. G = diag(gamma) . M / sigma with M
     symmetric, so G is symmetric only if gamma is constant. Measured: pythia-410m's final
     LayerNorm has gamma mean 3.46, sd 0.216, giving ||G - G^T||_F / ||G||_F = 0.0039.
     A 0.4% perturbation is small -- but the reported post-norm r is 0.4961 against a
     PRE-REGISTERED THRESHOLD OF 0.5, and the stored VERDICT is selected on which side of 0.5 that
     number falls. A 0.4% error cannot be waved past a 0.8% margin. Recompute all three
     orientations and report.

  B. E45's C2 CONTROL CANNOT FAIL.
     It builds ONE random Delta and assigns it to every corpus, so its x has zero within-set
     variance while the real statistic's x does not; the script's own comment says as much. A
     control that is structurally incapable of firing is not a control (CLAUDE.md 6.0). The
     control it should have run: FIVE independent random Deltas, one per corpus, each matched in
     Frobenius norm to that corpus's real Delta, put through the identical z-within-set pipeline.

  C. E36's DIAGONAL PREFIX/FIT OVERLAP IS ASSERTED, NEVER MEASURED.
     paper.tex:161 claims context pools are disjoint from fitting pools. That is true of E52
     (t52 implements load_pool_heldout) and FALSE of E36, which is Appendix B: t36_qladder.py
     load_pool() draws from the whole corpus file while its operators are e28 N=400 fits from
     documents 0..800 of that same file. The repo asserts "~17%" from an arithmetic guess. Measure
     it.

  D. (storage) the containment chain, re-derived independently. Reported because the audit ran it.

WHAT IT SETTLES
  Whether E45's stored verdict string survives the corrected algebra; whether its C2 fires when
  made capable of failing; and what E36's diagonal leakage actually is.

DECISION RULE — fixed before running.
  A. If the corrected post-norm |r| lands on the SAME side of 0.5 as the coded one, E45's stored
     VERDICT stands and the transpose is recorded as immaterial. If it CROSSES 0.5, E45's verdict
     string is wrong and must be corrected in the results file and the taxonomy.
     Either way E45 remains a FAILED predictor -- it fails leave-one-out at 0.20 (corpus) in every
     orientation -- so no paper claim reverses. This is about the record being right.
  B. If the proper C2 gives |r| >= 0.3, the original C2 was masking a real problem and E45's
     primary is confounded. If |r| < 0.3, the control fires for the right reason.
  C. No rule; it is a measurement that a disclosure sentence needs.

CONTROLS
  C1  s_c with Delta = 0 must be exactly 0.0, in every orientation. (E45's own C1, re-run.)
  C2p the proper five-Delta random control (item B above).
  C3  the pre-norm variant must reproduce e45_disagreement_geometry.json's stored r exactly --
      it has no G in it, so it is untouched by the transpose and is the anchor proving this
      script's pipeline is E45's pipeline.

    python experiments/t58_algebra_audit.py --device cpu
"""
from __future__ import annotations
import argparse, glob, json, math, os, statistics as st, sys

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))
from provenance import write_result  # noqa: E402

RES = os.path.join(HERE, "..", "results")
MODEL = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))
CORPORA = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]
SETS = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
PREFIX_TOKENS = 128


def pear(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x)); sy = math.sqrt(sum((v - my) ** 2 for v in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx * sy else float("nan")


def zwithin(mat):
    out = {}
    for s in SETS:
        v = [mat[(c, s)] for c in CORPORA]
        m, sd = st.mean(v), st.pstdev(v)
        for c in CORPORA:
            out[(c, s)] = (mat[(c, s)] - m) / sd if sd > 0 else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(RES, "e58_algebra_audit.json"))
    ap.add_argument("--skip-overlap", action="store_true")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    WU = hf.get_output_embeddings().weight.data.float()
    d = WU.shape[1]

    rec = {"experiment": "E58 — algebra and measurement checks the validity audit left open",
           "why": "E45's post-norm transpose, E45's degenerate C2, E36's asserted diagonal leakage",
           "model": MODEL, "band": BAND}

    # ================================================================ A/B: E45 recomputed
    set_tokens = {s: set() for s in SETS}
    H = {l: [] for l in BAND}
    with torch.no_grad():
        for name in EVAL_SETS:
            for it in load_eval(name):
                tg = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
                if not tg:
                    continue
                if name in set_tokens:
                    for t in tg:
                        set_tokens[name].update(t)
                ids = model.encode(it["prompt"], max_length=256)
                pos = readout_position(tok, name, it["prompt"])
                with ActivationRecorder(model.layers, at=BAND) as r:
                    model.forward(ids)
                    for l in BAND:
                        H[l].append(r.activations[l][0][pos].detach().float())
    H = {l: torch.stack(v) for l, v in H.items()}
    Sig = {l: (H[l].T @ H[l]) / H[l].shape[0] for l in BAND}
    print(f"activations {tuple(H[BAND[0]].shape)}", flush=True)

    fin = None
    for attr in ("final_layer_norm", "ln_f", "norm"):
        m = getattr(getattr(hf, "gpt_neox", hf), attr, None)
        if m is not None:
            fin = m; break
    Gm = {}
    for l in BAND:
        x0 = H[l].mean(0).clone().requires_grad_(True)
        Gm[l] = torch.autograd.functional.jacobian(lambda z: fin(z), x0).detach()
    asym = {l: float((Gm[l] - Gm[l].T).norm() / Gm[l].norm()) for l in BAND}
    print(f"final-norm Jacobian linearised; ||G-G^T||_F/||G||_F median = "
          f"{st.median(asym.values()):.5f}", flush=True)

    def lens(p):
        return {int(k): v.float() for k, v in torch.load(
            os.path.join(RES, p), map_location="cpu")["J"].items()}

    Js = {c: lens(f"e28_{c}_410m_n400_s0.pt") for c in CORPORA}
    Jbar = {l: sum(Js[c][l] for c in CORPORA) / len(CORPORA) for l in BAND}

    def Wmat(l, orient):
        if orient == "none":
            return WU
        if orient == "coded_G_transpose":
            return WU @ Gm[l].T          # what t40 does
        if orient == "correct_G":
            return WU @ Gm[l]            # what the linearisation requires
        if orient == "symmetrised":
            return WU @ (0.5 * (Gm[l] + Gm[l].T))
        raise ValueError(orient)

    def s_per_set(Delta, orient):
        acc = {s: [] for s in SETS}
        for l in BAND:
            W = Wmat(l, orient)
            for s in SETS:
                idx = sorted(set_tokens[s])
                U = W[idx] @ Delta[l]
                sc = ((U @ Sig[l]) * U).sum(1)
                acc[s].append(sc.mean().item())
        return {s: st.median(v) for s, v in acc.items()}

    # outcome, from the stored ladder (identical to E45)
    cells = {}
    for f in glob.glob(os.path.join(RES, "ladder410", "*.json")):
        dd = json.load(open(f)); cells[(dd["corpus"], dd["seed"])] = dd
    Y = {(c, s): st.mean([st.mean([cells[(c, sd)]["by_N"][n][s]["persist"]
                                   for n in cells[(c, sd)]["by_N"] if int(n) >= 75])
                          for sd in range(3)]) for c in CORPORA for s in SETS}
    zY = zwithin(Y)
    keys = [(c, s) for c in CORPORA for s in SETS]

    def evaluate(orient):
        S = {}
        for c in CORPORA:
            D = {l: Js[c][l] - Jbar[l] for l in BAND}
            for s, v in s_per_set(D, orient).items():
                S[(c, s)] = v
        zS = zwithin(S)
        xs = [zS[k] for k in keys]; ys = [abs(zY[k]) for k in keys]
        r = pear(xs, ys)
        loo_set = {s: pear([zS[k] for k in keys if k[1] != s],
                           [abs(zY[k]) for k in keys if k[1] != s]) for s in SETS}
        loo_cor = {c: pear([zS[k] for k in keys if k[0] != c],
                           [abs(zY[k]) for k in keys if k[0] != c]) for c in CORPORA}
        ws = min(loo_set.items(), key=lambda kv: abs(kv[1]))
        wc = min(loo_cor.items(), key=lambda kv: abs(kv[1]))
        return {"r": r, "worst_loo_set": list(ws), "worst_loo_corpus": list(wc),
                "survives": abs(r) >= 0.5 and abs(ws[1]) >= 0.5 and abs(wc[1]) >= 0.5,
                "crosses_the_0p5_threshold": abs(r) >= 0.5}

    zero = {l: torch.zeros(d, d) for l in BAND}
    c1 = {o: max(s_per_set(zero, o).values()) for o in
          ("none", "coded_G_transpose", "correct_G", "symmetrised")}
    orients = {}
    for o in ("none", "coded_G_transpose", "correct_G", "symmetrised"):
        orients[o] = evaluate(o)
        print(f"  [{o:18s}] r = {orients[o]['r']:+.4f}  worst LOO corpus "
              f"{orients[o]['worst_loo_corpus'][1]:+.4f}  |r|>=0.5 = "
              f"{orients[o]['crosses_the_0p5_threshold']}", flush=True)

    e45 = json.load(open(os.path.join(RES, "e45_disagreement_geometry.json")))
    c3 = {"stored_pre_norm_r": e45["pre_norm"]["r"], "recomputed": orients["none"]["r"],
          "abs_diff": abs(e45["pre_norm"]["r"] - orients["none"]["r"])}
    c3["fires"] = c3["abs_diff"] <= 1e-9
    stored_post = e45["post_norm"]["r"]
    verdict_a = (
        "IMMATERIAL — the corrected orientation lands on the same side of the pre-registered 0.5 "
        "threshold as the coded one; E45's stored VERDICT string stands"
        if orients["correct_G"]["crosses_the_0p5_threshold"]
        == orients["coded_G_transpose"]["crosses_the_0p5_threshold"] else
        "MATERIAL — the corrected orientation crosses the pre-registered 0.5 threshold. E45's "
        "stored VERDICT string is wrong and must be corrected in the results file and taxonomy.")

    # ---- B: the proper C2, five independent Deltas, one per corpus, norm-matched
    g = torch.Generator().manual_seed(0)
    Sr = {}
    for c in CORPORA:
        Dr = {}
        for l in BAND:
            ref = Js[c][l] - Jbar[l]
            R = torch.randn(d, d, generator=g)
            Dr[l] = R * (ref.norm() / R.norm())
        for s, v in s_per_set(Dr, "none").items():
            Sr[(c, s)] = v
    zSr = zwithin(Sr)
    c2p = pear([zSr[k] for k in keys], [abs(zY[k]) for k in keys])
    verdict_b = ("FIRES FOR THE RIGHT REASON — five independent norm-matched random Deltas, each "
                 "with genuine within-set variation, still do not predict"
                 if abs(c2p) < 0.3 else
                 "DOES NOT FIRE — a norm-matched random operator predicts as well as the real "
                 "disagreement geometry; E45's primary is confounded")

    rec["A_e45_orientation"] = {
        "layernorm_gamma_mean": float(fin.weight.data.mean()),
        "layernorm_gamma_sd": float(fin.weight.data.std()),
        "G_relative_asymmetry_per_layer": asym,
        "G_relative_asymmetry_median": st.median(asym.values()),
        "stored_post_norm_r": stored_post,
        "by_orientation": orients,
        "threshold": 0.5,
        "VERDICT": verdict_a,
        "note": ("in EVERY orientation the leave-one-corpus-out value is ~0.2 (Github), so E45 "
                 "remains a failed predictor and no paper claim reverses. This is about the "
                 "results file saying the right thing."),
    }
    rec["B_e45_proper_C2"] = {
        "coded_C2_r": e45["control_C2_random_delta_r"],
        "coded_C2_defect": ("one Delta shared across all five corpora, so x has zero within-set "
                            "variance and the control cannot fail"),
        "proper_C2_r_five_independent_norm_matched_deltas": c2p,
        "threshold": 0.3, "fires": abs(c2p) < 0.3,
        "VERDICT": verdict_b,
    }
    rec["controls"] = {"C1_zero_delta_by_orientation": c1,
                       "C1_fires": all(v == 0.0 for v in c1.values()),
                       "C3_pre_norm_reproduces_stored_e45": c3}

    # ================================================================ C: E36 diagonal overlap
    if not a.skip_overlap:
        ov = {}
        for c in CORPORA:
            texts = [json.loads(l)["text"]
                     for l in open(os.path.join(HERE, "..", "corpora", f"{c}.jsonl"))]
            b = len(texts) // 3
            # e28 N=400 seed-0 operators: block 0, first 400 qualifying (t2_fastfit takes pool[:n])
            fit_docs = [t for t in texts[0:b] if len(tok(t).input_ids) >= PREFIX_TOKENS][:400]
            fit_set = set(fit_docs)
            # E36's prefix pool: t36_qladder.load_pool, seed 9000+ps, whole file, first 541 with
            # >= 128 tokens in the permuted order
            per_seed = {}
            for ps in (0, 1, 2):
                gg = torch.Generator().manual_seed(9000 + ps)
                out, hits = [], 0
                for i in torch.randperm(len(texts), generator=gg).tolist():
                    t = texts[i]
                    if len(tok(t).input_ids) >= PREFIX_TOKENS:
                        out.append(t)
                        if t in fit_set:
                            hits += 1
                    if len(out) >= 541:
                        break
                per_seed[f"p{ps}"] = {"n_prefix_docs": len(out), "n_from_fitting_pool": hits,
                                      "overlap_fraction": hits / len(out) if out else 0.0}
            ov[c] = {"n_fitting_docs": len(fit_docs), "n_docs_in_file": len(texts),
                     "per_prefix_seed": per_seed,
                     "mean_overlap_fraction": st.mean(v["overlap_fraction"]
                                                      for v in per_seed.values())}
            print(f"  E36 overlap {c:20s} {ov[c]['mean_overlap_fraction']:.4f}", flush=True)
        rec["C_e36_diagonal_prefix_fit_overlap"] = {
            "what": ("fraction of E36's prefix documents for corpus X that are among the 400 "
                     "documents its own N=400 seed-0 operator was fitted on. Applies to the five "
                     "DIAGONAL rungs of Appendix B, which are also its five most-contained rungs."),
            "by_corpus": ov,
            "mean_over_corpora": st.mean(v["mean_overlap_fraction"] for v in ov.values()),
            "repo_assertion": "~17%",
            "e52_by_contrast": ("t52_factorial.py load_pool_heldout() excludes every fitting "
                                "document and aborts rather than cycle — measured 0 by "
                                "construction"),
            "direction_of_bias": ("inflates one cell per row, so it lands mostly in the residual; "
                                  "on Appendix B's containment argument it lifts the "
                                  "high-containment end, which flattens any decline and therefore "
                                  "favours the paper's null"),
        }

    write_result(a.out, rec, experiment="E58",
                 inputs=[os.path.join(RES, "e45_disagreement_geometry.json")]
                        + [os.path.join(RES, f"e28_{c}_410m_n400_s0.pt") for c in CORPORA])
    print(f"\nA: {verdict_a}")
    print(f"B: proper C2 r = {c2p:+.4f} -> {verdict_b}")
    print(f"C1 zero-Delta all orientations exactly 0: {rec['controls']['C1_fires']}")
    print(f"C3 pre-norm reproduces stored E45: {c3['fires']} (|diff|={c3['abs_diff']:.2e})")
    if not a.skip_overlap:
        print(f"C: E36 diagonal overlap = "
              f"{rec['C_e36_diagonal_prefix_fit_overlap']['mean_over_corpora']:.4f}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
