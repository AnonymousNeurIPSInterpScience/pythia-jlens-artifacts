#!/usr/bin/env python3
"""t66_fitter_equivalence_cuda.py — E66: do the programme's TWO fitters agree on CUDA?

WHY THIS IS NOT JUST AN E65 PROBLEM
  Two fitters produced the lenses this paper cites:
    * `fastfit.fast_fit`      — device-resident; produced E28's 83 ladder lenses (t2_fastfit) and
                                E33's operators.
    * `trainval.py`'s loop    — accumulates `jlens.fitting.jacobian_for_prompt`, whose per-prompt
                                Jacobians come back on CPU and are moved to the accumulator each
                                step; produced every INSTREAM/OOD lens used by E48 and E52.
  The paper compares those populations directly: EXPERIMENT_VALIDITY_AUDIT v2 3.4(a) flags that
  section 3's two adjacent tables are "different operator populations from different fitters".

  E60 established fast_fit == jlens.fitting.fit at 0.00e+00 ON CPU, and this session measured
  fast_fit == trainval's loop at 0.00e+00 on CPU at 70m. **Neither has ever been checked on CUDA**,
  which is where every stored lens was actually fitted (device=cuda in all four provenance blocks).
  E65 v1 then found a fresh fast_fit CUDA fit reading +11% above the stored trainval lens under a
  configuration that matches on fitter-inputs, prompts, weights, band and device.

  So this settles whether that discrepancy is a fitter difference on GPU — and if it is, it reaches
  every cross-fitter comparison in the paper, not just E65.

WHAT IT DOES
  On ONE GPU, on the SAME 200 Pile-CC seed-block-0 prompts, at 410m, band [9,21], target_layer=-2,
  skip_first=16, dim_batch=128, fp32, TF32 off:
    A  fit with fast_fit
    B  fit with trainval's loop
    C  load the STORED lens_INSTREAM_Pile-CC_410m_n200_s0.pt
  then compare all three operator-wise (Frobenius, max abs, relative) and read-wise (the admitted
  battery at Q0, persist and min).

DECISION RULE — fixed before running.
  EQUIVALENT      max relative operator difference A-vs-B <= 1e-3 AND the two reads differ by
                  less than the 1.84e-4 Pile-CC seed SD. The E65 discrepancy is then NOT a fitter
                  difference and something else is responsible.
  FITTERS DIVERGE max relative difference > 1e-3 or the reads differ by more than one seed SD.
                  Then every cross-fitter comparison in the paper carries an undisclosed
                  confound and the audit's 3.4(a) becomes a correction, not a disclosure.
  CONTROL C1  arm B must reproduce the STORED lens (arm C) to <= 1e-3 relative. B is supposed to be
              the very code path that produced C. If it does not, the stored lens is not what its
              own record says and that is a larger problem than either verdict.

    python experiments/t66_fitter_equivalence_cuda.py --device cuda
"""
from __future__ import annotations
import argparse, json, os, statistics, sys

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))
from provenance import write_result  # noqa: E402

MID = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))
K = [1, 2, 5, 10, 20, 50, 100]
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
CORPUS, SEED_BLOCK, N_FIT = "Pile-CC", 0, 200
STORED = "results/e48/lens_INSTREAM_Pile-CC_410m_n200_s0.pt"
SEED_SD = 1.84e-4          # Pile-CC seed SD at 410m
REL_TOL = 1e-3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--items-per-chunk", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results",
                                                  "e66_fitter_equivalence_cuda.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from jlens.fitting import jacobian_for_prompt
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from fastfit import fast_fit
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of

    tok = AutoTokenizer.from_pretrained(MID)
    hf = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    d, L, HALF = model.d_model, len(BAND), len(BAND) // 2

    texts = [json.loads(l)["text"]
             for l in open(os.path.join(HERE, "..", "corpora", f"{CORPUS}.jsonl"))]
    b = len(texts) // 3
    pool = [t for t in texts[SEED_BLOCK * b:(SEED_BLOCK + 1) * b]
            if len(tok(t).input_ids) >= 128][:N_FIT]
    print(f"{len(pool)} fitting prompts", flush=True)

    items = []
    for name in EVAL_SETS:
        for it in load_eval(name):
            tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
            if tgt:
                items.append({"set": name,
                              "ids": tok(it["prompt"], add_special_tokens=True).input_ids,
                              "pos": readout_position(tok, name, it["prompt"]), "tgt": tgt})
    ps, pt, pi = [], [], []
    for ii, it in enumerate(items):
        for t in it["tgt"]:
            ps.append(it["set"]); pt.append(t); pi.append(ii)
    SET_IDX = {s: torch.tensor([i for i, p in enumerate(ps) if p == s], dtype=torch.long)
               for s in EVAL_SETS}
    IP = {}
    for k, ii in enumerate(pi):
        IP.setdefault(ii, []).append(k)
    P_n = len(pt)

    A = torch.empty(len(items), L, d)
    with torch.no_grad():
        for ii, it in enumerate(items):
            t = torch.tensor([it["ids"]], device=a.device)
            with ActivationRecorder(model.layers, at=BAND) as rec:
                model.forward(t)
                A[ii] = torch.stack([rec.activations[l][0][it["pos"]].detach().float()
                                     for l in BAND])

    def score(T):
        H = torch.stack([A[:, j, :] @ T[BAND[j]].T for j in range(L)], 1)
        flat = H.reshape(-1, d)
        R = torch.empty(P_n, L)
        with torch.no_grad():
            for i0 in range(0, len(items), a.items_per_chunk):
                i1 = min(i0 + a.items_per_chunk, len(items))
                lg = model.unembed(flat[i0 * L:i1 * L]).float()
                for ii in range(i0, i1):
                    blk = lg[(ii - i0) * L:(ii - i0 + 1) * L]
                    for k in IP[ii]:
                        cand = torch.stack([(blk > blk[:, i:i + 1]).sum(1) + 1 for i in pt[k]])
                        R[k] = cand.min(0).values.float()
        mn = R.min(dim=1).values
        return {"min": float(statistics.mean(
                    [float(torch.stack([(mn <= kk).float() for kk in K]).mean(0)[SET_IDX[s]].mean())
                     for s in ADMITTED])),
                "persist": float(statistics.mean(
                    [float(torch.stack([((R <= kk).float().sum(1) >= HALF).float()
                                        for kk in K]).mean(0)[SET_IDX[s]].mean())
                     for s in ADMITTED]))}

    print("arm A: fast_fit ...", flush=True)
    ra = fast_fit(model, pool, dim_batch=128, max_seq_len=128, checkpoint_path=None,
                  checkpoint_every=0, log_every=0, target_layer=-2, skip_first=16)
    Ja = {l: ra.jacobians[l].float().cpu() for l in BAND}

    print("arm B: trainval loop ...", flush=True)
    Jsum, n_used, n_skip = None, 0, 0
    for pr in pool:
        try:
            pj, _, _ = jacobian_for_prompt(model, pr, BAND, target_layer=-2, dim_batch=128,
                                           max_seq_len=128, skip_first=16)
        except ValueError:
            n_skip += 1
            continue
        if Jsum is None:
            Jsum = {l: torch.zeros_like(pj[l].float()) for l in BAND}
        for l in BAND:
            Jsum[l] += pj[l].float().to(Jsum[l].device)
        n_used += 1
    Jb = {l: (Jsum[l] / n_used).cpu() for l in BAND}
    print(f"  trainval used {n_used}, skipped {n_skip}", flush=True)

    Jc = {int(k): v.float() for k, v in torch.load(
        os.path.join(HERE, "..", STORED), map_location="cpu", weights_only=True)["J"].items()}

    def cmp(X, Y):
        worst_abs = worst_rel = 0.0
        for l in BAND:
            dd = float((X[l] - Y[l]).abs().max())
            worst_abs = max(worst_abs, dd)
            worst_rel = max(worst_rel, dd / max(float(Y[l].abs().max()), 1e-12))
        return {"max_abs": worst_abs, "max_rel": worst_rel}

    sa, sb, sc = score(Ja), score(Jb), score(Jc)
    ab, bc, ac = cmp(Ja, Jb), cmp(Jb, Jc), cmp(Ja, Jc)
    print(f"  reads  fast={sa['persist']:.6f}  trainval={sb['persist']:.6f}  "
          f"stored={sc['persist']:.6f}", flush=True)

    c1 = {"what": "arm B (trainval, the path that produced the stored lens) vs arm C (stored)",
          "operator": bc, "read_diff": abs(sb["persist"] - sc["persist"]),
          "rel_tolerance": REL_TOL, "fires": bc["max_rel"] <= REL_TOL}
    diverge = ab["max_rel"] > REL_TOL or abs(sa["persist"] - sb["persist"]) > SEED_SD

    rec = {
        "experiment": "E66 — do the programme's two fitters agree on CUDA?",
        "prereg": "in-file docstring, fixed before running", "status": "PRE-REGISTERED",
        "model": MID, "band": BAND, "device": a.device, "corpus": CORPUS,
        "n_fit": len(pool), "n_items": len(items), "n_pairs": P_n,
        "arms": {"A_fast_fit": {"read": sa}, "B_trainval_loop": {"read": sb, "n_used": n_used,
                                                                 "n_skipped": n_skip},
                 "C_stored_lens": {"read": sc, "path": STORED}},
        "operator_differences": {"A_vs_B": ab, "B_vs_C": bc, "A_vs_C": ac},
        "read_differences": {"A_minus_B": sa["persist"] - sb["persist"],
                             "B_minus_C": sb["persist"] - sc["persist"],
                             "A_minus_C": sa["persist"] - sc["persist"],
                             "pile_cc_seed_sd": SEED_SD},
        "controls": {"C1_trainval_reproduces_stored": c1},
        "e65_v1_context": ("E65 v1 fitted with fast_fit on CUDA and read 0.050230 against the "
                           "stored lens's 0.045092 — a +11% gap under a configuration matching on "
                           "prompts, weights, band, device and fitter-inputs."),
    }
    if not c1["fires"]:
        rec["VERDICT"] = (f"UNCLEAR — arm B does not reproduce the stored lens "
                          f"(max_rel {bc['max_rel']:.3e} > {REL_TOL}). The stored lens is not what "
                          f"its own provenance says it is, which is a larger problem than either "
                          f"branch of this experiment.")
    elif diverge:
        rec["VERDICT"] = (f"FITTERS DIVERGE ON CUDA — max relative operator difference "
                          f"{ab['max_rel']:.3e} and a read gap of "
                          f"{abs(sa['persist']-sb['persist']):.6f} against a seed SD of {SEED_SD}. "
                          f"Every cross-fitter comparison in the paper carries an undisclosed "
                          f"confound; the audit's 3.4(a) becomes a correction, not a disclosure.")
    else:
        rec["VERDICT"] = (f"EQUIVALENT — the two fitters agree on CUDA to "
                          f"{ab['max_rel']:.3e} relative and their reads differ by "
                          f"{abs(sa['persist']-sb['persist']):.2e}, inside the {SEED_SD} seed SD. "
                          f"E65 v1's discrepancy is NOT a fitter difference.")
    write_result(a.out, rec, experiment="E66",
                 inputs=[os.path.join(HERE, "..", STORED)])
    print(f"\nC1 trainval-vs-stored: max_rel={bc['max_rel']:.3e} -> "
          f"{'FIRES' if c1['fires'] else 'DOES NOT FIRE'}")
    print(f"A-vs-B operator max_rel = {ab['max_rel']:.3e}")
    print(f"\nVERDICT: {rec['VERDICT']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
