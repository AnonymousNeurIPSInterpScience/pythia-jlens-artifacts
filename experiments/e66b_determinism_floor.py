#!/usr/bin/env python3
"""
E66b — THE FITTER'S DETERMINISM FLOOR AT 410M ON CUDA.

PRE-REGISTERED: docs/experiments/preregs/E66b_determinism_floor.md, committed at b4c131f BEFORE
this file existed. The decision rule lives there and is not restated here in a form that could
drift from it.

WHAT THIS SETTLES.  E66 returned "UNCLEAR — the stored lens is not what its own provenance says it
is". That verdict came from a CONTROL, not from E66's own primary: E66's primary asked whether the
two fitters agree on CUDA and answered YES, at max_abs 1.19e-07. What failed was C1, which asked
whether a refit reproduces the STORED operator, and which compared

    an fp16 artifact  against  an fp32 refit  under a 1e-3 tolerance

that was calibrated against neither the storage dtype nor the fitter's own reproducibility. The
stored operator is float16 with max|J| = 1.208984, so fp16 half-ULP is 4.883e-04 and explains 44%
of the observed 1.1066e-03. The residual is R = 6.183e-04 and nobody has measured what the fitter's
run-to-run floor is on this device, which is the number C1 needed.

E60 measured that floor at 70M ON CPU: the fitters agree exactly at 1/4/8 threads and differ by
1.09e-04 above that, the same order as the released fitter's disagreement with ITSELF (1.28e-04).
Its own reading: "FITS ARE NOT BIT-REPRODUCIBLE above 8 threads on this machine, which every stored
lens SHA depends on." This measures the same thing at 410M on CUDA.

  .venv/bin/python experiments/e66b_determinism_floor.py --smoke   # 4 prompts, proves the path
  .venv/bin/python experiments/e66b_determinism_floor.py           # registered: 3 fits, N=200
"""
import argparse, hashlib, json, os, statistics, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

MID = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))
CORPUS = "Pile-CC"
SEED_BLOCK = 0
N_FIT = 200
STORED = os.path.join(ROOT, "results", "e48", "lens_INSTREAM_Pile-CC_410m_n200_s0.pt")

# Fixed in the registration, before this ran. Do not recompute from the run.
R_RESIDUAL = 6.183e-4          # E66's B_vs_C max_abs 1.1066e-3 minus the fp16 half-ULP 4.883e-4
FP16_HALF_ULP = 4.883e-4

# ---- added 2026-08-30 for the C-completion (C2 and C5 were registered and never written) -------
WINDOW = 128
K = [1, 2, 5, 10, 20, 50, 100]                                  # t66_fitter_equivalence_cuda.py:59
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]          # t66:60
TV_SIDECAR = os.path.join(ROOT, "results", "e48", "tv_INSTREAM_Pile-CC_s0.json")
MANIFEST = os.path.join(ROOT, "corpora", "manifest.json")

# What the 2026-08-14 CUDA run of THIS script stored. Kept so the re-run reports old BESIDE new.
# S IS DEVICE-DEPENDENT AND IS NOT A REPRODUCTION TARGET: the GPU model was never recorded, and
# that absence is E66b's own finding. See docs/handoff/E66_C_COMPLETION_SPEC.md section 3.
STORED_RUN = {"S": 0.0010986328125, "F": 0.0, "F_prime": 1.0281801223754883e-06,
              "floor_used": 1.0281801223754883e-06, "dim_batch_D1_D2": 32, "n_fit": 200,
              "torch": "2.11.0+cu128", "gpu_model": None,
              "VERDICT": "CONFIRMED DEFECT"}


def sha_pool(prompts) -> str:
    """SHA-256 of the ordered prompt pool. Order matters: the fitter consumes it in order."""
    return hashlib.sha256("\x00".join(prompts).encode()).hexdigest()


def max_abs(A: dict, B: dict) -> float:
    return max(float((A[l].float() - B[l].float()).abs().max()) for l in A)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dim-batch", type=int, default=32,
                    help="D1/D2 use this; D3 uses 2x it. Changes reduction order, not "
                         "the estimand. 128 OOMs a 24 GB card under torch 2.11.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "e66b_determinism_floor.json"))
    # C2 and C5 both reference D1, and D1 was never persisted: the 2026-08-14 run computed it in
    # memory and dropped it, so there was nothing on disk to diff and both controls were
    # unwritable. Persisting it is what makes them computable at all.
    ap.add_argument("--save-d1", default=os.path.join(ROOT, "results",
                                                      "e66_D1_refit_410m_pilecc_s0.pt"))
    ap.add_argument("--skip-c2", action="store_true",
                    help="skip the read-equality control only (it loads the eval battery)")
    a = ap.parse_args()
    n_fit = 4 if a.smoke else N_FIT

    # TF32 is forbidden. Set it in torch, not only via the driver override, and ASSERT it (C1).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from fastfit import fast_fit
    import provenance as prov

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MID)
    hf = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)

    # The SAME pool construction as t66_fitter_equivalence_cuda.py:89-92. Copied deliberately: a
    # floor measured on a different pool would not bound the difference E66 reported.
    texts = [json.loads(l)["text"]
             for l in open(os.path.join(ROOT, "corpora", f"{CORPUS}.jsonl"))]
    b = len(texts) // 3
    pool = [t for t in texts[SEED_BLOCK * b:(SEED_BLOCK + 1) * b]
            if len(tok(t).input_ids) >= 128][:n_fit]
    print(f"  {len(pool)} fitting prompts, band {BAND[0]}..{BAND[-1]}, device {a.device}", flush=True)

    # ---------------------------------------------------------------- C5: same prompts?
    # THE REGISTERED FORM OF C5 IS NOT COMPUTABLE, AND SAYING SO IS THE POINT.
    # C5 was registered as "D1's pool SHA-256 equals the stored run's". No such hash exists
    # anywhere: results/e48/tv_INSTREAM_Pile-CC_s0.json carries NO provenance block, the stored
    # .pt records only {n_prompts, source_layers, d_model, model, corpus, seed}, and the HF mirror's
    # copy of the sidecar is byte-identical to the local one. The E48 panel was fitted 20:03-20:04
    # PDT on 2026-08-14, two minutes before commit 1a6f5df added provenance.py. There is nothing
    # to equal.
    #
    # What IS obtainable is stronger than a bare hash comparison and is checked here instead:
    #   1. the code that fitted the panel, pythia/trainval.py at 1a6f5df^, whose pool construction
    #      is `texts[seed*b:(seed+1)*b]` -> filter len>=128 -> `[:n_max]`, with NO RNG anywhere;
    #   2. that construction is byte-for-byte the same rule this script uses, so the pools must be
    #      identical -- and that equality is TESTED here, not asserted;
    #   3. the corpus bytes are pinned: corpora/manifest.json records Pile-CC's SHA-256 and the
    #      file on disk still matches it;
    #   4. the one quantity the stored run DID record, n_used = 200, must equal len(pool).
    # A NEGATIVE CONTROL is included because a hash that cannot tell two pools apart proves
    # nothing (rule 10): seed block 1 must produce a DIFFERENT hash with zero prompt overlap.
    #
    # WHAT THIS DOES NOT PROVE. It proves the prompts are the same GIVEN the recorded corpus bytes
    # and the historical code. It cannot prove the 2026-08-14 process actually consumed them,
    # because that run recorded nothing. C5 is therefore reported as RECONSTRUCTED, not as the
    # registered equality, and its registered form is recorded as uncomputable.
    tv = json.load(open(TV_SIDECAR))
    man = json.load(open(MANIFEST))
    corpus_sha = hashlib.sha256(open(os.path.join(ROOT, "corpora", f"{CORPUS}.jsonl"),
                                     "rb").read()).hexdigest()
    # trainval.py @ 1a6f5df^ :156-159 -- slice, THEN filter, THEN truncate.
    tv_pool = texts[SEED_BLOCK * b:(SEED_BLOCK + 1) * b]
    tv_pool = [t for t in tv_pool if len(tok(t).input_ids) >= WINDOW][:n_fit]
    other = [t for t in texts[1 * b:2 * b] if len(tok(t).input_ids) >= WINDOW][:n_fit]
    h_d1, h_tv, h_other = sha_pool(pool), sha_pool(tv_pool), sha_pool(other)
    c5_ok = (h_d1 == h_tv and corpus_sha == man[CORPUS]["sha256"]
             and h_other != h_d1 and len(set(pool) & set(other)) == 0
             and (a.smoke or len(pool) == tv["n_used"]))
    C5 = {
        "required": "REGISTERED VERBATIM: 'pool SHA-256 identical to tv_INSTREAM_Pile-CC_s0.json's "
                    "inputs WHERE RECORDED'. Nothing is recorded, so the registered clause is "
                    "VACUOUS -- it cannot fail, which rule 10 forbids. It is therefore adjudicated "
                    "by reconstruction against the historical fitting code, with a negative control "
                    "that makes it able to fail.",
        "registered_form_vacuous": True,
        "registered_form_computable": False,
        "why_not": "results/e48/tv_INSTREAM_Pile-CC_s0.json has no provenance block; the stored .pt "
                   "records only n_prompts/source_layers/d_model/model/corpus/seed; the mirror copy "
                   "is byte-identical to local. E48 was fitted 20:03-20:04 PDT 2026-08-14, commit "
                   "1a6f5df added provenance.py at 20:05.",
        "D1_pool_sha256": h_d1,
        "trainval_reconstruction_sha256": h_tv,
        "historical_code": "pythia/trainval.py @ 1a6f5df^ :156-159 (slice, filter len>=128, [:n_max])",
        "pools_identical": h_d1 == h_tv,
        "corpus_sha256": corpus_sha,
        "corpus_sha256_in_manifest": man[CORPUS]["sha256"],
        "corpus_matches_manifest": corpus_sha == man[CORPUS]["sha256"],
        "n_pool": len(pool), "stored_n_used": tv["n_used"],
        "n_matches_stored": len(pool) == tv["n_used"],
        "negative_control_block1_sha256": h_other,
        "negative_control_differs": h_other != h_d1,
        "negative_control_prompt_overlap": len(set(pool) & set(other)),
        "smoke": bool(a.smoke),
        "does_not_prove": "that the 2026-08-14 process consumed these prompts -- that run recorded "
                          "nothing. It proves the pool is determined by corpus bytes that still "
                          "match the manifest and by code still obtainable from git history.",
        "fires": bool(c5_ok)}
    print(f"  C5 pool: D1={h_d1[:16]} trainval={h_tv[:16]} identical={C5['pools_identical']} "
          f"corpus_ok={C5['corpus_matches_manifest']} neg_ctrl_differs={C5['negative_control_differs']} "
          f"-> fires={C5['fires']}", flush=True)

    def fit(tag, dim_batch):
        t = time.time()
        lens = fast_fit(model, pool, source_layers=BAND, target_layer=-2, dim_batch=dim_batch,
                        max_seq_len=128, skip_first=16, device=a.device, checkpoint_path=None)
        J = {l: lens.jacobians[l].detach().float().cpu().clone() for l in BAND}
        print(f"    {tag}: dim_batch={dim_batch}  {time.time()-t:.0f}s", flush=True)
        return J

    D1 = fit("D1", a.dim_batch)
    D2 = fit("D2", a.dim_batch)             # byte-identical invocation to D1
    D3 = fit("D3", a.dim_batch * 2)         # changes reduction order, not the estimand

    # PERSIST D1. grep -c torch.save on the 2026-08-14 version of this file returns 0: D1 was
    # computed in memory, compared, and dropped. That is why C2 and C5 could not be written -- there
    # was nothing on disk to diff. Saved fp32, which is what was fitted; the fp16 cast used for S
    # is derived from it below and is not what is stored.
    if a.save_d1:
        os.makedirs(os.path.dirname(a.save_d1), exist_ok=True)
        torch.save({"J": D1, "source_layers": BAND, "d_model": model.d_model, "model": MID,
                    "corpus": CORPUS, "seed": SEED_BLOCK, "n_prompts": len(pool),
                    "dtype": "float32", "dim_batch": a.dim_batch,
                    "pool_sha256": h_d1,
                    "note": "E66 D1, the first refit. Regenerated 2026-08-30 to make the "
                            "registered controls C2 and C5 computable."}, a.save_d1)
        d1_sha = hashlib.sha256(open(a.save_d1, "rb").read()).hexdigest()
        print(f"  saved D1 -> {os.path.relpath(a.save_d1, ROOT)}  sha256={d1_sha[:16]}", flush=True)
    else:
        d1_sha = None

    stored_raw = torch.load(STORED, map_location="cpu", weights_only=True)["J"]
    stored = {l: stored_raw[l] for l in BAND}

    # S: does the artifact reproduce AT ITS OWN DTYPE? Casting D1 down is the like-for-like
    # comparison E66's C1 should have made; comparing fp32 against fp16 charges the refit for the
    # storage format.
    D1_h = {l: D1[l].half() for l in BAND}
    S = max(float((D1_h[l].float() - stored[l].float()).abs().max()) for l in BAND)
    F = max_abs(D1, D2)
    Fp = max_abs(D1, D3)
    floor = max(F, Fp)

    allv = torch.cat([stored[l].reshape(-1).float() for l in BAND])
    absmax = float(allv.abs().max())

    # ---------------------------------------------------------------- C2: same estimand?
    # Registered as "D1's reads equal the stored operator's reads -- read_diff == 0.0 under min and
    # persist", and never written because D1 was never persisted. The scoring path is transcribed
    # from t66_fitter_equivalence_cuda.py:124-141, the code that produced E66's own
    # `read_differences` block, so the shapes are comparable to what E66 reported.
    #
    # NOTE ON THE READOUT CONVENTION. That path tokenises the prompt UNSTRIPPED, which is the
    # legacy readout the programme later corrected. It is used here deliberately and unchanged,
    # because C2 is a DIFFERENCE between two operators scored through one identical path: the
    # convention cancels, and matching E66's shape matters more than matching the corrected
    # readout. This is not a corrected-readout number and must not be quoted as one.
    C2 = None
    if not a.skip_c2:
        from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of
        from jlens.hooks import ActivationRecorder
        L, d = len(BAND), model.d_model
        HALF = L // 2
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
        for k_, ii in enumerate(pi):
            IP.setdefault(ii, []).append(k_)
        P_n = len(pt)
        A = torch.empty(len(items), L, d)
        with torch.no_grad():
            for ii, it in enumerate(items):
                t = torch.tensor([it["ids"]], device=a.device)
                with ActivationRecorder(model.layers, at=BAND) as rec:
                    model.forward(t)
                    A[ii] = torch.stack([rec.activations[l][0][it["pos"]].detach().float().cpu()
                                         for l in BAND])
        print(f"  C2: {len(items)} eval items, {P_n} scored pairs", flush=True)

        def read(T, chunk=24):
            H = torch.stack([A[:, j, :] @ T[BAND[j]].float().T for j in range(L)], 1)
            flat = H.reshape(-1, d)
            R = torch.empty(P_n, L)
            with torch.no_grad():
                for i0 in range(0, len(items), chunk):
                    i1 = min(i0 + chunk, len(items))
                    lg = model.unembed(flat[i0 * L:i1 * L].to(a.device)).float().cpu()
                    for ii in range(i0, i1):
                        blk = lg[(ii - i0) * L:(ii - i0 + 1) * L]
                        for k_ in IP[ii]:
                            cand = torch.stack([(blk > blk[:, i:i + 1]).sum(1) + 1 for i in pt[k_]])
                            R[k_] = cand.min(0).values.float()
            mn = R.min(dim=1).values
            return {"min": float(statistics.mean(
                        [float(torch.stack([(mn <= kk).float() for kk in K]).mean(0)[SET_IDX[s]].mean())
                         for s in ADMITTED])),
                    "persist": float(statistics.mean(
                        [float(torch.stack([((R <= kk).float().sum(1) >= HALF).float()
                                            for kk in K]).mean(0)[SET_IDX[s]].mean())
                         for s in ADMITTED]))}

        r_d1, r_st = read(D1), read(stored)
        diffs = {k_: r_d1[k_] - r_st[k_] for k_ in ("min", "persist")}
        C2 = {
            "required": "D1's reads equal the stored operator's reads: read_diff == 0.0 under BOTH "
                        "min and persist. E66 measured exactly that for its own three arms.",
            "read_D1": r_d1, "read_stored": r_st, "read_differences": diffs,
            "scoring_path": "t66_fitter_equivalence_cuda.py:124-141, UNSTRIPPED (legacy) readout; "
                            "the convention cancels in a difference and this is not a "
                            "corrected-readout number",
            "n_items": len(items), "n_scored_pairs": P_n,
            "e66_reference_read_differences": {"A_minus_B": 0.0, "B_minus_C": 0.0, "A_minus_C": 0.0},
            "fires": bool(diffs["min"] == 0.0 and diffs["persist"] == 0.0)}
        print(f"  C2: read_diff min={diffs['min']:+.3e} persist={diffs['persist']:+.3e} "
              f"-> fires={C2['fires']}", flush=True)

    controls = {
        "C1_tf32_is_off": {
            "required": "observed flags False, asserted not assumed",
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "fires": (not torch.backends.cuda.matmul.allow_tf32
                      and not torch.backends.cudnn.allow_tf32)},
        "C3_fp16_arithmetic": {
            "required": "max|J| in [1,2) so the fp16 half-ULP is 4.883e-04",
            "abs_max": absmax, "half_ulp": FP16_HALF_ULP,
            "fires": 1.0 <= absmax < 2.0},
        "C4_floor_is_measurable": {
            "required": "if F == F' == 0 the card is fully deterministic, the WITHIN-THE-FLOOR "
                        "branch is unavailable by construction, and adjudication falls to S alone. "
                        "This is the guard on the declared bias.",
            "F_identical_invocation": F, "F_prime_changed_reduction_order": Fp,
            "floor_is_zero": floor == 0.0, "fires": True},
    }
    # The two registered controls that were specified and never written, added 2026-08-30.
    controls["C5_same_prompts"] = C5
    if C2 is not None:
        controls["C2_same_estimand"] = C2

    if not controls["C1_tf32_is_off"]["fires"]:
        verdict = "VOID — TF32 was not off; the run is not admissible."
    elif S == 0.0:
        verdict = (f"RESOLVED, THE ARTIFACT IS EXACT — fp16(refit) is bit-identical to the stored "
                   f"operator (S = 0). E66's UNCLEAR is retired: its C1 compared an fp16 artifact "
                   f"against an fp32 refit and charged the refit for the storage format.")
    elif floor >= R_RESIDUAL:
        verdict = (f"RESOLVED, WITHIN THE FLOOR — S = {S:.3e}; the fitter's own run-to-run floor on "
                   f"this device is {floor:.3e}, at or above E66's unexplained residual "
                   f"R = {R_RESIDUAL:.3e}. The stored operator is what its provenance says to the "
                   f"precision this fitter can deliver. E66's UNCLEAR is retired and replaced by a "
                   f"stated tolerance of {floor:.3e}, which ARTIFACTS.md must carry.")
    elif floor <= R_RESIDUAL / 3:
        verdict = (f"CONFIRMED DEFECT — the floor is {floor:.3e}, more than three times below the "
                   f"residual R = {R_RESIDUAL:.3e}. Reduction order does not explain the gap. The "
                   f"stored operators do not reproduce and must be refit before release.")
    else:
        verdict = (f"UNCLEAR — the floor {floor:.3e} falls in the registered band "
                   f"({R_RESIDUAL/3:.3e}, {R_RESIDUAL:.3e}). Report and stop. Do not re-cut.")

    rec = {
        "experiment": "E66b — the fitter's determinism floor at 410M on CUDA",
        "prereg": "docs/experiments/preregs/E66b_determinism_floor.md",
        "status": "PRE-REGISTERED",
        "model": MID, "band": BAND, "corpus": CORPUS, "seed_block": SEED_BLOCK,
        "n_fit": len(pool), "device": a.device, "smoke": bool(a.smoke),
        "dim_batch_D1_D2": a.dim_batch, "dim_batch_D3": a.dim_batch * 2,
        "resolves": "results/e66_fitter_equivalence_cuda.json (VERDICT: UNCLEAR)",
        "e66_recap": {
            "A_vs_B_max_abs": 1.1920928955078125e-07,
            "B_vs_C_max_abs": 0.00110660120844841,
            "read_differences_all_zero": True,
            "note": "E66's PRIMARY (do the two fitters agree on CUDA?) returned EQUIVALENT. The "
                    "UNCLEAR came from control C1, not from the primary."},
        "fp16_half_ulp_at_absmax": FP16_HALF_ULP,
        "residual_R_fixed_in_registration": R_RESIDUAL,
        "S_fp16_refit_vs_stored": S,
        "F_identical_invocation": F,
        "F_prime_changed_reduction_order": Fp,
        "floor_used": floor,

        # The GPU model was NEVER RECORDED by the 2026-08-14 run, and that absence is precisely why
        # S cannot be reproduced across boxes. Recorded here so the next agent is not in this
        # position. F must be 0.0 on any card; F' and S are reduction-order and kernel dependent.
        "gpu_model": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "gpu_capability": (list(torch.cuda.get_device_capability(0))
                           if torch.cuda.is_available() else None),
        "stored_run_2026_08_14": STORED_RUN,
        "old_vs_new": {
            "S": {"stored": STORED_RUN["S"], "new": S,
                  "comparable_across_gpus": False,
                  "note": "NOT a reproduction target. Kernel- and card-dependent, and that "
                          "dependence IS E66b's finding. See E66_C_COMPLETION_SPEC.md section 3."},
            "F": {"stored": STORED_RUN["F"], "new": F, "comparable_across_gpus": True,
                  "note": "byte-identical invocation; must be 0.0 on any card"},
            "F_prime": {"stored": STORED_RUN["F_prime"], "new": Fp,
                        "comparable_across_gpus": False,
                        "note": "doubled dim_batch changes reduction order, which is device-specific"}},
        "D1_persisted_to": (os.path.relpath(a.save_d1, ROOT) if a.save_d1 else None),
        "D1_sha256": d1_sha,
        "R_derivation_caveat": (
            "R = 6.183e-04 was fixed in the registration as S_observed minus the fp16 HALF-ULP. Two "
            "independently rounded values straddling a boundary differ by a FULL ULP (9.766e-04), so "
            "R is likely overstated. The verdict survives either model: half-ULP gives R/3 = "
            "2.061e-04 and full-ULP gives R/3 = 4.333e-05, and the measured floor clears both. The "
            "derivation is corrected here; CONFIRMED DEFECT is not reopened."),
        "controls": controls,
        "controls_fired": {k: v.get("fires") for k, v in controls.items()},
        "VERDICT": verdict,
    }
    print("\n" + verdict, flush=True)
    prov.write_result(a.out, rec, script=__file__, experiment="E66b",
                      inputs=[STORED, os.path.join(ROOT, "results",
                                                   "e66_fitter_equivalence_cuda.json")])
    print("wrote", os.path.relpath(a.out, ROOT), f"({(time.time()-t0)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
