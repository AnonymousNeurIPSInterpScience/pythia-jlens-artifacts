#!/usr/bin/env python3
"""
E66c — do operators fitted UNDER RECORDED PROVENANCE reproduce?

PRE-REGISTERED: docs/experiments/preregs/E66c_post_provenance_repro.md, committed at c9c4969
before this file existed.

E66b showed the E48 panel does not reproduce. The archaeology found why that is untraceable: those
lenses were fitted at 20:03-20:04 PDT on 2026-08-14, and commit 1a6f5df at 20:05 is the one that
added provenance.py. They are the last artifacts made before the provenance layer existed, so their
sidecars carry argv=None and env=null, and the box that made them was destroyed that night.

This asks the question that actually bears on the release: does an operator fitted WITH provenance
reproduce? Target is results/r6/lens_R6_Pile-CC_b0_410m_n200.pt, whose fit is recorded in
results/r6_within_source_410m.json with torch 2.11.0+cu128, TF32 off on all three flags, and both
corpus inputs SHA-256 hashed. Its pool is deterministic: file order, no RNG.

  /opt/conda/bin/python experiments/e66c_post_provenance_repro.py --dim-batch 32
"""
import argparse, hashlib, json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

MID = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))
SOURCE, N_BLOCKS, N_FIT, WINDOW = "Pile-CC", 4, 200, 128
STORED_B0 = os.path.join(ROOT, "results", "r6", "lens_R6_Pile-CC_b0_410m_n200.pt")
STORED_B1 = os.path.join(ROOT, "results", "r6", "lens_R6_Pile-CC_b1_410m_n200.pt")
RECORDED_TORCH = "2.11.0+cu128"          # results/r6_within_source_410m.json:provenance.env.torch
FP16_ULP = 9.766e-4                      # one fp16 step in the [1,2) binade; the storage threshold


def disjoint_blocks(corpus, tok, n_blocks, n_fit, window):
    """Transcribed verbatim from experiments/r6_within_source.py:124-133."""
    texts = [json.loads(l)["text"] for l in open(os.path.join(ROOT, "corpora", f"{corpus}.jsonl"))]
    qual = [t for t in texts if len(tok(t).input_ids) >= window]
    need = n_blocks * n_fit
    if len(qual) < need:
        raise SystemExit(f"ABORT: {corpus} supplies {len(qual)} full-window docs, need {need}.")
    return [qual[i * n_fit:(i + 1) * n_fit] for i in range(n_blocks)]


def max_abs_h(A, B):
    return max(float((A[l].float() - B[l].float()).abs().max()) for l in A)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dim-batch", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "e66c_post_provenance_repro.json"))
    a = ap.parse_args()

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

    blocks = disjoint_blocks(SOURCE, tok, N_BLOCKS, N_FIT, WINDOW)
    pool = blocks[0]
    print(f"  {len(pool)} prompts, block 0 of {N_BLOCKS}, device {a.device}, "
          f"dim_batch {a.dim_batch}", flush=True)

    lens = fast_fit(model, pool, source_layers=BAND, target_layer=-2, dim_batch=a.dim_batch,
                    max_seq_len=WINDOW, skip_first=16, device=a.device)
    refit = {l: lens.jacobians[l].detach().float().cpu().clone() for l in BAND}
    refit_h = {l: refit[l].half() for l in BAND}
    print(f"    fitted in {time.time()-t0:.0f}s", flush=True)

    b0 = torch.load(STORED_B0, map_location="cpu", weights_only=True)["J"]
    b1 = torch.load(STORED_B1, map_location="cpu", weights_only=True)["J"]
    S6 = max_abs_h(refit_h, {l: b0[l] for l in BAND})
    S_other = max_abs_h(refit_h, {l: b1[l] for l in BAND})

    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]
    controls = {
        "C1_tf32_off": {"required": "both flags False",
                        "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                        "cudnn": bool(torch.backends.cudnn.allow_tf32),
                        "fires": not torch.backends.cuda.matmul.allow_tf32
                                 and not torch.backends.cudnn.allow_tf32},
        "C2_same_pool": {"required": "200 prompts, blocks disjoint, endpoints recorded",
                         "n": len(pool), "first_sha16": sha(pool[0]), "last_sha16": sha(pool[-1]),
                         "b0_x_b1_overlap": len(set(blocks[0]) & set(blocks[1])),
                         "fires": len(pool) == N_FIT and len(set(blocks[0]) & set(blocks[1])) == 0},
        "C3_torch_matches_record": {"required": f"torch == {RECORDED_TORCH}",
                                    "recorded": RECORDED_TORCH, "observed": torch.__version__,
                                    "fires": torch.__version__ == RECORDED_TORCH},
        "C4_comparison_can_fail": {
            "required": f"a different block's operator must exceed one fp16 ULP ({FP16_ULP:.3e})",
            "max_abs_refit_b0_vs_stored_b1": S_other, "fires": S_other > FP16_ULP},
    }

    if not controls["C1_tf32_off"]["fires"]:
        verdict = "VOID — TF32 was not off."
    elif not controls["C4_comparison_can_fail"]["fires"]:
        verdict = ("VOID — C4 did not fire: the comparison cannot distinguish a different block's "
                   "operator, so a match would mean nothing.")
    elif S6 == 0.0:
        verdict = ("REPRODUCES — S6 = 0. An operator fitted under recorded provenance is "
                   "BIT-IDENTICAL to a refit at its storage dtype. The E48 panel is a dated "
                   "exception whose cause is known: it predates the provenance layer by two minutes.")
    elif S6 <= FP16_ULP:
        verdict = (f"REPRODUCES TO STORAGE — S6 = {S6:.3e}, within one fp16 step "
                   f"({FP16_ULP:.3e}). The refit agrees to the resolution the file is stored at.")
    else:
        verdict = (f"DOES NOT REPRODUCE — S6 = {S6:.3e} exceeds one fp16 ULP ({FP16_ULP:.3e}) and "
                   f"is the same scale as the E48 panel's 1.1e-03. Recorded provenance is then not "
                   f"sufficient for reproduction and the harness has a real problem.")

    rec = {
        "experiment": "E66c — do operators fitted under recorded provenance reproduce?",
        "prereg": "docs/experiments/preregs/E66c_post_provenance_repro.md",
        "status": "PRE-REGISTERED",
        "model": MID, "band": BAND, "source": SOURCE, "block": 0,
        "n_fit": len(pool), "dim_batch": a.dim_batch, "device": a.device,
        "target": os.path.relpath(STORED_B0, ROOT),
        "recorded_env_torch": RECORDED_TORCH, "observed_torch": torch.__version__,
        "fp16_ulp_threshold": FP16_ULP,
        "S6_fp16_refit_vs_stored": S6,
        "S_refit_vs_other_block": S_other,
        "why_this_target": ("E48's panel was fitted 20:03-20:04 on 2026-08-14; commit 1a6f5df at "
                            "20:05 added provenance.py. r6 was fitted 2026-08-20 with a full "
                            "environment block and a deterministic file-order pool."),
        "controls": controls,
        "controls_fired": {k: v.get("fires") for k, v in controls.items()},
        "VERDICT": verdict,
    }
    print("\n" + verdict, flush=True)
    for k, v in controls.items():
        print(f"  {k}: fires={v.get('fires')}", flush=True)
    prov.write_result(a.out, rec, script=__file__, experiment="E66c",
                      inputs=[STORED_B0, os.path.join(ROOT, "corpora", f"{SOURCE}.jsonl")])
    print("wrote", os.path.relpath(a.out, ROOT), f"({(time.time()-t0)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
