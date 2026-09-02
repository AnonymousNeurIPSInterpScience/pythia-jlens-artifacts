#!/usr/bin/env python3
"""t65_ckpt_geometry.py — E65 Phase 0: does the readout geometry have a TRAINING-axis floor?

PRE-REGISTRATION: docs/experiments/preregs/E65_training_axis_floor.md, written before this ran.

WHY THIS IS PHASE 0 AND NOT AN AFTERTHOUGHT
  E38/E39 established a PARAMETER-axis floor: concept-token unembedding is effectively rank-1 below
  410M (mean pairwise cosine 0.959 at 70M, 0.922 at 160M, 0.031 at 410M), so no transport can
  separate what W_U has collapsed. This asks whether an EARLY CHECKPOINT of a large model has the
  same collapsed geometry.

  It needs no Jacobian, no fitting, no eval battery scoring and no GPU — only `embed_out` and the
  concept-token id list. So it can make the paid Phase 1 unnecessary, or tell it where to look.

THE STATISTIC, identical to E39
  Over the 1310 concept-token rows of W_U: mean pairwise cosine, and effective rank (participation
  ratio of the singular values). E39's stored values are the reference and the final checkpoint must
  land on them (control C1b below).

CONTROLS
  C1b FINAL-CHECKPOINT ANCHOR. The last checkpoint IS the released model, which E39 already
      measured. Its mean cosine must reproduce e38_jgeometry.json to <= 1e-3. If it does not, the
      checkpoint-loading path is not the path that produced our stored geometry and nothing here is
      comparable.
  C2b TOKEN SET IDENTITY. The concept-token id list must be identical at every checkpoint. Pythia
      shares one tokenizer across all checkpoints and sizes, so this is expected to hold trivially —
      it is asserted rather than assumed, because if it ever failed every cross-checkpoint
      comparison would be meaningless.

    python experiments/t65_ckpt_geometry.py --model 410m --smoke
    python experiments/t65_ckpt_geometry.py --model 410m
"""
from __future__ import annotations
import argparse, json, os, sys

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))
from provenance import write_result  # noqa: E402

MODELS = {"410m": "EleutherAI/pythia-410m-deduped", "1b": "EleutherAI/pythia-1b-deduped"}
# The published grid: powers of two to 512, then every 1000 to 143000. We take the whole early
# ramp (where a floor would live) and a log-ish tail. FIXED HERE, BEFORE THE RUN.
CKPTS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
         1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
E39_REF = {"410m": 0.030666086822748184, "1b": 0.019916435703635216}


def concept_token_ids(tok):
    """The 1310 concept tokens of the admitted battery — E39's exact set."""
    from anchor_evals import EVAL_SETS, load_eval, token_ids_of
    ids = set()
    for name in EVAL_SETS:
        for it in load_eval(name):
            for w in it.get("intermediates", []):
                t = token_ids_of(tok, w)
                if t:
                    ids.update(t)
    return sorted(ids)


def geometry(W, idx):
    """mean pairwise cosine and effective rank over the concept-token rows of W_U."""
    R = W[idx].float()
    R = R / R.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n = R.shape[0]
    G = R @ R.T
    # mean over the strict upper triangle
    mean_cos = float((G.sum() - n) / (n * (n - 1)))
    sv = torch.linalg.svdvals(W[idx].float())
    p = sv / sv.sum()
    eff_rank = float(torch.exp(-(p * (p.clamp_min(1e-12)).log()).sum()))
    return mean_cos, eff_rank, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="410m", choices=list(MODELS))
    ap.add_argument("--smoke", action="store_true", help="two checkpoints only, no file written")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    mid = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(mid)
    idx_list = concept_token_ids(tok)
    idx = torch.tensor(idx_list, dtype=torch.long)
    print(f"{len(idx_list)} concept tokens (E39 reference: 1310)", flush=True)

    ckpts = [0, 143000] if a.smoke else CKPTS
    rows, first_ids = {}, None
    for step in ckpts:
        rev = f"step{step}"
        try:
            hf = AutoModelForCausalLM.from_pretrained(mid, revision=rev, dtype=torch.float32)
        except Exception as e:                     # recorded, never swallowed
            rows[rev] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            print(f"  {rev:12s} LOAD FAILED: {type(e).__name__}", flush=True)
            continue
        W = hf.get_output_embeddings().weight.data
        ids_here = concept_token_ids(AutoTokenizer.from_pretrained(mid, revision=rev))
        if first_ids is None:
            first_ids = ids_here
        mc, er, n = geometry(W, idx)
        rows[rev] = {"step": step, "mean_cos": mc, "eff_rank": er, "n_concept_tokens": n,
                     "token_set_identical": ids_here == first_ids,
                     "wu_fro": float(W.float().norm())}
        print(f"  {rev:12s} mean_cos={mc:.4f}  eff_rank={er:8.1f}  |W_U|={rows[rev]['wu_fro']:.1f}"
              f"  tokens_same={rows[rev]['token_set_identical']}", flush=True)
        del hf, W
    if a.smoke:
        print("\nsmoke only; no file written")
        return 0

    ok = [v for v in rows.values() if "error" not in v]
    final = rows.get("step143000", {})
    c1b = {"reference_e39_mean_cos": E39_REF[a.model],
           "final_checkpoint_mean_cos": final.get("mean_cos"),
           "abs_diff": (abs(final["mean_cos"] - E39_REF[a.model])
                        if "mean_cos" in final else None),
           "tolerance": 1e-3}
    c1b["fires"] = c1b["abs_diff"] is not None and c1b["abs_diff"] <= 1e-3
    c2b = {"n_checkpoints_checked": len(ok),
           "all_token_sets_identical": all(v["token_set_identical"] for v in ok),
           "fires": all(v["token_set_identical"] for v in ok)}

    # the gate: which checkpoints are geometrically degenerate by the paper's own 410M threshold
    THRESH = 0.031
    degenerate = [k for k, v in rows.items() if "mean_cos" in v and v["mean_cos"] > THRESH]
    clean = [k for k, v in rows.items() if "mean_cos" in v and v["mean_cos"] <= THRESH]

    rec = {
        "experiment": "E65 Phase 0 — the readout-geometry gate across training checkpoints",
        "prereg": "docs/experiments/preregs/E65_training_axis_floor.md",
        "status": "PRE-REGISTERED",
        "model": mid, "checkpoints_requested": ckpts,
        "statistic": ("mean pairwise cosine and effective rank over the concept-token rows of "
                      "W_U — E39's exact statistic, band-free and Jacobian-free"),
        "degeneracy_threshold_mean_cos": THRESH,
        "threshold_source": ("the paper's own 410M value, 0.031; above it the readout is in the "
                             "regime E38/E39 shows no transport can separate"),
        "by_checkpoint": rows,
        "n_degenerate": len(degenerate), "degenerate_checkpoints": degenerate,
        "n_clean": len(clean),
        "earliest_clean_checkpoint": (min((rows[k]["step"] for k in clean), default=None)),
        "controls": {"C1b_final_checkpoint_anchor": c1b, "C2b_token_set_identity": c2b},
    }
    if not c1b["fires"]:
        rec["VERDICT"] = ("UNCLEAR — the final checkpoint does not reproduce E39's stored geometry; "
                          "the checkpoint-loading path is not the one that produced our numbers")
    elif not degenerate:
        rec["VERDICT"] = (
            f"NO GEOMETRY FLOOR — every checkpoint from step{min(c['step'] for c in ok)} onward "
            f"sits at or below the 410M threshold of {THRESH}. The parameter-axis degeneracy has "
            f"no training-axis analogue at this scale, so Phase 1 has no narrow window to target "
            f"and should sample the whole range.")
    else:
        rec["VERDICT"] = (
            f"GEOMETRY FLOOR AT step{rec['earliest_clean_checkpoint']} — {len(degenerate)} early "
            f"checkpoint(s) exceed the 410M threshold and are geometrically degenerate. Phase 1 "
            f"should concentrate around the transition; readouts before it are uninterpretable "
            f"for the same reason 70M/160M readouts are.")

    out = a.out or os.path.join(HERE, "..", "results", f"e65_ckpt_geometry_{a.model}.json")
    write_result(out, rec, experiment="E65-P0", inputs=[])
    print(f"\nC1b final-checkpoint anchor: {c1b['abs_diff']} vs tol {c1b['tolerance']} -> "
          f"{'FIRES' if c1b['fires'] else 'DOES NOT FIRE'}")
    print(f"C2b token set identical across checkpoints: {c2b['fires']}")
    print(f"\nVERDICT: {rec['VERDICT']}\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
