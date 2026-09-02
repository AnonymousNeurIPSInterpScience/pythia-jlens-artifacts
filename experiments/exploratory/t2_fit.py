#!/usr/bin/env python3
"""t2_fit.py — fit a J-lens on a Pythia model. No centering, ever.

Corpus is wikitext-103-raw-v1, matching the Gemma-2 fit provenance
(eab/results/e1_box/e7_fit_provenance.json) so N and corpus are not confounded when the
two families are discussed.

Writes the lens AND a provenance JSON beside it (repo convention: provenance JSON beside
every artifact). Resumable via the anchor's own checkpointing.

    .venv/bin/python t2_fit.py --model EleutherAI/pythia-70m-deduped -n 200 \
        --out results/lenses/ladder/lens_70m_n200.pt
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))

from jlens.fitting import SKIP_FIRST_N_POSITIONS, fit  # noqa: E402
from jlens.hf import Layout, from_hf  # noqa: E402

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")


def load_prompts(n: int, min_chars: int = 400) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
                      streaming=True)
    out: list[str] = []
    for row in ds:
        t = row["text"].strip()
        if len(t) >= min_chars and not t.startswith("="):
            out.append(t)
            if len(out) >= n:
                break
    if len(out) < n:
        raise SystemExit(f"only found {len(out)}/{n} prompts")
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("-n", "--n-prompts", type=int, default=200)
    ap.add_argument("--dim-batch", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=getattr(torch, a.dtype)).to(a.device)
    try:
        model = from_hf(hf, tok)
        autodetect = True
    except ValueError:
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
        autodetect = False

    prompts = load_prompts(a.n_prompts)
    ckpt = a.out.replace(".pt", "_ckpt.pt")
    lens = fit(model, prompts, dim_batch=a.dim_batch, max_seq_len=a.max_seq_len,
               checkpoint_path=ckpt, checkpoint_every=max(1, a.n_prompts // 10))
    lens.save(a.out)
    wall = time.perf_counter() - t0

    sv0 = {l: float(torch.linalg.svdvals(lens.jacobians[l].float())[0])
           for l in lens.source_layers}
    prov = {
        "artifact": a.out,
        "sha256": sha256(a.out),
        "bytes": os.path.getsize(a.out),
        "model": a.model,
        "corpus": "wikitext-103-raw-v1",
        "n_prompts_requested": a.n_prompts,
        "n_prompts_used": lens.n_prompts,
        "dim_batch": a.dim_batch,
        "max_seq_len": a.max_seq_len,
        "skip_first_positions": SKIP_FIRST_N_POSITIONS,
        "device": a.device, "dtype": a.dtype,
        "n_layers": model.n_layers, "d_model": model.d_model,
        "source_layers": lens.source_layers,
        "centered": False,
        "layout_autodetected": autodetect,
        "layout": {"path": model.layout.path, "lm_head": model.layout.lm_head,
                   "norm": model.layout.norm, "embed": model.layout.embed},
        "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "env": {"transformers": transformers.__version__, "torch": torch.__version__},
        "fit_wall_seconds": round(wall, 1),
        "fit_secs_per_prompt": round(wall / max(1, lens.n_prompts), 3),
        "per_layer": [{
            "layer": l,
            "fro_norm": round(float(lens.jacobians[l].norm()), 4),
            "top_sv": round(sv0[l], 4),
            "stable_rank": round(float(lens.jacobians[l].float().norm() ** 2 / sv0[l] ** 2), 3),
        } for l in lens.source_layers],
    }
    pj = a.out.replace(".pt", "_provenance.json")
    json.dump(prov, open(pj, "w"), indent=1)
    if os.path.exists(ckpt):
        os.remove(ckpt)

    print(f"\n{a.model}  N={lens.n_prompts}  {wall:.0f}s ({prov['fit_secs_per_prompt']}s/prompt)")
    print(f"  stable_rank by layer: "
          f"{[r['stable_rank'] for r in prov['per_layer']]}")
    print(f"  wrote {a.out}  sha256={prov['sha256'][:16]}…")
    print(f"  wrote {pj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
