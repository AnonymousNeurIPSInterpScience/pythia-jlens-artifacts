#!/usr/bin/env python3
"""t2_fastfit.py — driver for fastfit.fast_fit, with the anchor-agreement gate built in.

Runs `--verify` first (unless --skip-verify): fits the same 3 prompts with BOTH the anchor's
fit() and fast_fit(), and refuses to proceed unless every layer agrees. The gate report is
embedded in the provenance JSON, so no lens produced here can be cited without it.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time

import torch

torch.set_num_threads(min(8, os.cpu_count() or 8))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jlens.fitting import SKIP_FIRST_N_POSITIONS  # noqa: E402
from jlens.hf import Layout, from_hf  # noqa: E402
from fastfit import fast_fit, verify_against_anchor  # noqa: E402

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")


def load_prompts(n, min_chars=400, corpus="wikitext", offset=0, stride=1,
                 doc_split=None, tokenizer=None, full_window_tokens=None,
                 corpus_file=None, seed_block=None):
    if corpus_file is not None:
        import json as _json
        pool = [_json.loads(l)["text"] for l in open(corpus_file)]
        if seed_block is not None:
            b = len(pool) // 3
            pool = pool[seed_block * b:(seed_block + 1) * b]
        if full_window_tokens is not None and tokenizer is not None:
            pool = [t for t in pool
                    if len(tokenizer(t)["input_ids"]) >= full_window_tokens]
        if len(pool) < n:
            raise SystemExit(f"pool {corpus_file} block {seed_block} yielded "
                             f"{len(pool)}/{n} full-window prompts")
        return pool[:n]
    """Fitting corpus. `corpus` selects the DISTRIBUTION the Jacobian is averaged over —
    the independent variable of RESEARCH_NOTES.tex C2 (F2, distribution mismatch).

      wikitext      wikitext-103-raw-v1 prose (the default used for every lens so far)
      pile          the Pile validation split — Pythia's ACTUAL pretraining distribution
      code          the Pile's code subset, as a domain-OOD contrast
      eval:<name>   the anchor's own evaluation prompts for <name>, i.e. a TASK-ADJACENT fit.
                    This is the refit-on-target cell the literature has never reported.
    """
    if corpus.startswith("eval:"):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
        from anchor_evals import load_eval
        items = load_eval(corpus.split(":", 1)[1])
        out = [i["prompt"].rstrip() for i in items]
        # the fitter needs > skip_first+1 tokens; repeat the pool if the set is small
        while len(out) < n * stride + offset:
            out = out + out
        return out[offset::stride][:n]

    from datasets import load_dataset
    if corpus == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
                          streaming=True)
        key, skip = "text", (lambda t: t.startswith("="))
    elif corpus in ("pile", "code"):
        ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        key, skip = "text", (lambda t: False)
    else:
        raise ValueError(f"unknown corpus {corpus!r}")
    out, docs, doc_id = [], [], 0
    for row in ds:
        t = row[key].strip()
        if corpus == "wikitext" and t.startswith("="):
            # wikitext-103 raw writes headers as ' = Title = ' and ' = = Section = = ' --- with
            # SPACES between the equals signs. A startswith("==") guard therefore never fires and
            # every SUBSECTION would look like a new article. Count leading '=' TOKENS instead:
            # level 1 is an article, level >= 2 is a section inside it.
            level = 0
            for tokn in t.split():
                if tokn == "=":
                    level += 1
                else:
                    break
            if level == 1:
                doc_id += 1
        if corpus == "code" and not any(k in t for k in ("def ", "import ", "{", ";")):
            continue
        if full_window_tokens is not None and tokenizer is not None:
            # HARD length match: reject anything that will not fill the window
            if len(tokenizer(t)["input_ids"]) < full_window_tokens:
                continue
        if len(t) >= min_chars and not skip(t):
            out.append(t); docs.append(doc_id)
            if doc_split is None and len(out) >= n * stride + offset:
                break
            if doc_split is not None and len(out) >= 40 * n:
                break            # oversample rows, then keep only the chosen document half
    if doc_split is not None:
        uniq = sorted(set(docs))
        if len(uniq) < 4:
            raise SystemExit(f"corpus {corpus!r} exposed only {len(uniq)} documents; a "
                             "document-level split needs several")
        half = set(uniq[0::2]) if doc_split == "A" else set(uniq[1::2])
        sel = [t for t, dxx in zip(out, docs) if dxx in half]
        if len(sel) < n:
            raise SystemExit(f"document half {doc_split} yielded only {len(sel)}/{n} prompts "
                             f"from {len(half)} documents")
        print(f"  doc-split {doc_split}: {n} prompts from {len(half)} disjoint documents "
              f"(pool had {len(uniq)} docs, {len(out)} rows)")
        return sel[:n]
    need = n * stride + offset
    if len(out) < need:
        raise SystemExit(f"corpus {corpus!r} yielded only {len(out)}/{need} prompts")
    return out[offset::stride][:n]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", "--n-prompts", type=int, default=200)
    ap.add_argument("--dim-batch", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--corpus-file", default=None,
                    help="E28: a .jsonl pool built ONCE by build_corpora.py and shipped identically "
                         "to every box. Required for multi-box runs -- if each box streamed the "
                         "Pile independently, 'seed k' would denote different prompts on different "
                         "machines and the seed variance would be uninterpretable.")
    ap.add_argument("--seed-block", type=int, default=None,
                    help="which DISJOINT third of the pool this seed draws from (0,1,2). Seeds "
                         "share no prompts, so seed variance is a genuine resampling variance.")
    ap.add_argument("--require-full-window", action="store_true",
                    help="HARD LENGTH MATCH. Keep only prompts that reach max_seq_len tokens, so "
                         "every per-prompt J is averaged over EXACTLY max_seq_len - skip_first "
                         "positions in every corpus. Without this, corpora differ in mean position "
                         "count (wikitext 116.4 tok vs pile 127.4 vs code 128.0) and a corpus "
                         "contrast is partly a LENGTH contrast -- the confound that forced the F8 "
                         "narrowness retraction. Required for any cross-corpus operator claim.")
    ap.add_argument("--doc-split", default=None, choices=[None, "A", "B"],
                    help="DOCUMENT-LEVEL partition -- the CORRECT sampling floor (Prompt 7 sec2.3: "
                         "'use block or cluster resampling at the DOCUMENT level'). Assigns every "
                         "row to its source document and gives disjoint halves of the DOCUMENT set, "
                         "so the two operators share no article. Contiguous splits overstate the "
                         "floor (they contain document shift); interleaved splits understate it "
                         "(their halves share documents). Only this one is a pure sampling term.")
    ap.add_argument("--prompt-stride", type=int, default=1,
                    help="take every k-th prompt. stride=2 with offset 0/1 gives the\n                         ODD-EVEN split, which is EXCHANGEABLE. The contiguous\n                         (offset) split is NOT: F7 measured split-half disagreement at\n                         2-5x odd-even, so a contiguous floor contains ordering bias\n                         and OVERSTATES the sampling term.")
    ap.add_argument("--prompt-offset", type=int, default=0,
                    help="skip this many prompts before taking n. Makes the two "
                         "halves of a corpus DISJOINT, which the E27 sampling "
                         "floor requires -- overlapping halves would understate it.")
    ap.add_argument("--corpus", default="wikitext",
                    help="fit distribution: wikitext | pile | code | eval:<name>")
    ap.add_argument("--target-layer", type=int, default=None,
                    help="layer to differentiate TO. None = library default (final). "
                         "-2 = PENULTIMATE, which is the paper's default recipe (A.7): the "
                         "final block calibrates next-token prediction and can add noisy "
                         "artifacts. Negative indices count from the end. NOTE this changes "
                         "the source-layer count: sources must be < target, so penultimate "
                         "yields one fewer fitted layer than final.")
    ap.add_argument("--skip-first", type=int, default=None,
                    help="leading positions excluded from the position average. None = the "
                         "vendored anchor default SKIP_FIRST_N_POSITIONS = 16 (confirmed at "
                         "jlens/fitting.py:42; rationale in-code: 'early positions act as "
                         "attention sinks and have atypical residual statistics'). The PAPER "
                         "reports skipping gives no meaningful improvement and describes the "
                         "unmodified position average as its default, so 0 is the paper-literal "
                         "value. These disagree; 16 is the registered primary.")
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

    gate = None
    if not a.skip_verify:
        print("=== GATE: fast_fit vs anchor fit() on 3 prompts ===", flush=True)
        # Gate at MATCHED dim_batch: this validates the fitter, not the batch size.
        # dim_batch is separately measured to perturb J by ~1.4e-4 relative at fp32 on this
        # stack (fast db=256 vs anchor db=64), which is why the whole ladder is fit at one
        # dim_batch rather than mixing.
        gate = verify_against_anchor(model, load_prompts(3, corpus=a.corpus),
                                     dim_batch_fast=a.dim_batch,
                                     dim_batch_ref=a.dim_batch,
                                     max_seq_len=a.max_seq_len,
                                     target_layer=a.target_layer,
                                     skip_first=a.skip_first)
        print(f"  ALL_LAYERS_AGREE={gate['ALL_LAYERS_AGREE']}  "
              f"worst_abs={gate['worst_max_abs_diff']:.3e}  "
              f"worst_rel={gate['worst_rel_diff']:.3e}", flush=True)
        if not gate["ALL_LAYERS_AGREE"]:
            print("GATE FAILED — refusing to fit", flush=True)
            json.dump({"gate": gate, "status": "GATE_FAILED"},
                      open(a.out.replace(".pt", "_GATE_FAILED.json"), "w"), indent=1)
            return 2
        if a.verify_only:
            json.dump({"gate": gate, "status": "GATE_ONLY"},
                      open(a.out.replace(".pt", "_gate.json"), "w"), indent=1)
            return 0

    prompts = load_prompts(a.n_prompts, corpus=a.corpus, offset=a.prompt_offset,
                           stride=a.prompt_stride, doc_split=a.doc_split,
                           tokenizer=(model.tokenizer if a.require_full_window else None),
                           full_window_tokens=(a.max_seq_len if a.require_full_window else None),
                           corpus_file=a.corpus_file, seed_block=a.seed_block)
    ckpt = a.out.replace(".pt", "_ckpt.pt")
    kw = {}
    if a.target_layer is not None:
        kw["target_layer"] = a.target_layer
    if a.skip_first is not None:
        kw["skip_first"] = a.skip_first
    t_fit = time.perf_counter()
    lens = fast_fit(model, prompts, dim_batch=a.dim_batch, max_seq_len=a.max_seq_len,
                    checkpoint_path=ckpt, checkpoint_every=a.checkpoint_every,
                    device=a.device, **kw)
    fit_wall = time.perf_counter() - t_fit
    lens.save(a.out)

    # SVD on the accelerator. On CPU, torch.linalg.svdvals thrashes MKL threads on a
    # high-core container: 11 SVDs of 768x768 took >12 min at 3481% CPU. Measured 2026-08-10.
    _sd = a.device if a.device != "cpu" else "cpu"
    sv0 = {l: float(torch.linalg.svdvals(lens.jacobians[l].float().to(_sd))[0])
           for l in lens.source_layers}
    prov = {
        "artifact": a.out, "sha256": sha256(a.out), "bytes": os.path.getsize(a.out),
        "model": a.model, "corpus": a.corpus,
        "n_prompts_requested": a.n_prompts, "n_prompts_used": lens.n_prompts,
        "dim_batch": a.dim_batch, "max_seq_len": a.max_seq_len,
        "require_full_window": a.require_full_window,
        "corpus_file": a.corpus_file, "seed_block": a.seed_block,
        # EFFECTIVE values, not the library constant. Before 2026-08-10 this field wrote
        # SKIP_FIRST_N_POSITIONS unconditionally, so an overridden fit recorded a false value.
        "skip_first_positions": (SKIP_FIRST_N_POSITIONS if a.skip_first is None
                                 else a.skip_first),
        "skip_first_is_library_default": a.skip_first is None,
        "target_layer_arg": a.target_layer,
        # Derived from the artifact: sources are exactly 0..target-1, so this cannot drift
        # from what was actually fitted.
        "target_layer_effective": max(lens.source_layers) + 1,
        "target_layer_is_penultimate": max(lens.source_layers) + 1 == model.n_layers - 2,
        "device": a.device, "dtype": a.dtype,
        "n_layers": model.n_layers, "d_model": model.d_model,
        "source_layers": lens.source_layers, "centered": False,
        "fitter": "fastfit.py:fast_fit (device-resident accumulator)",
        "anchor_agreement_gate": gate,
        "jacobian_dispersion": getattr(lens, "dispersion", None),
        "layout_autodetected": autodetect,
        "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "env": {"transformers": transformers.__version__, "torch": torch.__version__},
        "fit_wall_seconds": round(fit_wall, 1),
        "fit_secs_per_prompt": round(fit_wall / max(1, lens.n_prompts), 3),
        "total_wall_seconds": round(time.perf_counter() - t0, 1),
        "per_layer": [{"layer": l,
                       "fro_norm": round(float(lens.jacobians[l].norm()), 4),
                       "top_sv": round(sv0[l], 4),
                       "stable_rank": round(
                           float(lens.jacobians[l].float().norm() ** 2 / sv0[l] ** 2), 3)}
                      for l in lens.source_layers],
    }
    json.dump(prov, open(a.out.replace(".pt", "_provenance.json"), "w"), indent=1)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"\n{a.model}  N={lens.n_prompts}  fit {fit_wall:.0f}s "
          f"({prov['fit_secs_per_prompt']}s/prompt)  db={a.dim_batch}")
    print(f"  stable_rank: {[r['stable_rank'] for r in prov['per_layer']]}")
    if prov.get("jacobian_dispersion"):
        d = prov["jacobian_dispersion"]
        print(f"  dispersion : {[round(d[l]['dispersion'], 3) for l in sorted(d)]}")
        print(f"  heavy_tail : {[round(d[l]['heavy_tail_ratio'], 2) for l in sorted(d)]}")
    print(f"  wrote {a.out}  sha256={prov['sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
