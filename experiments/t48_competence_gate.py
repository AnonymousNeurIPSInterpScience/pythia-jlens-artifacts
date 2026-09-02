#!/usr/bin/env python3
"""t48_competence_gate.py — E48 PREREQUISITE: is the model COMPETENT on each fitting corpus?

WHY THIS EXISTS, AND WHY IT RUNS BEFORE A SINGLE READ IS SCORED
  E48 asks whether a Jacobian lens fitted on an out-of-stream corpus Q reads WORSE than the
  unfitted logit lens. A negative read has three possible causes and they are not the same
  finding:

    1. OPERATOR COLLAPSE   J^Q is degenerate — it reads at or below the layer-derangement floor.
                           Excluded by the pre-registered DEGENERACY branch (clause c).
    2. MODEL INCOMPETENCE  the model does not model Q at all, so E_{x~Q}[dh_final/dh_l] is an
                           average of derivatives taken at points where the forward pass is
                           confused. That measures MODEL CONFUSION, not distributional bias of
                           the estimator, and it would fake a crossover.
    3. DISTRIBUTIONAL BIAS structured operator, competent model, still degrades under shift.
                           THIS is the S3 claim.

  Nothing in the programme separates (2) from (3) yet. `trainval.py` was supposed to record
  `model_cross_entropy` per fit and it recorded NaN for every E48 cell, because the CE block sits
  inside `except Exception: pass` (trainval.py:89-102) — the exact swallowed-exception failure
  mode CLAUDE.md §6.0b bans. So the competence axis has NO data. This script supplies it.

PRE-REGISTRATION — fixed and committed to this docstring BEFORE the script was run.

  PRIMARY   mean per-token cross-entropy (nats) of EleutherAI/pythia-410m-deduped on the EXACT
            prompt window each J was fitted on: the same seed block, the same >=128-token filter,
            the same first-200 prompts, truncated to 128 tokens, positions `skip_first`..127
            scored (the fitter's `skip_first=16` window).

  GATE      an OOD corpus PASSES iff
                CE(Q) <= max_instream_CE + (max_instream_CE - min_instream_CE)
            i.e. it is at most ONE in-stream spread worse than the WORST of the five Pile
            components. Rationale: E33 already demonstrated that all five in-stream corpora
            yield operators that read (Github at the logit-lens level, USPTO +75%), so the
            in-stream CE range is a MEASURED interval of "competent enough to fit a usable J".
            Github anchors the permissive end: it is code, it is the register-far outlier, and
            its operator is still structured.

  SECONDARY bits-per-byte, BPB = CE_nats * n_tokens / (ln 2 * n_utf8_bytes), reported for every
            corpus. Per-token CE is tokenizer-efficiency-dependent: a corpus that shatters into
            many cheap tokens (LaTeX, code) gets a flattering CE. BPB removes that. The gate
            adjudicates on CE as worded in the E48 handoff; BPB is reported alongside.
            **If CE and BPB disagree about which rungs pass, that is FLAGGED TO THE OPERATOR, not
            resolved here.** Reinterpreting a gate after seeing it is the failure this file exists
            to prevent.

  CONTROLS
    C1 SANITY-LOW    Pile-CC (ordinary English web text, definitely in-stream) must have the
                     lowest or near-lowest CE of the five in-stream corpora. If an in-stream
                     corpus scores like noise the CE path itself is broken.
    C2 UNIFORM-NULL  CE of a UNIFORM next-token distribution is ln(vocab) = ln(50304) = 10.826.
                     Every corpus must sit far below it. A corpus near 10.8 means the model is
                     emitting nothing, and the "CE" is not measuring what we think.
    C3 SEED-SPREAD   the 3 disjoint seed blocks of one corpus must agree to within a small
                     fraction of the between-corpus spread. If seed spread ~ corpus spread, the
                     corpus axis is not resolvable at n=200 and the gate has no power.

  DECLARED BIAS
    CE is measured on the SAME prompts the lens was fitted on. That is deliberate — the question
    is whether the model was competent AT THE POINTS WHERE THE DERIVATIVE WAS TAKEN — but it
    means this is not a held-out estimate and must not be quoted as a corpus-difficulty number.

    A low CE does not prove the corpus is in-distribution and a high CE does not prove it is out.
    M1 (containment) is the exposure instrument; this is only the competence gate.

    python t48_competence_gate.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import math
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

MODELS = {"70m": "EleutherAI/pythia-70m-deduped", "160m": "EleutherAI/pythia-160m-deduped",
          "410m": "EleutherAI/pythia-410m-deduped", "1b": "EleutherAI/pythia-1b-deduped"}

INSTREAM = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github"]
OOD = ["OOD_News_2024", "OOD_arXiv_2023", "OOD_CommonPile"]
# reported but NOT part of the gate: M1 measured these as partially present, so they are
# neither an in-stream anchor nor an eligible OOD rung.
EXTRA = ["OOD_Wikipedia_2023", "TRAP_FineWeb", "CONTROL_PubMed_2023"]


def pool_for(corpus: str, seed: int, tok, n_max: int, min_tokens: int):
    """The EXACT pool trainval.py builds — same slice, same filter, same truncation point."""
    path = os.path.join(HERE, "..", "corpora", f"{corpus}.jsonl")
    texts = [json.loads(l)["text"] for l in open(path)]
    b = len(texts) // 3
    pool = texts[seed * b:(seed + 1) * b]
    pool = [t for t in pool if len(tok(t).input_ids) >= min_tokens][:n_max]
    return pool


def ce_of_pool(model, hf, pool, *, max_seq_len: int, skip_first: int, device: str):
    """Mean per-token CE in nats over the scored window, plus the bytes/tokens for BPB.

    NO exception is swallowed. If a prompt fails, this raises — a NaN CE is exactly how the
    competence axis went missing in the first place.

    DO NOT be tempted to write `model.unembed(model.forward(ids).last_hidden_state)` here.
    jlens' `HFLensModel.forward` delegates to the TEXT MODULE (`hf_model.gpt_neox`), whose
    `last_hidden_state` has ALREADY passed through `final_layer_norm`; `unembed` applies
    `_final_norm` again. The double norm is silent and costs ~3.3 nats — measured 6.32 vs the
    true 3.02 on one English sentence at 70M. The lens READ path is unaffected (it hooks the
    transformer BLOCKS, which are pre-norm, so exactly one norm is applied); only a
    logits-from-the-final-state path like this one is exposed to it.
    """
    tot_nll, tot_tok, tot_bytes, per_doc = 0.0, 0, 0, []
    with torch.no_grad():
        for text in pool:
            ids = model.encode(text, max_length=max_seq_len)
            x = ids[0] if ids.dim() > 1 else ids
            logits = hf(ids).logits[0].float()              # [T, V] — the model's own head
            # predict token t+1 from position t; skip the first `skip_first` positions to match
            # the fitter's valid_position_mask
            lo = max(skip_first, 0)
            if logits.shape[0] - 1 <= lo:
                raise ValueError(f"prompt yields {logits.shape[0]} tokens, <= skip_first={lo}")
            pred = logits[lo:-1]
            tgt = x[lo + 1:]
            nll = torch.nn.functional.cross_entropy(pred, tgt, reduction="sum").item()
            n = tgt.numel()
            tot_nll += nll
            tot_tok += n
            per_doc.append(nll / n)
            # bytes of the SCORED span only, so BPB and CE describe the same text
            span = model.tokenizer.decode(tgt.tolist())
            tot_bytes += len(span.encode("utf-8"))
    ce = tot_nll / tot_tok
    bpb = tot_nll / (math.log(2) * tot_bytes) if tot_bytes else float("nan")
    return {"ce_nats": ce, "ppl": math.exp(ce), "bits_per_byte": bpb,
            "n_docs": len(pool), "n_scored_tokens": tot_tok, "n_scored_bytes": tot_bytes,
            "ce_per_doc_sd": statistics.stdev(per_doc) if len(per_doc) > 1 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="410m", choices=list(MODELS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-max", type=int, default=200, help="matches the E48 fit N")
    ap.add_argument("--max-seq-len", type=int, default=128, help="matches the fitter's window")
    ap.add_argument("--skip-first", type=int, default=16, help="matches the fitter's skip_first")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--corpora", default=None, help="comma list; default = in-stream + OOD + extra")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e48_competence_gate_410m.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from t2_fastfit import PYTHIA_LAYOUT_T5

    mid = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(mid)
    hf = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    vocab = model.unembed(torch.zeros(1, model.d_model)).shape[-1]

    corpora = ([c for c in a.corpora.split(",")] if a.corpora else INSTREAM + OOD + EXTRA)
    seeds = [int(s) for s in a.seeds.split(",")]

    rec = {
        "experiment": "E48 prerequisite — model-competence gate on every fitting corpus",
        "status": "PRE-REGISTERED (rule in this file's docstring, fixed before the run)",
        "model": mid, "device": a.device,
        "window": {"max_seq_len": a.max_seq_len, "skip_first": a.skip_first, "n_max": a.n_max},
        "gate_rule": ("PASS iff CE(Q) <= max_instream_CE + (max_instream_CE - min_instream_CE); "
                      "one in-stream spread above the WORST in-stream corpus"),
        "uniform_null_ce_nats": math.log(vocab), "vocab": int(vocab),
        "why_trainval_ce_is_nan": ("trainval.py:89-102 wraps the CE block in `except Exception: "
                                   "pass` and calls `model.forward(...).logits`; jlens' "
                                   "HFLensModel.forward returns the TEXT MODULE output (hidden "
                                   "states), which has no `.logits`. Every fit in the programme "
                                   "therefore recorded model_cross_entropy = NaN."),
        "second_bug_avoided_here": ("the obvious fix — unembed the returned last_hidden_state — "
                                    "is ALSO wrong: gpt_neox's last_hidden_state is already "
                                    "final-layer-normed and jlens' unembed norms again. Measured "
                                    "6.32 vs the true 3.02 nats at 70M on one English sentence. "
                                    "This script takes logits from the HF head directly. The "
                                    "lens read path is NOT affected: it hooks the transformer "
                                    "blocks, which are pre-norm."),
        "cells": {}, "by_corpus": {},
    }

    t0 = time.time()
    for c in corpora:
        if not os.path.exists(os.path.join(HERE, "..", "corpora", f"{c}.jsonl")):
            print(f"  {c}: MISSING corpus file, skipping", flush=True)
            continue
        per_seed = []
        for s in seeds:
            pool = pool_for(c, s, tok, a.n_max, a.max_seq_len)
            if len(pool) < a.n_max:
                print(f"  NOTE {c}/s{s}: pool supplies {len(pool)}/{a.n_max}", flush=True)
            r = ce_of_pool(model, hf, pool, max_seq_len=a.max_seq_len,
                           skip_first=a.skip_first, device=a.device)
            r["corpus"], r["seed"] = c, s
            rec["cells"][f"{c}|s{s}"] = r
            per_seed.append(r)
            print(f"  {c:22s} s{s}  CE={r['ce_nats']:.4f}  ppl={r['ppl']:8.2f}  "
                  f"bpb={r['bits_per_byte']:.4f}  n={r['n_docs']}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            json.dump(rec, open(a.out, "w"), indent=1)
        ces = [r["ce_nats"] for r in per_seed]
        bpbs = [r["bits_per_byte"] for r in per_seed]
        rec["by_corpus"][c] = {
            "ce_mean": statistics.mean(ces),
            "ce_seed_sd": statistics.stdev(ces) if len(ces) > 1 else 0.0,
            "ppl_mean": math.exp(statistics.mean(ces)),
            "bpb_mean": statistics.mean(bpbs),
            "bpb_seed_sd": statistics.stdev(bpbs) if len(bpbs) > 1 else 0.0,
            "tier": ("in-stream" if c in INSTREAM else "OOD" if c in OOD else "reported-only"),
        }
        json.dump(rec, open(a.out, "w"), indent=1)

    # ---------------- adjudication, exactly as pre-registered
    ins = {c: rec["by_corpus"][c]["ce_mean"] for c in INSTREAM if c in rec["by_corpus"]}
    if len(ins) == len(INSTREAM):
        lo_ce, hi_ce = min(ins.values()), max(ins.values())
        thresh = hi_ce + (hi_ce - lo_ce)
        rec["instream_ce_range"] = {"min": lo_ce, "max": hi_ce,
                                    "argmin": min(ins, key=ins.get), "argmax": max(ins, key=ins.get)}
        rec["gate_threshold_ce"] = thresh
        rec["gate"] = {c: {"ce": rec["by_corpus"][c]["ce_mean"],
                           "passes": rec["by_corpus"][c]["ce_mean"] <= thresh,
                           "margin_nats": thresh - rec["by_corpus"][c]["ce_mean"]}
                       for c in OOD if c in rec["by_corpus"]}
        # SECONDARY, disclosed: the same gate computed on bits-per-byte
        insb = {c: rec["by_corpus"][c]["bpb_mean"] for c in INSTREAM}
        lo_b, hi_b = min(insb.values()), max(insb.values())
        tb = hi_b + (hi_b - lo_b)
        rec["gate_threshold_bpb"] = tb
        rec["gate_bpb"] = {c: {"bpb": rec["by_corpus"][c]["bpb_mean"],
                               "passes": rec["by_corpus"][c]["bpb_mean"] <= tb}
                           for c in OOD if c in rec["by_corpus"]}
        agree = all(rec["gate"][c]["passes"] == rec["gate_bpb"][c]["passes"] for c in rec["gate"])
        rec["ce_bpb_agree"] = agree

        # controls
        c1 = ins["Pile-CC"] <= sorted(ins.values())[1]
        c2 = max(rec["by_corpus"][c]["ce_mean"] for c in rec["by_corpus"]) < 0.8 * math.log(vocab)
        seed_sds = [rec["by_corpus"][c]["ce_seed_sd"] for c in rec["by_corpus"]]
        corpus_spread = (max(rec["by_corpus"][c]["ce_mean"] for c in rec["by_corpus"])
                         - min(rec["by_corpus"][c]["ce_mean"] for c in rec["by_corpus"]))
        c3_ratio = (max(seed_sds) / corpus_spread) if corpus_spread else float("inf")
        rec["controls"] = {
            "C1_pilecc_lowest_or_second": {"fires": bool(c1),
                                           "pilecc_ce": ins["Pile-CC"],
                                           "instream_ces": ins},
            "C2_far_below_uniform_null": {"fires": bool(c2),
                                          "max_corpus_ce": max(rec["by_corpus"][c]["ce_mean"]
                                                               for c in rec["by_corpus"]),
                                          "uniform_null": math.log(vocab)},
            "C3_seed_spread_small": {"fires": bool(c3_ratio < 0.25),
                                     "max_seed_sd": max(seed_sds),
                                     "between_corpus_spread": corpus_spread,
                                     "ratio": c3_ratio},
        }
        failed = [c for c in rec["gate"] if not rec["gate"][c]["passes"]]
        rec["VERDICT"] = (
            "ALL THREE OOD RUNGS PASS — a negative read on them cannot be attributed to model "
            "incompetence; E48 may be scored as registered"
            if not failed else
            f"GATE FAILED for {failed} — a negative read there would be model-confusion, not "
            f"distributional bias. STOP: per the E48 handoff, clause (b) must be RE-SPECIFIED in "
            f"the pre-registration BEFORE any read is scored. Operator decision required.")
        if not agree:
            rec["VERDICT"] += ("  ||  FLAG: the CE gate and the bits-per-byte gate DISAGREE on at "
                               "least one rung. Not resolved here — operator decision.")

    write_result(a.out, rec, experiment="E48-gate",
                 inputs=[os.path.join(HERE, "..", "corpora", f"{c}.jsonl") for c in corpora
                         if os.path.exists(os.path.join(HERE, "..", "corpora", f"{c}.jsonl"))])

    print(f"\n{'corpus':24s} {'tier':14s} {'CE(nats)':>10} {'ppl':>10} {'bpb':>8} {'seedSD':>8}")
    for c, d in rec["by_corpus"].items():
        print(f"{c:24s} {d['tier']:14s} {d['ce_mean']:10.4f} {d['ppl_mean']:10.2f} "
              f"{d['bpb_mean']:8.4f} {d['ce_seed_sd']:8.4f}")
    if "gate" in rec:
        print(f"\nin-stream CE range [{rec['instream_ce_range']['min']:.4f}, "
              f"{rec['instream_ce_range']['max']:.4f}]  ->  gate threshold "
              f"{rec['gate_threshold_ce']:.4f} nats")
        for c, g in rec["gate"].items():
            print(f"  {c:24s} CE={g['ce']:.4f}  {'PASS' if g['passes'] else 'FAIL'}  "
                  f"(margin {g['margin_nats']:+.4f})")
        for k, v in rec["controls"].items():
            print(f"  control {k:32s} {'FIRES' if v['fires'] else 'DOES NOT FIRE'}")
        print(f"\nVERDICT: {rec['VERDICT']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
