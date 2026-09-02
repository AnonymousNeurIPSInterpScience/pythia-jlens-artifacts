#!/usr/bin/env python3
"""test_anchor_fidelity.py — assert our readout IS the anchor's readout, numerically.

RIGOR_SKILL axis 8 (numerical faithfulness) demands assertion against an independent
reference, not a spot-check. Before this file, nothing in `tests/` compared our
pass@k readout path against either (a) the anchor's own `JacobianLens.apply`, or (b) the
model's true logits from `transformers`. The double-final-norm bug (KL 2.395 nats, 50,234
of 50,304 ranks changed) survived every review precisely because no such test existed.

Three assertions, each against a reference we did not write:

  A1  our J-lens readout   == jlens.lens.JacobianLens.apply(..., use_jacobian=True)
  A2  our logit-lens readout == jlens.lens.JacobianLens.apply(..., use_jacobian=False)
  A3  the anchor's own `model_logits` == hf(input_ids).logits at the same position
      (i.e. the vendored readout chain final_norm -> lm_head is the real one)

A1/A2 pin US to the anchor. A3 pins the ANCHOR to transformers. Together they close the
chain from raw residual to scored rank.

Run: .venv/bin/python tests/test_anchor_fidelity.py
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from jlens.hf import Layout, from_hf  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402
from anchor_evals import (load_eval, readout, readout_position,  # noqa: E402
                          transport_jlens, transport_logit)

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")
MODEL = "EleutherAI/pythia-70m-deduped"
# Lenses were regrouped under results/lenses/<family>/ on 2026-08-31; the ladder operators
# live in results/lenses/ladder/.
LENS = os.path.join(_HERE, "..", "results", "lenses", "ladder", "lens_70m_n200_db128.pt")
MAX_SEQ = 256

_n_pass = 0
_n_fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _n_pass, _n_fail
    if cond:
        _n_pass += 1
        print(f"  ok  {name}   [{detail}]")
    else:
        _n_fail += 1
        print(f"  FAIL {name}   [{detail}]")


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    try:
        model = from_hf(hf, tok)
    except ValueError:
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    lens = JacobianLens.load(LENS)
    layers = list(lens.source_layers)

    # Real eval prompts, not toys: the readout position rule is part of what we are testing.
    prompts = [it["prompt"].rstrip() for it in load_eval("multihop")[:3]]
    prompts += [it["prompt"].rstrip() for it in load_eval("poetry")[:2]]   # last_newline rule
    names = ["multihop"] * 3 + ["poetry"] * 2

    T_j = transport_jlens(lens, model=model)
    T_i = transport_logit()

    worst_j = worst_i = worst_gt = 0.0
    worst_rank_j = worst_rank_i = 0
    for ev, prompt in zip(names, prompts):
        pos = readout_position(model.tokenizer, ev, prompt)

        ours_j = readout(model, T_j, prompt, layers, pos, max_seq_len=MAX_SEQ)
        ours_i = readout(model, T_i, prompt, layers, pos, max_seq_len=MAX_SEQ)

        ref_j, model_logits, input_ids = lens.apply(
            model, prompt, layers=layers, positions=[pos], max_seq_len=MAX_SEQ,
            use_jacobian=True)
        ref_i, _, _ = lens.apply(
            model, prompt, layers=layers, positions=[pos], max_seq_len=MAX_SEQ,
            use_jacobian=False)

        for l in layers:
            a, b = ours_j[l], ref_j[l][0]
            worst_j = max(worst_j, float((a - b).abs().max()))
            worst_rank_j = max(worst_rank_j, int((a.argsort(descending=True)[:10]
                                                  != b.argsort(descending=True)[:10]).sum()))
            c, d = ours_i[l], ref_i[l][0]
            worst_i = max(worst_i, float((c - d).abs().max()))
            worst_rank_i = max(worst_rank_i, int((c.argsort(descending=True)[:10]
                                                  != d.argsort(descending=True)[:10]).sum()))

        # A3: the anchor's own final-layer readout vs the real model.
        with torch.no_grad():
            true = hf(input_ids).logits[0, pos].float().cpu()
        worst_gt = max(worst_gt, float((model_logits[0] - true).abs().max()))

    check("A1_jlens_readout_matches_anchor_apply", worst_j < 1e-3,
          f"max|diff| = {worst_j:.2e} over {len(prompts)} prompts x {len(layers)} layers")
    check("A1b_jlens_top10_order_identical", worst_rank_j == 0,
          f"{worst_rank_j} top-10 positions differ")
    check("A2_logit_readout_matches_anchor_apply", worst_i < 1e-3,
          f"max|diff| = {worst_i:.2e}")
    check("A2b_logit_top10_order_identical", worst_rank_i == 0,
          f"{worst_rank_i} top-10 positions differ")
    check("A3_anchor_final_readout_matches_hf_logits", worst_gt < 5e-3,
          f"max|diff| vs hf(...).logits = {worst_gt:.2e}")

    # A4: the double-norm bug must STAY dead. unembed(forward(...)) double-applies the
    # final LayerNorm; assert it is measurably different from the true logits, so that a
    # future refactor cannot silently reintroduce it and pass the suite.
    ids = model.encode(prompts[0], max_length=MAX_SEQ)
    with torch.no_grad():
        out = model.forward(ids)
        hs = out if torch.is_tensor(out) else out[0]
        doubled = model.unembed(hs[0, -1]).float().cpu()
        true_last = hf(ids).logits[0, -1].float().cpu()
    kl = torch.nn.functional.kl_div(
        torch.log_softmax(doubled, -1), torch.log_softmax(true_last, -1),
        log_target=True, reduction="sum")
    check("A4_double_norm_is_still_detectably_wrong", float(kl) > 0.1,
          f"KL(double-normed || true) = {float(kl):.3f} nats — the bug is detectable, "
          f"so this test would catch its return")

    # ---------------------------------------------------------------------------- A5
    # THE GAP THIS CLOSES. A1/A2 gate `anchor_evals.readout`, which transports ONE position
    # vector at a time. But NO SCORER USES `readout`: trainval.py, t33, t48, t36 and t52 all
    # inline a BATCHED path, `model.unembed(acts[l] @ T[l].T)`, over [items x layers] at once.
    # The two are the same algebra, and the batched one changes fp32 reduction order. Until this
    # assertion existed the anchored path and the path that produced every number in the
    # programme were joined only by inspection.
    from anchor_evals import EVAL_SETS  # noqa: E402
    import torch as _t
    BAND5 = [l for l in lens.source_layers][:5]
    max_batch_diff = 0.0
    max_rank_diff = 0
    max_top10_diff = 0
    for name in ("multihop", "typo"):
        for it in load_eval(name)[:4]:
            prompt = it["prompt"]
            pos = readout_position(tok, name, prompt)
            ref = readout(model, transport_jlens(lens), prompt, BAND5, pos, MAX_SEQ)
            # the scorers' path: stack the band, one matmul per layer, one batched unembed
            ids = model.encode(prompt, max_length=MAX_SEQ)
            from jlens.hooks import ActivationRecorder as _AR
            with _t.no_grad(), _AR(model.layers, at=BAND5) as rec:
                model.forward(ids)
                A = _t.stack([rec.activations[l][0][pos].detach().float() for l in BAND5])
                H = _t.stack([A[j] @ lens.jacobians[BAND5[j]].float().T
                              for j in range(len(BAND5))])
                batched = model.unembed(H).float().cpu()
            for j, l in enumerate(BAND5):
                d = float((batched[j] - ref[l]).abs().max())
                max_batch_diff = max(max_batch_diff, d)
                r1 = int(ref[l].argmax()); r2 = int(batched[j].argmax())
                max_rank_diff += int(r1 != r2)
                t1 = ref[l].topk(10).indices.tolist()
                t2 = batched[j].topk(10).indices.tolist()
                max_top10_diff += sum(1 for x, y in zip(t1, t2) if x != y)
    # THRESHOLD, derived rather than tuned. A3 above measures the anchor's OWN readout against
    # hf(...).logits at 2.32e-03, so 2.3e-3 IS the fp32 noise floor of this readout chain.
    # Demanding the batched path agree with `readout` to 1e-3 would demand it be tighter than the
    # anchor is with transformers, which is not a meaningful bar. The metric is RANK-based, so the
    # assertion that matters is that no rank moves; the logit delta is reported descriptively.
    check("A5_batched_scorer_path_preserves_top1", max_rank_diff == 0,
          f"{max_rank_diff} top-1 disagreements, but {max_top10_diff} top-10 ORDER differences "
          f"(so this is NOT rank-identity beyond top-1) "
          f"over 8 prompts x 5 layers; max|logit diff| = {max_batch_diff:.2e}, at the same "
          f"2.3e-3 fp32 floor A3 measures for the anchor itself. This is the path every scorer "
          f"actually uses, and until this assertion it was joined to the anchored path only by "
          f"inspection.")

    # ---------------------------------------------------------------- A6: the FITTER gate
    # paper.tex cites this file for "our fitter equals jlens.fitting.fit at 2e-05". Until now that
    # claim pointed at nothing: 2e-05 is the `atol` DEFAULT of fastfit.verify_against_anchor(), no
    # test called it, and no results file stored its output. A tolerance is not a measurement.
    # This runs the gate and reports the number, so the sentence has something behind it.
    from fastfit import verify_against_anchor

    # Prompts must be LONG ENOUGH to be in the regime the fitter actually runs in. skip_first=16
    # drops any prompt with <= 17 tokens, and a fit over the 2 survivors of a 6-prompt eval sample
    # disagrees at 1.4e-04 purely from fp32 accumulation order on a handful of positions. Every
    # real fit uses 128-token windows, so build 128-token windows.
    pool = [it["prompt"].rstrip()
            for n in ("multihop", "poetry", "multilingual") for it in load_eval(n)]
    fit_prompts, _i = [], 0
    while len(fit_prompts) < 8 and _i < len(pool):
        s = ""
        while len(tok(s).input_ids) < 130 and _i < len(pool):
            s += pool[_i] + "\n"; _i += 1
        fit_prompts.append(s)
    # THE FITTER EQUIVALENCE IS NOT BIT-EXACT, AND IT IS THREAD-DEPENDENT. Measured on this
    # machine: torch.set_num_threads(8) gives max|diff| = 0.00e+00, and (10) gives 1.09e-04, on
    # byte-identical prompts with no hooks attached and grad enabled in both. fast_fit is a
    # device-resident reimplementation and the anchor round-trips through CPU, so the two differ
    # in fp32 reduction order; whether that difference cancels depends on how BLAS splits the
    # reduction. `verify_against_anchor`'s atol=2e-05 default therefore PASSES OR FAILS BY THREAD
    # COUNT, and nothing had ever called it, so nobody found out.
    #
    # This test asserts the ENVELOPE across thread counts rather than a number that happens to
    # hold at one of them. The tolerance is set from the measured worst case, and the measured
    # value is printed so a real regression (orders of magnitude, not reduction order) is visible.
    _saved_threads = torch.get_num_threads()
    envelope, per_nt = 0.0, {}
    try:
        for _nt in (8, 10):
            torch.set_num_threads(_nt)
            r = verify_against_anchor(model, fit_prompts, dim_batch_fast=64, dim_batch_ref=32,
                                      max_seq_len=128, target_layer=-2, skip_first=16)
            per_nt[_nt] = r["worst_rel_diff"]
            envelope = max(envelope, r["worst_rel_diff"])
            rep = r
    finally:
        torch.set_num_threads(_saved_threads)
    check("A6_fastfit_matches_anchor_fitting_fit_up_to_fp32_reduction_order", envelope < 1e-3,
          f"worst RELATIVE diff over thread counts {sorted(per_nt)} = "
          + ", ".join(f"{k}t:{v:.2e}" for k, v in sorted(per_nt.items()))
          + f"; envelope {envelope:.2e} < 1e-3, over {len(rep['layers'])} layers, "
            f"{rep['n_prompts']} prompts, target_layer=-2, skip_first=16. NOTE: this is NOT the "
            f"2e-05 that paper.tex asserts — 2e-05 is verify_against_anchor's atol DEFAULT, it "
            f"was never run, and it is not achievable at every thread count. The defensible "
            f"claim is agreement up to fp32 reduction order at the 1e-4 level.")

    print(f"\n=== {_n_pass}/{_n_pass + _n_fail} PASSED ===")
    return 1 if _n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
