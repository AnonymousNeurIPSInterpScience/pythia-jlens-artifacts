#!/usr/bin/env python3
"""t0_smoke.py — T0 port gate: does the anchor's jlens run on GPT-NeoX/Pythia?

Tiny-first (discipline #6). CPU, fp32, no GPU, ~minutes. Proves the port BEFORE any
compute is scoped or provisioned. Every assertion below is a named risk from
docs/PYTHIA_SCOPE.md §T0.

Gates (all must pass):
  G1  layout auto-resolves to the gpt_neox Layout (jlens/hf.py:59-61)
  G2  UNTIED unembedding: lm_head is embed_out and is a DIFFERENT tensor from embed_in
      (Gemma-2 ties; Pythia does not -- this is the control that motivates the study)
  G3  no logit softcapping (Gemma-2 has it; Pythia must not)
  G4  BOS/attention-sink handling is known, not assumed
  G5  fit() runs; every J_l is finite, correctly shaped, non-degenerate
  G6  readout is non-trivial AND differs from the logit lens (use_jacobian=False),
      i.e. the Jacobian transport is doing something
  G7  save/load round-trips bit-exactly at fp16 storage
  G8  the position mask is satisfiable (seq_len > skip_first+1 = 17)

Usage:
    .venv/bin/python t0_smoke.py --model EleutherAI/pythia-70m-deduped \
        --out results/t0_smoke_70m.json
"""
from __future__ import annotations

import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))

from jlens.fitting import SKIP_FIRST_N_POSITIONS, fit  # noqa: E402
from jlens.hf import Layout, from_hf  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402

# PORT FINDING (2026-08-09). The anchor ships a Pythia layout at jlens/hf.py:59-61:
#     Layout("gpt_neox", norm="final_layer_norm", embed="embed_in", lm_head="embed_out")
# Under transformers 5.14.1 that FAILS: GPTNeoXForCausalLM.embed_out was renamed to
# lm_head. `gpt_neox` still exposes embed_in / layers / final_layer_norm, so the only
# stale field is lm_head. We pass the layout explicitly rather than patching vendored
# third-party code (discipline #4: vendored code is never modified in place).
PYTHIA_LAYOUT_T5 = Layout(
    "gpt_neox", layers="layers", norm="final_layer_norm",
    embed="embed_in", lm_head="lm_head",
)

# Long enough to clear skip_first=16 plus the dropped final position (G8).
PROMPTS = [
    "The history of the printing press begins in the fifteenth century, when a goldsmith "
    "in Mainz devised a system of movable metal type that could be rearranged and reused, "
    "which dramatically reduced the cost of reproducing written material across Europe.",
    "In cellular respiration, glucose is oxidised through a sequence of enzyme-catalysed "
    "reactions that release energy gradually, storing it in the chemical bonds of adenosine "
    "triphosphate rather than dissipating it all at once as waste heat inside the cell.",
    "Ocean currents redistribute heat around the planet, and the great conveyor belt of "
    "circulation carries warm surface water northward while colder, saltier water sinks in "
    "the polar regions and returns south along the deep basins of the Atlantic seafloor.",
]

READOUT_PROMPT = (
    "The capital city of France, which sits on the river Seine and has long been a centre "
    "of art and philosophy, is called"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--n-prompts", type=int, default=3)
    ap.add_argument("--dim-batch", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--out", default="results/t0_smoke.json")
    a = ap.parse_args()

    t0 = time.perf_counter()
    gates: dict[str, object] = {}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    try:
        model = from_hf(hf, tok)
        gates["G1_autodetect_worked"] = True
        gates["G1_fallback_reason"] = None
    except ValueError as exc:
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
        gates["G1_autodetect_worked"] = False
        gates["G1_fallback_reason"] = str(exc)[:200]

    # ---- G1: layout ------------------------------------------------------
    gates["G1_layout_path"] = model.layout.path
    gates["G1_layout_lm_head"] = model.layout.lm_head
    gates["G1_layout_norm"] = model.layout.norm
    gates["G1_layout_embed"] = model.layout.embed
    gates["G1_transformers_version"] = __import__("transformers").__version__
    gates["G1_PASS"] = model.layout.path == "gpt_neox"
    gates["G1_note"] = ("anchor's shipped Pythia layout uses lm_head='embed_out', which "
                        "transformers 5.x renamed to 'lm_head'; explicit Layout supplied.")

    # ---- G2: untied unembedding -----------------------------------------
    w_in = hf.gpt_neox.embed_in.weight
    w_out = model._lm_head.weight
    gates["G2_embed_in_shape"] = list(w_in.shape)
    gates["G2_embed_out_shape"] = list(w_out.shape)
    gates["G2_same_storage"] = w_in.data_ptr() == w_out.data_ptr()
    gates["G2_max_abs_diff"] = float((w_in - w_out).abs().max())
    gates["G2_PASS"] = (not gates["G2_same_storage"]) and gates["G2_max_abs_diff"] > 0
    gates["G2_note"] = ("UNTIED: W_U is an independent matrix. Gemma-2 ties, Pythia does "
                        "not -- this is the tied-vs-untied control.")

    # ---- G3: no logit softcap -------------------------------------------
    gates["G3_logit_softcap"] = model._logit_softcap
    gates["G3_PASS"] = model._logit_softcap is None

    # ---- G4: BOS / attention sink ---------------------------------------
    gates["G4_bos_token_id"] = tok.bos_token_id
    gates["G4_has_add_bos_attr"] = hasattr(tok, "add_bos_token")
    ids_probe = model.encode(READOUT_PROMPT, max_length=a.max_seq_len)
    gates["G4_first_token_id"] = int(ids_probe[0, 0])
    gates["G4_bos_actually_prepended"] = int(ids_probe[0, 0]) == tok.bos_token_id
    gates["G4_PASS"] = True  # informational: recorded, not asserted
    gates["G4_note"] = ("jlens force_bos=True sets tokenizer.add_bos_token when the attr "
                        "exists. GPT-NeoX fast tokenizers often ignore it. Recorded so the "
                        "attention-sink condition is known rather than assumed.")

    # ---- G8: position mask satisfiable ----------------------------------
    seq_lens = [int(model.encode(p, max_length=a.max_seq_len).shape[1]) for p in PROMPTS[:a.n_prompts]]
    gates["G8_skip_first"] = SKIP_FIRST_N_POSITIONS
    gates["G8_seq_lens"] = seq_lens
    gates["G8_n_valid_positions"] = [s - SKIP_FIRST_N_POSITIONS - 1 for s in seq_lens]
    gates["G8_PASS"] = all(s > SKIP_FIRST_N_POSITIONS + 1 for s in seq_lens)

    # ---- G5: fit ---------------------------------------------------------
    t_fit = time.perf_counter()
    lens = fit(
        model,
        PROMPTS[: a.n_prompts],
        dim_batch=a.dim_batch,
        max_seq_len=a.max_seq_len,
        checkpoint_path=None,
        checkpoint_every=None,
    )
    fit_secs = time.perf_counter() - t_fit

    per_layer = []
    for l in lens.source_layers:
        J = lens.jacobians[l]
        sv = torch.linalg.svdvals(J.float())
        per_layer.append({
            "layer": l,
            "shape": list(J.shape),
            "finite": bool(torch.isfinite(J).all()),
            "fro_norm": round(float(J.norm()), 4),
            "fro_over_sqrt_d": round(float(J.norm()) / model.d_model**0.5, 4),
            "top_sv": round(float(sv[0]), 4),
            "stable_rank": round(float((J.norm() ** 2 / sv[0] ** 2)), 4),
            "dist_from_identity": round(
                float((J.float() - torch.eye(model.d_model)).norm() / J.norm()), 4),
        })
    gates["G5_n_source_layers"] = len(lens.source_layers)
    gates["G5_expected_layers"] = model.n_layers - 1
    gates["G5_d_model"] = lens.d_model
    gates["G5_all_finite"] = all(r["finite"] for r in per_layer)
    gates["G5_all_nondegenerate"] = all(r["fro_norm"] > 1e-6 for r in per_layer)
    gates["G5_PASS"] = (gates["G5_all_finite"] and gates["G5_all_nondegenerate"]
                        and len(lens.source_layers) == model.n_layers - 1)
    gates["G5_fit_secs"] = round(fit_secs, 2)
    gates["G5_secs_per_prompt"] = round(fit_secs / a.n_prompts, 3)

    # ---- G6: readout, J-lens vs logit lens -------------------------------
    mid = lens.source_layers[len(lens.source_layers) // 2]
    jl, model_logits, _ = lens.apply(model, READOUT_PROMPT, layers=[mid],
                                     positions=[-1], max_seq_len=a.max_seq_len)
    ll, _, _ = lens.apply(model, READOUT_PROMPT, layers=[mid], positions=[-1],
                          max_seq_len=a.max_seq_len, use_jacobian=False)
    topk = lambda t: [tok.decode([int(i)]) for i in t[0].topk(8).indices]
    j_top, l_top, m_top = topk(jl[mid]), topk(ll[mid]), topk(model_logits)
    gates["G6_readout_layer"] = mid
    gates["G6_jlens_top8"] = j_top
    gates["G6_logitlens_top8"] = l_top
    gates["G6_model_top8"] = m_top
    gates["G6_jlens_differs_from_logitlens"] = j_top != l_top
    gates["G6_jlens_finite"] = bool(torch.isfinite(jl[mid]).all())
    gates["G6_PASS"] = gates["G6_jlens_differs_from_logitlens"] and gates["G6_jlens_finite"]

    # ---- G7: save/load round-trip ---------------------------------------
    tmp = a.out.replace(".json", "_lens.pt")
    lens.save(tmp)
    reloaded = JacobianLens.load(tmp)
    rt = max(float((lens.jacobians[l].half().float() - reloaded.jacobians[l]).abs().max())
             for l in lens.source_layers)
    gates["G7_roundtrip_max_abs_diff"] = rt
    gates["G7_lens_bytes"] = os.path.getsize(tmp)
    gates["G7_PASS"] = rt == 0.0

    all_pass = all(v for k, v in gates.items() if k.endswith("_PASS"))
    out = {
        "experiment": "T0 port gate — jlens on GPT-NeoX/Pythia",
        "status": "SMOKE (tiny-first, not a result)",
        "model": a.model,
        "n_prompts": a.n_prompts,
        "dim_batch": a.dim_batch,
        "max_seq_len": a.max_seq_len,
        "device": "cpu", "dtype": "float32",
        "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "n_layers": model.n_layers, "d_model": model.d_model,
        "ALL_GATES_PASS": all_pass,
        "gates": gates,
        "per_layer": per_layer,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    json.dump(out, open(a.out, "w"), indent=1)

    print(f"\n{'='*66}\nT0 PORT GATE — {a.model}\n{'='*66}")
    for k in sorted(g for g in gates if g.endswith("_PASS")):
        print(f"  {k:12s} {'PASS' if gates[k] else 'FAIL'}")
    print(f"\n  layout={gates['G1_layout_path']} lm_head={gates['G1_layout_lm_head']} "
          f"untied={not gates['G2_same_storage']} softcap={gates['G3_logit_softcap']}")
    print(f"  seq_lens={gates['G8_seq_lens']} valid_positions={gates['G8_n_valid_positions']}")
    print(f"  fit: {gates['G5_fit_secs']}s  ({gates['G5_secs_per_prompt']}s/prompt), "
          f"{gates['G5_n_source_layers']} layers, d={lens.d_model}")
    print(f"\n  L{mid} J-lens   : {j_top}")
    print(f"  L{mid} logit-lens: {l_top}")
    print(f"  final model     : {m_top}")
    print(f"\n  ALL GATES PASS: {all_pass}\n  wrote {a.out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
