#!/usr/bin/env python3
"""fastfit.py — same estimator as jlens.fitting.fit, without the CPU round-trip bottleneck.

WHY. Measured on an RTX 4090: pythia-160m fits at 11.49 s/prompt with the GPU at ~0-24%
utilisation and the CPU pegged at 4014%. The estimator is not GPU-bound; it is bottlenecked on
per-pass host round-trips. The anchor's `jacobian_for_prompt` keeps the J accumulator in CPU
memory and does

    jacobians[layer][a:b, :] = rows.cpu()

once per (backward pass x source layer) -- that is `ceil(d/dim_batch) * n_layers` synchronising
device->host copies per prompt (132 for 160m, 480 for 1b at dim_batch=64), each of which stalls the
CUDA queue. It then accumulates d^2 per layer on the CPU and, per prompt, materialises several more
d x d CPU temporaries for a convergence diagnostic.

WHAT CHANGES (performance only -- the estimator is untouched):
  1. the running sum lives on the COMPUTE DEVICE; one host copy per checkpoint instead of
     ~n_passes * n_layers per prompt;
  2. the per-prompt buffer is allocated once and reused, not rebuilt every prompt;
  3. the convergence diagnostic is computed on-device;
  4. checkpoints are written every `checkpoint_every` prompts (default 50, not 1).

WHAT DOES NOT CHANGE: the cotangent construction, the position mask, the mean over valid
positions, the sum-over-later-targets reduction, dtype (fp32 accumulation), and the
`sum / n_done` final mean. `dim_batch` is a pure batching parameter -- the repo has already
measured `db 32 == db 64` at `worst_rel_max_diff = 0`.

THIS FILE IS A FORK OF VENDORED LOGIC AND IS THEREFORE GATED. `verify_against_anchor()`
re-runs the anchor's own `fit()` on the same prompts and asserts agreement; `--verify` runs it.
Never use fastfit for a result without a passing gate recorded in the provenance JSON.
"""
from __future__ import annotations

import math
import os
import time

import torch

from jlens.fitting import (SKIP_FIRST_N_POSITIONS, _atomic_save, _check_layer_indices,
                           valid_position_mask)
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens


def fast_fit(
    model,
    prompts,
    *,
    source_layers=None,
    target_layer=None,
    dim_batch: int = 256,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 50,
    resume: bool = True,
    device: torch.device | str | None = None,
    log_every: int = 20,
    on_prompt=None,
) -> JacobianLens:
    """Device-resident reimplementation of `jlens.fitting.fit`. Same estimator, same output."""
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(source_layers, target_layer, n_layers)
    dev = torch.device(device) if device is not None else next(
        p.device for p in model._hf_model.parameters())

    # running state, ON DEVICE
    jac_sum = {l: torch.zeros(d_model, d_model, dtype=torch.float32, device=dev)
               for l in source_layers}
    # JACOBIAN DISPERSION accumulator (RESEARCH_NOTES.tex sec3 (C0)). One scalar per layer, because
    #     E||J - J_bar||_F^2  =  E||J||_F^2 - ||J_bar||_F^2
    # so the per-prompt squared Frobenius norms are sufficient. No extra passes, no extra
    # memory beyond len(source_layers) floats. The anchor states per-example Jacobians are
    # heavy-tailed but never reports dispersion at any layer or scale.
    sumsq = {l: 0.0 for l in source_layers}
    max_prompt_fro = {l: 0.0 for l in source_layers}   # heavy-tail witness
    n_done, next_idx = 0, 0
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, expected in (("source_layers", source_layers),
                              ("target_layer", target_layer),
                              ("skip_first", skip_first)):
            if key in state and state[key] != expected:
                raise ValueError(f"checkpoint {key}={state[key]!r} != {expected!r}")
        for l in source_layers:
            jac_sum[l] = state["jacobian_sum"][l].to(dev, torch.float32)
        n_done, next_idx = state["n_done"], state["next_idx"]
        for l in source_layers:                       # dispersion state (absent in old ckpts)
            sumsq[l] = float(state.get("sumsq", {}).get(l, 0.0))
            max_prompt_fro[l] = float(state.get("max_prompt_fro", {}).get(l, 0.0))
        print(f"  resumed from checkpoint at {next_idx}/{len(prompts)} prompts", flush=True)

    # per-prompt buffer, allocated ONCE
    buf = {l: torch.zeros(d_model, d_model, dtype=torch.float32, device=dev)
           for l in source_layers}
    n_passes = math.ceil(d_model / dim_batch)
    sqrt_d = math.sqrt(d_model)

    def write_checkpoint():
        if checkpoint_path:
            _atomic_save({"jacobian_sum": {l: jac_sum[l].cpu() for l in source_layers},
                          "n_done": n_done, "next_idx": next_idx,
                          "sumsq": dict(sumsq), "max_prompt_fro": dict(max_prompt_fro),
                          "source_layers": source_layers, "target_layer": target_layer,
                          "skip_first": skip_first}, checkpoint_path)

    t_start = time.perf_counter()
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        t0 = time.perf_counter()
        input_ids = model.encode(prompt, max_length=max_seq_len)
        seq_len = input_ids.shape[1]
        try:
            position_mask = valid_position_mask(seq_len, skip_first=skip_first)
        except ValueError as exc:
            print(f"  skipping prompt {prompt_idx}: {exc}", flush=True)
            next_idx = prompt_idx + 1
            continue

        with (ActivationRecorder(model.layers, at=[*source_layers, target_layer],
                                 start_graph_at=min(source_layers)) as rec,
              torch.enable_grad()):
            model.forward(input_ids.expand(dim_batch, -1))
            target_activation = rec.activations[target_layer]
            source_activations = [rec.activations[l] for l in source_layers]

            valid_positions = position_mask.nonzero(as_tuple=True)[0].to(
                target_activation.device)
            batch_indices = torch.arange(dim_batch, device=target_activation.device)
            cotangent = torch.zeros_like(target_activation)

            for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
                n = min(dim_batch, d_model - dim_start)
                cotangent.zero_()
                cotangent[batch_indices[:n, None], valid_positions[None, :],
                          dim_start + batch_indices[:n, None]] = 1.0
                grads = torch.autograd.grad(
                    outputs=target_activation, inputs=source_activations,
                    grad_outputs=cotangent, retain_graph=(pass_idx < n_passes - 1))
                for layer, grad in zip(source_layers, grads, strict=True):
                    pos = valid_positions.to(grad.device, non_blocking=True)
                    # identical reduction to the anchor; result STAYS on device
                    buf[layer][dim_start:dim_start + n, :] = (
                        grad[:n, pos, :].float().mean(dim=1).to(dev))
                del grads

        # OPTIONAL per-prompt hook (E2 robust aggregation). Default None => zero overhead and
        # the mean path is byte-identical, so the anchor gate still certifies this function.
        if on_prompt is not None:
            on_prompt(n_done, buf)

        per_layer_fro = {l: float(buf[l].norm()) for l in source_layers}
        prompt_norm = max(per_layer_fro.values()) / sqrt_d
        for layer in source_layers:
            jac_sum[layer] += buf[layer]
            sumsq[layer] += per_layer_fro[layer] ** 2          # dispersion accumulator
            max_prompt_fro[layer] = max(max_prompt_fro[layer], per_layer_fro[layer])
        n_done += 1
        next_idx = prompt_idx + 1

        if log_every and (n_done % log_every == 0 or n_done == 1):
            el = time.perf_counter() - t_start
            print(f"  prompt {next_idx}/{len(prompts)}  seq={seq_len}  "
                  f"{time.perf_counter()-t0:.2f}s  ||J||/sqrt(d)={prompt_norm:.3f}  "
                  f"avg={el/max(1,n_done):.2f}s/prompt", flush=True)
        if checkpoint_every and next_idx % checkpoint_every == 0:
            write_checkpoint()

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    lens = JacobianLens(jacobians={l: (jac_sum[l] / n_done).cpu() for l in source_layers},
                        n_prompts=n_done, d_model=d_model)
    lens.dispersion = jacobian_dispersion(jac_sum, sumsq, max_prompt_fro, n_done, source_layers)
    return lens


def jacobian_dispersion(jac_sum, sumsq, max_prompt_fro, n_done, source_layers) -> dict:
    """Per-layer dispersion of the per-prompt Jacobians about their mean (RESEARCH_NOTES.tex sec3 (C0)).

        disp = E||J - J_bar||_F^2 / ||J_bar||_F^2 = (E||J||_F^2 - ||J_bar||_F^2) / ||J_bar||_F^2

    disp == 0 means every prompt yields the same Jacobian (the mean is a perfect local model).
    Large disp means the mean is a poor local approximation at any particular input -- which is
    the mechanism behind both the sample-size question and the OOD question.
    Also returns a heavy-tail witness: the largest single-prompt ||J||_F relative to the RMS.
    """
    out = {}
    for l in source_layers:
        mean_fro_sq = float((jac_sum[l] / n_done).norm() ** 2)
        e_fro_sq = sumsq[l] / n_done
        var = max(0.0, e_fro_sq - mean_fro_sq)
        rms = e_fro_sq ** 0.5
        out[l] = {
            "dispersion": round(var / mean_fro_sq, 6) if mean_fro_sq > 0 else float("inf"),
            "mean_fro": round(mean_fro_sq ** 0.5, 6),
            "rms_per_prompt_fro": round(rms, 6),
            "max_per_prompt_fro": round(max_prompt_fro[l], 6),
            "heavy_tail_ratio": round(max_prompt_fro[l] / rms, 4) if rms > 0 else float("inf"),
            "n_prompts": n_done,
        }
    return out


def verify_against_anchor(model, prompts, *, dim_batch_fast=256, dim_batch_ref=64,
                          max_seq_len=128, atol=2e-5, rtol=1e-4,
                          target_layer=None, skip_first=None):
    """GATE: fast_fit must reproduce the anchor's own fit() on the same prompts.

    `target_layer` / `skip_first` are threaded through to BOTH sides. Before 2026-08-10 they
    were not, so a lens fitted at `target_layer=-2` was gated against a `target_layer=final`
    reference — the gate would have validated a configuration that was never fitted.
    """
    from jlens.fitting import fit as anchor_fit

    kw = {}
    if target_layer is not None:
        kw["target_layer"] = target_layer
    if skip_first is not None:
        kw["skip_first"] = skip_first

    ref = anchor_fit(model, prompts, dim_batch=dim_batch_ref, max_seq_len=max_seq_len,
                     checkpoint_path=None, checkpoint_every=None, **kw)
    fast = fast_fit(model, prompts, dim_batch=dim_batch_fast, max_seq_len=max_seq_len,
                    checkpoint_path=None, checkpoint_every=0, log_every=0, **kw)
    report = {"dim_batch_ref": dim_batch_ref, "dim_batch_fast": dim_batch_fast,
              "target_layer_arg": target_layer, "skip_first_arg": skip_first,
              "n_prompts": len(prompts), "layers": {}}
    ok = True
    for l in ref.source_layers:
        a, b = ref.jacobians[l].float(), fast.jacobians[l].float()
        max_abs = float((a - b).abs().max())
        denom = float(a.abs().max()) or 1.0
        rel = max_abs / denom
        passed = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
        ok &= passed
        report["layers"][l] = {"max_abs_diff": max_abs, "max_rel_diff": rel,
                               "ref_absmax": denom, "pass": passed}
    report["ALL_LAYERS_AGREE"] = bool(ok)
    report["worst_max_abs_diff"] = max(v["max_abs_diff"] for v in report["layers"].values())
    report["worst_rel_diff"] = max(v["max_rel_diff"] for v in report["layers"].values())
    return report
