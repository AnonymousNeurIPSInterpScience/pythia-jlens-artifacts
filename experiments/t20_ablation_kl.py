#!/usr/bin/env python3
"""t20_ablation_kl.py — E20. The anchor's Figure 53 DV: ablation KL.

WHY THIS ONE. It is the cheapest write result in the paper: no oracle, no alpha, no band
compounding, and the anchor reports a concrete comparison --- J-lens ablation induces ~2x the
output KL of logit- or tuned-lens ablation on multihop. Every other write result in this
programme is retracted or inadmissible (T5 RETRACTED, T8 1 admissible cell of 8 runnable), so
"the J-lens fails at writes" is currently UNSUPPORTED: we have not tried the two cheapest write
DVs. This is one of them.

PRE-REGISTERED (RESEARCH_NOTES sec13 E20). Fixed before the first number:
  PRIMARY OUTCOME  ratio KL(J^P) / KL(I) per set, per model.
  DECISION RULE    the anchor's claim reproduces on Pythia if the ratio > 1.5 on multihop at
                   >= 410M. Ratio <= 1.0 FALSIFIES it at that scale. 1.0 < ratio <= 1.5 is
                   inconclusive and stays inconclusive.
  CONTROLS         J^shuf MANDATORY -- otherwise this cannot distinguish "the J-lens direction
                   matters" from "ablating any fitted direction of that norm matters". Plus a
                   matched-norm random direction (RIGOR axis 3: a control must be shown to fire,
                   and it is matched on ||v|| so it cannot win on dose alone).
  DOSE             ||Delta|| per layer AND per position, recorded for every arm (D12 / W4).

THE OPERATOR IS ASKED TO RULE ON ONE THING (RIGOR axis 9, and rule 7 "ask when in doubt about
Gurnee et al."). The E20 block says "zero the residual component along the intermediate's lens
vector at the readout position, all positions". That fixes WHERE along the sequence but not at
WHICH LAYER. Three readings are available and they are not equivalent:
    (a) one layer at a time, swept          <-- IMPLEMENTED, because it is the only reading that
                                                contains the other two as reductions, and because
                                                E20's stated virtue is "no band compounding"
    (b) all band layers at once (clamped)       compounds; the thing E20 was chosen to avoid
    (c) a single hand-picked layer              a researcher degree of freedom with no rule
We therefore sweep (a) and store the whole per-layer curve. The headline is the BAND MEAN of the
per-layer ratio; band-max is stored alongside and is NOT the primary (it is an existential
statistic, and sec F1 is a whole finding about what those launder).

DIRECTIONS, per transport. v_c^l = T_l^T W_U[c], i.e. `joperator.token_direction`:
    logit       T = I               v_c = W_U[c]              the free baseline; nothing fitted
    jlens       T = J_l             v_c = J_l^T W_U[c]        the real arm
    shuffled    T = J_pi(l)         pi a derangement          the decisive control
    rand_match  random direction, norm matched to the jlens arm's ||v_c|| at that layer
NOTE (external review B5/G6): the logit lens has no lens-native coordinate write. Applied this way it is
a SHARED hidden-state intervention parameterised by whichever lens supplies v_c. Reported as such.

EXPLORATORY unless --prereg is cited. 70m/160m/410m are burned per PREREG_PYTHIA_T7_v2 sec0.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time

import torch

torch.set_num_threads(min(8, os.cpu_count() or 8))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jlens.hf import Layout, from_hf                                    # noqa: E402
from jlens.lens import JacobianLens                                     # noqa: E402
from anchor_evals import (EVAL_SETS, load_eval, readout_position,       # noqa: E402
                          token_ids_of)
from joperator import (ablate_delta, ablation_kl, control_seed,        # noqa: E402
                       token_direction)
from t13_transport_controls import _derangement                         # noqa: E402
from t17_reaggregate import band_of                                     # noqa: E402

PYTHIA_LAYOUT_T5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm",
                          embed="embed_in", lm_head="lm_head")


class AblationHooks:
    """h -> h - proj_v(h) at ONE layer, at all positions from `skip_positions` on.

    Records ||Delta|| per position (and the per-layer reductions of it), which is mandatory:
    an intervention whose dose is not measured is not analysable, and a control that is not
    dose-matched can win on magnitude alone (RIGOR axis 4 --- a 21x dose gap once produced a
    finding that evaporated at matched dose).
    """

    def __init__(self, blocks, layer: int, v: torch.Tensor, skip_positions: int = 0) -> None:
        self.blocks, self.layer, self.v = blocks, layer, v.float()
        self.skip_positions = skip_positions
        self.per_position: torch.Tensor | None = None
        self.delta_norm_mean = self.delta_norm_max = self.delta_norm_sum = 0.0
        self.delta_norm_last = 0.0
        self._handles: list = []

    def __enter__(self) -> AblationHooks:
        if self._handles:
            raise RuntimeError("AblationHooks is already entered; re-entering would register a "
                               "second hook and ablate TWICE (idempotent here, but the dose "
                               "record would be wrong)")

        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            v = self.v.to(t.device)
            k = self.skip_positions
            d = torch.zeros_like(t, dtype=torch.float32)
            d[..., k:, :] = ablate_delta(t[..., k:, :], v)
            new = (t.float() + d).to(t.dtype)
            dn = d.norm(dim=-1).reshape(-1)
            self.per_position = dn.detach().cpu()
            self.delta_norm_mean = float(dn.mean())
            self.delta_norm_max = float(dn.max())
            self.delta_norm_sum = float(dn.sum())
            self.delta_norm_last = float(dn[-1])
            return new if torch.is_tensor(output) else (new, *output[1:])

        self._handles.append(self.blocks[self.layer].register_forward_hook(hook))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.no_grad()
def logits_at(model, prompt: str, position: int, hooks=None, max_length: int = 128):
    """Logits at `position` WITHOUT double-applying the final norm.

    `model.forward()` calls the bare decoder, whose forward ALREADY ends with final_layer_norm;
    `model.unembed()` applies it AGAIN (KL 2.395 nats, 50,234/50,304 ranks changed). So the
    readout is `_lm_head` alone, exactly as `joperator.final_logits` does --- this function only
    generalises that one from position -1 to an arbitrary position, and `--self-test` asserts the
    two agree bit-for-bit at position -1.
    """
    ids = model.encode(prompt, max_length=max_length)
    if hooks is None:
        out = model.forward(ids)
        hs = out if torch.is_tensor(out) else out[0]
        return model._lm_head(hs[0, position].float()).float()
    with hooks:
        out = model.forward(ids)
        hs = out if torch.is_tensor(out) else out[0]
        return model._lm_head(hs[0, position].float()).float()


def build_directions(lens: JacobianLens, layers: list[int], WU: torch.Tensor,
                     seed: int) -> tuple[dict, dict]:
    """{arm: {layer: J-like matrix or None}}; the direction itself is per-token."""
    # The model is placed on --device; the lens must follow it or every matmul
    # against WU raises a device mismatch. This path had never been exercised:
    # all six prior W3 results record device=cpu.
    dev = WU.device
    J = {l: lens.jacobians[l].float().to(dev) for l in layers}
    perm = _derangement(layers, seed)
    return ({"logit": None, "jlens": J, "shuffled": {l: J[perm[l]] for l in layers},
             "rand_match": "RANDOM"},
            {str(k): v for k, v in perm.items()})


def direction(arm, mats, layer: int, WU_row: torch.Tensor, ref_norm: float,
              gen: torch.Generator) -> torch.Tensor:
    if arm == "logit":
        return WU_row.float()
    if arm == "rand_match":
        v = torch.randn(WU_row.shape[0], generator=gen, dtype=torch.float32)
        return v / v.norm().clamp_min(1e-12) * ref_norm
    return token_direction(mats[layer], WU_row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--evals", default="all")
    ap.add_argument("--max-items", type=int, default=None,
                    help="cap items PER SET. Recorded in the JSON; None = all (R8).")
    ap.add_argument("--layers", default="band", choices=["band", "all"])
    ap.add_argument("--skip-positions", type=int, default=1,
                    help="leading positions left unablated. Default 1 excludes the "
                         "attention-sink token, whose residual norm is 7-9x every real token on "
                         "Pythia and would otherwise dominate an all-positions dose.")
    ap.add_argument("--min-pairs", type=int, default=30,
                    help="declared floor: below this many scored (item, intermediate) pairs on "
                         "multihop the run emits NOT ADJUDICABLE instead of a verdict. Smoke "
                         "runs must not be able to produce a citable number.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="assert logits_at(pos=-1) == joperator.final_logits, then exit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.device)
    try:
        model = from_hf(hf, tok)
    except ValueError:
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)

    if a.self_test:
        from joperator import final_logits
        p = "The capital of France is"
        d = float((logits_at(model, p, -1) - final_logits(model, p)).abs().max())
        print(f"self-test  max|logits_at(-1) - final_logits| = {d:.3e}")
        assert d == 0.0, "readout path diverges from the audited one"
        print("SELF-TEST PASSED")
        return 0

    lens = JacobianLens.load(a.lens)          # always loads to CPU; hooks move v per forward
    layers = list(lens.source_layers)
    band = band_of(model.n_layers, layers) if a.layers == "band" else layers
    WU = model._lm_head.weight.detach().float()
    blocks = model.model.layers if hasattr(model, "model") else model.layers
    mats, perm = build_directions(lens, layers, WU, a.seed)
    arms = ["logit", "jlens", "shuffled", "rand_match"]
    sets = EVAL_SETS if a.evals == "all" else a.evals.split(",")

    results, n_forward = {}, 0
    for ev in sets:
        items = load_eval(ev)
        if a.max_items:
            items = items[:a.max_items]
        # per (arm, layer) accumulators
        kl = {arm: {l: [] for l in band} for arm in arms}
        dose = {arm: {l: [] for l in band} for arm in arms}
        dose_pos = {arm: {l: [] for l in band} for arm in arms}
        dose_first_pair = {arm: {} for arm in arms}
        n_pairs = n_unscorable = 0
        for item in items:
            prompt = item["prompt"].rstrip()
            pos = readout_position(tok, ev, prompt)
            clean = logits_at(model, prompt, pos)
            n_forward += 1
            for word in item["intermediates"]:
                ids = token_ids_of(tok, word)
                if not ids:
                    n_unscorable += 1
                    continue
                n_pairs += 1
                c = ids[0]
                for l in band:
                    ref = float(token_direction(mats["jlens"][l], WU[c]).norm())
                    for arm in arms:
                        # control_seed, NOT an arithmetic combination: `base + layer` aliases
                        # across (base, layer) pairs and, when it omits the item, makes the
                        # random control IDENTICAL for every item (measured cosine 1.000000,
                        # zero across-item variance). `hash()` is salted per process and would
                        # also destroy reproducibility.
                        g = torch.Generator().manual_seed(
                            control_seed(a.seed, f"{ev}|{item['prompt'][:96]}|{c}", l, arm))
                        v = direction(arm, mats[arm], l, WU[c], ref, g)
                        # pass UNENTERED — logits_at enters it, exactly as final_logits does
                        h = AblationHooks(blocks, l, v, a.skip_positions)
                        lg = logits_at(model, prompt, pos, hooks=h)
                        n_forward += 1
                        kl[arm][l].append(ablation_kl(clean, lg))
                        dose[arm][l].append(h.delta_norm_mean)
                        dose_pos[arm][l].append(h.delta_norm_sum)
                        # ||Delta|| PER POSITION is mandatory, but storing it for every
                        # (item, layer, arm) would dominate the file. Store the full vector
                        # for the first scored pair of each set: auditable, bounded.
                        if n_pairs == 1:
                            dose_first_pair[arm][l] = [round(x, 4)
                                                       for x in h.per_position.tolist()]

        def mean(xs):
            return float(sum(xs) / len(xs)) if xs else float("nan")

        per_layer = {arm: {str(l): {"kl_mean": round(mean(kl[arm][l]), 6),
                                    "dose_mean_per_position": round(mean(dose[arm][l]), 4),
                                    "dose_sum_over_positions": round(mean(dose_pos[arm][l]), 4)}
                           for l in band} for arm in arms}
        band_kl = {arm: mean([mean(kl[arm][l]) for l in band]) for arm in arms}
        ratio_by_layer = {str(l): (mean(kl["jlens"][l]) / mean(kl["logit"][l])
                                   if mean(kl["logit"][l]) > 0 else float("nan")) for l in band}
        results[ev] = {
            "n_items_scored": len(items), "n_intermediate_pairs_scored": n_pairs,
            "n_intermediates_unscorable_multitoken": n_unscorable,
            "band_mean_kl": {arm: round(band_kl[arm], 6) for arm in arms},
            "PRIMARY_ratio_jlens_over_logit": round(band_kl["jlens"] / band_kl["logit"], 4)
            if band_kl["logit"] > 0 else None,
            "ratio_jlens_over_shuffled": round(band_kl["jlens"] / band_kl["shuffled"], 4)
            if band_kl["shuffled"] > 0 else None,
            "ratio_jlens_over_rand_match": round(band_kl["jlens"] / band_kl["rand_match"], 4)
            if band_kl["rand_match"] > 0 else None,
            "ratio_by_layer_jlens_over_logit": {k: round(v, 4) for k, v in ratio_by_layer.items()},
            "band_max_ratio_SECONDARY": round(max(ratio_by_layer.values()), 4)
            if ratio_by_layer else None,
            "per_layer": per_layer,
            "dose_per_position_first_pair": {arm: {str(l): v for l, v in d.items()}
                                             for arm, d in dose_first_pair.items()},
        }
        r = results[ev]["PRIMARY_ratio_jlens_over_logit"]
        print(f"  {ev:13s} band-mean KL  jlens={band_kl['jlens']:.4f} logit={band_kl['logit']:.4f} "
              f"shuf={band_kl['shuffled']:.4f} rand={band_kl['rand_match']:.4f}   ratio={r}",
              flush=True)

    # POWER FLOOR (RIGOR axis 6: "0 = n below the declared floor"). Without this the script
    # emits a confident VERDICT off a 2-item smoke -- measured: a 2-item run at 410m printed
    # "REPRODUCES, ratio 1.68" while the shuffled CONTROL was simultaneously beating the real
    # arm (0.0104 vs 0.0092). A verdict that can be produced by a smoke is a trap.
    mh = results.get("multihop", {}).get("PRIMARY_ratio_jlens_over_logit")
    mh_pairs = results.get("multihop", {}).get("n_intermediate_pairs_scored", 0)
    if mh is not None and mh_pairs < a.min_pairs:
        verdict = (f"NOT ADJUDICABLE — multihop scored {mh_pairs} (item, intermediate) pairs, "
                   f"below the declared floor of {a.min_pairs}. Ratio {mh} is a SMOKE VALUE, "
                   f"not a result; it may not be cited.")
    elif mh is None:
        verdict = "NOT ADJUDICABLE — multihop was not scored; the decision rule names it"
    elif mh > 1.5:
        verdict = f"REPRODUCES — multihop ratio {mh} > 1.5, the anchor's ~2x claim holds here"
    elif mh <= 1.0:
        verdict = f"FALSIFIED at this scale — multihop ratio {mh} <= 1.0"
    else:
        verdict = f"INCONCLUSIVE — multihop ratio {mh} in (1.0, 1.5]"

    out = {
        "experiment": "T20 / E20 — ablation KL, the anchor's Figure 53 DV",
        "status": "CONFIRMATORY" if a.prereg else "EXPLORATORY",
        "prereg": a.prereg,
        "prereg_sha256": (hashlib.sha256(open(a.prereg, "rb").read()).hexdigest()
                          if a.prereg and os.path.exists(a.prereg) else None),
        "decision_rule": ("ratio KL(J^P)/KL(I), band mean, on multihop: >1.5 reproduces the "
                          "anchor; <=1.0 falsifies at this scale; (1.0,1.5] inconclusive"),
        "primary_outcome": "band-mean ratio KL(jlens)/KL(logit) per set",
        "layer_protocol": ("one layer at a time, swept over the band; NO band compounding. "
                           "Headline is the band MEAN of the per-layer ratio; band-max is stored "
                           "as a secondary and is explicitly not primary (existential statistic)."),
        "cross_lens_caveat": ("the logit lens has no lens-native coordinate write (external review "
                              "B5/G6); applied here it is a SHARED hidden-state intervention "
                              "parameterised by whichever lens supplies v_c"),
        "VERDICT": verdict,
        "model": a.model, "lens": a.lens,
        "lens_sha256": hashlib.sha256(open(a.lens, "rb").read()).hexdigest(),
        "n_layers": model.n_layers, "source_layers": layers, "band": band,
        "layers_arg": a.layers, "skip_positions": a.skip_positions,
        "max_items_effective": a.max_items, "min_pairs_floor": a.min_pairs, "evals": sets, "arms": arms, "seed": a.seed,
        "derangement_layer_map": perm, "device": a.device, "n_forward_passes": n_forward,
        "results": results,
        "env": {"transformers": transformers.__version__, "torch": torch.__version__},
        "argv": sys.argv,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n  VERDICT: {verdict}")
    print(f"wrote {a.out} ({out['runtime_s']}s, {n_forward} forwards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
