#!/usr/bin/env python3
"""effrank1_adjudication.py — EFFRANK1: adjudicate eff_rank 371 (E38) vs 725.888 (E65).

PRE-REGISTRATION: docs/experiments/preregs/EFFRANK1_effective_rank_adjudication.md, written before
this file existed.
  PRIMARY  does each stored value reproduce under its OWN definition, on one shared spectrum?
  RULE     NAME COLLISION / ONE IS WRONG / BOTH WRONG — see the prereg.
  C1  recomputed mean_cos reproduces E38's to 1e-6 at all four models; n_concept_tokens == 1310.
  C2  er_entropy - er_energy90 > 1 at 410M (the definitions are not accidentally equal).
  C3  er_entropy(70M) > 1 where E38 stores eff_rank_I = 1.
  C4  spectrum integrity: sum(sigma^2) > 0 and sigma sorted descending.

Recomputation only. Fits nothing, changes no prior threshold.
    .venv/bin/python experiments/effrank1_adjudication.py
"""
from __future__ import annotations
import argparse, json, os, sys

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))
from provenance import write_result  # noqa: E402

MODELS = {"70m": "EleutherAI/pythia-70m-deduped", "160m": "EleutherAI/pythia-160m-deduped",
          "410m": "EleutherAI/pythia-410m-deduped", "1b": "EleutherAI/pythia-1b-deduped"}

# stored values under adjudication
E38_STORED = {"70m": {"mean_cos_I": 0.9585257768630981, "eff_rank_I": 1},
              "160m": {"mean_cos_I": 0.921846866607666, "eff_rank_I": 1},
              "410m": {"mean_cos_I": 0.030666086822748184, "eff_rank_I": 371},
              "1b": {"mean_cos_I": 0.019916435703635216, "eff_rank_I": 508}}
E65_STORED_410M_FINAL = {"mean_cos": 0.030666090548038483, "eff_rank": 725.887939453125}


def concept_token_ids(tok):
    """E39's exact set — identical helper to t65_ckpt_geometry.py:52-62."""
    from anchor_evals import EVAL_SETS, load_eval, token_ids_of
    ids = set()
    for name in EVAL_SETS:
        for it in load_eval(name):
            for w in it.get("intermediates", []):
                t = token_ids_of(tok, w)
                if t:
                    ids.update(t)
    return sorted(ids)


def all_stats(V: torch.Tensor) -> dict:
    """Every rank statistic, from ONE svdvals call on ONE matrix."""
    Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-9)
    G = Vn @ Vn.T
    n = G.shape[0]
    mean_cos = float((G.sum() - n) / (n * (n - 1)))

    s = torch.linalg.svdvals(V)                      # descending
    s2 = s.pow(2)
    tot2 = float(s2.sum())

    # E38, t38_jgeometry.py:167-169 — 90% energy participation, integer
    e = s2.cumsum(0) / s2.sum()
    er_energy90 = int((e < 0.90).sum().item()) + 1
    er_energy99 = int((e < 0.99).sum().item()) + 1

    # E65, t65_ckpt_geometry.py:73-75 — Roy-Vetterli entropy rank on sigma (NOT squared)
    p = s / s.sum()
    er_entropy = float(torch.exp(-(p * (p.clamp_min(1e-12)).log()).sum()))

    er_stable = float(s2.sum() / s2[0])
    return {"n": n, "mean_cos": mean_cos, "er_energy90": er_energy90,
            "er_energy99": er_energy99, "er_entropy": er_entropy,
            "er_stable": er_stable, "sigma1_energy_share": float(s2[0] / s2.sum()),
            "sigma_sum_sq": tot2,
            "sigma_descending": bool(torch.all(s[:-1] >= s[1:] - 1e-6)),
            "sigma_top5": [float(x) for x in s[:5]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    by_model = {}
    for key, mid in MODELS.items():
        tok = AutoTokenizer.from_pretrained(mid)
        idx = torch.tensor(concept_token_ids(tok), dtype=torch.long)
        hf = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32)
        W = hf.get_output_embeddings().weight.data
        st = all_stats(W[idx].float())
        st["model"] = mid
        st["e38_stored_eff_rank_I"] = E38_STORED[key]["eff_rank_I"]
        st["e38_stored_mean_cos_I"] = E38_STORED[key]["mean_cos_I"]
        st["er_energy90_reproduces_e38"] = (st["er_energy90"] == E38_STORED[key]["eff_rank_I"])
        st["mean_cos_abs_diff_vs_e38"] = abs(st["mean_cos"] - E38_STORED[key]["mean_cos_I"])
        by_model[key] = st
        print(f"{key:5s} mean_cos={st['mean_cos']:+.9f}  energy90={st['er_energy90']:4d} "
              f"(stored {E38_STORED[key]['eff_rank_I']:4d})  entropy={st['er_entropy']:9.3f}  "
              f"energy99={st['er_energy99']:4d}  stable={st['er_stable']:7.2f}  "
              f"s1share={st['sigma1_energy_share']:.4f}", flush=True)
        del hf, W

    m410 = by_model["410m"]
    entropy_diff = abs(m410["er_entropy"] - E65_STORED_410M_FINAL["eff_rank"])
    e38_ok = all(v["er_energy90_reproduces_e38"] for v in by_model.values())
    e65_ok = entropy_diff <= 1e-3

    controls = {
        "C1_same_object": {
            "required": "mean_cos abs_diff <= 1e-6 vs E38 at all four models; n == 1310 at all four",
            "max_mean_cos_abs_diff": max(v["mean_cos_abs_diff_vs_e38"] for v in by_model.values()),
            "n_concept_tokens": {k: v["n"] for k, v in by_model.items()},
            "fires": (max(v["mean_cos_abs_diff_vs_e38"] for v in by_model.values()) <= 1e-6
                      and all(v["n"] == 1310 for v in by_model.values()))},
        "C2_definitions_not_equal": {
            "required": "er_entropy - er_energy90 > 1 at 410M",
            "observed": m410["er_entropy"] - m410["er_energy90"],
            "fires": (m410["er_entropy"] - m410["er_energy90"]) > 1.0},
        "C3_rank_one_is_definitional": {
            "required": "er_entropy(70M) > 1 where E38 stores eff_rank_I = 1",
            "observed_er_entropy_70m": by_model["70m"]["er_entropy"],
            "observed_er_energy90_70m": by_model["70m"]["er_energy90"],
            "fires": by_model["70m"]["er_entropy"] > 1.0},
        "C4_spectrum_integrity": {
            "required": "sum(sigma^2) > 0 and sigma descending at every model",
            "fires": all(v["sigma_sum_sq"] > 0 and v["sigma_descending"]
                         for v in by_model.values())},
    }
    controls_fired = {k: v["fires"] for k, v in controls.items()}

    if e38_ok and e65_ok:
        branch = "NAME COLLISION"
        verdict = (
            f"NAME COLLISION — both stored numbers are correct and they are DIFFERENT STATISTICS of "
            f"the same matrix. E38's `eff_rank_I` is the 90%-energy participation count "
            f"(#{{i: cumsum(sigma^2)/sum < 0.90}} + 1), reproduced EXACTLY at all four models "
            f"(70m/160m/410m/1b = {by_model['70m']['er_energy90']}/{by_model['160m']['er_energy90']}/"
            f"{by_model['410m']['er_energy90']}/{by_model['1b']['er_energy90']}). E65's `eff_rank` is "
            f"the Roy-Vetterli entropy rank exp(-sum p log p), p = sigma/sum(sigma), reproduced at "
            f"410M to abs_diff {entropy_diff:.3e}. Neither file is defective; the field NAME is. "
            f"The paper's sentence 'The effective ranks are 1, 1, 371 and 508' is TRUE of the "
            f"90%-energy definition and must name it, because under the entropy definition the same "
            f"four matrices give "
            f"{by_model['70m']['er_entropy']:.1f}/{by_model['160m']['er_entropy']:.1f}/"
            f"{by_model['410m']['er_entropy']:.1f}/{by_model['1b']['er_entropy']:.1f} — "
            f"and in particular the 'rank-one' reading at 70M/160M is a property of the 90% cutoff, "
            f"not of the matrix.")
    elif e38_ok != e65_ok:
        branch = "ONE IS WRONG"
        verdict = (f"ONE IS WRONG — E38 reproduces: {e38_ok}; E65 reproduces: {e65_ok} "
                   f"(entropy abs_diff {entropy_diff:.3e}). STOP AND ALERT.")
    else:
        branch = "BOTH WRONG"
        verdict = (f"BOTH WRONG — neither stored value reproduces under its own definition "
                   f"(E38 exact-match {e38_ok}; E65 abs_diff {entropy_diff:.3e}). STOP AND ALERT: "
                   f"S1's geometry sentence is unsupported pending re-measurement.")

    rec = {
        "experiment": "EFFRANK1 — adjudication of eff_rank 371 (E38) vs 725.888 (E65)",
        "prereg": "docs/experiments/preregs/EFFRANK1_effective_rank_adjudication.md",
        "status": "PRE-REGISTERED",
        "recomputes_not_remeasures": True,
        "adjudicates": ["results/e38_jgeometry.json -> E39.*.eff_rank_I",
                        "results/e65_ckpt_geometry_410m.json -> by_checkpoint.step143000.eff_rank"],
        "definitions": {
            "er_energy90": "#{i : cumsum(sigma^2)_i / sum(sigma^2) < 0.90} + 1  "
                           "[experiments/t38_jgeometry.py:167-169]  INTEGER",
            "er_entropy": "exp(-sum p log p), p = sigma / sum(sigma)  "
                          "[experiments/t65_ckpt_geometry.py:73-75]  CONTINUOUS (Roy-Vetterli)",
            "er_energy99": "as er_energy90 at 0.99 — reference only",
            "er_stable": "||W||_F^2 / sigma_1^2 — reference only"},
        "concept_token_set": "E39's 1310, via concept_token_ids over all six released eval sets",
        "dtype": "float32", "device": "cpu",
        "by_model": by_model,
        "e65_stored_410m_final": E65_STORED_410M_FINAL,
        "PRIMARY": {"e38_energy90_reproduces_all_models": e38_ok,
                    "e65_entropy_reproduces_410m": e65_ok,
                    "e65_entropy_abs_diff": entropy_diff},
        "controls": controls, "controls_fired": controls_fired,
        "branch": branch, "VERDICT": verdict,
        "paper_claim_under_adjudication": {
            "location": "paper/audit_paper_5pp.tex:693",
            "text": "The effective ranks are 1, 1, 371 and 508.",
            "energy90_values": [by_model[k]["er_energy90"] for k in ["70m", "160m", "410m", "1b"]],
            "entropy_values": [by_model[k]["er_entropy"] for k in ["70m", "160m", "410m", "1b"]]},
    }
    out = a.out or os.path.join(HERE, "..", "results", "effrank1_adjudication.json")
    write_result(out, rec, experiment=rec["experiment"], inputs=[])
    print("\n" + verdict)
    print(f"\ncontrols: {controls_fired}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
