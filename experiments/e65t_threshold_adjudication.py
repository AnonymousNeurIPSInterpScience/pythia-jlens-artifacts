#!/usr/bin/env python3
"""e65t_threshold_adjudication.py — E65T: is E65 Phase 0's "geometry floor" real?

PRE-REGISTRATION: docs/experiments/preregs/E65T_threshold_defect_adjudication.md, written first.
  PRIMARY  er_energy90 and mean_cos at the 7 flagged checkpoints.
  RULE     FINDING NOT REAL / FINDING REAL / PARTIAL — see the prereg.
  REGIME   read off results/e38_jgeometry.json: mean_cos >= 0.92 AND er_energy90 == 1.
           No new threshold is invented here.
  C1 recomputation reproduces E65 Phase 0 (mean_cos 1e-6, er_entropy 1e-3) at all 19 checkpoints.
  C2 the regime returns TRUE for 70m and 160m final W_U  (makes the rule able to fail).
  C3 the regime returns FALSE for 410m and 1b final W_U.
  C4 the mean_cos > 0.031 set reproduces the stored degenerate_checkpoints exactly.

    .venv/bin/python experiments/e65t_threshold_adjudication.py
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

MID = "EleutherAI/pythia-410m-deduped"
CKPTS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
         1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
STORED_THRESH = 0.031          # E65's stored threshold; used ONLY to reproduce its flagged set
REGIME_MEAN_COS = 0.92         # read off e38_jgeometry.json E39 (70m 0.9585, 160m 0.9218)
REGIME_ENERGY90 = 1            # read off e38_jgeometry.json E39 (70m 1, 160m 1)
FINAL_MODELS = {"70m": "EleutherAI/pythia-70m-deduped", "160m": "EleutherAI/pythia-160m-deduped",
                "410m": MID, "1b": "EleutherAI/pythia-1b-deduped"}


def concept_token_ids(tok):
    from anchor_evals import EVAL_SETS, load_eval, token_ids_of
    ids = set()
    for name in EVAL_SETS:
        for it in load_eval(name):
            for w in it.get("intermediates", []):
                t = token_ids_of(tok, w)
                if t:
                    ids.update(t)
    return sorted(ids)


def stats(V):
    Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-9)
    G = Vn @ Vn.T
    n = G.shape[0]
    mean_cos = float((G.sum() - n) / (n * (n - 1)))
    s = torch.linalg.svdvals(V)
    s2 = s.pow(2)
    e = s2.cumsum(0) / s2.sum()
    er_energy90 = int((e < 0.90).sum().item()) + 1
    p = s / s.sum()
    er_entropy = float(torch.exp(-(p * (p.clamp_min(1e-12)).log()).sum()))
    return {"mean_cos": mean_cos, "er_energy90": er_energy90, "er_entropy": er_entropy,
            "sigma1_energy_share": float(s2[0] / s2.sum()), "n": n}


def in_regime(st):
    return bool(st["mean_cos"] >= REGIME_MEAN_COS and st["er_energy90"] == REGIME_ENERGY90)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stored = json.load(open(os.path.join(HERE, "..", "results",
                                         "e65_ckpt_geometry_410m.json")))
    stored_ck = stored["by_checkpoint"]
    stored_degen = stored["degenerate_checkpoints"]

    tok = AutoTokenizer.from_pretrained(MID)
    idx = torch.tensor(concept_token_ids(tok), dtype=torch.long)

    rows, mc_diffs, ent_diffs = {}, [], []
    for step in CKPTS:
        rev = f"step{step}"
        hf = AutoModelForCausalLM.from_pretrained(MID, revision=rev, dtype=torch.float32)
        st = stats(hf.get_output_embeddings().weight.data[idx].float())
        st["step"] = step
        st["flagged_by_stored_rule"] = st["mean_cos"] > STORED_THRESH
        st["in_e38_degenerate_regime"] = in_regime(st)
        sd = stored_ck.get(rev, {})
        st["stored_mean_cos"] = sd.get("mean_cos")
        st["stored_eff_rank_entropy"] = sd.get("eff_rank")
        if sd:
            mc_diffs.append(abs(st["mean_cos"] - sd["mean_cos"]))
            ent_diffs.append(abs(st["er_entropy"] - sd["eff_rank"]))
        rows[rev] = st
        print(f"  {rev:12s} mean_cos={st['mean_cos']:+.6f}  energy90={st['er_energy90']:4d}  "
              f"entropy={st['er_entropy']:8.2f}  s1share={st['sigma1_energy_share']:.4f}  "
              f"flagged={st['flagged_by_stored_rule']!s:5s}  "
              f"E38regime={st['in_e38_degenerate_regime']}", flush=True)
        del hf

    # C2/C3 — the regime applied to the four final checkpoints
    regime_check = {}
    for k, mid in FINAL_MODELS.items():
        tk = AutoTokenizer.from_pretrained(mid)
        ii = torch.tensor(concept_token_ids(tk), dtype=torch.long)
        hf = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32)
        s = stats(hf.get_output_embeddings().weight.data[ii].float())
        regime_check[k] = {**s, "in_e38_degenerate_regime": in_regime(s)}
        del hf
        print(f"  FINAL {k:5s} mean_cos={s['mean_cos']:+.6f} energy90={s['er_energy90']:4d} "
              f"regime={regime_check[k]['in_e38_degenerate_regime']}", flush=True)

    flagged = [r for r in CKPTS if rows[f"step{r}"]["flagged_by_stored_rule"]]
    flagged_names = [f"step{r}" for r in flagged]
    in_reg = [f"step{r}" for r in flagged if rows[f"step{r}"]["in_e38_degenerate_regime"]]

    controls = {
        "C1_reproduces_e65_phase0": {
            "required": "mean_cos abs_diff <= 1e-6 and er_entropy abs_diff <= 1e-3, all 19",
            "max_mean_cos_abs_diff": max(mc_diffs), "max_entropy_abs_diff": max(ent_diffs),
            "n_checkpoints_compared": len(mc_diffs),
            "fires": max(mc_diffs) <= 1e-6 and max(ent_diffs) <= 1e-3},
        "C2_regime_not_vacuous": {
            "required": "E38 degenerate regime TRUE for 70m and 160m final W_U",
            "observed": {k: regime_check[k]["in_e38_degenerate_regime"] for k in ["70m", "160m"]},
            "fires": (regime_check["70m"]["in_e38_degenerate_regime"]
                      and regime_check["160m"]["in_e38_degenerate_regime"])},
        "C3_regime_not_universal": {
            "required": "E38 degenerate regime FALSE for 410m and 1b final W_U",
            "observed": {k: regime_check[k]["in_e38_degenerate_regime"] for k in ["410m", "1b"]},
            "fires": (not regime_check["410m"]["in_e38_degenerate_regime"]
                      and not regime_check["1b"]["in_e38_degenerate_regime"])},
        "C4_flagged_set_reproduces": {
            "required": f"mean_cos > {STORED_THRESH} selects exactly {stored_degen}",
            "observed": flagged_names, "stored": stored_degen,
            "fires": flagged_names == stored_degen},
    }
    controls_fired = {k: v["fires"] for k, v in controls.items()}

    if len(in_reg) == 0:
        branch = "FINDING NOT REAL"
        verdict = (
            f"FINDING NOT REAL — 0 of the {len(flagged_names)} checkpoints flagged by E65's stored "
            f"0.031 rule is in the degenerate regime as E38/E39 exhibits it (mean_cos >= 0.92 AND "
            f"90%-energy rank == 1). At the flagged checkpoints mean_cos is "
            f"{min(rows[c]['mean_cos'] for c in flagged_names):.4f}-"
            f"{max(rows[c]['mean_cos'] for c in flagged_names):.4f} against the regime's 0.92, and "
            f"the 90%-energy rank is "
            f"{min(rows[c]['er_energy90'] for c in flagged_names)}-"
            f"{max(rows[c]['er_energy90'] for c in flagged_names)} against the regime's 1. The E65 "
            f"pre-registration's self-flag is CONFIRMED: the 0.031 threshold does not select the "
            f"regime it was justified by. The 'geometry floor' is an artifact of the threshold. "
            f"paper/audit_paper_5pp.tex:918 ('degenerate from step 1000 through step 64000') "
            f"restates it as fact and must be corrected or disclosed. No new threshold is proposed.")
    elif len(in_reg) == len(flagged_names):
        branch = "FINDING REAL"
        verdict = (f"FINDING REAL — all {len(flagged_names)} flagged checkpoints satisfy the E38 "
                   f"degenerate regime. The prereg's self-flag is withdrawn.")
    else:
        branch = "PARTIAL"
        verdict = (f"PARTIAL — {len(in_reg)} of {len(flagged_names)} flagged checkpoints satisfy "
                   f"the E38 degenerate regime: {in_reg}. No floor claim either way.")

    rec = {
        "experiment": "E65T — adjudication of E65 Phase 0's geometry-floor threshold",
        "prereg": "docs/experiments/preregs/E65T_threshold_defect_adjudication.md",
        "status": "PRE-REGISTERED", "recomputes_not_remeasures": True,
        "adjudicates": ["results/e65_ckpt_geometry_410m.json -> degenerate_checkpoints",
                        "docs/experiments/preregs/E65_training_axis_floor.md "
                        "(PRE-REGISTRATION DEFECT, FLAGGED NOT FIXED)",
                        "paper/audit_paper_5pp.tex:918"],
        "model": MID, "checkpoints": CKPTS, "dtype": "float32", "device": "cpu",
        "stored_threshold_reproduced_only": STORED_THRESH,
        "e38_degenerate_regime": {
            "definition": "mean_cos >= 0.92 AND er_energy90 == 1",
            "source": "results/e38_jgeometry.json -> E39 (70m/160m exhibit it; 410m/1b do not)",
            "no_new_threshold_invented": True},
        "by_checkpoint": rows, "final_checkpoint_regime_check": regime_check,
        "PRIMARY": {"flagged_by_stored_rule": flagged_names,
                    "n_flagged": len(flagged_names),
                    "flagged_and_in_e38_regime": in_reg,
                    "n_flagged_and_in_regime": len(in_reg)},
        "controls": controls, "controls_fired": controls_fired,
        "branch": branch, "VERDICT": verdict,
    }
    out = a.out or os.path.join(HERE, "..", "results", "e65t_threshold_adjudication.json")
    write_result(out, rec, experiment=rec["experiment"], inputs=[])
    print("\n" + verdict)
    print(f"\ncontrols: {controls_fired}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
