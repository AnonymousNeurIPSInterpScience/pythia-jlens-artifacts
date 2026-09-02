#!/usr/bin/env python3
"""e65r_phase1_rstrip_rescore.py — E65R: E65 Phase 1 at the corrected readout.

PRE-REGISTRATION: docs/experiments/preregs/E65R_phase1_rstrip_rescore.md, written before this file.
  PRIMARY  C1_S: arm S at step143000 vs e48_crossover_410m_rstrip.json J|Pile-CC|s0 persist.
  RULE     C1 CLEARS AT THE CORRECTED READOUT / C1 STILL FAILS / UNINTERPRETABLE.
  C-U        arm U reproduces results/e65_ckpt_readout_410m.json to 1e-6 on 6 fields x 10 ckpts.
  C-REF      |stripped ref - unstripped ref| > tolerance.
  C-DERANGE  5 distinct derangement draws per cell at step143000, both arms.
  C-ITEMS    exactly 157 of 541 admitted items read a different token between arms.

RESCORE ONLY — loads results/e65_lenses/*.pt, refits nothing. Stored outputs are not modified.
    .venv/bin/python experiments/e65r_phase1_rstrip_rescore.py
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))
from provenance import write_result  # noqa: E402

MID = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))
CKPTS = [0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
K = [1, 2, 5, 10, 20, 50, 100]
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
N_DERANGE = 5
LENS_DIR = os.path.join(HERE, "..", "results", "e65_lenses")
RES = os.path.join(HERE, "..", "results")


def derangement(band, seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        perm = [band[i] for i in torch.randperm(len(band), generator=g).tolist()]
        if all(p != l for p, l in zip(perm, band)):
            return perm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--items-per-chunk", type=int, default=24)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of

    tok = AutoTokenizer.from_pretrained(MID)
    L, HALF = len(BAND), len(BAND) // 2

    # ---- eval items under BOTH conventions
    items_U, items_S, n_tok_differ = [], [], 0
    for name in EVAL_SETS:
        for it in load_eval(name):
            tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
            if not tgt:
                continue
            raw = it["prompt"]
            st = raw.rstrip()
            iu = tok(raw, add_special_tokens=True).input_ids
            is_ = tok(st, add_special_tokens=True).input_ids
            pu = readout_position(tok, name, raw)
            ps = readout_position(tok, name, st)
            items_U.append({"set": name, "ids": iu, "pos": pu, "tgt": tgt})
            items_S.append({"set": name, "ids": is_, "pos": ps, "tgt": tgt})
            if iu[pu] != is_[ps]:
                n_tok_differ += 1

    def index_pairs(items):
        pair_set, pair_tgt, pair_item = [], [], []
        for ii, it in enumerate(items):
            for t in it["tgt"]:
                pair_set.append(it["set"]); pair_tgt.append(t); pair_item.append(ii)
        SET_IDX = {s: torch.tensor([i for i, p in enumerate(pair_set) if p == s], dtype=torch.long)
                   for s in EVAL_SETS}
        IP = {}
        for pi, ii in enumerate(pair_item):
            IP.setdefault(ii, []).append(pi)
        return SET_IDX, IP, pair_tgt, len(pair_tgt)

    IDX_U = index_pairs(items_U)
    IDX_S = index_pairs(items_S)
    print(f"items={len(items_U)} pairs={IDX_U[3]} | read token differs on {n_tok_differ} items",
          flush=True)

    # ---- references
    e48_u = json.load(open(os.path.join(RES, "e48_crossover_410m.json")))["arms_admitted_mean"]
    e48_s = json.load(open(os.path.join(RES, "e48_crossover_410m_rstrip.json")))["arms_admitted_mean"]
    REF_U_STORED = 0.04546590892132372                     # as stored in the E65 run
    REF_S = e48_s["J|Pile-CC|s0"]["persist"]
    seeds_s = [e48_s[f"J|Pile-CC|s{i}"]["persist"] for i in range(3)]
    seed_sd_s = statistics.stdev(seeds_s)
    TOL_S = 2 * seed_sd_s
    print(f"corrected ref = {REF_S:.10f}  seed SD = {seed_sd_s:.6e}  tol = {TOL_S:.6e}", flush=True)

    stored = json.load(open(os.path.join(RES, "e65_ckpt_readout_410m.json")))["by_checkpoint"]

    rows = {"U": {}, "S": {}}
    t0 = time.time()
    for step in CKPTS:
        rev = f"step{step}"
        lp = os.path.join(LENS_DIR, f"lens_410m_{rev}_trainval.pt")
        if not os.path.exists(lp):
            rows["U"][rev] = rows["S"][rev] = {"error": "lens missing"}
            continue
        lens = torch.load(lp, map_location="cpu")
        J = {l: lens["J"][l].float() for l in BAND}

        hf = AutoModelForCausalLM.from_pretrained(MID, revision=rev,
                                                  dtype=torch.float32).to(a.device).eval()
        model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
        d = model.d_model

        for arm, items, (SET_IDX, IP, pair_tgt, P_n) in (("U", items_U, IDX_U),
                                                         ("S", items_S, IDX_S)):
            A = torch.empty(len(items), L, d)
            with torch.no_grad():
                for ii, it in enumerate(items):
                    t = torch.tensor([it["ids"]], device=a.device)
                    with ActivationRecorder(model.layers, at=BAND) as rec:
                        model.forward(t)
                        A[ii] = torch.stack([rec.activations[l][0][it["pos"]].detach().float()
                                             for l in BAND])

            def score(T):
                H = A if T is None else torch.stack([A[:, j, :] @ T[BAND[j]].T for j in range(L)], 1)
                flat = H.reshape(-1, d)
                R = torch.empty(P_n, L, dtype=torch.float32)
                with torch.no_grad():
                    for i0 in range(0, len(items), a.items_per_chunk):
                        i1 = min(i0 + a.items_per_chunk, len(items))
                        lg = model.unembed(flat[i0 * L:i1 * L]).float()
                        for ii in range(i0, i1):
                            blk = lg[(ii - i0) * L:(ii - i0 + 1) * L]
                            for pi in IP[ii]:
                                cand = torch.stack([(blk > blk[:, i:i + 1]).sum(1) + 1
                                                    for i in pair_tgt[pi]])
                                R[pi] = cand.min(0).values.float()
                mn = R.min(dim=1).values
                return {"min": torch.stack([(mn <= k).float() for k in K]).mean(0),
                        "persist": torch.stack([((R <= k).float().sum(1) >= HALF).float()
                                                for k in K]).mean(0)}

            def adm(v):
                return statistics.mean([float(v[SET_IDX[s]].mean()) for s in ADMITTED])

            js, isc = score(J), score(None)
            shufs = []
            for di in range(N_DERANGE):
                Js = {l: J[q] for l, q in zip(BAND, derangement(BAND, 7000 + di))}
                shufs.append(adm(score(Js)["persist"]))
            rows[arm][rev] = {
                "step": step, "jlens_persist": adm(js["persist"]), "jlens_min": adm(js["min"]),
                "logit_persist": adm(isc["persist"]), "logit_min": adm(isc["min"]),
                "shuf_persist_mean": statistics.mean(shufs), "shuf_persist_max": max(shufs),
                "shuf_draws": shufs,
                "clears_derangement": adm(js["persist"]) > max(shufs),
                "clears_logit": adm(js["persist"]) > adm(isc["persist"])}
            del A
        u, s = rows["U"][rev], rows["S"][rev]
        print(f"  {rev:12s} U_J={u['jlens_persist']:.6f} S_J={s['jlens_persist']:.6f} "
              f"S_logit={s['logit_persist']:.6f} S_shufmax={s['shuf_persist_max']:.6f} "
              f"clears(dr,lg)=({s['clears_derangement']},{s['clears_logit']}) "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del hf, model, J

    # ---------- controls
    FIELDS = ["jlens_persist", "jlens_min", "logit_persist", "logit_min",
              "shuf_persist_mean", "shuf_persist_max"]
    diffs = {}
    for rev, v in rows["U"].items():
        if "error" in v or rev not in stored:
            continue
        for f in FIELDS:
            diffs[f"{rev}.{f}"] = abs(v[f] - stored[rev][f])
    max_u_diff = max(diffs.values()) if diffs else float("inf")

    fin_s = rows["S"]["step143000"]
    c1_s_diff = abs(fin_s["jlens_persist"] - REF_S)
    c1_s_fires = c1_s_diff <= TOL_S

    controls = {
        "C_U_reproduces_stored": {
            "required": "abs_diff <= 1e-6 on 6 fields x 10 checkpoints vs e65_ckpt_readout_410m.json",
            "max_abs_diff": max_u_diff, "n_compared": len(diffs),
            "worst_field": max(diffs, key=diffs.get) if diffs else None,
            "fires": max_u_diff <= 1e-6},
        "C_REF_differ": {
            "required": "|stripped ref - unstripped ref| > C1 tolerance",
            "stripped_ref": REF_S, "unstripped_ref_as_stored": REF_U_STORED,
            "abs_diff": abs(REF_S - REF_U_STORED), "tolerance": TOL_S,
            "fires": abs(REF_S - REF_U_STORED) > TOL_S},
        "C_DERANGE_distinct": {
            "required": "5 distinct derangement draws per cell at step143000, both arms",
            "U_distinct": len(set(rows["U"]["step143000"]["shuf_draws"])),
            "S_distinct": len(set(rows["S"]["step143000"]["shuf_draws"])),
            "fires": (len(set(rows["U"]["step143000"]["shuf_draws"])) == 5
                      and len(set(rows["S"]["step143000"]["shuf_draws"])) == 5)},
        "C_ITEMS_157": {
            "required": "exactly 157 of 541 admitted items read a different token between arms",
            "observed_differ": n_tok_differ, "observed_total": len(items_U),
            "fires": n_tok_differ == 157 and len(items_U) == 541},
    }
    controls_fired = {k: v["fires"] for k, v in controls.items()}

    # ---------- Phase-1 rule, applied ONLY if C1_S fires
    ok = [v for v in rows["S"].values() if "error" not in v]
    steps_sorted = sorted(ok, key=lambda v: v["step"])
    tstar = None
    for i, v in enumerate(steps_sorted):
        if all(w["clears_derangement"] and w["clears_logit"] for w in steps_sorted[i:]):
            tstar = v["step"]; break
    n_clear = len([v for v in ok if v["clears_derangement"] and v["clears_logit"]])

    if not controls["C_U_reproduces_stored"]["fires"]:
        branch = "UNINTERPRETABLE"
        verdict = (f"UNINTERPRETABLE — arm U does not reproduce the stored unstripped run "
                   f"(max abs_diff {max_u_diff:.3e} > 1e-6 on "
                   f"{controls['C_U_reproduces_stored']['worst_field']}). Neither arm is comparable "
                   f"to the stored file; both are discarded.")
        phase1 = None
    elif c1_s_fires:
        branch = "C1 CLEARS AT THE CORRECTED READOUT"
        if tstar is None:
            p1 = "UNCLEAR — J never clears both at any checkpoint including the last"
        elif tstar == CKPTS[0]:
            p1 = "NO FLOOR — J clears both at the earliest measured checkpoint"
        elif n_clear < len(ok) and tstar is not None:
            p1 = f"NON-MONOTONE — clears then fails; t* = {tstar}, {n_clear}/{len(ok)} clear both"
        else:
            p1 = f"FLOOR EXISTS — t* = {tstar} > earliest measured checkpoint"
        phase1 = p1
        verdict = (f"C1 CLEARS AT THE CORRECTED READOUT — arm S final = "
                   f"{fin_s['jlens_persist']:.8f} against the corrected reference {REF_S:.8f}, "
                   f"abs_diff {c1_s_diff:.3e} <= tol {TOL_S:.3e}. E65 Phase 1's UNCLEAR was a "
                   f"readout artifact. Registered Phase-1 rule on arm S: {p1}")
    else:
        branch = "C1 STILL FAILS"
        phase1 = None
        verdict = (f"C1 STILL FAILS — arm S final = {fin_s['jlens_persist']:.8f} against the "
                   f"corrected reference {REF_S:.8f}, abs_diff {c1_s_diff:.3e} > tol {TOL_S:.3e}. "
                   f"The readout convention is NOT the explanation for E65 Phase 1's C1 failure. "
                   f"E65 Phase 1 remains UNCLEAR. Per the registration this is NOT re-run with new "
                   f"settings. A measured null: the corrected readout raises the final-checkpoint "
                   f"read from {rows['U']['step143000']['jlens_persist']:.6f} to "
                   f"{fin_s['jlens_persist']:.6f} but does not reconcile it with the E48 cell.")

    rec = {
        "experiment": "E65R — E65 Phase 1 rescored at the corrected readout",
        "prereg": "docs/experiments/preregs/E65R_phase1_rstrip_rescore.md",
        "status": "PRE-REGISTERED", "rescore_only_no_refit": True,
        "preserves": ["results/e65_ckpt_readout_410m.json",
                      "results/e65_ckpt_readout_410m_trainval.json",
                      "results/e65_ckpt_readout_1b.json"],
        "model": MID, "band": BAND, "K": K, "admitted_sets": ADMITTED,
        "checkpoints": CKPTS, "fitting_corpus": "Pile-CC", "N": 200, "seed_block": 0,
        "n_derangements": N_DERANGE, "device": a.device, "dtype": "float32",
        "operators_from": "results/e65_lenses/lens_410m_step<STEP>_trainval.pt (already fitted)",
        "n_items": len(items_U), "n_pairs": IDX_U[3],
        "n_items_read_token_differs": n_tok_differ,
        "arms": {"U": "unstripped (legacy) — reproduces the stored run",
                 "S": "STRIPPED (anchor rule) — prompt.rstrip() before tokenisation"},
        "by_checkpoint_unstripped": rows["U"], "by_checkpoint_stripped": rows["S"],
        "reference_corrected": {
            "source": "results/e48_crossover_410m_rstrip.json "
                      "-> arms_admitted_mean['J|Pile-CC|s0'].persist",
            "value": REF_S, "seed_values": seeds_s, "seed_sd": seed_sd_s, "tolerance_2sd": TOL_S},
        "reference_as_stored_in_e65": {
            "value": REF_U_STORED,
            "note": "the E65 run labels this e48_crossover_410m.json J|Pile-CC|s0 persist, but that "
                    "field currently holds 0.04509159799199551; the stored value is the 3-seed mean"},
        "PRIMARY": {"C1_S_final": fin_s["jlens_persist"], "C1_S_reference": REF_S,
                    "C1_S_abs_diff": c1_s_diff, "C1_S_tolerance": TOL_S, "C1_S_fires": c1_s_fires},
        "phase1_rule_on_arm_S": phase1,
        "t_star_steps_arm_S": tstar, "n_checkpoints_clearing_both_arm_S": n_clear,
        "controls": controls, "controls_fired": controls_fired,
        "branch": branch, "VERDICT": verdict,
        "scope_limit": "410M only. Only 1 of 10 1B operators was persisted; the 1B arm of E65 "
                       "Phase 1 remains at its stored UNCLEAR and is out of scope.",
    }
    out = a.out or os.path.join(RES, "e65r_phase1_rstrip_rescore.json")
    write_result(out, rec, experiment=rec["experiment"], inputs=[])
    print("\n" + verdict)
    print(f"\ncontrols: {controls_fired}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
