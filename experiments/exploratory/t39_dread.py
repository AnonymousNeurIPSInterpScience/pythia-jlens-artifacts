#!/usr/bin/env python3
"""t39_dread.py — E44 / M5: D_read, the read-relevant operator distance. Predictor #18.

WHY THIS EXISTS
  M4 (D_act) = E_{x~Q} ||(J^P - J^Q) h(x)||^2 = tr(D^T D Sigma_Q), i.e. the Frobenius norm of the
  operator difference in the metric induced by the activation second moment Sigma_Q. It corrects
  plain Frobenius, which is the Sigma_Q = I special case and so assumes isotropic activations.

  It failed twice as a predictor of read performance (E34-a, E34b), and the reason is structural.
  The readout composes TWO objects:

      logits = W_U (J h)        so a difference D moves logits by  W_U D h

  Ranks change only if W_U D h REORDERS tokens. A large ||D h|| contributes nothing if it lies in
  ker(W_U) or in directions where the scored rows of W_U are near-parallel. D_act omits W_U
  entirely: it is the right metric for "how differently do these operators act on the data" and the
  wrong one for "does that difference change what the lens reads."

  This is sharp below 410M, where E39 measured the concept-token rows of W_U at EFFECTIVE RANK 1
  (mean pairwise cosine 0.96). There W_U annihilates nearly every difference D can produce, so
  D_act can be large while the read is provably unchanged.

  D_read folds the unembedding back in, restricted to the rows the metric actually scores:

      D_read(P,Q) = E_{x~Q} || W_c (J^P - J^Q) h(x) ||^2
                  = tr( D^T W_c^T W_c D Sigma_Q )

  where W_c is W_U restricted to the ~1310 concept token ids of the eval battery. Warped by
  Sigma_Q on the input side AND W_c^T W_c on the output side.

STANDING PRIOR, stated so a positive is not over-read
  Seventeen predictors of the corpus effect have failed leave-one-out at n=5. This is the
  eighteenth. What distinguishes it: every previous candidate omitted either Sigma_Q or W_U.
  This is the first that omits neither. That is a reason to try it, NOT a reason to expect it to
  work, and it is held to exactly the control that killed the other seventeen.

PRE-REGISTRATION — fixed before any number below existed
  PRIMARY   r(D_read(J^P, I), per-corpus read gap) over the 5 corpora at 410M, with
            leave-one-corpus-out. Secondary: reference = J^eval (jeval_410m_skip0.pt).
  RULE      * |r| >= 0.8 AND survives every leave-one-corpus-out => ACCEPT, D_read predicts the
              corpus floor and is the first quantity that does.
            * fails LOO => LEVERAGE-DRIVEN, reported as such, joins the other seventeen.
            * |r| < 0.8 full-sample => NULL.
  CONTROL   C1 D_read(J, J) = 0 exactly.
  CONTROL   C2 D_read must NOT be a relabelling of D_act: report r(D_read, D_act) across the same
            corpora. If |r| > 0.95 the two are the same statistic and nothing was learned.
  CONTROL   C3 the same quantity computed with W_c replaced by a random row-subset of W_U of equal
            size. If that predicts equally well, "concept rows" is doing no work.
  BIAS      n = 5 corpora. Github has been the leverage point in four prior analyses.

    python t39_dread.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))

MODEL = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))
CORPORA = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]


def pear(x, y):
    n = len(x); mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in x)); sy = math.sqrt(sum((v - my) ** 2 for v in y))
    return sum((p - mx) * (q - my) for p, q in zip(x, y)) / (sx * sy) if sx * sy else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e44_dread_410m.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    WU = hf.get_output_embeddings().weight.data.float()

    cids, H = set(), {l: [] for l in BAND}
    with torch.no_grad():
        for name in EVAL_SETS:
            for it in load_eval(name):
                tg = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
                if not tg:
                    continue
                for t in tg:
                    cids.update(t)
                ids = model.encode(it["prompt"], max_length=256)
                pos = readout_position(tok, name, it["prompt"])
                with ActivationRecorder(model.layers, at=BAND) as r:
                    model.forward(ids)
                    for l in BAND:
                        H[l].append(r.activations[l][0][pos].detach().float())
    cids = sorted(cids)
    H = {l: torch.stack(v) for l, v in H.items()}
    Wc = WU[cids]
    g = torch.Generator().manual_seed(0)
    Wr = WU[torch.randperm(WU.shape[0], generator=g)[:len(cids)]]        # C3
    print(f"{len(cids)} concept rows, {H[BAND[0]].shape[0]} activations", flush=True)

    def dread(Ja, Jb, W):
        """E||W (Ja-Jb) h||^2, normalised by E||W Ja h||^2. Median over the band."""
        out = []
        for l in BAND:
            D = (Ja[l] - Jb[l])
            num = ((H[l] @ D.T) @ W.T).pow(2).sum(1).mean().item()
            den = ((H[l] @ Ja[l].T) @ W.T).pow(2).sum(1).mean().item()
            out.append(num / den if den > 0 else float("nan"))
        return statistics.median(out)

    def dact(Ja, Jb):
        out = []
        for l in BAND:
            D = (Ja[l] - Jb[l])
            num = (H[l] @ D.T).pow(2).sum(1).mean().item()
            den = (H[l] @ Ja[l].T).pow(2).sum(1).mean().item()
            out.append(num / den if den > 0 else float("nan"))
        return statistics.median(out)

    def lens(p):
        return {int(k): v.float() for k, v in torch.load(
            os.path.join(HERE, "..", "results", p), map_location="cpu")["J"].items()}

    I = {l: torch.eye(model.d_model) for l in BAND}
    Js = {c: lens(f"e28_{c}_410m_n400_s0.pt") for c in CORPORA}
    Jev = lens("jeval_410m_skip0.pt")

    c1 = dread(Js[CORPORA[0]], Js[CORPORA[0]], Wc)
    print(f"C1  D_read(J,J) = {c1:.3e}  {'FIRES' if abs(c1) < 1e-9 else 'DOES NOT FIRE'}", flush=True)

    rec = {"experiment": "E44 / M5 — D_read, the read-relevant operator distance (predictor #18)",
           "prereg": "in-file docstring, fixed before running", "model": MODEL, "band": BAND,
           "n_concept_rows": len(cids), "control_C1_dread_self": c1,
           "control_C1_fires": abs(c1) < 1e-9, "by_corpus": {}}
    for c in CORPORA:
        rec["by_corpus"][c] = {
            "dread_vs_I": dread(Js[c], I, Wc),
            "dread_vs_Jeval": dread(Js[c], Jev, Wc),
            "dact_vs_I": dact(Js[c], I),
            "dread_random_rows_vs_I": dread(Js[c], I, Wr),
        }
        print(f"  {c:22s} D_read(vs I)={rec['by_corpus'][c]['dread_vs_I']:.4f}  "
              f"D_read(vs Jeval)={rec['by_corpus'][c]['dread_vs_Jeval']:.4f}  "
              f"D_act(vs I)={rec['by_corpus'][c]['dact_vs_I']:.4f}", flush=True)

    e33 = json.load(open(os.path.join(HERE, "..", "results", "e33_logit_baseline_410m_v2.json")))
    L = e33["persist_admitted_mean"]["logit_I"]
    gap = {c: e33["e28_asymptotes"][c]["mean"] - L for c in CORPORA}
    rec["gap_vs_logit"] = gap

    for key in ["dread_vs_I", "dread_vs_Jeval", "dact_vs_I", "dread_random_rows_vs_I"]:
        xs = [rec["by_corpus"][c][key] for c in CORPORA]
        ys = [gap[c] for c in CORPORA]
        r = pear(xs, ys)
        loo = {c: pear([v for j, v in zip(CORPORA, xs) if j != c],
                       [v for j, v in zip(CORPORA, ys) if j != c]) for c in CORPORA}
        worst = min(loo.items(), key=lambda kv: abs(kv[1]))
        rec.setdefault("correlations", {})[key] = {
            "r": r, "loo": loo, "worst_loo": worst,
            "survives": abs(r) >= 0.8 and abs(worst[1]) >= 0.8}
        print(f"  r({key:24s}, gap) = {r:+.3f}   worst LOO {worst[1]:+.3f} (-{worst[0]})   "
              f"{'SURVIVES' if rec['correlations'][key]['survives'] else 'fails'}")

    # C2: is D_read just D_act relabelled?
    c2 = pear([rec["by_corpus"][c]["dread_vs_I"] for c in CORPORA],
              [rec["by_corpus"][c]["dact_vs_I"] for c in CORPORA])
    rec["control_C2_r_dread_vs_dact"] = c2
    rec["control_C2_distinct"] = abs(c2) <= 0.95
    print(f"\nC2  r(D_read, D_act) = {c2:+.3f}  "
          f"{'DISTINCT statistics' if abs(c2) <= 0.95 else 'SAME statistic — nothing learned'}")

    s = rec["correlations"]["dread_vs_I"]
    rec["VERDICT"] = ("ACCEPT — D_read predicts the corpus floor" if s["survives"] else
                      "LEVERAGE-DRIVEN — significant but fails LOO; joins the other seventeen"
                      if abs(s["r"]) >= 0.8 else
                      "NULL — D_read does not predict the corpus floor. Predictor #18 fails.")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"\nVERDICT: {rec['VERDICT']}\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
