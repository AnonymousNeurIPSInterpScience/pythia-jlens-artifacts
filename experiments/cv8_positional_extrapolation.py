#!/usr/bin/env python3
"""
CV8 — POSITIONAL EXTRAPOLATION.  Is the corpus effect an extrapolation artifact?

PRE-REGISTERED: docs/experiments/preregs/CV8_positional_extrapolation.md, committed at 8b249af
BEFORE this file existed.  Read the registration first; the decision rule there is fixed and this
script does not restate it in a form that could drift from it.

THE QUESTION.  CV2 established that the fitter averages the Jacobian over positions 16..126 of a
128-token window, and that with a 128-token read prefix 0 of 551 items are read inside that range
(median read position 142).  Every corpus number in the paper is therefore produced by an operator
applied outside its own positional support, and the paper concedes in its Discussion that the
finding "may be that one corpus yields an operator extrapolating more tolerantly to position 142".

This script settles that.  It measures, for each of the eight fitting corpora, how well that
corpus's operator aligns with the transport ACTUALLY REQUIRED at each position, and asks whether
that alignment orders the corpora the way their read scores do.

WHY IT IS CHEAP, which is why not running it was indefensible.  jlens/fitting.py:198 computes
`grad[:n, positions, :]` -- a per-source-position tensor -- and immediately calls `.mean(dim=1)`.
The position-resolved Jacobian is computed by the released fitter and thrown away.  Slicing it
instead of averaging it costs the same forward and backward passes.

SCALE-FREE ON PURPOSE.  The released estimator sums over LATER target positions, so ||J(p)||_F
falls as p grows for a reason that has nothing to do with the science: there are fewer later
targets.  Measured in the smoke at 70M: 188.0 at p=16 down to 46.6 at p=180.  A magnitude-sensitive
metric would report that arithmetic as a finding.  C1 asserts the metric is invariant to it.

  .venv/bin/python experiments/cv8_positional_extrapolation.py --smoke     # tiny, CPU, minutes
  .venv/bin/python experiments/cv8_positional_extrapolation.py             # full, CPU, ~2 h
"""
import argparse, itertools, json, math, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

MODEL_ID = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))                 # floor(0.38*24)..floor(0.92*24); tests/test_band_rule.py
PREFIX_TOKENS = 128
SEQ_LEN = 192                             # long enough that probe position 177 exists
SKIP_FIRST = 16

# The probe grid, taken from CV2's MEASURED read distribution and fixed in the registration.
IN_SUPPORT = [16, 24, 32, 48, 64, 80, 96, 112, 126]
OUT_SUPPORT = [133, 142, 150, 160, 177]           # prefixed reads span 133..177, median 142
PROBE = IN_SUPPORT + OUT_SUPPORT
P_READ = 142

# The eight-corpus panel r9 adjudicates. Left = the name used in the register and in r9; right =
# the operator file stem, which carries an INSTREAM_/OOD_ prefix the register does not.
PANEL = {
    "Github":            "INSTREAM_Github",
    "Pile-CC":           "INSTREAM_Pile-CC",
    "StackExchange":     "INSTREAM_StackExchange",
    "USPTO_Backgrounds": "INSTREAM_USPTO_Backgrounds",
    "Wikipedia_en":      "INSTREAM_Wikipedia_en",
    "OOD_arXiv_2023":    "OOD_arXiv_2023",
    "OOD_CommonPile":    "OOD_CommonPile",
    "OOD_News_2024":     "OOD_News_2024",
}


# --------------------------------------------------------------------------- statistics
def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):                     # average ties, so a tie cannot fake a perfect rho
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else 0.0


def exact_permutation(a, b):
    """Exact two-sided p and the alpha=0.05 critical |rho|, by enumerating all n! relabelings.

    n=8 gives 40,320 permutations, so the null is ENUMERATED, not approximated and not read off a
    table. DN1 is in this programme's register because a threshold was once set without being
    calibrated; this one is calibrated here.
    """
    obs = spearman(a, b)
    null = [spearman(a, list(p)) for p in itertools.permutations(b)]
    absnull = sorted(abs(x) for x in null)
    n = len(absnull)
    p_two = sum(1 for x in absnull if x >= abs(obs) - 1e-12) / n
    crit = absnull[min(n - 1, int(math.ceil(0.95 * n)) - 1)]
    return obs, p_two, crit, n


def cos_f(A, B):
    """Frobenius cosine. Scale-free in BOTH arguments by construction; C1 asserts it."""
    a = A.reshape(-1).double()
    b = B.reshape(-1).double()
    na, nb = a.norm(), b.norm()
    return float((a @ b) / (na * nb)) if na > 0 and nb > 0 else 0.0


# --------------------------------------------------------------------------- the measurement
def build_items(tok, rstrip=True):
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of
    items = []
    for name in EVAL_SETS:
        for it in load_eval(name):
            tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
            if not tgt:
                continue
            p = it["prompt"].rstrip() if rstrip else it["prompt"]
            ids = tok(p, add_special_tokens=True).input_ids
            pos = readout_position(tok, name, p)
            items.append({"set": name, "ids": ids, "pos": pos, "tgt": tgt})
    bad = [i for i, it in enumerate(items) if not it["ids"] or it["ids"][0] != tok.bos_token_id]
    if bad:
        raise SystemExit(f"ABORT: {len(bad)}/{len(items)} items carry no BOS; every cell would be "
                         f"scored on a different sequence than the stored runs.")
    return items


def load_prefix_pool(corpus, tok, n, seed=0):
    """Same construction as t36_qladder.load_pool, so the sequences match the stored ladder."""
    path = os.path.join(ROOT, "corpora", f"{corpus}.jsonl")
    texts = [json.loads(l)["text"] for l in open(path)]
    g = torch.Generator().manual_seed(9000 + seed)
    idx = torch.randperm(len(texts), generator=g).tolist()
    out = []
    for i in idx:
        ids = tok(texts[i]).input_ids
        if len(ids) >= PREFIX_TOKENS:
            out.append(ids[:PREFIX_TOKENS])
        if len(out) >= n:
            break
    if len(out) < n:
        out = (out * (n // max(1, len(out)) + 1))[:n]
    return out


def prefixed_sequence(item, prefix_ids, bos_id):
    body = item["ids"][1:] if item["ids"] and item["ids"][0] == bos_id else item["ids"]
    ids = [bos_id] + list(prefix_ids) + list(body)
    pos = item["pos"] if item["pos"] < 0 else item["pos"] + len(prefix_ids)
    return ids, pos


@torch.no_grad()
def _noop():
    pass


def position_jacobians(model, ids, band, probe, device):
    """{layer: [len(probe), d_model, d_model]} — the estimator SLICED at probe positions.

    This is jlens.fitting.jacobian_for_prompt with one line changed: where it does
    `.mean(dim=1)` over valid positions, this keeps the positions. Same forward, same backwards,
    same cotangent construction, same fp32 accumulation.
    """
    from jlens.hooks import ActivationRecorder
    from jlens.fitting import valid_position_mask

    d_model = model.d_model
    dim_batch = 64
    input_ids = torch.tensor([ids[:SEQ_LEN]], dtype=torch.long, device=device)
    seq_len = input_ids.shape[1]
    mask = valid_position_mask(seq_len, skip_first=SKIP_FIRST)
    valid = mask.nonzero(as_tuple=True)[0]
    keep = torch.tensor([p for p in probe if p < seq_len - 1], dtype=torch.long)
    out = {l: torch.zeros(len(keep), d_model, d_model, dtype=torch.float32) for l in band}
    # target_layer=-2, NOT n_layers-1. trainval.py:310 fits with target_layer=-2, so the stored
    # operators transport into layer n_layers-2. Measuring G against n_layers-1 would compare each
    # operator against a transport it was never estimating, and every alignment number would be
    # apples to oranges. The tiny-first smoke caught this; it is why that rule exists.
    target_layer = model.n_layers - 2
    n_passes = math.ceil(d_model / dim_batch)

    with ActivationRecorder(model.layers, at=[*band, target_layer],
                            start_graph_at=min(band)) as rec, torch.enable_grad():
        model.forward(input_ids.expand(dim_batch, -1))
        tgt = rec.activations[target_layer]
        srcs = [rec.activations[l] for l in band]
        vpos = valid.to(tgt.device)
        bidx = torch.arange(dim_batch, device=tgt.device)
        cot = torch.zeros_like(tgt)
        for pi, d0 in enumerate(range(0, d_model, dim_batch)):
            n = min(dim_batch, d_model - d0)
            cot.zero_()
            cot[bidx[:n, None], vpos[None, :], d0 + bidx[:n, None]] = 1.0
            grads = torch.autograd.grad(tgt, srcs, grad_outputs=cot,
                                        retain_graph=(pi < n_passes - 1))
            for l, g in zip(band, grads):
                out[l][:, d0:d0 + n, :] = g[:n, keep.to(g.device), :].float().permute(1, 0, 2).cpu()
            del grads
    return out, [int(p) for p in keep]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-seq", type=int, default=128, help="read sequences averaged into G(p)")
    ap.add_argument("--prefix-corpus", default="Pile-CC")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cv8_positional_extrapolation.json"))
    a = ap.parse_args()
    if a.smoke:
        a.n_seq = 4

    torch.backends.cuda.matmul.allow_tf32 = False       # TF32 is forbidden (CLAUDE.md §7)
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from t2_fastfit import PYTHIA_LAYOUT_T5
    import provenance as prov

    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(a.device)
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    print(f"  model d_model={model.d_model} n_layers={model.n_layers} band={BAND[0]}..{BAND[-1]}",
          flush=True)

    items = build_items(tok)
    print(f"  {len(items)} eval items", flush=True)

    # C3: the items that BUILD G and the items that are SCORED are disjoint halves.
    build_idx = list(range(0, len(items), 2))
    score_idx = list(range(1, len(items), 2))
    pool = load_prefix_pool(a.prefix_corpus, tok, len(items), seed=0)

    use = build_idx[:a.n_seq]
    print(f"  building G(p) from {len(use)} sequences, prefix={a.prefix_corpus}", flush=True)

    acc, kept = None, None
    for n, ii in enumerate(use):
        ids, _ = prefixed_sequence(items[ii], pool[ii], tok.bos_token_id)
        if len(ids) < max(PROBE) + 2:
            ids = ids + pool[ii][: max(PROBE) + 2 - len(ids)]      # pad to reach probe 177
        J, kept = position_jacobians(model, ids, BAND, PROBE, a.device)
        if acc is None:
            acc = {l: torch.zeros_like(J[l]) for l in BAND}
        for l in BAND:
            acc[l] += J[l]
        del J
        if n % 8 == 0 or n == len(use) - 1:
            el = time.time() - t_start
            print(f"    {n+1}/{len(use)}  {el/60:.1f} min", flush=True)
    G = {l: acc[l] / len(use) for l in BAND}
    pidx = {p: i for i, p in enumerate(kept)}
    print(f"  G(p) built for positions {kept}", flush=True)

    # ---- load the eight operators ------------------------------------------------------------
    ops = {}
    for name, stem in PANEL.items():
        p = os.path.join(ROOT, "results", "e48", f"lens_{stem}_410m_n200_s0.pt")
        if not os.path.exists(p):
            raise SystemExit(f"ABORT: missing operator {p}")
        d = torch.load(p, map_location="cpu", weights_only=True)
        ops[name] = {l: d["J"][l].float() for l in BAND}

    # ---- alignment ---------------------------------------------------------------------------
    align = {c: {str(p): float(sum(cos_f(ops[c][l], G[l][pidx[p]]) for l in BAND) / len(BAND))
                 for p in kept} for c in PANEL}

    # ---- controls ----------------------------------------------------------------------------
    controls = {}

    c1_worst = 0.0
    for alpha in (0.5, 2.0, 100.0):
        for c in PANEL:
            base = cos_f(ops[c][BAND[0]], G[BAND[0]][pidx[P_READ]])
            scaled = cos_f(ops[c][BAND[0]], G[BAND[0]][pidx[P_READ]] * alpha)
            c1_worst = max(c1_worst, abs(base - scaled))
    # AMENDED TOLERANCE, 2026-08-29, disclosed in the registration. The literal "== 0.0" as first
    # registered is unachievable: cos_F sums 1,048,576 products in float64, and the smoke measured
    # 2.18e-11. The control's INTENT -- that the metric cannot see the target-count artifact -- is
    # unchanged, and 1e-9 is nine orders below the smallest between-corpus alignment difference.
    controls["C1_scale_invariance"] = {
        "required": "cos_F unchanged to < 1e-9 when G is multiplied by alpha in {0.5, 2, 100}; the "
                    "control for the target-count artifact that motivated a scale-free metric. "
                    "AMENDED from '== 0.0', which is unreachable in floating point.",
        "max_abs_difference": c1_worst, "tolerance": 1e-9, "fires": c1_worst < 1e-9}

    g = torch.Generator().manual_seed(1234)
    rand = {l: torch.randn(G[l].shape[1], G[l].shape[2], generator=g) for l in BAND}
    a_sup = {c: align[c][str(64)] for c in PANEL}
    a_rand = {c: float(sum(cos_f(ops[c][l], rand[l]) for l in BAND) / len(BAND)) for c in PANEL}
    c2_ok = min(a_sup.values()) >= 0.30 and max(abs(v) for v in a_rand.values()) <= 0.05
    controls["C2_metric_is_not_vacuous"] = {
        "required": "in-support alignment >= 0.30 for all 8 corpora AND |alignment| vs a Gaussian "
                    "matrix <= 0.05. IF THIS FAILS THE INSTRUMENT IS BROKEN AND NO VERDICT ISSUES.",
        "min_in_support_p64": min(a_sup.values()), "max_abs_vs_random": max(abs(v) for v in a_rand.values()),
        "per_corpus_in_support": a_sup, "per_corpus_vs_random": a_rand, "fires": bool(c2_ok)}

    overlap = len(set(build_idx[:a.n_seq]) & set(score_idx))
    controls["C3_no_leakage"] = {
        "required": "the items building G and the items scored are disjoint",
        "n_overlap": overlap, "n_build": len(use), "n_score": len(score_idx),
        "fires": overlap == 0}

    cv2 = json.load(open(os.path.join(ROOT, "results", "cv2_position_support.json")))
    pre = cv2["overall"]["prefixed"]
    c4_ok = (pre["min"] <= P_READ <= pre["max"]) and pre["n_in_support"] == 0
    controls["C4_probe_matches_the_real_read"] = {
        "required": "p_read inside CV2's measured prefixed read range, and CV2's n_in_support == 0",
        "p_read": P_READ, "cv2_min": pre["min"], "cv2_max": pre["max"],
        "cv2_median": pre["median_read_pos"], "cv2_n_in_support": pre["n_in_support"],
        "fires": bool(c4_ok)}

    # The rule tests/test_band_rule.py asserts is floor(0.38L)..floor(0.92L) INTERSECTED WITH
    # layers strictly below the penultimate target (cv6_per_family_ladder.band_for). Omitting the
    # intersection gives [9,22] and disagrees with every band on disk; the registration quoted the
    # rule without its second clause, and the AMENDMENT records that.
    L = model.n_layers
    rule_band = [l for l in range(math.floor(0.38 * L), math.floor(0.92 * L) + 1) if l < L - 2]
    c5_ok = BAND == rule_band
    controls["C5_band_rule"] = {
        "required": "band == floor(0.38L)..floor(0.92L) intersected with layers strictly below the "
                    "penultimate target layer, at L=24",
        "band": BAND, "rule_band": rule_band, "target_layer": L - 2, "fires": bool(c5_ok)}

    # ---- PRIMARY -----------------------------------------------------------------------------
    r9 = json.load(open(os.path.join(ROOT, "results", "r9_permutation_calibrated_min.json")))
    reads = {c: r9["by_aggregation"]["min"]["per_corpus"][c]["real"] for c in PANEL}
    corpora = sorted(PANEL)
    x = [align[c][str(P_READ)] for c in corpora]
    y = [reads[c] for c in corpora]
    rho, p_two, crit, n_perm = exact_permutation(x, y)

    sd = lambda v: (sum((t - sum(v) / len(v)) ** 2 for t in v) / (len(v) - 1)) ** 0.5
    spread_read = sd([align[c][str(P_READ)] for c in corpora])
    spread_sup = sd([align[c][str(64)] for c in corpora])

    if not controls["C2_metric_is_not_vacuous"]["fires"]:
        verdict = ("VOID — C2 did not fire: the alignment metric is vacuous on this run, so no "
                   "verdict is issued. See the registration.")
    elif p_two <= 0.05:
        verdict = (f"EXTRAPOLATION EXPLAINS THE ORDERING — exact two-sided p = {p_two:.4f} <= 0.05 "
                   f"(rho = {rho:+.3f}). The corpus ordering at position {P_READ} tracks how well "
                   f"each operator aligns with the transport required there. The unqualified corpus "
                   f"claim must be retired and reported as a decomposition.")
    elif p_two >= 0.20:
        verdict = (f"EXTRAPOLATION DOES NOT EXPLAIN THE ORDERING — exact two-sided p = {p_two:.4f} "
                   f">= 0.20, rho = {rho:+.3f}, below the alpha=0.05 critical |rho| of {crit:.3f} "
                   f"for n=8. No STRONG monotone relationship between alignment at position "
                   f"{P_READ} and the read. The positional hedge is retired; this design could not "
                   f"have detected a moderate relationship and the claim is bounded accordingly.")
    else:
        verdict = (f"UNCLEAR — exact two-sided p = {p_two:.4f} falls in the registered 0.05..0.20 "
                   f"band (rho = {rho:+.3f}). Report and stop. Do not re-cut.")

    rec = {
        "experiment": "CV8 — positional extrapolation: is the corpus effect an extrapolation artifact?",
        "prereg": "docs/experiments/preregs/CV8_positional_extrapolation.md",
        "status": "PRE-REGISTERED",
        "model": MODEL_ID, "band": BAND, "device": a.device,
        "smoke": bool(a.smoke),
        "n_sequences_for_G": len(use), "prefix_corpus": a.prefix_corpus,
        "probe_positions": kept, "p_read": P_READ,
        "in_support_positions": IN_SUPPORT, "out_of_support_positions": OUT_SUPPORT,
        "alignment_by_corpus_by_position": align,
        "read_scores_min_from_r9": reads,
        "PRIMARY": {
            "statistic": "Spearman rho between alignment at p_read and the stored min read score",
            "rho": rho, "exact_two_sided_p": p_two, "n_permutations": n_perm,
            "critical_abs_rho_at_alpha_0.05": crit,
            "minimum_detectable": (f"n=8 detects only |rho| >= {crit:.3f} at alpha=0.05 two-sided; "
                                   f"a moderate relationship would NOT be detected by this design"),
            "corpora_in_order": corpora,
            "alignment_at_p_read": x, "read_score": y},
        "SECONDARY_alignment_spread": {
            "between_corpus_sd_at_p_read": spread_read,
            "between_corpus_sd_in_support_p64": spread_sup,
            "ratio": (spread_read / spread_sup) if spread_sup else None,
            "note": "descriptive; no threshold is attached, per the registration"},
        "controls": controls,
        "controls_fired": {k: v.get("fires") for k, v in controls.items()},
        "VERDICT": verdict,
    }
    print("\n" + verdict, flush=True)
    for k, v in controls.items():
        print(f"  control {k}: fires={v.get('fires')}", flush=True)

    prov.write_result(a.out, rec, script=__file__, experiment="CV8",
                      inputs=[os.path.join(ROOT, "results", "r9_permutation_calibrated_min.json"),
                              os.path.join(ROOT, "results", "cv2_position_support.json")])
    print("wrote", os.path.relpath(a.out, ROOT), f"({(time.time()-t_start)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
