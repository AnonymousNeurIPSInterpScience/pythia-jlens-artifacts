#!/usr/bin/env python3
"""r6_within_source.py — R6: is "corpus" a factor, or a bundle?

PRE-REGISTRATION: specs/experiments/R6_within_source_resampling.md, committed before this ran.
Source slate: specs/POSTREVIEW_EXPERIMENTS.md §3 Tier B, item R6.

THE QUESTION. The paper's fit axis is labelled "corpus". A corpus bundles formatting, boundaries,
token-frequency profile, syntax and surprisal, and the paper controls prompt length and nothing
else (its own Limitation 3). If the fit-axis effect survives when the SOURCE is held fixed and only
the SAMPLE varies, the effect is not a bundle artifact. If it collapses, the factor is misnamed.

WHY IT MATTERS MORE AFTER R1. R1 cut the fit axis's SHARE from 91.2% to 50.7%, but in absolute
terms the fit axis barely moved (SS x0.85) -- what grew was the read axis (x10.27). The fit effect
is still the larger of the two, and still needs showing to be a SOURCE effect and not a SAMPLING one.

READOUT: STRIPPED, per R1. These numbers are not comparable to pre-R1 figures.

    python r6_within_source.py --device cuda --out ../results/r6_within_source_410m.json
"""
from __future__ import annotations
import argparse, itertools, json, os, statistics as st, sys, time

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

# TF32 IS FORBIDDEN (CLAUDE.md §7): a 10-bit mantissa breaks the anchor gate's tolerance, silently,
# and corrupts every number a run produces. NVIDIA_TF32_OVERRIDE is a DRIVER override and does NOT
# change torch's flags -- that exact mismatch failed a preflight before, which is why trainval.py
# sets both. The box this was written for shipped with TF32 ON and its preflight failed on it.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert not torch.backends.cuda.matmul.allow_tf32, "TF32 matmul is ON — refusing to fit"
assert not torch.backends.cudnn.allow_tf32, "TF32 cudnn is ON — refusing to fit"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))
from provenance import sha256_file, write_result  # noqa: E402

MODEL = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))
K = [1, 2, 5, 10, 20, 50, 100]
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
SOURCES = ["Pile-CC", "Wikipedia_en"]
N_BLOCKS = 4
N_FIT = 200
WINDOW = 128


def derangement(band, seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        perm = [band[i] for i in torch.randperm(len(band), generator=g).tolist()]
        if all(p != l for p, l in zip(perm, band)):
            return perm


def build_items(tok, load_eval, readout_position, token_ids_of, EVAL_SETS, rstrip=True):
    """The eval battery at the ANCHOR's readout token (R1). BOS is asserted, not assumed."""
    items = []
    for name in EVAL_SETS:
        for it in load_eval(name):
            tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
            if not tgt:
                continue
            p = it["prompt"].rstrip() if rstrip else it["prompt"]
            items.append({"set": name, "ids": tok(p, add_special_tokens=True).input_ids,
                          "pos": readout_position(tok, name, p), "tgt": tgt})
    bad = [i for i, it in enumerate(items) if not it["ids"] or it["ids"][0] != tok.bos_token_id]
    if bad:
        raise SystemExit(f"ABORT: {len(bad)}/{len(items)} items carry no BOS — jlens/hf.py:111's "
                         f"tokenizer mutation did not reach the item table.")
    return items


def make_scorer(model, items, device):
    """One activation cache, then transport -> unembed -> rank -> pass@k. Same path as t52/t36."""
    from jlens.hooks import ActivationRecorder
    L, d = len(BAND), model.d_model
    pair_set, pair_tgt, pair_item = [], [], []
    for ii, it in enumerate(items):
        for t in it["tgt"]:
            pair_set.append(it["set"]); pair_tgt.append(t); pair_item.append(ii)
    # keyed on ADMITTED explicitly, not on whatever sets happen to appear: a truncated item list
    # would otherwise silently average over fewer than five sets and the admitted-mean would be a
    # different estimand. Empty sets are reported, never skipped silently.
    SET_IDX = {s: torch.tensor([i for i, p in enumerate(pair_set) if p == s], dtype=torch.long)
               for s in ADMITTED}
    _empty = [s for s in ADMITTED if len(SET_IDX[s]) == 0]
    ITEM_PAIRS = {}
    for pi, ii in enumerate(pair_item):
        ITEM_PAIRS.setdefault(ii, []).append(pi)
    P_n, HALF = len(pair_tgt), L // 2

    A = torch.empty(len(items), L, d)
    with torch.no_grad():
        for ii, it in enumerate(items):
            t = torch.tensor([it["ids"]], device=device)
            with ActivationRecorder(model.layers, at=BAND) as rec:
                model.forward(t)
                A[ii] = torch.stack([rec.activations[l][0][it["pos"]].detach().float() for l in BAND])

    def score(T):
        H = A if T is None else torch.stack([A[:, j, :] @ T[BAND[j]].T.cpu() for j in range(L)], dim=1)
        flat = H.reshape(-1, d)
        R = torch.empty(P_n, L, dtype=torch.float32)
        with torch.no_grad():
            for i0 in range(0, len(items), 24):
                i1 = min(i0 + 24, len(items))
                lg = model.unembed(flat[i0 * L:i1 * L].to(device)).float().cpu()
                for ii in range(i0, i1):
                    blk = lg[(ii - i0) * L:(ii - i0 + 1) * L]
                    for pi in ITEM_PAIRS[ii]:
                        cand = torch.stack([(blk > blk[:, i:i + 1]).sum(1) + 1 for i in pair_tgt[pi]])
                        R[pi] = cand.min(0).values.float()
        mn = R.min(dim=1).values
        out = {"min": torch.stack([(mn <= k).float() for k in K]).mean(0),
               "persist": torch.stack([((R <= k).float().sum(1) >= HALF).float() for k in K]).mean(0)}
        live = [s for s in ADMITTED if len(SET_IDX[s]) > 0]
        return {ag: st.mean([float(v[SET_IDX[s]].mean()) for s in live]) for ag, v in out.items()}
    return score


def disjoint_blocks(corpus, tok, n_blocks, n_fit, window):
    """n_blocks DISJOINT pools of n_fit full-window documents, in file order. C1 verifies it."""
    path = os.path.join(HERE, "..", "corpora", f"{corpus}.jsonl")
    texts = [json.loads(l)["text"] for l in open(path)]
    qual = [t for t in texts if len(tok(t).input_ids) >= window]
    need = n_blocks * n_fit
    if len(qual) < need:
        raise SystemExit(f"ABORT: {corpus} supplies {len(qual)} full-window docs, need {need}. "
                         f"Reusing documents would destroy the disjointness this experiment is about.")
    return [qual[i * n_fit:(i + 1) * n_fit] for i in range(n_blocks)]


def two_way(vals, groups):
    """between-source vs within-source sums of squares over the operator-level reads."""
    g = st.mean(vals)
    ss_tot = sum((v - g) ** 2 for v in vals)
    gm = {}
    for v, s in zip(vals, groups):
        gm.setdefault(s, []).append(v)
    ss_between = sum(len(v) * (st.mean(v) - g) ** 2 for v in gm.values())
    ss_within = sum((v - st.mean(gm[s])) ** 2 for v, s in zip(vals, groups))
    return {"SS_between_source": ss_between, "SS_within_source": ss_within, "SS_total": ss_tot,
            "between_source_share": ss_between / (ss_between + ss_within) if ss_tot else None,
            "group_means": {k: st.mean(v) for k, v in gm.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dim-batch", type=int, default=128)
    ap.add_argument("--lens-dir", default=os.path.join(HERE, "..", "results", "r6"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "r6_within_source_410m.json"))
    ap.add_argument("--smoke", action="store_true", help="2 blocks; proves the path only")
    ap.add_argument("--n-fit", type=int, default=None, help="override prompts per block")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from fastfit import fast_fit
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of

    n_blocks = 2 if a.smoke else N_BLOCKS
    n_fit = a.n_fit if a.n_fit else (8 if a.smoke else N_FIT)
    os.makedirs(a.lens_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    print(f"model on {a.device}; band {BAND[0]}..{BAND[-1]}", flush=True)

    items = build_items(tok, load_eval, readout_position, token_ids_of, EVAL_SETS, rstrip=True)
    if a.smoke:
        items = items[:40]
    print(f"{len(items)} eval items (STRIPPED readout)", flush=True)
    _seen = {s: sum(1 for it in items if it["set"] == s) for s in ADMITTED}
    if not a.smoke and any(v == 0 for v in _seen.values()):
        raise SystemExit(f"ABORT: an admitted set has no items: {_seen}. The admitted-mean would "
                         f"be a different estimand than every other result in the repo.")
    print(f"  items per admitted set: {_seen}", flush=True)

    # ---------------------------------------------------------------- pools + C1 disjointness
    pools, overlaps = {}, {}
    for src in SOURCES:
        blocks = disjoint_blocks(src, tok, n_blocks, n_fit, WINDOW)
        pools[src] = blocks
        for i, j in itertools.combinations(range(n_blocks), 2):
            overlaps[f"{src}|b{i}xb{j}"] = len(set(blocks[i]) & set(blocks[j]))
    c1 = {"required": "0 shared documents between any pair of within-source blocks",
          "pairwise_shared_document_counts": overlaps,
          "max_shared": max(overlaps.values()) if overlaps else 0,
          "fires": all(v == 0 for v in overlaps.values()),
          "power": ("e58 measured 24-26% overlap when the exclusion step is omitted, so a "
                    "disjointness check on this corpus family demonstrably can fail")}
    print(f"C1 disjointness: max shared docs = {c1['max_shared']} -> {'FIRES' if c1['fires'] else 'FAILS'}",
          flush=True)

    # ---------------------------------------------------------------- fit
    ops, fit_meta = {}, {}
    t0 = time.time()
    for src in SOURCES:
        for b, prompts in enumerate(pools[src]):
            key = f"{src}|b{b}"
            path = os.path.join(a.lens_dir, f"lens_R6_{src}_b{b}_410m_n{n_fit}.pt")
            if os.path.exists(path):
                J = torch.load(path, map_location="cpu", weights_only=True)["J"]
                print(f"  [cached] {key}", flush=True)
            else:
                ts = time.time()
                lens = fast_fit(model, prompts, source_layers=BAND, target_layer=-2,
                                dim_batch=a.dim_batch, max_seq_len=WINDOW, skip_first=16,
                                device=a.device)
                # the library's own writer, so these lenses load exactly like every other .pt in
                # results/ (fp16 storage, key "J") rather than a bespoke format
                lens.save(path)
                J = torch.load(path, map_location="cpu", weights_only=True)["J"]
                print(f"  fitted {key} in {time.time()-ts:.0f}s -> {os.path.basename(path)}", flush=True)
            ops[key] = {l: J[l].float() for l in BAND}
            fit_meta[key] = {"path": os.path.relpath(path, os.path.join(HERE, "..")),
                             "n_prompts": len(prompts), "source": src, "block": b}
    print(f"  {len(ops)} operators in {(time.time()-t0)/60:.1f} min", flush=True)

    # ---------------------------------------------------------------- read
    score = make_scorer(model, items, a.device)
    reads = {k: score(J) for k, J in ops.items()}
    reads["__logit__"] = score(None)
    perm = derangement(BAND, 7000)
    floors = {k: score({l: ops[k][q] for l, q in zip(BAND, perm)}) for k in ops}
    for k in sorted(ops):
        print(f"  {k:24s} persist={reads[k]['persist']:.5f}  own-derangement={floors[k]['persist']:.5f}",
              flush=True)

    # ---------------------------------------------------------------- decomposition + controls
    keys = sorted(ops)
    res = {}
    for ag in ("persist", "min"):
        vals = [reads[k][ag] for k in keys]
        res[ag] = two_way(vals, [k.split("|")[0] for k in keys])
        res[ag]["per_operator"] = {k: reads[k][ag] for k in keys}

    g = torch.Generator().manual_seed(4242)
    null = []
    src_labels = [k.split("|")[0] for k in keys]
    vals_p = [reads[k]["persist"] for k in keys]
    for _ in range(2000):
        idx = torch.randperm(len(keys), generator=g).tolist()
        null.append(two_way(vals_p, [src_labels[i] for i in idx])["between_source_share"])
    null = [v for v in null if v is not None]
    obs = res["persist"]["between_source_share"]
    p95 = sorted(null)[int(0.95 * (len(null) - 1))]
    c2 = {"required": ("the observed between-source share must exceed the shuffled-label null's "
                       "95th percentile; chance level for 8 operators in 2 groups is ~1/7 = 0.143"),
          "null_mean": st.mean(null), "null_p95": p95, "observed": obs,
          "exceeds_null_p95": obs > p95, "n_perm": len(null),
          "fires": obs > p95}
    c3 = {"required": "every operator clears its OWN layer-deranged floor under persist",
          "per_operator": {k: {"real": reads[k]["persist"], "own_derangement": floors[k]["persist"],
                               "clears": reads[k]["persist"] > floors[k]["persist"]} for k in keys},
          "n_clearing": sum(1 for k in keys if reads[k]["persist"] > floors[k]["persist"]),
          "n_operators": len(keys)}
    c3["fires"] = c3["n_clearing"] == c3["n_operators"]

    if obs is None:
        verdict = "UNCLEAR — the decomposition is degenerate."
    elif obs >= 0.70:
        verdict = (f"ACCEPT — corpus identity is a real factor. The between-source share is "
                   f"{obs:.4f} >= 0.70: holding the source fixed and varying only the sample "
                   f"leaves most of the fit-axis variance between sources, so the effect is not a "
                   f"sampling artifact.")
    elif obs <= 0.30:
        verdict = (f"REJECT — the effect is sample noise, not source identity. The between-source "
                   f"share is {obs:.4f} <= 0.30. STOP AND ALERT THE OPERATOR: the paper's factor is "
                   f"not what it is called and must be renamed.")
    else:
        verdict = (f"UNCLEAR — the between-source share is {obs:.4f}, between 0.30 and 0.70. Report "
                   f"the number and stop; do not re-cut it (CLAUDE.md §2.9).")

    prereg = "specs/experiments/R6_within_source_resampling.md"
    rec = {"experiment": "R6 — within-source resampling: is 'corpus' a factor or a bundle?",
           "prereg": prereg,
           "prereg_sha256": sha256_file(os.path.join(HERE, "..", prereg)),
           "status": "PRE-REGISTERED",
           "decision_rule_verbatim": (
               "ACCEPT (corpus identity is a real factor): between-source share >= 0.70. | REJECT "
               "(the effect is sample noise, not source identity): between-source share <= 0.30. "
               "STOP and alert -- the paper's factor is not what it is called. | UNCLEAR: anything "
               "between. Report the number and stop; do not re-cut it."),
           "readout_convention": ("STRIPPED — the anchor rule (R1). NOT comparable to pre-R1 "
                                  "figures and must not be printed beside them."),
           "tf32": {"allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                    "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                    "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
                    "note": ("asserted OFF in-process. The rented box's own preflight reported "
                             "TF32 ENABLED, so the driver-level override alone was not sufficient "
                             "and the torch flags are set explicitly here.")},
           "model": MODEL, "band": BAND, "K": K, "admitted_sets": ADMITTED,
           "sources": SOURCES, "n_blocks_per_source": n_blocks, "n_fit_per_block": n_fit,
           "window_tokens": WINDOW, "smoke": a.smoke, "device": a.device,
           "n_items": len(items), "operators": fit_meta,
           "reads": reads, "own_derangement_floors": floors,
           "PRIMARY_between_source_share_persist": obs,
           "by_aggregation": res,
           "controls": {"C1_document_disjointness": c1, "C2_shuffled_label_null": c2,
                        "C3_derangement_floor": c3},
           "declared_bias": [
               "two sources, not eight: this tests whether SOURCE IDENTITY beats SAMPLING, not "
               "whether it beats FORMATTING, which is R7",
               "the pools are drawn under the same require_full_window filter, whose selectivity "
               "differs by corpus (SUSPICIONS.md S-3). Stated, not solved",
               "blocks are contiguous in file order, so any within-file ordering structure "
               "(e.g. crawl date) is confounded with block identity"],
           "VERDICT": verdict}
    write_result(a.out, rec, experiment="R6",
                 inputs=[os.path.join(HERE, "..", "corpora", f"{s}.jsonl") for s in SOURCES])
    for k, v in rec["controls"].items():
        print(f"  {k:34s} {v['fires']}")
    print(f"\nbetween-source share (persist) = {obs}")
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
