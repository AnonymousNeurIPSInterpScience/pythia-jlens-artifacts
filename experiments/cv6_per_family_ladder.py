#!/usr/bin/env python3
"""
CV6 — the per-family corpus contrast at 1.4B and 2.8B.

Pre-registration: docs/experiments/preregs/CV6_per_family_ladder.md, committed BEFORE this file.
DO NOT reinterpret the decision rule. It is transcribed verbatim into DECISION_RULE below and
applied by --adjudicate without a branch that can soften it.

WHAT THIS ANSWERS
  Does the per-family corpus effect replicate at 1.4B and 2.8B, and does the per-family ordering
  hold? The pooled estimand is retired (D3: multilingual is 49.2% of scored pairs at 0.9%
  competence), so there is NO pooled primary here and none may be computed after the fact.

THREE MODES, one file so the rule cannot drift between them:

  --model {1.4b,2.8b}     PHASE A + B for one model. Fits 15 operators (5 in-stream corpora x
                          3 disjoint seed blocks, N=25), caching h_t ON THIS DEVICE at this
                          dtype first, then scoring every operator against that one cache.
                          Writes results/cv6/cv6_<short>_n<N>.json after EVERY arm (crash-safe).

  --c0-reproduce-d3       CONTROL C0, free, CPU. Scores the 15 STORED 410M operators in
                          results/e48/ with the scorer below and compares against the stored
                          results/d3_corpus_by_family_410m.json. This proves the vectorised
                          scorer here is the same instrument as D3's per-pair loop. It must be
                          run on CPU: D3 ran on CPU and D2 measured a CUDA-vs-CPU cell-level
                          divergence of 2.774e-04, so a GPU rerun would not be a null test.

  --adjudicate            Combines the per-model files with D3 and applies the DECISION RULE.
                          Writes results/cv6_per_family_ladder.json.

THE READOUT IS THE CORRECTED ONE. Prompts are rstrip()ed before readout_position, per
docs/experiments/READOUT_DEFECT.md. The e48 `tv_*.json` reads are EXPOSED (unstripped) and are
NOT used here for anything.

POOLING CONVENTION, named because trap 4 says naming it is mandatory: within a family, pairs are
FLAT-POOLED (every (item, intermediate) pair counts once), k-summary is the FLAT MEAN over
K=(1,2,5,10,20,50,100), z is the max over the band per pair then the mean over pairs, and the
per-family seed SD is the RMS over the 5 corpora of each corpus's 3-seed sample SD. This is D3's
convention exactly, which is what makes the 410M comparison a comparison.

  # local, free, both are gates on the paid run
  .venv/bin/python experiments/cv6_per_family_ladder.py --c1-only --model 410m --device cpu
  .venv/bin/python experiments/cv6_per_family_ladder.py --c0-reproduce-d3

  # on the box
  python experiments/cv6_per_family_ladder.py --model 1.4b --device cuda
  python experiments/cv6_per_family_ladder.py --model 2.8b --device cuda

  # local, free
  .venv/bin/python experiments/cv6_per_family_ladder.py --adjudicate
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, time
import statistics as st

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

# The env var above is a DRIVER-level override and does not move torch's own flags. R18 forbids
# TF32; set the asserted state as the actual state rather than trusting the driver.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in ("src", "jacobian-lens", "experiments"):
    sys.path.insert(0, os.path.join(ROOT, _p))

from anchor_evals import load_eval, token_ids_of, readout_position   # noqa: E402

MODELS = {"410m": "EleutherAI/pythia-410m-deduped",
          "1.4b": "EleutherAI/pythia-1.4b-deduped",
          "2.8b": "EleutherAI/pythia-2.8b-deduped"}
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
INSTREAM = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github"]
SEEDS = [0, 1, 2]
KS = (1, 2, 5, 10, 20, 50, 100)
N_DEFAULT = 25
DIM_BATCH = 128
MAX_SEQ_LEN = 128
SKIP_FIRST = 16
TARGET_LAYER = -2
BAND_LO, BAND_HI = 0.38, 0.92     # the standing rule, normalised L38-L92

# e48_crossover_410m_rstrip.json : arms_admitted_mean.logit_I.min
C1_TARGET = 0.19810852520167826

DECISION_RULE = (
    "Evaluated PER FAMILY; the verdict is the count across the five. "
    "REPLICATES if R(f,2.8B) >= 10 on >= 4 of 5 families AND ordering tau >= 0.6 on >= 3 of 5. "
    "ATTENUATES if R(f,2.8B) < 5 on >= 4 of 5 families. "
    "UNCLEAR otherwise, including a split where some families replicate and others do not — "
    "a split is an informative outcome, not a failure. "
    "Do not pool to force a verdict. Do not add models to chase a threshold. Do not drop a family."
)


# ------------------------------------------------------------------ the standing band rule
def band_for(n_layers: int) -> list[int]:
    """Normalised L38-L92 intersected with layers STRICTLY below the penultimate target.

    target_layer=-2 resolves to n_layers-2 and jlens requires every source layer strictly below
    it. This is trainval.py's rule with band_frac=(0.38,0.92); it is NOT tuned and C2 asserts the
    emitted band equals what this returns.
    """
    target_eff = n_layers - 2
    return [l for l in range(int(BAND_LO * n_layers), int(BAND_HI * n_layers) + 1)
            if l < target_eff]


def ksummary(curve) -> float:
    """FLAT MEAN over the 7 k values — the programme's convention (docs/context/CONFIG_MATRIX.md).
    NOT a trapezoid AUC over log k: the two differ by 4.3% on identical ranks and CV3's C1 caught
    exactly that substitution."""
    return sum(curve) / len(curve)


def kendall_tau(order_a: list[str], order_b: list[str]) -> float | None:
    """tau-a over a strict ordering of the same 5 corpora. No ties are possible in a
    best-to-worst list, so tau-a == tau-b here."""
    ra = {c: i for i, c in enumerate(order_a)}
    rb = {c: i for i, c in enumerate(order_b)}
    items = list(order_a)
    conc = disc = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            x, y = items[i], items[j]
            s = (ra[x] - ra[y]) * (rb[x] - rb[y])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    n = conc + disc
    return (conc - disc) / n if n else None


# ------------------------------------------------------------------ phase B: the h_t cache
def cache_activations(model, tok, band, device, strip=True):
    """One forward pass per eval item, at that item's own readout position, no prefix.

    Byte-for-byte D3's caching loop, except the stack lands on `device`. The prompt is rstrip()ed
    BEFORE readout_position — that strip is the readout correction and it moves 157 of 551 items.
    `strip=False` is the LEGACY (defective) readout and exists only to make C0 falsifiable.
    """
    from jlens.hooks import ActivationRecorder
    pairs, acts, rows = [], {l: [] for l in band}, []
    with torch.no_grad():
        for name in ADMITTED:
            for it in load_eval(name):
                pr = it["prompt"] if isinstance(it["prompt"], str) else \
                    " ".join(m.get("content", "") for m in it["prompt"])
                pr = pr.rstrip() if strip else pr
                pos = readout_position(tok, name, pr)
                ids = model.encode(pr, max_length=256)
                with ActivationRecorder(model.layers, at=band) as rec:
                    model.forward(ids)
                    for l in band:
                        acts[l].append(rec.activations[l][0][pos].detach().float())
                r = len(rows)
                rows.append((name, r))
                for w in it["intermediates"]:
                    sy = token_ids_of(tok, w)
                    if sy:
                        pairs.append((name, r, sy))
    A = {l: torch.stack(acts[l]).to(device) for l in band}
    return A, pairs, len(rows)


def pair_tensors(pairs, device):
    """Pad the synonym-id lists into one [P, S] gather target plus a mask, so a whole layer scores
    in three tensor ops instead of P python iterations. Equivalence to D3's per-pair loop is not
    argued, it is CONTROLLED: --c0-reproduce-d3."""
    P = len(pairs)
    S = max(len(sy) for _, _, sy in pairs)
    ids = torch.zeros(P, S, dtype=torch.long)
    mask = torch.zeros(P, S, dtype=torch.bool)
    rows = torch.zeros(P, dtype=torch.long)
    for i, (_, r, sy) in enumerate(pairs):
        rows[i] = r
        ids[i, :len(sy)] = torch.tensor(sy, dtype=torch.long)
        mask[i, :len(sy)] = True
    return rows.to(device), ids.to(device), mask.to(device)


def score_arm(model, A, band, rows, syn_ids, syn_mask, pairs, Jmap):
    """rank and z for every (pair, layer). Jmap=None is the identity arm (the logit lens)."""
    P = rows.shape[0]
    rank = torch.empty(P, len(band), dtype=torch.int32)
    zsc = torch.empty(P, len(band))
    with torch.no_grad():
        for li, l in enumerate(band):
            h = A[l] if Jmap is None else A[l] @ Jmap[l].T
            lg = model.unembed(h).float()                        # [n_items, V]
            mu = lg.mean(1)
            sd = lg.std(1)
            L = lg.index_select(0, rows)                          # [P, V]
            got = L.gather(1, syn_ids)
            got = got.masked_fill(~syn_mask, float("-inf"))
            best = got.max(1).values                              # [P]
            rank[:, li] = ((L > best.unsqueeze(1)).sum(1) + 1).to(torch.int32).cpu()
            zsc[:, li] = ((best - mu.index_select(0, rows)) /
                          sd.index_select(0, rows)).float().cpu()
            del lg, L, got
    minrank = rank.min(1).values
    maxz = zsc.max(1).values
    per = {}
    for f in ADMITTED:
        idx = [i for i, (name, _, _) in enumerate(pairs) if name == f]
        r = [int(minrank[i]) for i in idx]
        z = [float(maxz[i]) for i in idx]
        per[f] = {"rank": ksummary([sum(1 for x in r if x <= k) / len(r) for k in KS]),
                  "z": st.mean(z), "n_pairs": len(idx)}
    return per


# ------------------------------------------------------------------ phase A: the fit
def load_fit_prompts(corpus, seed, tok, n):
    """The disjoint seed block, full-window filtered, first n. trainval.py's selection exactly, so
    CV6's N=25 is the PREFIX of the N=200 pool the 410M operators used."""
    path = os.path.join(ROOT, "corpora", f"{corpus}.jsonl")
    texts = [json.loads(l)["text"] for l in open(path)]
    b = len(texts) // 3
    pool = texts[seed * b:(seed + 1) * b]
    pool = [t for t in pool if len(tok(t).input_ids) >= MAX_SEQ_LEN][:n]
    if len(pool) < n:
        raise SystemExit(f"ABORT: {corpus} seed {seed} supplies {len(pool)}/{n} full-window "
                         f"prompts — C4 (N identical across corpora) cannot hold")
    return pool


def fit_operator(model, prompts, band, device, dim_batch):
    from jlens.fitting import jacobian_for_prompt
    Jsum = {l: torch.zeros(model.d_model, model.d_model, dtype=torch.float32, device=device)
            for l in band}
    n = 0
    t0 = time.time()
    for i, p in enumerate(prompts):
        pj, _, _ = jacobian_for_prompt(model, p, band, target_layer=TARGET_LAYER,
                                       dim_batch=dim_batch, max_seq_len=MAX_SEQ_LEN,
                                       skip_first=SKIP_FIRST)
        for l in band:
            Jsum[l] += pj[l].float().to(device)
        n += 1
        del pj
    if n != len(prompts):
        raise SystemExit(f"ABORT: {n}/{len(prompts)} prompts contributed; C4 requires all of them")
    return {l: Jsum[l] / n for l in band}, n, round(time.time() - t0, 1)


# ------------------------------------------------------------------ memory / time probe
def probe(a):
    """ONE prompt at the real band and the real dim_batch, before committing to 15 operators.

    WHY THIS EXISTS. `ARTIFACTS.md` records the N=200 full-band 2.8B fit peaking at 76.2 GB of
    81.5 on an H100. The band does NOT bound that: the graph root is min(source_layers), but every
    parameter still requires grad, so the forward builds and retains a graph from the embedding
    upward regardless of where the band starts. A 44 GB card therefore cannot hold 2.8B at
    dim_batch=128 and no amount of band narrowing changes it. This measures the peak rather than
    trusting the arithmetic, and it costs one prompt.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from jlens.hf import from_hf
    from t2_fastfit import PYTHIA_LAYOUT_T5

    mid = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(mid)
    hf = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(a.device).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    band = band_for(model.n_layers)
    prompts = load_fit_prompts(INSTREAM[0], 0, tok, 1)
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    fit_operator(model, prompts, band, a.device, a.dim_batch)
    dt = time.time() - t
    peak_a = peak_r = None
    if a.device.startswith("cuda"):
        peak_a = torch.cuda.max_memory_allocated() / 2**30
        peak_r = torch.cuda.max_memory_reserved() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"  peak allocated {peak_a:.1f} GB   reserved {peak_r:.1f} GB   of {total:.1f} GB")
    print(f"  {a.model} band={band[0]}..{band[-1]} db={a.dim_batch}: {dt:.1f} s/prompt")
    print(f"  projection: {dt*a.n/60:.1f} min/operator, {dt*a.n*15/3600:.2f} h for 15 operators")
    return 0


# ------------------------------------------------------------------ per-model run
def run_model(a):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from jlens.hf import from_hf
    from t2_fastfit import PYTHIA_LAYOUT_T5

    mid = MODELS[a.model]
    dev = a.device
    tok = AutoTokenizer.from_pretrained(mid)
    hf = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(dev).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    n_layers = model.n_layers
    band = band_for(n_layers)
    print(f"{a.model}  n_layers={n_layers}  d_model={model.d_model}  band={band} "
          f"({len(band)} layers)  device={dev}", flush=True)

    print("caching h_t (one forward per eval item, corrected readout) ...", flush=True)
    t_cache = time.time()
    A, pairs, n_items = cache_activations(model, tok, band, dev)
    rows, syn_ids, syn_mask = pair_tensors(pairs, dev)
    npf = {f: sum(1 for p in pairs if p[0] == f) for f in ADMITTED}
    print(f"  {n_items} items, {len(pairs)} pairs {npf}  ({time.time()-t_cache:.0f}s)", flush=True)

    if a.cache_out:
        cp = a.cache_out
        os.makedirs(os.path.dirname(cp) or ".", exist_ok=True)
        torch.save({"A": {l: A[l].cpu() for l in band}, "band": band, "model": mid,
                    "pairs": [(f, r, sy) for f, r, sy in pairs], "n_items": n_items,
                    "device_fitted_on": dev, "dtype": "float32",
                    "readout": "STRIPPED (corrected), no prefix, max_length=256"}, cp)
        print(f"  h_t cache -> {os.path.relpath(cp, ROOT)}", flush=True)

    out = {
        "experiment": "CV6 — the per-family corpus contrast, one model",
        "prereg": "docs/experiments/preregs/CV6_per_family_ladder.md",
        "status": "PRE-REGISTERED",
        "decision_rule_verbatim": DECISION_RULE,
        "model": mid, "short": a.model, "n_layers": n_layers, "d_model": model.d_model,
        "band": band, "band_rule": f"floor({BAND_LO}*L)..floor({BAND_HI}*L), layers < L-2",
        "band_matches_rule": band == band_for(n_layers),
        "N": a.n, "dim_batch": a.dim_batch, "max_seq_len": MAX_SEQ_LEN,
        "skip_first": SKIP_FIRST, "target_layer_arg": TARGET_LAYER,
        "target_layer_effective": n_layers - 2,
        "device": dev, "dtype": "float32",
        "tf32": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                 "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE")},
        "gpu": (torch.cuda.get_device_name(0) if dev.startswith("cuda") else None),
        "gpu_total_gb": (torch.cuda.get_device_properties(0).total_memory / 2**30
                         if dev.startswith("cuda") else None),
        "peak_vram_gb": None,          # filled after the last arm
        "torch": torch.__version__,
        # The box has no .git (lab push excludes it), so provenance.git_state() records commit
        # None. The script SHA still pins the code exactly; this carries the tree it came from.
        "git_commit_at_launch": os.environ.get("CV6_GIT_COMMIT"),
        "K": list(KS), "admitted_sets": ADMITTED, "corpora": INSTREAM, "seeds": SEEDS,
        "readout_convention": "STRIPPED (corrected), no prefix, flat-mean-7",
        "pooling_convention": ("flat pool over (item, intermediate) pairs WITHIN a family; "
                               "rank = min over band then flat-mean-7 pass@k; z = max over band "
                               "then mean over pairs"),
        "n_items": n_items, "n_pairs": len(pairs), "n_pairs_per_family": npf,
        "per_arm": {}, "per_arm_fp16_roundtrip": {}, "fits": {},
    }
    dest = a.out or os.path.join(ROOT, "results", "cv6", f"cv6_{a.model}_n{a.n}.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def flush(final=False):
        if final:
            try:
                from provenance import write_result
                write_result(dest, out, script=__file__, experiment="CV6",
                             inputs=[os.path.join(ROOT, "corpora", f"{c}.jsonl") for c in INSTREAM])
                return
            except Exception as e:                     # printed, never swallowed
                print(f"  !! provenance stamp FAILED: {e!r}", file=sys.stderr)
        json.dump(out, open(dest, "w"), indent=1, default=str)

    # ---- C1 first: the identity arm is emitted BEFORE any J arm is graded
    t = time.time()
    out["per_arm"]["logit_I"] = score_arm(model, A, band, rows, syn_ids, syn_mask, pairs, None)
    pooled = st.mean(out["per_arm"]["logit_I"][f]["rank"] for f in ADMITTED)
    out["logit_I_pooled_rank_flatmean7"] = pooled
    print(f"  logit_I  pooled rank {pooled:.11f}  ({time.time()-t:.0f}s)  " +
          " ".join(f"{f[:5]}={out['per_arm']['logit_I'][f]['z']:.3f}" for f in ADMITTED), flush=True)
    if a.model == "410m":
        d = abs(pooled - C1_TARGET)
        print(f"  C1 vs stored 410M constant: |diff| = {d:.3e}  "
              f"{'FIRES' if d <= 1e-6 else 'FAILED'}", flush=True)
        out["C1_410m_check"] = {"target": C1_TARGET, "observed": pooled, "abs_diff": d,
                                "fires": d <= 1e-6}
    flush()
    if a.c1_only:
        print("--c1-only: stopping after the identity arm", flush=True)
        flush(final=True)
        return 0

    # ---- the 15 J arms, fitted then scored against the SAME cache
    for c in INSTREAM:
        for s in SEEDS:
            arm = f"J|{c}|s{s}"
            if arm in out["per_arm"]:
                continue
            prompts = load_fit_prompts(c, s, tok, a.n)
            sha = hashlib.sha256("\n".join(prompts).encode()).hexdigest()
            t = time.time()
            J, n_used, fit_s = fit_operator(model, prompts, band, dev, a.dim_batch)
            out["fits"][arm] = {"n_prompts": n_used, "fit_seconds": fit_s,
                                "prompts_sha256": sha, "corpus": c, "seed_block": s,
                                "fro_norm_per_layer": {str(l): float(J[l].norm()) for l in band}}
            out["per_arm"][arm] = score_arm(model, A, band, rows, syn_ids, syn_mask, pairs, J)
            # DIAGNOSTIC, not a decision input: the 410M operators in results/e48/ are STORED at
            # fp16, so D3 scored an fp16-rounded operator. This measures what that rounding is
            # worth here, so the ladder comparison does not have to assume it is immaterial.
            Jh = {l: J[l].half().float() for l in band}
            out["per_arm_fp16_roundtrip"][arm] = score_arm(model, A, band, rows, syn_ids,
                                                           syn_mask, pairs, Jh)
            if a.save_lens_dir:
                os.makedirs(a.save_lens_dir, exist_ok=True)
                lp = os.path.join(a.save_lens_dir,
                                  f"lens_INSTREAM_{c}_{a.model}_n{a.n}_s{s}.pt")
                torch.save({"J": {l: J[l].half().cpu() for l in band}, "n_prompts": n_used,
                            "source_layers": band, "d_model": model.d_model, "model": mid,
                            "corpus": c, "seed": s, "fitted_device": dev, "fitted_dtype": "float32",
                            "prompts_sha256": sha}, lp)
            del J, Jh
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
            zz = " ".join(f"{f[:5]}={out['per_arm'][arm][f]['z']:.3f}" for f in ADMITTED)
            print(f"  {arm:26} fit {fit_s:6.0f}s  total {time.time()-t:6.0f}s  {zz}", flush=True)
            flush()

    if dev.startswith("cuda"):
        out["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
    flush(final=True)
    print(f"\nwrote {os.path.relpath(dest, ROOT)}")
    return 0


# ------------------------------------------------------------------ C0: scorer equivalence
def c0_reproduce_d3(a):
    """Score the STORED 410M operators with this file's scorer and diff against D3.

    A control that cannot fail is not a control: this one fails if the vectorised gather scores a
    different token, pools differently, or summarises k differently from D3's per-pair loop.
    """
    if a.device != "cpu":
        raise SystemExit("C0 must run on CPU — D3 ran on CPU and D2 measured a CUDA-vs-CPU "
                         "divergence of 2.774e-04, so a GPU rerun is not a null test")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from jlens.hf import from_hf
    from t2_fastfit import PYTHIA_LAYOUT_T5

    d3p = os.path.join(ROOT, "results", "d3_corpus_by_family_410m.json")
    d3 = json.load(open(d3p))
    mid = MODELS["410m"]
    tok = AutoTokenizer.from_pretrained(mid)
    hf = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    band = band_for(model.n_layers)
    if band != d3["band"]:
        raise SystemExit(f"ABORT: band rule gives {band}, D3 stored {d3['band']}")
    print(f"C0: band={band} matches D3", flush=True)

    A, pairs, n_items = cache_activations(model, tok, band, "cpu")
    rows, syn_ids, syn_mask = pair_tensors(pairs, "cpu")
    npf = {f: sum(1 for p in pairs if p[0] == f) for f in ADMITTED}
    if npf != d3["n_pairs_per_family"]:
        raise SystemExit(f"ABORT: pairs {npf} != D3 {d3['n_pairs_per_family']}")
    print(f"  {n_items} items, {len(pairs)} pairs — matches D3", flush=True)

    arms = {"logit_I": None}
    for c in INSTREAM:
        for s in SEEDS:
            arms[f"{c}|s{s}"] = os.path.join(ROOT, "results", "e48",
                                             f"lens_INSTREAM_{c}_410m_n200_s{s}.pt")
    worst_z = worst_r = 0.0
    rowsout = {}
    for arm, path in arms.items():
        Jm = None if path is None else {l: torch.load(path, map_location="cpu")["J"][l].float()
                                        for l in band}
        per = score_arm(model, A, band, rows, syn_ids, syn_mask, pairs, Jm)
        ref = d3["per_arm"][arm]
        dz = max(abs(per[f]["z"] - ref[f]["z"]) for f in ADMITTED)
        dr = max(abs(per[f]["rank"] - ref[f]["rank"]) for f in ADMITTED)
        worst_z, worst_r = max(worst_z, dz), max(worst_r, dr)
        rowsout[arm] = {"max_abs_dz": dz, "max_abs_drank": dr}
        print(f"  {arm:22} max|dz|={dz:.3e}  max|drank|={dr:.3e}", flush=True)

    # NEGATIVE CONTROL. A control that cannot fail is not a control. Re-cache at the LEGACY
    # (unstripped) readout — the defect that moved 157 of 551 items — and score the identity arm
    # against the same D3 reference. If C0 still reads zero here, C0 is not sensitive to the one
    # thing it is supposed to pin, and the zero above means nothing.
    An, pn, _ = cache_activations(model, tok, band, "cpu", strip=False)
    rn, sin, smn = pair_tensors(pn, "cpu")
    neg = score_arm(model, An, band, rn, sin, smn, pn, None)
    ndz = max(abs(neg[f]["z"] - d3["per_arm"]["logit_I"][f]["z"]) for f in ADMITTED)
    ndr = max(abs(neg[f]["rank"] - d3["per_arm"]["logit_I"][f]["rank"]) for f in ADMITTED)
    neg_fires = ndz > 1e-3
    print(f"  NEGATIVE (legacy unstripped readout, logit_I): max|dz|={ndz:.3e} "
          f"max|drank|={ndr:.3e}  {'separates' if neg_fires else 'DOES NOT SEPARATE'}", flush=True)
    del An

    fires = worst_z <= 1e-6 and worst_r <= 1e-9 and neg_fires
    out = {"experiment": "CV6 control C0 — the scorer reproduces D3 on the stored 410M operators",
           "status": "CONTROL", "model": mid, "band": band, "device": "cpu",
           "n_items": n_items, "n_pairs": len(pairs), "n_pairs_per_family": npf,
           "compared_against": "results/d3_corpus_by_family_410m.json",
           "per_arm_deviation": rowsout,
           "worst_abs_dz": worst_z, "worst_abs_drank": worst_r,
           "required": ("max|dz| <= 1e-6 and max|drank| <= 1e-9 over all 16 arms x 5 families, "
                        "AND the negative control separates"),
           "negative_control": {
               "what": "same scorer, LEGACY unstripped readout, identity arm, same D3 reference",
               "why": "proves C0 can fail; a zero that a wrong readout also produces is not a test",
               "max_abs_dz": ndz, "max_abs_drank": ndr,
               "required": "max|dz| > 1e-3", "separates": neg_fires,
               "per_family_z_legacy": {f: neg[f]["z"] for f in ADMITTED}},
           "fires": fires,
           "what_it_does_not_cover": ("nothing about the FIT path — it scores stored operators. "
                                      "It also says nothing about GPU scoring; D2 measured a "
                                      "CUDA-vs-CPU divergence of 2.774e-04 at the cell level.")}
    dest = os.path.join(ROOT, "results", "cv6", "cv6_c0_scorer_equivalence.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        from provenance import write_result
        write_result(dest, out, script=__file__, experiment="CV6-C0", inputs=[d3p])
    except Exception as e:
        print(f"  !! provenance stamp FAILED: {e!r}", file=sys.stderr)
        json.dump(out, open(dest, "w"), indent=1, default=str)
    print(f"\nC0 {'FIRES' if fires else 'FAILED'}  worst|dz|={worst_z:.3e} "
          f"worst|drank|={worst_r:.3e}  negative max|dz|={ndz:.3e}")
    return 0 if fires else 3


# ------------------------------------------------------------------ adjudication
def per_family_table(per_arm):
    """R(f) = spread_z(f) / pooled_seed_sd_z(f), computed WITHIN the family."""
    fam = {}
    for f in ADMITTED:
        rowsc = {}
        for c in INSTREAM:
            v = [per_arm[f"J|{c}|s{s}"][f] for s in SEEDS if f"J|{c}|s{s}" in per_arm]
            zs = [x["z"] for x in v]
            rs = [x["rank"] for x in v]
            rowsc[c] = {"z_mean": st.mean(zs),
                        "z_sd": st.stdev(zs) if len(zs) > 2 else 0.0,
                        "rank_mean": st.mean(rs),
                        "rank_sd": st.stdev(rs) if len(rs) > 2 else 0.0,
                        "n_seeds": len(zs)}
        def sp(k, sk):
            m = [rowsc[c][k] for c in INSTREAM]
            pooled = math.sqrt(sum(rowsc[c][sk] ** 2 for c in INSTREAM) / len(INSTREAM))
            return max(m) - min(m), pooled, ((max(m) - min(m)) / pooled if pooled else None)
        zs_, zp, zr = sp("z_mean", "z_sd")
        rs_, rp, rr = sp("rank_mean", "rank_sd")
        fam[f] = {"per_corpus": rowsc,
                  "z_spread": zs_, "z_pooled_seed_sd": zp, "R": zr,
                  "rank_spread": rs_, "rank_pooled_seed_sd": rp, "rank_spread_over_sd": rr,
                  "order_by_z": sorted(INSTREAM, key=lambda c: -rowsc[c]["z_mean"]),
                  "order_by_rank": sorted(INSTREAM, key=lambda c: -rowsc[c]["rank_mean"]),
                  "n_pairs": per_arm[f"J|{INSTREAM[0]}|s0"][f]["n_pairs"]}
    return fam


def adjudicate(a):
    d3p = os.path.join(ROOT, "results", "d3_corpus_by_family_410m.json")
    d3 = json.load(open(d3p))
    models, inputs = {}, [d3p]
    for short in ("1.4b", "2.8b"):
        p = os.path.join(ROOT, "results", "cv6", f"cv6_{short}_n{a.n}.json")
        if not os.path.exists(p):
            raise SystemExit(f"ABORT: {os.path.relpath(p, ROOT)} missing — run --model {short} first")
        models[short] = json.load(open(p))
        inputs.append(p)

    c0p = os.path.join(ROOT, "results", "cv6", "cv6_c0_scorer_equivalence.json")
    c0 = json.load(open(c0p)) if os.path.exists(c0p) else None
    if c0:
        inputs.append(c0p)

    out = {"experiment": "CV6 — does the per-family corpus effect replicate at 1.4B and 2.8B?",
           "prereg": "docs/experiments/preregs/CV6_per_family_ladder.md",
           "status": "PRE-REGISTERED",
           "decision_rule_verbatim": DECISION_RULE,
           "admitted_sets": ADMITTED, "corpora": INSTREAM, "seeds": SEEDS, "N": a.n,
           "K": list(KS),
           "readout_convention": "STRIPPED (corrected), no prefix, flat-mean-7",
           "pooling_convention": ("flat pool over (item, intermediate) pairs WITHIN a family; "
                                  "seed SD is the RMS over the 5 corpora of each corpus's 3-seed "
                                  "sample SD — trap 4: the convention is named because it moves "
                                  "'x seed SD' ratios ~37%"),
           "reference_410m": {"source": "results/d3_corpus_by_family_410m.json (EXPLORATORY)",
                              "N": 200,
                              "order_by_z": {f: d3["by_family"][f]["order_by_z"] for f in ADMITTED},
                              "R": {f: d3["by_family"][f]["z_spread_over_sd"] for f in ADMITTED},
                              "z_pooled_seed_sd": {f: d3["by_family"][f]["z_pooled_seed_sd"]
                                                   for f in ADMITTED}},
           "by_model": {}, "controls": {}}

    for short, rec in models.items():
        fam = per_family_table(rec["per_arm"])
        fam16 = per_family_table(rec["per_arm_fp16_roundtrip"]) if rec.get("per_arm_fp16_roundtrip") else {}
        for f in ADMITTED:
            fam[f]["kendall_tau_vs_410m"] = kendall_tau(fam[f]["order_by_z"],
                                                        d3["by_family"][f]["order_by_z"])
            if fam16:
                fam[f]["fp16_roundtrip_diagnostic"] = {
                    "R": fam16[f]["R"],
                    "max_abs_dz_mean": max(abs(fam16[f]["per_corpus"][c]["z_mean"] -
                                               fam[f]["per_corpus"][c]["z_mean"])
                                           for c in INSTREAM),
                    "order_unchanged": fam16[f]["order_by_z"] == fam[f]["order_by_z"]}
        out["by_model"][short] = {
            "model": rec["model"], "band": rec["band"], "n_layers": rec["n_layers"],
            "device": rec["device"], "dtype": rec["dtype"], "gpu": rec.get("gpu"),
            "N": rec["N"], "n_pairs_per_family": rec["n_pairs_per_family"],
            "logit_I": rec["per_arm"]["logit_I"],
            "logit_I_pooled_rank_flatmean7": rec.get("logit_I_pooled_rank_flatmean7"),
            "by_family": fam}

    # ---- controls
    c1 = {"required": ("logit_I emitted for every model and family; at 410M the pooled admitted-5 "
                       f"flat-mean-7 min-rank equals {C1_TARGET}"),
          "emitted_per_model": {s: sorted(models[s]["per_arm"]["logit_I"].keys())
                                for s in models},
          "410m_anchor": (c0 and "C0 scored the identity arm at 410M; see cv6_c0_scorer_equivalence.json"),
          "fires": all(set(models[s]["per_arm"]["logit_I"].keys()) == set(ADMITTED) for s in models)}
    c2 = {"required": "emitted band == band_for(n_layers); a tuned band voids the run",
          "per_model": {s: {"band": models[s]["band"],
                            "rule": band_for(models[s]["n_layers"]),
                            "match": models[s]["band"] == band_for(models[s]["n_layers"])}
                        for s in models},
          "fires": all(models[s]["band"] == band_for(models[s]["n_layers"]) for s in models)}
    c3 = {"required": "pooled_seed_sd_z(f, m) > 0 for every (family, model); else that cell is VOID",
          "per_model": {s: {f: out["by_model"][s]["by_family"][f]["z_pooled_seed_sd"]
                            for f in ADMITTED} for s in models},
          "void_cells": [f"{s}/{f}" for s in models for f in ADMITTED
                         if not out["by_model"][s]["by_family"][f]["z_pooled_seed_sd"] > 0],
          "fires": all(out["by_model"][s]["by_family"][f]["z_pooled_seed_sd"] > 0
                       for s in models for f in ADMITTED)}
    nseen = {s: sorted({models[s]["fits"][k]["n_prompts"] for k in models[s]["fits"]})
             for s in models}
    c4 = {"required": f"N == {a.n} and identical across all 15 arms of every model",
          "n_prompts_seen": nseen,
          "n_arms": {s: len(models[s]["fits"]) for s in models},
          "prompt_sha_identical_across_models": {
              f"{c}|s{s_}": (models["1.4b"]["fits"][f"J|{c}|s{s_}"]["prompts_sha256"] ==
                             models["2.8b"]["fits"][f"J|{c}|s{s_}"]["prompts_sha256"])
              for c in INSTREAM for s_ in SEEDS},
          "fires": all(nseen[s] == [a.n] for s in models) and
                   all(len(models[s]["fits"]) == 15 for s in models)}
    c5 = {"required": "activations and operators on the same device at the same dtype",
          "per_model": {s: {"device": models[s]["device"], "dtype": models[s]["dtype"],
                            "gpu": models[s].get("gpu"), "tf32": models[s].get("tf32")}
                        for s in models},
          "fires": all(models[s]["dtype"] == "float32" for s in models)}
    c0c = {"required": "the scorer reproduces D3 on the stored 410M operators",
           "result": (c0 and {"worst_abs_dz": c0["worst_abs_dz"],
                              "worst_abs_drank": c0["worst_abs_drank"], "fires": c0["fires"]}),
           "fires": bool(c0 and c0["fires"])}
    out["controls"] = {"C0_scorer_equivalence": c0c, "C1_logit_identity_arm": c1,
                       "C2_band_matches_rule": c2, "C3_seed_sd_nondegenerate": c3,
                       "C4_N_identical": c4, "C5_device_precision": c5}

    # ---- THE DECISION RULE. Transcribed, not interpreted.
    fam28 = out["by_model"]["2.8b"]["by_family"]
    R28 = {f: fam28[f]["R"] for f in ADMITTED}
    TAU28 = {f: fam28[f]["kendall_tau_vs_410m"] for f in ADMITTED}
    n_ge10 = sum(1 for f in ADMITTED if R28[f] is not None and R28[f] >= 10)
    n_lt5 = sum(1 for f in ADMITTED if R28[f] is not None and R28[f] < 5)
    n_tau = sum(1 for f in ADMITTED if TAU28[f] is not None and TAU28[f] >= 0.6)
    if n_ge10 >= 4 and n_tau >= 3:
        verdict = "REPLICATES"
    elif n_lt5 >= 4:
        verdict = "ATTENUATES"
    else:
        verdict = "UNCLEAR"
    out["PRIMARY"] = {"R_per_family_2.8b": R28, "kendall_tau_vs_410m_2.8b": TAU28,
                      "n_families_R_ge_10": n_ge10, "n_families_R_lt_5": n_lt5,
                      "n_families_tau_ge_0.6": n_tau}
    out["VERDICT"] = verdict
    out["verdict_note"] = {
        "REPLICATES": "the corpus effect is not an artifact of the low-competence regime at 410M",
        "ATTENUATES": "the effect is a small-model phenomenon; corpus sensitivity is itself "
                      "capability-gated",
        "UNCLEAR": "report the per-family table and STOP. Do not pool, do not add models, do not "
                   "drop a family.",
    }[verdict]
    out["declared_bias"] = [
        "Capability and scale are confounded; every larger model is also differently trained. No "
        "outcome licenses a causal reading — only 'the effect does / does not survive up the ladder'.",
        "The 5 corpora are a fixed panel, not a sample. n=5 is the replication unit for any claim "
        "about KINDS of corpus.",
        f"N={a.n} here vs N=200 at 410M. S2 established reads are flat in N, so this is expected to "
        "be immaterial, but it is a difference and an attenuation must not be attributed to scale "
        "without checking N.",
        "Per-family seed SDs are thin: 3 seeds, 96-394 pairs per family. Thin denominators are what "
        "inflated the retired 58x figure. The SD is reported beside every ratio.",
        "The 410M reference is D3, which is EXPLORATORY and was computed BEFORE this "
        "pre-registration. The per-family estimand may not be described as chosen on capability "
        "grounds alone.",
    ]

    dest = os.path.join(ROOT, "results", "cv6_per_family_ladder.json")
    try:
        from provenance import write_result
        write_result(dest, out, script=__file__, experiment="CV6", inputs=inputs)
    except Exception as e:
        print(f"  !! provenance stamp FAILED: {e!r}", file=sys.stderr)
        json.dump(out, open(dest, "w"), indent=1, default=str)

    print("\n" + "=" * 96)
    for short in ("1.4b", "2.8b"):
        b = out["by_model"][short]
        print(f"\n{short}  band={b['band'][0]}..{b['band'][-1]}  N={b['N']}  {b.get('gpu') or b['device']}")
        print(f"  {'family':13} {'n':>4} {'z spread':>9} {'z seed SD':>10} {'R':>8} "
              f"{'tau':>6}  best->worst (z)")
        for f in ADMITTED:
            x = b["by_family"][f]
            print(f"  {f:13} {x['n_pairs']:4} {x['z_spread']:9.4f} {x['z_pooled_seed_sd']:10.5f} "
                  f"{(x['R'] if x['R'] is not None else float('nan')):8.1f} "
                  f"{(x['kendall_tau_vs_410m'] if x['kendall_tau_vs_410m'] is not None else float('nan')):6.2f}"
                  f"  " + " > ".join(y[:11] for y in x["order_by_z"]))
    print(f"\ncontrols: " + ", ".join(f"{k.split('_')[0]}={v['fires']}"
                                      for k, v in out["controls"].items()))
    print(f"R>=10 on {n_ge10}/5   R<5 on {n_lt5}/5   tau>=0.6 on {n_tau}/5")
    print(f"VERDICT = {verdict}  —  {out['verdict_note']}")
    print("wrote", os.path.relpath(dest, ROOT))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("-n", "--n", type=int, default=N_DEFAULT)
    ap.add_argument("--dim-batch", type=int, default=DIM_BATCH)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache-out", default=None,
                    help="persist the h_t cache as a .pt so a re-score needs no forward passes")
    ap.add_argument("--save-lens-dir", default=None)
    ap.add_argument("--c1-only", action="store_true",
                    help="emit the identity arm and stop — the free gate on the paid run")
    ap.add_argument("--c0-reproduce-d3", action="store_true")
    ap.add_argument("--adjudicate", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="one prompt: peak VRAM and s/prompt, before committing to 15 operators")
    a = ap.parse_args()

    if a.probe:
        if not a.model:
            ap.error("--probe needs --model")
        return probe(a)
    if a.c0_reproduce_d3:
        return c0_reproduce_d3(a)
    if a.adjudicate:
        return adjudicate(a)
    if not a.model:
        ap.error("one of --model, --c0-reproduce-d3, --adjudicate is required")
    return run_model(a)


if __name__ == "__main__":
    raise SystemExit(main())
