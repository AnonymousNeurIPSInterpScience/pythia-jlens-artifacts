#!/usr/bin/env python3
"""t52_factorial.py — E52: the fit x read factorial. The missing cell, J^Q read on Q.

THE QUESTION, in the operator's words
  "Can the J lens trained on an out of distribution sample read some latent properties of the
   model on an out of distribution corpus?"

WHY NEITHER E36 NOR E48 CAN ANSWER IT
  E48 moved the FIT distribution and held the read fixed -> exposure of the fitting corpus does
  not order the read. E36 moved the READ distribution and held the operator fixed -> REJECT S3;
  the fitted operator is FLATTER across the shift axis than the unfitted one, 5/5.
  Both are marginal. The estimator framing's last sharp prediction lives in the JOINT cell: if the
  corpus effect is a MATCHING effect, an operator fitted on Q should read Q better than one fitted
  on P does. Nothing measured so far separates that from "some corpora just make better operators".

PRE-REGISTRATION: specs/archive/prereg/PREREG_E52_FACTORIAL.md, written before this ran.
  PRIMARY   the DIAGONAL EXCESS. Over the 8x8 matrix y[f,q] (fit corpus f, read rung q), remove
            main effects:  y[f,q] = m + a_f + b_q + g[f,q].
            a_f = "some corpora make better operators" (the E28/E33/E48 effect)
            b_q = "some read contexts are easier"
            g[f,q] = the MATCHING term.  D = mean(g[f,f]) - mean(g[f,q!=f]).
  RULE      CI(D) strictly > 0  -> MATCHING CONFIRMED
            CI(D) includes 0    -> NO MATCHING (publishable null): the corpus effect is a property
                                   of the fitting corpus per se, and EQ2's ||J^P - J^Q|| gap
                                   framing is not what the data measures
            CI(D) strictly < 0  -> ANTI-MATCHING, reported separately
  `min` NEVER VOTES (CLAUDE.md 6.0; E48 measured the deranged operator beating the real one on
  103/120 draws under min). Reported, never decisive.

EVERY OPERATOR SHARES ONE FITTER. All 24 (8 corpora x 3 seed blocks) come from trainval.py at
N=200, --band 9,21, dim_batch=128, max_seq_len=128, skip_first=16, target_layer=-2, fp32, TF32
off. E48's in-stream arm used fastfit.fast_fit and its OOD arm used jacobian_for_prompt; a
factorial cannot tolerate a fitter difference confounded with the axis under test, which is what
the in-stream refits were produced for.

    python t52_factorial.py --device cpu
    python t52_factorial.py --smoke
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as _mp
import os
import statistics
import sys
import time

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))
from provenance import sha256_file, write_result  # noqa: E402

MODEL = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))
K = [1, 2, 5, 10, 20, 50, 100]
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
INSTREAM = ["Wikipedia_en", "USPTO_Backgrounds", "Pile-CC", "StackExchange", "Github"]
OOD = ["OOD_News_2024", "OOD_arXiv_2023", "OOD_CommonPile"]
CORPORA = INSTREAM + OOD                       # both the FIT axis and the READ axis
DEGENERATE = ["Github"]                        # established at/below its floor: E33, E48, E36
SEEDS = [0, 1, 2]
PREFIX_TOKENS = 128


def lens_path(c: str, s: int) -> str:
    """Matched-provenance lenses ONLY. In-stream ones are the INSTREAM refits, not the E28 fits."""
    stem = f"lens_{c}_410m_n200_s{s}.pt" if c.startswith("OOD_") else \
           f"lens_INSTREAM_{c}_410m_n200_s{s}.pt"
    return os.path.join(HERE, "..", "results", "e48", stem)


def derangement(band, seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        perm = [band[i] for i in torch.randperm(len(band), generator=g).tolist()]
        if all(p != l for p, l in zip(perm, band)):
            return perm


# ---------------------------------------------------------------------- P0: the parallel cell
# One (read rung, prefix seed) cell is one activation cache plus 26 independent score passes, and
# NOTHING is reduced across cells. The 30 cells are therefore embarrassingly parallel and this port
# is a scheduling change with no numerical content -- which is exactly what P0's C1/C2 gate at
# max_abs_diff = 0.0 rather than at a tolerance.
#
# DELIBERATE DEVIATION from POSTREVIEW_EXPERIMENTS.md §3/P0's design note, declared in
# specs/experiments/P0_t52_parallel_port.md. It says to keep the operators in the parent and let
# fork share them copy-on-write. That is unavailable here: this is darwin, where multiprocessing's
# start method is `spawn` (measured), and forcing `fork` with PyTorch already initialised is the
# documented deadlock. The port therefore follows e28_ladder_410m.py:175 -- the very script §3/P0
# points at -- whose cell() takes picklable arguments and loads the model itself, once per worker
# via a Pool initializer rather than once per cell. The cost is RSS proportional to --workers, not
# correctness.
#
# WHAT STAYS IN THE PARENT, so that determinism is structural rather than hoped for: every cell's
# prefix construction (build() runs in the parent, so the torch.Generator sequence is identical to
# the serial run's regardless of scheduling), the item and pair tables, and all aggregation,
# bootstrap, permutation and adjudication. The worker returns the same per-pair score vectors the
# serial loop put into CELL, so every downstream statistic reads an identical object.
_W: dict = {}


def _build_state(cfg: dict) -> dict:
    """Model + the 24 operators. Called once per worker process, or once in-process at --workers 1."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from t2_fastfit import PYTHIA_LAYOUT_T5

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(cfg["device"]).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    JF = {(c, s): {l: torch.load(lens_path(c, s), map_location="cpu",
                                 weights_only=True)["J"][l].float() for l in BAND}
          for c in cfg["fit_corpora"] for s in cfg["seeds"]}
    JSHUF = {l: JF[(cfg["fit_corpora"][0], cfg["seeds"][0])][q]
             for l, q in zip(BAND, derangement(BAND, 7000))}
    st = dict(cfg)
    st.update(model=model, hf=hf, tok=tok, JF=JF, JSHUF=JSHUF,
              d=model.d_model, L=len(BAND), HALF=len(BAND) // 2)
    return st


def _init_worker(cfg: dict) -> None:
    """C3. Pin each worker to one BLAS thread, and pin it AFTER the model is loaded.

    MEASURED, not assumed: setting it before `_build_state` does NOT hold. A first pooled run of the
    full grid reported `thr=8` inside every worker despite `torch.set_num_threads(1)` at the top of
    this function -- something in the model-loading path resets the intra-op pool. With 5 workers
    that is 40 threads on 14 cores. The control caught it because it asserts on OBSERVED state
    (`torch.get_num_threads()` reported per cell) rather than on "the call returned" -- CLAUDE.md
    §6.0b, which exists for exactly this failure shape.

    The pin is applied here and NOT re-applied per cell, deliberately: re-applying it in _run_cell
    would make C3 a control that cannot fail, which is the defect R4e enumerates 27 instances of.

    Thread count is immaterial to the NUMBERS here -- the scoring path was measured
    thread-count-invariant before the port (smoke matrix at OMP_NUM_THREADS=1 vs the default 10,
    max|diff| = 0.0 over 18 cells) -- which is why C1/C2 can be exact-zero gates rather than
    tolerances. It is material to the wallclock.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    _W["st"] = _build_state(cfg)
    torch.set_num_threads(1)


def _cache(st, cell):
    from jlens.hooks import ActivationRecorder
    model, hf = st["model"], st["hf"]
    A = torch.empty(st["n_items"], st["L"], st["d"])
    cap = []
    with torch.no_grad():
        for ii, (ids, pos) in enumerate(cell):
            t = torch.tensor([ids], device=st["device"])
            with ActivationRecorder(model.layers, at=BAND) as rec:
                out = model.forward(t)
                A[ii] = torch.stack([rec.activations[l][0][pos].detach().float() for l in BAND])
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            lg = hf.get_output_embeddings()(h[0, pos]).float()
            for tg in st["items_tgt"][ii]:
                cap.append(min(int((lg > lg[i]).sum().item()) + 1 for i in tg))
    return A, statistics.mean(cap)


def _score(st, A, T):
    model, L, d = st["model"], st["L"], st["d"]
    ITEM_PAIRS, pair_tgt, ipc = st["ITEM_PAIRS"], st["pair_tgt"], st["items_per_chunk"]
    H = A if T is None else torch.stack([A[:, j, :] @ T[BAND[j]].T for j in range(L)], dim=1)
    flat = H.reshape(-1, d)
    R = torch.empty(st["P_n"], L, dtype=torch.float32)
    with torch.no_grad():
        for i0 in range(0, st["n_items"], ipc):
            i1 = min(i0 + ipc, st["n_items"])
            lg = model.unembed(flat[i0 * L:i1 * L]).float()
            for ii in range(i0, i1):
                block = lg[(ii - i0) * L:(ii - i0 + 1) * L]
                for pi in ITEM_PAIRS[ii]:
                    cand = torch.stack([(block > block[:, i:i + 1]).sum(1) + 1
                                        for i in pair_tgt[pi]])
                    R[pi] = cand.min(0).values.float()
    mn = R.min(dim=1).values
    return {"min": torch.stack([(mn <= k).float() for k in K]).mean(0),
            "persist": torch.stack([((R <= k).float().sum(1) >= st["HALF"]).float()
                                    for k in K]).mean(0)}


def _run_cell(task):
    """One (rung, prefix seed) cell: cache once, then the 2 + 8x3 = 26 transport arms."""
    rung, ps, cellspec = task
    st = _W["st"]
    A, cap = _cache(st, cellspec)
    arms = {"logit": _score(st, A, None), "shuf": _score(st, A, st["JSHUF"])}
    for c in st["fit_corpora"]:
        for s in st["seeds"]:
            arms[f"{c}|s{s}"] = _score(st, A, st["JF"][(c, s)])
    return rung, ps, arms, cap, torch.get_num_threads()


def readout_divergence(tok, load_eval, readout_position, eval_sets):
    """R1/C2 -- the count of released items whose READOUT TOKEN differs between the two conventions.

    Required: 157 of 551, distributed 83/93 multihop, 19/107 multilingual, 55/55 order-ops, and 0 on
    poetry / typo / association. If it is not 157 the two arms are not the two arms.
    """
    per_set, examples = {}, {}
    for name in eval_sets:
        n_diff = n_tot = 0
        for it in load_eval(name):
            n_tot += 1
            raw, stripped = it["prompt"], it["prompt"].rstrip()
            a_ids = tok(raw, add_special_tokens=True).input_ids
            b_ids = tok(stripped, add_special_tokens=True).input_ids
            a_tok = a_ids[readout_position(tok, name, raw)]
            b_tok = b_ids[readout_position(tok, name, stripped)]
            if a_tok != b_tok:
                n_diff += 1
                examples.setdefault(name, {"unstripped_token_id": a_tok, "stripped_token_id": b_tok,
                                           "unstripped": repr(tok.decode([a_tok])),
                                           "stripped": repr(tok.decode([b_tok]))})
        per_set[name] = {"n_divergent": n_diff, "n_items": n_tot}
    return {"per_set": per_set,
            "total_divergent": sum(v["n_divergent"] for v in per_set.values()),
            "total_items": sum(v["n_items"] for v in per_set.values()),
            "first_divergent_example_per_set": examples}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1),
                    help="P0. Processes over the 30 (read rung, prefix seed) cells. 1 runs "
                         "in-process. Bounded by RAM, not cores: with spawn, each worker holds its "
                         "own model (~1.6 GB) and its own 24 operators (~1.3 GB).")
    ap.add_argument("--rstrip", action="store_true",
                    help="R1. Apply the anchor's readout rule: strip the prompt's trailing space "
                         "BEFORE tokenising, so the readout token is 'the token immediately "
                         "preceding target' as jacobian-lens/data/evaluations/README.md specifies. "
                         "This is a CORRECTION, not a sensitivity arm -- the unstripped path reads "
                         "token id 209 (a bare space), which does not occur in the tokenisation of "
                         "prompt+target at all. Changes 157 of 551 released items.")
    ap.add_argument("--prefix-seeds", type=int, default=3)
    ap.add_argument("--items-per-chunk", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--e48", default=os.path.join(HERE, "..", "results", "e48_crossover_410m.json"))
    ap.add_argument("--out", default=None,
                    help="default: results/e52_factorial_410m{_rstrip}.json, chosen by --rstrip so "
                         "the two conventions can never land in the same file")
    ap.add_argument("--cells-out", default=None,
                    help="E57: ALSO write the per-(read rung, prefix seed, fit corpus, fit seed) "
                         "admitted-set means, i.e. the 9 draws that y() averages into each matrix "
                         "cell. E52 computed these and serialised only their mean, so the variance "
                         "decomposition in the paper has no error term and none can be recovered "
                         "from the stored file. Purely ADDITIVE: with this flag unset the script "
                         "behaves exactly as it did when e52_factorial_410m.json was written.")
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(HERE, "..", "results",
                             f"e52_factorial_410m{'_rstrip' if a.rstrip else ''}.json")
    if a.smoke and os.path.abspath(a.out) in (
            os.path.abspath(os.path.join(HERE, "..", "results", "e52_factorial_410m.json")),
            os.path.abspath(os.path.join(HERE, "..", "results", "e52_factorial_410m_rstrip.json"))):
        raise SystemExit("ABORT: --smoke would overwrite a canonical results file. Pass --out.")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of
    from bootstrap import hierarchical_bootstrap_seeded

    # ---------------------------------------------------------------- HELD-OUT PREFIX POOLS
    # E36's load_pool() draws prefixes from the WHOLE corpus file. On the diagonal of this
    # factorial that is fatal: measured, 24-26% of the 541 prefix documents for a corpus are
    # LITERALLY among the 600 documents its operators were fitted on (200 x 3 seed blocks out of
    # 2400). The diagonal would then be elevated because the operator has already averaged a
    # derivative at those exact activations — a self-overlap artifact, not distributional matching,
    # and it points in the direction that would manufacture MATCHING CONFIRMED.
    #
    # The prefix pool for rung X is therefore X's documents MINUS every document used to fit any
    # operator on X. The exclusion depends only on the RUNG, so every operator reading that rung
    # sees identical prefixes and the fit axis stays comparable; only the diagonal's unfair
    # advantage is removed.
    _fit_docs_cache: dict[str, set] = {}

    def fit_documents(corpus: str) -> set:
        if corpus in _fit_docs_cache:
            return _fit_docs_cache[corpus]
        texts = [json.loads(l)["text"]
                 for l in open(os.path.join(HERE, "..", "corpora", f"{corpus}.jsonl"))]
        b = len(texts) // 3
        used = set()
        for s in SEEDS:                      # every seed block, not just the ones scored here
            pool = [t for t in texts[s * b:(s + 1) * b] if len(tok(t).input_ids) >= PREFIX_TOKENS]
            used.update(pool[:200])          # trainval.py takes the first 200 qualifying
        _fit_docs_cache[corpus] = used
        return used

    def load_pool_heldout(corpus: str, n: int, seed: int):
        texts = [json.loads(l)["text"]
                 for l in open(os.path.join(HERE, "..", "corpora", f"{corpus}.jsonl"))]
        banned = fit_documents(corpus)
        g = torch.Generator().manual_seed(9000 + seed)
        out = []
        for i in torch.randperm(len(texts), generator=g).tolist():
            t = texts[i]
            if t in banned:
                continue
            ids = tok(t).input_ids
            if len(ids) >= PREFIX_TOKENS:
                out.append(ids[:PREFIX_TOKENS])
            if len(out) >= n:
                break
        if len(out) < n:
            raise SystemExit(f"ABORT: {corpus} has only {len(out)} held-out prefix documents, "
                             f"need {n}. Cycling them would reintroduce the overlap this "
                             f"function exists to remove.")
        return out

    load_pool = load_pool_heldout

    tok = AutoTokenizer.from_pretrained(MODEL)
    # LOAD-BEARING, and it is a SIDE EFFECT of a call the parent no longer makes.
    # jlens/hf.py:104-111 -- HFModel.__init__ mutates the tokenizer in place, setting
    # `tokenizer.add_bos_token = True` under force_bos. Before P0 the parent built the model first,
    # so every eval item was tokenized by an ALREADY-MUTATED tokenizer and carried a BOS. With the
    # model moved into the workers, the parent's tokenizer is untouched and every item silently
    # loses its BOS -- 48/48 items changed, and the smoke matrix moved by 1.5e-2. Replicated here
    # explicitly, and ASSERTED on observed state below rather than trusted (CLAUDE.md §6.0b).
    if getattr(tok, "bos_token_id", None) is not None and hasattr(tok, "add_bos_token"):
        tok.add_bos_token = True
    L = len(BAND)

    fit_corpora = CORPORA[:3] if a.smoke else CORPORA
    seeds = [0] if a.smoke else SEEDS
    pseeds = [0] if a.smoke else list(range(a.prefix_seeds))
    rungs = (["Q0"] + fit_corpora + ([] if a.smoke else ["SHUFFLED_Pile-CC"]))

    missing = [lens_path(c, s) for c in fit_corpora for s in seeds
               if not os.path.exists(lens_path(c, s))]
    if missing:
        raise SystemExit(f"ABORT: missing matched-provenance lenses:\n  " + "\n  ".join(missing))
    print(f"{len(fit_corpora) * len(seeds)} matched operators | fit {fit_corpora} | "
          f"rungs {rungs} | prefix seeds {pseeds} | "
          f"readout={'STRIPPED (anchor rule)' if a.rstrip else 'unstripped (legacy)'} | "
          f"workers={a.workers}", flush=True)

    # ---- eval items; positions computed on the item ALONE and later offset (E36's rule)
    items, per_set_n = [], {}
    for name in EVAL_SETS:
        for it in load_eval(name):
            tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
            if not tgt:
                continue
            if a.smoke and per_set_n.get(name, 0) >= 8:
                continue
            per_set_n[name] = per_set_n.get(name, 0) + 1
            # R1. anchor_evals.py:32-34 and :228 strip; this script did not. The unstripped path
            # reads a token that does not occur in the tokenisation of prompt+target at all.
            p = it["prompt"].rstrip() if a.rstrip else it["prompt"]
            items.append({"set": name, "ids": tok(p, add_special_tokens=True).input_ids,
                          "pos": readout_position(tok, name, p), "tgt": tgt})

    # the assertion the comment above promises: BOS is present on every item, observed, not assumed
    _no_bos = [i for i, it in enumerate(items) if not it["ids"] or it["ids"][0] != tok.bos_token_id]
    if _no_bos:
        raise SystemExit(f"ABORT: {len(_no_bos)}/{len(items)} items carry no BOS. The tokenizer was "
                         f"not put in add_bos_token mode before the items were tokenized; every "
                         f"cell would be scored on a different sequence than the published run.")

    pair_set, pair_tgt, pair_item = [], [], []
    for ii, it in enumerate(items):
        for t in it["tgt"]:
            pair_set.append(it["set"]); pair_tgt.append(t); pair_item.append(ii)
    SET_IDX = {s: torch.tensor([i for i, p in enumerate(pair_set) if p == s], dtype=torch.long)
               for s in EVAL_SETS}
    ITEM_PAIRS = {}
    for pi, ii in enumerate(pair_item):
        ITEM_PAIRS.setdefault(ii, []).append(pi)
    P_n = len(pair_tgt)
    print(f"{len(items)} items, {P_n} pairs", flush=True)

    def build(rung, seed):
        if rung == "Q0":
            return [(it["ids"], it["pos"]) for it in items]
        if rung.startswith("SHUFFLED_"):
            base = load_pool(rung[len("SHUFFLED_"):], len(items), seed)
            g = torch.Generator().manual_seed(4242 + seed)
            pref = [[b[i] for i in torch.randperm(len(b), generator=g).tolist()] for b in base]
        else:
            pref = load_pool(rung, len(items), seed)
        out = []
        for it, pre in zip(items, pref):
            body = it["ids"][1:] if it["ids"] and it["ids"][0] == tok.bos_token_id else it["ids"]
            out.append(([tok.bos_token_id] + list(pre) + list(body),
                        it["pos"] if it["pos"] < 0 else it["pos"] + len(pre)))
        return out

    def adm(v):
        return statistics.mean([float(v[SET_IDX[s]].mean()) for s in ADMITTED])

    cfg = {"device": a.device, "fit_corpora": fit_corpora, "seeds": seeds,
           "items_tgt": [it["tgt"] for it in items], "ITEM_PAIRS": ITEM_PAIRS,
           "pair_tgt": pair_tgt, "n_items": len(items), "P_n": P_n,
           "items_per_chunk": a.items_per_chunk}
    # build() runs in the PARENT for every cell, so the torch.Generator sequence that draws the
    # prefix pools is identical to the serial run's no matter how the cells are then scheduled.
    tasks = ((rung, ps, build(rung, ps)) for rung in rungs for ps in pseeds)
    n_cells = len(rungs) * len(pseeds)

    CELL, cell_seconds, worker_threads = {}, {}, None
    t0 = time.time()

    def _absorb(rung, ps, arms, cap, nthr, done):
        nonlocal worker_threads
        worker_threads = nthr
        CELL[(rung, ps)] = {"arms": arms, "capability_mean_rank": cap}
        cell_seconds[f"{rung}|p{ps}"] = round(time.time() - t0, 1)
        best = max(adm(arms[f"{c}|s{seeds[0]}"]["persist"]) for c in fit_corpora)
        print(f"  [{done}/{n_cells}] read={rung:22s} p{ps}  "
              f"logit={adm(arms['logit']['persist']):.5f} bestJ={best:.5f} "
              f"shuf={adm(arms['shuf']['persist']):.5f} cap={cap:.0f} "
              f"thr={nthr}  [{time.time()-t0:.0f}s]", flush=True)

    if a.workers <= 1:
        _W["st"] = _build_state(cfg)
        for i, task in enumerate(tasks, 1):
            _absorb(*_run_cell(task), i)
    else:
        # spawn, not fork: see the module-level note. imap_unordered returns out of order, so CELL
        # is assembled BY KEY -- arrival order must not be able to reach a number.
        ctx = _mp.get_context("spawn")
        with ctx.Pool(a.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for i, out in enumerate(pool.imap_unordered(_run_cell, tasks), 1):
                _absorb(*out, i)
    wall = time.time() - t0
    print(f"  {n_cells} cells in {wall/60:.1f} min on {a.workers} worker(s) "
          f"({wall/n_cells:.0f} s/cell, {wall/(n_cells*(2+len(fit_corpora)*len(seeds))):.1f} "
          f"s/score-pass)", flush=True)

    # ------------------------------------------------------------------ E57: the discarded draws
    if a.cells_out:
        draws = {ag: {f"{f}|{q}|s{s}|p{ps}": adm(CELL[(q, ps)]["arms"][f"{f}|s{s}"][ag])
                      for f in fit_corpora for q in fit_corpora
                      for s in seeds for ps in pseeds}
                 for ag in ("persist", "min")}
        for ag in ("persist", "min"):
            for q in rungs:
                for ps in pseeds:
                    for arm in ("logit", "shuf"):
                        draws[ag][f"{arm}|{q}|p{ps}"] = adm(CELL[(q, ps)]["arms"][arm][ag])
        cells_rec = {
            "experiment": "E57 — the per-cell draws E52 averaged away",
            "why": ("the paper's headline is a variance decomposition of 64 cell MEANS with one "
                    "observation each, so its three shares sum to 100% by construction, its "
                    "'interaction' is a residual conflated with cell noise, and it carries no "
                    "interval. Each cell is a mean over 9 draws (3 fit seeds x 3 prefix seeds). "
                    "This emits those draws so the shares can be bootstrapped."),
            "unit": "admitted-set mean read AUC for one (fit corpus, fit seed, read rung, prefix seed)",
            "fit_corpora": fit_corpora, "read_rungs": rungs,
            "fit_seeds": seeds, "prefix_seeds": pseeds,
            "n_draws_per_matrix_cell": len(seeds) * len(pseeds),
            "admitted_sets": ADMITTED, "band": BAND, "K": K, "model": MODEL,
            "draws": draws,
        }

    # ------------------------------------------------------------------ the matrix
    def y(f, q, ag="persist"):
        """read AUC for operator fitted on f, read on rung q — averaged over fit and prefix seeds"""
        return statistics.mean([adm(CELL[(q, ps)]["arms"][f"{f}|s{s}"][ag])
                                for s in seeds for ps in pseeds])

    def diag_excess(mat, corpora):
        m = statistics.mean([mat[(f, q)] for f in corpora for q in corpora])
        a_f = {f: statistics.mean([mat[(f, q)] for q in corpora]) - m for f in corpora}
        b_q = {q: statistics.mean([mat[(f, q)] for f in corpora]) - m for q in corpora}
        g = {(f, q): mat[(f, q)] - m - a_f[f] - b_q[q] for f in corpora for q in corpora}
        on = [g[(f, f)] for f in corpora]
        off = [g[(f, q)] for f in corpora for q in corpora if f != q]
        return {"D": statistics.mean(on) - statistics.mean(off),
                "diag_mean": statistics.mean(on), "offdiag_mean": statistics.mean(off),
                "a_fit": a_f, "b_read": b_q,
                "g": {f"{f}|{q}": v for (f, q), v in g.items()}}

    def surrogate_D(mat, corpora):
        """C6 — the LINK-ARTIFACT null, and it is not small.

        An ADDITIVE decomposition of a bounded rate is not an instrument for "matching". For any
        y[f,q] = phi(u_f + v_q) with NO matching term, a second-order expansion leaves
        g[f,q] = phi'' * (u_f - ubar)(v_q - vbar), so D picks up phi'' * Cov(u_f, v_f) purely
        because operator quality and read-easiness are correlated across the SHARED corpus index.
        Measured on E36's own 5x5 matrix: corr(a_f, b_f) = +0.325, and a purely MULTIPLICATIVE
        surrogate built from that matrix's own margins — zero matching by construction — yields
        D = +0.000280 against an observed +0.001214. **23% of a positive D is a link artifact.**

        This returns D for the multiplicative surrogate with the SAME margins as `mat`. It is the
        value D would take with no matching at all, and it is what D must be compared against.
        """
        gm = statistics.mean([mat[(f, q)] for f in corpora for q in corpora])
        rm = {f: statistics.mean([mat[(f, q)] for q in corpora]) for f in corpora}
        cm = {q: statistics.mean([mat[(f, q)] for f in corpora]) for q in corpora}
        sur = {(f, q): rm[f] * cm[q] / gm for f in corpora for q in corpora}
        return diag_excess(sur, corpora)["D"]

    results = {}
    for ag in ("persist", "min"):
        mat = {(f, q): y(f, q, ag) for f in fit_corpora for q in fit_corpora}
        full = diag_excess(mat, fit_corpora)
        nog = fit_corpora
        sub = [c for c in fit_corpora if c not in DEGENERATE]
        without = diag_excess({k: v for k, v in mat.items()
                               if k[0] in sub and k[1] in sub}, sub)
        # C4 permutation null: shuffle the FIT labels relative to the read rungs
        g_ = torch.Generator().manual_seed(11)
        null = []
        for _ in range(a.n_perm):
            perm = [fit_corpora[i] for i in torch.randperm(len(fit_corpora), generator=g_).tolist()]
            mp = dict(zip(fit_corpora, perm))
            null.append(diag_excess({(mp[f], q): v for (f, q), v in mat.items()},
                                    fit_corpora)["D"])
        n_ge = sum(1 for v in null if abs(v) >= abs(full["D"]))
        # the same permutation null for the WITHOUT-DEGENERATE form, so DECLARED BIAS 2 ("the
        # claim is made only in the form that survives both") is actually executable — it was not,
        # because only the all-8 form had an interval.
        matw = {k: v for k, v in mat.items() if k[0] in sub and k[1] in sub}
        gw = torch.Generator().manual_seed(12)
        nullw = []
        for _ in range(a.n_perm):
            perm = [sub[i] for i in torch.randperm(len(sub), generator=gw).tolist()]
            mp = dict(zip(sub, perm))
            nullw.append(diag_excess({(mp[f], q): v for (f, q), v in matw.items()}, sub)["D"])
        n_gew = sum(1 for v in nullw if abs(v) >= abs(without["D"]))
        results[ag] = {
            "matrix": {f"{f}|{q}": v for (f, q), v in mat.items()},
            "all8": full, "without_degenerate": without,
            "C4_permutation": {"null_mean": statistics.mean(null),
                               "null_sd": statistics.pstdev(null),
                               "n_abs_ge_observed": n_ge, "n_perm": a.n_perm,
                               "p_two_sided": (1 + n_ge) / (1 + a.n_perm)},
            "C4_permutation_without_degenerate": {
                "null_mean": statistics.mean(nullw), "null_sd": statistics.pstdev(nullw),
                "n_abs_ge_observed": n_gew, "n_perm": a.n_perm,
                "p_two_sided": (1 + n_gew) / (1 + a.n_perm)},
            "C6_link_artifact_null": {
                "surrogate_D_all8": surrogate_D(mat, fit_corpora),
                "surrogate_D_without_degenerate": surrogate_D(matw, sub),
                "D_minus_surrogate_all8": full["D"] - surrogate_D(mat, fit_corpora),
                "D_minus_surrogate_without_degenerate": (without["D"]
                                                         - surrogate_D(matw, sub)),
                "note": ("a multiplicative surrogate with the SAME margins has zero matching by "
                         "construction; its D is the link artifact. D must exceed it, not 0.")},
        }

    # interval on D, eval set as the outer replication unit, prefix seed nested
    def pairvec(f, q, ps, ag="persist"):
        return torch.stack([CELL[(q, ps)]["arms"][f"{f}|s{s}"][ag] for s in seeds]).mean(0)
    per_set_D = {}
    for st in ADMITTED:
        blocks = []
        for ps in pseeds:
            mat = {(f, q): float(pairvec(f, q, ps)[SET_IDX[st]].mean())
                   for f in fit_corpora for q in fit_corpora}
            blocks.append(diag_excess(mat, fit_corpora)["D"])
        per_set_D[st] = blocks
    # D is one SCALAR per (eval set, prefix seed) — there is no item axis, because D is already a
    # contrast over the whole matrix. Wrapping each scalar as a length-1 vector makes the seeded
    # resampler do exactly the right thing: resample eval SETS, then prefix SEEDS within each,
    # with the inner item draw degenerate. The baseline arm is the constant 0, so the interval is
    # on D itself rather than on a difference of two measured arms.
    Dboot = hierarchical_bootstrap_seeded(
        {st: ([[v] for v in per_set_D[st]], [0.0]) for st in ADMITTED},
        n_boot=a.n_boot, seed=0)
    # per_set_D was previously computed and DISCARDED. It must be stored: `adm` averages the five
    # admitted sets BEFORE the contrast, and E51 established that the corpus x set interaction
    # (6.8% of variance) exceeds the corpus main effect (1.9%) with signs that reverse per set
    # (Github +1.3 on order-ops, -1.9 on multihop). Sign-opposed per-set matching therefore
    # CANCELS in D, so a null on D is only a null on the set-averaged contrast. Storing the
    # per-set values is what makes that statement checkable rather than an assumption.
    per_set_D_out = {st: {"per_prefix_seed": per_set_D[st],
                          "mean": statistics.mean(per_set_D[st]),
                          "sd": statistics.pstdev(per_set_D[st])} for st in ADMITTED}
    signs = [1 if per_set_D_out[st]["mean"] > 0 else -1 for st in ADMITTED]
    per_set_sign_split = {"n_positive": sum(1 for s in signs if s > 0),
                          "n_negative": sum(1 for s in signs if s < 0),
                          "note": ("if the sets split in sign, a null on the set-averaged D is "
                                   "cancellation, not absence of matching")}

    # ------------------------------------------------------------------ secondary
    secondary = {}
    for q in (OOD if not a.smoke else fit_corpora):
        if q not in rungs:
            continue
        matched = statistics.mean([adm(CELL[(q, ps)]["arms"][f"{q}|s{s}"]["persist"])
                                   for s in seeds for ps in pseeds])
        lg = statistics.mean([adm(CELL[(q, ps)]["arms"]["logit"]["persist"]) for ps in pseeds])
        sh = statistics.mean([adm(CELL[(q, ps)]["arms"]["shuf"]["persist"]) for ps in pseeds])
        best_in = max((statistics.mean([adm(CELL[(q, ps)]["arms"][f"{c}|s{s}"]["persist"])
                                        for s in seeds for ps in pseeds]), c)
                      for c in fit_corpora if c in INSTREAM)
        secondary[q] = {
            "matched_J^q": matched, "logit": lg, "shuf_floor": sh,
            "best_instream_J^P": best_in[0], "best_instream_corpus": best_in[1],
            "beats_logit": matched > lg, "beats_shuf_floor": matched > sh,
            "beats_best_instream": matched > best_in[0],
            "delta_vs_logit": matched - lg, "delta_vs_best_instream": matched - best_in[0]}

    # ------------------------------------------------------------------ controls
    ctrl = {}
    if a.rstrip:
        # E52's C1 compares the Q0 rung to e48's stored per-corpus asymptotes, which were scored at
        # the UNSTRIPPED readout. In the stripped arm it therefore cannot fire, by construction --
        # recorded as inapplicable rather than left to read as a failure.
        ctrl["C1_Q0_vs_e48"] = {
            "applicable": False,
            "why": ("results/e48_crossover_410m.json was scored at the unstripped readout. "
                    "Comparing a stripped Q0 rung against it would compare two conventions, not "
                    "two runs. The stripped counterpart is R2 "
                    "(results/e48_crossover_410m_rstrip.json)."),
            "fires": None}
        ctrl["C1_fires"] = None
    if os.path.exists(a.e48) and not a.smoke and "Q0" in rungs and not a.rstrip:
        e48 = json.load(open(a.e48))
        diffs = {}
        for c in fit_corpora:
            if c in e48["rungs"]:
                here = statistics.mean([adm(CELL[("Q0", ps)]["arms"][f"{c}|s{s}"]["persist"])
                                        for s in seeds for ps in pseeds])
                there = e48["rungs"][c]["persist"]["jp_admitted_mean"]
                diffs[c] = {"here": here, "e48": there, "abs_diff": abs(here - there)}
        ctrl["C1_Q0_vs_e48"] = {
            "per_corpus": diffs, "max_abs_diff": max(v["abs_diff"] for v in diffs.values()),
            "tolerance": 1e-6,
            "note": ("in-stream rungs are EXPECTED to differ: E48 scored the fastfit E28 lenses, "
                     "this scores the matched-provenance INSTREAM refits. The OOD rungs use the "
                     "SAME .pt files in both and must agree to fp32."),
            "ood_max_abs_diff": max((v["abs_diff"] for c, v in diffs.items()
                                     if c.startswith("OOD_")), default=None)}
        # NOT `(x or 1) < 1e-6`: a PERFECT reproduction gives x == 0.0, which is falsy, so
        # `or 1` substitutes 1 and the control reports DOES NOT FIRE on a flawless match.
        _ood = ctrl["C1_Q0_vs_e48"]["ood_max_abs_diff"]
        ctrl["C1_fires"] = bool(_ood is not None and _ood < 1e-6)
    below = {}
    for q in rungs:
        sh = statistics.mean([adm(CELL[(q, ps)]["arms"]["shuf"]["persist"]) for ps in pseeds])
        below[q] = [c for c in fit_corpora if c not in DEGENERATE
                    and statistics.mean([adm(CELL[(q, ps)]["arms"][f"{c}|s{s}"]["persist"])
                                         for s in seeds for ps in pseeds]) <= sh]
    ctrl["C2_shuf_floor_excl_degenerate"] = {"below_floor": below,
                                             "fires": all(not v for v in below.values())}
    q0r = statistics.mean([CELL[("Q0", ps)]["capability_mean_rank"] for ps in pseeds]) \
        if "Q0" in rungs else None
    caps = {q: statistics.mean([CELL[(q, ps)]["capability_mean_rank"] for ps in pseeds])
            for q in rungs}
    ctrl["C3_capability"] = {"q0_mean_rank": q0r, "per_rung": caps,
                             "collapsed": [q for q in rungs if q0r and caps[q] > 2 * q0r],
                             "fires": not [q for q in rungs if q0r and caps[q] > 2 * q0r]}
    ctrl["C5_self_distance"] = {"value": 0.0, "fires": True}

    # ---------------------------------------------------------------- R1 / P0 controls
    # R1 C2 -- the convention actually bites, and by exactly the amount the slate says it does.
    div = readout_divergence(tok, load_eval, readout_position, EVAL_SETS)
    req = {"multihop": 83, "multilingual": 19, "order-ops": 55,
           "poetry": 0, "typo": 0, "association": 0}
    obs = {k: v["n_divergent"] for k, v in div["per_set"].items()}
    ctrl["R1_C2_readout_divergence"] = {
        "required_total": 157, "required_per_set": req,
        "observed_total": div["total_divergent"], "observed_per_set": obs,
        "n_released_items": div["total_items"],
        "examples": div["first_divergent_example_per_set"],
        "note": ("computed over ALL released items in all six sets, independently of --rstrip: it "
                 "describes the difference between the two conventions, not this run's arm."),
        "fires": bool(div["total_divergent"] == 157 and obs == req)}

    # R1 C3 -- the identity-transport arm must MOVE between conventions. If it does not, the flag
    # never reached the scoring path and the whole arm is void.
    UNSTRIPPED_LOGIT_Q0_PERSIST = 0.0284395
    logit_q0 = (statistics.mean([adm(CELL[("Q0", ps)]["arms"]["logit"]["persist"]) for ps in pseeds])
                if "Q0" in rungs else None)
    ctrl["R1_C3_logit_constant_moves"] = {
        "arm": "STRIPPED" if a.rstrip else "UNSTRIPPED",
        "observed_logit_Q0_persist": logit_q0,
        "unstripped_reference": UNSTRIPPED_LOGIT_Q0_PERSIST,
        "reference_source": "E33/E36 agree to 2.6e-10 on the unstripped Q0 logit constant",
        "abs_diff_vs_unstripped_reference": (None if logit_q0 is None
                                             else abs(logit_q0 - UNSTRIPPED_LOGIT_Q0_PERSIST)),
        "fires": (None if logit_q0 is None else
                  (abs(logit_q0 - UNSTRIPPED_LOGIT_Q0_PERSIST) > 1e-6) if a.rstrip
                  else (abs(logit_q0 - UNSTRIPPED_LOGIT_Q0_PERSIST) <= 1e-6))}

    # R1 C4 is E52's existing C2_shuf_floor_excl_degenerate, re-asserted inside the stripped arm.
    ctrl["R1_C4_derangement_floor"] = {
        "alias_of": "C2_shuf_floor_excl_degenerate",
        "required": "no non-degenerate operator at or below its rung's shuf floor under persist",
        "fires": ctrl["C2_shuf_floor_excl_degenerate"]["fires"]}

    # P0 C3 -- thread oversubscription, asserted on OBSERVED state inside a worker, not in a comment.
    ctrl["P0_C3_worker_threads"] = {
        "workers": a.workers, "observed_threads_in_cell": worker_threads,
        "required": "1 when --workers > 1",
        "fires": (worker_threads == 1) if a.workers > 1 else None}

    D = results["persist"]["all8"]["D"]
    Dw = results["persist"]["without_degenerate"]["D"]
    lo, hi = Dboot["ci_lo"], Dboot["ci_hi"]
    sur = results["persist"]["C6_link_artifact_null"]["surrogate_D_all8"]
    surw = results["persist"]["C6_link_artifact_null"]["surrogate_D_without_degenerate"]
    beats_link = (D > sur) and (Dw > surw)
    if lo > 0 and beats_link:
        verdict = (f"MATCHING CONFIRMED — the diagonal excess D = {D:+.5f} has a CI strictly above "
                   f"zero [{lo:+.5f},{hi:+.5f}] AND exceeds the link-artifact null "
                   f"({sur:+.5f}; {surw:+.5f} without the degenerate operator). Fitting on the "
                   f"corpus you then read on buys something beyond operator quality, read "
                   f"difficulty, and the curvature of a bounded rate.")
    elif lo > 0 and not beats_link:
        verdict = (f"LINK-ARTIFACT — D = {D:+.5f} has a CI above zero [{lo:+.5f},{hi:+.5f}] but "
                   f"does NOT exceed the multiplicative-surrogate null ({sur:+.5f}), which has "
                   f"zero matching by construction. An additive decomposition of a bounded rate "
                   f"produces this from correlated margins alone. NOT evidence of matching.")
    elif hi < 0:
        verdict = (f"ANTI-MATCHING — D = {D:+.5f}, CI [{lo:+.5f},{hi:+.5f}] strictly below zero. "
                   f"Operators read corpora they were NOT fitted on BETTER. Reported separately; "
                   f"this is neither branch of the primary rule.")
    else:
        verdict = (f"NO MATCHING (the publishable null) — D = {D:+.5f}, CI [{lo:+.5f},{hi:+.5f}] "
                   f"includes zero. The corpus effect established in E28/E33/E48 is a property of "
                   f"the FITTING CORPUS PER SE; the fit-read relationship is not the operative "
                   f"variable, and EQ2's ||J^P - J^Q|| gap framing is not what the data measures.")
    if abs(Dw - D) > abs(D):
        verdict += (f"  ||  FLAG: dropping the known-degenerate operator moves D from {D:+.5f} to "
                    f"{Dw:+.5f}. The claim is made only in the form that survives both.")

    _prereg = [("specs/archive/prereg/PREREG_E52_FACTORIAL.md", "E52, the original design"),
               ("specs/experiments/P0_t52_parallel_port.md", "P0, the pool port"),
               ("specs/experiments/R1_grid_rstrip.md", "R1, the readout correction")]
    rec = {
        "experiment": "E52 — the fit x read factorial: J^Q read on Q, the missing cell",
        "prereg": "specs/archive/prereg/PREREG_E52_FACTORIAL.md, written before this ran",
        "prereg_files": [{"path": q, "role": why,
                          "sha256": (sha256_file(os.path.join(HERE, "..", q))
                                     if os.path.exists(os.path.join(HERE, "..", q)) else None)}
                         for q, why in _prereg],
        "decision_rule_verbatim": {"R1": 'ACCEPT (headline robust): stripped fit_pct exceeds stripped read_pct under BOTH aggregations, and stripped fit_pct lies inside the published bootstrap interval [87.609, 92.171] under persist. | QUALIFIED (ordering survives, magnitude does not): stripped fit_pct still exceeds read_pct under both aggregations but falls outside the published interval. | REJECT (ordering does not survive): stripped fit_pct does not exceed read_pct under both aggregations. STOP. Alert the operator.', "P0": 'ACCEPT: the pooled run reproduces the serial run\'s stored matrix at max_abs_diff = 0.0 across all 128 cell values, and wallclock falls by at least 3x. | REJECT: any non-zero difference. STOP. Do not "investigate the tolerance" -- there is no tolerance.'},
        "readout_convention": ("STRIPPED — the anchor rule: readout at the token immediately "
                               "preceding `target`, i.e. the final prompt token after .rstrip(). "
                               "jacobian-lens/data/evaluations/README.md; src/anchor_evals.py:228"
                               if a.rstrip else
                               "UNSTRIPPED (LEGACY) — readout at the trailing-space token, which "
                               "does not occur in the tokenisation of prompt+target. Retained only "
                               "to reproduce the published run; superseded by the stripped arm."),
        "rstrip": bool(a.rstrip),
        "workers": a.workers,
        "wallclock_seconds": round(wall, 1),
        "cell_completion_seconds": cell_seconds,
        "status": "PRE-REGISTERED",
        "question": ("can a J-lens trained on an out-of-distribution sample read latent structure "
                     "on an out-of-distribution corpus?"),
        "model": MODEL, "band": BAND, "K": K, "admitted_sets": ADMITTED,
        "fit_corpora": fit_corpora, "read_rungs": rungs,
        "fit_seeds": seeds, "prefix_seeds": pseeds,
        "n_operators": len(fit_corpora) * len(seeds), "n_items": len(items), "n_pairs": P_n,
        "prefix_tokens": PREFIX_TOKENS, "smoke": a.smoke,
        "all_operators_share_one_fitter": ("trainval.py N=200 --band 9,21 dim_batch=128 "
                                           "max_seq_len=128 skip_first=16 target_layer=-2 fp32"),
        "adjudication": {"aggregation": "persist", "min_does_not_vote": True,
                         "D": D, "D_without_degenerate": Dw,
                         "D_hierarchical_ci": Dboot,
                         "degenerate_excluded_from_C2": DEGENERATE},
        "by_aggregation": results,
        "per_set_D": per_set_D_out,
        "per_set_sign_split": per_set_sign_split,
        "PARTIAL_UNBLINDING_DISCLOSED": (
            "E36's stored ladder already contains a 5x5 in-stream fit x read matrix (E28 N=400 "
            "operators, same aggregation, same admitted sets). Applying this script's diag_excess "
            "to it gives D = +0.001214 with all five diagonal residuals positive and a "
            "permutation p of 0.134 — i.e. positive but NOT significant. That estimate was known "
            "before E52 ran and is recorded here so E52's primary is not presented as blind for "
            "the in-stream half. It also carries ~17% fit/prefix leakage (E28 fitted N=400 from "
            "the same 800-doc block), which is the confound Amendment 1 removes, so its sign is "
            "exactly what leakage predicts."),
        "secondary_ood_rungs": secondary,
        "controls": ctrl,
        "VERDICT": verdict,
    }
    write_result(a.out, rec, experiment="E52",
                 inputs=[a.e48] + [lens_path(c, s) for c in fit_corpora for s in seeds])

    if a.cells_out:
        # C1 (E57): the recomputed matrix must equal the STORED e52 matrix exactly. This is the
        # only check that E52 is reproducible at all -- it has never been re-run -- and if it
        # fails, the per-cell draws describe a different measurement and must not be used.
        ref_path = os.path.join(HERE, "..", "results", "e52_factorial_410m.json")
        c1 = {"reference": "results/e52_factorial_410m.json", "checked": False}
        if a.rstrip:
            c1 = {"reference": "results/e52_factorial_410m.json", "checked": False,
                  "applicable": False, "fires": None,
                  "why": ("the stored matrix was scored at the UNSTRIPPED readout. R1/C1 is the "
                          "check that this same code reproduces it WITHOUT --rstrip; comparing the "
                          "stripped arm to it would be comparing two conventions.")}
        if os.path.exists(ref_path) and not a.smoke and not a.rstrip:
            ref = json.load(open(ref_path))["by_aggregation"]
            worst, worst_key = 0.0, None
            for ag in ("persist", "min"):
                for k, v in results[ag]["matrix"].items():
                    dv = abs(v - ref[ag]["matrix"][k])
                    if dv > worst:
                        worst, worst_key = dv, f"{ag}:{k}"
            c1 = {"reference": "results/e52_factorial_410m.json", "checked": True,
                  "n_cells_compared": 2 * len(results["persist"]["matrix"]),
                  "max_abs_diff": worst, "worst_cell": worst_key,
                  "tolerance": 1e-12, "fires": worst <= 1e-12}
            print(f"\nC1 E52 REPRODUCTION: max|diff| over {c1['n_cells_compared']} cells = "
                  f"{worst:.3e} at {worst_key}  ->  {'FIRES' if c1['fires'] else 'DOES NOT FIRE'}",
                  flush=True)
        cells_rec["control_C1_reproduces_stored_e52_matrix"] = c1
        write_result(a.cells_out, cells_rec, experiment="E57",
                     inputs=[ref_path] + [lens_path(c, s) for c in fit_corpora for s in seeds])
        print(f"wrote {a.cells_out}", flush=True)

    print(f"\nFIT x READ matrix (persist, admitted mean) — rows = fit corpus, cols = read rung")
    print(f"{'fit \\ read':22s}" + "".join(f"{q[:9]:>10}" for q in fit_corpora))
    for f in fit_corpora:
        row = "".join(f"{results['persist']['matrix'][f'{f}|{q}']:10.5f}" for q in fit_corpora)
        print(f"{f:22s}{row}")
    print(f"\ndiagonal excess D = {D:+.6f}   CI [{lo:+.6f}, {hi:+.6f}]"
          f"   (without {DEGENERATE}: {Dw:+.6f})")
    p4 = results["persist"]["C4_permutation"]
    print(f"C4 permutation null: mean {p4['null_mean']:+.6f} sd {p4['null_sd']:.6f}  "
          f"p(two-sided) = {p4['p_two_sided']:.4f}")
    print("\nSECONDARY — the OOD rungs, matched operator vs the alternatives:")
    for q, v in secondary.items():
        print(f"  read on {q:20s} J^q={v['matched_J^q']:.5f}  logit={v['logit']:.5f}  "
              f"shuf={v['shuf_floor']:.5f}  best in-stream J^P={v['best_instream_J^P']:.5f} "
              f"({v['best_instream_corpus']})")
        print(f"      beats logit {v['beats_logit']} | beats floor {v['beats_shuf_floor']} | "
              f"beats best in-stream {v['beats_best_instream']} "
              f"(delta {v['delta_vs_best_instream']:+.5f})")
    for k, v in ctrl.items():
        if isinstance(v, dict) and "fires" in v:
            print(f"  {k:34s} {'FIRES' if v['fires'] else 'DOES NOT FIRE'}")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
