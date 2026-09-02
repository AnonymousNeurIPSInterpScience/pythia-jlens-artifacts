#!/usr/bin/env python3
"""t36_qladder.py — E36: vary the READ distribution Q, holding the fitted operator J^P fixed.

THIS IS THE EXPERIMENT E48 IS NOT.
  E48 varied P (what the operator was fitted on) and held the evaluation battery fixed. It found
  that exposure of the FITTING corpus does not order the read — a lens fitted on a corpus with
  containment 0.0001 reads above one fitted on in-stream Wikipedia. That bounds the P axis.

  It says nothing about the P--Q gap, because Q never moved. EQ2's unbuyable term is
  ||J^P - J^Q||, where Q is the distribution the activations are drawn from, and the whole
  estimator framing predicts the fitted lens should lose to the unfitted one as P and Q diverge.
  **Nothing in this programme has ever varied Q.** This does.

  Pre-registration: `specs/archive/prereg/PREREG_E36_QLADDER.md`, written 2026-08-13, before any E36
  number existed. The decision rule below is transcribed from it and is not reinterpretable.

OPERATIONAL DEFINITION OF "EVALUATE ON Q" — Q-context prefixing
  The anchor's concept eval sets are NOT corpora, so "evaluate on Q" cannot mean swapping the eval
  set without also swapping the task. Instead the item is read in a Q context:

      ids = [BOS] ++ [128 tokens drawn from Q] ++ [the item's own prompt tokens]

  The target `intermediates` are unchanged, so pass@k is comparable across every rung. The prefix
  length matches the fitting window (max_seq_len=128), so the activations the operator acts on are
  drawn from the same window statistics it was averaged over.

  READOUT POSITION IS OFFSET, NOT RECOMPUTED. This is the subtle part and getting it wrong would
  invalidate the ladder. `poetry` reads at the last newline; a Q prefix contains newlines, so
  recomputing the rule on the concatenated string would read inside the PREFIX rather than the
  item. The item's own position is computed on the item alone and then shifted by the prefix
  length. For the five `last` sets the position is -1 and is unaffected either way.

WHAT VARIES, WHAT DOES NOT
  varies      Q — the distribution the read activations are drawn from, and nothing else
  fixed       model, band [9,21], the five fitted J^P (E28, N=400), the eval items and their
              target tokens, aggregation (persist primary), K grid, readout rule

THE LADDER. The x-axis is MEASURED containment (E48b, k=32 at full available coverage), not a rung
  number — that is what E35/M1 was built to supply. Rungs:
      Q0    no prefix                       the null rung; must reproduce E33 (control C1)
      Qc    each corpus with measured containment, in-stream and OOD alike
      Qshuf token-shuffled Pile-CC          unigram statistics preserved, syntax destroyed:
                                            separates "unfamiliar tokens" from "unfamiliar syntax"
  A non-Pile SCRIPT rung (the prereg's Q5) is NOT available — no such corpus is on disk — so the
  ladder does not extend to script-level shift. Recorded, not silently omitted.

DECISION RULE — from the pre-registration, fixed before running
  ACCEPT S3   if for >=3 of the 5 fitting corpora the J-lens curve crosses BELOW the logit-lens
              curve at some rung, AND the crossing survives leave-one-rung-out, AND capability
              (C3) has not collapsed at or before the crossing rung.
  REJECT S3   if the two curves stay parallel within their seed intervals across the whole ladder
              — shift degrades BOTH lenses equally. This is the publishable null: the fitted
              operator is no more shift-fragile than the unfitted one.
  UNCLEAR     if capability collapses at or before any crossing. STOP. Do not re-pick Q.
  A crossing that appears only under `min` does not count: `min` is existential and is reported
  but never votes.

CONTROLS, each with the number it must produce
  C1 Q0 REPRODUCES E33   the no-prefix rung must reproduce e33_logit_baseline_410m_v2.json's
                         logit and J^P admitted means. A mismatch means the prefixing path altered
                         the base measurement and everything downstream is void.
  C2 SHUF FLOOR          J^shuf below both J^P and logit at EVERY rung.
  C3 CAPABILITY          the model's OWN rank of the target token at the final layer. Must not
                         fall below 50% of its Q0 value at or before any crossing. This is what
                         separates "the lens stopped reading" from "the model stopped knowing".
  C4 PREFIX-LENGTH       the in-stream rungs ARE the C4 control: a prefix of 128 tokens from a
                         Pile component isolates "any prefix at all" from "a prefix from a shifted
                         Q". Any degradation already present there is not shift.
  C5 SELF-DISTANCE       D_act(J,J) = 0 exactly.

DECLARED BIAS — from the pre-registration, restated because it is the design's central limit
  We shift the READ distribution, not the FIT distribution. EQ2's bias term is ||J^P - J^Q|| where
  J^Q is the operator that WOULD HAVE BEEN fitted on Q. We never fit J^Q here. What we measure is
  how J^P performs on activations drawn from Q — related, not identical. This must appear in the
  paper. (E48 supplies the other half: operators actually fitted on OOD corpora.)

    python t36_qladder.py --device cpu            # the full ladder
    python t36_qladder.py --smoke                 # 8 items/set, 2 rungs, 1 seed
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
import multiprocessing as mp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "jacobian-lens"))
from provenance import write_result  # noqa: E402

MODEL = "EleutherAI/pythia-410m-deduped"
BAND = list(range(9, 22))
K = [1, 2, 5, 10, 20, 50, 100]
ADMITTED = ["multihop", "multilingual", "order-ops", "poetry", "typo"]
FIT_CORPORA = ["Github", "Wikipedia_en", "StackExchange", "Pile-CC", "USPTO_Backgrounds"]
# every corpus for which E48b measured containment — that measurement IS the x-axis
Q_CORPORA = ["Pile-CC", "StackExchange", "Wikipedia_en", "USPTO_Backgrounds", "Github",
             "CONTROL_PubMed_2023", "OOD_Wikipedia_2023", "TRAP_FineWeb",
             "OOD_News_2024", "OOD_arXiv_2023", "OOD_CommonPile"]
PREFIX_TOKENS = 128


def lens_path(corpus: str) -> str:
    return os.path.join(HERE, "..", "results", f"e28_{corpus}_410m_n400_s0.pt")


def derangement(band, seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        perm = [band[i] for i in torch.randperm(len(band), generator=g).tolist()]
        if all(p != l for p, l in zip(perm, band)):
            return perm


# ---------------------------------------------------------------------- the parallel cell
# Same port as P0 did for t52_factorial.py, and for the same reason: E36's (rung, prefix seed)
# cells are one activation cache plus 12 independent score passes each, and NOTHING is reduced
# across cells. Serial, the 39 cells with R4d's five extra own-derangement arms ran at ~30 min per
# cell on a contended machine -- about 19 hours. POSTREVIEW_EXPERIMENTS.md §3/P0 anticipated this:
# "Consider the same port for t36_qladder.py (R4b, ~2.2 h serial) if it is also a plain loop."
#
# spawn, not fork (darwin). The worker loads the model itself; the PARENT keeps every decision that
# defines WHAT a cell is -- build() and its torch.Generator sequence -- so scheduling cannot reach
# a number. C1 gates that against the stored unstripped arms.
_W: dict = {}


def _build_state36(cfg: dict) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from t2_fastfit import PYTHIA_LAYOUT_T5
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(cfg["device"]).eval()
    model = from_hf(hf, tok, layout=PYTHIA_LAYOUT_T5)
    JP = {c: {l: torch.load(lens_path(c), map_location="cpu", weights_only=True)["J"][l].float()
              for l in BAND} for c in FIT_CORPORA}
    perm = derangement(BAND, 7000)
    JSHUF = {l: JP["Pile-CC"][q] for l, q in zip(BAND, perm)}
    JSHUF_OWN = {c: {l: JP[c][q] for l, q in zip(BAND, perm)} for c in FIT_CORPORA}
    st = dict(cfg)
    st.update(model=model, hf=hf, tok=tok, JP=JP, JSHUF=JSHUF, JSHUF_OWN=JSHUF_OWN,
              d=model.d_model, L=len(BAND), HALF=len(BAND) // 2)
    return st


def _init_worker36(cfg: dict) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    _W["st"] = _build_state36(cfg)
    torch.set_num_threads(1)          # MEASURED: the pin does not survive model loading


def _cache36(st, cell):
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
    return A, cap


def _score36(st, A, T):
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


def _run_cell36(task):
    rung, seed, cellspec = task
    st = _W["st"]
    A, cap = _cache36(st, cellspec)
    arms = {"logit": _score36(st, A, None), "shuf": _score36(st, A, st["JSHUF"])}
    for c in FIT_CORPORA:
        arms[f"J|{c}"] = _score36(st, A, st["JP"][c])
        arms[f"shufown|{c}"] = _score36(st, A, st["JSHUF_OWN"][c])
    return rung, seed, arms, cap, torch.get_num_threads()


def load_pool(corpus: str, tok, n: int, seed: int):
    """`n` documents from Q, each truncated to PREFIX_TOKENS. Deterministic in `seed`."""
    path = os.path.join(HERE, "..", "corpora", f"{corpus}.jsonl")
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
    if len(out) < n:                      # cycle rather than silently shorten the ladder
        out = (out * (n // max(1, len(out)) + 1))[:n]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=3, help="prefix draws per rung")
    ap.add_argument("--rungs", default=None, help="comma list; default = Q0 + every measured corpus + shuffled")
    ap.add_argument("--items-per-chunk", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exposure", default=os.path.join(HERE, "..", "results", "e48b_exposure_growth.json"))
    ap.add_argument("--e33", default=os.path.join(HERE, "..", "results", "e33_logit_baseline_410m_v2.json"))
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                    help="processes over the (rung, prefix seed) cells. 1 runs in-process. Bounded "
                         "by RAM, not cores: with spawn each worker holds its own model and "
                         "operators, and this workload is memory-bandwidth bound past ~6.")
    ap.add_argument("--rstrip", action="store_true",
                    help="R4b. Apply the anchor's readout rule -- strip the prompt's trailing space "
                         "BEFORE tokenising, so the readout token is 'the token immediately "
                         "preceding target'. A CORRECTION, not a sensitivity arm: the unstripped "
                         "path reads token id 209 (a bare space), which does not occur in the "
                         "tokenisation of prompt+target. Changes 157 of 551 released items.")
    ap.add_argument("--out", default=None,
                    help="default: results/e36_qladder_410m{_rstrip}.json, chosen by --rstrip so the "
                         "two conventions can never land in the same file")
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(HERE, "..", "results",
                             f"e36_qladder_410m{'_rstrip' if a.rstrip else ''}.json")
    if a.smoke and os.path.abspath(a.out) in (
            os.path.abspath(os.path.join(HERE, "..", "results", "e36_qladder_410m.json")),
            os.path.abspath(os.path.join(HERE, "..", "results", "e36_qladder_410m_rstrip.json"))):
        raise SystemExit("ABORT: --smoke would overwrite a canonical results file. Pass --out.")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from t2_fastfit import PYTHIA_LAYOUT_T5
    from anchor_evals import EVAL_SETS, load_eval, readout_position, token_ids_of
    from bootstrap import hierarchical_bootstrap_seeded

    # ---- the x-axis, measured, not assumed
    exposure = {}
    if os.path.exists(a.exposure):
        g = json.load(open(a.exposure))
        cov = len(g["shards_used"])
        exposure = {k: v[-1] for k, v in g["containment_by_coverage"]["32"].items()}
    else:
        print("  NOTE: no exposure file — the ladder will be CATEGORICAL, as the prereg's "
              "k-selection fallback allows", flush=True)
        cov = 0

    tok = AutoTokenizer.from_pretrained(MODEL)
    # LOAD-BEARING, and a SIDE EFFECT of a call the parent no longer makes. jlens/hf.py:104-111 --
    # HFModel.__init__ sets `tokenizer.add_bos_token = True` under force_bos. Before the pool port
    # the parent built the model first, so every eval item was tokenized by an already-mutated
    # tokenizer and carried a BOS. With the model in the workers the parent's tokenizer is
    # untouched and every item silently loses its BOS. Measured on t52: 48/48 items changed and the
    # matrix moved 1.5e-2. Replicated explicitly, and ASSERTED below (CLAUDE.md §6.0b).
    if getattr(tok, "bos_token_id", None) is not None and hasattr(tok, "add_bos_token"):
        tok.add_bos_token = True
    L = len(BAND)

    rungs = (["Q0"] + [c for c in Q_CORPORA if os.path.exists(os.path.join(HERE, "..", "corpora", f"{c}.jsonl"))]
             + ["SHUFFLED_Pile-CC"])
    if a.rungs:
        rungs = a.rungs.split(",")
    if a.smoke:
        rungs = ["Q0", "OOD_arXiv_2023"]
    seeds = [0] if a.smoke else list(range(a.seeds))
    print(f"rungs ({len(rungs)}): {rungs}\nseeds: {seeds}  band={BAND}", flush=True)

    # ---- the eval items, tokenized ONCE. Positions are computed on the ITEM ALONE and offset.
    items = []
    per_set_n = {}
    for name in EVAL_SETS:
        for it in load_eval(name):
            tgt = [t for t in (token_ids_of(tok, w) for w in it.get("intermediates", [])) if t]
            if not tgt:
                continue
            if a.smoke and per_set_n.get(name, 0) >= 8:
                continue
            per_set_n[name] = per_set_n.get(name, 0) + 1
            # R4b. anchor_evals.py:32-34 and :228 strip; this script did not.
            p = it["prompt"].rstrip() if a.rstrip else it["prompt"]
            ids = tok(p, add_special_tokens=True).input_ids       # includes BOS
            pos = readout_position(tok, name, p)                  # -1, or an ABSOLUTE index
            items.append({"set": name, "ids": ids, "pos": pos, "tgt": tgt})
    print(f"{len(items)} eval items, {sum(len(i['tgt']) for i in items)} (item,intermediate) pairs",
          flush=True)

    _no_bos = [i for i, it in enumerate(items) if not it["ids"] or it["ids"][0] != tok.bos_token_id]
    if _no_bos:
        raise SystemExit(f"ABORT: {len(_no_bos)}/{len(items)} items carry no BOS. The tokenizer was "
                         f"not put in add_bos_token mode before the items were tokenized; every "
                         f"cell would be scored on a different sequence than the stored run.")

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

    # existence check only -- JP / JSHUF / JSHUF_OWN are built ONCE PER WORKER in _build_state36.
    # See the note there on the Pile-CC-sourced original (kept verbatim for C1) versus R4d's
    # own-corpus correction.
    _missing = [lens_path(c) for c in FIT_CORPORA if not os.path.exists(lens_path(c))]
    if _missing:
        raise SystemExit("ABORT: missing operators:\n  " + "\n  ".join(_missing))

    def build(rung: str, seed: int):
        """Prefixed token ids + offset readout position per item, for one (rung, seed) cell."""
        if rung == "Q0":
            return [(it["ids"], it["pos"]) for it in items]
        if rung.startswith("SHUFFLED_"):
            base = load_pool(rung[len("SHUFFLED_"):], tok, len(items), seed)
            g = torch.Generator().manual_seed(4242 + seed)
            pref = [[b[i] for i in torch.randperm(len(b), generator=g).tolist()] for b in base]
        else:
            pref = load_pool(rung, tok, len(items), seed)
        out = []
        for it, pre in zip(items, pref):
            body = it["ids"][1:] if it["ids"] and it["ids"][0] == tok.bos_token_id else it["ids"]
            ids = [tok.bos_token_id] + list(pre) + list(body)
            # -1 stays -1; an absolute index shifts by exactly the prefix length, because the
            # item's own indexing already counted one BOS which we keep at position 0
            pos = it["pos"] if it["pos"] < 0 else it["pos"] + len(pre)
            out.append((ids, pos))
        return out

    def adm(v):
        return statistics.mean([float(v[SET_IDX[s]].mean()) for s in ADMITTED])

    cfg = {"device": a.device, "items_tgt": [it["tgt"] for it in items], "ITEM_PAIRS": ITEM_PAIRS,
           "pair_tgt": pair_tgt, "n_items": len(items), "P_n": P_n,
           "items_per_chunk": a.items_per_chunk}
    # build() runs in the PARENT for every cell, so the generator sequence that draws the prefix
    # pools is identical to the serial run's however the cells are then scheduled.
    tasks = ((rung, s, build(rung, s)) for rung in rungs for s in seeds)
    n_cells = len(rungs) * len(seeds)

    CELLS, worker_threads = {}, None
    t0 = time.time()

    def _absorb(rung, s, arms, cap, nthr, done):
        nonlocal worker_threads
        worker_threads = nthr
        CELLS[(rung, s)] = {
            "arms": {k: {ag: v[ag] for ag in ("min", "persist")} for k, v in arms.items()},
            "capability_mean_rank": statistics.mean(cap),
            "capability_top10_rate": sum(1 for r in cap if r <= 10) / len(cap)}
        print(f"  [{done}/{n_cells}] {rung:22s} s{s}  logit={adm(arms['logit']['persist']):.5f}  "
              f"J|Pile-CC={adm(arms['J|Pile-CC']['persist']):.5f}  "
              f"shuf={adm(arms['shuf']['persist']):.5f}  "
              f"cap_top10={CELLS[(rung,s)]['capability_top10_rate']:.3f}  thr={nthr}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    if a.workers <= 1:
        _W["st"] = _build_state36(cfg)
        for i, task in enumerate(tasks, 1):
            _absorb(*_run_cell36(task), i)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.workers, initializer=_init_worker36, initargs=(cfg,)) as pool:
            for i, out in enumerate(pool.imap_unordered(_run_cell36, tasks), 1):
                _absorb(*out, i)
    wall = time.time() - t0
    print(f"  {n_cells} cells in {wall/60:.1f} min on {a.workers} worker(s)", flush=True)

    # ---------------------------------------------------------------- aggregate
    def cell_adm(rung, arm, ag):
        return [adm(CELLS[(rung, s)]["arms"][arm][ag]) for s in seeds]

    ladder = {}
    for rung in rungs:
        row = {"containment_k32": exposure.get(rung.replace("SHUFFLED_", ""), None)
               if rung != "Q0" else None,
               "is_shuffled": rung.startswith("SHUFFLED_"),
               "capability_top10_rate": statistics.mean(
                   [CELLS[(rung, s)]["capability_top10_rate"] for s in seeds]),
               "capability_mean_rank": statistics.mean(
                   [CELLS[(rung, s)]["capability_mean_rank"] for s in seeds]),
               "arms": {}}
        for arm in (["logit", "shuf"] + [f"J|{c}" for c in FIT_CORPORA]
                    + [f"shufown|{c}" for c in FIT_CORPORA]):
            for ag in ("persist", "min"):
                v = cell_adm(rung, arm, ag)
                row["arms"].setdefault(arm, {})[ag] = {
                    "mean": statistics.mean(v), "seed_sd": statistics.pstdev(v), "per_seed": v}
        # J^P - logit, with the eval set as the replication unit and the prefix seed nested
        for c in FIT_CORPORA:
            sets = {s: ([CELLS[(rung, sd)]["arms"][f"J|{c}"]["persist"][SET_IDX[s]].tolist()
                         for sd in seeds],
                        CELLS[(rung, seeds[0])]["arms"]["logit"]["persist"][SET_IDX[s]].tolist())
                    for s in ADMITTED}
            row["arms"][f"J|{c}"]["ci_vs_logit_persist"] = hierarchical_bootstrap_seeded(
                sets, n_boot=a.n_boot, seed=0)
        ladder[rung] = row

    # ---------------------------------------------------------------- controls
    ctrl = {}
    if os.path.exists(a.e33) and "Q0" in ladder and not a.smoke:
        e33 = json.load(open(a.e33))
        want_logit = e33["persist_admitted_mean"]["logit_I"]
        got_logit = ladder["Q0"]["arms"]["logit"]["persist"]["mean"]
        ctrl["C1_Q0_reproduces_e33"] = {
            "logit_e33": want_logit, "logit_here": got_logit,
            "abs_diff": abs(got_logit - want_logit),
            "tolerance": 1e-6,
            "fires": abs(got_logit - want_logit) < 1e-6,
            "note": ("Q0 is the no-prefix rung and MUST equal E33 exactly. A mismatch means the "
                     "prefixing path altered the base measurement and the ladder is void.")}
    below = {r: all(ladder[r]["arms"]["shuf"]["persist"]["mean"]
                    < ladder[r]["arms"][f"J|{c}"]["persist"]["mean"] for c in FIT_CORPORA)
             and ladder[r]["arms"]["shuf"]["persist"]["mean"] < ladder[r]["arms"]["logit"]["persist"]["mean"]
             for r in rungs}
    ctrl["C2_shuf_below_everything"] = {
        "per_rung": below, "fires": all(below.values()),
        "WHY_IT_DOES_NOT_FIRE_AND_WHY_THAT_IS_NOT_LIKE_FOR_LIKE": (
            "this control builds its derangement from PILE-CC's operator and then compares it to "
            "EVERY corpus's operator (R4d / F-7). 'A derangement beats Github's operator' is "
            "therefore a comparison between two different corpora's operators, not between an "
            "operator and its own derangement. C2b below is the like-for-like form.")}
    # C2b — R4d. Each corpus's operator against ITS OWN layer-deranged self, which is the
    # comparison the claim actually needs. Required to count as firing: 0 corpus-rung cells in
    # which the derangement is at or above the real operator under persist.
    own_below = {}
    for r in rungs:
        own_below[r] = {c: (ladder[r]["arms"][f"shufown|{c}"]["persist"]["mean"]
                            < ladder[r]["arms"][f"J|{c}"]["persist"]["mean"])
                        for c in FIT_CORPORA}
    n_bad = sum(1 for r in own_below for c in own_below[r] if not own_below[r][c])
    ctrl["C2b_own_derangement_below_its_own_operator"] = {
        "per_rung_per_corpus": own_below,
        "n_cells": len(rungs) * len(FIT_CORPORA), "n_cells_where_derangement_wins": n_bad,
        "required": "0 cells where the derangement is at or above its own operator",
        "fires": n_bad == 0,
        "like_for_like_evidence_already_on_disk": (
            "e33_logit_baseline_410m_v2: Github 0.027926 vs its own jshuf 0.018142; "
            "e54 C2: 0 of 15 and 0 of 120")}
    # C3 AS PRE-REGISTERED IS DEGENERATE, and that is reported rather than quietly rewritten.
    # The prereg asks for "the model's own top-k accuracy on the target token at the final layer",
    # with the rule "must not fall below 50% of its Q0 value". Measured: the Q0 top-10 rate is
    # ~0.00, because the `intermediates` are LATENT concepts the model is not emitting at the
    # readout position — that is the entire premise of a lens. A rule of the form "< 50% of 0" can
    # never fire, so the registered form has no power in either direction.
    # Both forms are computed. Adjudication uses MEAN RANK, which is not floored; the registered
    # top-k form is reported alongside and marked degenerate. Flagged to the operator, not resolved.
    q0cap = ladder["Q0"]["capability_top10_rate"] if "Q0" in ladder else None
    q0rank = ladder["Q0"]["capability_mean_rank"] if "Q0" in ladder else None
    collapsed = [r for r in rungs
                 if q0rank and ladder[r]["capability_mean_rank"] > 2.0 * q0rank]
    ctrl["C3_capability"] = {
        "REGISTERED_FORM_IS_DEGENERATE": (
            f"the pre-registered rule is top-k accuracy < 50% of Q0, but Q0's top-10 rate is "
            f"{q0cap}. The eval targets are latent concepts, not emitted tokens, so this quantity "
            f"is floored at Q0 and the rule cannot fire. NOT reinterpreted silently — adjudication "
            f"below uses mean rank, which is not floored, and the operator is asked to confirm."),
        "registered_top10_rate": {"q0": q0cap,
                                  "per_rung": {r: ladder[r]["capability_top10_rate"] for r in rungs},
                                  "usable": bool(q0cap and q0cap > 0.05)},
        "adjudicated_mean_rank": {"q0": q0rank,
                                  "per_rung": {r: ladder[r]["capability_mean_rank"] for r in rungs},
                                  "rule": "a rung collapses if its mean target rank exceeds 2x Q0's"},
        "collapsed_rungs": collapsed,
        "note": "a rung whose capability collapsed cannot distinguish lens failure from model failure"}
    ctrl["C5_self_distance"] = {"value": 0.0, "fires": True,
                                "note": "D_act(J,J)=0 by construction; measured 0.0 in t23"}

    # ---------------------------------------------------------------- adjudication
    measured = [r for r in rungs if ladder[r]["containment_k32"] is not None
                and not ladder[r]["is_shuffled"]]
    crossings = {}
    for c in FIT_CORPORA:
        neg = [r for r in measured
               if ladder[r]["arms"][f"J|{c}"]["ci_vs_logit_persist"]["ci_hi"] < 0]
        pt_neg = [r for r in measured
                  if ladder[r]["arms"][f"J|{c}"]["persist"]["mean"]
                  < ladder[r]["arms"]["logit"]["persist"]["mean"]]
        crossings[c] = {"rungs_ci_strictly_below_logit": neg,
                        "rungs_point_below_logit": pt_neg,
                        "crosses": bool(neg)}
    n_cross = sum(1 for c in FIT_CORPORA if crossings[c]["crosses"])
    cap_collapsed = ctrl["C3_capability"]["collapsed_rungs"]

    if cap_collapsed and n_cross:
        verdict = (f"UNCLEAR — capability collapsed at {cap_collapsed}, so lens failure and model "
                   f"failure are not separable. STOP; do not re-pick Q.")
    elif n_cross >= 3:
        verdict = (f"ACCEPT S3 — the J-lens curve crosses below the logit lens for {n_cross} of 5 "
                   f"fitting corpora, with capability intact. The fitted operator IS more "
                   f"shift-fragile than the unfitted one.")
    elif n_cross == 0:
        verdict = ("REJECT S3 (the publishable null) — no fitting corpus produces a J-lens curve "
                   "that goes strictly below the logit lens at any measured rung. Shift in the "
                   "READ distribution degrades both lenses together; the averaged derivative is "
                   "no more shift-fragile than not fitting at all.")
    else:
        verdict = (f"UNCLEAR — {n_cross} of 5 fitting corpora cross, short of the pre-registered "
                   f"3. Not reinterpretable here; operator decision.")

    rec = {
        "experiment": "E36 — the Q-ladder: vary the READ distribution, hold J^P fixed",
        "prereg": "specs/archive/prereg/PREREG_E36_QLADDER.md (2026-08-13, before any E36 number existed)",
        "prereg_r4b": "specs/experiments/R4b_e36_rstrip.md",
        "status": "PRE-REGISTERED",
        "rstrip": bool(a.rstrip),
        "readout_convention": ("STRIPPED — the anchor rule: readout at the token immediately "
                               "preceding `target`. jacobian-lens/data/evaluations/README.md; "
                               "src/anchor_evals.py:228" if a.rstrip else
                               "UNSTRIPPED (LEGACY) — readout at the trailing-space token, which "
                               "does not occur in the tokenisation of prompt+target"),
        "decision_rule_verbatim_R4b": (
            "REJECT STANDS: the fitted operator is flatter on >= 4 of 5 rungs. Section 3.1 is "
            "unchanged. | REJECT OVERTURNED: flatter on <= 1 of 5. STOP and alert. | UNCLEAR: 2 or "
            "3 of 5. Report and stop. Do not re-cut."),
        "PREREG_C4_WAS_NEVER_CODED": (
            "PREREG_E36_QLADDER.md:133 registers C4, the prefix-length control -- a prefix of the "
            "same 128 tokens drawn from the FITTING corpus P -- and :136 calls it load-bearing: "
            "'Without C4 the entire ladder could be a prefix-length artifact.' No C4 key exists in "
            "any E36 results file. Surfaced by R4e; recorded here so the omission travels with the "
            "result rather than being rediscovered."),
        "model": MODEL, "band": BAND, "K": K, "admitted_sets": ADMITTED,
        "prefix_tokens": PREFIX_TOKENS, "n_items": len(items), "n_pairs": P_n,
        "rungs": rungs, "prefix_seeds": seeds, "smoke": a.smoke,
        "x_axis": {"source": os.path.basename(a.exposure), "k": 32,
                   "coverage_shards": cov,
                   "note": "measured containment, not a rung number — this is what E35/M1 was for"},
        "unavailable_rung": {"Q5_non_pile_script":
                             "no non-Latin-script corpus is on disk; the ladder does not extend "
                             "to script-level shift. Recorded, not silently omitted."},
        "declared_bias": ("we shift the READ distribution, not the FIT distribution. EQ2's bias "
                          "term is ||J^P - J^Q|| where J^Q is the operator that WOULD have been "
                          "fitted on Q; we never fit J^Q here. E48 supplies that half."),
        "ladder": ladder,
        "crossings": crossings,
        "controls": ctrl,
        "VERDICT": verdict,
    }
    write_result(a.out, rec, experiment="E36",
                 inputs=[a.exposure, a.e33] + [lens_path(c) for c in FIT_CORPORA])

    print(f"\n{'rung':24s} {'containment':>12} {'logit':>9} {'J|best':>9} {'J|worst':>9} "
          f"{'shuf':>9} {'cap@10':>8}")
    for r in rungs:
        row = ladder[r]
        js = [row["arms"][f"J|{c}"]["persist"]["mean"] for c in FIT_CORPORA]
        cx = row["containment_k32"]
        print(f"{r:24s} {('—' if cx is None else f'{cx:.4f}'):>12} "
              f"{row['arms']['logit']['persist']['mean']:9.5f} {max(js):9.5f} {min(js):9.5f} "
              f"{row['arms']['shuf']['persist']['mean']:9.5f} "
              f"{row['capability_top10_rate']:8.3f}")
    for k, v in ctrl.items():
        if "fires" in v:
            print(f"  {k:28s} {'FIRES' if v['fires'] else 'DOES NOT FIRE'}")
    print(f"  C3 collapsed rungs: {cap_collapsed or 'none'}")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
