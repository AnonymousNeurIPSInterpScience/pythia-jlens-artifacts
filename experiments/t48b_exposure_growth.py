#!/usr/bin/env python3
"""t48b_exposure_growth.py — E48 PREREQUISITE: is the "genuinely OOD" designation actually earned?

WHY THIS EXISTS
  E48's pre-registration divides the rungs into "in-stream" and "genuinely OOD", and clause (b)
  is scored ONLY on the OOD three. The evidence offered for that division is M1 containment, and
  the argument is explicitly about SHAPE, not level:

      "The discriminator is the GROWTH RATE, not the level: present corpora are found
       proportionally as coverage rises; absent ones stay pinned at zero."
      — PREREG_E48_CROSSOVER.md

  Two problems with that as it stood, both found while preparing to score E48:

  1. The numbers lived ONLY on a rented GPU box (`/workspace/m1_5shard.json`). The committed
     `results/e35_containment_shard0.json` contains the five Pile components and the random
     control and NO OOD corpus at all — so there was no stored 1-shard baseline for the OOD
     rungs, and therefore no stored growth RATE for exactly the corpora the rate is used to
     classify. Discipline #1 and #2 both. The file is now pulled and SHA-verified to
     results/m1/m1_5shard.json; this script supplies the missing curve.

  2. A growth rate needs more than two points to be called flat.

  M1's per-shard artifacts make the curve free: each `merged/shardNN.npz` is a BOOLEAN
  "was this Q k-gram found" bitmap over a fixed, deterministic k-gram index. Containment at any
  coverage is the OR of any subset of shards. No GPU, no re-streaming of the Pile, no refit.

WHAT IT MEASURES
  containment_c(P) = |{k-grams of P found in the first c shards}| / |distinct k-grams of P|
  for c = 1..C, every corpus, every k in {8,13,20,32}. Then the growth ratio
  containment_C / containment_1, which is the quantity the pre-registration's discriminator
  names and which no file currently stores.

CONTROLS
  C1 RANDOM FLOOR   `_CONTROL_random` (200 documents of uniform random token ids) must sit at
                    exactly 0.000 at every coverage and every k. If it grows, the index or the
                    bitmap alignment is wrong and no other row means anything.
  C2 INDEX MATCH    the rebuilt k-gram index must have the SAME length as every shard bitmap.
                    A mismatch means the corpora or tokenizer changed since the shards were
                    written, and the bitmaps would be silently misaligned.
  C3 REPRODUCTION   at c = (all shards present in m1_5shard.json) the recomputed containment
                    must reproduce that file's stored values. This proves the recomputation
                    path is the same one that produced the pre-registration's table.

DECLARED BIAS — READ BEFORE USING THIS TO CALL ANYTHING "ABSENT"
  * Containment is over TOKEN k-grams, so it is sensitive to FORMATTING, not just content.
    Measured on OOD_CommonPile by re-wrapping the identical text: at k=32, 28.8% of k-grams
    survive an 80-col re-wrap (its own wrap), 2.43% a 72-col, 1.32% a 100-col and 0.14% a
    single-line reflow. So a corpus whose text IS in the stream under a different line wrap can
    score as low as ~0.1-2%. Any containment in that band is UNINTERPRETABLE as absence.
  * M1 returns EVIDENCE, not a Boolean (metrics_shift.py's own rule 3). "Provably non-member"
    requires an exhaustive scan; a partial-coverage scan can only say "no match found".
  * Shards are a preshuffled stream, so shard order is arbitrary and c shards is an unbiased
    c/20 sample of the corpus. That is what licenses reading the curve as a coverage curve.

    python t48b_exposure_growth.py --shards "results/m1/merged/shard*.npz"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from provenance import write_result  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=os.path.join(HERE, "..", "results", "m1", "merged", "shard*.npz"))
    ap.add_argument("--corpora", default=os.path.join(HERE, "..", "corpora"))
    ap.add_argument("--exclude", default="OOD_News_2024_dedup",
                    help="comma list of corpora to leave OUT of the index. The index must be "
                         "rebuilt with EXACTLY the corpus set that existed when the shard "
                         "bitmaps were written, or every bitmap position points at the wrong "
                         "k-gram. OOD_News_2024_dedup was created after the shards and is a "
                         "strict subset of OOD_News_2024, so it adds no new k-grams and the "
                         "TOTAL index length is unchanged — which is exactly why a length-only "
                         "check cannot catch this.")
    ap.add_argument("--reference", default=os.path.join(HERE, "..", "results", "m1", "m1_5shard.json"),
                    help="stored containment to reproduce for control C3")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e48b_exposure_growth.json"))
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from t35_containment import KS, WINDOW, build_index, gate_tokenizer, load_q_corpora

    files = sorted(glob.glob(a.shards))
    if not files:
        raise SystemExit(f"ABORT: --shards matched no files: {a.shards}")
    print(f"{len(files)} shard bitmaps: {os.path.basename(files[0])} .. "
          f"{os.path.basename(files[-1])}", flush=True)

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
    # G1/G2 exactly as t35 runs it. `gates["vocab_size"]` is the RAW 20B tokenizer vocab
    # (50277), NOT `tok.vocab_size` (50254). The random control is drawn against that bound, so
    # using the wrong one draws DIFFERENT tokens, whose hashes sort into different slots, which
    # silently permutes the whole index. That bug produced a first run in which the random
    # control scored 0.457 — identical to every real corpus — and C1/C3 caught it.
    gates = gate_tokenizer(tok, os.path.join(HERE, "..", "thirdparty", "pythia", "utils",
                                             "20B_tokenizer.json"))
    print(f"G1/G2 tokenizer gate: PASS (vocab {gates['vocab_size']})", flush=True)

    excl = {x for x in a.exclude.split(",") if x}
    paths = [p for p in sorted(glob.glob(os.path.join(a.corpora, "*.jsonl")))
             if os.path.basename(p).replace(".jsonl", "") not in excl]
    print(f"index corpora ({len(paths)}), excluding {sorted(excl) or 'nothing'}", flush=True)
    q = load_q_corpora(paths, tok, None)
    rng = np.random.default_rng(0)
    # the index MUST be built exactly as t35 built it, including the random control and its
    # position in the dict — the owner bitmask is positional
    q["_CONTROL_random"] = [rng.integers(0, gates["vocab_size"], WINDOW, dtype=np.uint16)
                            for _ in range(200)]
    idx = build_index(q)
    names = list(q)

    # ---- C2: PER-CORPUS k-gram counts must match the stored reference, not just the total.
    # A length-only check is worthless here: OOD_News_2024_dedup is a strict subset of
    # OOD_News_2024, so including it leaves the total index length EXACTLY unchanged while
    # shifting every owner bit. Per-corpus counts catch that; the total does not.
    c2 = {"per_corpus_counts_vs_reference": {}, "length_vs_bitmaps": {}}
    ref_all = json.load(open(a.reference))["containment"] if os.path.exists(a.reference) else None
    ok = True
    for k in KS:
        uniq, _, per_q = idx[k]
        lens = {int(np.load(f)[str(k)].shape[0]) for f in files}
        c2["length_vs_bitmaps"][str(k)] = {"index_len": int(len(uniq)),
                                           "bitmap_lens": sorted(lens),
                                           "fires": lens == {len(uniq)}}
        ok &= (lens == {len(uniq)})
        if ref_all:
            bad = {n: (per_q[n], ref_all[str(k)][n]["n_distinct_kgrams"])
                   for n in per_q if n in ref_all[str(k)]
                   and per_q[n] != ref_all[str(k)][n]["n_distinct_kgrams"]}
            c2["per_corpus_counts_vs_reference"][str(k)] = {
                "n_checked": sum(1 for n in per_q if n in ref_all[str(k)]),
                "mismatches": bad, "fires": not bad}
            ok &= (not bad)
    if not ok:
        raise SystemExit(
            f"C2 FAILED — the rebuilt index does not match what produced the shard bitmaps:\n"
            f"{json.dumps(c2, indent=1)}\n"
            "Every bitmap position would point at a different k-gram and every number below "
            "would be silently wrong. Do NOT relax this check.")
    print("C2 INDEX MATCH: FIRES (per-corpus k-gram counts and total length both reproduce)",
          flush=True)

    # ---- the growth curve
    curve = {str(k): {n: [] for n in names} for k in KS}
    acc = {k: None for k in KS}
    for c, f in enumerate(files, start=1):
        z = np.load(f)
        for k in KS:
            acc[k] = z[str(k)].copy() if acc[k] is None else (acc[k] | z[str(k)])
        for k in KS:
            uniq, owners, per_q = idx[k]
            hit = acc[k]
            for bit, n in enumerate(names):
                mask = (owners & np.uint32(1 << bit)) != 0
                tot = int(mask.sum())
                curve[str(k)][n].append(float(hit[mask].sum() / tot) if tot else 0.0)
        print(f"  coverage {c}/{len(files)} shards done", flush=True)

    # ---- C1: the random control must be exactly zero everywhere
    c1_max = max(max(curve[str(k)]["_CONTROL_random"]) for k in KS)
    print(f"C1 RANDOM FLOOR: max over all k and coverages = {c1_max:.6f}  ->  "
          f"{'FIRES' if c1_max < 0.01 else 'DOES NOT FIRE'}", flush=True)
    if c1_max >= 0.01:
        raise SystemExit(
            f"C1 FAILED — 200 documents of UNIFORM RANDOM token ids score containment "
            f"{c1_max:.4f}. Random 32-token sequences are not in the Pile. The index and the "
            f"bitmaps are misaligned; nothing in this run means anything. ABORTING rather than "
            f"writing a results file.")

    # ---- C3: reproduce the stored 5-shard file at the matching coverage
    c3 = {"checked": False}
    if os.path.exists(a.reference):
        ref = json.load(open(a.reference))["containment"]
        n_ref = None
        # find the coverage whose values match best; the reference is the first N shards
        best = None
        for c in range(1, len(files) + 1):
            err = max(abs(curve[str(k)][n][c - 1] - ref[str(k)][n]["containment"])
                      for k in KS for n in ref[str(k)] if n in curve[str(k)])
            if best is None or err < best[1]:
                best = (c, err)
        n_ref, err = best
        c3 = {"checked": True, "matching_coverage_shards": n_ref, "max_abs_diff": err,
              "fires": err < 1e-6}
        print(f"C3 REPRODUCTION: best match at {n_ref} shards, max|diff| = {err:.3e}  ->  "
              f"{'FIRES' if c3['fires'] else 'DOES NOT FIRE'}", flush=True)
        if not c3["fires"]:
            raise SystemExit(
                f"C3 FAILED — this recomputation does not reproduce the stored containment in "
                f"{os.path.basename(a.reference)} at ANY coverage (best {err:.3e} at {n_ref} "
                f"shards). The recomputation path is not the one that produced the "
                f"pre-registration's table, so its growth curve cannot be used to adjudicate "
                f"rung eligibility. ABORTING rather than writing a results file.")

    K_MAIN = "32"
    growth = {n: (curve[K_MAIN][n][-1] / curve[K_MAIN][n][0]
                  if curve[K_MAIN][n][0] > 0 else float("inf") if curve[K_MAIN][n][-1] > 0
                  else float("nan")) for n in names}
    rec = {
        "experiment": "E48b — containment vs stream coverage: does the OOD designation earn itself?",
        "why": ("the E48 pre-registration classifies rungs by the GROWTH RATE of containment with "
                "coverage, and no stored file contained a growth rate for the OOD corpora at all "
                "— results/e35_containment_shard0.json has only the five Pile components"),
        "shards_used": [os.path.basename(f) for f in files],
        "n_shards_total_in_stream": 20,
        "k_values": [int(k) for k in KS], "window_tokens": WINDOW,
        "primary_k": int(K_MAIN),
        "controls": {"C1_random_floor": {"max_over_all": c1_max, "fires": c1_max == 0.0},
                     "C2_index_matches_bitmaps": c2,
                     "C3_reproduces_stored_5shard": c3},
        "containment_by_coverage": curve,
        "growth_ratio_first_to_last_k32": growth,
        "formatting_sensitivity_measured": {
            "note": ("fraction of OOD_CommonPile's own k=32 k-grams that survive re-wrapping the "
                     "IDENTICAL text — the band in which a containment value cannot be read as "
                     "absence"),
            "80col_its_own_wrap": 0.2877, "72col": 0.0243, "100col": 0.0132,
            "single_line_reflow": 0.0014},
        "reading": ("A corpus is 'not found' only if its containment is BELOW the formatting "
                    "floor AND flat in coverage. Containment inside 0.001-0.03 at k=32 is "
                    "consistent with present-but-reformatted and must not be called absence."),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write_result(a.out, rec, experiment="E48b", inputs=files + [a.reference])

    print(f"\nCONTAINMENT vs COVERAGE at k={K_MAIN}  ({len(files)}/20 shards)")
    print(f"{'corpus':24s}" + "".join(f"{c:>9d}" for c in range(1, len(files) + 1)) + f"{'growth':>9}")
    for n in sorted(names, key=lambda x: -curve[K_MAIN][x][-1]):
        print(f"{n:24s}" + "".join(f"{v:9.4f}" for v in curve[K_MAIN][n])
              + f"{growth[n]:9.2f}x")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
