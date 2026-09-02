#!/usr/bin/env python3
"""t35_containment.py — E35 / M1: n-gram containment of candidate Q corpora against Pythia's
actual training stream, by INVERTING the query.

THE IDEA
  Containment asks "what fraction of Q's k-grams appear in the training stream?"  The naive
  implementation indexes the 300-billion-token side (601 GB, 20 shards).  But Q is seven orders of
  magnitude smaller:

      Pythia deduped stream   300B tokens  -> ~3e11 k-grams   601 GB
      all candidate Q corpora ~6M tokens   -> ~6e6  k-grams   ~50 MB

  So index Q, not the Pile.  Build a Bloom filter over Q's k-grams, stream the Pile ONCE, probe
  every position, and verify the rare Bloom hits exactly against a sorted array.  Peak disk is one
  30 GB shard, deleted after each pass.  Storage cost ~ 0.

  Two things become free in this direction:
    * more Q's cost nothing   -- one filter, tagged by q-id; one stream answers every candidate
    * the k-sweep costs nothing -- parallel filters for every k in the same pass

SECOND ARTIFACT, same pass
  Every stream k-gram with (hash % --sketch-mod == 0) is written to a sorted uint64 Parquet.
  At mod=100 that is ~3e9 rows (~13 GB Parquet) and is a reusable APPROXIMATE containment oracle
  for every FUTURE Q, queryable with DuckDB and no further download.  Hash-modulo sampling keeps
  the same k-grams on both sides, so the sampled containment is unbiased.

CORRECTNESS GATES (rule 0b -- assert on observed state, never on "it ran")
  G1 tokenizer identity: our corpora must tokenise BIT-IDENTICALLY to the stream's 20B tokenizer,
     or every k-gram comparison is meaningless.  Checked against utils/20B_tokenizer.json.
  G2 uint16: the stream is uint16; the tokenizer's vocab must fit.
  G3 --smoke plants known documents in a synthetic stream so M1-C1 and M1-C2 fire with NO download.

PRE-REGISTRATION: docs/experiments/preregs/superseded/PREREG_E36_QLADDER.md, section "E35".  k is selected by a rule
fixed in advance -- the largest k in the sweep whose containment range across candidates is >= 0.3
-- because a small k saturates at 1.0, which is exactly how E28's classifier-AUC manipulation
check failed (1.0000 for every pair, could not rank).

    python t35_containment.py --smoke                       # tiny-first, no download, ~1 min
    python t35_containment.py --shards 1 --data /path/document
    python t35_containment.py --shards 20 --data /path/document --sketch-out sketch.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

BASE = np.uint64(1000003)          # rolling-hash base
KS = (8, 13, 20, 32)               # the pre-registered k sweep
WINDOW = 128                       # match the fitting window: the operator never saw a 129th token


# ---------------------------------------------------------------- hashing
def kgram_hashes(tokens: np.ndarray, k: int) -> np.ndarray:
    """64-bit rolling hashes of every k-gram, as k multiply-add passes (vectorised).

    Wraparound uint64 arithmetic is the hash. Collisions across 2e10 items in a 2^64 space are
    ~1e-8 by the birthday bound, which is far below every other error term here.
    """
    n = len(tokens)
    if n < k:
        return np.empty(0, dtype=np.uint64)
    out = np.zeros(n - k + 1, dtype=np.uint64)
    for j in range(k):
        out *= BASE
        out += tokens[j:j + n - k + 1].astype(np.uint64)
    return out


class Bloom:
    """Packed-bit Bloom over Q's k-grams. Two hashes by default: a deliberate space/speed trade --
    more bits per element and fewer probes is faster than the space-optimal setting, and the
    resulting ~1e-3 false-positive rate costs only an exact re-check on the hits."""

    def __init__(self, n_items: int, bits_per_item: int = 64, n_hash: int = 2):
        self.m = max(1 << 10, int(n_items * bits_per_item))
        self.n_hash = n_hash
        self.bits = np.zeros((self.m + 7) // 8, dtype=np.uint8)

    def _idx(self, h: np.ndarray, i: int) -> np.ndarray:
        # double hashing: h1 + i*h2, mixed so the two probes decorrelate
        h2 = (h ^ (h >> np.uint64(29))) * np.uint64(0x9E3779B97F4A7C15)
        return ((h + np.uint64(i) * h2) % np.uint64(self.m)).astype(np.int64)

    def add(self, h: np.ndarray) -> None:
        for i in range(self.n_hash):
            idx = self._idx(h, i)
            np.bitwise_or.at(self.bits, idx >> 3, (1 << (idx & 7)).astype(np.uint8))

    def maybe(self, h: np.ndarray) -> np.ndarray:
        keep = np.ones(len(h), dtype=bool)
        for i in range(self.n_hash):
            idx = self._idx(h, i)
            keep &= ((self.bits[idx >> 3] >> (idx & 7).astype(np.uint8)) & 1).astype(bool)
            if not keep.any():
                break
        return keep


# ---------------------------------------------------------------- gates
def gate_tokenizer(tok, raw_path: str) -> dict:
    """G1/G2. The stream was tokenised with utils/20B_tokenizer.json. If our corpora tokenise
    differently, every comparison below is meaningless."""
    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(raw_path)
    probes = ["def foo(x):\n    return x+1",
              "The Treaty of Versailles was signed in 1919.",
              "SELECT * FROM users WHERE id = 42;",
              "中文测试 — emoji \U0001F600"]
    mismatches = [p for p in probes if tok(p).input_ids != raw.encode(p).ids]
    vocab = raw.get_vocab_size()
    g = {"G1_tokenizer_identical": not mismatches, "G1_mismatched_probes": mismatches,
         "G2_vocab_fits_uint16": vocab < 65536, "vocab_size": vocab}
    if mismatches or vocab >= 65536:
        raise SystemExit(f"ABORT: tokenizer gate failed -> {g}")
    return g


# ---------------------------------------------------------------- Q side
def load_q_corpora(paths, tok, max_docs=None):
    q = {}
    for p in paths:
        name = os.path.basename(p).replace(".jsonl", "")
        toks = []
        for i, line in enumerate(open(p)):
            if max_docs and i >= max_docs:
                break
            t = tok(json.loads(line)["text"]).input_ids[:WINDOW]
            if len(t) >= WINDOW:
                toks.append(np.array(t, dtype=np.uint16))
        q[name] = toks
        print(f"  Q[{name}] {len(toks)} docs x {WINDOW} tokens", flush=True)
    return q


def build_index(q_corpora):
    """{k: (sorted unique hashes, owner-bitmask per hash, {qname: n_distinct})}"""
    idx = {}
    for k in KS:
        per_q, all_h, all_o = {}, [], []
        for bit, (name, docs) in enumerate(q_corpora.items()):
            hs = [kgram_hashes(d, k) for d in docs]
            hs = np.unique(np.concatenate(hs)) if hs else np.empty(0, np.uint64)
            per_q[name] = len(hs)
            all_h.append(hs)
            all_o.append(np.full(len(hs), 1 << bit, dtype=np.uint32))
        H = np.concatenate(all_h) if all_h else np.empty(0, np.uint64)
        O = np.concatenate(all_o) if all_o else np.empty(0, np.uint32)
        order = np.argsort(H, kind="stable")
        H, O = H[order], O[order]
        # merge owners of identical hashes (a k-gram shared by two Q's)
        uniq, inv = np.unique(H, return_inverse=True)
        owners = np.zeros(len(uniq), dtype=np.uint32)
        np.bitwise_or.at(owners, inv, O)
        idx[k] = (uniq, owners, per_q)
        print(f"  k={k:<3d} {len(uniq):>10,} distinct Q k-grams", flush=True)
    return idx


# ---------------------------------------------------------------- stream
def stream_tokens(paths, smoke_planted=None, chunk=50_000_000, sl=(0, 1)):
    """Yield uint16 chunks of the token stream. --smoke synthesises one instead.

    The shard .bin files are RAW uint16 with no header -- MMapIndexedDataset only uses the .idx
    for document boundaries, and containment does not need them: a k-gram spanning a document
    boundary is a negligible and uniform source of extra matches. So we memory-map the .bin
    directly, which avoids the 1.76 GB .idx download and the unshard step entirely.
    """
    if smoke_planted is not None:
        yield smoke_planted
        return
    i, n = sl
    ov = max(KS) - 1               # overlap so k-grams spanning a boundary are not lost;
                                   # double-counting is harmless because `found` is a boolean OR
    for p in paths:
        arr = np.memmap(p, dtype=np.uint16, mode="r")
        lo = len(arr) * i // n
        hi = len(arr) * (i + 1) // n
        print(f"  shard {os.path.basename(p)} slice {i}/{n}: "
              f"tokens [{lo:,}, {hi:,}) of {len(arr):,}", flush=True)
        for st in range(lo, hi, chunk):
            yield np.asarray(arr[max(0, st - ov):min(st + chunk, hi)])
        del arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="synthetic stream, no download")
    ap.add_argument("--bins", nargs="*", default=None,
                    help="raw .bin shard files (uint16, no header)")
    ap.add_argument("--corpora", default=os.path.join(HERE, "..", "corpora"))
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--sketch-mod", type=int, default=100)
    ap.add_argument("--sketch-out", default=None)
    ap.add_argument("--chunk", type=int, default=50_000_000)
    ap.add_argument("--slice", default="0/1", help="i/N — process only slice i of each shard")
    ap.add_argument("--found-out", default=None, help="save this worker's found bitmaps (.npz)")
    ap.add_argument("--merge", default=None, help="glob of found-bitmap .npz to OR together")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "e35_containment.json"))
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
    gates = gate_tokenizer(tok, os.path.join(HERE, "..", "thirdparty", "pythia", "utils",
                                             "20B_tokenizer.json"))
    print(f"G1/G2 tokenizer gate: PASS (vocab {gates['vocab_size']})", flush=True)

    paths = sorted(glob.glob(os.path.join(a.corpora, "*.jsonl")))
    q_corpora = load_q_corpora(paths, tok, a.max_docs)

    # M1-C2 control: random-token documents must score containment ~ 0
    rng = np.random.default_rng(0)
    q_corpora["_CONTROL_random"] = [rng.integers(0, gates["vocab_size"], WINDOW, dtype=np.uint16)
                                    for _ in range(200)]
    print(f"  Q[_CONTROL_random] 200 docs (M1-C2)", flush=True)

    idx = build_index(q_corpora)
    names = list(q_corpora)

    blooms, found = {}, {}
    for k in KS:
        uniq, owners, _ = idx[k]
        b = Bloom(len(uniq))
        b.add(uniq)
        blooms[k] = b
        found[k] = np.zeros(len(uniq), dtype=bool)
        print(f"  k={k:<3d} bloom {b.bits.nbytes/1e6:.1f} MB", flush=True)

    # ---- merge mode: OR every worker's bitmap, then score. No streaming.
    if a.merge:
        import glob as _g
        files = sorted(_g.glob(a.merge))
        if not files:
            raise SystemExit(f"ABORT: --merge matched no files: {a.merge}")
        for f in files:
            z = np.load(f)
            for k in KS:
                found[k] |= z[str(k)]
        print(f"merged {len(files)} worker bitmaps", flush=True)
        n_tok = n_probe = 0
        planted = None
    else:
        planted = None
    if a.smoke:
        # G3: a synthetic stream that CONTAINS the first 50 docs of the first real corpus
        # verbatim, embedded in random tokens. M1-C1 must then score high on that corpus and
        # M1-C2 must score ~0 on the random control.
        real = q_corpora[names[0]][:50]
        parts = []
        for d in real:
            parts.append(rng.integers(0, gates["vocab_size"], 500, dtype=np.uint16))
            parts.append(d)
        parts.append(rng.integers(0, gates["vocab_size"], 500, dtype=np.uint16))
        planted = np.concatenate(parts)
        print(f"\nSMOKE: synthetic stream {len(planted):,} tokens, "
              f"{len(real)} docs of '{names[0]}' planted verbatim", flush=True)
    elif not a.bins and not a.merge:
        raise SystemExit("ABORT: --bins is required unless --smoke or --merge")

    sl = tuple(int(x) for x in a.slice.split("/"))
    sketch, t0 = [], time.time()
    if not a.merge:
        n_tok = n_probe = 0
    for chunk in ([] if a.merge else stream_tokens(a.bins or [], planted, a.chunk, sl)):
        n_tok += len(chunk)
        for k in KS:
            h = kgram_hashes(chunk, k)
            if not len(h):
                continue
            n_probe += len(h)
            cand = blooms[k].maybe(h)
            if cand.any():
                hits = np.unique(h[cand])
                uniq = idx[k][0]
                pos = np.searchsorted(uniq, hits)
                pos = pos[pos < len(uniq)]
                exact = pos[uniq[pos] == hits[:len(pos)]] if len(pos) else pos
                found[k][exact] = True
            if a.sketch_out and k == 13:
                sketch.append(h[(h % np.uint64(a.sketch_mod)) == 0])
        print(f"  {n_tok/1e6:9.1f}M tokens  {time.time()-t0:6.0f}s", flush=True)

    if a.found_out:
        os.makedirs(os.path.dirname(a.found_out) or ".", exist_ok=True)
        np.savez_compressed(a.found_out, **{str(k): found[k] for k in KS})
        print(f"wrote worker bitmap {a.found_out}", flush=True)

    # ---- containment per (Q, k)
    res = {}
    for k in KS:
        uniq, owners, per_q = idx[k]
        res[str(k)] = {}
        for bit, name in enumerate(names):
            mask = (owners & np.uint32(1 << bit)) != 0
            tot = int(mask.sum())
            hit = int((mask & found[k]).sum())
            res[str(k)][name] = {"containment": hit / tot if tot else None,
                                 "n_distinct_kgrams": tot, "n_found": hit}

    # ---- pre-registered k selection: largest k whose range across REAL Q's is >= 0.30
    real_names = [n for n in names if not n.startswith("_CONTROL")]
    ranges = {}
    for k in KS:
        v = [res[str(k)][n]["containment"] for n in real_names]
        ranges[k] = max(v) - min(v)
    eligible = [k for k in KS if ranges[k] >= 0.30]
    k_sel = max(eligible) if eligible else None

    ctrl = {
        "M1_C2_random_containment": {str(k): res[str(k)]["_CONTROL_random"]["containment"]
                                     for k in KS},
        "M1_C2_fires": all((res[str(k)]["_CONTROL_random"]["containment"] or 0) < 0.01 for k in KS),
    }
    if a.smoke:
        planted_name = names[0]
        ctrl["M1_C1_planted_corpus"] = planted_name
        ctrl["M1_C1_containment"] = {str(k): res[str(k)][planted_name]["containment"] for k in KS}
        ctrl["M1_C1_fires"] = all(res[str(k)][planted_name]["containment"] > 0.1 for k in KS)

    rec = {"experiment": "E35 / M1 — n-gram containment against the Pythia stream (inverted query)",
           "prereg": "docs/experiments/preregs/superseded/PREREG_E36_QLADDER.md, section E35",
           "mode": "SMOKE (synthetic stream)" if a.smoke else "STREAM",
           "gates": gates, "k_sweep": list(KS), "window": WINDOW,
           "tokens_streamed": int(n_tok), "kgrams_probed": int(n_probe),
           "runtime_s": round(time.time() - t0, 1),
           "containment": res, "k_ranges": {str(k): ranges[k] for k in KS},
           "k_selected": k_sel,
           "k_selection_rule": "largest k in the sweep whose containment range across real Q "
                               "corpora is >= 0.30; fixed before running",
           "controls": ctrl}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)

    if a.sketch_out and sketch:
        import pyarrow as pa, pyarrow.parquet as pq
        s = np.unique(np.concatenate(sketch))
        pq.write_table(pa.table({"h": pa.array(s, type=pa.uint64())}), a.sketch_out,
                       compression="zstd")
        print(f"sketch: {len(s):,} rows -> {a.sketch_out} "
              f"({os.path.getsize(a.sketch_out)/1e9:.2f} GB)")

    print(f"\n{'corpus':26s}" + "".join(f"{'k='+str(k):>10}" for k in KS))
    for n in names:
        print(f"{n:26s}" + "".join(f"{res[str(k)][n]['containment']:10.4f}" for k in KS))
    print(f"\n{'range (real Q only)':26s}" + "".join(f"{ranges[k]:10.4f}" for k in KS))
    print(f"\nk selected: {k_sel}   (rule: largest k with range >= 0.30)")
    print(f"M1-C2 random control fires: {ctrl['M1_C2_fires']}  "
          f"{ctrl['M1_C2_random_containment']}")
    if a.smoke:
        print(f"M1-C1 planted control fires: {ctrl['M1_C1_fires']}  {ctrl['M1_C1_containment']}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
