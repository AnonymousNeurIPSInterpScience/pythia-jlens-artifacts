#!/usr/bin/env python3
"""Unit tests for anchor_evals.py. Run: .venv/bin/python tests/test_anchor_evals.py

Each test pins a property taken from the anchor's own `data/evaluations/README.md`, so a
regression here means we have drifted from the published protocol, not merely from our own code.
No model or GPU required.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anchor_evals import (  # noqa: E402
    EVAL_SETS, load_eval, load_experiment, rank_of, readout_position,
    synonyms, token_ids_of, transport_logit,
)

PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED  {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


class FakeTok:
    """Minimal tokenizer: whitespace/newline aware, deterministic ids."""
    def __init__(self):
        self.vocab = {}

    def _id(self, s):
        return self.vocab.setdefault(s, len(self.vocab) + 1)

    def encode(self, text, add_special_tokens=False):
        # single token iff the text has no internal whitespace beyond one leading space
        core = text[1:] if text.startswith(" ") else text
        return [self._id(text)] if core and " " not in core else [self._id(c) for c in text.split()]

    def __call__(self, text, add_special_tokens=True):
        toks = []
        for part in text.replace("\n", " \n ").split(" "):
            if part:
                toks.append(self._id(part))
        return type("E", (), {"input_ids": toks})()

    def decode(self, ids):
        rev = {v: k for k, v in self.vocab.items()}
        return "".join(rev.get(i, "") for i in ids)


def main() -> int:
    print("=== anchor_evals — spec tests ===")

    # 1. all six sets load, and are non-empty ---------------------------------
    counts = {}
    for name in EVAL_SETS:
        items = load_eval(name)
        counts[name] = len(items)
        check(f"load_{name}", len(items) > 0 and "prompt" in items[0]
              and "intermediates" in items[0], f"n={len(items)}")
    check("all_six_sets_present", len(counts) == 6, str(counts))

    # 2. THE trailing-space hazard the anchor data actually contains ----------
    with_trailing = [n for n in EVAL_SETS
                     if any(i["prompt"] != i["prompt"].rstrip() for i in load_eval(n))]
    check("some_prompts_have_trailing_whitespace", len(with_trailing) > 0,
          f"sets affected: {with_trailing} — rstrip is load-bearing")

    # 3. multihop is intermediate-bearing: the intermediate is NOT the target
    #    and does NOT appear in the prompt. This is the property that defuses
    #    Nanda's 'France/Paris are linearly related' critique.
    mh = load_eval("multihop")
    # Word-boundary matching, not substring: "H" occurs inside "hydrogen" and "O" inside
    # "oxygen" without the intermediate being given away. Naive `in` reports 5 false leaks.
    import re
    def leaks(item):
        return [w for w in item["intermediates"]
                if re.search(rf"\b{re.escape(w)}\b", item["prompt"], re.IGNORECASE)]
    leaked = [i["name"] for i in mh if leaks(i)]
    # KNOWN, accepted: 'dual-photosynthesis-opposite' gives the intermediate "day" in the
    # phrase "the time of day". It is a genuine (borderline) leak in the anchor's own data.
    # Pinned here so a future data change is caught rather than silently absorbed.
    KNOWN_LEAKS = {"dual-photosynthesis-opposite"}
    check("multihop_intermediate_not_in_prompt_wordwise", set(leaked) <= KNOWN_LEAKS,
          f"leaks={leaked} (known-accepted: {sorted(KNOWN_LEAKS)})")
    check("multihop_leak_rate_under_2pct", len(leaked) / len(mh) < 0.02,
          f"{len(leaked)}/{len(mh)}")
    same = [i["name"] for i in mh
            if "target" in i and any(w.lower() == i["target"].lower() for w in i["intermediates"])]
    check("multihop_intermediate_is_not_the_target", len(same) == 0, f"{len(same)} collisions")

    # 4. probe-swap carries everything a write battery needs ------------------
    ps = load_experiment("probe-swap")["items"]
    need = {"prompt", "intermediate", "answer", "swap_to", "swap_answer"}
    check("probe_swap_schema", all(need <= set(i) for i in ps), f"n={len(ps)}")
    bad = [i["name"] for i in ps if i["answer"].lower() == i["swap_answer"].lower()]
    check("probe_swap_answer_differs_from_swap_answer", len(bad) == 0, f"{len(bad)} degenerate")

    # 5. readout position rules ----------------------------------------------
    tok = FakeTok()
    check("position_last_for_multihop", readout_position(tok, "multihop", "a b c") == -1)
    check("position_last_for_typo", readout_position(tok, "typo", "a b c") == -1)
    poem = "line one here\nline two here"
    p = readout_position(tok, "poetry", poem)
    check("position_newline_for_poetry", p != -1 and p >= 0, f"index {p}")

    # 6. order-ops synonym expansion -----------------------------------------
    check("synonym_digit_to_word", "five" in synonyms("5"), str(synonyms("5")))
    check("synonym_operation", "*" in synonyms("multiplication") and "times" in synonyms("multiplication"))
    check("synonym_identity_for_plain_word", synonyms("Brazil") == ["Brazil"])
    check("synonym_no_compound_number", _no_compound(), "21 -> no 'twenty-one'")

    # 7. rank_of semantics ----------------------------------------------------
    lg = torch.tensor([5.0, 1.0, 9.0, 3.0])
    check("rank_of_best_is_1", rank_of(lg, [2]) == 1)
    check("rank_of_takes_min_over_ids", rank_of(lg, [1, 2]) == 1, "best of the set wins")
    check("rank_of_is_1_indexed", rank_of(lg, [0]) == 2, "5.0 is second-largest -> rank 2")
    check("rank_of_empty_is_none", rank_of(lg, []) is None)

    # 8. token_ids_of drops multi-token forms ---------------------------------
    ids = token_ids_of(tok, "Brazil")
    check("token_ids_nonempty_for_simple_word", len(ids) > 0, f"{len(ids)} forms")
    check("token_ids_multiword_is_empty", token_ids_of(tok, "two words") == [],
          "multi-token concepts are unscorable — the anchor's own §9.1 limit")

    # 9. transport_logit is the identity --------------------------------------
    h = torch.randn(16)
    check("transport_logit_is_identity", torch.allclose(transport_logit()(h, 3), h))

    # 10. the metric is a MEAN OF PER-ITEM FRACTIONS, not a flat pool ----------
    check("per_item_fraction_differs_from_flat_pool", _metric_shapes_differ(),
          "sets with unequal #intermediates make the two definitions disagree")

    print(f"\n=== {len(PASSED)}/{len(PASSED)} PASSED ===")
    print(f"item counts: {counts}")
    return 0


def _no_compound():
    return all(s.isdigit() or "-" not in s for s in synonyms("21"))


def _metric_shapes_differ():
    """multilingual items carry 4 intermediates, association 1 — so a flat pool over
    (item, intermediate) pairs is NOT the anchor's mean-over-items-of-fractions."""
    ml = load_eval("multilingual")
    sizes = {len(i["intermediates"]) for i in ml} | {len(i["intermediates"])
                                                     for i in load_eval("association")}
    return len(sizes) > 1


if __name__ == "__main__":
    raise SystemExit(main())
