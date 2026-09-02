#!/usr/bin/env python3
"""anchor_evals.py — the anchor's own §A.6 evaluation suite, implemented to its published spec.

DATA PROVENANCE. The six evaluation sets and the intervention prompt sets ship in the anchor's
companion repo, `anthropics/jacobian-lens` @ 581d398, **Apache-2.0**, at `data/evaluations/` and
`data/experiments/`. They are read from the local vendored checkout; nothing is downloaded.

  evaluations/lens-eval-{multihop,multilingual,order-ops,poetry,typo,association}.json
  experiments/{probe-swap,flexible-generalization,verbal-report,...}.json

WHY THIS EXISTS. Our hand-rolled country->capital battery reinvented the anchor's
`flexible-generalization` set — the EASY one, whose intermediate is the prompt subject and whose
answer is directly associated with it. Nanda's review of the anchor names exactly that design as
confounded ("pairs like France and Paris, which were linearly related"). The anchor's `multihop`
and `probe-swap` sets are constructed so the trigger and the answer rarely co-occur without the
intermediate, which is the design that defuses the critique. Use those.

METRIC, verbatim from `data/evaluations/README.md`:

    pass@k = mean over ITEMS of the FRACTION of `intermediates` whose min-over-layers lens rank <= k

Note this is a mean of per-item fractions, NOT a flat pool over all (item, intermediate) pairs.
The two differ whenever items carry different numbers of intermediates — multilingual items have
four, association items have one.

READOUT POSITION, per set (README):
  multihop / multilingual / order-ops : the token immediately preceding `target`
                                        == the final prompt token, after rstrip
  poetry                              : the last newline token (end of line 1 of the couplet)
  typo / association                  : the final prompt token

`.rstrip()` IS LOAD-BEARING: the released prompts end in a trailing space (e.g. multihop's
"... is the "), and an un-stripped trailing space becomes the readout token and destroys the
baseline.

TRANSPORT ABSTRACTION. The three lenses are one operator over a different transport
`v_t = T_l^T W_U[t]` / readout `unembed(T_l h)`:
    logit lens  T = I          (no fitting)
    tuned lens  T = A_l        (learned affine; linear part)
    J-lens      T = J_l        (averaged Jacobian)
`transport_*` below return callables `(h, layer) -> transported h` so a single readout path
serves all three.

NOTE ON THE PAPER-LITERAL DIRECTION. The anchor defines v_t as the rows of `W_U J_l`, i.e.
`J_l^T W_U[t]`, with no final-norm gain folded in, while the readout path is
`lm_head(final_norm(J h))`. Folding the gain (`J_l^T diag(g) W_U[t]`) is arguably more faithful
to the readout. Measured on our lenses the two differ negligibly in direction
(cos = 0.9999 at pythia-70m, 0.9978 at pythia-410m), so we keep the paper-literal form and
record the check. See `fold_norm_gain=` to switch.
"""
from __future__ import annotations

import json
import math
import os
from typing import Callable

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
ANCHOR_DATA = os.path.join(_HERE, "..", "jacobian-lens", "data")

EVAL_SETS = ["multihop", "multilingual", "order-ops", "poetry", "typo", "association"]

# Readout position rule per set (README). "last" == final prompt token after rstrip.
_POSITION_RULE = {
    "multihop": "last", "multilingual": "last", "order-ops": "last",
    "typo": "last", "association": "last", "poetry": "last_newline",
}

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_OPS = {
    "multiplication": ["*", "×", "times", "multiply", "multiplication", "product"],
    "addition": ["+", "plus", "add", "addition", "sum"],
    "subtraction": ["-", "−", "minus", "subtract", "subtraction", "difference"],
    "division": ["/", "÷", "divided", "divide", "division", "quotient"],
}


# --------------------------------------------------------------------------- data

def load_eval(name: str) -> list[dict]:
    """Load one of the six §A.6 evaluation sets from the vendored anchor repo."""
    if name not in EVAL_SETS:
        raise ValueError(f"unknown eval set {name!r}; expected one of {EVAL_SETS}")
    path = os.path.join(ANCHOR_DATA, "evaluations", f"lens-eval-{name}.json")
    with open(path) as f:
        return json.load(f)["items"]


def load_experiment(name: str) -> dict:
    """Load one of the intervention prompt sets (e.g. 'probe-swap')."""
    path = os.path.join(ANCHOR_DATA, "experiments", f"{name}.json")
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- tokens

def _number_word(s: str) -> str | None:
    n = int(s)
    if n < 20:
        return _ONES[n]
    if n < 100 and n % 10 == 0:
        return _TENS[n // 10 - 2]
    return None  # compounds like "twenty-one" are multi-token anyway


def synonyms(word: str) -> list[str]:
    """Surface variants of an intermediate. order-ops expands digits to number words and
    operation names to symbol/word forms, per the README's synonym-set rule."""
    out = [word]
    if word.isdigit():
        w = _number_word(word)
        if w:
            out.append(w)
    out += _OPS.get(word.lower(), [])
    return out


def token_ids_of(tok, word: str) -> list[int]:
    """Single-token ids for a word across surface forms (leading space, casing).

    Multi-token surface forms are dropped: the J-lens has one vector per vocabulary token, so a
    concept with no single-token form cannot be scored (the anchor's own §9.1 limitation).
    """
    ids: set[int] = set()
    for w in synonyms(word):
        for form in {w, w.lower(), w.capitalize(), " " + w, " " + w.lower(), " " + w.capitalize()}:
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
    return sorted(ids)


def rank_of(logits: torch.Tensor, ids: list[int]) -> int | None:
    """Best 1-indexed rank of any id in `ids` within a [vocab] logit vector."""
    if not ids:
        return None
    best = logits[ids].max()
    return int((logits > best).sum().item()) + 1


def readout_position(tok, eval_name: str, prompt: str) -> int:
    """Position index to read at, per the anchor's per-set convention."""
    rule = _POSITION_RULE[eval_name]
    if rule == "last":
        return -1
    ids = tok(prompt, add_special_tokens=True).input_ids
    newlines = [i for i, t in enumerate(ids) if "\n" in tok.decode([t])]
    return newlines[-1] if newlines else -1


# --------------------------------------------------------------------------- transports

def transport_jlens(lens, fold_norm_gain: bool = False, model=None) -> Callable:
    """T = J_l. `fold_norm_gain` left-multiplies by the final-norm diagonal gain."""
    gain = None
    if fold_norm_gain:
        if model is None:
            raise ValueError("fold_norm_gain requires model=")
        gain = model._final_norm.weight.detach().float()

    def T(h: torch.Tensor, layer: int) -> torch.Tensor:
        J = lens.jacobians[layer].to(h.device).float()
        out = h.float() @ J.T
        return out * gain.to(h.device) if gain is not None else out
    return T


def transport_logit() -> Callable:
    """T = I. The logit lens: no fitting, no corpus, no distribution dependence."""
    def T(h: torch.Tensor, layer: int) -> torch.Tensor:
        return h.float()
    return T


def transport_tuned(translators: dict, include_bias: bool = True) -> Callable:
    """T = A_l (+ b_l). `translators[layer] = (A, b)` with A:[d,d], b:[d].

    The bias is why a tuned lens is affine rather than linear. The anchor reports that the tuned
    lens's early-layer advantage over the logit lens comes almost entirely from `b`, so
    `include_bias=False` isolates the linear part and is the apples-to-apples transport for a
    `v_t = T^T W_U[t]` comparison. Report both.
    """
    def T(h: torch.Tensor, layer: int) -> torch.Tensor:
        A, b = translators[layer]
        out = h.float() @ A.to(h.device).float().T
        return out + b.to(h.device).float() if include_bias else out
    return T


# --------------------------------------------------------------------------- readout

@torch.no_grad()
def readout(model, transport: Callable, prompt: str, layers, position: int,
            max_seq_len: int = 256) -> dict[int, torch.Tensor]:
    """{layer: [vocab] logits} at one position, through the given transport."""
    from jlens.hooks import ActivationRecorder
    ids = model.encode(prompt, max_length=max_seq_len)
    with ActivationRecorder(model.layers, at=list(layers)) as rec:
        model.forward(ids)
        acts = {l: rec.activations[l][0].detach() for l in layers}
    return {l: model.unembed(transport(acts[l][position].float(), l)).float().cpu()
            for l in layers}


# --------------------------------------------------------------------------- metric

@torch.no_grad()
def pass_at_k(model, transport: Callable, eval_name: str, layers,
              ks=(1, 2, 5, 10, 25, 100), max_items: int | None = None,
              max_seq_len: int = 256) -> dict:
    """pass@k per the anchor README, plus the normalized AUC over log k.

    pass@k = mean over ITEMS of the FRACTION of that item's intermediates whose
             min-over-layers (and min-over-synonyms) rank is <= k.
    """
    items = load_eval(eval_name)
    if max_items:
        items = items[:max_items]
    per_item = {k: [] for k in ks}
    n_unscorable = 0
    ranks_all = []
    for item in items:
        prompt = item["prompt"].rstrip()          # load-bearing; see module docstring
        pos = readout_position(model.tokenizer, eval_name, prompt)
        out = readout(model, transport, prompt, layers, pos, max_seq_len)
        best_ranks = []
        for word in item["intermediates"]:
            ids = token_ids_of(model.tokenizer, word)
            if not ids:
                n_unscorable += 1
                continue
            r = min(rank_of(out[l], ids) for l in layers)
            best_ranks.append(r)
            ranks_all.append(r)
        if not best_ranks:
            continue
        for k in ks:
            per_item[k].append(sum(r <= k for r in best_ranks) / len(best_ranks))

    xs = [math.log(k) for k in ks]

    def _auc(curve_ys: list[float]) -> float:
        a = sum((curve_ys[i] + curve_ys[i + 1]) / 2 * (xs[i + 1] - xs[i])
                for i in range(len(ks) - 1))
        return a / (xs[-1] - xs[0])

    # ---- form B: the shipped data/evaluations/README.md metric (mean of per-item fractions)
    curve = {k: (sum(v) / len(v) if v else float("nan")) for k, v in per_item.items()}
    auc = _auc([curve[k] for k in ks])

    # ---- form A: the PAPER's §A.6 metric (per-item BINARY over the min-over-layers rank)
    #
    #     recovered_k(i) = 1[ exists l : rank_l(v_i) <= k ]
    #     pass@k         = (1/N) sum_i recovered_k(i)
    #
    # §A.6 specifies ONE intermediate per item, in which case item-level and
    # intermediate-level are the same set. Four of the six RELEASED sets carry several
    # intermediates per item (multilingual 4, order-ops 2), so the formula has to be
    # generalized. We take the direct generalization: the indicator is applied per
    # (item, intermediate) pair — a flat pool — which reduces EXACTLY to the paper's formula
    # when every item has one intermediate, and does not silently reweight items by how many
    # intermediates they happen to carry.
    #
    # INTERPRETIVE CHOICE, flagged rather than buried. The alternative generalizations are
    # "item counts if ANY intermediate is recovered" (more permissive) and "...if ALL are"
    # (stricter). Both are defensible; neither reduces to the paper's formula more naturally
    # than the flat pool. Registered in docs/experiments/preregs/superseded/PREREG_PYTHIA_T7_v2.md §3.
    curve_paper = {k: (sum(r <= k for r in ranks_all) / len(ranks_all) if ranks_all
                       else float("nan")) for k in ks}
    auc_paper = _auc([curve_paper[k] for k in ks])

    med = float(torch.tensor(ranks_all, dtype=torch.float).median()) if ranks_all else float("nan")
    return {
        "eval": eval_name, "n_items": len(items), "n_items_scored": len(per_item[ks[0]]),
        "n_intermediate_pairs_scored": len(ranks_all),
        "n_intermediates_unscorable_multitoken": n_unscorable,
        "layers": list(layers), "ks": list(ks),
        # PRIMARY (paper §A.6)
        "pass_at_k_paper": {str(k): round(curve_paper[k], 4) for k in ks},
        "normalized_auc_over_logk_paper": round(auc_paper, 4),
        # SECONDARY (shipped README; every pre-2026-08-10 T7 number in results/ uses this)
        "pass_at_k": {str(k): round(curve[k], 4) for k in ks},
        "normalized_auc_over_logk": round(auc, 4),
        "metric_note": ("paper = per-(item,intermediate) binary on min-over-layers rank; "
                        "readme = mean over items of the fraction of that item's "
                        "intermediates recovered. Identical when items have 1 intermediate."),
        "median_best_rank": med,
    }
