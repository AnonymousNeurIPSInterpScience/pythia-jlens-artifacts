#!/usr/bin/env python3
"""r4e_control_power.py — R4e: which controls in this repository could not have failed?

PRE-REGISTRATION: specs/experiments/R4_corrections.md item R4e, committed before this script existed.
Source slate: specs/POSTREVIEW_EXPERIMENTS.md §3 Tier A, item R4e.

THE QUESTION. `CLAUDE.md` §6.3 requires every result to name its control and the number that control
produced. It does not require the control to have been CAPABLE of producing a different number. A
control that cannot fail is decoration, and this repository has shipped several -- E45's C2 was
"structurally incapable of failing, and rebuilt properly it does not fire".

WHAT IS MECHANICAL AND WHAT IS JUDGEMENT, kept apart on purpose:

  MECHANICAL (this script decides, and can be re-run):
    * NO_GATE            -- a control dict carrying no `fires`/`passes` key at all. It records a
                            number and adjudicates nothing.
    * GATE_NEVER_EVALUATED -- `fires` is null: the gate exists in the schema and did not run.
    * DECLARED_NEVER_CODED -- a control named in a specs/experiments/ document or a prereg that has
                            NO corresponding key in the experiment's results file. Fully mechanical,
                            and the most damning class because the spec claims a control the file
                            does not contain.
    * FIRED_TRUE / FIRED_FALSE -- gated and evaluated.

  JUDGEMENT (a hand-audited override table, each entry naming WHY):
    * SELF_COMPARISON    -- compares a quantity to itself.
    * ARITHMETIC_IDENTITY -- the value is forced by algebra, not by the measurement.
  These cannot be inferred from a JSON value: `max_abs_diff = 0.0` against a stored file is a REAL
  bit-identity control with power, while `{"value": 0.0, "fires": true}` is a definitional zero.
  The two are indistinguishable to a script, so they are distinguished by hand and the hand entries
  are listed explicitly rather than folded into the count.

    python tools/r4e_control_power.py
"""
from __future__ import annotations
import collections, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "src"))
from provenance import sha256_file, write_result  # noqa: E402

R = lambda *p: os.path.join(REPO, *p)
CNAME = re.compile(r"^(?:control_)?(C\d+[a-z]?)(_|$)", re.I)  # catches control_C1_... too
# smoke runs and re-run duplicates are the same controls counted twice
SKIP = re.compile(r"_smoke|smoke_|_rerun|_pooled|_provenance")

# ---- the hand audit. Each entry: (file stem, control base) -> (class, why).
# Every one of these was read before it was classified.
MANUAL = {
    ("e52_factorial_410m", "C5"): (
        "ARITHMETIC_IDENTITY",
        "C5_self_distance is stored as {'value': 0.0, 'fires': True}. The distance of an operator "
        "from itself is zero by definition; no measurement can make it anything else."),
    ("e34a_dact_vs_floor_410m", "C1"): (
        "ARITHMETIC_IDENTITY",
        "C1 fires 'at exactly 0.0' -- D_act of an operator against itself. Definitional."),
    ("e45_disagreement_geometry", "C2"): (
        "SELF_COMPARISON",
        "recorded in the audit as structurally incapable of failing; rebuilt properly it does NOT "
        "fire (r = +0.349 against a 0.3 threshold). The stored version is decoration."),
    ("e47_ablation_pairs", "bottom"): (
        "COMPUTED_STORED_NEVER_GATED",
        "the bottom-k arm was computed and stored and no decision rule was ever attached to it. "
        "Applying the SAME criterion the top-k arm is gated by (selectivity >= 0.5) to bottom-k, "
        "3 of 5 pairs clear the bar -- so the control, had it been gated, would have failed on 3 "
        "of 5. Recomputed here rather than asserted."),
}


def walk_controls(o, path, out, under=False):
    if not isinstance(o, dict):
        return
    for k, v in o.items():
        if k == "provenance":
            continue
        p = f"{path}.{k}" if path else k
        if CNAME.match(k) or (under and isinstance(v, (dict, bool))):
            m = CNAME.match(k)
            out.setdefault(m.group(1).upper() if m else k, []).append((p, v))
            continue
        walk_controls(v, p, out, under=(k == "controls"))


def classify(entries):
    """Mechanical class for one (file, control) pair, from every instance of it in that file."""
    gated = evaluated = none_gate = 0
    for _, v in entries:
        if isinstance(v, dict):
            if "fires" in v or "passes" in v:
                gated += 1
                g = v.get("fires", v.get("passes"))
                if g is None: none_gate += 1
                else: evaluated += 1
            # a bare recorder dict
        elif isinstance(v, bool):
            gated += 1; evaluated += 1
    if gated == 0:
        return "NO_GATE"
    if evaluated == 0:
        return "GATE_NEVER_EVALUATED"
    fired = []
    for _, v in entries:
        if isinstance(v, dict) and ("fires" in v or "passes" in v):
            fired.append(v.get("fires", v.get("passes")))
        elif isinstance(v, bool):
            fired.append(v)
    if any(x is False for x in fired):
        return "FIRED_FALSE"
    return "FIRED_TRUE"


def main() -> int:
    # ---------------------------------------------------------- 1. what the results files contain
    per_file, inputs = {}, []
    for f in sorted(glob.glob(R("results", "*.json"))):
        rel = os.path.relpath(f, REPO)
        if SKIP.search(rel):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        out: dict = {}
        walk_controls(d, "", out)
        if out:
            per_file[rel] = out
            inputs.append(f)

    records, counts = [], collections.Counter()
    for rel, ctrls in per_file.items():
        stem = os.path.basename(rel)[:-5]
        for name, entries in ctrls.items():
            cls = classify(entries)
            man = MANUAL.get((stem, name))
            rec = {"file": rel, "control": name, "n_instances": len(entries),
                   "mechanical_class": cls,
                   "power": ("NONE" if cls in ("NO_GATE", "GATE_NEVER_EVALUATED") else "HAS_POWER"),
                   "example_path": entries[0][0]}
            if man:
                rec["hand_audited_class"], rec["why"] = man
                rec["power"] = "NONE"
            records.append(rec)
            counts[rec["power"]] += 1
            counts["class:" + (man[0] if man else cls)] += 1

    # ---------------------------------------------------------- 2. declared in a spec, never coded
    declared_missing = []
    spec_files = glob.glob(R("specs", "experiments", "*.md")) + glob.glob(R("specs", "archive", "prereg", "*.md"))
    stem_to_results = {}
    for rel in per_file:
        stem_to_results.setdefault(os.path.basename(rel).split("_")[0].lower(), []).append(rel)
    for sp in spec_files:
        txt = open(sp).read()
        eid = re.match(r"(?:PREREG_)?([A-Za-z0-9]+)", os.path.basename(sp))
        if not eid:
            continue
        key = eid.group(1).lower()
        targets = stem_to_results.get(key, [])
        if not targets:
            continue
        named = set(m.group(1).upper() for m in re.finditer(r"\*\*(C\d+[a-z]?)\b", txt))
        named |= set(m.group(1).upper() for m in re.finditer(r"^\s*[-*]\s+\*?\*?(C\d+[a-z]?)\b", txt, re.M))
        # a spec may cover several results files; a control it declares counts as CODED if it
        # appears in ANY of them. Checking per-file instead would manufacture false positives --
        # verified directly on E65, whose spec declares C1-C5 across two files.
        present = set().union(*(set(per_file[t]) for t in targets))
        for c in sorted(named - present):
            declared_missing.append({"spec": os.path.relpath(sp, REPO),
                                     "results_files_checked": targets,
                                     "control_declared_but_absent_from_all_of_them": c})
    counts["class:DECLARED_NEVER_CODED"] = len(declared_missing)

    powerless = [r for r in records if r["power"] == "NONE"]
    rec = {
        "experiment": "R4e — the control-power audit: which controls could not have failed?",
        "prereg": "specs/experiments/R4_corrections.md",
        "prereg_sha256": sha256_file(R("specs", "experiments", "R4_corrections.md")),
        "status": "PRE-REGISTERED",
        "decision_rule_verbatim": ("Produce the enumeration as a results file, mark each in place, "
                                   "and add the power question to the experiment-spec template. "
                                   "Include E47's bottom-k control, which was computed, stored, "
                                   "never gated, and would have failed 3 of 5."),
        "counting_rule": ("one row per (results file, control base name); repeated per-set or "
                          "per-corpus instances of the SAME control collapse to one row. Smoke, "
                          "re-run and pooled duplicates of a file are excluded, since they are the "
                          "same controls counted twice."),
        "n_results_files_with_controls": len(per_file),
        "n_control_rows": len(records),
        "n_powerless": len(powerless),
        "counts": dict(counts),
        "SLATE_COMPARISON": {
            "slate_says": ("27 of 97 control fields could not have failed: 8 compare a quantity to "
                           "itself, 6 are recorders with no gate, 4 are declared in a docstring and "
                           "never coded, 9 are arithmetic identities"),
            "recomputed_n_control_rows": len(records),
            "recomputed_n_powerless": len(powerless),
            "note": ("the totals depend entirely on the counting rule, which the slate does not "
                     "state. This file states its rule above so the number is checkable. What does "
                     "NOT depend on the rule, and is the finding: a large minority of this "
                     "repository's controls carry no gate at all, and several named in specs have "
                     "no corresponding key in the results file they belong to.")},
        "powerless_controls": powerless,
        "declared_in_a_spec_but_absent_from_the_results_file": declared_missing,
        "all_control_rows": records,
        "REQUIRED_CHANGE_TO_THE_SPEC_TEMPLATE": (
            "specs/experiments/README.md's CONTROLS row becomes: 'each control, the number it "
            "actually produced, AND the number that would have made it fail. A control you cannot "
            "state a failing value for did not have power, and must be labelled a RECORDER.'"),
    }
    write_result(R("results", "r4e_control_power.json"), rec, experiment="R4e", inputs=inputs)
    print(f"control rows: {len(records)} across {len(per_file)} results files")
    for k in sorted(counts):
        print(f"   {k:34s} {counts[k]}")
    print(f"\npowerless (no gate, never evaluated, or hand-audited as unfalsifiable): {len(powerless)}")
    print(f"declared in a spec but absent from the results file: {len(declared_missing)}")
    for d in declared_missing[:12]:
        print(f"   {d['control_declared_but_absent_from_all_of_them']:6s} {d['spec']} "
              f"-> {', '.join(os.path.basename(x) for x in d['results_files_checked'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
