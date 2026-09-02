#!/usr/bin/env python3
"""
t1_verify.py — what a repro module actually exercises, and whether its payload moved.

WHY THIS IS A SCRIPT AND NOT A SHELL HEREDOC. The reproduction has to be reproducible too. Checks
run as `python - <<EOF` exist only in one terminal's scrollback: they cannot be re-run, diffed,
hash-stamped or reviewed, which is precisely the property this repository demands of everything
else. This file is the checker, in git, with a hash.

TWO QUESTIONS, kept separate because conflating them overstates coverage.

  1. DID THE PAYLOAD MOVE?  `canonical_payload_sha` excludes the provenance block, so a rerun that
     changes only a timestamp or hostname does not move it. If it moves, a NUMBER moved.

  2. WHAT DID THE MODULE ACTUALLY EXERCISE?  A module's TIER is a label; what it ran is evidence.
     `cv7_rung` and `cv6_ladder` are labelled T1 but invoke `--adjudicate`, which returns before
     any model is loaded -- they re-aggregate stored JSON, which is T0 work. Counting them as
     "reads reproduced from the released operators" would overstate the claim.

     The evidence used here is the results file's own `provenance.inputs`: a run that consumed
     operators records `.pt` paths with SHA-256. Three classes come out of it:

       SCORES_PT     .pt inputs recorded  -> genuinely re-scored from released operators
       NO_PT_USED    no .pt, and none needed (--adjudicate re-aggregation; answer-competence runs
                     that score the MODEL and never touch a lens)
       PT_UNRECORDED the module DECLARES .pt inputs in INPUTS=(), but the results file records
                     none -- the script predates input stamping. The operator was used; the
                     evidence that it was is missing.

     That third class is not a failure. It is the pre-provenance set, and it is exactly the set
     the sweep's pre-hoc map already flags as hardest to diagnose.

  .venv/bin/python tools/t1_verify.py --module cv3_margins --before /tmp/t1_before
  .venv/bin/python tools/t1_verify.py --classify-all
"""
import argparse, glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def module_spec(mid: str) -> dict:
    p = os.path.join(ROOT, "repro", "exp", f"{mid}.sh")
    if not os.path.isfile(p):
        raise SystemExit(f"no such module: {p}")
    src = open(p, encoding="utf-8", errors="ignore").read()
    grab = lambda k: (re.search(rf'{k}="([^"]*)"', src) or [None, None])[1]
    arr = lambda k: [t.strip('"\'') for b in re.findall(rf"{k}=\(([^)]*)\)", src) for t in b.split()]
    return {"id": mid, "tier": grab("MODULE_TIER"), "cost": grab("MODULE_COST"),
            "inputs": arr("INPUTS"), "outputs": arr("OUTPUTS"),
            "invocations": [f"{a}{b}".strip() for a, b in re.findall(r"run_py\s+(\S+\.py)([^;\n]*)", src)]}


def classify(spec: dict) -> dict:
    """What the module actually exercised, from the stored evidence."""
    declares_pt = any(i.endswith(".pt") for i in spec["inputs"])
    recorded_pt = None
    for o in spec["outputs"]:
        f = os.path.join(ROOT, o)
        if not o.endswith(".json") or not os.path.isfile(f):
            continue
        try:
            rec = json.load(open(f))
        except Exception:
            continue
        ins = (rec.get("provenance") or {}).get("inputs") or []
        recorded_pt = sum(1 for i in ins if str(i.get("path", "")).endswith(".pt"))
        break
    if recorded_pt:
        cls, why = "SCORES_PT", f"{recorded_pt} .pt operators recorded as hashed inputs"
    elif declares_pt:
        cls, why = ("PT_UNRECORDED",
                    "module declares .pt inputs but the results file records none; the script "
                    "predates input stamping")
    else:
        adj = any("--adjudicate" in v for v in spec["invocations"])
        cls, why = ("NO_PT_USED",
                    "re-aggregates stored JSON (--adjudicate); no operator involved" if adj
                    else "no operator involved (scores the model directly, or recomputes)")
    return {"class": cls, "why": why, "declares_pt": declares_pt, "recorded_pt": recorded_pt}


def payload_verdict(spec: dict, before_dir: str) -> list:
    from provenance import canonical_payload_sha
    out = []
    for o in spec["outputs"]:
        new = os.path.join(ROOT, o)
        old = os.path.join(before_dir, os.path.basename(o))
        if not os.path.isfile(new):
            out.append({"output": o, "verdict": "DID_NOT_RUN", "detail": "output absent"}); continue
        if not os.path.isfile(old):
            out.append({"output": o, "verdict": "NO_BASELINE",
                        "detail": f"no backup at {old}; back up BEFORE running"}); continue
        a, b = json.load(open(old)), json.load(open(new))
        ha, hb = canonical_payload_sha(a), canonical_payload_sha(b)
        out.append({"output": o, "verdict": "BIT_IDENTICAL" if ha == hb else "MOVED",
                    "payload_before": ha[:16], "payload_after": hb[:16]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module")
    ap.add_argument("--before", default="/tmp/t1_before")
    ap.add_argument("--classify-all", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.classify_all:
        rows = []
        for f in sorted(glob.glob(os.path.join(ROOT, "repro", "exp", "*.sh"))):
            mid = os.path.basename(f)[:-3]
            if mid == "_lib":
                continue
            spec = module_spec(mid)
            if spec["tier"] != "T1":
                continue
            c = classify(spec)
            rows.append({"module": mid, **c})
        from collections import Counter
        cnt = Counter(r["class"] for r in rows)
        print(f"  {len(rows)} modules labelled T1\n")
        for k in ("SCORES_PT", "PT_UNRECORDED", "NO_PT_USED"):
            print(f"   {k:16s} {cnt.get(k,0):3d}")
        print()
        for r in sorted(rows, key=lambda r: (r["class"], r["module"])):
            print(f"   {r['class']:16s} {r['module']:20s} {r['why']}")
        print(f"\n  T1 as 'reads re-scored from released operators' = {cnt.get('SCORES_PT',0)}"
              f" of {len(rows)}. The other {len(rows)-cnt.get('SCORES_PT',0)} reproduce something"
              f" real, but not that.")
        if a.json:
            json.dump(rows, open(a.json, "w"), indent=1); print(f"  wrote {a.json}")
        return 0

    if not a.module:
        ap.error("give --module or --classify-all")
    spec = module_spec(a.module)
    c = classify(spec)
    print(f"  module   {spec['id']}   tier={spec['tier']}")
    print(f"  exercises {c['class']}: {c['why']}")
    for r in payload_verdict(spec, a.before):
        extra = (f"  {r.get('payload_before')} -> {r.get('payload_after')}"
                 if r["verdict"] in ("BIT_IDENTICAL", "MOVED") else f"  {r.get('detail','')}")
        print(f"  {r['verdict']:14s} {r['output']}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
