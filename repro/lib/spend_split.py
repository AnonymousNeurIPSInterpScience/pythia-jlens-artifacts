#!/usr/bin/env python3
"""spend_split.py — split vast.ai spend into prior arcs vs this programme.

The account predates this programme. Quoting the all-time total as the cost of the
Jacobian-lens work overstates it by ~10x, so this recomputes the split from the billing
records rather than leaving a number in a document that nobody can check.

Boundary: 7e82292, the first commit adding pythia/ (2026-08-09).

    vastai show invoices --raw -s 2020-01-01 > /tmp/inv.json
    python repro/lib/spend_split.py /tmp/inv.json
"""
import collections, datetime, json, sys

CUT = "2026-08-09"          # first commit adding pythia/  (7e82292)

def main(path):
    rows = [r for r in json.load(open(path))
            if isinstance(r, dict) and (r.get("type") or "").lower() == "charge"]
    if not rows:
        return print("no charge records — did you pass --raw -s <early date>?")
    day = lambda r: datetime.datetime.fromtimestamp(
        float(r["timestamp"]), datetime.UTC).strftime("%Y-%m-%d")
    acc = {k: collections.Counter() for k in ("prior", "pythia")}
    inst = {k: set() for k in ("prior", "pythia")}
    for r in rows:
        try: amt = abs(float(r.get("amount") or 0))
        except (TypeError, ValueError): amt = 0.0
        k = "pythia" if day(r) >= CUT else "prior"
        acc[k]["usd"] += amt
        inst[k].add(str(r.get("instance_id")))
        if " gpu " in (r.get("description") or "").lower():
            acc[k]["gpu_h"] += float(r.get("quantity") or 0)
    tot = acc["prior"]["usd"] + acc["pythia"]["usd"]
    print(f"  boundary {CUT} (7e82292, first commit adding pythia/)\n")
    print(f"  {'':<44}{'$':>9}{'%':>7}{'GPU-h':>9}{'inst':>6}")
    for k, label in (("prior",  "prior arcs (not part of this release)"),
                     ("pythia", "THE ESTIMATOR PROGRAMME (Pythia)")):
        a = acc[k]
        print(f"  {label:<44}{a['usd']:>9.2f}{a['usd']/tot*100:>6.1f}%"
              f"{a['gpu_h']:>9.1f}{len(inst[k]):>6}")
    print(f"  {'all-time':<44}{tot:>9.2f}{100.0:>6.1f}%"
          f"{acc['prior']['gpu_h']+acc['pythia']['gpu_h']:>9.1f}"
          f"{len(inst['prior'])+len(inst['pythia']):>6}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/inv.json")
