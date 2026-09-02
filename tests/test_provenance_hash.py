#!/usr/bin/env python3
"""test_provenance_hash.py — R4f / finding M-2: the payload hash must survive a JSON round-trip.

WHY THIS TEST EXISTS. `payload_sha256` is the only mechanism in the repository that can say "this
file's content is the content that was written". It could not do that for a payload keyed by
integers: `json.dumps(..., sort_keys=True)` sorts int keys NUMERICALLY and str keys
LEXICOGRAPHICALLY, and stringifies int keys only after sorting. A dict keyed 9..21 -- which is what
a per-layer result looks like -- therefore hashes differently in memory and after `json.load`, so
the stored hash did not verify against the file it was stored in.

Measured on the tree at 2026-08-19: 29 of 225 non-sidecar results files carry a `payload_sha256`,
and exactly one fails to verify on a round-trip -- `results/e58_algebra_audit.json`, whose
`A_e45_orientation.G_relative_asymmetry_per_layer` is keyed 9..21. Re-keying that payload to int
reproduces the stored hash EXACTLY, which is what identifies the mechanism.

The first check below FAILS against the unfixed `legacy_payload_sha` and passes against the fixed
`canonical_payload_sha`. That is the regression.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from provenance import (canonical_payload_sha, legacy_payload_sha,  # noqa: E402
                        verify_payload_sha)

_p = _f = 0
def check(name, cond, detail=""):
    global _p, _f
    if cond: _p += 1; print(f"  ok  {name}   [{detail}]")
    else:    _f += 1; print(f"  FAIL {name}   [{detail}]")

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# the shape that breaks it: >= 10 numeric keys, so numeric and lexicographic order disagree
INT_KEYED = {"experiment": "T", "per_layer": {l: float(l) / 7 for l in range(9, 22)}}
STR_KEYED = json.loads(json.dumps(INT_KEYED))          # exactly what a JSON round-trip produces

check("round_trip_invariance_is_the_regression",
      canonical_payload_sha(INT_KEYED) == canonical_payload_sha(STR_KEYED),
      "int-keyed and round-tripped payloads hash the same under the fixed rule")

check("the_unfixed_rule_really_did_differ",
      legacy_payload_sha(INT_KEYED) != legacy_payload_sha(STR_KEYED),
      "a control on the control: if these agreed, the test above could not fail and would be "
      "one of R4e's 27 powerless controls")

check("key_order_does_not_matter",
      canonical_payload_sha({"b": 1, "a": 2}) == canonical_payload_sha({"a": 2, "b": 1}),
      "sort_keys still normalises insertion order")

check("content_change_still_changes_the_hash",
      canonical_payload_sha({"a": 1}) != canonical_payload_sha({"a": 2}),
      "the fix must not make the hash insensitive to content")

check("provenance_block_is_excluded",
      canonical_payload_sha({"a": 1, "provenance": {"utc": "x"}})
      == canonical_payload_sha({"a": 1, "provenance": {"utc": "y"}}),
      "a timestamp must not enter the content hash")

# ---- the live file the finding is about
p = os.path.join(REPO, "results", "e58_algebra_audit.json")
if os.path.exists(p):
    d = json.load(open(p))
    def intify(o):
        if isinstance(o, dict):
            return {(int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): intify(v)
                    for k, v in o.items()}
        if isinstance(o, list): return [intify(x) for x in o]
        return o
    stored = d["provenance"]["payload_sha256"]
    # HISTORY, 2026-08-24. These two assertions used to REQUIRE e58 to be broken: they witnessed
    # a real defect (the file's stored hash was reachable only by re-keying the payload to int,
    # which is the pre-R4f rule) and they were the reason the defect could not be forgotten.
    #
    # The defect is now repaired: e58 was re-emitted from experiments/t58_algebra_audit.py, which
    # writes under the canonical rule, so the file verifies as written. A witness that asserts a
    # bug still exists MUST be inverted when the bug is fixed, or it fails forever and gets muted,
    # which is how a green suite stops meaning anything. The mechanism knowledge is kept above and
    # in the int-key regression at the top of this file, which is what actually guards the rule.
    check("e58_verifies_as_written",
          verify_payload_sha(d)["rule"] in ("canonical", "legacy"),
          "e58's stored hash is reachable from the file as written; the 2026-08-22 defect "
          "(reachable only after re-keying to int) is repaired, not hidden")
    check("e58_is_not_reachable_only_by_intifying",
          not (verify_payload_sha(d)["rule"] == "MISMATCH"
               and legacy_payload_sha(intify(d)) == stored),
          "the specific broken state this file was in must not recur: a stored hash that only "
          "an int re-key reproduces means the emitter regressed to the pre-R4f rule")

print(f"\n=== {_p}/{_p + _f} PASSED ===")
sys.exit(1 if _f else 0)
