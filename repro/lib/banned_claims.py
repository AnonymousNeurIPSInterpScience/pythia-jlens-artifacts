#!/usr/bin/env python3
"""banned_claims.py — fail the build if a disproved claim is stated as a finding.

WHY THIS FILE EXISTS. The layer-shuffled ("deranged") Jacobian scoring 5/6 against the
logit lens is an ARTIFACT OF MIN-OVER-LAYERS SCORING, not a result. min is existential:
it asks whether a concept surfaces at *any* single layer, so it cannot distinguish a
working operator from a broken one. Under persistence the shuffled operator scores 0/6,
loses to J^P on 5 of 6 reads and 6 of 6 writes, and its generic component is negative on
5 of 6 sets.

The reading it produced -- "most of the J-lens advantage is a generic basis change" --
is also logically impossible: if a layer-deranged J matched the real operator, then
d h_final / d h_l would not depend on l, intermediate layers would do nothing
distinguishable, and no transformer would need more than one layer. That contradicts
backpropagation as understood since Werbos.

It was reported as a finding, stood for weeks, propagated into four documents, and was
re-raised repeatedly after retraction -- including once as a proposed paper TITLE. Asking
a person or a model to remember is not a control. This is the control.

    python repro/lib/banned_claims.py          # scan the repo, exit 1 on any violation

A mention is ALLOWED only when the same line, or a line within RETRACTION_WINDOW lines
above it, marks it as disproved. Retraction records are the point; restatements are not.

FALSE POSITIVES, AND WHY THEY GET AN EXEMPTION RATHER THAN A LOOSER REGEX. The patterns are
deliberately paranoid: they match a co-occurrence of tokens, not a parsed claim. So a sentence
can trip one while saying something entirely different -- "an earlier control fired on 5 of 6
evaluation sets, re-run with random derangements" is a control's FIRING RATE, not the
derangement's SCORE against a baseline. Three ways to resolve that, and only one is honest:

  (a) reword the source until the regex stops matching -- spell "5" as "five", break "5 of 6"
      with an article. This silences the gate and leaves it equally wrong for the next true
      sentence. It is regex-dodging and it is not done here.
  (b) loosen the regex with negative lookarounds. Every exception makes the tripwire weaker at
      catching the thing it exists to catch, and the claim it exists to catch came back four
      times.
  (c) an explicit, per-occurrence, self-justifying exemption. That is EXEMPT below.

An exemption is a marker on the flagged line or the line immediately above it, reading
`banned-claims:not-the-claim` followed by a statement of what the quantity actually is. It is
deliberately NOT windowed like CLEARED: a four-line window would let a genuinely banned sentence
sit under someone else's exemption. Every exemption is greppable, and the count is printed on a
clean run so they cannot proliferate unnoticed.

An exemption says "this is not the claim". CLEARED says "this IS the claim, and it is being
recorded as retracted". Do not use one for the other.
"""
from __future__ import annotations
import glob, os, re, sys

RETRACTION_WINDOW = 4

# (name, pattern that identifies the claim, why it is banned)
BANNED = [
    ("shuffled-J beats the baseline",
     re.compile(r"(shuffl|derang)\w*[^.\n]{0,180}?\b5\s*(/|of)\s*6"
                r"|(?:\b5\s*(?:/|of)\s*6)[^.\n]{0,180}?(shuffl|derang)", re.I),
     "5/6 is a min-over-layers artifact; under persistence it is 0/6"),
    ("generic basis change",
     re.compile(r"generic (basis|change of basis)|most of the (advantage|6/6) is generic"
                r"|advantage is (a )?generic", re.I),
     "the H2 'generic basis change' hypothesis is disproved and deleted"),
    ("bimodal partition",
     re.compile(r"bimodal partition", re.I),
     "computed under the defective metric; inverts when the aggregation changes"),
]

# Marks that make a mention a retraction rather than a restatement.
CLEARED = re.compile(
    r"retract|disprov|DISPROVED|artifact|artefact|false|must not reappear|do not restate"
    r"|banned|not a (result|finding)|metric test|REMOVED|deleted|slop|BANNED"
    # a DENIAL of the claim is not a restatement of it
    r"|there is no|destroys the operator|buys \\?emph\{?less than nothing|no such|is not a",
    re.I)

# An explicit per-occurrence exemption: the line trips a pattern but is not the banned claim.
# Checked ONLY on the flagged line and the one immediately above it, never over a window, so an
# exemption cannot shelter a neighbouring sentence. See the module docstring for why this exists
# instead of a looser pattern.
EXEMPT = re.compile(r"banned-claims:not-the-claim", re.I)

SCAN = ["*.md", "*.tex", "docs/**/*.md", "docs/**/*.tex",
        "paper/*.tex", "repro/*.sh", "repro/README.md", "docs/*.md"]
SKIP_DIRS = ("docs/archive/", "docs/archive/", ".venv/", "jacobian-lens/",
             "thirdparty/", "docs/prereg/")
# PROMPTS.jsonl is a verbatim transcript of what was said, including the errors.
# Rewriting it would falsify the record, so it is excluded by design.
SKIP_FILES = ("PROMPTS.jsonl",
              "repro/lib/banned_claims.py")


def scan() -> int:
    files: list[str] = []
    for pat in SCAN:
        files += glob.glob(pat)
    files = sorted({f for f in files
                    if not any(f.startswith(d) for d in SKIP_DIRS)
                    and f not in SKIP_FILES and os.path.isfile(f)})

    violations = []
    exemptions = []
    for f in files:
        try:
            lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines):
            for name, pat, why in BANNED:
                if not pat.search(line):
                    continue
                # NOT windowed, unlike CLEARED: the marker must be on this line or the one
                # directly above, so it cannot shelter a neighbour.
                if EXEMPT.search(line) or (i and EXEMPT.search(lines[i - 1])):
                    exemptions.append((f, i + 1, name))
                    continue
                # Strip the exemption marker before testing CLEARED. The marker contains the
                # word "banned", which CLEARED also matches, so without this an exemption would
                # additionally act as a WINDOWED retraction marker and shelter its neighbours.
                # The self-test's "claim two lines under a marker" case is exactly that hole.
                window = [EXEMPT.sub("", w) for w in lines[max(0, i - RETRACTION_WINDOW): i + 2]]
                if any(CLEARED.search(w) for w in window):
                    continue                      # properly marked as retracted
                violations.append((f, i + 1, name, why, line.strip()[:110]))

    if violations:
        print(f"\n  {len(violations)} BANNED CLAIM(S) STATED AS FINDINGS:\n")
        for f, ln, name, why, txt in violations:
            print(f"  {f}:{ln}")
            print(f"    claim : {name}")
            print(f"    why   : {why}")
            print(f"    text  : {txt}\n")
        print("  A mention is allowed ONLY if it is marked as retracted/disproved on the")
        print(f"  same line or within {RETRACTION_WINDOW} lines above it.\n")
        return 1

    print(f"  ok   {len(files)} files scanned, 0 banned claims stated as findings")
    if exemptions:
        # Printed on every clean run so exemptions cannot accumulate quietly. Each one is a
        # standing assertion that a pattern hit is not the claim; re-read them when the count moves.
        print(f"  note {len(exemptions)} explicit exemption(s), each stating why it is not the claim:")
        for f, ln, name in exemptions:
            print(f"         {f}:{ln}  ({name})")
    return 0


def self_test() -> int:
    """Show that the gate still fails on the inputs it exists to catch.

    An exemption mechanism is exactly the kind of thing that quietly turns a control into a no-op,
    so this constructs the cases that MUST fail and asserts they do. Run it whenever EXEMPT,
    CLEARED or BANNED changes.
    """
    import shutil
    import tempfile

    banned_line = ("The layer-shuffled Jacobian beats the free logit lens on 5 of 6 evaluation "
                   "sets, which is the headline of this section.")
    cases = [
        # (name, file body, must_fail)
        ("bare banned claim", banned_line, True),
        ("claim marked as retracted", "This was retracted: " + banned_line, False),
        ("claim carrying the exemption marker",
         "% banned-claims:not-the-claim\n" + banned_line, False),
        # The exemption must NOT be windowed. A marker two lines up may not shelter the line below.
        ("claim two lines under a marker",
         "% banned-claims:not-the-claim\nsome unrelated sentence.\n" + banned_line, True),
        # ...nor may a marker shelter the lines that follow the one it covers.
        ("second claim after an exempted one",
         "% banned-claims:not-the-claim\n" + banned_line + "\n" + banned_line, True),
    ]

    cwd = os.getcwd()
    failures = []
    for name, body, must_fail in cases:
        tmp = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmp, "paper"))
            with open(os.path.join(tmp, "paper", "case.tex"), "w") as fh:
                fh.write(body + "\n")
            os.chdir(tmp)
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = scan()
        finally:
            os.chdir(cwd)
            shutil.rmtree(tmp, ignore_errors=True)
        got_fail = rc != 0
        ok = got_fail == must_fail
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: "
              f"expected {'violation' if must_fail else 'clean'}, "
              f"got {'violation' if got_fail else 'clean'}")
        if not ok:
            failures.append(name)

    if failures:
        print(f"\n  SELF-TEST FAILED on {len(failures)} case(s): {', '.join(failures)}")
        return 1
    print(f"\n  self-test ok: {len(cases)} cases, the gate still fails on what it must")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    sys.exit(scan())
