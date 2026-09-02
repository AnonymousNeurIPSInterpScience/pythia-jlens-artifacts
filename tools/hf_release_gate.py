#!/usr/bin/env python3
"""
Pre-review gate for the artifact mirror. THE CONSTRAINT IS DOUBLE-BLIND ANONYMITY.

WHY THIS EXISTS. NeurIPS 2026 workshop submissions are double-blind: the draft must
carry no identifying information "in the main text, figures, or supplementary
materials", and a linked artifact is supplementary material. The mirror is hosted
under an account whose name is identifying, and three things make that sticky
in ways a quick fix does not address (the release procedure §"the mirror"):

  1. The account name is in every URL a reviewer would follow.
  2. The mirror's GIT HISTORY retains ten prior-arc `.pt` (~3.75 GB) from separate
     research lines, and ARTIFACTS.md publishes a working revision-pinned download
     recipe for them. History is not removed by deleting files from the tip.
  3. The mirror was renamed from a predecessor id on 2026-08-12 and HF keeps a
     redirect. A RENAME IS NOT ANONYMISATION — it carries both the history and the
     redirect.

THE PLAN OF RECORD (operator, 2026-08-22): do not attempt to anonymise this mirror in
place. Take the final commit of each of GitHub and HF and re-host on purpose-built
anonymising front-ends — `anonymous.4open.science` for the code and an anonymous HF
mirror for the artifacts. That defeats (1), (2) and (3) at once, because the anonymous
copy starts from a single squashed state with no history and no redirect.

Licensing is a SECONDARY consideration and is largely benign: CommonPile is openly
licensed by construction, FineWeb is ODC-By, Wikipedia is CC BY-SA, PubMed abstracts
are NLM-redistributable, arXiv abstracts are covered, and the five Pile components are
redistributable as part of the Pile. `OOD_News_2024` is the only genuine copyright
exposure. While the mirror is private, none of this is live.

  .venv/bin/python tools/hf_release_gate.py            # report
  .venv/bin/python tools/hf_release_gate.py --strict   # exit 1 if not review-safe

Exit codes: 0 = review-safe, 1 = would deanonymise or would publish plaintext.
"""
import argparse, configparser, os, re, sys

REPO_ID = os.environ.get("HF_REPO", "")
REPO_TYPE = "model"


# Any of these appearing in a reviewer-visible URL breaks double-blind.
# The literal strings live in tools/deanon_patterns.json, which is UNTRACKED on purpose: a gate
# that ships its own pattern list republishes exactly what it exists to remove. This file used to
# hardcode them and was itself the last leak the sweep reported.
def _identifying():
    import json as _j
    f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "tools", "deanon_patterns.json")
    if not os.path.exists(f):
        sys.exit("tools/deanon_patterns.json absent - cannot certify anonymity without the list "
                 "of what makes it fail. Rebuild from the release-hygiene procedure.")
    return [rx for _lbl, rx, _why in _j.load(open(f))["patterns"]]


IDENTIFYING = _identifying()

PLAINTEXT_PREFIX, PLAINTEXT_SUFFIX = "corpora/", ".jsonl"
HIGH_RISK = {"corpora/OOD_News_2024.jsonl", "corpora/OOD_News_2024_dedup.jsonl"}


def token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    c = configparser.ConfigParser()
    c.read(os.path.expanduser("~/hf_cache/stored_tokens"))
    for prof in c.sections():
        if "hf_token" in c[prof]:
            return c[prof]["hf_token"]
    sys.exit("no token: set HF_TOKEN or populate ~/hf_cache/stored_tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi(token=token())
    info = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=False)
    files = sorted(s.rfilename for s in info.siblings)
    plaintext = [f for f in files
                 if f.startswith(PLAINTEXT_PREFIX) and f.endswith(PLAINTEXT_SUFFIX)]
    import re as _re
    identifying = [t for t in IDENTIFYING if _re.search(t, REPO_ID)]

    print(f"repo:    {REPO_ID}")
    print(f"private: {info.private}")
    print(f"files:   {len(files)}")

    fail = False

    print("\n--- DOUBLE-BLIND (the binding constraint) ---")
    if identifying:
        print(f"  FAIL  repo id contains an identifying token: {identifying}")
        print("        A reviewer following this URL learns who produced it.")
        fail = True
    else:
        print("  ok    repo id carries no identifying token")
    print("  NOTE  git history retains prior-arc artifacts and ARTIFACTS.md publishes a")
    print("        revision-pinned recipe for them; a rename keeps history AND a redirect.")
    print("        Deleting files from the tip does NOT fix this.")
    print("  PLAN  re-host the final commit on an anonymous HF mirror +")
    print("        anonymous.4open.science for the code. Do not anonymise in place.")

    print("\n--- PLAINTEXT (secondary) ---")
    print(f"  corpus plaintext on mirror: {len(plaintext)}")
    for f in sorted(HIGH_RISK & set(plaintext)):
        print(f"    {f}  HIGH RISK (2024 news scrape)")
    if plaintext and not info.private:
        print("  FAIL  plaintext is publicly readable")
        fail = True
    elif plaintext:
        print("  ok    mirror is private, so nothing is redistributed today")

    print("\nSTATUS:", "NOT REVIEW-SAFE — use the anonymous mirror" if fail
          else "review-safe as configured")
    sys.exit(1 if (fail and a.strict) else 0)


if __name__ == "__main__":
    main()
