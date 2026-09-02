#!/usr/bin/env bash
# repro/05_mirror_results.sh — RETIRED 2026-09-01. Do not use.
#
# This was the PUSH side of the mirror: it uploaded results/ to a private HF repo. It is retired
# because the release no longer has a push step a reviewer or a maintainer needs:
#
#   * The canonical artifact mirror is PUBLIC, already populated, and frozen for review:
#         https://huggingface.co/AnonymousInterpScience/pythia-jlens-artifacts
#   * It was ADD-ONLY. It never removed a path that had gone away locally, so after the results
#     tree was restructured (operators moved to results/lenses/{e28,plad,ladder,misc}/) the mirror
#     kept BOTH layouts, and served pre-R4g provenance sidecars recording `corpus: "wikitext"`
#     for Github-fitted operators. A reviewer fetching those got wrong metadata.
#   * Its replacement for maintenance is tools/hf_canonical_sync.py, which compares by content
#     hash and both ADDS and DELETES so the mirror matches the local tree exactly.
#
# The reviewer-facing direction is FETCH, and it is the supported path:
#     bash repro/04_fetch_results.sh        # results JSON, ~8 MB   <-- start here
#     bash repro/03_fetch_artifacts.sh --all # operators, ~16 GB
#     bash repro/06_data.sh --build          # rebuild the corpus pool at its pinned revisions

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
cat <<'MSG'

  repro/05_mirror_results.sh is RETIRED and does nothing.

  To FETCH (what a reviewer wants):
      bash repro/04_fetch_results.sh
      bash repro/03_fetch_artifacts.sh --all

  To re-sync the mirror from a maintainer's tree:
      .venv/bin/python tools/hf_canonical_sync.py            # plan
      .venv/bin/python tools/hf_canonical_sync.py --apply
      .venv/bin/python tools/hf_canonical_sync.py --verify

MSG
exit 0
