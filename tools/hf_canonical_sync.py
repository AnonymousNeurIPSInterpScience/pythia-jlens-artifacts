#!/usr/bin/env python3
"""hf_canonical_sync.py — make an HF repo match the canonical local scientific surface, EXACTLY.

Written because repro/03_fetch_artifacts.sh, repro/04_fetch_results.sh, repro/05_mirror_results.sh
and tools/hf_sync_results.py are all ADD-ONLY or path-stale: none of them removes a path that no
longer exists locally, so the mirror still carries the pre-restructure `results/e28_*.pt` layout and
the pre-R4g provenance sidecars that record `corpus: "wikitext"` for a Github lens.

Comparison is CONTENT-EXACT and needs no downloads: HF's tree API returns the git blob sha1 for
every file and an LFS sha256 for LFS files, and both are computable locally.

    .venv/bin/python tools/hf_canonical_sync.py --plan            # default: show, change nothing
    .venv/bin/python tools/hf_canonical_sync.py --apply
    .venv/bin/python tools/hf_canonical_sync.py --verify

By default raw `corpora/*.jsonl` already on the remote are LEFT ALONE (--purge-raw-corpora to
delete them). They are not part of the canonical scientific surface, but deleting the operator's
own private backup copy is not something this tool does silently.
"""
from __future__ import annotations
import argparse, hashlib, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_REPO = "AnonymousInterpScience/pythia-jlens-artifacts"

# ---- the canonical scientific surface
CANON_DIRS = ["results"]
CANON_FILES = ["corpora/manifest.json", "corpora/manifest_ood.json",
               "corpora/dataset_revisions.json"]
SKIP_BASENAMES = {".DS_Store"}
SKIP_SUFFIXES = (".tmp", ".bak", ".partial", ".part", ".swp")


def local_surface() -> dict[str, str]:
    out = {}
    for d in CANON_DIRS:
        base = os.path.join(ROOT, d)
        for dp, _, fns in os.walk(base):
            for fn in fns:
                if fn in SKIP_BASENAMES or fn.endswith(SKIP_SUFFIXES):
                    continue
                p = os.path.join(dp, fn)
                out[os.path.relpath(p, ROOT)] = p
    for f in CANON_FILES:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            out[f] = p
        else:
            print(f"  WARNING: canonical file missing locally: {f}", file=sys.stderr)
    return out


def git_blob_sha1(path: str) -> str:
    n = os.path.getsize(path)
    h = hashlib.sha1()
    h.update(b"blob %d\0" % n)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_surface(api, repo):
    from huggingface_hub import HfApi  # noqa: F401
    tree = api.list_repo_tree(repo, repo_type="model", recursive=True, expand=True)
    out = {}
    for x in tree:
        if getattr(x, "size", None) is None:
            continue
        lfs = getattr(x, "lfs", None)
        out[x.path] = {"size": x.size, "blob_id": getattr(x, "blob_id", None),
                       "lfs_sha256": getattr(lfs, "sha256", None) if lfs else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--purge-raw-corpora", action="store_true",
                    help="also delete raw corpora/*.jsonl from the remote")
    ap.add_argument("--batch", type=int, default=120)
    a = ap.parse_args()

    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    api = HfApi()

    loc = local_surface()
    rem = remote_surface(api, a.repo)
    print(f"local canonical files : {len(loc)}")
    print(f"remote files          : {len(rem)}")

    add, update, identical = [], [], []
    for rel, path in sorted(loc.items()):
        r = rem.get(rel)
        if r is None:
            add.append(rel); continue
        if r["size"] != os.path.getsize(path):
            update.append(rel); continue
        if r["lfs_sha256"]:
            same = r["lfs_sha256"] == sha256(path)
        else:
            same = r["blob_id"] == git_blob_sha1(path)
        (identical if same else update).append(rel)

    protected_raw = {p for p in rem if p.startswith("corpora/") and p.endswith(".jsonl")}
    keep_meta = set(CANON_FILES)
    stale = [p for p in sorted(rem)
             if p not in loc and p != ".gitattributes"
             and not (p in protected_raw and not a.purge_raw_corpora)
             and p not in keep_meta]

    print(f"\n  identical : {len(identical)}")
    print(f"  ADD       : {len(add)}")
    print(f"  UPDATE    : {len(update)}")
    print(f"  DELETE    : {len(stale)}")
    if protected_raw and not a.purge_raw_corpora:
        print(f"  (kept {len(protected_raw)} raw corpora/*.jsonl on the remote; "
              f"--purge-raw-corpora to delete)")

    def head(lbl, xs, n=12):
        if xs:
            print(f"\n{lbl}:")
            for x in xs[:n]:
                print("   ", x)
            if len(xs) > n:
                print(f"    ... and {len(xs)-n} more")
    head("ADD", add); head("UPDATE", update); head("DELETE (stale on remote)", stale)

    if a.verify:
        ok = not add and not update and not stale
        print(f"\nVERIFY: {'MATCH' if ok else 'MISMATCH'} "
              f"(add {len(add)}, update {len(update)}, delete {len(stale)})")
        return 0 if ok else 1

    if not a.apply:
        print("\n--plan only. Nothing changed. Re-run with --apply.")
        return 0

    ops = ([CommitOperationDelete(path_in_repo=p) for p in stale]
           + [CommitOperationAdd(path_in_repo=r, path_or_fileobj=loc[r]) for r in add + update])
    print(f"\napplying {len(ops)} operations in batches of {a.batch} ...")
    for i in range(0, len(ops), a.batch):
        chunk = ops[i:i + a.batch]
        api.create_commit(repo_id=a.repo, repo_type="model", operations=chunk,
                          commit_message=f"canonical sync {i//a.batch+1}: "
                                         f"{sum(isinstance(o,CommitOperationDelete) for o in chunk)} del, "
                                         f"{sum(isinstance(o,CommitOperationAdd) for o in chunk)} add/update")
        print(f"  batch {i//a.batch+1}: {len(chunk)} ops committed", flush=True)
    print("done. re-run with --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
