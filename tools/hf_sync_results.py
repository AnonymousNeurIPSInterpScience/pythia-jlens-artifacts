#!/usr/bin/env python3
"""
Mirror results/ (and corpus manifests) to the private Hugging Face artifact repo.

Uploads only what is MISSING remotely; never deletes, never overwrites silently.
Verifies afterwards by re-reading the remote file list.

  .venv/bin/python tools/hf_sync_results.py            # dry run: report the diff
  .venv/bin/python tools/hf_sync_results.py --push     # upload the missing files

Token: read from ~/hf_cache/stored_tokens (the profile carrying `hf_token`), matching
docs/reproducibility/ARTIFACTS.md. Never printed.

CORPUS PLAINTEXT IS SYNCED, AND THE REPO MUST STAY PRIVATE.
All corpora are mirrored, including the out-of-stream ones. The reproducibility case
is decisive for those: OOD_News_2024 is a 2024 scrape whose source URLs rot, so it is
NOT reconstructible from the builder, and it is the clause-(d) load-bearing rung of
E48. Without the plaintext that experiment is unreproducible by anyone.

The mirror is PRIVATE. The binding constraint on it is NOT licensing but DOUBLE-BLIND
ANONYMITY: the account name is identifying, the git history retains prior-arc
artifacts from separate research lines, and a rename keeps both the history and an
id redirect. Plan of record: re-host the final commit on an anonymous HF mirror and
anonymous.4open.science rather than anonymising in place.
Run tools/hf_release_gate.py before exposing this to reviewers.
"""
import argparse, configparser, os, sys, glob
from collections import Counter

REPO_ID = os.environ.get("HF_REPO") or sys.exit(
    "set HF_REPO first — the mirror id is not hardcoded here (double-blind)")
REPO_TYPE = "model"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# what this script is allowed to send
INCLUDE_GLOBS = ["results/**/*", "corpora/*.json", "corpora/*.jsonl"]
EXCLUDE_SUBSTR = [".DS_Store", "__pycache__"]


def token():
    p = os.path.expanduser("~/hf_cache/stored_tokens")
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    c = configparser.ConfigParser()
    c.read(p)
    for prof in c.sections():
        if "hf_token" in c[prof]:
            return c[prof]["hf_token"]
    sys.exit(f"no token: set HF_TOKEN or populate {p}")


def local_files():
    out = set()
    for g in INCLUDE_GLOBS:
        for p in glob.glob(os.path.join(ROOT, g), recursive=True):
            if not os.path.isfile(p):
                continue
            rel = os.path.relpath(p, ROOT)
            if any(s in rel for s in EXCLUDE_SUBSTR):
                continue
            out.add(rel)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--batch", type=int, default=15, help="files per commit")
    a = ap.parse_args()

    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=token())

    remote = {s.rfilename for s in
              api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=False).siblings}
    local = local_files()
    missing = sorted(local - remote)
    size = sum(os.path.getsize(os.path.join(ROOT, m)) for m in missing)

    print(f"remote: {len(remote)} files")
    print(f"local (in scope): {len(local)} files")
    print(f"MISSING remotely: {len(missing)} files, {size/1e9:.2f} GB")
    print("  by ext:", dict(Counter(os.path.splitext(m)[1] for m in missing).most_common()))
    if not missing:
        print("nothing to do")
        return
    if not a.push:
        for m in missing[:20]:
            print("   ", m)
        if len(missing) > 20:
            print(f"    ... and {len(missing)-20} more")
        print("\ndry run — pass --push to upload")
        return

    sent = 0
    for i in range(0, len(missing), a.batch):
        chunk = missing[i:i + a.batch]
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=os.path.join(ROOT, p))
               for p in chunk]
        api.create_commit(
            repo_id=REPO_ID, repo_type=REPO_TYPE, operations=ops,
            commit_message=f"sync results: {len(chunk)} files ({i//a.batch + 1})")
        sent += len(chunk)
        print(f"  uploaded {sent}/{len(missing)}", flush=True)

    # verify against a fresh listing — never trust the return of the upload call
    remote2 = {s.rfilename for s in
               api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=False).siblings}
    still = sorted(set(missing) - remote2)
    print(f"\nVERIFY: remote now {len(remote2)} files; {len(still)} of the intended set still absent")
    for s in still[:10]:
        print("   MISSING", s)
    sys.exit(1 if still else 0)


if __name__ == "__main__":
    main()
