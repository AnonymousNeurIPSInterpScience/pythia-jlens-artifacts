#!/usr/bin/env python3
"""provenance.py — the one place a result learns where it came from.

WHY THIS EXISTS
  Discipline #2 says never report an unstored number. That is necessary and not sufficient: this
  programme has twice cited a number that WAS stored and still could not be traced to the code
  that produced it.

    * `4614x the seed x set noise floor` — the paper's headline, in six documents, produced by no
      script in any commit. When it was finally recomputed, six of seven numbers reproduced and
      that one did not.
    * `Github 0.02808, t = -1.65` — load-bearing for a pre-registered clause, in no results file.

  Both were recoverable only because their INPUTS happened to be stored. A results file that
  records its own script, that script's hash, the commit, the argv and the SHA-256 of every input
  is recoverable by construction. That is what this module emits.

WHAT A STAMP CONTAINS, and why each field is there
  script      path + SHA-256 of the FILE THAT RAN. A commit hash alone is not enough: this
              programme routinely runs with a dirty tree, and "commit abc123" then names code
              that was never executed.
  git         commit, branch, and `dirty`, with the SHA-256 of the working diff when dirty. A
              dirty result is not disqualified — most real results are produced mid-edit — but it
              must SAY so, and carry enough to reconstruct what actually ran.
  inputs      SHA-256 of every declared input artifact. This is what makes a chain auditable:
              e48c consumes e48 and e48b, and records their hashes, so a changed upstream file
              is detectable rather than silent.
  argv        the exact command line, so flags that change the science (--band, --rstrip,
              --n-derangements) are recorded rather than remembered.
  env         python / torch / transformers / platform. TF32 state too: a 10-bit mantissa breaks
              the anchor gate at 2e-5 and has cost this programme a run before.
  payload_sha SHA-256 of the RESULT CONTENT with the provenance block excluded. This is the
              reproducibility check: rerun, compare payload_sha256, and a match proves the
              science is identical even though the timestamp is not. Without excluding the
              timestamp, no two runs ever agree and the check is worthless.

USAGE — two lines in any experiment script

    from provenance import write_result
    write_result(out_path, rec, experiment="E48", inputs=[lens_a, lens_b])

  and the file lands with a `provenance` block. `tools/verify_provenance.py` audits the tree.

DELIBERATE NON-GOAL
  This does not try to make runs bit-reproducible. It makes them ACCOUNTABLE: given a results
  file, you can always find the code, the inputs and the command that produced it, and tell
  whether the tree was clean at the time.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)             # this module lives in src/; the repo is one level up
SCHEMA_VERSION = 1


def sha256_file(path: str) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                              timeout=20, check=True).stdout.strip()
    except Exception:
        return None                      # a tarball export has no .git; that is recorded, not fatal


def git_state() -> dict:
    """Git provenance, with the REASON recorded when there is none.

    NEVER raises and never aborts a run. The published tree is a FLATTENED SNAPSHOT with no `.git`
    at all -- that is what anonymous.4open.science serves -- so a reviewer reproducing a result
    there must not hit a hard failure. Aborting would break the harness on precisely the surface
    this repository is published on.

    But a bare `None` is also wrong, because it cannot be told apart from "the stamp was never
    written". That ambiguity is not hypothetical: the E48 operator panel carries `commit: None`, and
    it took an archaeology pass to establish that the cause was those lenses being fitted two
    minutes before `provenance.py` existed, rather than a git call that failed. So the absence is
    recorded as an explicit, greppable sentinel that says WHICH of the two it is.
    """
    has_repo = os.path.isdir(os.path.join(REPO, ".git"))
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        commit = ("<no git repository: published or flattened tree. THIS DOES NOT AFFECT "
                  "REPRODUCTION — see the reproducibility appendix, 'What reproduces from the "
                  "flattened tree'. Every result is reproduced from the stored inputs and the "
                  "script hash recorded alongside this field, neither of which needs git.>"
                  if not has_repo else
                  "<git present but rev-parse failed: unexpected, and NOT the published-tree case. "
                  "Reproduction is unaffected, but this box's git is broken and should be checked.>")
    status = _git("status", "--porcelain")
    diff = _git("diff", "HEAD")
    dirty = bool(status)
    return {
        "commit": commit,
        "commit_short": commit[:12] if commit and not commit.startswith("<") else commit,
        "git_repository_present": has_repo,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
        # the full diff is not stored (it can be huge and can contain licensed text); its hash is
        # enough to tell two dirty states apart, and to prove a rerun used the same edits
        "dirty_files": sorted(l[3:] for l in status.splitlines())[:64] if status else [],
        "dirty_diff_sha256": sha256_bytes(diff.encode()) if dirty and diff else None,
        "describe": _git("describe", "--always", "--dirty"),
    }


def hf_model_state(model_id: str, revision: str | None = None, hf_model=None) -> dict:
    """The resolved Hugging Face commit for a model, read from the local cache. No network.

    Nothing in the fitting path pins a model revision: `from_pretrained(MODEL_ID)` takes whatever
    `main` happens to be. Weights are stable in practice, but "in practice" is not a provenance
    record, and an operator fitted against a silently updated checkpoint would be unreproducible
    with nothing in the file to say so.

    The cache alone cannot answer this: `pythia-410m-deduped` has 20 snapshots cached here because
    E65 pulls per-step revisions, so "which one loaded" is only knowable from the loaded object.
    `config._commit_hash` is what `from_pretrained` actually resolved, so that is authoritative and
    everything else is a fallback that says so.
    """
    out = {"model_id": model_id, "revision_requested": revision or "main (unpinned)"}
    try:
        cfg = getattr(hf_model, "config", None)
        resolved = getattr(cfg, "_commit_hash", None) if cfg is not None else None
        out["revision_resolved"] = resolved or "<unresolved: no _commit_hash on the loaded config>"
    except Exception as e:
        out["revision_resolved"] = f"<unresolved: {type(e).__name__}>"
    return out


def env_state() -> dict:
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    tf32 = None
    threads = None
    try:
        import torch
        tf32 = {"allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE")}
        # THREAD COUNT CHANGES THE ANSWER, so it is provenance, not trivia. e60_fitter_determinism
        # measured the two fitters agreeing EXACTLY at 1, 4 and 8 threads and diverging by 1.09e-04
        # above that -- the same order as the released fitter's disagreement with itself. A result
        # that does not record how many threads produced it cannot be reproduced deliberately, only
        # by luck. This was missing until 2026-08-29.
        threads = {"torch_num_threads": torch.get_num_threads(),
                   "torch_num_interop_threads": torch.get_num_interop_threads(),
                   "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                   "os_cpu_count": os.cpu_count()}
    except Exception:
        pass
    # NO HOSTNAME. `platform.node()` was recorded here until 2026-08-31 and it was a standing
    # liability, not a reproducibility aid: nothing in this repository ever read the value, while
    # every write stamped a machine name into the results file. Because a redaction is a
    # property of the FILE and not of this code, each rerun -- and each `04_fetch_results.sh` fetch,
    # which overwrites local files -- put the name straight back. Measured: a T0 sweep re-leaked 12
    # files minutes after a clean H12, twice; and a background fetch once overwrote 58 already
    # redacted files with the mirror's copies, with nothing warning.
    #
    # `platform.platform()` and `machine` stay: "macOS-26.5.2-arm64-arm-64bit-Mach-O" and "arm64"
    # describe the ENVIRONMENT, which is load-bearing for reproduction, and identify nobody.
    # tools/anonymize_release.py still redacts a `hostname` key, deliberately: files written before
    # this change, or fetched from a mirror that holds pre-change copies, still carry one.
    return {"python": sys.version.split()[0], "torch": ver("torch"),
            "transformers": ver("transformers"), "numpy": ver("numpy"),
            "platform": platform.platform(), "machine": platform.machine(),
            "tf32": tf32, "threads": threads}


def _str_keys(o):
    """Recursively coerce every dict key to `str`. R4f / finding M-2.

    THE DEFECT. `json.dumps(..., sort_keys=True)` sorts INT keys numerically and STR keys
    lexicographically, and it stringifies int keys only AFTER sorting. A payload keyed by layer
    number therefore serialises as 9,10,...,21 in memory and as "10","11",...,"21","9" once it has
    been through `json.load`. The two byte streams differ, so the stored hash does not verify
    against the file it was stored in -- which is the one thing the hash exists to do.

    Measured: of 29 results files carrying a `payload_sha256`, exactly one fails to verify on a
    round-trip -- `results/e58_algebra_audit.json`, whose
    `A_e45_orientation.G_relative_asymmetry_per_layer` is keyed 9..21. Re-keying that payload to
    int reproduces the stored hash EXACTLY, which is what identifies the mechanism rather than
    merely correlating with it.
    """
    if isinstance(o, dict):
        return {str(k): _str_keys(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_str_keys(v) for v in o]
    return o


def legacy_payload_sha(payload: dict) -> str:
    """The pre-R4f rule, kept so the 29 hashes already on disk stay checkable.

    Do not use for new results. `verify_payload_sha` reports which rule a file was written under.
    """
    body = {k: v for k, v in payload.items() if k != "provenance"}
    return sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                   default=str).encode())


def canonical_payload_sha(payload: dict) -> str:
    """SHA-256 of the result CONTENT, provenance block excluded.

    This is the number to compare across reruns. Including the timestamp (or the hostname, or a
    dirty-diff hash) would make every run differ and the check would mean nothing.

    Keys are coerced to `str` first (R4f), so the hash is invariant under a JSON round-trip. That
    invariance is the whole point: a hash that changes when you read the file back cannot verify
    the file.
    """
    body = {k: v for k, v in payload.items() if k != "provenance"}
    return sha256_bytes(json.dumps(_str_keys(body), sort_keys=True, separators=(",", ":"),
                                   default=str).encode())


def verify_payload_sha(payload: dict) -> dict:
    """Check a loaded results payload against its stored hash under both rules.

    Returns {'stored', 'canonical', 'legacy', 'rule'} where `rule` is 'canonical' (written after
    R4f), 'legacy' (written before it, and still valid under the rule in force at the time),
    'absent', or 'MISMATCH' -- which is a real failure and not a convention question.
    """
    prov = payload.get("provenance") or {}
    stored = prov.get("payload_sha256") if isinstance(prov, dict) else None
    canon, leg = canonical_payload_sha(payload), legacy_payload_sha(payload)
    rule = ("absent" if not stored else
            "canonical" if stored == canon else
            "legacy" if stored == leg else "MISMATCH")
    return {"stored": stored, "canonical": canon, "legacy": leg, "rule": rule}


def _rel_argv0(argv: list[str]) -> list[str]:
    """argv with argv[0] made repo-relative when it points inside the repo.

    Everything after argv[0] is left byte-identical: `--band 9,21` is not cosmetic, and a rewriter
    that touched flags would corrupt the record it exists to protect.
    """
    if not argv:
        return list(argv)
    a0 = argv[0]
    try:
        ap = os.path.abspath(a0)
        if os.path.commonpath([ap, REPO]) == REPO:
            a0 = os.path.relpath(ap, REPO)
    except (ValueError, OSError):
        pass
    return [a0, *argv[1:]]


def stamp(experiment: str, *, script: str | None = None, inputs=(), extra: dict | None = None,
          payload: dict | None = None) -> dict:
    """The provenance record. `inputs` are paths whose SHA-256 becomes part of the chain."""
    # A --help run measures nothing, so a results file that records --help as the command that
    # produced it is a false receipt. Two files carried exactly that (e48c, which is the source of
    # the paper's 10,862x containment claim, and e33b), and a repro module generated from that argv
    # would print usage and produce no output. Refuse to stamp rather than record a lie.
    if any(a in ("--help", "-h") for a in sys.argv[1:]):
        raise SystemExit(
            "provenance.stamp() refused: --help/-h is in argv, so this process is not a measurement.\n"
            "A results file stamped from a help run records a command that cannot reproduce it.")
    script = os.path.abspath(script or sys.argv[0])
    rel = os.path.relpath(script, REPO) if os.path.exists(script) else script
    rec = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": {"path": rel, "sha256": sha256_file(script),
                   "n_lines": sum(1 for _ in open(script, "rb")) if os.path.isfile(script) else None},
        "git": git_state(),
        # argv[0] is normalised to a repo-relative path. Stored verbatim it was an absolute path
        # under the operator's home directory, which is both a double-blind leak and unusable in a
        # generated repro module: cv7_1b_rung.json recorded /Users/<name>/... as its invocation.
        # Only argv[0] is rewritten; every flag is preserved exactly, because flags are the science.
        "argv": _rel_argv0(sys.argv),
        "cwd": os.path.relpath(os.getcwd(), REPO),
        "inputs": [{"path": os.path.relpath(os.path.abspath(p), REPO),
                    "sha256": sha256_file(p),
                    "bytes": os.path.getsize(p) if os.path.isfile(p) else None,
                    "exists": os.path.isfile(p)} for p in inputs],
        "env": env_state(),
    }
    if payload is not None:
        rec["payload_sha256"] = canonical_payload_sha(payload)
    if extra:
        rec["extra"] = extra
    return rec


def write_result(path: str, payload: dict, *, experiment: str, inputs=(),
                 extra: dict | None = None, script: str | None = None, indent: int = 1) -> dict:
    """Attach provenance and write ATOMICALLY.

    Atomic because a results file half-written by a crashed run is worse than no file: it looks
    valid to every downstream reader. tmp-then-rename on the same filesystem cannot tear.
    """
    payload = dict(payload)
    payload.pop("provenance", None)
    payload["provenance"] = stamp(experiment, script=script, inputs=inputs, extra=extra,
                                  payload=payload)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=indent, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    p = payload["provenance"]
    print(f"  wrote {os.path.relpath(path, REPO)}\n"
          f"    payload sha256 {p['payload_sha256'][:16]}…  "
          f"script {p['script']['sha256'][:12] if p['script']['sha256'] else '?'}…  "
          f"commit {p['git']['commit_short']}{' DIRTY' if p['git']['dirty'] else ''}", flush=True)
    return payload


def verify_result(path: str) -> dict:
    """Re-check a stored record against the tree as it is NOW. Reports, never mutates."""
    try:
        d = json.load(open(path))
    except Exception as e:
        return {"path": path, "ok": False, "reason": f"unreadable: {e}"}
    p = d.get("provenance")
    if not p:
        return {"path": path, "ok": False, "reason": "no provenance block",
                "has_provenance": False}
    out = {"path": path, "has_provenance": True, "experiment": p.get("experiment"),
           "commit": p.get("git", {}).get("commit_short"),
           "dirty_when_written": p.get("git", {}).get("dirty")}
    # A legacy record may carry a provenance block written before `script` was stamped (11 Arc-1
    # files do: t20_*, t22_*, en1_*). Report that as a finding; do NOT raise. A KeyError here
    # aborts the whole --verify sweep partway, which reads as "the tree is broken" when the real
    # state is "one old file predates a schema field" -- and it hides every file after it.
    if not isinstance(p.get("script"), dict) or "path" not in p["script"]:
        # A legacy record may carry a provenance block written before `script` was stamped (11
        # Arc-1 files do: t20_*, t22_*, en1_*). Report it; do NOT raise. A KeyError here aborted
        # the whole --verify sweep partway, which reads as "the tree is broken" when the real
        # state is "one old file predates a schema field" -- and it hid every file after it.
        # The payload and input checks below still run and still mean something.
        out["legacy_no_script"] = True
        out["script_exists"] = None
        out["script_unchanged"] = False
        out["script_unchanged_reason"] = "provenance block predates the `script` field"
    else:
        sp = os.path.join(REPO, p["script"]["path"])
        now = sha256_file(sp)
        recorded = p["script"]["sha256"]
        out["script_exists"] = now is not None
        # None == None must NOT read as "unchanged". A check that passes because it could not run
        # is the failure mode CLAUDE.md 6.0b names: never accept a proxy for the thing you care
        # about.
        out["script_unchanged"] = bool(now is not None and recorded is not None and now == recorded)
        if now is None or recorded is None:
            out["script_unchanged_reason"] = ("script file not found now" if now is None
                                              else "no script hash was recorded at write time")
    out["payload_sha_matches"] = (canonical_payload_sha(d) == p.get("payload_sha256"))
    bad = []
    for i in p.get("inputs", []):
        cur = sha256_file(os.path.join(REPO, i["path"]))
        if cur is None:
            bad.append({"path": i["path"], "why": "missing now"})
        elif cur != i["sha256"]:
            bad.append({"path": i["path"], "why": "changed since the run"})
    out["inputs_checked"] = len(p.get("inputs", []))
    out["inputs_changed_or_missing"] = bad
    out["ok"] = bool(out["script_unchanged"] and out["payload_sha_matches"] and not bad)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="inspect or verify a results file's provenance")
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    for p in a.paths:
        r = verify_result(p)
        print(json.dumps(r, indent=1))
