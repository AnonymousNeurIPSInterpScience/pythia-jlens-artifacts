#!/usr/bin/env python3
"""
operator_probe.py — GROUND TRUTH on whether a run actually reads operator files.

WHY THIS EXISTS. Classifying a module by reading its source is inference, and inference has been
wrong twice here: once by asking "does the script mention from_pretrained" (which counted
`--adjudicate` runs that never reach it), and once by asking "does the results file record .pt
inputs" (which missed scripts that load operators without stamping them). Both were proxies.

This is not a proxy. It installs a CPython audit hook, runs the target script for real, and records
every filesystem open of a `.pt`. If the process reads an operator, this sees it. If it does not,
there is nothing to see. There is no reading of source and no guessing about which branch a flag
takes.

    .venv/bin/python tools/operator_probe.py experiments/cv7_1b_rung.py --adjudicate

Everything after the script path is passed through to the script unchanged. The report goes to
stderr so it cannot be confused with the script's own stdout.

COST. The probe adds an audit hook, which is a small per-open overhead and nothing else. It does not
change numerics, so a probed run and an unprobed run compute the same thing -- but do NOT use a
probed run as a reproduction check, because the point of a reproduction check is to run the module
exactly as the harness runs it. Probe to classify; run the module to reproduce.
"""
import os
import runpy
import sys

_OPENED: list[str] = []
_MODELS: list[str] = []


def _hook(event: str, args) -> None:
    # "open" fires for every file the interpreter opens, including inside torch and safetensors.
    if event == "open" and args:
        p = args[0]
        if isinstance(p, (str, bytes, os.PathLike)):
            s = os.fspath(p) if not isinstance(p, bytes) else p.decode("utf-8", "replace")
            low = s.lower()
            if low.endswith(".pt"):
                if s not in _OPENED:
                    _OPENED.append(s)
            elif low.endswith((".safetensors", ".bin")) and "snapshots" in low:
                if s not in _MODELS:
                    _MODELS.append(s)


def _report() -> None:
    e = sys.stderr
    print("\n" + "=" * 72, file=e)
    print("  OPERATOR PROBE — what this run actually opened", file=e)
    print("=" * 72, file=e)
    print(f"  operator files (.pt) opened : {len(_OPENED)}", file=e)
    for p in _OPENED[:12]:
        print(f"      {p}", file=e)
    if len(_OPENED) > 12:
        print(f"      ... and {len(_OPENED) - 12} more", file=e)
    print(f"  model weight files opened   : {len(_MODELS)}", file=e)
    for p in _MODELS[:4]:
        print(f"      {p}", file=e)

    if _OPENED:
        cls = ("TYPE 1 — OPERATOR REPRODUCTION. This run read the stored operators off disk and "
               "re-derived its numbers from them.")
    elif _MODELS:
        cls = ("TYPE 2 — MODEL MEASUREMENT. This run loaded the model but read no operator. Its "
               "quantity does not involve a lens.")
    else:
        cls = ("TYPE 3 — RE-AGGREGATION. This run read neither an operator nor model weights. It "
               "recomputed from numbers already stored on disk.")
    print(f"\n  {cls}", file=e)
    print("=" * 72, file=e)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: operator_probe.py <script.py> [args passed to the script...]")
    target = sys.argv[1]
    if not os.path.isfile(target):
        raise SystemExit(f"no such script: {target}")
    sys.argv = [target, *sys.argv[2:]]
    sys.addaudithook(_hook)
    code = 0
    try:
        runpy.run_path(target, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        _report()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
