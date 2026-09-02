#!/usr/bin/env python3
"""
emit_exp_modules.py - generate a repro/exp/ module from what a results file already records.

WHY. repro/exp/README.md says each module is "THE canonical invocation ... with the flags that
produced the stored result". Eleven experiments had one; twenty-three paper-cited results did not.
That was never a lost-information problem - `provenance.argv`, `provenance.script` and
`provenance.inputs` are in every stamped results file - it was an ergonomics and guard problem: a
reviewer had to read JSON to find the command, and nothing enforced inputs, exit codes or outputs.

So generate the modules from the record instead of hand-writing them. Faithful by construction: the
argv is the argv that ran, not someone's memory of it.

    .venv/bin/python tools/emit_exp_modules.py --list      # what is missing a module
    .venv/bin/python tools/emit_exp_modules.py --write     # emit them

A generated module carries a GENERATED banner and says which results file it was derived from.
Hand-written modules are never overwritten.
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "repro", "exp")

# --- SCOPE ---------------------------------------------------------------------------------------
# The scope is DERIVED, not listed: every results file named by the paper, by its table generator,
# or by its figure generator needs a module.
#
# It used to be a hardcoded dict of 17, and that dict WAS the coverage: `--list` reported
# "0 modules would be written" while thirteen cited-or-used results had no module at all. A list
# cannot report a gap it does not know about. Deriving the scope from the paper means a new citation
# cannot arrive without a module -- the tool goes red on its own the moment the paper cites a file
# nothing can regenerate.
#
# The paper names its results files in the register table as \texttt{name} with LaTeX-escaped
# underscores; the table generator names them as load("name.json").
PAPER = os.path.join(ROOT, "paper", "audit_paper_5pp.tex")
GENERATORS = [os.path.join(ROOT, "paper", "tools", "make_audit_tables.py"),
              os.path.join(ROOT, "tools", "build_config_matrix.py")]

# Module id and one-line "why" for the results we have already named. Anything derived that is not
# in here gets an id from its stem and an explicit UNSTATED why, which is a visible prompt to write
# one rather than a silent default.
KNOWN = {
    "cv1_answer_competence":            ("cv1_competence",   "can the model do the task at all?"),
    "cv2_position_support":             ("cv2_position",     "is the read inside the Jacobian's fitting support?"),
    "cv3_margins_410m":                 ("cv3_margins",      "score effect or ranking artifact?"),
    "cv4_phase1_capability":            ("cv4_capability",   "answer competence across the ladder"),
    "cv5_metric_sensitivity_410m":      ("cv5_sensitivity",  "was rank blind where z is sighted?"),
    "cv6_per_family_ladder":            ("cv6_ladder",       "does the per-family corpus effect replicate at 1.4B and 2.8B?"),
    "cv7_1b_rung":                      ("cv7_rung",         "the 1B rung of the per-family ladder"),
    "d1_min_union_diagnostic_410m":     ("d1_union",         "is min's preference for a derangement a union artifact?"),
    "da1_derangement_adjudication_410m": ("da1_derangement", "what does the layer-derangement control actually identify?"),
    "r10_t59_readout_reconciliation":   ("r10_t59_readout",  "is the stored E59 dose ladder legacy-scored?"),
    "d3_corpus_by_family_410m":         ("d3_by_family",     "the 410M per-family corpus decomposition"),
    "dn1_design_nulls":                 ("dn1_design_nulls", "does the design null agree with the permutation null?"),
    "e33_logit_baseline_410m_v2":       ("e33_logit_v2",     "the free-lens baseline at the corrected readout"),
    "e37_rank_ablation_70m_wikitext":   ("e37_rank_70m",     "is rank the cause of the small-model floor? (70M)"),
    "e37_rank_ablation_160m_wikitext":  ("e37_rank_160m",    "is rank the cause of the small-model floor? (160M)"),
    "e38_jgeometry":                    ("e38_geometry",     "the operator geometry behind the 160M precondition"),
    "e49_derangement_stability":        ("e49_stability",    "how much is one derangement draw worth?"),
    "e54_aggregation_audit_rstrip_v2":  ("e54_aggregation",  "the derangement audit at the corrected readout"),
    "e55_matrix_robustness_rstrip":     ("e55_matrix",       "does the matching effect replicate on an independent operator population?"),
    "e57_grid_variance_ci_rstrip":      ("e57_variance_ci",  "an interval on the headline variance decomposition"),
    "e58_algebra_audit":                ("e58_algebra",      "algebra and measurement checks the validity audit left open"),
    "e59_read_dose_410m":               ("e59_read_dose",    "is the fit/read asymmetry a dose effect?"),
    "e61_randomized_null_410m":         ("e61_adjudicate",   "the randomised-network null, adjudicated"),
    "e65_ckpt_geometry_410m":           ("e65_ckpt_geometry","trained small model, or undertrained?"),
    "os1_operator_space_410m":          ("os1_operator_space","how far apart are the fitted operators?"),
    "paper_clustered_derangement_t":    ("paper_clustered_t","the corpus-clustered t on the derangement gap"),
    "r5_corpus_axis_uncertainty_rstrip":("r5_corpus_axis",   "corpus-axis uncertainty, stated as such"),
    "r6_within_source_410m":            ("r6_within_source", "corpus identity, or sampling within a corpus?"),
    "r7_matched_pools_410m":            ("r7_matched_pools", "does the effect survive matching the pools on lexical composition?"),
    "r8_ladder_flatness":               ("r8_flatness",      "does S2 survive at the corrected readout?"),
    "r9_permutation_calibrated_min":    ("r9_calibrated_min","the published statistic against its own null"),
}


def _results_stems():
    import glob as _g
    out = set()
    for p in _g.glob(os.path.join(ROOT, "results", "**", "*.json"), recursive=True):
        out.add(os.path.basename(p)[:-5])
    return out


def derive_targets():
    """Every results file the paper or its generators name. The register is the scope."""
    import re, glob as _g
    stems = _results_stems()
    found = set()
    if os.path.exists(PAPER):
        tex = open(PAPER, encoding="utf-8", errors="ignore").read()
        for m in re.findall(r"\\texttt\{([^}]*)\}", tex):
            n = m.replace("\\_", "_").strip()
            if n in stems:
                found.add(n)
        for n in re.findall(r"results/([A-Za-z0-9_/.\-]+)\.json", tex):
            if n in stems:
                found.add(n)
    for g in GENERATORS:
        if not os.path.exists(g):
            continue
        src = open(g, encoding="utf-8", errors="ignore").read()
        # Only an actual load counts. Matching every quoted "*.json" in the source pulled in six
        # files the generator merely mentions in a comment or writes as an alternate output, and a
        # scope that over-reports is as useless as one that under-reports: it trains you to ignore it.
        for n in re.findall(r'load\(\s*["\']([A-Za-z0-9_/.\-]+)\.json["\']', src):
            n = n.split("/")[-1]
            if n in stems:
                found.add(n)
    tg, used = {}, set()
    for n in sorted(found):
        if n in KNOWN:
            mid, why = KNOWN[n]
        else:
            bits = n.split("_")
            mid = "_".join(bits[:2]) if len(bits) > 1 else n
            why = "WHY UNSTATED — add it to KNOWN"
        # Two results files can shorten to the same id (e55_matrix_robustness and its _rstrip
        # sibling both give "e55_matrix"). Silently colliding would make one module overwrite the
        # other, so fall back to the full stem, which is unique by construction.
        if mid in used:
            mid = n
        used.add(mid)
        tg[n] = (mid, why)
    return tg


def covered_results():
    """results paths already claimed by an existing module's OUTPUTS. Exact, not substring."""
    import re, glob as _g
    out = set()
    for f in _g.glob(os.path.join(EXP, "*.sh")):
        src = open(f, encoding="utf-8", errors="ignore").read()
        for blk in re.findall(r"OUTPUTS=\(([^)]*)\)", src):
            for tok in blk.split():
                tok = tok.strip('"\'')
                if tok.startswith("results/") and tok.endswith(".json"):
                    out.add(os.path.basename(tok)[:-5])
    return out

TEMPLATE = '''#!/usr/bin/env bash
# {mid} — {why}
#
# GENERATED by tools/emit_exp_modules.py from results/{res}.json. Do not hand-edit; regenerate.
# The invocation below is `provenance.argv` of that file: the command that actually produced the
# stored number, not a reconstruction. INPUTS are `provenance.inputs`.
#
# Regenerating will NOT reproduce the payload byte-for-byte where the script has changed since;
# `docs/reproducibility/SCRIPT_PROVENANCE.md` names the exact code each result was produced by, and
# `repro/scripts_at_run/` carries it.
#
# PATHS IN THE INVOCATION BELOW ARE REPO-RELATIVE. The raw record may hold an absolute path from
# the machine that ran it -- a home directory, or a rented GPU box's /workspace that no longer
# exists. Only paths were rewritten; every flag is byte-identical to `provenance.argv`.
source "$(dirname "${{BASH_SOURCE[0]}}")/_lib.sh"
MODULE_ID="{mid}"; MODULE_TITLE="{title}"
MODULE_WHY="{why}"
MODULE_COST="{cost}"
MODULE_TIER="{tier}"
INPUTS=({inputs})
OUTPUTS=(results/{res}.json)
main() {{ run_py {cmd}; }}
module_main "$@"
'''


def _dir_has_pt(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.isdir(p):
        return False
    return any(f.endswith(".pt") for _, _, fs in os.walk(p) for f in fs)


def _loads_model(script_rel):
    """Does this script pull a model off the hub? That is the T0/T1 line."""
    import re
    p = os.path.join(ROOT, script_rel)
    if not os.path.isfile(p):
        p = os.path.join(ROOT, "experiments", os.path.basename(script_rel))
    if not os.path.isfile(p):
        return False
    return bool(re.search(r"from_pretrained|GPTNeoXFor|AutoModel",
                          open(p, encoding="utf-8", errors="ignore").read()))


def portable_argv(argv):
    """The stored argv, rewritten so a reviewer can actually run it from a fresh clone.

    Three classes of unrunnable path are in the stored record, and none of them is a wrong number:

      1. an absolute path under a home directory  (cv7_1b_rung recorded /Users/<name>/...).
         It is a double-blind leak AND it hardcodes a directory nobody else has.
      2. an absolute path under a rented GPU box's /workspace  (r6, r7). The box is long destroyed.
      3. a bare script name with no directory  (r6, r7 again), which resolves only if you happen to
         be standing in experiments/.

    Flags are never touched, and neither is any value that is not a path: `--band 9,21` is science.
    The rewrite is recorded in the module header so the difference from the raw record is visible.
    """
    out = []
    for i, tok in enumerate(argv):
        if tok in ("--help", "-h"):
            continue
        t = tok
        if t.startswith("/workspace/"):
            t = t[len("/workspace/"):]
        elif os.path.isabs(t):
            try:
                if os.path.commonpath([t, ROOT]) == ROOT:
                    t = os.path.relpath(t, ROOT)
            except (ValueError, OSError):
                pass
        if i == 0 and "/" not in t and t.endswith(".py"):
            if os.path.exists(os.path.join(ROOT, "experiments", t)):
                t = f"experiments/{t}"
        out.append(t)
    return out


def load(res):
    p = os.path.join(ROOT, "results", res + ".json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    targets = derive_targets()
    existing = {f[:-3] for f in os.listdir(EXP) if f.endswith(".sh")}
    claimed = covered_results()
    made, skipped = [], []
    for res, (mid, why) in sorted(targets.items()):
        # "already covered" is decided on the OUTPUTS a module declares, not on its filename.
        # Matching on filename hid two real gaps: repro/exp/e61_randomized_null.sh fits the
        # trainval operators and does NOT emit e61_randomized_null_410m.json, and cv7_1b.sh is a
        # detached launcher that never sources _lib.sh, so it makes none of the harness guarantees.
        if res in claimed:
            skipped.append((mid, "already an OUTPUT of an existing module")); continue
        if mid in existing:
            skipped.append((mid, f"repro/exp/{mid}.sh exists but does not declare results/{res}.json"))
            continue
        o = load(res)
        if o is None:
            skipped.append((mid, f"results/{res}.json absent")); continue
        pr = o.get("provenance") or {}
        argv = pr.get("argv")
        if not isinstance(argv, list) or not argv:
            skipped.append((mid, "no provenance.argv — HAND-WRITE this one")); continue
        cmd = " ".join(portable_argv(argv))
        # argv[0] is the script; run_py takes it relative to the repo root
        ins = []
        for i in (pr.get("inputs") or []):
            p_ = i.get("path") if isinstance(i, dict) else None
            if p_ and not p_.startswith(("/", "<")):
                ins.append(p_)
        ins = sorted(set(ins))[:6]
        # `free` is what _lib.sh's budget gate recognises (it matches `free` or `free *`).
        # A recomputation over stored JSON, or anything explicitly --device cpu, spends nothing.
        recompute = argv[0].startswith("tools/") or bool(o.get("recomputes_not_remeasures"))
        plat = str((pr.get("env") or {}).get("platform", ""))
        # A run is free if it recomputes from stored JSON, or was explicitly --device cpu, or ran on
        # the laptop (macOS => no CUDA in this programme; repro/01_setup_local.sh prints so).
        # Getting this wrong is not cosmetic: an unrecognised cost string makes _lib.sh demand --yes
        # and the module cannot run unattended, which is how four generated modules failed first time.
        oncpu = ("--device cpu" in cmd or str(o.get("device", "")).startswith("cpu")
                 or plat.startswith(("macOS", "Darwin")))
        cost = ("free (recomputation over stored results)" if recompute
                else "free (CPU)" if oncpu
                else "GPU — price with ./lab cost before running")
        # TIER is what a reviewer sorts on, and it answers a different question from COST.
        #   T0  pure recomputation over stored JSON — no model, no operator, no GPU. Seconds.
        #   T1  scores stored .pt operators against the battery on CPU. Minutes to hours.
        #   T2  fits operators. GPU, fp32, TF32 off. Hours and dollars.
        # A reviewer with no GPU and no patience can still run every T0 and see the statistics
        # reproduce, which is the promise repro/RUN_ALL.sh makes.
        #
        # Decide it on EVIDENCE, not on the platform the run happened to use. Keying T1 off "this
        # ran on macOS" mislabelled dn1 and e61: both are pure JSON recomputations that happened to
        # run on the laptop, and a reviewer told they need the 7.7 GB artifact mirror to run them
        # would not bother. What actually separates the tiers is whether a model gets loaded and
        # whether any declared input is a `.pt`.
        needs_pt = any(p_.endswith(".pt") or _dir_has_pt(p_) for p_ in ins)
        loads_model = _loads_model(argv[0])
        tier = "T2" if not oncpu else ("T1" if (needs_pt or loads_model) else "T0")
        body = TEMPLATE.format(mid=mid, why=why, res=res, title=why[:48],
                               cost=cost, tier=tier,
                               inputs=" ".join(ins) if ins else "",
                               cmd=cmd)
        made.append((mid, res, cmd))
        if a.write:
            f = os.path.join(EXP, mid + ".sh")
            open(f, "w").write(body)
            os.chmod(f, 0o755)

    print(f"  {len(made)} module(s) {'written' if a.write else 'would be written'}")
    for mid, res, cmd in made:
        print(f"      {mid:20s} <- {res}")
        print(f"        {cmd[:110]}")
    if skipped:
        print(f"\n  {len(skipped)} skipped:")
        for mid, why in skipped:
            print(f"      {mid:20s} {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
