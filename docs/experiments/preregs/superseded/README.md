# `preregs/superseded/` — the original pre-registrations, kept because results files pin them

These six documents were superseded by the live pre-registrations in the parent directory. They are
**not history for its own sake**. Each is named, by path and by SHA-256, inside a stored results
file, so deleting them would break the integrity chain that `prereg_sha256` exists to provide:

| document | pinned by |
|---|---|
| `PREREG_E52_FACTORIAL.md` | `e52_factorial_410m*.json` (5 files), sha `740ddb1d6aeb` |
| `PREREG_E36_QLADDER.md` | `e36_qladder_410m*.json` |
| `PREREG_E48_CROSSOVER.md` | `e48_crossover_410m*.json` |
| `PREREG_E38_JGEOMETRY.md` | `e38_jgeometry.json` |
| `PREREG_PYTHIA_T7_v2.md`, `PREREG_PYTHIA_T7.md` | the T-series |

The recorded paths inside those results files are the ones that were true when the experiment ran
(`specs/archive/prereg/…`, `pythia/specs/…`). Those are **payload** keys, so repathing them would
move `payload_sha256` and break every downstream integrity check. They are therefore left exactly as
written.

The bytes **as registered** — which for ten R-series documents differ from the file on disk, because
the 2026-08-22 repath rewrote paths inside them — are frozen in `../as_registered/`.

Frozen. Do not edit, including to fix a stale number: a stale number inside a pre-registration is
correct behaviour under the visible-retraction discipline.
