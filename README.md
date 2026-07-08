# odor-search-analysis

Analysis pipeline for **odor-driven *Drosophila* local-search assays** — SLEAP
pose estimation turned into trajectories, kinematics, odor-approach behavior, and
statistics.

This package is a uv-managed rebuild of the original `extinction_tracking/exp003`
scripts and shared `lib/`. See `../pipeline_and_plan.html` for the full pipeline
explainer and the phased implementation plan.

## Status

**Phase 0 — scaffold.** The package structure, dependencies, and CLI surface are
in place; the shared algorithms (`detection`, `tracking`, `io`, `diagnostics`,
`constants`) have been moved into `src/odor_search/`. Individual pipeline stages
are stubs, filled in over Phases 1–10.

## Quickstart

```bash
uv sync                     # create .venv and install (incl. dev tools)
uv run odor-search --help   # see the pipeline stages
uv run pytest               # run the smoke tests
```

## Layout

```
src/odor_search/     # the package (importable as `odor_search`)
  constants.py       # arena/node names, skeleton, calibration constants
  detection.py       # arena + cylinder detection and mm-calibration
  tracking.py        # arena assignment, trajectory cleaning, Schmitt triggers
  io.py              # SLEAP loading, frame timing, prediction discovery
  diagnostics.py     # detection QC figures
  cli.py             # `odor-search` command-line entry point
configs/             # per-experiment config TOMLs (Phase 1)
tests/               # unit + smoke tests
data/                # raw data (gitignored)
runs/                # per-run outputs + manifest.json (gitignored)
```

## Pipeline stages

`detect → track → qc → kinematics → behavior → stats → temporal → figures`
(or `all`). Each becomes an `odor-search <stage>` subcommand that reads a config
and writes into a per-run directory.
