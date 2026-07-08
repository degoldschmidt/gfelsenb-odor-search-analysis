# configs/

Per-experiment configuration files (added in **Phase 1**).

Each stage of the pipeline reads one config (e.g. `exp003.toml`) that declares
data paths, detection/cleaning/behavior thresholds, the fixed odor map, the
skip-list of truncated videos, and the temporal-analysis bins. Keeping one
config per experiment (exp002, exp003, …) is what lets cross-experiment
comparisons be defined consistently rather than by ad-hoc constants scattered
across scripts.

Usage (once implemented):

```bash
uv run odor-search all --config configs/exp003.toml --run 2026-07-08
```
