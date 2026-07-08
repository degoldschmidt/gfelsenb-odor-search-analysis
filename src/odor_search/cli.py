"""Command-line entry point for the odor_search analysis pipeline.

This is the Phase-0 scaffold: the argument surface (stages, ``--config``,
``--run``) is wired up, but each stage is a stub that will be filled in during
later phases (detection, tracking, QC, kinematics, behavior, stats, temporal,
figures). Run ``odor-search --help`` to see the available commands.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__

# Pipeline stages, in execution order. ``all`` runs the whole chain.
STAGES: tuple[str, ...] = (
    "detect",      # arena + cylinder detection and calibration
    "track",       # arena assignment + trajectory cleaning
    "qc",          # mistracking detection (cylinder-lock + jump) and exclusion
    "kinematics",  # displacement, speed, spike removal
    "behavior",    # approach + border segmentation (Schmitt triggers)
    "stats",       # group comparisons
    "temporal",    # extinction / decay over time
    "figures",     # publication figures + diagnostics
    "all",         # run every stage above, in order
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odor-search",
        description="Analysis pipeline for odor-driven Drosophila local-search assays.",
    )
    parser.add_argument(
        "--version", action="version", version=f"odor-search {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="STAGE")
    for name in STAGES:
        sp = sub.add_parser(name, help=f"Run the {name} stage")
        sp.add_argument(
            "--config", metavar="TOML",
            help="Path to the experiment config (added in Phase 1).",
        )
        sp.add_argument(
            "--run", metavar="RUN_ID",
            help="Run identifier; artifacts land under runs/<RUN_ID>/.",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    print(
        f"[odor-search] stage '{args.command}' is not implemented yet "
        f"(Phase-0 scaffold). config={args.config!r} run={args.run!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
