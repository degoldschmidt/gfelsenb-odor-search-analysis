"""Command-line entry point for the odor_search analysis pipeline.

Two kinds of command:

* ``input <path>`` — a pre-flight validator for a raw-data location. This is
  implemented (see :mod:`odor_search.validate`).
* the pipeline **stages** (``detect``, ``track``, ``qc``, …) — Phase-0 stubs
  that will be filled in during later phases. Each takes ``--config``/``--run``.

Run ``odor-search --help`` to see everything.
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
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # Pre-flight input validation (implemented).
    ip = sub.add_parser(
        "input", help="Validate a raw-data location before running the pipeline"
    )
    ip.add_argument("path", help="Directory of recordings (or a single .avi) to validate")
    ip.add_argument(
        "--predictions-dir", metavar="DIR",
        help="Where to look for .predictions.slp if not beside the video",
    )
    ip.add_argument(
        "--time-column", default="Item5",
        help="Elapsed-time column expected in the timing CSV (default: Item5)",
    )
    ip.add_argument(
        "--min-frames", type=int, default=100,
        help="Warn if an .slp has fewer labeled frames (default: 100)",
    )
    ip.add_argument("--json", action="store_true", help="Emit a JSON report")
    ip.add_argument(
        "--color", choices=("auto", "always", "never"), default="auto",
        help="Colorize output: green=valid, yellow=warning, red=error "
             "(default: auto = only when writing to a terminal).",
    )

    # Pipeline stages (stubs for now).
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


def _resolve_color(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    import os  # auto: color only for an interactive terminal, honoring NO_COLOR.

    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _run_input(args: argparse.Namespace) -> int:
    from .validate import format_report, report_to_json, validate_input

    report = validate_input(
        args.path,
        predictions_dir=args.predictions_dir,
        time_column=args.time_column,
        min_slp_frames=args.min_frames,
    )
    if args.json:
        print(report_to_json(report))
    else:
        print(format_report(report, color=_resolve_color(args.color)))
    return 0 if report.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "input":
        return _run_input(args)

    print(
        f"[odor-search] stage '{args.command}' is not implemented yet "
        f"(Phase-0 scaffold). config={args.config!r} run={args.run!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
