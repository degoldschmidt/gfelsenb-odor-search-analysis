"""Phase-0 smoke tests: the package installs, imports, and the CLI runs.

These lock in the two Phase-0 acceptance criteria: ``import odor_search`` works
from anywhere (no ``sys.path`` hacks), and the moved ``lib`` modules import as a
proper package.
"""

from __future__ import annotations

import odor_search
from odor_search import constants
from odor_search.cli import STAGES, main


def test_version() -> None:
    assert odor_search.__version__ == "0.1.0"


def test_constants_moved_intact() -> None:
    assert constants.ARENA_NAMES == ["topleft", "topright", "bottomleft", "bottomright"]
    assert constants.N_NODES == 5
    assert constants.ARENA_DIAMETER_MM == 75.0


def test_package_relative_imports_resolve() -> None:
    # tracking imports from .constants and .detection; detection imports cv2/scipy.
    # If these resolve, the lib -> package move is wired correctly.
    from odor_search.detection import ArenaCircle, detect_arena_circles  # noqa: F401
    from odor_search.tracking import schmitt_trigger

    assert callable(schmitt_trigger)


def test_cli_no_command_prints_help() -> None:
    assert main([]) == 1


def test_cli_stage_stub_runs() -> None:
    assert main(["detect"]) == 0
    assert "detect" in STAGES
    assert "all" in STAGES
