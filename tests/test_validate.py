"""Tests for the `odor-search input` pre-flight validator.

Filesystem discovery, pairing and the CSV/log checks are exercised with dummy
files in a tmp dir. The video/prediction checks (which need real binaries) are
tested only for graceful error handling, not success.
"""

from __future__ import annotations

from odor_search import validate
from odor_search.cli import main
from odor_search.validate import (
    _check_log,
    _check_timing,
    find_videos,
    format_report,
    validate_input,
)


def _touch(path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_find_videos_recursive(tmp_path):
    _touch(tmp_path / "2026-03-26" / "localsearch_A.avi")
    _touch(tmp_path / "2026-03-27" / "localsearch_B.avi")
    _touch(tmp_path / "2026-03-27" / "notes.txt")
    assert [v.name for v in find_videos(tmp_path)] == [
        "localsearch_A.avi",
        "localsearch_B.avi",
    ]


def test_find_videos_single_file_and_wrong_suffix(tmp_path):
    v = tmp_path / "x.avi"
    _touch(v)
    _touch(tmp_path / "x.csv")
    assert find_videos(v) == [v]
    assert find_videos(tmp_path / "x.csv") == []
    assert find_videos(tmp_path / "does_not_exist") == []


def test_check_timing_ok(tmp_path):
    v = tmp_path / "rec.avi"
    _touch(v)
    _touch(tmp_path / "rec.csv", "Item1,Item5\n0,0.0\n1,0.05\n")
    chk = _check_timing(v, "Item5")
    assert chk.level == validate.LEVEL_OK
    assert "2 rows" in chk.detail


def test_check_timing_missing_column_and_file(tmp_path):
    v = tmp_path / "rec.avi"
    _touch(v)
    assert _check_timing(v, "Item5").level == validate.LEVEL_ERROR  # no csv
    _touch(tmp_path / "rec.csv", "a,b\n1,2\n")
    assert _check_timing(v, "Item5").level == validate.LEVEL_ERROR  # no Item5


def test_check_log(tmp_path):
    v = tmp_path / "day" / "rec.avi"
    _touch(v)
    assert _check_log(v).level == validate.LEVEL_WARN
    _touch(tmp_path / "day" / "log_2026-03-26.txt")
    assert _check_log(v).level == validate.LEVEL_OK


def test_validate_input_empty_dir(tmp_path):
    report = validate_input(tmp_path)
    assert report.recordings == []
    assert report.ok is False
    assert "FAIL" in format_report(report)


def test_validate_input_orchestrates_without_raising(tmp_path):
    # A dummy (non-real) video with a valid timing CSV but no predictions:
    # every check must resolve to a Check, not raise.
    _touch(tmp_path / "rec.avi")
    _touch(tmp_path / "rec.csv", "Item5\n0.0\n0.05\n")
    report = validate_input(tmp_path)
    assert len(report.recordings) == 1
    rec = report.recordings[0]
    assert set(rec.checks) == {"video", "timing", "predictions", "log"}
    assert rec.checks["timing"].level == validate.LEVEL_OK
    assert rec.checks["predictions"].level == validate.LEVEL_ERROR  # no .slp
    assert rec.ok is False


def test_cli_input_empty_dir_returns_1(tmp_path):
    assert main(["input", str(tmp_path)]) == 1


def test_format_report_color_toggle(tmp_path):
    _touch(tmp_path / "rec.avi")
    _touch(tmp_path / "rec.csv", "Item5\n0.0\n0.05\n")
    report = validate_input(tmp_path)
    plain = format_report(report, color=False)
    colored = format_report(report, color=True)
    assert "\033[" not in plain
    assert "\033[31m" in colored  # red, because predictions error out (no .slp)
    assert "\033[0m" in colored
