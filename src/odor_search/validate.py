"""Pre-flight validation of a raw-data location (``odor-search input <path>``).

Given a directory of recordings (or a single ``.avi``), this discovers the
expected raw layout and checks, per recording:

* **video** — the ``.avi`` opens in OpenCV, and its frame index is intact. A
  broken container index makes ``CAP_PROP_FRAME_COUNT <= 0``, which is the known
  silent-failure mode (unreliable seeking; SLEAP writes an empty prediction).
* **timing** — a paired timing CSV exists and carries the elapsed-time column.
* **predictions** — a SLEAP ``.predictions.slp`` exists, loads, has enough
  labeled frames, and *reports the node (skeleton) names it contains* — the
  values that belong in the experiment config, not hard-coded constants.
* **log** — a per-day ``log_*.txt`` is present next to the video.

Each check is ``ok`` / ``warn`` / ``error``; a recording passes if it has no
errors (warnings are allowed). The command exits non-zero if any recording
fails, so it can gate a pipeline run.

Heavy dependencies (OpenCV, sleap-io, pandas) are imported lazily inside the
check functions so ``import odor_search.validate`` stays cheap and the
filesystem-only discovery/pairing logic is unit-testable without them.

In Phase 1 the defaults below (time column, min frames, and eventually the
expected skeleton) are supplied by the experiment config rather than hardcoded.
"""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Defaults — superseded by the experiment config in Phase 1.
DEFAULT_TIME_COLUMN = "Item5"        # elapsed-seconds column in the timing CSV
DEFAULT_MIN_SLP_FRAMES = 100         # fewer labeled frames => likely silent SLEAP failure
PREDICTION_SUFFIX = ".predictions.slp"

LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_ERROR = "error"


@dataclass
class Check:
    """The result of one validation check on one recording."""

    level: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.level != LEVEL_ERROR


@dataclass
class Recording:
    """One video and the outcome of its checks."""

    video: Path
    checks: dict[str, Check] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks.values())

    @property
    def n_warn(self) -> int:
        return sum(c.level == LEVEL_WARN for c in self.checks.values())

    @property
    def n_error(self) -> int:
        return sum(c.level == LEVEL_ERROR for c in self.checks.values())


@dataclass
class Report:
    root: Path
    recordings: list[Recording]

    @property
    def ok(self) -> bool:
        return bool(self.recordings) and all(r.ok for r in self.recordings)


def find_videos(path: str | Path) -> list[Path]:
    """Return the ``.avi`` files under ``path`` (or ``[path]`` if it is one)."""
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix == ".avi" else []
    if not path.is_dir():
        return []
    return sorted(Path(p) for p in glob.glob(str(path / "**" / "*.avi"), recursive=True))


def _video_stem(video: Path) -> str:
    return video.name[: -len(".avi")]


def _resolve_slp(video: Path, predictions_dir: Path | None) -> Path | None:
    """Locate the prediction file: beside the video, else in ``predictions_dir``."""
    beside = video.with_name(_video_stem(video) + PREDICTION_SUFFIX)
    if beside.exists():
        return beside
    if predictions_dir is not None:
        cand = Path(predictions_dir) / (_video_stem(video) + PREDICTION_SUFFIX)
        if cand.exists():
            return cand
    return None


def _check_video(video: Path) -> Check:
    if not video.exists():
        return Check(LEVEL_ERROR, "video file missing")
    try:
        import cv2

        cap = cv2.VideoCapture(str(video))
        opened = cap.isOpened()
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    except Exception as exc:  # pragma: no cover - defensive
        return Check(LEVEL_ERROR, f"cannot open ({type(exc).__name__}: {exc})")
    if not opened:
        return Check(LEVEL_ERROR, "OpenCV could not open the file")
    if n_frames <= 0:
        return Check(
            LEVEL_WARN,
            "frame index broken (frame_count<=0); seeking unreliable — "
            "rebuild with `ffmpeg -i in.avi -c copy out.avi`",
        )
    return Check(LEVEL_OK, f"{n_frames} frames")


def _resolve_timing(video: Path, timing_dir: Path | None) -> Path | None:
    """Locate the timing CSV.

    Tries the video's exact stem first, then — for per-arena crops named
    ``..._arenaN`` — the shared full-video CSV (suffix stripped), since one
    timing file serves all four crops of a recording. Searches ``timing_dir``
    if given, else the video's own directory.
    """
    search_dir = timing_dir if timing_dir is not None else video.parent
    stem = _video_stem(video)
    candidates = [search_dir / f"{stem}.csv"]
    shared = re.sub(r"_arena\d+$", "", stem)
    if shared != stem:
        candidates.append(search_dir / f"{shared}.csv")
    for csv in candidates:
        if csv.exists():
            return csv
    return None


def _check_timing(
    video: Path, time_column: str, timing_dir: Path | None = None
) -> Check:
    csv = _resolve_timing(video, timing_dir)
    if csv is None:
        return Check(LEVEL_ERROR, f"timing CSV missing for {_video_stem(video)}")
    try:
        import pandas as pd

        df = pd.read_csv(csv)
    except Exception as exc:
        return Check(LEVEL_ERROR, f"CSV unreadable ({type(exc).__name__})")
    if time_column not in df.columns:
        have = list(df.columns)[:6]
        return Check(LEVEL_ERROR, f"no '{time_column}' column (have: {have})")
    if len(df) == 0:
        return Check(LEVEL_ERROR, "timing CSV is empty")
    return Check(LEVEL_OK, f"{csv.name}: {len(df)} rows, '{time_column}' present")


def _check_predictions(
    video: Path, predictions_dir: Path | None, min_frames: int
) -> Check:
    slp = _resolve_slp(video, predictions_dir)
    if slp is None:
        return Check(LEVEL_ERROR, "no .predictions.slp found")
    try:
        import sleap_io as sio

        labels = sio.load_slp(str(slp))
        n_frames = len(labels)
    except Exception as exc:
        return Check(LEVEL_ERROR, f"SLP unloadable ({type(exc).__name__}: {exc})")

    nodes: list[str] = []
    try:
        nodes = [node.name for node in labels.skeletons[0].nodes]
    except Exception:  # pragma: no cover - skeleton shape varies
        pass

    if n_frames < min_frames:
        return Check(
            LEVEL_WARN,
            f"only {n_frames} labeled frames (<{min_frames}) — possible silent "
            "tracking failure",
        )
    node_str = ", ".join(nodes) if nodes else "unknown"
    return Check(LEVEL_OK, f"{n_frames} frames; nodes: {node_str}")


def _check_log(video: Path, log_dir: Path | None = None) -> Check:
    search_dir = log_dir if log_dir is not None else video.parent
    # Match both exp003 (`log_*.txt`) and Ana (`*_log.txt`) naming.
    logs = sorted(search_dir.glob("log_*.txt")) + sorted(search_dir.glob("*_log.txt"))
    if not logs:
        where = "the --log-dir" if log_dir is not None else "the video's directory"
        return Check(LEVEL_WARN, f"no log file (log_*.txt / *_log.txt) in {where}")
    return Check(LEVEL_OK, logs[0].name)


def validate_input(
    path: str | Path,
    *,
    predictions_dir: str | Path | None = None,
    timing_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    time_column: str = DEFAULT_TIME_COLUMN,
    min_slp_frames: int = DEFAULT_MIN_SLP_FRAMES,
) -> Report:
    """Discover recordings under ``path`` and run all checks on each.

    ``predictions_dir`` / ``timing_dir`` / ``log_dir`` allow the crop-centric
    layout where those files live in sibling directories rather than beside the
    video (as in exp003).
    """
    pred_dir = Path(predictions_dir) if predictions_dir is not None else None
    tim_dir = Path(timing_dir) if timing_dir is not None else None
    lg_dir = Path(log_dir) if log_dir is not None else None
    recordings: list[Recording] = []
    for video in find_videos(path):
        rec = Recording(video=video)
        rec.checks["video"] = _check_video(video)
        rec.checks["timing"] = _check_timing(video, time_column, tim_dir)
        rec.checks["predictions"] = _check_predictions(video, pred_dir, min_slp_frames)
        rec.checks["log"] = _check_log(video, lg_dir)
        recordings.append(rec)
    return Report(root=Path(path), recordings=recordings)


_GLYPH = {LEVEL_OK: "OK  ", LEVEL_WARN: "WARN", LEVEL_ERROR: "ERR "}
_CHECK_ORDER = ("video", "timing", "predictions", "log")

# ANSI colors: green = valid, yellow = warning (works, but beware), red = error.
_ANSI = {LEVEL_OK: "\033[32m", LEVEL_WARN: "\033[33m", LEVEL_ERROR: "\033[31m"}
_ANSI_BOLD = "\033[1m"
_ANSI_RESET = "\033[0m"


def _paint(text: str, level: str, color: bool, *, bold: bool = False) -> str:
    """Wrap ``text`` in the ANSI color for ``level`` when ``color`` is on."""
    if not color or level not in _ANSI:
        return text
    return f"{_ANSI_BOLD if bold else ''}{_ANSI[level]}{text}{_ANSI_RESET}"


def format_report(report: Report, *, color: bool = False) -> str:
    lines: list[str] = []
    lines.append(f"Validating: {report.root}  ({len(report.recordings)} recording(s))")
    if not report.recordings:
        lines.append("")
        lines.append(f"  No .avi recordings found under {report.root}")
        lines.append("")
        lines.append(_paint("Result: FAIL (nothing to validate)", LEVEL_ERROR, color, bold=True))
        return "\n".join(lines)

    for rec in report.recordings:
        flag = "" if rec.ok else _paint("   <-- FAIL", LEVEL_ERROR, color, bold=True)
        lines.append("")
        lines.append(f"  {_video_stem(rec.video)}{flag}")
        for name in _CHECK_ORDER:
            chk = rec.checks.get(name)
            if chk is None:
                continue
            glyph = _paint(_GLYPH[chk.level], chk.level, color, bold=True)
            lines.append(f"    {name:<12} {glyph}  {chk.detail}")

    n_ok = sum(r.ok for r in report.recordings)
    n_bad = len(report.recordings) - n_ok
    n_warn = sum(r.n_warn for r in report.recordings)
    n_err = sum(r.n_error for r in report.recordings)
    verdict = _paint(
        "PASS" if report.ok else "FAIL",
        LEVEL_OK if report.ok else LEVEL_ERROR,
        color,
        bold=True,
    )
    lines.append("")
    lines.append(
        f"Summary: {n_ok} OK, {n_bad} with issues "
        f"({n_warn} warning(s), {n_err} error(s))  ->  {verdict}"
    )
    return "\n".join(lines)


def report_to_json(report: Report) -> str:
    payload = {
        "root": str(report.root),
        "ok": report.ok,
        "recordings": [
            {
                "video": str(rec.video),
                "ok": rec.ok,
                "checks": {
                    name: {"level": chk.level, "detail": chk.detail}
                    for name, chk in rec.checks.items()
                },
            }
            for rec in report.recordings
        ],
    }
    return json.dumps(payload, indent=2)
