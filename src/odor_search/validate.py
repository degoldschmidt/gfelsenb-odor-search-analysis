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


def _check_timing(video: Path, time_column: str) -> Check:
    csv = video.with_suffix(".csv")
    if not csv.exists():
        return Check(LEVEL_ERROR, f"timing CSV missing ({csv.name})")
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
    return Check(LEVEL_OK, f"{len(df)} rows, '{time_column}' present")


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


def _check_log(video: Path) -> Check:
    logs = sorted(video.parent.glob("log_*.txt"))
    if not logs:
        return Check(LEVEL_WARN, "no log_*.txt in the video's directory")
    return Check(LEVEL_OK, logs[0].name)


def validate_input(
    path: str | Path,
    *,
    predictions_dir: str | Path | None = None,
    time_column: str = DEFAULT_TIME_COLUMN,
    min_slp_frames: int = DEFAULT_MIN_SLP_FRAMES,
) -> Report:
    """Discover recordings under ``path`` and run all checks on each."""
    pred_dir = Path(predictions_dir) if predictions_dir is not None else None
    recordings: list[Recording] = []
    for video in find_videos(path):
        rec = Recording(video=video)
        rec.checks["video"] = _check_video(video)
        rec.checks["timing"] = _check_timing(video, time_column)
        rec.checks["predictions"] = _check_predictions(video, pred_dir, min_slp_frames)
        rec.checks["log"] = _check_log(video)
        recordings.append(rec)
    return Report(root=Path(path), recordings=recordings)


_GLYPH = {LEVEL_OK: "OK  ", LEVEL_WARN: "WARN", LEVEL_ERROR: "ERR "}
_CHECK_ORDER = ("video", "timing", "predictions", "log")


def format_report(report: Report) -> str:
    lines: list[str] = []
    lines.append(f"Validating: {report.root}  ({len(report.recordings)} recording(s))")
    if not report.recordings:
        lines.append("")
        lines.append(f"  No .avi recordings found under {report.root}")
        lines.append("")
        lines.append("Result: FAIL (nothing to validate)")
        return "\n".join(lines)

    for rec in report.recordings:
        flag = "" if rec.ok else "   <-- FAIL"
        lines.append("")
        lines.append(f"  {_video_stem(rec.video)}{flag}")
        for name in _CHECK_ORDER:
            chk = rec.checks.get(name)
            if chk is None:
                continue
            lines.append(f"    {name:<12} {_GLYPH[chk.level]}  {chk.detail}")

    n_ok = sum(r.ok for r in report.recordings)
    n_bad = len(report.recordings) - n_ok
    n_warn = sum(r.n_warn for r in report.recordings)
    n_err = sum(r.n_error for r in report.recordings)
    lines.append("")
    lines.append(
        f"Summary: {n_ok} OK, {n_bad} with issues "
        f"({n_warn} warning(s), {n_err} error(s))  ->  "
        f"{'PASS' if report.ok else 'FAIL'}"
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
