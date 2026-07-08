"""I/O utilities: SLEAP prediction loading and frame timing."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sleap_io as sio


def load_sleap_predictions(slp_path: str | Path) -> sio.Labels:
    return sio.load_slp(str(slp_path))


def load_frame_times(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load frame timing from a localsearch CSV.

    Column 'Item5' is elapsed time since recording start (seconds).

    Returns:
        dt: inter-frame intervals in seconds (length n_frames)
        cumulative: elapsed time per frame in seconds (length n_frames)
    """
    df = pd.read_csv(csv_path)
    cumulative = df["Item5"].values
    dt = np.diff(cumulative, prepend=0.0)
    return dt, cumulative


def find_latest_prediction(basename: str, predictions_dir: str | Path) -> str | None:
    """Find the latest .predictions.slp file for a given video basename."""
    pred_dir = Path(predictions_dir)
    matches = [
        f for f in pred_dir.iterdir()
        if f.name.startswith(basename) and f.name.endswith(".predictions.slp")
    ]
    if not matches:
        return None
    latest = max(matches, key=lambda f: f.stat().st_mtime)
    return str(latest)
