"""Trajectory building, cleaning, and approach detection."""

from __future__ import annotations

import numpy as np

from .constants import ARENA_NAMES, THORAX_IDX
from .detection import ArenaCircle


def build_arena_trajectories(
    positions: np.ndarray,
    arena_circles: dict[str, ArenaCircle],
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    """Build one trajectory per arena using frame-by-frame assignment.

    For each frame, each detection (track) is assigned to the arena whose
    circle contains it. If multiple detections land in the same arena on
    the same frame, the one closest to the arena center is kept.

    Returns:
        merged: dict arena_name -> (n_frames, n_nodes, 2) trajectory
        arena_tracks: dict arena_name -> list of track indices that contributed
    """
    n_frames, n_tracks, n_nodes, _ = positions.shape

    merged = {
        name: np.full((n_frames, n_nodes, 2), np.nan)
        for name in ARENA_NAMES
    }
    arena_track_set: dict[str, set[int]] = {name: set() for name in ARENA_NAMES}

    for frame_idx in range(n_frames):
        for arena_name in ARENA_NAMES:
            ac = arena_circles[arena_name]
            best_track = -1
            best_dist = np.inf

            for t in range(n_tracks):
                thorax = positions[frame_idx, t, THORAX_IDX, :]
                if np.any(np.isnan(thorax)):
                    continue

                dx = thorax[0] - ac.center_x
                dy = thorax[1] - ac.center_y
                dist = np.sqrt(dx ** 2 + dy ** 2)

                if dist <= ac.radius_px * 1.05 and dist < best_dist:
                    best_dist = dist
                    best_track = t

            if best_track >= 0:
                merged[arena_name][frame_idx] = positions[frame_idx, best_track, :, :]
                arena_track_set[arena_name].add(best_track)

    arena_tracks = {name: sorted(tracks) for name, tracks in arena_track_set.items()}
    return merged, arena_tracks


def _interpolate_gaps(arr_1d: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN gaps up to max_gap frames."""
    arr = arr_1d.copy()
    is_nan = np.isnan(arr)
    if not is_nan.any():
        return arr

    n = len(arr)
    changes = np.diff(is_nan.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if is_nan[0]:
        starts = np.concatenate([[0], starts])
    if is_nan[-1]:
        ends = np.concatenate([ends, [n]])

    for s, e in zip(starts, ends):
        gap_len = e - s
        if gap_len <= max_gap and s > 0 and e < n:
            v0 = arr[s - 1]
            v1 = arr[e]
            if not (np.isnan(v0) or np.isnan(v1)):
                arr[s:e] = np.linspace(v0, v1, gap_len + 2)[1:-1]

    return arr


def clean_merged_trajectory(
    trajectory: np.ndarray,
    arena_circle: ArenaCircle,
    max_displacement_px: float = 40.0,
    interpolate_gap: int = 60,
    max_rounds: int = 10,
) -> np.ndarray:
    """Clean a merged per-arena trajectory.

    Iterates until no jumps remain:
    1. Remove out-of-arena points
    2. Remove jumps (> max_displacement_px)
    3. Interpolate short NaN gaps
    4. Repeat until clean
    """
    cleaned = trajectory.copy()
    n_frames, n_nodes, _ = cleaned.shape
    ac = arena_circle
    margin = ac.radius_px * 0.05

    for node_idx in range(n_nodes):
        xy = cleaned[:, node_idx, :]

        # Out-of-arena removal
        dx = xy[:, 0] - ac.center_x
        dy = xy[:, 1] - ac.center_y
        dist = np.sqrt(dx ** 2 + dy ** 2)
        outside = dist > (ac.radius_px + margin)
        xy[outside] = np.nan

        # Iterative: remove jumps -> interpolate -> check
        for _round in range(max_rounds):
            n_removed = 0
            for _pass in range(5):
                disp = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
                jump_frames = np.where(disp > max_displacement_px)[0]
                if len(jump_frames) == 0:
                    break
                for jf in jump_frames:
                    xy[jf + 1] = np.nan
                n_removed += len(jump_frames)

            for dim in range(2):
                xy[:, dim] = _interpolate_gaps(xy[:, dim], interpolate_gap)

            disp = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
            remaining = np.sum(disp[~np.isnan(disp)] > max_displacement_px)
            if remaining == 0:
                break

        # Final hard pass
        for _pass in range(5):
            disp = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
            jump_frames = np.where(disp > max_displacement_px)[0]
            if len(jump_frames) == 0:
                break
            for jf in jump_frames:
                xy[jf + 1] = np.nan

        cleaned[:, node_idx, :] = xy

    return cleaned


def schmitt_trigger(
    dist_mm: np.ndarray, valid: np.ndarray,
    activate_mm: float, deactivate_mm: float,
) -> np.ndarray:
    """Schmitt trigger for approach detection.

    Active when distance drops below activate_mm,
    deactivates when distance rises above deactivate_mm.
    """
    n = len(dist_mm)
    active = np.zeros(n, dtype=bool)
    state = False
    for i in range(n):
        if not valid[i]:
            active[i] = state
            continue
        d = dist_mm[i]
        if state:
            state = d <= deactivate_mm
        else:
            state = d < activate_mm
        active[i] = state
    return active


def schmitt_trigger_upper(
    dist_mm: np.ndarray, valid: np.ndarray,
    activate_mm: float, deactivate_mm: float,
) -> np.ndarray:
    """Schmitt trigger for border detection.

    Active when distance rises above activate_mm,
    deactivates when distance drops below deactivate_mm.
    """
    n = len(dist_mm)
    active = np.zeros(n, dtype=bool)
    state = False
    for i in range(n):
        if not valid[i]:
            active[i] = state
            continue
        d = dist_mm[i]
        if state:
            state = d >= deactivate_mm
        else:
            state = d > activate_mm
        active[i] = state
    return active
