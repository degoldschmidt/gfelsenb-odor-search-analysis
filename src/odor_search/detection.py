"""Arena and center cylinder detection from video background models."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

from .constants import ARENA_NAMES, ARENA_DIAMETER_MM


class ArenaCircle(NamedTuple):
    center_x: float
    center_y: float
    radius_px: float
    px_per_mm: float


class CenterCylinder(NamedTuple):
    center_x: float
    center_y: float
    radius_px: float


def compute_background(
    video_path: str, n_samples: int = 200, sample_stride: int = 50
) -> np.ndarray:
    """Build background image by median-averaging sampled frames.

    Normally samples ``n_samples`` frames spread across the video by seeking
    to frame positions. If the container reports no frame count
    (``CAP_PROP_FRAME_COUNT <= 0`` — e.g. an AVI with a missing/broken idx1
    index), frame-accurate seeking is unreliable and returns garbage, so we
    fall back to decoding sequentially and keeping every ``sample_stride``-th
    bright frame. ffmpeg/SLEAP decode such files fine; only OpenCV seeking
    fails. See the "data integrity" notes in analysis_documentation.html.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    if total_frames > 0:
        # Healthy file: sample by seeking to evenly-spaced frame positions.
        positions = np.linspace(0, total_frames - 1, n_samples).astype(int)
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.mean() < 30:
                continue
            frames.append(gray.astype(np.float32))
    else:
        # Broken/missing index: seeking is unreliable. Decode sequentially
        # and keep every sample_stride-th bright frame until we have enough.
        idx = 0
        while len(frames) < n_samples:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if gray.mean() >= 30:
                    frames.append(gray.astype(np.float32))
            idx += 1
    cap.release()

    if len(frames) < 10:
        raise RuntimeError(f"Not enough bright frames from {video_path}")

    return np.median(frames, axis=0).astype(np.uint8)


def detect_arena_circles(
    video_path: str,
) -> tuple[dict[str, ArenaCircle], np.ndarray]:
    """Detect 4 circular arenas using background model + Hough circles."""
    bg = compute_background(video_path)
    frame_h, frame_w = bg.shape[:2]
    mid_x, mid_y = frame_w / 2, frame_h / 2

    param_sets = [
        {"dp": 1, "minDist": 300, "param1": 50, "param2": 30,
         "minRadius": 200, "maxRadius": 600},
        {"dp": 1, "minDist": 250, "param1": 40, "param2": 25,
         "minRadius": 180, "maxRadius": 650},
        {"dp": 1.5, "minDist": 200, "param1": 35, "param2": 20,
         "minRadius": 150, "maxRadius": 700},
    ]

    bg_blur = cv2.GaussianBlur(bg, (9, 9), 2)

    circles = None
    for params in param_sets:
        circles = cv2.HoughCircles(bg_blur, cv2.HOUGH_GRADIENT, **params)
        if circles is not None and len(circles[0]) >= 4:
            break
        circles = None

    if circles is None:
        raise RuntimeError(f"Could not detect 4 arena circles in {video_path}")

    detected = np.round(circles[0]).astype(int)

    assignment: dict[str, tuple[int, int, int]] = {}
    best_dist: dict[str, float] = {}

    for c in detected[:12]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        if cx < mid_x and cy < mid_y:
            name = "topleft"
        elif cx >= mid_x and cy < mid_y:
            name = "topright"
        elif cx < mid_x and cy >= mid_y:
            name = "bottomleft"
        else:
            name = "bottomright"

        qcx = mid_x / 2 if cx < mid_x else mid_x + mid_x / 2
        qcy = mid_y / 2 if cy < mid_y else mid_y + mid_y / 2
        d = np.sqrt((cx - qcx) ** 2 + (cy - qcy) ** 2)

        if name not in best_dist or d < best_dist[name]:
            assignment[name] = (cx, cy, r)
            best_dist[name] = d

    if len(assignment) != 4:
        raise RuntimeError(
            f"Could not assign circles to all 4 quadrants; got {list(assignment.keys())}"
        )

    result = {}
    for name, (cx, cy, r) in assignment.items():
        px_per_mm = (2 * r) / ARENA_DIAMETER_MM
        result[name] = ArenaCircle(
            center_x=float(cx), center_y=float(cy),
            radius_px=float(r), px_per_mm=px_per_mm,
        )
    return result, bg


def compute_radial_profile(
    bg: np.ndarray, cx: int, cy: int, max_radius: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean intensity at each radius from (cx, cy).

    Returns (radii, mean_intensity) arrays.
    """
    radii = np.arange(1, max_radius)
    mean_intensity = np.zeros(len(radii))

    for i, r in enumerate(radii):
        n_samples = max(36, int(2 * np.pi * r))
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        rx = (cx + r * np.cos(angles)).astype(int)
        ry = (cy + r * np.sin(angles)).astype(int)
        valid = (
            (rx >= 0) & (rx < bg.shape[1]) & (ry >= 0) & (ry < bg.shape[0])
        )
        if valid.any():
            mean_intensity[i] = np.mean(bg[ry[valid], rx[valid]].astype(float))

    return radii, mean_intensity


def score_radial_profile(
    radii: np.ndarray, mean_intensity: np.ndarray, px_per_mm: float,
    center_range_mm: tuple[float, float] = (0.0, 0.8),
    peak_range_mm: tuple[float, float] = (1.0, 2.0),
    dip_range_mm: tuple[float, float] = (2.0, 3.2),
    expected_dip_mm: float = 2.4,
    proximity_weight: float = 2.0,
) -> tuple[float, float]:
    """Score a radial profile for the full cylinder pattern.

    The expected profile has three features:
      1. Dark center   (r < 0.8mm)  — inside the cylinder
      2. Bright peak   (r ~ 1.5mm)  — the cylinder rim/wall
      3. Dark dip      (r ~ 2.4mm)  — shadow at cylinder base

    The score rewards profiles that show all three features clearly.
    Returns (score, radius_px) where radius_px is the dip position.
    """
    mi_smooth = uniform_filter1d(mean_intensity, 5)
    radii_mm = radii / px_per_mm

    # 1. Center intensity (should be a local minimum — dark inside cylinder)
    center_mask = (radii_mm >= center_range_mm[0]) & (radii_mm <= center_range_mm[1])
    if center_mask.sum() < 2:
        return -np.inf, 20.0
    center_val = np.mean(mi_smooth[center_mask])

    # 2. Peak in bright-rim range (should be higher than center)
    peak_mask = (radii_mm >= peak_range_mm[0]) & (radii_mm <= peak_range_mm[1])
    if peak_mask.sum() < 2:
        return -np.inf, 20.0
    peak_indices = np.where(peak_mask)[0]
    peak_local = np.argmax(mi_smooth[peak_mask])
    peak_idx = peak_indices[peak_local]
    peak_val = mi_smooth[peak_idx]

    # 3. Dip after the peak (should be lower than peak)
    dip_mask = ((radii_mm >= dip_range_mm[0]) & (radii_mm <= dip_range_mm[1])
                & (radii > radii[peak_idx]))
    if dip_mask.sum() < 2:
        return -np.inf, 20.0
    dip_indices = np.where(dip_mask)[0]
    dip_local = np.argmin(mi_smooth[dip_mask])
    dip_idx = dip_indices[dip_local]
    dip_val = mi_smooth[dip_idx]

    # Reject if shape is wrong: need center < peak AND dip < peak
    if peak_val <= center_val or peak_val <= dip_val:
        return -np.inf, 20.0

    # Score: reward both dark-center→peak rise AND peak→dip contrast
    rise_contrast = peak_val - center_val
    dip_contrast = peak_val - dip_val
    score = rise_contrast + dip_contrast

    # Prefer dip near expected 2.4mm
    dip_mm = radii_mm[dip_idx]
    score -= proximity_weight * abs(dip_mm - expected_dip_mm)

    return score, float(radii[dip_idx])


def detect_center_cylinders(
    bg: np.ndarray,
    arena_circles: dict[str, ArenaCircle],
    cylinder_radius_mm: float = 2.4,
    search_radius_px: int = 35,
    search_step_px: int = 3,
    max_profile_radius: int = 50,
    offset_penalty: float = 2.0,
) -> dict[str, CenterCylinder]:
    """Detect center cylinders by searching for the characteristic radial profile.

    For each arena, searches a grid of (x, y) offsets around the arena center
    to find the position that produces the best-matching radial intensity
    profile (dark center → bright rim at ~1.5mm → dark dip at ~2.4mm).

    The cylinder radius is fixed at cylinder_radius_mm (the physical cylinder
    is the same object in every arena); only the (x, y) center is searched.

    A quadratic offset penalty biases the search toward the arena center,
    preventing lock-on to spurious features far from the cylinder.
    """
    cylinders: dict[str, CenterCylinder] = {}

    offsets = range(-search_radius_px, search_radius_px + 1, search_step_px)

    for arena_name, ac in arena_circles.items():
        cx0, cy0 = int(ac.center_x), int(ac.center_y)
        px_per_mm = ac.px_per_mm
        fixed_radius_px = cylinder_radius_mm * px_per_mm

        best_adj_score = -np.inf
        best_cx, best_cy = float(cx0), float(cy0)

        # Coarse grid search for best center position
        for dx in offsets:
            for dy in offsets:
                cx, cy = cx0 + dx, cy0 + dy
                radii, intensity = compute_radial_profile(
                    bg, cx, cy, max_profile_radius
                )
                score, _ = score_radial_profile(
                    radii, intensity, px_per_mm
                )
                # Quadratic penalty for distance from arena center
                offset_mm = np.sqrt(dx**2 + dy**2) / px_per_mm
                adj_score = score - offset_penalty * offset_mm**2

                if adj_score > best_adj_score:
                    best_adj_score = adj_score
                    best_cx, best_cy = float(cx), float(cy)

        # Fine refinement: ±step around the coarse best, step=1
        if search_step_px > 1:
            fine = range(-search_step_px, search_step_px + 1)
            coarse_cx, coarse_cy = int(best_cx), int(best_cy)
            for dx_fine in fine:
                for dy_fine in fine:
                    cx, cy = coarse_cx + dx_fine, coarse_cy + dy_fine
                    radii, intensity = compute_radial_profile(
                        bg, cx, cy, max_profile_radius
                    )
                    score, _ = score_radial_profile(
                        radii, intensity, px_per_mm
                    )
                    dx_total = cx - cx0
                    dy_total = cy - cy0
                    offset_mm = np.sqrt(dx_total**2 + dy_total**2) / px_per_mm
                    adj_score = score - offset_penalty * offset_mm**2

                    if adj_score > best_adj_score:
                        best_adj_score = adj_score
                        best_cx, best_cy = float(cx), float(cy)

        cylinders[arena_name] = CenterCylinder(
            center_x=best_cx,
            center_y=best_cy,
            radius_px=fixed_radius_px,
        )

    return cylinders
