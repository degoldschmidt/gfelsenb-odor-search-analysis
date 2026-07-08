"""Comprehensive diagnostic image generation for arena/cylinder detection."""

from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as MplCircle, FancyArrowPatch

from .constants import ARENA_NAMES, ARENA_COLORS, ARENA_RADIUS_MM
from .detection import ArenaCircle, CenterCylinder


def save_detection_diagnostics(
    bg: np.ndarray,
    arena_circles: dict[str, ArenaCircle],
    center_cylinders: dict[str, CenterCylinder],
    out_dir: str,
    basename: str,
    arena_labels: dict[str, str] | None = None,
    flagged_arenas: set[str] | None = None,
    approach_activate_mm: float = 10.0,
    approach_deactivate_mm: float = 20.0,
):
    """Generate comprehensive diagnostic images for detection results.

    Produces:
      1. overview.png        — full frame with all arenas and cylinders overlaid
      2. arena_details.png   — zoomed crop per arena with radial profile
      3. radial_profiles.png — cylinder detection radial intensity profiles

    Args:
        bg: grayscale background image
        arena_circles: detected arena geometries
        center_cylinders: detected cylinder geometries
        out_dir: output directory for this video's diagnostics
        basename: video basename for titles
        arena_labels: optional dict mapping arena name -> label string
        flagged_arenas: set of arena names that need manual review
        approach_activate_mm: Schmitt trigger activation distance
        approach_deactivate_mm: Schmitt trigger deactivation distance
    """
    os.makedirs(out_dir, exist_ok=True)
    if flagged_arenas is None:
        flagged_arenas = set()

    _save_overview(
        bg, arena_circles, center_cylinders, out_dir, basename,
        arena_labels, flagged_arenas, approach_activate_mm, approach_deactivate_mm,
    )
    _save_arena_details(
        bg, arena_circles, center_cylinders, out_dir, basename,
        arena_labels, flagged_arenas, approach_activate_mm, approach_deactivate_mm,
    )
    _save_radial_profiles(
        bg, arena_circles, center_cylinders, out_dir, basename,
        flagged_arenas,
    )


def _save_overview(bg, arena_circles, center_cylinders, out_dir, basename,
                   arena_labels, flagged_arenas, activate_mm, deactivate_mm):
    """Full-frame overview with transparent overlays."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(bg, cmap="gray", interpolation="none")

    for arena_name in ARENA_NAMES:
        ac = arena_circles[arena_name]
        cyl = center_cylinders[arena_name]
        color = ARENA_COLORS[arena_name]
        is_flagged = arena_name in flagged_arenas

        # Arena boundary — red if flagged
        border_color = "#ff0000" if is_flagged else color
        border_lw = 4.0 if is_flagged else 2.5
        ax.add_patch(MplCircle(
            (ac.center_x, ac.center_y), ac.radius_px,
            fill=False, edgecolor=border_color, linewidth=border_lw,
            linestyle="-", alpha=0.9,
        ))
        # Thin filled arena tint
        tint = "#ff0000" if is_flagged else color
        ax.add_patch(MplCircle(
            (ac.center_x, ac.center_y), ac.radius_px,
            fill=True, facecolor=tint, alpha=0.10 if is_flagged else 0.06,
            edgecolor="none",
        ))

        # Cylinder (filled red)
        ax.add_patch(MplCircle(
            (cyl.center_x, cyl.center_y), cyl.radius_px,
            fill=True, facecolor="red", alpha=0.25, edgecolor="red",
            linewidth=1.5,
        ))

        # Schmitt trigger zones
        activate_px = activate_mm * ac.px_per_mm
        deactivate_px = deactivate_mm * ac.px_per_mm

        ax.add_patch(MplCircle(
            (cyl.center_x, cyl.center_y), activate_px,
            fill=False, edgecolor="red", linewidth=1.2, linestyle="--", alpha=0.7,
        ))
        ax.add_patch(MplCircle(
            (cyl.center_x, cyl.center_y), deactivate_px,
            fill=False, edgecolor="orange", linewidth=1.2, linestyle=":", alpha=0.7,
        ))
        # Shaded approach zone
        ax.add_patch(MplCircle(
            (cyl.center_x, cyl.center_y), activate_px,
            fill=True, facecolor="red", alpha=0.08, edgecolor="none",
        ))

        # Labels
        label_text = arena_name
        if arena_labels and arena_name in arena_labels:
            label_text = f"{arena_name}\n{arena_labels[arena_name]}"
        if is_flagged:
            label_text += "\nFLAGGED"
        label_ec = "#ff0000" if is_flagged else color
        label_fc = "#ffcccc" if is_flagged else "white"
        ax.text(
            ac.center_x, ac.center_y - ac.radius_px - 15,
            label_text, ha="center", va="bottom", fontsize=10,
            fontweight="bold", color=border_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=label_fc, alpha=0.8, edgecolor=label_ec),
        )

        # Measurements
        r_mm = ac.radius_px / ac.px_per_mm
        cyl_mm = cyl.radius_px / ac.px_per_mm
        info = f"arena r={r_mm:.1f}mm ({ac.radius_px:.0f}px)\ncyl r={cyl_mm:.1f}mm ({cyl.radius_px:.0f}px)\n{ac.px_per_mm:.2f} px/mm"
        ax.text(
            ac.center_x, ac.center_y + ac.radius_px + 10,
            info, ha="center", va="top", fontsize=7, color="white",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
        )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=2.5, label="Arena boundary"),
        MplCircle((0, 0), 1, facecolor="red", alpha=0.3, edgecolor="red", label="Cylinder"),
        Line2D([0], [0], color="red", linewidth=1.2, linestyle="--", label=f"Activate <{activate_mm}mm"),
        Line2D([0], [0], color="orange", linewidth=1.2, linestyle=":", label=f"Deactivate >{deactivate_mm}mm"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              framealpha=0.8, fancybox=True)

    ax.set_title(f"Detection Overview — {basename}", fontsize=13, fontweight="bold")
    ax.set_xlim(0, bg.shape[1])
    ax.set_ylim(bg.shape[0], 0)
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_arena_details(bg, arena_circles, center_cylinders, out_dir, basename,
                        arena_labels, flagged_arenas, activate_mm, deactivate_mm):
    """Per-arena zoomed detail images."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    layout = {"topleft": (0, 0), "topright": (0, 1),
              "bottomleft": (1, 0), "bottomright": (1, 1)}

    for arena_name, (row, col) in layout.items():
        ax = axes[row, col]
        ac = arena_circles[arena_name]
        cyl = center_cylinders[arena_name]
        color = ARENA_COLORS[arena_name]
        is_flagged = arena_name in flagged_arenas

        # Crop region with margin
        margin = ac.radius_px * 0.15
        x0 = max(0, int(ac.center_x - ac.radius_px - margin))
        x1 = min(bg.shape[1], int(ac.center_x + ac.radius_px + margin))
        y0 = max(0, int(ac.center_y - ac.radius_px - margin))
        y1 = min(bg.shape[0], int(ac.center_y + ac.radius_px + margin))

        ax.imshow(bg[y0:y1, x0:x1], cmap="gray", interpolation="none",
                  extent=[x0, x1, y1, y0])

        # Arena boundary — red if flagged
        border_color = "#ff0000" if is_flagged else color
        ax.add_patch(MplCircle(
            (ac.center_x, ac.center_y), ac.radius_px,
            fill=False, edgecolor=border_color,
            linewidth=4.0 if is_flagged else 2.5, alpha=0.9,
        ))
        ax.add_patch(MplCircle(
            (ac.center_x, ac.center_y), ac.radius_px,
            fill=True,
            facecolor="#ff0000" if is_flagged else color,
            alpha=0.10 if is_flagged else 0.05, edgecolor="none",
        ))

        # Cylinder
        ax.add_patch(MplCircle(
            (cyl.center_x, cyl.center_y), cyl.radius_px,
            fill=True, facecolor="red", alpha=0.3, edgecolor="red", linewidth=2,
        ))

        # Schmitt zones
        activate_px = activate_mm * ac.px_per_mm
        deactivate_px = deactivate_mm * ac.px_per_mm

        ax.add_patch(MplCircle(
            (cyl.center_x, cyl.center_y), activate_px,
            fill=True, facecolor="red", alpha=0.08, edgecolor="red",
            linewidth=1.5, linestyle="--",
        ))
        ax.add_patch(MplCircle(
            (cyl.center_x, cyl.center_y), deactivate_px,
            fill=False, edgecolor="orange", linewidth=1.5, linestyle=":",
        ))

        # Crosshair at cylinder center
        ch_len = cyl.radius_px + 8
        ax.plot([cyl.center_x - ch_len, cyl.center_x + ch_len],
                [cyl.center_y, cyl.center_y],
                color="red", linewidth=0.8, alpha=0.5)
        ax.plot([cyl.center_x, cyl.center_x],
                [cyl.center_y - ch_len, cyl.center_y + ch_len],
                color="red", linewidth=0.8, alpha=0.5)

        # Scale bar (10mm)
        scale_px = 10.0 * ac.px_per_mm
        bar_y = y1 - margin * 0.5
        bar_x0 = x0 + margin * 0.4
        ax.plot([bar_x0, bar_x0 + scale_px], [bar_y, bar_y],
                color="white", linewidth=3, solid_capstyle="butt")
        ax.text(bar_x0 + scale_px / 2, bar_y - 5, "10 mm",
                ha="center", va="bottom", fontsize=9, color="white",
                fontweight="bold")

        # Flagged warning badge
        if is_flagged:
            ax.text(x0 + margin * 0.5, y0 + margin * 0.5,
                    "NEEDS REVIEW", fontsize=12, fontweight="bold",
                    color="white", va="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#cc0000", alpha=0.9))

        # Title
        title = arena_name
        if arena_labels and arena_name in arena_labels:
            title += f"  —  {arena_labels[arena_name]}"
        r_mm = ac.radius_px / ac.px_per_mm
        cyl_mm = cyl.radius_px / ac.px_per_mm
        title += f"\narena r={r_mm:.1f}mm, cyl r={cyl_mm:.1f}mm, {ac.px_per_mm:.2f} px/mm"
        title_color = "#cc0000" if is_flagged else color
        ax.set_title(title, fontsize=10, fontweight="bold", color=title_color)

        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_axis_off()

    fig.suptitle(f"Arena Details — {basename}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "arena_details.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_radial_profiles(bg, arena_circles, center_cylinders, out_dir, basename,
                          flagged_arenas):
    """Radial intensity profiles showing cylinder detection."""
    from .detection import compute_radial_profile
    from scipy.ndimage import uniform_filter1d

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    layout = {"topleft": (0, 0), "topright": (0, 1),
              "bottomleft": (1, 0), "bottomright": (1, 1)}

    for arena_name, (row, col) in layout.items():
        ax = axes[row, col]
        ac = arena_circles[arena_name]
        cyl = center_cylinders[arena_name]
        color = ARENA_COLORS[arena_name]
        is_flagged = arena_name in flagged_arenas

        # Profile from the detected cylinder center (not arena center)
        radii, mean_intensity = compute_radial_profile(
            bg, int(cyl.center_x), int(cyl.center_y), 50)
        mi_smooth = uniform_filter1d(mean_intensity, 5)
        radii_mm = radii / ac.px_per_mm

        ax.plot(radii_mm, mean_intensity, color="gray", alpha=0.4, linewidth=1,
                label="Raw")
        ax.plot(radii_mm, mi_smooth, color=color, linewidth=2, label="Smoothed")

        # Mark detected cylinder radius (the minimum dip)
        cyl_mm = cyl.radius_px / ac.px_per_mm
        ax.axvline(cyl_mm, color="red", linewidth=1.5, linestyle="--",
                   label=f"Cylinder r={cyl_mm:.1f}mm")

        # Mark the dip point on the curve
        cyl_idx = np.argmin(np.abs(radii - cyl.radius_px))
        if cyl_idx < len(mi_smooth):
            ax.plot(cyl_mm, mi_smooth[cyl_idx], "rv", markersize=10, zorder=5,
                    label="Detected dip")

        ax.set_xlabel("Radius from cylinder center (mm)")
        ax.set_ylabel("Mean intensity")

        title_color = "#cc0000" if is_flagged else color
        title = f"{arena_name} (cyl={cyl.radius_px:.0f}px = {cyl_mm:.1f}mm)"
        if is_flagged:
            title += "  FLAGGED"
        ax.set_title(title, fontsize=10, fontweight="bold", color=title_color)

        if is_flagged:
            ax.set_facecolor("#fff0f0")

        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Radial Intensity Profiles — {basename}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "radial_profiles.png"), dpi=150)
    plt.close(fig)
