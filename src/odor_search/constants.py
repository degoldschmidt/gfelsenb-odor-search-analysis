"""Shared constants for Drosophila tracking experiments."""

ARENA_NAMES = ["topleft", "topright", "bottomleft", "bottomright"]
NODE_NAMES = ["Head", "Thorax", "Abdomen_tip", "Wing_L", "Wing_R"]
HEAD_IDX = 0
THORAX_IDX = 1
N_NODES = len(NODE_NAMES)
ARENA_DIAMETER_MM = 75.0
ARENA_RADIUS_MM = ARENA_DIAMETER_MM / 2.0

FLY_COLORS_BGR = {
    "topleft": (255, 100, 100),
    "topright": (100, 255, 100),
    "bottomleft": (100, 100, 255),
    "bottomright": (255, 255, 100),
}

FLY_COLORS_HEX = {
    "topleft": "#6464FF",
    "topright": "#64FF64",
    "bottomleft": "#FF6464",
    "bottomright": "#64FFFF",
}

ARENA_COLORS = {
    "topleft": "#3498db",
    "topright": "#2ecc71",
    "bottomleft": "#e74c3c",
    "bottomright": "#f39c12",
}
