"""
Central configuration for the Intelligent Traffic Monitoring system.
Tweak these values to tune the pipeline for a specific video without
touching the processing code.
"""

# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------
RESIZE_WIDTH = 960                # frames are resized to this width (keeps aspect ratio)
GAUSSIAN_BLUR_KERNEL = (5, 5)      # noise reduction kernel
CLAHE_CLIP_LIMIT = 2.0             # contrast-limited adaptive histogram equalization
CLAHE_TILE_GRID_SIZE = (8, 8)

# ---------------------------------------------------------------------------
# Background subtraction / foreground mask cleanup
# ---------------------------------------------------------------------------
BG_HISTORY = 500
BG_VAR_THRESHOLD = 16
BG_DETECT_SHADOWS = True
MORPH_KERNEL_SIZE = (5, 5)
MORPH_OPEN_ITERATIONS = 2
MORPH_CLOSE_ITERATIONS = 2

# ---------------------------------------------------------------------------
# Vehicle detection (contour based)
# ---------------------------------------------------------------------------
MIN_CONTOUR_AREA = 900             # discard blobs smaller than this (noise)
MAX_CONTOUR_AREA = 60000           # discard blobs larger than this (shadows/lighting blobs)
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 4.0

# Rough size-based vehicle-type classification (in pixels, tuned for a
# frame resized to RESIZE_WIDTH). Adjust after inspecting your own footage.
VEHICLE_TYPE_THRESHOLDS = {
    "bike":  (900, 3500),
    "car":   (3500, 12000),
    "bus":   (12000, 60000),
    "truck": (12000, 60000),
}

# ---------------------------------------------------------------------------
# Tracking (centroid tracker)
# ---------------------------------------------------------------------------
MAX_DISAPPEARED_FRAMES = 15        # frames a track can be missing before it's dropped
MAX_MATCH_DISTANCE = 80            # max pixel distance to associate a detection with a track

# ---------------------------------------------------------------------------
# Traffic density thresholds (vehicles currently on screen)
# ---------------------------------------------------------------------------
DENSITY_THRESHOLDS = {
    "LOW": 5,
    "MEDIUM": 12,
    # anything above MEDIUM is HIGH
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_VIDEO_CODEC = "mp4v"
DRAW_TRAJECTORY_LENGTH = 30         # number of past points to draw per vehicle trail
