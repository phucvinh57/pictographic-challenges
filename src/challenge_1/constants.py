from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IN_DIR = Path("input/challenge_1")
OUT_DIR = Path("output/challenge_1")

# Every length threshold is a fraction of the contour's own perimeter, floored at
# a pixel count. The ratio keeps a threshold proportional to the shape, so the same
# artwork traced at any resolution yields the same curves. The floor stops a tiny
# contour from resolving below rasterisation noise: thresholding plus marching
# squares place an edge within about half a pixel whatever the image size.
# Angles and the bow ratio are dimensionless already, so they need neither.

SIMPLIFY_RATIO = 0.0002
SIMPLIFY_FLOOR = 0.35

BREAK_SPAN_RATIO = 0.01
BREAK_SPAN_FLOOR = 4.0
BREAK_ANGLE_THRESHOLD = 30.0

STRAIGHT_MIN_RATIO = 0.0067
STRAIGHT_MIN_FLOOR = 4.0
STRAIGHT_TOLERANCE_RATIO = 0.00083
STRAIGHT_TOLERANCE_FLOOR = 1.0
STRAIGHT_BOW_RATIO = 0.01
DOMINANT_RATIO = 0.053
DOMINANT_FLOOR = 16.0

CORNER_ANGLE_THRESHOLD = 60.0
FIT_RATIO = 0.000625
FIT_FLOOR = 0.5
TANGENT_SPAN_RATIO = 0.0025
TANGENT_SPAN_FLOOR = 2.0

LINE_TOLERANCE = 0.005
