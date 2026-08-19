from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IN_DIR = Path("input/challenge_2")
OUT_DIR = Path("output/challenge_2")

CANNY_LOW = 50
CANNY_HIGH = 150
MIN_AREA = 64
TRANSITION_DELTA = 0.05
TRANSITION_FLATNESS = 0.3
SAMPLE_SPACING = 8
STROKE_WIDTH = 64.0
