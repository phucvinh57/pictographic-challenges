"""Binarize an image and thin it with the Zhang-Suen algorithm (via OpenCV)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .vectorize import skeleton_to_svg, write_svg


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu-threshold a grayscale image.

    Returns a binary image where foreground (ink) is 0 (black) and
    background is 255 (white).
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def thin_zhang_suen(binary: np.ndarray) -> np.ndarray:
    """Skeletonize a binary image using the Zhang-Suen thinning algorithm.

    `binary` must have foreground=0 (black) / background=255 (white), matching
    `binarize`'s output. cv2.ximgproc.thinning expects the foreground in
    white, so the image is inverted before and after thinning.
    """
    foreground_white = cv2.bitwise_not(binary)
    skeleton = cv2.ximgproc.thinning(
        foreground_white, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
    )
    return cv2.bitwise_not(skeleton)


def estimate_stroke_half_width(binary: np.ndarray, skeleton: np.ndarray) -> float:
    """Median distance from skeleton pixels to the nearest background pixel.

    Used to auto-scale spur pruning to the image's own ink thickness: Zhang-
    Suen thinning leaves spurious branches roughly on the order of the
    stroke's half-width, regardless of the image's absolute resolution.
    """
    foreground = (binary < 128).astype(np.uint8)
    distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    skeleton_mask = skeleton < 128
    if not np.any(skeleton_mask):
        return 0.0
    return float(np.median(distance[skeleton_mask]))


def process_image(
    input_path: Path, output_dir: Path, stroke_width: float = 45.0
) -> tuple[Path, Path, Path]:
    """Binarize + thin one image, writing `<stem>-binarize.png`,
    `<stem>-ZhangSuen-skeletonize.png`, and a vectorized `<stem>-ZhangSuen.svg`
    (fixed `stroke_width`, centerline traced from the skeleton) into
    `output_dir`.
    """
    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {input_path}")

    binary = binarize(gray)
    skeleton = thin_zhang_suen(binary)
    half_width = estimate_stroke_half_width(binary, skeleton)
    svg = skeleton_to_svg(
        skeleton, stroke_width=stroke_width, min_spur_length=0.5 * half_width
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    binary_path = output_dir / f"{stem}-binarize.png"
    skeleton_path = output_dir / f"{stem}-ZhangSuen-skeletonize.png"
    svg_path = output_dir / f"{stem}-ZhangSuen.svg"

    cv2.imwrite(str(binary_path), binary)
    cv2.imwrite(str(skeleton_path), skeleton)
    write_svg(svg, svg_path)

    return binary_path, skeleton_path, svg_path
