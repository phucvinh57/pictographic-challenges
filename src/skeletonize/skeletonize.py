from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .centerline import find_centerline_points, overlay_edges_midpoints


def binarize(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def detect_edges(
    gray: np.ndarray, low_threshold: float = 50.0, high_threshold: float = 150.0
) -> np.ndarray:
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return cv2.bitwise_not(edges)


def process_image(
    input_path: Path,
    output_dir: Path,
    width_min_ratio: float = 0.9,
    width_max_ratio: float = 1.1,
) -> tuple[Path, Path, Path]:
    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {input_path}")

    binary = binarize(gray)
    edges = detect_edges(gray)
    midpoints, _ = find_centerline_points(
        binary, width_min_ratio=width_min_ratio, width_max_ratio=width_max_ratio
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    binary_path = output_dir / f"{stem}-binarize.png"
    edges_path = output_dir / f"{stem}-canny-edges.png"
    overlay_path = output_dir / f"{stem}-edges-midpoints-overlay.png"

    cv2.imwrite(str(binary_path), binary)
    cv2.imwrite(str(edges_path), edges)
    cv2.imwrite(str(overlay_path), overlay_edges_midpoints(edges, midpoints))

    return binary_path, edges_path, overlay_path
