from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .centerline import find_centerline_points, overlay_edges_midpoints
from .vectorize import overlay_vector_paths, vectorize_midpoints


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
) -> tuple[Path, Path, Path, Path, Path]:
    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {input_path}")

    binary = binarize(gray)
    edges = detect_edges(gray)
    midpoints, stroke_width = find_centerline_points(
        binary, width_min_ratio=width_min_ratio, width_max_ratio=width_max_ratio
    )
    svg, vector_paths = vectorize_midpoints(midpoints, gray.shape, stroke_width)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    binary_path = output_dir / f"{stem}-binarize.png"
    edges_path = output_dir / f"{stem}-canny-edges.png"
    overlay_path = output_dir / f"{stem}-edges-midpoints-overlay.png"
    vector_overlay_path = output_dir / f"{stem}-vector-skeleton-overlay.png"
    svg_path = output_dir / f"{stem}.svg"

    cv2.imwrite(str(binary_path), binary)
    cv2.imwrite(str(edges_path), edges)
    cv2.imwrite(str(overlay_path), overlay_edges_midpoints(edges, midpoints))
    cv2.imwrite(str(vector_overlay_path), overlay_vector_paths(edges, vector_paths))
    svg_path.write_text(svg)

    return binary_path, edges_path, overlay_path, vector_overlay_path, svg_path
