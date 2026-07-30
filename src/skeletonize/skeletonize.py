from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .medial_axis import extract_medial_axis


def binarize(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def overlay_edges_axis(
    gray: np.ndarray,
    axis: np.ndarray,
    edge_color: tuple[int, int, int] = (0, 0, 255),
    axis_color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    edges = cv2.Canny(gray, 50.0, 150.0)
    canvas = np.full((*gray.shape, 3), 255, dtype=np.uint8)
    canvas[edges > 0] = edge_color
    canvas[axis < 128] = axis_color
    return canvas

def process_image(input_path: Path, output_dir: Path):
    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {input_path}")

    binary = binarize(gray)
    axis = extract_medial_axis(binary)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    overlay_path = output_dir / f"{stem}-edges-medial-axis-overlay.png"

    cv2.imwrite(str(overlay_path), overlay_edges_axis(gray, axis))
