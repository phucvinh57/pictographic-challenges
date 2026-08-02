from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .graph import build_skeleton_graph, remove_intersection_neighborhoods
from .medial_axis import estimate_stroke_width, extract_medial_axis
from .sharpen import (
    DEFAULT_SHARPENING_PARAMETERS,
    SharpeningParameters,
    sharpen_medial_axis,
)
from .vectorize import polylines_to_svg, rasterize_polylines


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

def process_image(
    input_path: Path,
    output_dir: Path,
    parameters: SharpeningParameters = DEFAULT_SHARPENING_PARAMETERS,
) -> None:
    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {input_path}")

    binary = binarize(gray)
    axis = extract_medial_axis(binary)
    stroke_width = estimate_stroke_width(binary, axis)
    graph = build_skeleton_graph(axis, binary)
    cut_axis = remove_intersection_neighborhoods(axis, graph.intersections)
    geometry = sharpen_medial_axis(axis, binary, stroke_width, parameters, graph=graph)
    sharpened = rasterize_polylines(geometry.polylines, geometry.width, geometry.height)
    svg = polylines_to_svg(geometry.polylines, geometry.width, geometry.height, stroke_width)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    cv2.imwrite(str(output_dir / f"{stem}-binarize.png"), binary)
    cv2.imwrite(str(output_dir / f"{stem}-skeleton.png"), axis)
    cv2.imwrite(str(output_dir / f"{stem}-skeleton-cut-intersections.png"), cut_axis)
    cv2.imwrite(str(output_dir / f"{stem}-sharpened.png"), sharpened)
    cv2.imwrite(
        str(output_dir / f"{stem}-edges-medial-axis-overlay.png"),
        overlay_edges_axis(gray, axis),
    )
    (output_dir / f"{stem}.svg").write_text(svg)
