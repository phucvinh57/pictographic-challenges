from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .graph import (
    IntersectionTangents,
    SampledGraph,
    build_skeleton_graph,
    intersection_tangents,
    merge_tangent_foci,
    remove_intersection_neighborhoods,
    sample_axis_lines,
)
from .medial_axis import extract_medial_axis

TANGENT_REACH = 3.0


def binarize(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def draw_axis_samples(
    axis: np.ndarray, samples: tuple[tuple[tuple[float, float], ...], ...]
) -> np.ndarray:
    """Render the axis in black with its sampled points highlighted in red."""
    canvas = cv2.cvtColor(axis, cv2.COLOR_GRAY2BGR)
    for edge_samples in samples:
        for x, y in edge_samples:
            cv2.circle(canvas, (round(x), round(y)), 2, (0, 0, 255), -1)
    return canvas


def draw_intersection_tangents(
    axis: np.ndarray,
    junctions: tuple[IntersectionTangents, ...],
) -> np.ndarray:
    """Render each junction's tangent crossings and their focus."""
    canvas = cv2.cvtColor(axis, cv2.COLOR_GRAY2BGR)
    scale = max(1, min(axis.shape) // 512)
    for junction in junctions:
        cv2.circle(
            canvas,
            junction.center,
            round(junction.radius),
            (190, 190, 190),
            scale,
            cv2.LINE_AA,
        )
        if junction.focus is not None:
            x, y = junction.focus
            cv2.circle(
                canvas, (round(x), round(y)), 3 * scale, (0, 0, 255), -1, cv2.LINE_AA
            )
    return canvas


def draw_sampled_graph(shape: tuple[int, int], graph: SampledGraph) -> np.ndarray:
    """Render the merged graph as straight segments between its points."""
    canvas = np.full((*shape, 3), 255, dtype=np.uint8)
    scale = max(1, min(shape) // 512)
    for start, end in graph.segments:
        first, second = graph.points[start], graph.points[end]
        cv2.line(
            canvas,
            (round(first[0]), round(first[1])),
            (round(second[0]), round(second[1])),
            (0, 0, 0),
            scale,
            cv2.LINE_AA,
        )
    return canvas


def process_image(
    input_path: Path,
    output_dir: Path,
    sample_spacing: float = 10.0,
    cut_radius_scale: float = 1.0,
) -> None:
    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {input_path}")

    binary = binarize(gray)
    axis = extract_medial_axis(binary)
    graph = build_skeleton_graph(axis, binary, cut_radius_scale)
    cut_axis = remove_intersection_neighborhoods(axis, graph.intersections)
    cut_graph = build_skeleton_graph(cut_axis)
    samples = sample_axis_lines(cut_graph.edges, sample_spacing)
    junctions = intersection_tangents(
        graph.intersections, cut_graph.edges, sample_spacing, reach=TANGENT_REACH
    )
    merged = merge_tangent_foci(samples, junctions)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    cv2.imwrite(str(output_dir / f"{stem}-1-binarize.png"), binary)
    cv2.imwrite(str(output_dir / f"{stem}-2-skeleton.png"), axis)
    cv2.imwrite(str(output_dir / f"{stem}-3-skeleton-cut-intersections.png"), cut_axis)
    cv2.imwrite(
        str(output_dir / f"{stem}-4-skeleton-samples.png"),
        draw_axis_samples(cut_axis, samples),
    )
    cv2.imwrite(
        str(output_dir / f"{stem}-5-tangent-crossings.png"),
        draw_intersection_tangents(cut_axis, junctions),
    )
    cv2.imwrite(
        str(output_dir / f"{stem}-6-merged-graph.png"),
        draw_sampled_graph(axis.shape, merged),
    )
