from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .graph import (
    build_skeleton_graph,
    corner_cuts,
    intersection_tangents,
    junction_cuts,
    merge_tangent_foci,
    remove_axis_cuts,
    sample_axis_lines,
    smooth_sampled_graph,
)
from .medial_axis import (
    axis_disc_contacts,
    axis_stroke_width,
    corner_runs,
    extract_medial_axis,
    overlap_spurs,
    transition_points,
)
from .svg import bezier_svg

TANGENT_REACH = 3.0


def binarize(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def process_image(
    input_path: Path,
    output_dir: Path,
    sample_spacing: float,
    stroke_width: float | None,
) -> None:
    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {input_path}")

    binary = binarize(gray)
    axis = extract_medial_axis(binary)
    graph = build_skeleton_graph(axis, binary, 180)
    discs = axis_disc_contacts(binary, axis)
    width = axis_stroke_width(discs) if stroke_width is None else stroke_width
    lines = tuple(edge.pixels for edge in graph.edges)
    runs = corner_runs(discs, lines)
    transitions = transition_points(discs, runs)
    spurs = overlap_spurs(
        discs,
        lines,
        frozenset(
            pixel
            for intersection in graph.intersections
            for pixel in intersection.pixels
        ),
        width / 2,
    )
    junctions = junction_cuts(
        graph, frozenset(point.pixel for point in transitions), spurs
    )
    cuts = junctions + corner_cuts(
        [(run.pixels, run.radius) for run in runs if all(run.bounded)], junctions
    )
    cut_axis = remove_axis_cuts(axis, cuts)
    cut_graph = build_skeleton_graph(cut_axis, None, 180)
    samples = sample_axis_lines(cut_graph.edges, sample_spacing)
    tangents = intersection_tangents(
        cuts, cut_graph.edges, sample_spacing, reach=TANGENT_REACH
    )
    merged = merge_tangent_foci(samples, tangents, sample_spacing)
    smoothed = smooth_sampled_graph(merged)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{input_path.stem}.svg").write_text(
        bezier_svg(axis.shape, smoothed, width)
    )
