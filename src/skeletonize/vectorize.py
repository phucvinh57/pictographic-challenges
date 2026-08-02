from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .graph import build_skeleton_graph
from .sharpen import Polyline, build_polylines, prune_short_branches

Point = tuple[float, float]
PixelPoint = tuple[int, int]


def trace_medial_axis(axis: np.ndarray) -> list[list[PixelPoint]]:
    """Trace a one-pixel medial axis into paths between ends and junctions."""
    graph = build_skeleton_graph(axis)
    return [list(edge.pixels) for edge in graph.edges]


def _simplify(points: Sequence[Point], epsilon: float) -> list[Point]:
    if len(points) < 3:
        return [(float(x), float(y)) for x, y in points]
    closed = points[0] == points[-1]
    array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(array, epsilon, closed)
    return [(float(x), float(y)) for x, y in simplified.reshape(-1, 2)]


def _simplify_preserving(
    points: Sequence[Point], sharp_indices: frozenset[int], epsilon: float
) -> tuple[list[Point], frozenset[int]]:
    """Simplify while keeping every sharp vertex exactly, at a known index."""
    count = len(points)
    if count == 0:
        return [], frozenset()
    if not sharp_indices:
        return _simplify(points, epsilon), frozenset()

    boundaries = sorted(set(sharp_indices) | {0, count - 1})
    result: list[Point] = []
    new_sharp: set[int] = set()

    for position in range(len(boundaries) - 1):
        start = boundaries[position]
        end = boundaries[position + 1]
        run = list(points[start : end + 1])
        simplified_run = _simplify(run, epsilon)
        if position == 0:
            if start in sharp_indices:
                new_sharp.add(0)
        else:
            simplified_run = simplified_run[1:]
        if simplified_run and end in sharp_indices:
            new_sharp.add(len(result) + len(simplified_run) - 1)
        result.extend(simplified_run)

    return result, frozenset(new_sharp)


def _catmull_rom_controls(
    previous: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    following: np.ndarray,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Centripetal Catmull-Rom tangents, robust to non-uniform point spacing.

    A uniform-parameter tangent (start + (end - previous) / 6) assumes evenly
    spaced points; after Douglas-Peucker simplification a short segment can
    sit next to a much longer one, and that assumption makes the tangent
    overshoot into a visible loop. Centripetal parametrization scales each
    tangent by actual point distances instead, which eliminates that loop.
    """

    def knot_delta(a: np.ndarray, b: np.ndarray) -> float:
        distance = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        return max(distance, 1e-6) ** alpha

    t0 = 0.0
    t1 = t0 + knot_delta(previous, start)
    t2 = t1 + knot_delta(start, end)
    t3 = t2 + knot_delta(end, following)

    m1 = (t2 - t1) * (
        (start - previous) / (t1 - t0)
        - (end - previous) / (t2 - t0)
        + (end - start) / (t2 - t1)
    )
    m2 = (t2 - t1) * (
        (end - start) / (t2 - t1)
        - (following - start) / (t3 - t1)
        + (following - end) / (t3 - t2)
    )
    return start + m1 / 3.0, end - m2 / 3.0


def _bezier_segments(
    points: list[Point], closed: bool, sharp_indices: frozenset[int] = frozenset()
) -> list[tuple[Point, Point, Point, Point]]:
    array = np.asarray(points, dtype=np.float64)
    count = len(array)
    if count < 2:
        return []
    if count == 2:
        start, end = array
        return [(tuple(start), tuple(start), tuple(end), tuple(end))]

    def at(index: int) -> np.ndarray:
        if closed:
            return array[index % count]
        return array[min(max(index, 0), count - 1)]

    segments = []
    for index in range(count if closed else count - 1):
        previous, start, end, following = (
            at(index - 1),
            at(index),
            at(index + 1),
            at(index + 2),
        )
        next_index = (index + 1) % count
        raw_control_1, raw_control_2 = _catmull_rom_controls(previous, start, end, following)
        control_1 = start if index in sharp_indices else raw_control_1
        control_2 = end if next_index in sharp_indices else raw_control_2
        segments.append(
            (tuple(start), tuple(control_1), tuple(control_2), tuple(end))
        )
    return segments


def _flatten_bezier(
    points: list[Point], closed: bool, sharp_indices: frozenset[int], samples: int
) -> list[Point]:
    segments = _bezier_segments(points, closed, sharp_indices)
    if not segments:
        return list(points)

    flattened: list[Point] = []
    parameter = np.linspace(0.0, 1.0, samples)
    for segment_index, (start, control_1, control_2, end) in enumerate(segments):
        p0, p1, p2, p3 = (np.array(point) for point in (start, control_1, control_2, end))
        curve = (
            np.outer((1 - parameter) ** 3, p0)
            + np.outer(3 * (1 - parameter) ** 2 * parameter, p1)
            + np.outer(3 * (1 - parameter) * parameter**2, p2)
            + np.outer(parameter**3, p3)
        )
        values = [(float(x), float(y)) for x, y in curve]
        flattened.extend(values[1:] if segment_index > 0 else values)
    return flattened


def _svg_path(points: list[Point], closed: bool, sharp_indices: frozenset[int] = frozenset()) -> str:
    segments = _bezier_segments(points, closed, sharp_indices)
    if not segments:
        return ""
    start = segments[0][0]
    commands = [f"M {start[0]:.2f} {start[1]:.2f}"]
    for _, control_1, control_2, end in segments:
        commands.append(
            f"C {control_1[0]:.2f} {control_1[1]:.2f} "
            f"{control_2[0]:.2f} {control_2[1]:.2f} "
            f"{end[0]:.2f} {end[1]:.2f}"
        )
    if closed:
        commands.append("Z")
    return " ".join(commands)


def polylines_to_svg(
    polylines: Sequence[Polyline],
    width: int,
    height: int,
    stroke_width: float,
    simplify_epsilon: float = 1.5,
    stroke_color: str = "black",
    background_color: str = "white",
) -> str:
    path_data = []
    for polyline in polylines:
        if len(polyline.points) < 2:
            continue
        simplified, sharp_indices = _simplify_preserving(
            polyline.points, polyline.sharp_indices, simplify_epsilon
        )
        data = _svg_path(simplified, polyline.closed, sharp_indices)
        if data:
            path_data.append(data)

    paths = "\n".join(
        f'  <path d="{data}" fill="none" stroke="{stroke_color}" '
        f'stroke-width="{stroke_width:.2f}" stroke-linecap="round" '
        f'stroke-linejoin="round" />'
        for data in path_data
    )
    background = f'  <rect width="{width}" height="{height}" fill="{background_color}" />'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n{background}\n{paths}\n</svg>\n'
    )


def rasterize_polylines(
    polylines: Sequence[Polyline],
    width: int,
    height: int,
    simplify_epsilon: float = 1.5,
    samples_per_segment: int = 12,
) -> np.ndarray:
    """Rasterize the same cubic Béziers polylines_to_svg emits, for comparison."""
    canvas = np.full((height, width), 255, dtype=np.uint8)
    for polyline in polylines:
        if len(polyline.points) < 2:
            continue
        simplified, sharp_indices = _simplify_preserving(
            polyline.points, polyline.sharp_indices, simplify_epsilon
        )
        flattened = _flatten_bezier(simplified, polyline.closed, sharp_indices, samples_per_segment)
        if len(flattened) < 2:
            continue
        contour = np.round(np.asarray(flattened, dtype=np.float64)).astype(np.int32)
        cv2.polylines(canvas, [contour], polyline.closed, 0, 1, cv2.LINE_8)
    return canvas


def medial_axis_to_svg(
    axis: np.ndarray,
    stroke_width: float,
    simplify_epsilon: float = 1.5,
    stroke_color: str = "black",
    background_color: str = "white",
) -> str:
    """Render the traced medial axis using the estimated original stroke width."""
    graph = build_skeleton_graph(axis)
    polylines = build_polylines(graph)
    polylines = prune_short_branches(polylines, minimum_length=stroke_width * 0.5)
    return polylines_to_svg(
        polylines,
        graph.width,
        graph.height,
        stroke_width,
        simplify_epsilon,
        stroke_color,
        background_color,
    )
