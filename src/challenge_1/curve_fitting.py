from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from .geometry import (
    AxisPoint,
    BezierCurve,
    chord_parameters,
    densify,
    distance,
    fit_error,
    generate_bezier,
    line_curve,
    signed_angle,
    unit,
    walk,
)


def corner_flags(
    points: Sequence[AxisPoint], threshold: float
) -> tuple[bool, ...]:
    """Say which of a closed path's vertices the path turns a corner at."""
    if len(points) < 3:
        return ()
    return tuple(
        abs(signed_angle(points[index - 1], point, points[(index + 1) % len(points)]))
        >= threshold
        for index, point in enumerate(points)
    )


def _cut_indices(
    points: Sequence[AxisPoint],
    corners: Sequence[bool],
    straight: Sequence[bool],
) -> list[int]:
    """Pick where the loop is broken into sections.

    Its corners, and both ends of every straight run: a straight is a section of
    its own, so it stays the line it was measured to be, and the curve either
    side of it starts where the line stops rather than eating into it.
    """
    size = len(points)
    cuts = {index for index, corner in enumerate(corners) if corner}
    for index, run in enumerate(straight):
        if run:
            cuts.update((index, (index + 1) % size))
    ordered = sorted(cuts)
    if not ordered:
        return [0, size // 2]
    if len(ordered) == 1:
        ordered.append((ordered[0] + size // 2) % size)
        ordered.sort()
    return ordered


def _cut_tangents(
    points: Sequence[AxisPoint],
    corners: Sequence[bool],
    straight: Sequence[bool],
    cuts: Sequence[int],
    span: float,
) -> dict[int, tuple[AxisPoint, AxisPoint]]:
    """Give each cut the direction the path arrives and leaves along.

    A corner turns, so each side is measured on its own. A straight run is its
    own direction, and where one meets a curve the curve takes it too, so the
    curve leaves along the line rather than across it. Anywhere else the two are
    one centred estimate, which is what keeps the sections' tangents equal and
    the join between them smooth.
    """
    tangents = {}
    size = len(points)
    for index in cuts:
        point = points[index]
        arriving = (
            unit(
                (
                    point[0] - points[index - 1][0],
                    point[1] - points[index - 1][1],
                )
            )
            if straight[index - 1]
            else None
        )
        leaving = (
            unit(
                (
                    points[(index + 1) % size][0] - point[0],
                    points[(index + 1) % size][1] - point[1],
                )
            )
            if straight[index]
            else None
        )
        before = walk(points, index, -1, span)
        after = walk(points, index, 1, span)
        if corners[index]:
            incoming = arriving or unit(
                (point[0] - before[0], point[1] - before[1])
            )
            outgoing = leaving or unit(
                (after[0] - point[0], after[1] - point[1])
            )
        elif arriving or leaving:
            incoming = outgoing = arriving or leaving
        else:
            incoming = outgoing = unit(
                (after[0] - before[0], after[1] - before[1])
            )
        tangents[index] = (incoming, outgoing)
    return tangents


def _fit_cubics(
    points: Sequence[AxisPoint],
    start_tangent: AxisPoint,
    end_tangent: AxisPoint,
    tolerance_squared: float,
) -> list[BezierCurve]:
    if len(points) == 2:
        span = distance(points[0], points[1]) / 3
        return [
            BezierCurve(
                points[0],
                (
                    points[0][0] + start_tangent[0] * span,
                    points[0][1] + start_tangent[1] * span,
                ),
                (
                    points[1][0] + end_tangent[0] * span,
                    points[1][1] + end_tangent[1] * span,
                ),
                points[1],
            )
        ]
    parameters = chord_parameters(points)
    curve = generate_bezier(points, parameters, start_tangent, end_tangent)
    error, split = fit_error(points, parameters, curve)
    if error <= tolerance_squared:
        return [curve]
    center = unit(
        (
            points[split - 1][0] - points[split + 1][0],
            points[split - 1][1] - points[split + 1][1],
        )
    )
    if center == (0.0, 0.0):
        center = unit(
            (
                points[split - 1][0] - points[split][0],
                points[split - 1][1] - points[split][1],
            )
        )
    left = _fit_cubics(
        points[: split + 1], start_tangent, center, tolerance_squared
    )
    right = _fit_cubics(
        points[split:], (-center[0], -center[1]), end_tangent, tolerance_squared
    )
    return [*left, *right]


def fit_closed_contour(
    points: Sequence[AxisPoint],
    corners: Sequence[bool],
    straight: Sequence[bool],
    tolerance: float,
    tangent_span: float = 3.0,
) -> tuple[BezierCurve, ...]:
    """Fit an adaptive cubic chain around a smoothed closed contour."""
    path = points
    corner_flags = list(corners)
    runs = list(straight)
    if len(path) < 3 or len(corner_flags) != len(path) or len(runs) != len(path):
        return ()
    cuts = _cut_indices(path, corner_flags, runs)
    tangents = _cut_tangents(path, corner_flags, runs, cuts, tangent_span)
    curves = []
    witness_spacing = min(1.0, max(0.25, tolerance))
    for start, end in pairwise([*cuts, cuts[0]]):
        if runs[start] and end == (start + 1) % len(path):
            curves.append(line_curve(path[start], path[end]))
            continue
        section = (
            path[start : end + 1]
            if start < end
            else [*path[start:], *path[: end + 1]]
        )
        dense = densify(section, witness_spacing)
        start_tangent = tangents[start][1]
        incoming = tangents[end][0]
        curves.extend(
            _fit_cubics(
                dense,
                start_tangent,
                (-incoming[0], -incoming[1]),
                tolerance * tolerance,
            )
        )
    if curves and curves[-1].end != curves[0].start:
        last = curves[-1]
        shift = (
            curves[0].start[0] - last.end[0],
            curves[0].start[1] - last.end[1],
        )
        curves[-1] = BezierCurve(
            last.start,
            last.first_control,
            (last.second_control[0] + shift[0], last.second_control[1] + shift[1]),
            curves[0].start,
        )
    return tuple(curves)
