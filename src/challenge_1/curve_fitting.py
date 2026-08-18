from itertools import pairwise

from common import debug

from .args import get_args
from .constants import FIT_FLOOR, TANGENT_SPAN_FLOOR
from .geometry import (
    AxisPoint,
    BezierCurve,
    Contour,
    chord_parameters,
    densify,
    distance,
    fit_error,
    generate_bezier,
    line_curve,
    perimeter,
    scaled,
    signed_angle,
    unit,
)


def _get_corner_flags(contour: Contour) -> list[bool]:
    """Say which of a closed path's vertices the path turns a corner at."""
    angle_threshold = get_args("corner_angle_threshold")
    if len(contour) < 3:
        return []
    return [
        abs(signed_angle(contour[i - 1], point, contour[(i + 1) % len(contour)]))
        >= angle_threshold
        for i, point in enumerate(contour)
    ]


def _cut_indices(
    contour: Contour,
    corners: list[bool],
    straight_flags: list[bool],
) -> list[int]:
    size = len(contour)
    cuts = {i for i, corner in enumerate(corners) if corner}
    for i, straight in enumerate(straight_flags):
        if straight:
            cuts.update((i, (i + 1) % size))

    ordered = sorted(cuts)
    if not ordered:
        return [0, size // 2]

    if len(ordered) == 1:
        # If there's only one cut, add a second cut opposite it, which cuts the contour in half.
        ordered.append((ordered[0] + size // 2) % size)
        ordered.sort()
    return ordered


def walk(contour: Contour, index: int, step: int, span: float) -> AxisPoint:
    """Follow the ring from a vertex until `span` of arc length is covered."""
    size = len(contour)
    current = index
    travelled = 0.0
    for _ in range(size - 1):
        following = (current + step) % size
        travelled += distance(contour[current], contour[following])
        current = following
        if travelled >= span:
            break
    return contour[current]


def _cut_tangents(
    contour: Contour,
    corners: list[bool],
    straight_flags: list[bool],
    cuts: list[int],
) -> dict[int, tuple[AxisPoint, AxisPoint]]:
    """Give each cut the direction the path arrives and leaves along.

    A corner turns, so each side is measured on its own. A straight run is its
    own direction, and where one meets a curve the curve takes it too, so the
    curve leaves along the line rather than across it. Anywhere else the two are
    one centred estimate, which is what keeps the sections' tangents equal and
    the join between them smooth.
    """
    tangents = {}
    span = scaled(
        get_args("tangent_span_ratio"), TANGENT_SPAN_FLOOR, perimeter(contour)
    )
    size = len(contour)
    for index in cuts:
        point = contour[index]
        arriving = (
            unit(
                (
                    point[0] - contour[index - 1][0],
                    point[1] - contour[index - 1][1],
                )
            )
            if straight_flags[index - 1]
            else None
        )
        leaving = (
            unit(
                (
                    contour[(index + 1) % size][0] - point[0],
                    contour[(index + 1) % size][1] - point[1],
                )
            )
            if straight_flags[index]
            else None
        )
        before = walk(contour, index, -1, span)
        after = walk(contour, index, 1, span)
        if corners[index]:
            incoming = arriving or unit((point[0] - before[0], point[1] - before[1]))
            outgoing = leaving or unit((after[0] - point[0], after[1] - point[1]))
        elif arriving or leaving:
            incoming = outgoing = arriving or leaving
        else:
            incoming = outgoing = unit((after[0] - before[0], after[1] - before[1]))
        tangents[index] = (incoming, outgoing)
    return tangents


def _fit_cubics(
    contour: Contour,
    start_tangent: AxisPoint,
    end_tangent: AxisPoint,
    tolerance_squared: float,
) -> list[BezierCurve]:
    if len(contour) == 2:
        span = distance(contour[0], contour[1]) / 3
        return [
            BezierCurve(
                contour[0],
                (
                    contour[0][0] + start_tangent[0] * span,
                    contour[0][1] + start_tangent[1] * span,
                ),
                (
                    contour[1][0] + end_tangent[0] * span,
                    contour[1][1] + end_tangent[1] * span,
                ),
                contour[1],
            )
        ]
    parameters = chord_parameters(contour)
    curve = generate_bezier(contour, parameters, start_tangent, end_tangent)
    error, split = fit_error(contour, parameters, curve)
    if error <= tolerance_squared:
        return [curve]
    center = unit(
        (
            contour[split - 1][0] - contour[split + 1][0],
            contour[split - 1][1] - contour[split + 1][1],
        )
    )
    if center == (0.0, 0.0):
        center = unit(
            (
                contour[split - 1][0] - contour[split][0],
                contour[split - 1][1] - contour[split][1],
            )
        )
    left = _fit_cubics(contour[: split + 1], start_tangent, center, tolerance_squared)
    right = _fit_cubics(
        contour[split:], (-center[0], -center[1]), end_tangent, tolerance_squared
    )
    return [*left, *right]


def fit_closed_contour(
    contour: Contour,
    straight_flags: list[bool],
) -> list[BezierCurve]:
    """Fit a closed contour with a sequence of Bezier curves."""
    size = len(contour)
    if size < 3:
        return []

    tolerance = scaled(get_args("fit_ratio"), FIT_FLOOR, perimeter(contour))
    corner_flags = _get_corner_flags(contour)
    cuts = _cut_indices(contour, corner_flags, straight_flags)
    tangents = _cut_tangents(contour, corner_flags, straight_flags, cuts)
    debug.count("corners", sum(corner_flags))
    debug.count("cuts", len(cuts))

    curves = []
    witness_spacing = min(1.0, max(0.25, tolerance))
    for start, end in pairwise([*cuts, cuts[0]]):
        if straight_flags[start] and end == (start + 1) % size:
            curves.append(line_curve(contour[start], contour[end]))
            continue
        section = (
            contour[start : end + 1]
            if start < end
            else contour[start:] + contour[: end + 1]
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
    debug.count("fitted curves", len(curves))
    return curves


__all__ = ["fit_closed_contour"]
