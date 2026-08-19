from itertools import pairwise

from common import debug

from .geometry import (
    chord_parameters,
    densify,
    distance,
    fit_error,
    generate_bezier,
    line_curve,
    path_length,
    scaled,
    signed_angle,
    unit,
)
from .settings import (
    DEFAULT_VECTORIZATION_SETTINGS,
    FIT_FLOOR,
    TANGENT_SPAN_FLOOR,
    VectorizationSettings,
)
from .types import AxisPoint, BezierCurve, ProcessedContour


def _corner_flags(
    points: tuple[AxisPoint, ...], closed: bool, angle_threshold: float
) -> list[bool]:
    if len(points) < 3:
        return [False] * len(points)
    if closed:
        return [
            abs(signed_angle(points[index - 1], point, points[(index + 1) % len(points)]))
            >= angle_threshold
            for index, point in enumerate(points)
        ]
    return [False] + [
        abs(signed_angle(points[index - 1], points[index], points[index + 1]))
        >= angle_threshold
        for index in range(1, len(points) - 1)
    ] + [False]


def _cut_indices(
    points: tuple[AxisPoint, ...],
    corners: list[bool],
    straight_flags: tuple[bool, ...],
    closed: bool,
) -> list[int]:
    size = len(points)
    cuts = {index for index, corner in enumerate(corners) if corner}
    for index, straight in enumerate(straight_flags):
        if straight:
            cuts.update((index, (index + 1) % size))
    if not closed:
        cuts.update((0, size - 1))
    ordered = sorted(cuts)
    if not ordered:
        return [0, size // 2]
    if closed and len(ordered) == 1:
        ordered.append((ordered[0] + size // 2) % size)
        ordered.sort()
    return ordered


def _walk(
    points: tuple[AxisPoint, ...],
    index: int,
    step: int,
    span: float,
    closed: bool,
) -> AxisPoint:
    size = len(points)
    current = index
    travelled = 0.0
    for _ in range(size - 1):
        following = (current + step) % size if closed else current + step
        if not 0 <= following < size:
            break
        travelled += distance(points[current], points[following])
        current = following
        if travelled >= span:
            break
    return points[current]


def _cut_tangents(
    contour: ProcessedContour,
    corners: list[bool],
    cuts: list[int],
    settings: VectorizationSettings,
) -> dict[int, tuple[AxisPoint, AxisPoint]]:
    points = contour.points
    straight_flags = contour.straight_flags
    span = scaled(
        settings.tangent_span_ratio,
        TANGENT_SPAN_FLOOR,
        path_length(points, contour.closed),
    )
    size = len(points)
    tangents = {}
    for index in cuts:
        point = points[index]
        arriving = (
            unit(
                (
                    point[0] - points[index - 1][0],
                    point[1] - points[index - 1][1],
                )
            )
            if (contour.closed or index > 0) and straight_flags[index - 1]
            else None
        )
        leaving = (
            unit(
                (
                    points[(index + 1) % size][0] - point[0],
                    points[(index + 1) % size][1] - point[1],
                )
            )
            if (contour.closed or index < size - 1)
            and index < len(straight_flags)
            and straight_flags[index]
            else None
        )
        before = _walk(points, index, -1, span, contour.closed)
        after = _walk(points, index, 1, span, contour.closed)
        if corners[index]:
            incoming = arriving or unit((point[0] - before[0], point[1] - before[1]))
            outgoing = leaving or unit((after[0] - point[0], after[1] - point[1]))
        elif arriving or leaving:
            incoming = outgoing = arriving or leaving
        elif not contour.closed and index == 0:
            incoming = outgoing = unit((after[0] - point[0], after[1] - point[1]))
        elif not contour.closed and index == size - 1:
            incoming = outgoing = unit((point[0] - before[0], point[1] - before[1]))
        else:
            incoming = outgoing = unit((after[0] - before[0], after[1] - before[1]))
        tangents[index] = (incoming, outgoing)
    return tangents


def _fit_cubics(
    points: list[AxisPoint],
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
    left = _fit_cubics(points[: split + 1], start_tangent, center, tolerance_squared)
    right = _fit_cubics(
        points[split:], (-center[0], -center[1]), end_tangent, tolerance_squared
    )
    return [*left, *right]


def fit_contour(
    contour: ProcessedContour,
    settings: VectorizationSettings = DEFAULT_VECTORIZATION_SETTINGS,
) -> list[BezierCurve]:
    size = len(contour.points)
    minimum = 3 if contour.closed else 2
    if size < minimum:
        return []

    tolerance = scaled(
        settings.fit_ratio,
        FIT_FLOOR,
        path_length(contour.points, contour.closed),
    )
    corners = _corner_flags(
        contour.points, contour.closed, settings.corner_angle_threshold
    )
    cuts = _cut_indices(
        contour.points, corners, contour.straight_flags, contour.closed
    )
    tangents = _cut_tangents(contour, corners, cuts, settings)
    debug.count("corners", sum(corners))
    debug.count("cuts", len(cuts))

    curves = []
    witness_spacing = min(1.0, max(0.25, tolerance))
    ends = [*cuts, cuts[0]] if contour.closed else cuts
    for start, end in pairwise(ends):
        if contour.straight_flags[start] and end == (start + 1) % size:
            curves.append(line_curve(contour.points[start], contour.points[end]))
            continue
        section = (
            contour.points[start : end + 1]
            if start < end
            else contour.points[start:] + contour.points[: end + 1]
        )
        dense = densify(section, witness_spacing)
        incoming = tangents[end][0]
        curves.extend(
            _fit_cubics(
                dense,
                tangents[start][1],
                (-incoming[0], -incoming[1]),
                tolerance * tolerance,
            )
        )
    if contour.closed and curves and curves[-1].end != curves[0].start:
        last = curves[-1]
        shift = (
            curves[0].start[0] - last.end[0],
            curves[0].start[1] - last.end[1],
        )
        curves[-1] = BezierCurve(
            last.start,
            last.first_control,
            (
                last.second_control[0] + shift[0],
                last.second_control[1] + shift[1],
            ),
            curves[0].start,
        )
    debug.count("fitted curves", len(curves))
    return curves
