from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import atan2, ceil, degrees, hypot, radians

AxisPoint = tuple[float, float]


@dataclass(frozen=True)
class BezierCurve:
    start: AxisPoint
    first_control: AxisPoint
    second_control: AxisPoint
    end: AxisPoint


def resample(points: Sequence[AxisPoint], spacing: float) -> tuple[AxisPoint, ...]:
    """Walk a polyline at an even arc length, keeping both of its ends."""
    if len(points) < 2:
        return tuple(points)

    segments = [
        (start, end, hypot(end[0] - start[0], end[1] - start[1]))
        for start, end in pairwise(points)
    ]
    if sum(length for _, _, length in segments) == 0:
        return (points[0],)

    walked = [points[0]]
    target = spacing
    traversed = 0.0
    for start, end, length in segments:
        while target < traversed + length:
            fraction = (target - traversed) / length
            walked.append(
                (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
            )
            target += spacing
        traversed += length

    walked.append(points[-1])
    return tuple(walked)


def catmull_rom(
    points: Sequence[AxisPoint], alpha: float
) -> tuple[BezierCurve, ...]:
    """Fit cubic Bezier spans that interpolate a Catmull-Rom point chain."""
    if len(points) < 2:
        return ()

    chain = list(points)
    closed = len(chain) > 2 and chain[0] == chain[-1]
    padded = (
        [chain[-2], *chain, chain[1]]
        if closed
        else [chain[0], *chain, chain[-1]]
    )
    knots = [0.0]
    for start, end in pairwise(padded):
        step = hypot(end[0] - start[0], end[1] - start[1])
        knots.append(knots[-1] + max(step, 1e-6) ** alpha)

    curves = []
    for index in range(1, len(padded) - 2):
        first, second = padded[index], padded[index + 1]
        before, after = padded[index - 1], padded[index + 2]
        span = knots[index + 1] - knots[index]
        leading = span / (3 * (knots[index + 1] - knots[index - 1]))
        trailing = span / (3 * (knots[index + 2] - knots[index]))
        curves.append(
            BezierCurve(
                start=first,
                first_control=(
                    first[0] + leading * (second[0] - before[0]),
                    first[1] + leading * (second[1] - before[1]),
                ),
                second_control=(
                    second[0] - trailing * (after[0] - first[0]),
                    second[1] - trailing * (after[1] - first[1]),
                ),
                end=second,
            )
        )
    return tuple(curves)


def _cross(first: AxisPoint, second: AxisPoint, third: AxisPoint) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        third[0] - first[0]
    ) * (second[1] - first[1])


def _closed_points(points: Sequence[AxisPoint]) -> list[AxisPoint]:
    unique = []
    for point in points:
        value = (float(point[0]), float(point[1]))
        if not unique or value != unique[-1]:
            unique.append(value)
    if len(unique) > 1 and unique[0] == unique[-1]:
        unique.pop()
    return unique


def _remove_staircase(points: list[AxisPoint]) -> list[AxisPoint]:
    if len(points) < 3:
        return points
    path = [*points, points[0]]
    clockwise = sum(
        start[0] * end[1] - end[0] * start[1]
        for start, end in pairwise(path)
    ) > 0
    result = []
    for index, point in enumerate(path):
        if index == 0 or index == len(path) - 1:
            result.append(point)
            continue
        before = path[index - 1]
        after = path[index + 1]
        before_length = abs(point[0] - before[0]) + abs(point[1] - before[1])
        after_length = abs(point[0] - after[0]) + abs(point[1] - after[1])
        area = _cross(before, point, after)
        if (
            before_length != 1
            and after_length != 1
            or area != 0
            and (area > 0) == clockwise
        ):
            result.append(point)
    return result[:-1]


def _penalty(first: AxisPoint, point: AxisPoint, last: AxisPoint) -> float:
    side_a = hypot(first[0] - point[0], first[1] - point[1])
    side_b = hypot(point[0] - last[0], point[1] - last[1])
    chord = hypot(last[0] - first[0], last[1] - first[1])
    if chord == 0:
        return 0.0
    semiperimeter = (side_a + side_b + chord) / 2
    area_squared = max(
        0.0,
        semiperimeter
        * (semiperimeter - side_a)
        * (semiperimeter - side_b)
        * (semiperimeter - chord),
    )
    return area_squared / chord


def _limit_penalties(points: list[AxisPoint], tolerance: float = 1.0) -> list[AxisPoint]:
    if len(points) < 3:
        return points
    path = [*points, points[0]]
    result = [path[0]]
    last = 0
    for index in range(1, len(path)):
        if index == last + 1:
            continue
        maximum = max(
            _penalty(path[last], path[middle], path[index])
            for middle in range(last + 1, index)
        )
        if maximum >= tolerance:
            last = index - 1
            result.append(path[last])
        if index == len(path) - 1:
            result.append(path[index])
    reduced = result[:-1] if len(result) > 1 and result[0] == result[-1] else result
    return reduced if len(reduced) >= 3 else points


def _remove_collinear(points: list[AxisPoint]) -> list[AxisPoint]:
    current = points
    while len(current) > 3:
        reduced = [
            point
            for index, point in enumerate(current)
            if abs(_cross(current[index - 1], point, current[(index + 1) % len(current)]))
            > 1e-9
        ]
        if len(reduced) < 3 or len(reduced) == len(current):
            break
        current = reduced
    return current


def simplify_staircase(points: Sequence[AxisPoint]) -> tuple[AxisPoint, ...]:
    """Collapse pixel staircases without applying a raster blur."""
    path = _closed_points(points)
    if len(path) < 3:
        return ()
    path = _remove_staircase(path)
    path = _limit_penalties(path)
    path = _remove_collinear(path)
    return (*path, path[0]) if len(path) >= 3 else ()


def _turn(first: AxisPoint, point: AxisPoint, last: AxisPoint) -> float:
    incoming = (point[0] - first[0], point[1] - first[1])
    outgoing = (last[0] - point[0], last[1] - point[1])
    return atan2(
        incoming[0] * outgoing[1] - incoming[1] * outgoing[0],
        incoming[0] * outgoing[0] + incoming[1] * outgoing[1],
    )


def _corner_flags(points: list[AxisPoint], threshold: float) -> list[bool]:
    return [
        abs(degrees(_turn(points[index - 1], point, points[(index + 1) % len(points)])))
        >= threshold
        for index, point in enumerate(points)
    ]


def _subdivide(
    points: list[AxisPoint],
    corners: list[bool],
    segment_length: float,
    outset_ratio: float,
) -> tuple[list[AxisPoint], list[bool], bool]:
    smoothed = []
    new_corners = []
    done = True
    size = len(points)
    for index, point in enumerate(points):
        following = (index + 1) % size
        end = points[following]
        smoothed.append(point)
        new_corners.append(corners[index])
        length = hypot(end[0] - point[0], end[1] - point[1])
        if length <= segment_length:
            continue
        previous = (index - 1) % size
        after = (following + 1) % size
        previous_length = hypot(
            points[previous][0] - point[0], points[previous][1] - point[1]
        )
        following_length = hypot(
            points[after][0] - end[0], points[after][1] - end[1]
        )
        if previous_length / length >= 2 or following_length / length >= 2:
            continue
        if corners[index]:
            previous = index
        if corners[following]:
            after = following
        if previous == index and after == following:
            continue
        outer_midpoint = ((point[0] + end[0]) / 2, (point[1] + end[1]) / 2)
        inner_midpoint = (
            (points[previous][0] + points[after][0]) / 2,
            (points[previous][1] + points[after][1]) / 2,
        )
        inserted = (
            outer_midpoint[0]
            + (outer_midpoint[0] - inner_midpoint[0]) / outset_ratio,
            outer_midpoint[1]
            + (outer_midpoint[1] - inner_midpoint[1]) / outset_ratio,
        )
        smoothed.append(inserted)
        new_corners.append(False)
        if (
            hypot(inserted[0] - point[0], inserted[1] - point[1]) > segment_length
            or hypot(inserted[0] - end[0], inserted[1] - end[1]) > segment_length
        ):
            done = False
    return smoothed, new_corners, done


def subdivide_closed_path(
    points: Sequence[AxisPoint],
    corner_threshold: float,
    segment_length: float = 4.0,
    max_iterations: int = 10,
    outset_ratio: float = 8.0,
) -> tuple[tuple[AxisPoint, ...], tuple[bool, ...]]:
    """Smooth a closed path while keeping detected corners fixed."""
    path = _closed_points(points)
    if len(path) < 3:
        return (), ()
    corners = _corner_flags(path, corner_threshold)
    for _ in range(max_iterations):
        path, corners, done = _subdivide(
            path, corners, segment_length, outset_ratio
        )
        if done:
            break
    return (*path, path[0]), (*corners, corners[0])


def _splice_flags(
    points: list[AxisPoint], corners: Sequence[bool], threshold: float
) -> list[bool]:
    flags = list(corners)
    increasing: bool | None = None
    displacement = 0.0
    for index, point in enumerate(points):
        angle = _turn(points[index - 1], point, points[(index + 1) % len(points)])
        current_increasing = angle >= 0
        if increasing is None:
            increasing = current_increasing
        elif increasing != current_increasing:
            flags[index] = True
            increasing = current_increasing
        displacement += angle
        if abs(displacement) >= radians(threshold):
            flags[index] = True
        if flags[index]:
            displacement = 0.0
    return flags


def _unit(vector: AxisPoint) -> AxisPoint:
    length = hypot(*vector)
    return (vector[0] / length, vector[1] / length) if length else (0.0, 0.0)


def _evaluate(curve: BezierCurve, parameter: float) -> AxisPoint:
    remaining = 1 - parameter
    weights = (
        remaining**3,
        3 * remaining * remaining * parameter,
        3 * remaining * parameter * parameter,
        parameter**3,
    )
    controls = (
        curve.start,
        curve.first_control,
        curve.second_control,
        curve.end,
    )
    return (
        sum(weight * point[0] for weight, point in zip(weights, controls)),
        sum(weight * point[1] for weight, point in zip(weights, controls)),
    )


def _parameters(points: Sequence[AxisPoint]) -> list[float]:
    values = [0.0]
    for first, second in pairwise(points):
        values.append(values[-1] + hypot(second[0] - first[0], second[1] - first[1]))
    if values[-1] == 0:
        return [index / (len(points) - 1) for index in range(len(points))]
    return [value / values[-1] for value in values]


def _generate_bezier(
    points: Sequence[AxisPoint],
    parameters: Sequence[float],
    start_tangent: AxisPoint,
    end_tangent: AxisPoint,
) -> BezierCurve:
    start, end = points[0], points[-1]
    c00 = c01 = c11 = x0 = x1 = 0.0
    for point, parameter in zip(points, parameters):
        remaining = 1 - parameter
        b0 = remaining**3
        b1 = 3 * remaining * remaining * parameter
        b2 = 3 * remaining * parameter * parameter
        b3 = parameter**3
        first = (start_tangent[0] * b1, start_tangent[1] * b1)
        second = (end_tangent[0] * b2, end_tangent[1] * b2)
        baseline = (
            start[0] * (b0 + b1) + end[0] * (b2 + b3),
            start[1] * (b0 + b1) + end[1] * (b2 + b3),
        )
        residual = (point[0] - baseline[0], point[1] - baseline[1])
        c00 += first[0] * first[0] + first[1] * first[1]
        c01 += first[0] * second[0] + first[1] * second[1]
        c11 += second[0] * second[0] + second[1] * second[1]
        x0 += first[0] * residual[0] + first[1] * residual[1]
        x1 += second[0] * residual[0] + second[1] * residual[1]
    determinant = c00 * c11 - c01 * c01
    if abs(determinant) > 1e-12:
        first_length = (x0 * c11 - x1 * c01) / determinant
        second_length = (c00 * x1 - c01 * x0) / determinant
    else:
        first_length = second_length = 0.0
    chord = hypot(end[0] - start[0], end[1] - start[1])
    minimum = chord * 1e-6
    if first_length < minimum or second_length < minimum:
        first_length = second_length = chord / 3
    return BezierCurve(
        start,
        (
            start[0] + start_tangent[0] * first_length,
            start[1] + start_tangent[1] * first_length,
        ),
        (
            end[0] + end_tangent[0] * second_length,
            end[1] + end_tangent[1] * second_length,
        ),
        end,
    )


def _fit_error(
    points: Sequence[AxisPoint],
    parameters: Sequence[float],
    curve: BezierCurve,
) -> tuple[float, int]:
    maximum = 0.0
    split = len(points) // 2
    for index in range(1, len(points) - 1):
        fitted = _evaluate(curve, parameters[index])
        error = (fitted[0] - points[index][0]) ** 2 + (
            fitted[1] - points[index][1]
        ) ** 2
        if error >= maximum:
            maximum = error
            split = index
    return maximum, split


def _fit_cubics(
    points: Sequence[AxisPoint],
    start_tangent: AxisPoint,
    end_tangent: AxisPoint,
    tolerance_squared: float,
) -> list[BezierCurve]:
    if len(points) == 2:
        distance = hypot(
            points[1][0] - points[0][0], points[1][1] - points[0][1]
        ) / 3
        return [
            BezierCurve(
                points[0],
                (
                    points[0][0] + start_tangent[0] * distance,
                    points[0][1] + start_tangent[1] * distance,
                ),
                (
                    points[1][0] + end_tangent[0] * distance,
                    points[1][1] + end_tangent[1] * distance,
                ),
                points[1],
            )
        ]
    parameters = _parameters(points)
    curve = _generate_bezier(points, parameters, start_tangent, end_tangent)
    error, split = _fit_error(points, parameters, curve)
    if error <= tolerance_squared:
        return [curve]
    center = _unit(
        (
            points[split - 1][0] - points[split + 1][0],
            points[split - 1][1] - points[split + 1][1],
        )
    )
    if center == (0.0, 0.0):
        center = _unit(
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


def _densify(points: Sequence[AxisPoint], spacing: float) -> tuple[AxisPoint, ...]:
    dense = [points[0]]
    for start, end in pairwise(points):
        length = hypot(end[0] - start[0], end[1] - start[1])
        divisions = max(1, ceil(length / spacing))
        dense.extend(
            (
                start[0] + (end[0] - start[0]) * index / divisions,
                start[1] + (end[1] - start[1]) * index / divisions,
            )
            for index in range(1, divisions + 1)
        )
    return tuple(dense)


def fit_closed_contour(
    points: Sequence[AxisPoint],
    corners: Sequence[bool],
    tolerance: float,
    splice_threshold: float = 45.0,
) -> tuple[BezierCurve, ...]:
    """Fit an adaptive cubic chain around a smoothed closed contour."""
    path = _closed_points(points)
    corner_flags = list(corners[:-1] if len(corners) == len(path) + 1 else corners)
    if len(path) < 3 or len(corner_flags) != len(path):
        return ()
    splice_flags = _splice_flags(path, corner_flags, splice_threshold)
    cuts = [index for index, splice in enumerate(splice_flags) if splice]
    if not cuts:
        cuts = [0, len(path) // 2]
    elif len(cuts) == 1:
        opposite = (cuts[0] + len(path) // 2) % len(path)
        cuts.append(opposite)
        cuts.sort()
    curves = []
    witness_spacing = min(1.0, max(0.25, tolerance))
    for start, end in pairwise([*cuts, cuts[0]]):
        section = (
            path[start : end + 1]
            if start < end
            else [*path[start:], *path[: end + 1]]
        )
        dense = _densify(section, witness_spacing)
        start_tangent = _unit(
            (dense[1][0] - dense[0][0], dense[1][1] - dense[0][1])
        )
        end_tangent = _unit(
            (dense[-2][0] - dense[-1][0], dense[-2][1] - dense[-1][1])
        )
        curves.extend(
            _fit_cubics(dense, start_tangent, end_tangent, tolerance * tolerance)
        )
    if curves:
        first = curves[0]
        last = curves[-1]
        curves[0] = BezierCurve(
            first.start, first.first_control, first.second_control, first.end
        )
        curves[-1] = BezierCurve(
            last.start, last.first_control, last.second_control, first.start
        )
    return tuple(curves)


def smooth_closed_contour(
    points: Sequence[AxisPoint],
    corner_threshold: float,
    tolerance: float,
) -> tuple[BezierCurve, ...]:
    simplified = simplify_staircase(points)
    smoothed, corners = subdivide_closed_path(simplified, corner_threshold)
    return fit_closed_contour(smoothed, corners, tolerance)


def _curve_flatness(curve: BezierCurve) -> float:
    dx = curve.end[0] - curve.start[0]
    dy = curve.end[1] - curve.start[1]
    length = hypot(dx, dy)
    if length == 0:
        return max(
            hypot(point[0] - curve.start[0], point[1] - curve.start[1])
            for point in (curve.first_control, curve.second_control)
        )
    return max(
        abs(dx * (point[1] - curve.start[1]) - dy * (point[0] - curve.start[0]))
        / length
        for point in (curve.first_control, curve.second_control)
    )


def _split_curve(curve: BezierCurve) -> tuple[BezierCurve, BezierCurve]:
    first = (
        (curve.start[0] + curve.first_control[0]) / 2,
        (curve.start[1] + curve.first_control[1]) / 2,
    )
    middle = (
        (curve.first_control[0] + curve.second_control[0]) / 2,
        (curve.first_control[1] + curve.second_control[1]) / 2,
    )
    last = (
        (curve.second_control[0] + curve.end[0]) / 2,
        (curve.second_control[1] + curve.end[1]) / 2,
    )
    left_middle = ((first[0] + middle[0]) / 2, (first[1] + middle[1]) / 2)
    right_middle = ((middle[0] + last[0]) / 2, (middle[1] + last[1]) / 2)
    center = (
        (left_middle[0] + right_middle[0]) / 2,
        (left_middle[1] + right_middle[1]) / 2,
    )
    return (
        BezierCurve(curve.start, first, left_middle, center),
        BezierCurve(center, right_middle, last, curve.end),
    )


def flatten_curves(
    curves: Sequence[BezierCurve], tolerance: float = 0.25
) -> tuple[AxisPoint, ...]:
    """Flatten a cubic chain closely enough for an antialiased raster preview."""
    if not curves:
        return ()
    points = [curves[0].start]
    pending = [(curve, 0) for curve in reversed(curves)]
    while pending:
        curve, depth = pending.pop()
        if depth >= 16 or _curve_flatness(curve) <= tolerance:
            points.append(curve.end)
            continue
        left, right = _split_curve(curve)
        pending.append((right, depth + 1))
        pending.append((left, depth + 1))
    return tuple(points)
