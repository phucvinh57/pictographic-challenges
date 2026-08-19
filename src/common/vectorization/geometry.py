from collections.abc import Sequence
from itertools import pairwise
from math import atan2, ceil, degrees, hypot

from .types import AxisPoint, BezierCurve


def distance(first: AxisPoint, second: AxisPoint) -> float:
    return hypot(second[0] - first[0], second[1] - first[1])


def path_length(points: Sequence[AxisPoint], closed: bool) -> float:
    return sum(distance(first, second) for first, second in pairwise(points)) + (
        distance(points[-1], points[0]) if closed and len(points) > 1 else 0.0
    )


def scaled(ratio: float, floor: float, scale: float) -> float:
    return max(floor, ratio * scale)


def cross_product(a: AxisPoint, b: AxisPoint, c: AxisPoint) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


def unit(vector: AxisPoint) -> AxisPoint:
    length = hypot(*vector)
    return (vector[0] / length, vector[1] / length) if length else (0.0, 0.0)


def signed_angle(a: AxisPoint, b: AxisPoint, c: AxisPoint) -> float:
    ab = (b[0] - a[0], b[1] - a[1])
    bc = (c[0] - b[0], c[1] - b[1])
    return degrees(
        atan2(
            ab[0] * bc[1] - ab[1] * bc[0],
            ab[0] * bc[0] + ab[1] * bc[1],
        )
    )


def offset(point: AxisPoint, line: tuple[AxisPoint, AxisPoint]) -> float:
    start, end = line
    length = distance(start, end)
    if length == 0:
        return distance(start, point)
    return abs(cross_product(start, end, point)) / length


def densify(points: Sequence[AxisPoint], spacing: float) -> list[AxisPoint]:
    dense = [points[0]]
    for start, end in pairwise(points):
        length = distance(start, end)
        divisions = max(1, ceil(length / spacing))
        dense.extend(
            (
                start[0] + (end[0] - start[0]) * index / divisions,
                start[1] + (end[1] - start[1]) * index / divisions,
            )
            for index in range(1, divisions + 1)
        )
    return dense


def chord_parameters(points: Sequence[AxisPoint]) -> list[float]:
    values = [0.0]
    for first, second in pairwise(points):
        values.append(values[-1] + distance(first, second))
    if values[-1] == 0:
        return [index / (len(points) - 1) for index in range(len(points))]
    return [value / values[-1] for value in values]


def generate_bezier(
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
    chord = distance(start, end)
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


def fit_error(
    points: Sequence[AxisPoint],
    parameters: Sequence[float],
    curve: BezierCurve,
) -> tuple[float, int]:
    start_x, start_y = curve.start
    first_x, first_y = curve.first_control
    second_x, second_y = curve.second_control
    end_x, end_y = curve.end
    maximum = 0.0
    split = len(points) // 2
    for index in range(1, len(points) - 1):
        parameter = parameters[index]
        remaining = 1 - parameter
        start_weight = remaining**3
        first_weight = 3 * remaining * remaining * parameter
        second_weight = 3 * remaining * parameter * parameter
        end_weight = parameter**3
        point_x, point_y = points[index]
        gap_x = (
            start_weight * start_x
            + first_weight * first_x
            + second_weight * second_x
            + end_weight * end_x
        ) - point_x
        gap_y = (
            start_weight * start_y
            + first_weight * first_y
            + second_weight * second_y
            + end_weight * end_y
        ) - point_y
        error = gap_x * gap_x + gap_y * gap_y
        if error >= maximum:
            maximum = error
            split = index
    return maximum, split


def line_curve(start: AxisPoint, end: AxisPoint) -> BezierCurve:
    return BezierCurve(
        start,
        (start[0] + (end[0] - start[0]) / 3, start[1] + (end[1] - start[1]) / 3),
        (end[0] - (end[0] - start[0]) / 3, end[1] - (end[1] - start[1]) / 3),
        end,
    )
