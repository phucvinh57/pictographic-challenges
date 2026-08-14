from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import atan2, ceil, degrees, hypot

AxisPoint = tuple[float, float]


def distance(first: AxisPoint, second: AxisPoint) -> float:
    """|AB|"""
    return hypot(second[0] - first[0], second[1] - first[1])


def cross_product(a: AxisPoint, b: AxisPoint, c: AxisPoint) -> float:
    """AB x AC"""
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


def unit(vector: AxisPoint) -> AxisPoint:
    length = hypot(*vector)
    return (vector[0] / length, vector[1] / length) if length else (0.0, 0.0)


def signed_angle(a: AxisPoint, b: AxisPoint, c: AxisPoint) -> float:
    """Signed angle in degrees between the vectors AB and BC."""
    ab = (b[0] - a[0], b[1] - a[1])
    bc = (c[0] - b[0], c[1] - b[1])
    return degrees(
        atan2(
            ab[0] * bc[1] - ab[1] * bc[0],
            ab[0] * bc[0] + ab[1] * bc[1],
        )
    )


@dataclass(frozen=True)
class BezierCurve:
    """A cubic Bézier span represented by its four control points."""

    start: AxisPoint
    first_control: AxisPoint
    second_control: AxisPoint
    end: AxisPoint


def offset(point: AxisPoint, line: tuple[AxisPoint, AxisPoint]) -> float:
    """Perpendicular distance from a point to the line through A and B."""
    start, end = line
    length = distance(start, end)
    if length == 0:
        return distance(start, point)
    return abs(cross_product(start, end, point)) / length


def walk(points: Sequence[AxisPoint], index: int, step: int, span: float) -> AxisPoint:
    """Follow the ring from a vertex until `span` of arc length is covered."""
    size = len(points)
    current = index
    travelled = 0.0
    for _ in range(size - 1):
        following = (current + step) % size
        travelled += distance(points[current], points[following])
        current = following
        if travelled >= span:
            break
    return points[current]


def densify(points: Sequence[AxisPoint], spacing: float) -> tuple[AxisPoint, ...]:
    """Resample a polyline so no segment is longer than `spacing`."""
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
    return tuple(dense)


def evaluate(curve: BezierCurve, parameter: float) -> AxisPoint:
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


def chord_parameters(points: Sequence[AxisPoint]) -> list[float]:
    """Assign each point a 0 to 1 parameter proportional to arc length."""
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
    """Least-squares fit one cubic to points, with both tangents fixed."""
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
    """Worst squared deviation from the curve, and where it happens."""
    maximum = 0.0
    split = len(points) // 2
    for index in range(1, len(points) - 1):
        fitted = evaluate(curve, parameters[index])
        error = (fitted[0] - points[index][0]) ** 2 + (
            fitted[1] - points[index][1]
        ) ** 2
        if error >= maximum:
            maximum = error
            split = index
    return maximum, split


def line_curve(start: AxisPoint, end: AxisPoint) -> BezierCurve:
    """A cubic whose controls sit on the straight line from A to B."""
    return BezierCurve(
        start,
        (start[0] + (end[0] - start[0]) / 3, start[1] + (end[1] - start[1]) / 3),
        (end[0] - (end[0] - start[0]) / 3, end[1] - (end[1] - start[1]) / 3),
        end,
    )
