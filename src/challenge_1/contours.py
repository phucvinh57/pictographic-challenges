from collections.abc import Sequence
from math import atan2, degrees, hypot

import numpy as np

from pictographic.curves import (
    AxisPoint,
    BezierCurve,
    corner_flags,
    fit_closed_contour,
)

_SEGMENTS: dict[int, tuple[tuple[str, str], ...]] = {
    1: (("L", "T"),),
    2: (("T", "R"),),
    3: (("L", "R"),),
    4: (("R", "B"),),
    6: (("T", "B"),),
    7: (("L", "B"),),
    8: (("B", "L"),),
    9: (("B", "T"),),
    11: (("B", "R"),),
    12: (("R", "L"),),
    13: (("R", "T"),),
    14: (("T", "L"),),
}
_SADDLES: dict[tuple[int, bool], tuple[tuple[str, str], ...]] = {
    (5, False): (("L", "T"), ("R", "B")),
    (5, True): (("R", "T"), ("L", "B")),
    (10, False): (("T", "R"), ("B", "L")),
    (10, True): (("T", "L"), ("B", "R")),
}


def _edge(side: str, row: int, column: int) -> tuple[str, int, int]:
    return {
        "T": ("h", row, column),
        "B": ("h", row + 1, column),
        "L": ("v", row, column),
        "R": ("v", row, column + 1),
    }[side]


def _crossing(field: np.ndarray, edge: tuple[str, int, int]) -> AxisPoint:
    kind, row, column = edge
    first = float(field[row, column])
    second = float(field[row, column + 1] if kind == "h" else field[row + 1, column])
    fraction = first / (first - second)
    return (
        (column + fraction - 0.5, row - 0.5)
        if kind == "h"
        else (column - 0.5, row + fraction - 0.5)
    )


def _cross(first: AxisPoint, second: AxisPoint, third: AxisPoint) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (third[0] - first[0]) * (
        second[1] - first[1]
    )


def _remove_duplicated_points(points: Sequence[AxisPoint]) -> list[AxisPoint]:
    unique = []
    for point in points:
        value = (float(point[0]), float(point[1]))
        if not unique or value != unique[-1]:
            unique.append(value)
    if len(unique) > 1 and unique[0] == unique[-1]:
        unique.pop()
    return unique


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


def _limit_penalties(points: list[AxisPoint], tolerance: float = 0.25) -> list[int]:
    if len(points) < 3:
        return list(range(len(points)))
    path = [*points, points[0]]
    result = [0]
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
            result.append(last)
        if index == len(path) - 1:
            result.append(index)
    reduced = result[:-1] if len(result) > 1 else result
    return reduced if len(reduced) >= 3 else list(range(len(points)))


def _remove_collinear(points: list[AxisPoint], indices: list[int]) -> list[int]:
    current = indices
    while len(current) > 3:
        reduced = [
            index
            for position, index in enumerate(current)
            if abs(
                _cross(
                    points[current[position - 1]],
                    points[index],
                    points[current[(position + 1) % len(current)]],
                )
            )
            > 1e-9
        ]
        if len(reduced) < 3 or len(reduced) == len(current):
            break
        current = reduced
    return current


def _offset(start: AxisPoint, end: AxisPoint, point: AxisPoint) -> float:
    length = hypot(end[0] - start[0], end[1] - start[1])
    if length == 0:
        return hypot(point[0] - start[0], point[1] - start[1])
    return abs(_cross(start, end, point)) / length


def _bend(first: AxisPoint, point: AxisPoint, last: AxisPoint) -> float:
    incoming = (point[0] - first[0], point[1] - first[1])
    outgoing = (last[0] - point[0], last[1] - point[1])
    return degrees(
        atan2(
            incoming[0] * outgoing[1] - incoming[1] * outgoing[0],
            incoming[0] * outgoing[0] + incoming[1] * outgoing[1],
        )
    )


def _breaks(polygon: list[AxisPoint], span: float, angle: float) -> list[bool]:
    size = len(polygon)
    turns = [
        _bend(polygon[index - 1], point, polygon[(index + 1) % size])
        for index, point in enumerate(polygon)
    ]
    lengths = [
        hypot(
            polygon[(index + 1) % size][0] - point[0],
            polygon[(index + 1) % size][1] - point[1],
        )
        for index, point in enumerate(polygon)
    ]

    def gather(index: int, step: int) -> tuple[float, float, float]:
        total = 0.0
        travelled = 0.0
        current = index
        for _ in range(size - 1):
            following = (current + step) % size
            length = lengths[current if step > 0 else following]
            if travelled + length > span:
                break
            travelled += length
            total += abs(turns[following])
            current = following
        edge = lengths[current if step > 0 else (current - 1) % size]
        return total, travelled, min(edge / 2, span / 2)

    breaks = []
    for index in range(size):
        behind, behind_arc, behind_edge = gather(index, -1)
        ahead, ahead_arc, ahead_edge = gather(index, 1)
        total = abs(turns[index]) + behind + ahead
        arc = behind_arc + ahead_arc + behind_edge + ahead_edge
        breaks.append(total >= angle and total * span >= angle * arc)
    return breaks


def _runs_straight(
    points: list[AxisPoint],
    start: int,
    end: int,
    minimum_length: float,
    tolerance: float,
    radius: float,
) -> bool:
    size = len(points)
    first, last = points[start], points[end]
    length = hypot(last[0] - first[0], last[1] - first[1])
    if length < minimum_length:
        return False
    stop = end if end > start else end + size
    bow = max(
        _offset(first, last, points[index % size]) for index in range(start, stop + 1)
    )
    return bow <= tolerance and length * length >= 8 * bow * radius


def _straight_runs(
    points: list[AxisPoint],
    indices: list[int],
    minimum_length: float,
    tolerance: float,
    radius: float,
    span: float,
    angle: float,
    dominant_length: float,
) -> tuple[list[int], tuple[bool, ...]]:
    polygon = [points[index] for index in indices]
    breaks = _breaks(polygon, span, angle)
    size = len(indices)
    for position, point in enumerate(polygon):
        following = (position + 1) % size
        if (
            hypot(polygon[following][0] - point[0], polygon[following][1] - point[1])
            >= dominant_length
        ):
            breaks[position] = breaks[following] = True
    marks = [position for position in range(size) if breaks[position]]
    if not marks:
        return indices, (False,) * size

    def straight_to(start: int, stop: int) -> bool:
        return _runs_straight(
            points,
            indices[marks[start]],
            indices[marks[stop % len(marks)]],
            minimum_length,
            tolerance,
            radius,
        )

    kept = []
    flags = []
    position = 0
    while position < len(marks):
        end = position + 1
        straight = straight_to(position, end)
        if straight:
            while end < len(marks) and straight_to(position, end + 1):
                end += 1
        kept.append(indices[marks[position]])
        flags.append(straight)
        if not straight:
            step = marks[position] + 1
            while step % size != marks[end % len(marks)]:
                kept.append(indices[step % size])
                flags.append(False)
                step += 1
        position = end
    return kept, tuple(flags)


def extract_contours(
    gray_img: np.ndarray, level: float
) -> tuple[tuple[AxisPoint, ...], ...]:
    field = np.pad((level + 0.5) - gray_img.astype(np.float64), 1, constant_values=-1.0)
    inside = field > 0
    cases = (
        inside[:-1, :-1].astype(np.uint8)
        | inside[:-1, 1:].astype(np.uint8) << 1
        | inside[1:, 1:].astype(np.uint8) << 2
        | inside[1:, :-1].astype(np.uint8) << 3
    )
    following: dict[tuple[str, int, int], tuple[str, int, int]] = {}
    for row, column in zip(*np.nonzero((cases > 0) & (cases < 15))):
        case = int(cases[row, column])
        if case in (5, 10):
            middle = (
                field[row, column]
                + field[row, column + 1]
                + field[row + 1, column]
                + field[row + 1, column + 1]
            )
            pairs = _SADDLES[case, bool(middle > 0)]
        else:
            pairs = _SEGMENTS[case]
        for start, end in pairs:
            following[_edge(start, int(row), int(column))] = _edge(
                end, int(row), int(column)
            )

    contours = []
    walked = set()
    for start in following:
        if start in walked:
            continue
        loop = []
        edge = start
        while edge not in walked and edge in following:
            walked.add(edge)
            loop.append(_crossing(field, edge))
            edge = following[edge]
        if len(loop) >= 4:
            contours.append(tuple(loop))
    return tuple(contours)


def preprocess_contours(
    contours: tuple[tuple[AxisPoint, ...], ...],
    simplify_tolerance: float = 0.25,
    minimum_straight: float = 8.0,
    straight_tolerance: float = 1.0,
    straight_radius: float = 100.0,
    break_span: float = 12.0,
    break_angle: float = 30.0,
    dominant_straight: float = 64.0,
) -> tuple[tuple[tuple[AxisPoint, ...], ...], tuple[tuple[bool, ...], ...]]:
    result = []
    straights = []
    for contour in contours:
        path = _remove_duplicated_points(contour)
        if len(path) < 3:
            continue
        indices = _remove_collinear(path, _limit_penalties(path, simplify_tolerance))

        if len(indices) >= 3:
            indices, straight = _straight_runs(
                path,
                indices,
                minimum_straight,
                straight_tolerance,
                straight_radius,
                break_span,
                break_angle,
                dominant_straight,
            )
            corners = [path[index] for index in indices]
            straights.append(straight)
            result.append((*corners, corners[0]))
    return tuple(result), tuple(straights)


def smooth_contours(
    contours: tuple[tuple[AxisPoint, ...], ...],
    straights: tuple[tuple[bool, ...], ...],
    angle_threshold: float,
    smooth_tolerance: float,
) -> tuple[tuple[BezierCurve, ...], ...]:
    result = []
    for contour, straight in zip(contours, straights, strict=True):
        corners = corner_flags(contour, angle_threshold)
        curves = fit_closed_contour(contour, corners, straight, smooth_tolerance)
        if curves:
            result.append(curves)
    return tuple(result)


def curve_anchors(
    contours: tuple[tuple[BezierCurve, ...], ...],
) -> tuple[tuple[AxisPoint, ...], ...]:
    return tuple(
        (curves[0].start, *(curve.end for curve in curves)) for curves in contours
    )
