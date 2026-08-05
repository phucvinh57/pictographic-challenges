from collections.abc import Sequence
from itertools import pairwise
from math import hypot

import cv2
import numpy as np

from pictographic.curves import (
    AxisPoint,
    BezierCurve,
    fit_closed_contour,
    subdivide_closed_path,
)


def _inside(mask: np.ndarray, point: tuple[int, int]) -> bool:
    x, y = point
    return bool(
        0 <= y < mask.shape[0]
        and 0 <= x < mask.shape[1]
        and mask[y, x]
    )


def _ahead(point: tuple[int, int], direction: int) -> tuple[int, int]:
    x, y = point
    dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[direction]
    return x + dx, y + dy


def _side_pixels(
    point: tuple[int, int], direction: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    x, y = point
    first, second = (
        ((-1, -1), (0, -1)),
        ((0, 0), (0, -1)),
        ((-1, 0), (0, 0)),
        ((-1, 0), (-1, -1)),
    )[direction]
    return (x + first[0], y + first[1]), (x + second[0], y + second[1])


def _boundary_start(mask: np.ndarray) -> tuple[int, int] | None:
    padded = np.pad(mask, 1, mode="constant")
    interior = (
        mask
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    rows, columns = np.nonzero(mask & ~interior)
    return (int(columns[0]), int(rows[0])) if len(columns) else None


def _trace_boundary(mask: np.ndarray, clockwise: bool) -> tuple[AxisPoint, ...]:
    start = _boundary_start(mask)
    if start is None:
        return ()
    current = previous = before_previous = start
    length = 0
    path = [start]
    directions = (0, 1, 2, 3) if clockwise else (3, 2, 1, 0)
    while current != start or length == 0:
        direction: int | None = None
        while True:
            following = None
            for candidate in directions:
                target = _ahead(current, candidate)
                if target == previous or target == before_previous:
                    continue
                first, second = _side_pixels(current, candidate)
                if _inside(mask, first) != _inside(mask, second):
                    following = candidate
                    break
            if following is None:
                raise ValueError("Could not trace a closed foreground boundary")
            if direction is not None and direction != following:
                break
            direction = following
            before_previous, previous = previous, current
            current = _ahead(current, following)
            length += 1
        path.append(current)
    return tuple((float(x), float(y)) for x, y in path)


def extract_contours(binary: np.ndarray) -> tuple[tuple[AxisPoint, ...], ...]:
    contours = []
    fields = ((binary < 128, True, False), (binary >= 128, False, True))
    for field, clockwise, exclude_border in fields:
        count, labels = cv2.connectedComponents(
            field.astype(np.uint8), connectivity=4
        )
        for label in range(1, count):
            mask = labels == label
            if exclude_border and (
                mask[0].any()
                or mask[-1].any()
                or mask[:, 0].any()
                or mask[:, -1].any()
            ):
                continue
            contour = _trace_boundary(mask, clockwise)
            if len(contour) >= 4:
                contours.append(contour)
    return tuple(contours)


def _cross(first: AxisPoint, second: AxisPoint, third: AxisPoint) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        third[0] - first[0]
    ) * (second[1] - first[1])


def _remove_duplicated_points(points: Sequence[AxisPoint]) -> list[AxisPoint]:
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


def _limit_penalties(
    points: list[AxisPoint], tolerance: float = 1.0
) -> list[AxisPoint]:
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

def process_staircases(
    contours: tuple[tuple[AxisPoint, ...], ...],
) -> tuple[tuple[AxisPoint, ...], ...]:
    result = []
    for contour in contours:
        path = _remove_duplicated_points(contour)
        if len(path) < 3:
            return ()
        path = _remove_staircase(path)
        path = _limit_penalties(path)
        path = _remove_collinear(path)

        if len(path) >= 3:
            result.append((*path, path[0]))
    return tuple(result)


def smooth_contours(
    contours: tuple[tuple[AxisPoint, ...], ...],
    angle_threshold: float,
    smooth_tolerance: float,
) -> tuple[tuple[BezierCurve, ...], ...]:
    result = []
    for contour in contours:
        smoothed, corners = subdivide_closed_path(contour, angle_threshold)
        curves = fit_closed_contour(smoothed, corners, smooth_tolerance)
        if curves:
            result.append(curves)
    return tuple(result)


def curve_anchors(
    contours: tuple[tuple[BezierCurve, ...], ...],
) -> tuple[tuple[AxisPoint, ...], ...]:
    return tuple(
        (curves[0].start, *(curve.end for curve in curves))
        for curves in contours
    )
