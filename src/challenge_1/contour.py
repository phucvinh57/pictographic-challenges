from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from skimage.measure import find_contours

from .geometry import cross_product, distance, offset, signed_angle
from .types import AxisPoint, Contour


def _convert_to_grayscale(image: MatLike) -> np.ndarray:
    if image.ndim == 2:
        return image

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    color = image[:, :, :3].astype(np.float64)
    alpha = image[:, :, 3:4].astype(np.float64) / 255.0
    composited = color * alpha + 255.0 * (1.0 - alpha)
    composited = np.rint(composited).astype(np.uint8)

    return cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)

def read_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return _convert_to_grayscale(image)


def extract_contours(image: np.ndarray) -> tuple[Contour, ...]:
    level, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    field = np.pad(
        (level + 0.5) - image.astype(np.float64),
        1,
        constant_values=-1.0,
    )

    contours: list[Contour] = []
    for contour in find_contours(
        field,
        level=0.0,
        fully_connected="high",
        positive_orientation="high",
    ):
        # scikit-image returns (row, column); the SVG pipeline uses (x, y).
        # The 0.5 offset is to center the contour on the pixel grid.
        points = tuple(
            (float(column - 0.5), float(row - 0.5)) for row, column in contour
        )
        if len(points) > 1 and points[0] == points[-1]:
            points = points[:-1]
        if len(points) >= 4:
            contours.append(points)
    return tuple(contours)


def _calc_penalty(point: AxisPoint, line: tuple[AxisPoint, AxisPoint]) -> float:
    """Calculate the distance from a point to a line defined by two points."""
    first, last = line
    a = distance(first, point)
    b = distance(point, last)
    chord = distance(first, last)
    if chord == 0:
        return 0.0
    semiperimeter = (a + b + chord) / 2
    # Using Heron's formula to calculate the area of the triangle formed by the three points
    area_squared = max(
        0.0,
        semiperimeter
        * (semiperimeter - a)
        * (semiperimeter - b)
        * (semiperimeter - chord),
    )
    # S^2 / chord
    return area_squared / chord


def _limit_penalties(points: list[AxisPoint], tolerance: float = 0.25) -> list[int]:
    """
    To reduce a closed contour to a smaller set of important points,
    while preserving points where the contour has significant curvature/change.

    A potrace-like approach, which is similar to the Ramer-Douglas-Peucker algorithm.
    """
    if len(points) < 3:
        return list(range(len(points)))
    close_path = [*points, points[0]]
    result = [0]
    last = 0

    for index in range(1, len(close_path)):
        if index == last + 1:
            continue
        maximum = max(
            _calc_penalty(close_path[middle], (close_path[last], close_path[index]))
            for middle in range(last + 1, index)
        )
        if maximum >= tolerance:
            last = index - 1
            result.append(last)
        if index == len(close_path) - 1:
            result.append(index)
    reduced = result[:-1] if len(result) > 1 else result
    return reduced if len(reduced) >= 3 else list(range(len(points)))


def _remove_collinear(points: list[AxisPoint], indices: list[int]) -> list[int]:
    """If three consecutive points are collinear, remove the middle one."""
    current = indices
    while len(current) > 3:
        reduced = [
            index
            for position, index in enumerate(current)
            if abs(
                cross_product(
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


def _get_break_points(
    polygon: list[AxisPoint],
    span_length_threshold: float = 12.0,
    angle_threshold: float = 30.0,
) -> list[bool]:
    size = len(polygon)
    turns = [
        signed_angle(polygon[index - 1], point, polygon[(index + 1) % size])
        for index, point in enumerate(polygon)
    ]
    lengths = [
        distance(point, polygon[(index + 1) % size])
        for index, point in enumerate(polygon)
    ]

    def gather(index: int, step: int) -> tuple[float, float, float]:
        total = 0.0
        travelled = 0.0
        current = index
        for _ in range(size - 1):
            next = (current + step) % size
            length = lengths[current if step > 0 else next]
            if travelled + length > span_length_threshold:
                break
            travelled += length
            total += abs(turns[next])
            current = next

        edge = lengths[current if step > 0 else (current - 1) % size]
        return total, travelled, min(edge / 2, span_length_threshold / 2)

    breaks = []
    for index in range(size):
        behind, behind_arc, behind_edge = gather(index, -1)
        ahead, ahead_arc, ahead_edge = gather(index, 1)

        total_angle = abs(turns[index]) + behind + ahead
        total_length = behind_arc + ahead_arc + behind_edge + ahead_edge

        breaks.append(
            total_angle >= angle_threshold
            # To detect if bend is sufficiently concentrated
            and total_angle * span_length_threshold >= angle_threshold * total_length
        )
    return breaks


def _is_straight_span(
    points: list[AxisPoint],
    start: int,
    end: int,
    minimum_length: float = 8.0,
    tolerance: float = 1.0,
    radius: float = 100.0,
) -> bool:
    size = len(points)
    first, last = points[start], points[end]
    length = distance(first, last)
    if length < minimum_length:
        return False

    stop = end if end > start else end + size
    bow = max(
        offset(points[index % size], (first, last)) for index in range(start, stop + 1)
    )
    # See https://en.wikipedia.org/wiki/Sagitta_(geometry)
    return bow <= tolerance and length * length >= 8 * bow * radius


def _identify_straight_runs(
    contour: list[AxisPoint],
    indices: list[int],
    dominant_length: float = 64.0,
) -> tuple[list[int], tuple[bool, ...]]:
    polygon = [contour[i] for i in indices]
    breaks = _get_break_points(polygon)
    size = len(indices)

    for i, point in enumerate(polygon):
        next = (i + 1) % size
        if distance(point, polygon[next]) >= dominant_length:
            breaks[i] = breaks[next] = True
    marks = [i for i in range(size) if breaks[i]]

    if not marks:
        return indices, (False,) * size

    def is_straight(start: int, stop: int) -> bool:
        return _is_straight_span(
            contour,
            indices[marks[start]],
            indices[marks[stop % len(marks)]],
        )

    kept = []
    flags = []
    i = 0

    while i < len(marks):
        end = i + 1
        straight = is_straight(i, end)
        if straight:
            while end < len(marks) and is_straight(i, end + 1):
                end += 1
        kept.append(indices[marks[i]])
        flags.append(straight)
        if not straight:
            step = marks[i] + 1
            while step % size != marks[end % len(marks)]:
                kept.append(indices[step % size])
                flags.append(False)
                step += 1
        i = end
    return kept, tuple(flags)


def process_contour(contour: Contour) -> tuple[Contour, tuple[bool, ...]]:
    path = list(contour)
    # A being a closed contour, we need at least 3 points to form a polygon.
    if len(path) < 3:
        return (), ()

    # Indices of points that are important to the shape of the contour, after removing collinear points.
    indices = _remove_collinear(path, _limit_penalties(path))
    if len(indices) < 3:
        return (), ()

    # Straight flags indicate whether the segment between two consecutive points is straight or not.
    # For example, straight_flags[i] is True if the segment between corners[i] and corners[(i + 1) % len(corners)] is straight.
    kept_indices, straight_flags = _identify_straight_runs(path, indices)
    corners = [contour[i] for i in kept_indices]

    return tuple(corners), straight_flags


__all__ = ["extract_contours", "process_contour"]
