from pathlib import Path

import cv2
from cv2.typing import MatLike
from skimage.measure import find_contours

from .types import AxisPoint, Contour
import numpy as np
from math import hypot, atan2, degrees


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


def extract_contours(image_path: Path) -> tuple[Contour, ...]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    gray_image = _convert_to_grayscale(image)
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    field = np.pad(
        (thresh + 0.5) - gray_image.astype(np.float64),
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


def _cross_product(a: AxisPoint, b: AxisPoint, c: AxisPoint) -> float:
    """AB x AC"""
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


def _calc_penalty(point: AxisPoint, line: tuple[AxisPoint, AxisPoint]) -> float:
    """Calculate the distance from a point to a line defined by two points."""
    first, last = line
    a = hypot(first[0] - point[0], first[1] - point[1])
    b = hypot(point[0] - last[0], point[1] - last[1])
    chord = hypot(last[0] - first[0], last[1] - first[1])
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
            _calc_penalty(close_path[last], (close_path[middle], close_path[index]))
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
                _cross_product(
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


def _bend(ab: AxisPoint, b: AxisPoint, c: AxisPoint) -> float:
    """
    Returns the signed angle in degrees between the vectors AB and BC.
    """
    ab = (b[0] - ab[0], b[1] - ab[1])
    bc = (c[0] - b[0], c[1] - b[1])
    return degrees(
        atan2(
            ab[0] * bc[1] - ab[1] * bc[0],
            ab[0] * bc[0] + ab[1] * bc[1],
        )
    )


def _distance(a: AxisPoint, b: AxisPoint) -> float:
    """|AB|"""
    return hypot(b[0] - a[0], b[1] - a[1])


def _breaks(
    polygon: list[AxisPoint], span_length_threshold: float, angle_threshold: float
) -> list[bool]:
    size = len(polygon)
    turns = [
        _bend(polygon[index - 1], point, polygon[(index + 1) % size])
        for index, point in enumerate(polygon)
    ]
    lengths = [
        _distance(point, polygon[(index + 1) % size])
        for index, point in enumerate(polygon)
    ]

    def gather(index: int, step: int) -> tuple[float, float, float]:
        total = 0.0
        travelled = 0.0
        current = index
        for _ in range(size - 1):
            following = (current + step) % size
            length = lengths[current if step > 0 else following]
            if travelled + length > span_length_threshold:
                break
            travelled += length
            total += abs(turns[following])
            current = following
        edge = lengths[current if step > 0 else (current - 1) % size]
        return total, travelled, min(edge / 2, span_length_threshold / 2)

    breaks = []
    for index in range(size):
        behind, behind_arc, behind_edge = gather(index, -1)
        ahead, ahead_arc, ahead_edge = gather(index, 1)

        total_bend = abs(turns[index]) + behind + ahead
        total_length = behind_arc + ahead_arc + behind_edge + ahead_edge

        breaks.append(
            total_bend >= angle_threshold
            # total_bend / total_length >= angle_threshold / span_length_threshold
            # To detect if bend is sufficiently concentrated
            and total_bend / total_length >= angle_threshold / span_length_threshold
        )
    return breaks


def process_contour(contours: tuple[Contour, ...]) -> None:
    result = []
    straights = []
    for contour in contours:
        path = list(contour)
        if len(path) < 3:
            continue
        indices = _remove_collinear(path, _limit_penalties(path))
        if len(indices) < 3:
            continue


__all__ = ["extract_contours", "process_contour"]
