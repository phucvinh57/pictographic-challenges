from math import hypot, sqrt
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import find_contours

from . import debug
from .args import get_args
from .geometry import Contour, cross_product, distance, offset, signed_angle


def read_image_in_gray_scale(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if image.ndim == 2:
        return image

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Composite over white in place: multiplying the uint8 colors by the float
    # alpha promotes straight to float64, so the colors never need their own copy.
    alpha = image[:, :, 3:4].astype(np.float64) / 255.0
    composited = image[:, :, :3] * alpha
    composited += 255.0 * (1.0 - alpha)
    np.rint(composited, out=composited)

    return cv2.cvtColor(composited.astype(np.uint8), cv2.COLOR_BGR2GRAY)


def extract_contours(image: np.ndarray) -> list[Contour]:
    level, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Fill the -1.0 background border first, then write the signed field
    field = np.full((image.shape[0] + 2, image.shape[1] + 2), -1.0)
    np.subtract(level + 0.5, image, out=field[1:-1, 1:-1], dtype=np.float64)

    contours: list[Contour] = []
    for contour in find_contours(
        field,
        level=0.0,
        fully_connected="high",
        positive_orientation="high",
    ):
        # scikit-image returns (row, column); the SVG pipeline uses (x, y).
        # The 0.5 offset is to center the contour on the pixel grid.
        points: Contour = [(x, y) for x, y in (contour[:, ::-1] - 0.5).tolist()]
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        if len(points) >= 4:
            contours.append(points)
    return contours


def _limit_penalties(contour: Contour) -> list[int]:
    """
    To reduce a closed contour to a smaller set of important points,
    while preserving points where the contour has significant curvature/change.

    A potrace-like approach, which is similar to the Ramer-Douglas-Peucker algorithm.
    """
    tolerance = get_args().simplify_tolerance
    if len(contour) < 3:
        return list(range(len(contour)))
    close_path = [*contour, contour[0]]
    result = [0]
    last = 0

    for index in range(1, len(close_path)):
        if index == last + 1:
            continue
        (anchor_x, anchor_y), (head_x, head_y) = close_path[last], close_path[index]
        span_x, span_y = head_x - anchor_x, head_y - anchor_y
        # A point's penalty is its triangle area squared over the chord, and the
        # cross product is twice that area, so `penalty >= tolerance` is just
        # `|cross| >= limit`: no square root per point, and no reason to look past
        # the first point that breaks it. A zero chord zeroes every penalty, and
        # the falsy limit short-circuits that case away.
        limit = sqrt(4 * tolerance * hypot(span_x, span_y))
        if limit and any(
            abs(span_x * (y - anchor_y) - (x - anchor_x) * span_y) >= limit
            for x, y in close_path[last + 1 : index]
        ):
            last = index - 1
            result.append(last)
        if index == len(close_path) - 1:
            result.append(index)
    reduced = result[:-1] if len(result) > 1 else result
    return reduced if len(reduced) >= 3 else list(range(len(contour)))


def _remove_collinear(points: Contour, indices: list[int]) -> list[int]:
    """If three consecutive points are collinear, remove the middle one."""
    epsilon = get_args().collinear_epsilon
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
            > epsilon
        ]
        if len(reduced) < 3 or len(reduced) == len(current):
            break
        current = reduced
    return current


def _get_break_points(polygon: Contour) -> list[bool]:
    args = get_args()
    span_length_threshold = args.break_span_length
    angle_threshold = args.break_angle_threshold
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


def _is_straight_span(points: Contour, start: int, end: int) -> bool:
    args = get_args()
    size = len(points)
    first, last = points[start], points[end]
    length = distance(first, last)
    if length < args.straight_min_length:
        return False

    stop = end if end > start else end + size
    bow = max(
        offset(points[index % size], (first, last)) for index in range(start, stop + 1)
    )
    # See https://en.wikipedia.org/wiki/Sagitta_(geometry)
    return (
        bow <= args.straight_tolerance
        and length * length >= 8 * bow * args.straight_radius
    )


def _identify_straight_runs(
    contour: Contour,
    indices: list[int],
) -> tuple[list[int], list[bool]]:
    dominant_length = get_args().dominant_length
    polygon = [contour[i] for i in indices]
    breaks = _get_break_points(polygon)
    size = len(indices)

    for i, point in enumerate(polygon):
        next = (i + 1) % size
        if distance(point, polygon[next]) >= dominant_length:
            breaks[i] = breaks[next] = True
    marks = [i for i in range(size) if breaks[i]]

    if not marks:
        return indices, [False] * size

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
    return kept, flags


def process_contour(contour: Contour) -> tuple[Contour, list[bool]]:
    # A being a closed contour, we need at least 3 points to form a polygon.
    if len(contour) < 3:
        return [], []

    debug.count("contour points", len(contour))

    # Indices of points that are important to the shape of the contour, after removing collinear points.
    penalised = _limit_penalties(contour)
    debug.count("after simplify", len(penalised))
    indices = _remove_collinear(contour, penalised)
    debug.count("after collinear removal", len(indices))
    if len(indices) < 3:
        return [], []

    # Straight flags indicate whether the segment between two consecutive points is straight or not.
    # For example, straight_flags[i] is True if the segment between corners[i] and corners[(i + 1) % len(corners)] is straight.
    kept_indices, straight_flags = _identify_straight_runs(contour, indices)
    corners = [contour[i] for i in kept_indices]
    debug.count("after straight runs", len(corners))
    debug.count("straight segments", sum(straight_flags))

    return corners, straight_flags


__all__ = ["extract_contours", "process_contour"]
