from itertools import pairwise
from math import hypot

from common import debug

from .geometry import distance, offset, path_length, scaled, signed_angle
from .settings import (
    BREAK_SPAN_FLOOR,
    DEFAULT_VECTORIZATION_SETTINGS,
    DOMINANT_FLOOR,
    SIMPLIFY_FLOOR,
    STRAIGHT_MIN_FLOOR,
    STRAIGHT_TOLERANCE_FLOOR,
    VectorizationSettings,
)
from .types import AxisPoint, Contour, ProcessedContour


def _limit_penalties(
    points: list[AxisPoint], closed: bool, settings: VectorizationSettings
) -> list[int]:
    minimum = 3 if closed else 2
    if len(points) < minimum:
        return list(range(len(points)))
    tolerance = scaled(
        settings.simplify_ratio,
        SIMPLIFY_FLOOR,
        path_length(points, closed),
    )
    path = [*points, points[0]] if closed else points
    result = [0]
    last = 0

    for index in range(1, len(path)):
        if index == last + 1:
            continue
        anchor_x, anchor_y = path[last]
        head_x, head_y = path[index]
        span_x, span_y = head_x - anchor_x, head_y - anchor_y
        chord = hypot(span_x, span_y)
        limit = tolerance * chord
        if any(
            abs(span_x * (y - anchor_y) - (x - anchor_x) * span_y) >= limit
            if chord
            else hypot(x - anchor_x, y - anchor_y) >= tolerance
            for x, y in path[last + 1 : index]
        ):
            last = index - 1
            result.append(last)
        if index == len(path) - 1:
            result.append(index)
    reduced = result[:-1] if closed and len(result) > 1 else result
    return reduced if len(reduced) >= minimum else list(range(len(points)))


def _closed_break_points(
    polygon: list[AxisPoint], scale: float, settings: VectorizationSettings
) -> list[bool]:
    span_length_threshold = scaled(
        settings.break_span_ratio, BREAK_SPAN_FLOOR, scale
    )
    angle_threshold = settings.break_angle_threshold
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
        total_angle = abs(turns[index]) + behind + ahead
        total_length = behind_arc + ahead_arc + behind_edge + ahead_edge
        breaks.append(
            total_angle >= angle_threshold
            and total_angle * span_length_threshold
            >= angle_threshold * total_length
        )
    return breaks


def _open_break_points(
    polygon: list[AxisPoint], scale: float, settings: VectorizationSettings
) -> list[bool]:
    size = len(polygon)
    if size < 2:
        return [True] * size
    span = scaled(settings.break_span_ratio, BREAK_SPAN_FLOOR, scale)
    angle_threshold = settings.break_angle_threshold
    turns = [0.0] + [
        signed_angle(polygon[index - 1], polygon[index], polygon[index + 1])
        for index in range(1, size - 1)
    ] + [0.0]
    lengths = [distance(first, second) for first, second in pairwise(polygon)]

    def gather(index: int, step: int) -> tuple[float, float, float]:
        total = 0.0
        travelled = 0.0
        current = index
        while 0 <= current + step < size:
            following = current + step
            length = lengths[current if step > 0 else following]
            if travelled + length > span:
                break
            travelled += length
            total += abs(turns[following])
            current = following
        edge_index = current if step > 0 else current - 1
        edge = lengths[edge_index] if 0 <= edge_index < len(lengths) else 0.0
        return total, travelled, min(edge / 2, span / 2)

    breaks = [False] * size
    breaks[0] = breaks[-1] = True
    for index in range(1, size - 1):
        behind, behind_arc, behind_edge = gather(index, -1)
        ahead, ahead_arc, ahead_edge = gather(index, 1)
        total_angle = abs(turns[index]) + behind + ahead
        total_length = behind_arc + ahead_arc + behind_edge + ahead_edge
        breaks[index] = (
            total_angle >= angle_threshold
            and total_angle * span >= angle_threshold * total_length
        )
    return breaks


def _is_straight_span(
    points: list[AxisPoint],
    start: int,
    end: int,
    scale: float,
    closed: bool,
    settings: VectorizationSettings,
) -> bool:
    size = len(points)
    first, last = points[start], points[end]
    length = distance(first, last)
    if length < scaled(settings.straight_min_ratio, STRAIGHT_MIN_FLOOR, scale):
        return False

    stop = end if end > start else end + size
    bow = max(
        offset(points[index % size], (first, last))
        for index in range(start, stop + 1)
    )
    ceiling = scaled(
        settings.straight_tolerance_ratio,
        STRAIGHT_TOLERANCE_FLOOR,
        scale,
    )
    return bow <= ceiling and bow <= settings.straight_bow_ratio * length


def _closed_straight_runs(
    points: list[AxisPoint],
    indices: list[int],
    scale: float,
    settings: VectorizationSettings,
) -> tuple[list[int], list[bool]]:
    dominant_length = scaled(settings.dominant_ratio, DOMINANT_FLOOR, scale)
    polygon = [points[index] for index in indices]
    breaks = _closed_break_points(polygon, scale, settings)
    size = len(indices)

    for index, point in enumerate(polygon):
        following = (index + 1) % size
        if distance(point, polygon[following]) >= dominant_length:
            breaks[index] = breaks[following] = True
    marks = [index for index in range(size) if breaks[index]]

    if not marks:
        return indices, [False] * size

    def is_straight(start: int, stop: int) -> bool:
        return _is_straight_span(
            points,
            indices[marks[start]],
            indices[marks[stop % len(marks)]],
            scale,
            True,
            settings,
        )

    kept = []
    flags = []
    position = 0
    while position < len(marks):
        end = position + 1
        straight = is_straight(position, end)
        if straight:
            while end < len(marks) and is_straight(position, end + 1):
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
    return kept, flags


def _open_straight_runs(
    points: list[AxisPoint],
    indices: list[int],
    scale: float,
    settings: VectorizationSettings,
) -> tuple[list[int], list[bool]]:
    dominant_length = scaled(settings.dominant_ratio, DOMINANT_FLOOR, scale)
    polygon = [points[index] for index in indices]
    breaks = _open_break_points(polygon, scale, settings)
    for index, (first, second) in enumerate(pairwise(polygon)):
        if distance(first, second) >= dominant_length:
            breaks[index] = breaks[index + 1] = True
    marks = [index for index, marked in enumerate(breaks) if marked]

    def is_straight(start: int, stop: int) -> bool:
        return _is_straight_span(
            points,
            indices[marks[start]],
            indices[marks[stop]],
            scale,
            False,
            settings,
        )

    kept = [indices[marks[0]]]
    flags: list[bool] = []
    position = 0
    while position < len(marks) - 1:
        end = position + 1
        straight = is_straight(position, end)
        if straight:
            while end < len(marks) - 1 and is_straight(position, end + 1):
                end += 1
            kept.append(indices[marks[end]])
            flags.append(True)
        else:
            for step in range(marks[position] + 1, marks[end] + 1):
                kept.append(indices[step])
                flags.append(False)
        position = end
    return kept, flags


def process_contour(
    contour: Contour,
    settings: VectorizationSettings = DEFAULT_VECTORIZATION_SETTINGS,
) -> ProcessedContour:
    minimum = 3 if contour.closed else 2
    if len(contour.points) < minimum:
        return ProcessedContour((), (), contour.closed)

    points = list(contour.points)
    debug.count("contour points", len(points))
    indices = _limit_penalties(points, contour.closed, settings)
    debug.count("after simplify", len(indices))
    if len(indices) < minimum:
        return ProcessedContour((), (), contour.closed)

    scale = path_length(points, contour.closed)
    if contour.closed:
        kept_indices, straight_flags = _closed_straight_runs(
            points, indices, scale, settings
        )
    else:
        kept_indices, straight_flags = _open_straight_runs(
            points, indices, scale, settings
        )
    corners = tuple(points[index] for index in kept_indices)
    debug.count("after straight runs", len(corners))
    debug.count("straight segments", sum(straight_flags))
    return ProcessedContour(
        corners,
        tuple(straight_flags),
        contour.closed,
    )
