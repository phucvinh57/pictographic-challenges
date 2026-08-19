from dataclasses import dataclass
from itertools import pairwise
from math import ceil

import numpy as np


@dataclass(frozen=True)
class BezierCurve:
    start: np.ndarray
    first_control: np.ndarray
    second_control: np.ndarray
    end: np.ndarray


@dataclass(frozen=True)
class SmoothedEdge:
    samples: np.ndarray
    curves: tuple[BezierCurve, ...]
    polyline: np.ndarray


def _without_repeated_points(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    keep = np.concatenate(
        (np.ones(1, dtype=bool), np.any(np.diff(points, axis=0) != 0, axis=1))
    )
    return points[keep]


def sample_polyline(points: np.ndarray, spacing: float) -> tuple[np.ndarray, bool]:
    points = _without_repeated_points(np.asarray(points, dtype=np.float64))
    if len(points) < 2:
        return points, False

    closed = len(points) > 2 and np.array_equal(points[0], points[-1])
    path = points if not closed else points[:-1]
    if closed:
        path = np.vstack((path, path[0]))

    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate((np.zeros(1), np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total == 0:
        return path[:1], closed

    segment_count = max(3 if closed else 1, ceil(total / spacing))
    distances = np.linspace(0.0, total, segment_count, endpoint=False)
    if not closed:
        distances = np.append(distances, total)

    segment_indices = np.searchsorted(cumulative, distances, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, len(lengths) - 1)
    local = (distances - cumulative[segment_indices]) / lengths[segment_indices]
    samples = (
        path[segment_indices]
        + (path[segment_indices + 1] - path[segment_indices]) * local[:, None]
    )
    if not closed:
        samples[0] = path[0]
        samples[-1] = path[-1]
    return samples, closed


def _fit_beziers(samples: np.ndarray, closed: bool) -> tuple[BezierCurve, ...]:
    if len(samples) < 2:
        return ()

    if closed:
        tangents = (np.roll(samples, -1, axis=0) - np.roll(samples, 1, axis=0)) / 2
        sections = tuple(
            (index, (index + 1) % len(samples)) for index in range(len(samples))
        )
    else:
        tangents = np.empty_like(samples)
        tangents[0] = samples[1] - samples[0]
        tangents[-1] = samples[-1] - samples[-2]
        if len(samples) > 2:
            tangents[1:-1] = (samples[2:] - samples[:-2]) / 2
        sections = tuple(pairwise(range(len(samples))))

    return tuple(
        BezierCurve(
            start=samples[start],
            first_control=samples[start] + tangents[start] / 3,
            second_control=samples[end] - tangents[end] / 3,
            end=samples[end],
        )
        for start, end in sections
    )


def _flatten_beziers(curves: tuple[BezierCurve, ...], spacing: float) -> np.ndarray:
    flattened: list[np.ndarray] = []
    for index, curve in enumerate(curves):
        control_length = sum(
            np.linalg.norm(second - first)
            for first, second in pairwise(
                (
                    curve.start,
                    curve.first_control,
                    curve.second_control,
                    curve.end,
                )
            )
        )
        sample_count = max(2, ceil(float(control_length) / spacing) + 1)
        parameters = np.linspace(0.0, 1.0, sample_count)
        remaining = 1.0 - parameters
        points = (
            remaining[:, None] ** 3 * curve.start
            + 3 * remaining[:, None] ** 2 * parameters[:, None] * curve.first_control
            + 3 * remaining[:, None] * parameters[:, None] ** 2 * curve.second_control
            + parameters[:, None] ** 3 * curve.end
        )
        flattened.extend(points[index > 0 :])
    return np.asarray(flattened, dtype=np.float64)


def smooth_edge(
    points: np.ndarray, sample_spacing: float, raster_spacing: float = 0.5
) -> SmoothedEdge:
    samples, closed = sample_polyline(points, sample_spacing)
    curves = _fit_beziers(samples, closed)
    if not curves:
        return SmoothedEdge(samples, curves, samples.copy())
    return SmoothedEdge(samples, curves, _flatten_beziers(curves, raster_spacing))


__all__ = ["BezierCurve", "SmoothedEdge", "sample_polyline", "smooth_edge"]
