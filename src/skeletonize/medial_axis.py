from __future__ import annotations

from typing import cast

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt

_NEIGHBORHOOD_KERNEL = np.ones((3, 3), dtype=np.uint8)


def _connected_components_8(pattern: np.ndarray) -> int:
    """Count 8-connected foreground components in a 3x3 pattern."""
    remaining = {
        (row, column)
        for row in range(3)
        for column in range(3)
        if pattern[row, column]
    }
    components = 0
    while remaining:
        components += 1
        pending = [remaining.pop()]
        while pending:
            row, column = pending.pop()
            connected = {
                point
                for point in remaining
                if abs(point[0] - row) <= 1 and abs(point[1] - column) <= 1
            }
            remaining.difference_update(connected)
            pending.extend(connected)
    return components


def _make_keep_table() -> np.ndarray:
    """Build the topology lookup table for every binary 3x3 pattern."""
    table = np.zeros(512, dtype=bool)
    for index in range(512):
        if not index & (1 << 4):
            continue

        pattern = np.array(
            [(index >> bit) & 1 for bit in range(9)], dtype=bool
        ).reshape(3, 3)
        without_center = pattern.copy()
        without_center[1, 1] = False

        # Endpoints and isolated pixels must survive. Other pixels survive
        # only when deleting the center would split or remove a component.
        table[index] = (
            pattern.sum() < 3
            or _connected_components_8(pattern)
            != _connected_components_8(without_center)
        )
    return table


_KEEP_TABLE = _make_keep_table()


def _pattern_index(image: np.ndarray, row: int, column: int) -> int:
    """Encode the 3x3 neighborhood at an unpadded image coordinate."""
    return (
        int(image[row, column])
        | (int(image[row, column + 1]) << 1)
        | (int(image[row, column + 2]) << 2)
        | (int(image[row + 1, column]) << 3)
        | (int(image[row + 1, column + 1]) << 4)
        | (int(image[row + 1, column + 2]) << 5)
        | (int(image[row + 2, column]) << 6)
        | (int(image[row + 2, column + 1]) << 7)
        | (int(image[row + 2, column + 2]) << 8)
    )


def _medial_axis_mask(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute a one-pixel, topology-preserving medial axis and its EDT."""
    distance = cast(np.ndarray, distance_transform_edt(ink))
    if not ink.any():
        return ink.copy(), distance

    # Pixels near a boundary are removed before pixels near a distance ridge.
    # On equal-distance plateaus, non-corner pixels go first. A fixed random
    # permutation avoids directional bias while keeping results reproducible.
    foreground_neighbors = convolve(
        ink.astype(np.uint8),
        _NEIGHBORHOOD_KERNEL,
        mode="constant",
        cval=0,
    )
    corner_score = 9 - foreground_neighbors
    rows, columns = np.nonzero(ink)
    tie_breaker = np.random.default_rng(0).permutation(rows.size)
    order = np.lexsort(
        (tie_breaker, corner_score[rows, columns], distance[rows, columns])
    )

    result = np.pad(ink.astype(np.uint8), 1, mode="constant")
    for position in order:
        row = int(rows[position])
        column = int(columns[position])
        if not _KEEP_TABLE[_pattern_index(result, row, column)]:
            result[row + 1, column + 1] = 0

    return result[1:-1, 1:-1].astype(bool), distance


def extract_medial_axis(binary: np.ndarray) -> np.ndarray:
    """Return the original ink's one-pixel medial axis in project polarity."""
    image = np.asarray(binary)
    if image.ndim != 2:
        raise ValueError("The binary image must be two-dimensional")

    axis, _ = _medial_axis_mask(image < 128)
    return np.where(axis, 0, 255).astype(np.uint8)


def estimate_stroke_width(binary: np.ndarray, axis: np.ndarray) -> float:
    """Estimate the original ink width from radii sampled along its medial axis."""
    image = np.asarray(binary)
    skeleton = np.asarray(axis)
    if image.ndim != 2 or skeleton.ndim != 2:
        raise ValueError("The binary image and medial axis must be two-dimensional")
    if image.shape != skeleton.shape:
        raise ValueError("The binary image and medial axis must have the same shape")

    distance = cast(np.ndarray, distance_transform_edt(image < 128))
    radii = distance[skeleton < 128]
    return float(2.0 * np.median(radii)) if radii.size else 0.0
