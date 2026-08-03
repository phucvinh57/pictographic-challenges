from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import cast

import cv2
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from scipy.spatial import KDTree

PixelPoint = tuple[int, int]

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


@dataclass(frozen=True)
class DiscContact:
    """One boundary patch a maximal-inscribed disc touches."""

    contour: int
    position: int
    point: PixelPoint
    corner: bool


@dataclass(frozen=True)
class AxisDisc:
    """An axis pixel's maximal-inscribed disc and the boundary it touches."""

    pixel: PixelPoint
    radius: float
    contacts: tuple[DiscContact, ...]


@dataclass(frozen=True)
class AxisDiscs:
    discs: dict[PixelPoint, AxisDisc]
    contour_lengths: tuple[int, ...]


@dataclass(frozen=True)
class CornerRun:
    """The stretch of axis a single boundary corner drives."""

    pixels: tuple[PixelPoint, ...]
    contacts: tuple[DiscContact, ...]
    radius: float
    bounded: tuple[bool, bool]


@dataclass(frozen=True)
class TransitionPoint:
    """An axis pixel where the disc's governing boundary element changes."""

    pixel: PixelPoint
    contact: DiscContact


@dataclass(frozen=True)
class _Boundary:
    points: np.ndarray
    labels: np.ndarray
    offsets: np.ndarray
    lengths: tuple[int, ...]
    corners: np.ndarray


def _contour_corners(points: np.ndarray, window: int, min_turn: float) -> np.ndarray:
    """Flag the contour pixels where the boundary turns a corner.

    The turn is measured between the chords reaching ``window`` pixels back and
    ``window`` pixels on, which averages out the staircase of a rasterised edge
    while a real corner still turns sharply over that span.
    """
    if len(points) <= 2 * window:
        return np.zeros(len(points), dtype=bool)

    before = points - np.roll(points, window, axis=0)
    after = np.roll(points, -window, axis=0) - points
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    dot = before[:, 0] * after[:, 0] + before[:, 1] * after[:, 1]
    return np.abs(np.arctan2(cross, dot)) >= min_turn


def _boundary_index(ink: np.ndarray, window: int, min_turn: float) -> _Boundary:
    """Order every boundary pixel along its contour and mark the corners.

    A contact is identified by how far along a contour it sits, so tracking a
    contact from one axis pixel to the next only needs the difference of two
    positions on the same contour.
    """
    contours, _ = cv2.findContours(
        ink.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    points: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    corners: list[np.ndarray] = []
    lengths: list[int] = []
    for index, contour in enumerate(contours):
        contour_points = contour.reshape(-1, 2)
        points.append(contour_points)
        labels.append(np.full(len(contour_points), index, dtype=np.int64))
        corners.append(_contour_corners(contour_points, window, min_turn))
        lengths.append(len(contour_points))
    if not points:
        return _Boundary(
            points=np.empty((0, 2), dtype=np.int64),
            labels=np.empty(0, dtype=np.int64),
            offsets=np.zeros(1, dtype=np.int64),
            lengths=(),
            corners=np.empty(0, dtype=bool),
        )
    return _Boundary(
        points=np.concatenate(points),
        labels=np.concatenate(labels),
        offsets=np.concatenate(([0], np.cumsum(lengths)))[:-1],
        lengths=tuple(lengths),
        corners=np.concatenate(corners),
    )


def _group_contacts(
    pixel: PixelPoint, indices: list[int], boundary: _Boundary, max_gap: int
) -> tuple[DiscContact, ...]:
    """Split the touched boundary pixels into contiguous contact patches."""
    labels = boundary.labels
    ordered = sorted(indices, key=lambda index: (int(labels[index]), index))
    runs: list[list[int]] = []
    for index in ordered:
        if (
            runs
            and labels[runs[-1][-1]] == labels[index]
            and index - runs[-1][-1] <= max_gap
        ):
            runs[-1].append(index)
        else:
            runs.append([index])

    if len(runs) > 1:
        first, last = runs[0], runs[-1]
        contour = int(labels[first[0]])
        if (
            labels[last[0]] == contour
            and boundary.lengths[contour] - last[-1] + first[0] <= max_gap
        ):
            runs[0] = last + first
            runs.pop()

    contacts = []
    for run in runs:
        nearest = min(
            run,
            key=lambda index: hypot(
                float(boundary.points[index, 0]) - pixel[0],
                float(boundary.points[index, 1]) - pixel[1],
            ),
        )
        contour = int(labels[nearest])
        contacts.append(
            DiscContact(
                contour=contour,
                position=nearest - int(boundary.offsets[contour]),
                point=(
                    int(boundary.points[nearest, 0]),
                    int(boundary.points[nearest, 1]),
                ),
                corner=bool(boundary.corners[nearest]),
            )
        )
    return tuple(contacts)


def axis_disc_contacts(
    binary: np.ndarray,
    axis: np.ndarray,
    tolerance: float = 1.0,
    slack: float = 0.05,
    corner_window: int = 5,
    corner_turn: float = 1.0,
) -> AxisDiscs:
    """Find the boundary patches every axis pixel's maximal disc touches.

    The disc of an axis pixel touches the boundary at two patches on a regular
    point, one at an end point and three or more at a junction. A thinned pixel
    sits up to a pixel off the true axis, which spreads the distances to the
    patches it should touch by a share of its radius, so the disc is widened by
    ``tolerance`` pixels and by ``slack`` of the radius before the patches are
    looked up. A patch is flagged as a corner contact when the boundary turns by
    at least ``corner_turn`` radians over ``corner_window`` pixels around it.
    """
    if binary.shape != axis.shape:
        raise ValueError("The binary image and skeleton must have the same shape")

    ink = np.asarray(binary) < 128
    boundary = _boundary_index(ink, corner_window, corner_turn)
    ys, xs = np.nonzero(np.asarray(axis) < 128)
    if not len(boundary.points) or not len(xs):
        return AxisDiscs({}, boundary.lengths)

    tree = KDTree(boundary.points.astype(float))
    pixels = np.column_stack((xs, ys)).astype(float)
    radii = cast(np.ndarray, tree.query(pixels)[0])

    discs = {}
    for pixel, radius, touched in zip(
        pixels, radii, tree.query_ball_point(pixels, radii * (1.0 + slack) + tolerance)
    ):
        point = (int(pixel[0]), int(pixel[1]))
        discs[point] = AxisDisc(
            pixel=point,
            radius=float(radius),
            contacts=_group_contacts(point, touched, boundary, max_gap=2),
        )
    return AxisDiscs(discs, boundary.lengths)


def axis_stroke_width(discs: AxisDiscs) -> float:
    """The ink's own stroke width, twice the median maximal-inscribed radius.

    Every axis point carries the disc that fits the ink around it, so its radius
    is the local half-width. The median ignores the ends, where the disc shrinks
    into the cap, and the junctions, where it swells to span two strokes at once.
    """
    if not discs.discs:
        return 0.0
    return 2.0 * float(
        np.median([disc.radius for disc in discs.discs.values()])
    )


def _corner_contact(disc: AxisDisc) -> DiscContact | None:
    for contact in disc.contacts:
        if contact.corner:
            return contact
    return None


def corner_runs(
    discs: AxisDiscs,
    chains: tuple[tuple[PixelPoint, ...], ...],
    min_run: int = 2,
) -> tuple[CornerRun, ...]:
    """Find the stretches of axis a boundary corner drives.

    An axis point is normally driven by two boundary edges, and its contacts
    slide along them as the axis is walked. A corner of the boundary cannot be
    slid across: the contact stops on it and the disc is driven by that vertex
    for as long as the axis stays inside the corner's fan. ``bounded`` says of
    each end of the run whether it borders a smooth piece of the axis rather
    than the end of the traced line, where an end or junction point sits
    instead. Runs shorter than ``min_run`` pixels are raster jitter.
    """
    runs = []
    for chain in chains:
        walk = [discs.discs[pixel] for pixel in chain if pixel in discs.discs]
        corners = [_corner_contact(disc) for disc in walk]

        start = None
        for index in range(len(corners) + 1):
            inside = index < len(corners) and corners[index] is not None
            if inside and start is None:
                start = index
            elif not inside and start is not None:
                span = walk[start:index]
                if len(span) >= min_run:
                    runs.append(
                        CornerRun(
                            pixels=tuple(disc.pixel for disc in span),
                            contacts=tuple(
                                cast(DiscContact, contact)
                                for contact in corners[start:index]
                            ),
                            radius=max(disc.radius for disc in span),
                            bounded=(start > 0, index < len(walk)),
                        )
                    )
                start = None
    return tuple(runs)


def transition_points(
    discs: AxisDiscs, runs: tuple[CornerRun, ...]
) -> tuple[TransitionPoint, ...]:
    """Take the borders of every corner-driven run, the transition points.

    They are the points where the disc changes which boundary element drives
    it, so they are the borders between the smooth pieces of the axis. The
    taxonomy gives every axis point one class, so a border whose disc already
    touches three boundary patches is a junction point and is left as one.
    """
    return tuple(
        TransitionPoint(run.pixels[index], run.contacts[index])
        for run in runs
        for index, bounded in ((0, run.bounded[0]), (-1, run.bounded[1]))
        if bounded and len(discs.discs[run.pixels[index]].contacts) == 2
    )
