from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, pairwise
from math import hypot
from typing import TypedDict, cast

import numpy as np
from scipy.ndimage import distance_transform_edt

PixelPoint = tuple[int, int]
AxisPoint = tuple[float, float]


class SkeletonNodeData(TypedDict):
    id: int
    x: int
    y: int
    adjacent: list[int]


class SkeletonEdgeData(TypedDict):
    id: int
    nodes: list[int]


class SkeletonIntersectionData(TypedDict):
    center: int
    nodes: list[int]
    radius: float | None


class SkeletonGraphData(TypedDict):
    width: int
    height: int
    nodes: list[SkeletonNodeData]
    edges: list[SkeletonEdgeData]
    endpoints: list[int]
    intersections: list[SkeletonIntersectionData]


@dataclass(frozen=True)
class SkeletonEdge:
    id: int
    pixels: tuple[PixelPoint, ...]


@dataclass(frozen=True)
class SkeletonIntersection:
    center: PixelPoint
    pixels: tuple[PixelPoint, ...]
    radius: float | None


@dataclass(frozen=True)
class EdgeTangent:
    edge_id: int
    point: AxisPoint
    direction: AxisPoint


@dataclass(frozen=True)
class IntersectionTangents:
    center: PixelPoint
    radius: float
    tangents: tuple[EdgeTangent, ...]
    crossings: tuple[AxisPoint, ...]
    focus: AxisPoint | None


@dataclass(frozen=True)
class SampledGraph:
    points: tuple[AxisPoint, ...]
    segments: tuple[tuple[int, int], ...]
    foci: tuple[int, ...]


@dataclass(frozen=True)
class SkeletonGraph:
    width: int
    height: int
    adjacency: dict[PixelPoint, tuple[PixelPoint, ...]]
    edges: tuple[SkeletonEdge, ...]
    endpoints: tuple[PixelPoint, ...]
    intersections: tuple[SkeletonIntersection, ...]

    def to_dict(self) -> SkeletonGraphData:
        ordered_points = sorted(self.adjacency, key=lambda point: (point[1], point[0]))
        node_ids = {point: index for index, point in enumerate(ordered_points)}
        return {
            "width": self.width,
            "height": self.height,
            "nodes": [
                {
                    "id": node_ids[point],
                    "x": point[0],
                    "y": point[1],
                    "adjacent": [node_ids[neighbor] for neighbor in self.adjacency[point]],
                }
                for point in ordered_points
            ],
            "edges": [
                {
                    "id": edge.id,
                    "nodes": [node_ids[point] for point in edge.pixels],
                }
                for edge in self.edges
            ],
            "endpoints": [node_ids[point] for point in self.endpoints],
            "intersections": [
                {
                    "center": node_ids[intersection.center],
                    "nodes": [node_ids[point] for point in intersection.pixels],
                    "radius": intersection.radius,
                }
                for intersection in self.intersections
            ],
        }


def sample_axis_lines(
    edges: tuple[SkeletonEdge, ...], spacing: float
) -> tuple[tuple[AxisPoint, ...], ...]:
    """Sample every traced axis line at an even arc-length spacing."""
    if spacing <= 0:
        raise ValueError("The sample spacing must be positive")

    samples = []
    for edge in edges:
        pixels = edge.pixels
        if not pixels:
            samples.append(())
            continue

        points = [(float(x), float(y)) for x, y in pixels]
        if len(points) == 1:
            samples.append((points[0],))
            continue

        segments = [
            (start, end, hypot(end[0] - start[0], end[1] - start[1]))
            for start, end in pairwise(points)
        ]
        total_length = sum(length for _, _, length in segments)
        if total_length == 0:
            samples.append((points[0],))
            continue

        edge_samples = [points[0]]
        target = spacing
        traversed = 0.0
        for start, end, length in segments:
            while target < traversed + length:
                fraction = (target - traversed) / length
                edge_samples.append(
                    (
                        start[0] + fraction * (end[0] - start[0]),
                        start[1] + fraction * (end[1] - start[1]),
                    )
                )
                target += spacing
            traversed += length

        if points[0] != points[-1]:
            edge_samples.append(points[-1])
        samples.append(tuple(edge_samples))

    return tuple(samples)


def neighboring_pixels(
    point: PixelPoint, pixels: set[PixelPoint]
) -> tuple[PixelPoint, ...]:
    x, y = point
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbor = (x + dx, y + dy)
            if not (dx or dy) or neighbor not in pixels:
                continue
            if dx and dy and ((x + dx, y) in pixels or (x, y + dy) in pixels):
                continue
            neighbors.append(neighbor)
    return tuple(sorted(neighbors, key=lambda item: (item[1], item[0])))


def _trace_edges(
    adjacency: dict[PixelPoint, tuple[PixelPoint, ...]],
) -> tuple[SkeletonEdge, ...]:
    visited: set[frozenset[PixelPoint]] = set()
    paths: list[tuple[PixelPoint, ...]] = []

    def walk(start: PixelPoint, following: PixelPoint) -> tuple[PixelPoint, ...]:
        path = [start, following]
        visited.add(frozenset((start, following)))
        current = following
        while len(adjacency[current]) == 2:
            candidates = [
                neighbor
                for neighbor in adjacency[current]
                if frozenset((current, neighbor)) not in visited
            ]
            if not candidates:
                break
            following = candidates[0]
            visited.add(frozenset((current, following)))
            path.append(following)
            current = following
            if current == start:
                break
        return tuple(path)

    ordered_points = sorted(adjacency, key=lambda point: (point[1], point[0]))
    for point in ordered_points:
        if len(adjacency[point]) == 2:
            continue
        for neighbor in adjacency[point]:
            link = frozenset((point, neighbor))
            if link not in visited:
                paths.append(walk(point, neighbor))

    for point in ordered_points:
        if len(adjacency[point]) != 2:
            continue
        unvisited = [
            neighbor
            for neighbor in adjacency[point]
            if frozenset((point, neighbor)) not in visited
        ]
        if unvisited:
            paths.append(walk(point, unvisited[0]))

    return tuple(SkeletonEdge(index, path) for index, path in enumerate(paths))


def _intersection_components(
    adjacency: dict[PixelPoint, tuple[PixelPoint, ...]],
) -> list[set[PixelPoint]]:
    candidates = {point for point, neighbors in adjacency.items() if len(neighbors) > 2}
    components = []
    while candidates:
        seed = min(candidates, key=lambda point: (point[1], point[0]))
        component = {seed}
        pending = [seed]
        candidates.remove(seed)
        while pending:
            point = pending.pop()
            connected = [neighbor for neighbor in adjacency[point] if neighbor in candidates]
            candidates.difference_update(connected)
            component.update(connected)
            pending.extend(connected)
        components.append(component)
    return components


def _inscribed_radius(distance: np.ndarray, center: PixelPoint) -> float:
    """The largest inscribed disc whose centre lies in the junction's own disc.

    Thinning does not always leave the branch pixel on the thickest part of a
    junction, and the radius sampled there can fall below the stroke's own —
    a disc that small stops short of the bend it exists to cut away. Searching
    the disc the branch pixel already sits in keeps every candidate inside the
    ink, so the result is scaled by the junction rather than by a global width.
    """
    x, y = center
    base = float(distance[y, x])
    span = int(base)
    if span < 1:
        return base

    height, width = distance.shape
    top, bottom = max(0, y - span), min(height, y + span + 1)
    left, right = max(0, x - span), min(width, x + span + 1)
    rows, columns = np.ogrid[top:bottom, left:right]
    within = (rows - y) ** 2 + (columns - x) ** 2 <= base * base
    return float(np.max(np.where(within, distance[top:bottom, left:right], base)))


def _component_center(
    component: set[PixelPoint],
    adjacency: dict[PixelPoint, tuple[PixelPoint, ...]],
) -> PixelPoint:
    mean_x = sum(point[0] for point in component) / len(component)
    mean_y = sum(point[1] for point in component) / len(component)
    return min(
        component,
        key=lambda point: (
            hypot(point[0] - mean_x, point[1] - mean_y),
            -len(adjacency[point]),
            point[1],
            point[0],
        ),
    )


def build_skeleton_graph(
    axis: np.ndarray, binary: np.ndarray | None = None, radius_scale: float = 1.0
) -> SkeletonGraph:
    """Build pixel adjacency, traced edges, endpoints, and junctions.

    ``radius_scale`` widens or narrows every junction's maximal-inscribed
    radius, which is the radius the cutting step removes and the one the
    tangent fit then works from.
    """
    if axis.ndim != 2:
        raise ValueError("The skeleton must be a two-dimensional image")
    if binary is not None and binary.shape != axis.shape:
        raise ValueError("The binary image and skeleton must have the same shape")
    if radius_scale <= 0:
        raise ValueError("The intersection radius scale must be positive")

    ys, xs = np.nonzero(axis < 128)
    pixels: set[PixelPoint] = set(zip(xs.tolist(), ys.tolist()))
    adjacency = {
        point: neighboring_pixels(point, pixels)
        for point in sorted(pixels, key=lambda item: (item[1], item[0]))
    }
    distances = (
        cast(np.ndarray, distance_transform_edt(binary < 128))
        if binary is not None
        else None
    )
    intersections = []
    for component in _intersection_components(adjacency):
        center = _component_center(component, adjacency)
        radius = (
            _inscribed_radius(distances, center) * radius_scale
            if distances is not None
            else None
        )
        intersections.append(
            SkeletonIntersection(
                center=center,
                pixels=tuple(sorted(component, key=lambda point: (point[1], point[0]))),
                radius=radius,
            )
        )

    height, width = axis.shape
    return SkeletonGraph(
        width=width,
        height=height,
        adjacency=adjacency,
        edges=_trace_edges(adjacency),
        endpoints=tuple(
            point for point, neighbors in adjacency.items() if len(neighbors) <= 1
        ),
        intersections=tuple(intersections),
    )


def _cut_intersection(
    point: AxisPoint,
    intersections: tuple[SkeletonIntersection, ...],
    tolerance: float,
) -> PixelPoint | None:
    """The junction whose removed disc this stroke end was left sitting on."""
    nearest: tuple[float, PixelPoint] | None = None
    for intersection in intersections:
        if intersection.radius is None:
            raise ValueError(
                "Intersection radii are unavailable; build the graph with binary"
            )
        distance = hypot(
            point[0] - intersection.center[0], point[1] - intersection.center[1]
        )
        if distance > intersection.radius + tolerance:
            continue
        if nearest is None or distance < nearest[0]:
            nearest = (distance, intersection.center)
    return None if nearest is None else nearest[1]


def _tail_run(pixels: tuple[PixelPoint, ...], spacing: float) -> tuple[PixelPoint, ...]:
    """The pixels a stroke end spends inside its first sample interval."""
    run = [pixels[0]]
    traversed = 0.0
    for start, end in pairwise(pixels):
        traversed += hypot(end[0] - start[0], end[1] - start[1])
        run.append(end)
        if traversed >= spacing:
            break
    return tuple(run)


def _tangent_direction(run: tuple[PixelPoint, ...]) -> AxisPoint | None:
    """Total-least-squares direction of a run, pointing out through its head."""
    if len(run) < 2:
        return None
    samples = np.asarray(run, dtype=float)
    origin = samples.mean(axis=0)
    direction = np.linalg.svd(samples - origin, full_matrices=False)[2][0]
    if float(direction @ (samples[0] - origin)) < 0:
        direction = -direction
    return (float(direction[0]), float(direction[1]))


def _line_crossing(
    first: EdgeTangent, second: EdgeTangent, parallel_tolerance: float
) -> AxisPoint | None:
    turn = (
        first.direction[0] * second.direction[1]
        - first.direction[1] * second.direction[0]
    )
    if abs(turn) < parallel_tolerance:
        return None
    offset_x = second.point[0] - first.point[0]
    offset_y = second.point[1] - first.point[1]
    step = (offset_x * second.direction[1] - offset_y * second.direction[0]) / turn
    return (
        first.point[0] + step * first.direction[0],
        first.point[1] + step * first.direction[1],
    )


def _centroid(points: list[AxisPoint]) -> AxisPoint:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _overlapping_groups(
    junctions: list[IntersectionTangents],
) -> list[list[IntersectionTangents]]:
    """Group junctions whose removed discs overlap.

    Thinning often leaves a crossing as two adjacent branch points rather than
    one. Their discs overlap, so no stroke survives between them and they
    describe a single junction that has to be reassembled.
    """
    groups: list[list[IntersectionTangents]] = []
    for junction in junctions:
        merged = [junction]
        for group in list(groups):
            if any(
                hypot(
                    junction.center[0] - member.center[0],
                    junction.center[1] - member.center[1],
                )
                < junction.radius + member.radius
                for member in group
            ):
                merged.extend(group)
                groups.remove(group)
        groups.append(merged)
    return groups


def _merge_junctions(group: list[IntersectionTangents]) -> IntersectionTangents:
    """Fuse a group into one junction sitting at the mean of their foci."""
    foci = [member.focus for member in group if member.focus is not None]
    widest = max(group, key=lambda member: member.radius)
    return IntersectionTangents(
        center=widest.center,
        radius=widest.radius,
        tangents=tuple(
            tangent for member in group for tangent in member.tangents
        ),
        crossings=tuple(
            crossing for member in group for crossing in member.crossings
        ),
        focus=_centroid(foci) if foci else None,
    )


def intersection_tangents(
    intersections: tuple[SkeletonIntersection, ...],
    edges: tuple[SkeletonEdge, ...],
    spacing: float,
    parallel_tolerance: float = 0.2,
    reach: float = 3.0,
    tolerance: float = 2.0,
) -> tuple[IntersectionTangents, ...]:
    """Take a tangent at every stroke end the intersection cut left behind.

    The tangent is fitted to the pixels a stroke end spends inside its first
    sample interval, so ``spacing`` sets the arc length the fit spans. Each
    junction's tangents are extended until they cross one another; the focus is
    the centroid of those crossings, which is the midpoint for the two
    crossings of a corner and the triangle centre for the three of a T.
    Crossings between near-parallel tangents run off to infinity and crossings
    beyond ``reach`` radii belong to no stroke, so both are discarded.

    Junctions whose discs overlap are then fused, and the fused junction sits
    at the mean of their foci.
    """
    if spacing <= 0:
        raise ValueError("The sample spacing must be positive")

    collected: dict[PixelPoint, list[EdgeTangent]] = {
        intersection.center: [] for intersection in intersections
    }
    for edge in edges:
        if len(edge.pixels) < 2:
            continue
        for pixels in (edge.pixels, edge.pixels[::-1]):
            run = _tail_run(pixels, spacing)
            center = _cut_intersection(run[0], intersections, tolerance)
            direction = _tangent_direction(run)
            if center is None or direction is None:
                continue
            collected[center].append(
                EdgeTangent(edge.id, (float(run[0][0]), float(run[0][1])), direction)
            )

    results = []
    for intersection in intersections:
        tangents = tuple(collected[intersection.center])
        radius = cast(float, intersection.radius)
        crossings = []
        for first, second in combinations(tangents, 2):
            crossing = _line_crossing(first, second, parallel_tolerance)
            if crossing is None:
                continue
            distance = hypot(
                crossing[0] - intersection.center[0],
                crossing[1] - intersection.center[1],
            )
            if distance <= reach * radius:
                crossings.append(crossing)
        results.append(
            IntersectionTangents(
                center=intersection.center,
                radius=radius,
                tangents=tangents,
                crossings=tuple(crossings),
                focus=_centroid(crossings) if crossings else None,
            )
        )
    return tuple(_merge_junctions(group) for group in _overlapping_groups(results))


def merge_tangent_foci(
    samples: tuple[tuple[AxisPoint, ...], ...],
    junctions: tuple[IntersectionTangents, ...],
) -> SampledGraph:
    """Add each junction's focus to the sampled strokes and rejoin them there.

    Every stroke keeps its own chain of samples; the ends the cut left behind
    gain a segment back to their junction's focus, so the strokes a junction
    separated meet again at one shared node.
    """
    points: list[AxisPoint] = []
    indices: dict[AxisPoint, int] = {}

    def node(point: AxisPoint) -> int:
        if point not in indices:
            indices[point] = len(points)
            points.append(point)
        return indices[point]

    segments = [
        (node(start), node(end))
        for edge_samples in samples
        for start, end in pairwise(edge_samples)
    ]
    foci = []
    for junction in junctions:
        if junction.focus is None:
            continue
        focus = node(junction.focus)
        foci.append(focus)
        segments.extend((focus, node(tangent.point)) for tangent in junction.tangents)
    return SampledGraph(tuple(points), tuple(segments), tuple(foci))


def remove_intersection_neighborhoods(
    axis: np.ndarray, intersections: tuple[SkeletonIntersection, ...]
) -> np.ndarray:
    """Remove skeleton pixels inside each junction's maximal-inscribed circle."""
    cleaned = axis.copy()
    skeleton_y, skeleton_x = np.nonzero(axis < 128)
    for intersection in intersections:
        if intersection.radius is None:
            raise ValueError(
                "Intersection radii are unavailable; build the graph with binary"
            )
        center_x, center_y = intersection.center
        within_circle = (
            (skeleton_x - center_x) ** 2 + (skeleton_y - center_y) ** 2
            <= intersection.radius**2
        )
        cleaned[skeleton_y[within_circle], skeleton_x[within_circle]] = 255
    return cleaned
