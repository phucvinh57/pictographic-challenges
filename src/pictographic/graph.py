from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import acos, degrees, hypot
from typing import TypedDict, cast

import numpy as np
from scipy.ndimage import distance_transform_edt

PixelPoint = tuple[int, int]
CORNER_LOOKAHEAD = 10 


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
    angle_threshold: float,
) -> tuple[SkeletonEdge, ...]:
    visited: set[frozenset[PixelPoint]] = set()
    paths: list[tuple[PixelPoint, ...]] = []
    breaks = {
        point
        for point, neighbors in adjacency.items()
        if len(neighbors) != 2
    }

    def walk(start: PixelPoint, following: PixelPoint) -> tuple[PixelPoint, ...]:
        path = [start, following]
        visited.add(frozenset((start, following)))
        current = following
        while current not in breaks:
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
        if point not in breaks:
            continue
        for neighbor in adjacency[point]:
            link = frozenset((point, neighbor))
            if link not in visited:
                paths.append(walk(point, neighbor))

    for point in ordered_points:
        if point in breaks:
            continue
        unvisited = [
            neighbor
            for neighbor in adjacency[point]
            if frozenset((point, neighbor)) not in visited
        ]
        if unvisited:
            paths.append(walk(point, unvisited[0]))

    lines = [
        line
        for path in paths
        for line in split_at_corners(path, angle_threshold)
    ]
    return tuple(SkeletonEdge(index, line) for index, line in enumerate(lines))


def split_at_corners(
    path: tuple[PixelPoint, ...], angle_threshold: float
) -> tuple[tuple[PixelPoint, ...], ...]:
    closed = len(path) > 2 and path[0] == path[-1]
    points = path[:-1] if closed else path
    size = len(points)
    if size < 2 * CORNER_LOOKAHEAD + 1:
        return (path,)

    indices = (
        range(size)
        if closed
        else range(CORNER_LOOKAHEAD, size - CORNER_LOOKAHEAD)
    )
    candidates = []
    for index in indices:
        point = points[index]
        before = points[(index - CORNER_LOOKAHEAD) % size]
        after = points[(index + CORNER_LOOKAHEAD) % size]
        incoming = (point[0] - before[0], point[1] - before[1])
        outgoing = (after[0] - point[0], after[1] - point[1])
        product = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
        lengths = hypot(*incoming) * hypot(*outgoing)
        turn = degrees(acos(max(-1.0, min(1.0, product / lengths))))
        if turn >= angle_threshold:
            candidates.append((index, turn))

    if not candidates:
        return (path,)

    groups: list[list[tuple[int, float]]] = []
    for candidate in candidates:
        if (
            not groups
            or candidate[0] - groups[-1][-1][0] > CORNER_LOOKAHEAD
        ):
            groups.append([candidate])
        else:
            groups[-1].append(candidate)
    if (
        closed
        and len(groups) > 1
        and groups[0][0][0] + size - groups[-1][-1][0]
        <= CORNER_LOOKAHEAD
    ):
        groups[0] = groups[-1] + groups[0]
        groups.pop()

    corners = sorted(
        max(group, key=lambda candidate: (candidate[1], -candidate[0]))[0]
        for group in groups
    )
    if not closed:
        boundaries = [0, *corners, size - 1]
        return tuple(
            tuple(points[start : end + 1])
            for start, end in pairwise(boundaries)
        )

    lines = []
    for start, end in pairwise([*corners, corners[0]]):
        line = (
            points[start : end + 1]
            if start < end
            else points[start:] + points[: end + 1]
        )
        lines.append(tuple(line))
    return tuple(lines)


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
    axis: np.ndarray,
    binary: np.ndarray | None,
    angle_threshold: float,
) -> SkeletonGraph:
    """Build adjacency and trace lines between endpoints, junctions, and corners."""
    image = np.asarray(axis)
    foreground = image if image.dtype == np.bool_ else image < 128
    ys, xs = np.nonzero(foreground)
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
            _inscribed_radius(distances, center) if distances is not None else None
        )
        intersections.append(
            SkeletonIntersection(
                center=center,
                pixels=tuple(sorted(component, key=lambda point: (point[1], point[0]))),
                radius=radius,
            )
        )

    height, width = image.shape
    return SkeletonGraph(
        width=width,
        height=height,
        adjacency=adjacency,
        edges=_trace_edges(adjacency, angle_threshold),
        endpoints=tuple(
            point for point, neighbors in adjacency.items() if len(neighbors) <= 1
        ),
        intersections=tuple(intersections),
    )
