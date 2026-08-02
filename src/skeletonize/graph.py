from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import TypedDict, cast

import numpy as np
from scipy.ndimage import distance_transform_edt

PixelPoint = tuple[int, int]


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
    axis: np.ndarray, binary: np.ndarray | None = None
) -> SkeletonGraph:
    """Build pixel adjacency, traced edges, endpoints, and junctions."""
    if axis.ndim != 2:
        raise ValueError("The skeleton must be a two-dimensional image")
    if binary is not None and binary.shape != axis.shape:
        raise ValueError("The binary image and skeleton must have the same shape")

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
            _inscribed_radius(distances, center) if distances is not None else None
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
