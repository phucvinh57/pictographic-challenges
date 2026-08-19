from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .constants import TRANSITION_DELTA, TRANSITION_FLATNESS
from .skeleton import MedialAxis

Pixel = tuple[int, int]

_OFFSETS = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)
_FILTER_SIZE = 5
_MIN_SEGMENT_SAMPLES = 5


class NodeType(Enum):
    ENDPOINT = 1
    JUNCTION = 2
    TRANSITION = 3


@dataclass
class Node:
    pos: Pixel
    radius: float
    type: NodeType


@dataclass
class RadiusProfile:
    distances: np.ndarray
    radii: np.ndarray
    filtered: np.ndarray
    fitted: np.ndarray
    transition_indices: list[int] = field(default_factory=list)
    decision: str = "none"


@dataclass
class Edge:
    start: Node
    end: Node
    pixels: list[Pixel] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    profile: RadiusProfile | None = None
    transition_sides: dict[Pixel, str] = field(default_factory=dict)


@dataclass
class Graph:
    nodes: dict[Pixel, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    junctions: list[Node] = field(default_factory=list)
    transitions: list[Node] = field(default_factory=list)
    _node_pixels: set[Pixel] = field(default_factory=set, repr=False)


@dataclass(frozen=True)
class _HingeFit:
    index: int
    fitted: np.ndarray


def _median_filter(values: np.ndarray) -> np.ndarray:
    if len(values) < _FILTER_SIZE:
        return values.copy()
    padding = _FILTER_SIZE // 2
    padded = np.pad(values, padding, mode="edge")
    return np.asarray(
        [
            np.median(padded[index : index + _FILTER_SIZE])
            for index in range(len(values))
        ],
        dtype=np.float64,
    )


def _line_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    matrix = np.column_stack((np.ones(len(x)), x))
    intercept, slope = np.linalg.lstsq(matrix, y, rcond=None)[0]
    fitted = matrix @ np.array((intercept, slope))
    error = float(np.sum(np.square(y - fitted)))
    return fitted, float(slope), error


def _bic(error: float, samples: int, parameters: int) -> float:
    variance = max(error / samples, np.finfo(np.float64).eps)
    return samples * float(np.log(variance)) + parameters * float(np.log(samples))


def _fit_hinge(
    distances: np.ndarray,
    radii: np.ndarray,
    transition_delta: float,
    transition_flatness: float,
) -> _HingeFit | None:
    minimum_length = 2 * _MIN_SEGMENT_SAMPLES - 1
    if len(radii) < minimum_length:
        return None

    _, _, line_error = _line_fit(distances, radii)
    line_bic = _bic(line_error, len(radii), 2)
    best: tuple[float, int, np.ndarray] | None = None

    first = _MIN_SEGMENT_SAMPLES - 1
    last = len(radii) - _MIN_SEGMENT_SAMPLES
    for index in range(first, last + 1):
        falling_distance = np.maximum(distances[index] - distances, 0.0)
        matrix = np.column_stack((np.ones(len(radii)), falling_distance))
        plateau, rise = np.linalg.lstsq(matrix, radii, rcond=None)[0]
        if rise <= 0:
            continue

        fitted = matrix @ np.array((plateau, rise))
        error = float(np.sum(np.square(radii - fitted)))
        score = _bic(error, len(radii), 3)
        if best is None or score < best[0]:
            best = score, index, fitted

    if best is None or best[0] >= line_bic:
        return None

    _, index, fitted = best
    _, falling_slope, _ = _line_fit(distances[: index + 1], radii[: index + 1])
    _, plateau_slope, _ = _line_fit(distances[index:], radii[index:])
    plateau_radius = float(np.mean(radii[index:]))
    radius_drop = float(radii[0] - plateau_radius)
    if falling_slope >= 0:
        return None
    if radius_drop < transition_delta * float(radii[0]):
        return None
    if abs(plateau_slope) > transition_flatness * abs(falling_slope):
        return None
    return _HingeFit(index=index, fitted=fitted)


def _make_profile(edge: Edge, medial: MedialAxis) -> RadiusProfile:
    coordinates = np.asarray(edge.pixels, dtype=np.float64)
    if len(coordinates) < 2:
        distances = np.zeros(len(coordinates), dtype=np.float64)
    else:
        steps = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
        distances = np.concatenate((np.zeros(1), np.cumsum(steps)))
    radii = np.asarray(
        [medial.distance[y, x] for x, y in edge.pixels], dtype=np.float64
    )
    filtered = _median_filter(radii)
    return RadiusProfile(
        distances=distances,
        radii=radii,
        filtered=filtered,
        fitted=np.full(len(radii), np.nan, dtype=np.float64),
    )


def _material_drop(
    junction_radius: float, stable_radius: float, transition_delta: float
) -> bool:
    return junction_radius - stable_radius >= transition_delta * junction_radius


def _add_transition(graph: Graph, edge: Edge, index: int, side: str) -> Node | None:
    if index <= 0 or index >= len(edge.pixels) - 1:
        return None
    pixel = edge.pixels[index]
    if pixel in graph._node_pixels:
        return None
    existing = graph.nodes.get(pixel)
    if existing is not None and existing.type is not NodeType.TRANSITION:
        return None
    if existing is None:
        profile = edge.profile
        if profile is None:
            raise ValueError("Edge radius profile is missing")
        existing = Node(
            pos=pixel,
            radius=float(profile.radii[index]),
            type=NodeType.TRANSITION,
        )
        graph.nodes[pixel] = existing
        graph.transitions.append(existing)
    previous_side = edge.transition_sides.get(pixel)
    edge.transition_sides[pixel] = (
        side if previous_side is None or previous_side == side else "S/E"
    )
    return existing


def _finish_edge_nodes(edge: Edge, transitions: list[tuple[int, Node]]) -> None:
    by_index = {index: node for index, node in transitions}
    ordered = [node for _, node in sorted(by_index.items())]
    edge.nodes = [edge.start, *ordered, edge.end]
    profile = edge.profile
    if profile is not None:
        profile.transition_indices = sorted({index for index, _ in transitions})


def _transitions_available(graph: Graph, edge: Edge, indices: list[int]) -> bool:
    return all(
        0 < index < len(edge.pixels) - 1
        and edge.pixels[index] not in graph._node_pixels
        for index in indices
    )


def _use_endpoint_minimum(
    graph: Graph,
    edge: Edge,
    oriented_radii: np.ndarray,
    reverse: bool,
    side: str,
    transition_delta: float,
) -> None:
    profile = edge.profile
    if profile is None:
        return
    minimum_radius = float(np.min(oriented_radii))
    if not _material_drop(float(oriented_radii[0]), minimum_radius, transition_delta):
        return
    minimum_indices = np.flatnonzero(
        np.isclose(oriented_radii, minimum_radius, rtol=1e-9, atol=1e-9)
    )
    oriented_index = int(minimum_indices[0])
    index = len(edge.pixels) - 1 - oriented_index if reverse else oriented_index
    pixel = edge.pixels[index]

    if index in (0, len(edge.pixels) - 1):
        transition = edge.start if index == 0 else edge.end
        if all(node.pos != transition.pos for node in graph.transitions):
            graph.transitions.append(transition)
        edge.transition_sides[pixel] = side
        profile.transition_indices = [index]
    else:
        transition = _add_transition(graph, edge, index, side)
        if transition is None:
            return
        _finish_edge_nodes(edge, [(index, transition)])

    profile.fitted.fill(minimum_radius)
    profile.decision = "minimum"


def _detect_endpoint_transition(
    graph: Graph,
    edge: Edge,
    transition_delta: float,
    transition_flatness: float,
) -> None:
    profile = edge.profile
    if profile is None:
        return
    if edge.start.type is NodeType.JUNCTION:
        oriented_distances = profile.distances
        oriented_radii = profile.filtered
        oriented_raw_radii = profile.radii
        reverse = False
        side = "S"
    elif edge.end.type is NodeType.JUNCTION:
        oriented_distances = profile.distances[-1] - profile.distances[::-1]
        oriented_radii = profile.filtered[::-1]
        oriented_raw_radii = profile.radii[::-1]
        reverse = True
        side = "E"
    else:
        return

    hinge = _fit_hinge(
        oriented_distances,
        oriented_radii,
        transition_delta,
        transition_flatness,
    )
    if hinge is None:
        _use_endpoint_minimum(
            graph,
            edge,
            oriented_raw_radii,
            reverse,
            side,
            transition_delta,
        )
        return
    index = len(edge.pixels) - 1 - hinge.index if reverse else hinge.index
    fitted = hinge.fitted[::-1] if reverse else hinge.fitted
    transition = _add_transition(graph, edge, index, side)
    if transition is None:
        _use_endpoint_minimum(
            graph,
            edge,
            oriented_raw_radii,
            reverse,
            side,
            transition_delta,
        )
        return
    profile.fitted[:] = fitted
    profile.decision = "knee"
    _finish_edge_nodes(edge, [(index, transition)])


def _detect_junction_transitions(
    graph: Graph,
    edge: Edge,
    transition_delta: float,
    transition_flatness: float,
) -> None:
    profile = edge.profile
    if profile is None or len(profile.radii) < 3:
        return

    center = len(profile.radii) // 2
    left_distances = profile.distances[: center + 1]
    left_radii = profile.filtered[: center + 1]
    left_raw_radii = profile.radii[: center + 1]
    right_distances = profile.distances[-1] - profile.distances[center:][::-1]
    right_radii = profile.filtered[center:][::-1]
    right_raw_radii = profile.radii[center:][::-1]

    sides = (
        ("S", left_distances, left_radii, left_raw_radii, False),
        ("E", right_distances, right_radii, right_raw_radii, True),
    )
    candidates: list[tuple[int, str, np.ndarray, str]] = []
    for side, distances, radii, raw_radii, reverse in sides:
        hinge = _fit_hinge(distances, radii, transition_delta, transition_flatness)
        if hinge is not None:
            oriented_index = hinge.index
            fitted = hinge.fitted
            decision = "knee"
        else:
            minimum_radius = float(np.min(raw_radii))
            if not _material_drop(
                float(raw_radii[0]), minimum_radius, transition_delta
            ):
                continue
            minimum_indices = np.flatnonzero(
                np.isclose(raw_radii, minimum_radius, rtol=1e-9, atol=1e-9)
            )
            oriented_index = int(minimum_indices[0])
            fitted = np.full(len(raw_radii), minimum_radius, dtype=np.float64)
            decision = "minimum"

        index = len(edge.pixels) - 1 - oriented_index if reverse else oriented_index
        candidates.append((index, side, fitted, decision))

    indices = [index for index, _, _, _ in candidates]
    if len(indices) != len(set(indices)):
        return
    available = [
        candidate
        for candidate in candidates
        if _transitions_available(graph, edge, [candidate[0]])
    ]

    transitions: list[tuple[int, Node]] = []
    decisions: list[str] = []
    for index, side, fitted, decision in available:
        transition = _add_transition(graph, edge, index, side)
        if transition is None:
            continue
        transitions.append((index, transition))
        decisions.append(decision)
        if side == "S":
            profile.fitted[: center + 1] = fitted
        else:
            profile.fitted[center:] = fitted[::-1]

    if not transitions:
        return
    profile.decision = decisions[0] if len(set(decisions)) == 1 else "/".join(decisions)
    _finish_edge_nodes(edge, transitions)


def _detect_transitions(
    graph: Graph,
    medial: MedialAxis,
    transition_delta: float,
    transition_flatness: float,
) -> None:
    for edge in graph.edges:
        edge.profile = _make_profile(edge, medial)
        edge.nodes = [edge.start, edge.end]
        start_junction = edge.start.type is NodeType.JUNCTION
        end_junction = edge.end.type is NodeType.JUNCTION
        if start_junction and end_junction:
            _detect_junction_transitions(
                graph, edge, transition_delta, transition_flatness
            )
        elif start_junction or end_junction:
            _detect_endpoint_transition(
                graph, edge, transition_delta, transition_flatness
            )


def build_graph(
    medial: MedialAxis,
    transition_delta: float = TRANSITION_DELTA,
    transition_flatness: float = TRANSITION_FLATNESS,
) -> Graph:
    pixels = {(int(x), int(y)) for y, x in np.argwhere(medial.axis)}
    graph = Graph()
    neighbors: dict[Pixel, tuple[Pixel, ...]] = {}
    for pixel in pixels:
        x, y = pixel
        neighbors[pixel] = tuple(
            sorted(
                (x + dx, y + dy) for dx, dy in _OFFSETS if (x + dx, y + dy) in pixels
            )
        )

    def to_node(members: set[Pixel], node_type: NodeType) -> Node:
        position = min(
            members,
            key=lambda pixel: (-medial.distance[pixel[1], pixel[0]], pixel),
        )
        x, y = position
        return Node(
            pos=position,
            radius=float(medial.distance[y, x]),
            type=node_type,
        )

    degrees = {pixel: len(adjacent) for pixel, adjacent in neighbors.items()}
    node_at: dict[Pixel, Node] = {}
    node_groups: list[tuple[Node, set[Pixel]]] = []

    def add_node(members: set[Pixel], node_type: NodeType) -> Node:
        node = to_node(members, node_type)
        graph.nodes[node.pos] = node
        node_groups.append((node, members))
        for pixel in members:
            node_at[pixel] = node
            graph._node_pixels.add(pixel)
        if node_type is NodeType.JUNCTION:
            graph.junctions.append(node)
        return node

    remaining_junctions = {pixel for pixel, degree in degrees.items() if degree >= 3}
    while remaining_junctions:
        seed = min(remaining_junctions)
        remaining_junctions.remove(seed)
        group = {seed}
        pending = [seed]

        while pending:
            current = pending.pop()
            for neighbor in neighbors[current]:
                if neighbor in remaining_junctions:
                    remaining_junctions.remove(neighbor)
                    group.add(neighbor)
                    pending.append(neighbor)

        add_node(group, NodeType.JUNCTION)

    for pixel, degree in sorted(degrees.items()):
        if degree == 1:
            add_node({pixel}, NodeType.ENDPOINT)

    unseen = set(pixels)
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        component = {seed}
        pending = [seed]

        while pending:
            current = pending.pop()
            for neighbor in neighbors[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)

        if not any(pixel in node_at for pixel in component):
            add_node({min(component)}, NodeType.JUNCTION)

    group_for = {node.pos: members for node, members in node_groups}

    def group_path(start: Pixel, end: Pixel, members: set[Pixel]) -> list[Pixel]:
        if start == end:
            return [start]
        previous: dict[Pixel, Pixel | None] = {start: None}
        pending = [start]
        while pending:
            current = pending.pop(0)
            for neighbor in neighbors[current]:
                if neighbor not in members or neighbor in previous:
                    continue
                previous[neighbor] = current
                if neighbor == end:
                    path = [end]
                    while previous[path[-1]] is not None:
                        previous_pixel = previous[path[-1]]
                        if previous_pixel is None:
                            break
                        path.append(previous_pixel)
                    return list(reversed(path))
                pending.append(neighbor)
        raise ValueError(f"Disconnected junction group between {start} and {end}")

    visited_links: set[tuple[Pixel, Pixel]] = set()

    def link_key(first: Pixel, second: Pixel) -> tuple[Pixel, Pixel]:
        return (first, second) if first < second else (second, first)

    for start, members in node_groups:
        for node_pixel in sorted(members):
            for first in neighbors[node_pixel]:
                if first in members:
                    continue

                first_link = link_key(node_pixel, first)
                if first_link in visited_links:
                    continue

                visited_links.add(first_link)
                path = group_path(start.pos, node_pixel, members)
                path.append(first)
                previous = node_pixel
                current = first

                while current not in node_at:
                    adjacent = neighbors[current]
                    if len(adjacent) != 2:
                        raise ValueError(f"Unexpected skeleton topology at {current}")

                    next_pixel = adjacent[1] if adjacent[0] == previous else adjacent[0]
                    visited_links.add(link_key(current, next_pixel))
                    path.append(next_pixel)
                    previous, current = current, next_pixel

                end = node_at[current]
                end_members = group_for[end.pos]
                tail = group_path(current, end.pos, end_members)
                path.extend(tail[1:])
                graph.edges.append(
                    Edge(start=start, end=end, pixels=path, nodes=[start, end])
                )

    graph.edges.sort(
        key=lambda edge: (
            min(edge.start.pos, edge.end.pos),
            max(edge.start.pos, edge.end.pos),
            tuple(edge.pixels),
        )
    )
    _detect_transitions(graph, medial, transition_delta, transition_flatness)
    return graph


__all__ = [
    "Edge",
    "Graph",
    "Node",
    "NodeType",
    "Pixel",
    "RadiusProfile",
    "build_graph",
]
