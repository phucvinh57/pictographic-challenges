from dataclasses import dataclass, field
from enum import Enum

import numpy as np

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


class NodeType(Enum):
    ENDPOINT = 1
    JUNCTION = 2


@dataclass
class Node:
    pos: Pixel
    radius: int
    type: NodeType


@dataclass
class Edge:
    start: Node
    end: Node
    nodes: list[Node] = field(default_factory=list)


@dataclass
class Graph:
    nodes: dict[Pixel, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    junctions: list[Node] = field(default_factory=list)


def build_graph(medial: MedialAxis) -> Graph:
    pixels = {(int(x), int(y)) for y, x in np.argwhere(medial.axis)}
    graph = Graph()

    # Every later phase needs a pixel's 8-connected neighbors.  Build this
    # once instead of repeatedly probing the same eight positions.
    neighbors: dict[Pixel, tuple[Pixel, ...]] = {}
    for pixel in pixels:
        x, y = pixel
        neighbors[pixel] = tuple(
            (x + dx, y + dy)
            for dx, dy in _OFFSETS
            if (x + dx, y + dy) in pixels
        )

    def to_node(pixel: Pixel) -> Node:
        x, y = pixel
        radius = int(medial.distance[y, x])
        return Node(
            pos=pixel,
            radius=radius,
            type=NodeType.ENDPOINT,
        )

    degrees = {pixel: len(adjacent) for pixel, adjacent in neighbors.items()}
    node_at: dict[Pixel, Node] = {}
    node_groups: list[tuple[Node, set[Pixel]]] = []

    def add_node(members: set[Pixel], node_type: NodeType) -> Node:
        node = to_node(min(members))
        node.type = node_type
        graph.nodes[node.pos] = node
        node_groups.append((node, members))
        for pixel in members:
            node_at[pixel] = node
        if node_type is NodeType.JUNCTION:
            graph.junctions.append(node)
        return node

    remaining_junctions = {
        pixel for pixel, degree in degrees.items() if degree >= 3
    }
    while remaining_junctions:
        seed = remaining_junctions.pop()
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

    for pixel, degree in degrees.items():
        if degree == 1:
            add_node({pixel}, NodeType.ENDPOINT)

    unseen = set(pixels)
    while unseen:
        seed = unseen.pop()
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

    # Store ordered endpoints rather than allocating a frozenset for every
    # traversed link.  Pixel tuples have a stable, lexicographic ordering.
    visited_links: set[tuple[Pixel, Pixel]] = set()

    def link_key(first: Pixel, second: Pixel) -> tuple[Pixel, Pixel]:
        return (first, second) if first < second else (second, first)

    for start, members in node_groups:
        for node_pixel in members:
            for first in neighbors[node_pixel]:
                if first in members:
                    continue

                first_link = link_key(node_pixel, first)
                if first_link in visited_links:
                    continue

                visited_links.add(first_link)
                previous = node_pixel
                current = first

                while current not in node_at:
                    adjacent = neighbors[current]
                    if len(adjacent) != 2:
                        raise ValueError(
                            f"Unexpected skeleton topology at {current}"
                        )

                    next_pixel = (
                        adjacent[1] if adjacent[0] == previous else adjacent[0]
                    )
                    visited_links.add(link_key(current, next_pixel))
                    previous, current = current, next_pixel

                end = node_at[current]
                graph.edges.append(Edge(start=start, end=end, nodes=[start, end]))

    return graph
