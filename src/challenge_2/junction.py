from dataclasses import dataclass
from itertools import combinations
from math import acos, degrees

import numpy as np

from .graph import Edge, Graph, Node, NodeType, Pixel

_TANGENT_SPAN = 10.0
_MAX_CONTINUATION_ERROR = 30.0
_MAX_SINGLE_CONTINUATION_ERROR = 25.0


@dataclass(frozen=True)
class TransitionReference:
    label: str
    position: Pixel


@dataclass(frozen=True)
class CandidateScore:
    first: str
    second: str
    first_error: float
    second_error: float
    accepted: bool


@dataclass(frozen=True)
class JunctionRewrite:
    old_position: Pixel
    old_positions: tuple[Pixel, ...]
    transitions: tuple[TransitionReference, ...]
    candidates: tuple[CandidateScore, ...]
    kind: str
    segments: tuple[tuple[Pixel, Pixel], ...]
    new_junction: Pixel | None = None
    reason: str | None = None


@dataclass(frozen=True)
class JunctionRewriteReport:
    junctions: tuple[JunctionRewrite, ...]


@dataclass(frozen=True)
class _Arm:
    edge_index: int
    transition_index: int
    side: str
    transition: Node
    tangent: tuple[float, float]

    @property
    def label(self) -> str:
        return f"E{self.edge_index}{self.side}"


@dataclass(frozen=True)
class _Candidate:
    first: _Arm
    second: _Arm
    first_error: float
    second_error: float

    @property
    def accepted(self) -> bool:
        return min(self.first_error, self.second_error) <= _MAX_CONTINUATION_ERROR


@dataclass(frozen=True)
class _Resolution:
    junctions: tuple[Node, ...]
    arms: tuple[_Arm, ...]
    candidates: tuple[_Candidate, ...]
    segments: tuple[tuple[Pixel, Pixel], ...]
    kind: str
    new_junction: Pixel | None = None
    reason: str | None = None

    @property
    def resolved(self) -> bool:
        return self.reason is None

    @property
    def junction(self) -> Node:
        return min(self.junctions, key=lambda node: node.pos)


@dataclass(frozen=True)
class _Piece:
    start: Pixel
    end: Pixel
    pixels: tuple[Pixel, ...]


def _unit(vector: np.ndarray) -> tuple[float, float] | None:
    length = float(np.linalg.norm(vector))
    if length <= np.finfo(np.float64).eps:
        return None
    return float(vector[0] / length), float(vector[1] / length)


def _local_samples(edge: Edge, index: int, direction: int) -> np.ndarray:
    samples = [np.asarray(edge.pixels[index], dtype=np.float64)]
    distance = 0.0
    current = index
    while 0 <= current + direction < len(edge.pixels):
        following = current + direction
        start = np.asarray(edge.pixels[current], dtype=np.float64)
        end = np.asarray(edge.pixels[following], dtype=np.float64)
        step = float(np.linalg.norm(end - start))
        if step <= np.finfo(np.float64).eps:
            current = following
            continue
        remaining = _TANGENT_SPAN - distance
        if step >= remaining:
            samples.append(start + (end - start) * (remaining / step))
            break
        samples.append(end)
        distance += step
        current = following
    return np.asarray(samples, dtype=np.float64)


def _fit_tangent(edge: Edge, index: int, direction: int) -> tuple[float, float] | None:
    samples = _local_samples(edge, index, direction)
    reverse = False
    if len(samples) < 2:
        samples = _local_samples(edge, index, -direction)
        reverse = True
    if len(samples) < 2:
        return None
    centered = samples - np.mean(samples, axis=0)
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    tangent = vectors[:, int(np.argmax(values))]
    outward = samples[-1] - samples[0]
    if reverse:
        outward = -outward
    if float(np.dot(tangent, outward)) < 0:
        tangent = -tangent
    return _unit(tangent)


def _transition_index(edge: Edge, side: str) -> int | None:
    profile = edge.profile
    if profile is None:
        return None
    matches = [
        index
        for index in profile.transition_indices
        if side in edge.transition_sides.get(edge.pixels[index], "")
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _make_arm(graph: Graph, edge_index: int, side: str) -> _Arm | None:
    edge = graph.edges[edge_index]
    index = _transition_index(edge, side)
    if index is None:
        return None
    direction = 1 if side == "S" else -1
    tangent = _fit_tangent(edge, index, direction)
    if tangent is None:
        return None
    transition = graph.nodes.get(edge.pixels[index])
    if transition is None:
        return None
    return _Arm(edge_index, index, side, transition, tangent)


def _continuation_error(arm: _Arm, other: _Arm) -> float:
    start = np.asarray(arm.transition.pos, dtype=np.float64)
    end = np.asarray(other.transition.pos, dtype=np.float64)
    chord = _unit(end - start)
    if chord is None:
        return 180.0
    alignment = -float(np.dot(np.asarray(arm.tangent), np.asarray(chord)))
    return degrees(acos(float(np.clip(alignment, -1.0, 1.0))))


def _branch_intersection(arms: tuple[_Arm, ...]) -> Pixel | None:
    tangents = np.asarray([arm.tangent for arm in arms], dtype=np.float64)
    matrix = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    positions = np.asarray([arm.transition.pos for arm in arms], dtype=np.float64)
    values = np.sum(matrix * positions, axis=1)
    intersection, _, rank, _ = np.linalg.lstsq(matrix, values, rcond=None)
    if rank < 2 or not np.all(np.isfinite(intersection)):
        return None
    return int(np.rint(intersection[0])), int(np.rint(intersection[1]))


def _select_disjoint_candidates(
    arms: tuple[_Arm, ...], candidates: tuple[_Candidate, ...]
) -> tuple[_Candidate, ...]:
    accepted = tuple(candidate for candidate in candidates if candidate.accepted)
    if len(arms) != 4 or len(accepted) <= 2:
        return accepted

    for count in range(len(arms) // 2, 0, -1):
        selections = []
        for selection in combinations(accepted, count):
            endpoints = [
                arm.label
                for candidate in selection
                for arm in (candidate.first, candidate.second)
            ]
            if len(endpoints) == len(set(endpoints)):
                selections.append(selection)
        if selections:
            return min(
                selections,
                key=lambda selection: (
                    sum(
                        candidate.first_error + candidate.second_error
                        for candidate in selection
                    ),
                    tuple(
                        (candidate.first.label, candidate.second.label)
                        for candidate in selection
                    ),
                ),
            )
    return ()


def _resolve_junction(graph: Graph, junctions: tuple[Node, ...]) -> _Resolution:
    member_positions = {junction.pos for junction in junctions}
    expected: list[tuple[int, str]] = []
    for edge_index, edge in enumerate(graph.edges):
        start_inside = edge.start.pos in member_positions
        end_inside = edge.end.pos in member_positions
        start_transition = _transition_index(edge, "S")
        end_transition = _transition_index(edge, "E")
        if (
            start_inside
            and end_inside
            and start_transition is None
            and end_transition is None
        ):
            continue
        if start_inside:
            expected.append((edge_index, "S"))
        if end_inside:
            expected.append((edge_index, "E"))

    arms: list[_Arm] = []
    for edge_index, side in expected:
        arm = _make_arm(graph, edge_index, side)
        if arm is None:
            reason = f"missing transition or tangent for E{edge_index}{side}"
            return _Resolution(
                junctions, tuple(arms), (), (), "unresolved", reason=reason
            )
        arms.append(arm)

    if len(arms) < 2:
        return _Resolution(
            junctions,
            tuple(arms),
            (),
            (),
            "unresolved",
            reason="fewer than two transition arms",
        )

    candidates = tuple(
        _Candidate(
            first,
            second,
            _continuation_error(first, second),
            _continuation_error(second, first),
        )
        for first, second in combinations(arms, 2)
        if first.transition.pos != second.transition.pos
    )
    accepted = _select_disjoint_candidates(tuple(arms), candidates)
    if (
        len(accepted) == 1
        and min(accepted[0].first_error, accepted[0].second_error)
        > _MAX_SINGLE_CONTINUATION_ERROR
    ):
        accepted = ()
    if accepted:
        segments = tuple(
            (candidate.first.transition.pos, candidate.second.transition.pos)
            for candidate in accepted
        )
        kind = "two" if len(accepted) == 2 else "one" if len(accepted) == 1 else "many"
        return _Resolution(junctions, tuple(arms), candidates, segments, kind)

    intersection = _branch_intersection(tuple(arms))
    if intersection is None:
        return _Resolution(
            junctions,
            tuple(arms),
            candidates,
            (),
            "unresolved",
            reason="transition branch lines have no unique intersection",
        )
    segments = tuple((arm.transition.pos, intersection) for arm in arms)
    return _Resolution(
        junctions,
        tuple(arms),
        candidates,
        segments,
        "zero",
        new_junction=intersection,
    )


def _line_pixels(start: Pixel, end: Pixel) -> tuple[Pixel, ...]:
    x, y = start
    end_x, end_y = end
    delta_x = abs(end_x - x)
    delta_y = abs(end_y - y)
    step_x = 1 if x < end_x else -1
    step_y = 1 if y < end_y else -1
    error = delta_x - delta_y
    pixels = []
    while True:
        pixels.append((x, y))
        if x == end_x and y == end_y:
            return tuple(pixels)
        doubled = 2 * error
        if doubled > -delta_y:
            error -= delta_y
            x += step_x
        if doubled < delta_x:
            error += delta_x
            y += step_y


def _add_piece(pieces: list[_Piece], pixels: list[Pixel] | tuple[Pixel, ...]) -> None:
    if len(pixels) < 2 or pixels[0] == pixels[-1]:
        return
    pieces.append(_Piece(pixels[0], pixels[-1], tuple(pixels)))


def _candidate_score(candidate: _Candidate, accepted: bool) -> CandidateScore:
    return CandidateScore(
        candidate.first.label,
        candidate.second.label,
        candidate.first_error,
        candidate.second_error,
        accepted,
    )


def _to_report(resolution: _Resolution) -> JunctionRewrite:
    kept = set(resolution.segments) if resolution.kind != "zero" else set()
    return JunctionRewrite(
        old_position=resolution.junction.pos,
        old_positions=tuple(junction.pos for junction in resolution.junctions),
        transitions=tuple(
            TransitionReference(arm.label, arm.transition.pos)
            for arm in resolution.arms
        ),
        candidates=tuple(
            _candidate_score(
                candidate,
                (candidate.first.transition.pos, candidate.second.transition.pos)
                in kept,
            )
            for candidate in resolution.candidates
        ),
        kind=resolution.kind,
        segments=resolution.segments,
        new_junction=resolution.new_junction,
        reason=resolution.reason,
    )


def _print_resolution(resolution: _Resolution) -> None:
    positions = ", ".join(str(junction.pos) for junction in resolution.junctions)
    if resolution.reason is not None:
        print(f"[junction] {positions}: {resolution.reason}")
        return
    if resolution.kind != "many":
        return
    print(f"[junction] {positions}: more than two accepted segments")
    for candidate in resolution.candidates:
        decision = "keep" if candidate.accepted else "drop"
        print(
            f"  {candidate.first.label}->{candidate.second.label}: "
            f"{candidate.first_error:.1f} deg / {candidate.second_error:.1f} deg "
            f"[{decision}]"
        )


def _piece_orientation(piece: _Piece, position: Pixel) -> tuple[list[Pixel], Pixel]:
    if piece.start == position:
        return list(piece.pixels), piece.end
    if piece.end == position:
        return list(reversed(piece.pixels)), piece.start
    raise ValueError(f"Piece does not touch {position}")


def _join_paths(path: list[Pixel], extension: list[Pixel]) -> None:
    window = 16
    positions = {
        pixel: index
        for index, pixel in enumerate(path[max(0, len(path) - window) : -1])
    }
    offset = max(0, len(path) - window)
    shortcuts = [
        (
            len(path) - 1 - (offset + positions[pixel]) + index,
            offset + positions[pixel],
            index,
        )
        for index, pixel in enumerate(extension[1:window], start=1)
        if pixel in positions
    ]
    if shortcuts:
        _, path_index, extension_index = max(shortcuts)
        del path[path_index + 1 :]
        path.extend(extension[extension_index + 1 :])
    else:
        path.extend(extension[1:])


def _component(
    start: Pixel, adjacency: dict[Pixel, list[int]], pieces: list[_Piece]
) -> set[Pixel]:
    seen = {start}
    pending = [start]
    while pending:
        position = pending.pop()
        for piece_index in adjacency[position]:
            piece = pieces[piece_index]
            other = piece.end if piece.start == position else piece.start
            if other not in seen:
                seen.add(other)
                pending.append(other)
    return seen


def _normalize_graph(
    graph: Graph,
    pieces: list[_Piece],
    protected: set[Pixel],
    forced_junctions: set[Pixel],
    radii: dict[Pixel, float],
) -> None:
    adjacency: dict[Pixel, list[int]] = {}
    for index, piece in enumerate(pieces):
        adjacency.setdefault(piece.start, []).append(index)
        adjacency.setdefault(piece.end, []).append(index)

    structural = protected | {
        position for position, incident in adjacency.items() if len(incident) != 2
    }
    unseen = set(adjacency)
    while unseen:
        seed = min(unseen)
        component = _component(seed, adjacency, pieces)
        unseen.difference_update(component)
        if not component.intersection(structural):
            anchor = min(component)
            structural.add(anchor)
            forced_junctions.add(anchor)

    original_nodes = dict(graph.nodes)
    nodes: dict[Pixel, Node] = {}
    junctions: list[Node] = []
    for position in sorted(structural):
        degree = len(adjacency.get(position, ()))
        original = original_nodes.get(position)
        if (
            position in forced_junctions
            or degree >= 3
            or original is not None
            and original.type is NodeType.JUNCTION
        ):
            node_type = NodeType.JUNCTION
        else:
            node_type = NodeType.ENDPOINT
        node = Node(position, radii.get(position, 0.0), node_type)
        nodes[position] = node
        if node_type is NodeType.JUNCTION:
            junctions.append(node)

    visited: set[int] = set()
    edges: list[Edge] = []
    for start_position in sorted(structural):
        for piece_index in sorted(adjacency.get(start_position, ())):
            if piece_index in visited:
                continue
            visited.add(piece_index)
            path, current = _piece_orientation(pieces[piece_index], start_position)
            previous_piece = piece_index
            while current not in structural:
                next_piece = next(
                    index for index in adjacency[current] if index != previous_piece
                )
                if next_piece in visited:
                    raise ValueError(f"Repeated provisional edge at {current}")
                visited.add(next_piece)
                extension, following = _piece_orientation(pieces[next_piece], current)
                _join_paths(path, extension)
                previous_piece = next_piece
                current = following
            edges.append(
                Edge(
                    start=nodes[start_position],
                    end=nodes[current],
                    pixels=path,
                    nodes=[nodes[start_position], nodes[current]],
                )
            )

    if len(visited) != len(pieces):
        raise ValueError("Failed to consume every provisional graph edge")
    edges.sort(
        key=lambda edge: (
            min(edge.start.pos, edge.end.pos),
            max(edge.start.pos, edge.end.pos),
            tuple(edge.pixels),
        )
    )
    graph.nodes = nodes
    graph.edges = edges
    graph.junctions = junctions
    graph.transitions = []
    graph._node_pixels = set(nodes)


def _junction_clusters(graph: Graph) -> tuple[tuple[Node, ...], ...]:
    junction_at = {junction.pos: junction for junction in graph.junctions}
    parent = {position: position for position in junction_at}

    def find(position: Pixel) -> Pixel:
        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = parent[position]
        return position

    def union(first: Pixel, second: Pixel) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if second_root < first_root:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root

    for edge in graph.edges:
        if (
            edge.start.type is NodeType.JUNCTION
            and edge.end.type is NodeType.JUNCTION
            and edge.start.pos != edge.end.pos
            and _transition_index(edge, "S") is None
            and _transition_index(edge, "E") is None
        ):
            union(edge.start.pos, edge.end.pos)

    grouped: dict[Pixel, list[Node]] = {}
    for position, junction in junction_at.items():
        grouped.setdefault(find(position), []).append(junction)
    return tuple(
        tuple(sorted(members, key=lambda node: node.pos))
        for _, members in sorted(grouped.items())
    )


def rewrite_junctions(graph: Graph) -> JunctionRewriteReport:
    original_edges = list(graph.edges)
    resolutions = tuple(
        _resolve_junction(graph, cluster) for cluster in _junction_clusters(graph)
    )
    for resolution in resolutions:
        _print_resolution(resolution)

    by_junction = {
        junction.pos: resolution
        for resolution in resolutions
        for junction in resolution.junctions
    }
    arm_for: dict[tuple[int, str], _Arm] = {}
    for resolution in resolutions:
        if not resolution.resolved:
            continue
        for arm in resolution.arms:
            arm_for[arm.edge_index, arm.side] = arm

    pieces: list[_Piece] = []
    for edge_index, edge in enumerate(original_edges):
        start_index = 0
        end_index = len(edge.pixels) - 1
        start_resolution = by_junction.get(edge.start.pos)
        end_resolution = by_junction.get(edge.end.pos)
        if (
            start_resolution is not None
            and start_resolution is end_resolution
            and start_resolution.resolved
            and (edge_index, "S") not in arm_for
            and (edge_index, "E") not in arm_for
        ):
            continue
        if edge.start.type is NodeType.JUNCTION:
            resolution = start_resolution
            if (
                resolution is not None
                and resolution.resolved
                and (edge_index, "S") in arm_for
            ):
                start_index = arm_for[edge_index, "S"].transition_index
        if edge.end.type is NodeType.JUNCTION:
            resolution = end_resolution
            if (
                resolution is not None
                and resolution.resolved
                and (edge_index, "E") in arm_for
            ):
                end_index = arm_for[edge_index, "E"].transition_index
        if start_index <= end_index:
            _add_piece(pieces, edge.pixels[start_index : end_index + 1])

    protected = {
        node.pos for node in graph.nodes.values() if node.type is NodeType.ENDPOINT
    }
    protected.update(
        junction.pos
        for resolution in resolutions
        if not resolution.resolved
        for junction in resolution.junctions
    )
    forced_junctions: set[Pixel] = set()
    radii = {position: node.radius for position, node in graph.nodes.items()}

    for resolution in resolutions:
        if not resolution.resolved:
            continue
        if resolution.new_junction is not None:
            forced_junctions.add(resolution.new_junction)
            protected.add(resolution.new_junction)
            radii[resolution.new_junction] = float(
                np.mean([arm.transition.radius for arm in resolution.arms])
            )
        for start, end in resolution.segments:
            _add_piece(pieces, _line_pixels(start, end))

    _normalize_graph(graph, pieces, protected, forced_junctions, radii)
    return JunctionRewriteReport(
        tuple(_to_report(resolution) for resolution in resolutions)
    )


__all__ = [
    "CandidateScore",
    "JunctionRewrite",
    "JunctionRewriteReport",
    "TransitionReference",
    "rewrite_junctions",
]
