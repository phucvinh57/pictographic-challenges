from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import atan2, cos, degrees, hypot, radians
from typing import cast

import numpy as np

from .graph import PixelPoint, SkeletonGraph, SkeletonIntersection, build_skeleton_graph

Point = tuple[float, float]


@dataclass(frozen=True)
class SharpeningParameters:
    junction_skip_factor: float = 1.35
    junction_window_factor: float = 2.0
    minimum_skip_width_factor: float = 0.35
    minimum_window_width_factor: float = 0.75
    minimum_fit_points: int = 8
    minimum_fit_span_factor: float = 0.6
    maximum_fit_residual_ratio: float = 0.20
    minimum_pair_sine: float = 0.26
    maximum_vertex_offset_factor: float = 2.0
    maximum_pair_spread_factor: float = 0.75
    junction_pairing_maximum_turn_degrees: float = 40.0
    junction_sharp_minimum_angle_degrees: float = 30.0
    junction_extension_factor: float = 3.0
    junction_curvature_deviation_factor: float = 0.5
    junction_extension_samples: int = 24


DEFAULT_SHARPENING_PARAMETERS = SharpeningParameters()


@dataclass(frozen=True)
class LineFit:
    origin: Point
    direction: Point
    residual_ratio: float


@dataclass(frozen=True)
class BranchStub:
    """A branch cut back to the rim of its junction's disc, plus its extension."""

    key: tuple[int, bool]
    cut_index: int
    point: Point
    direction: Point
    fit: LineFit
    extension: tuple[Point, ...]


@dataclass(frozen=True)
class Polyline:
    points: tuple[Point, ...]
    start_node: PixelPoint | None
    end_node: PixelPoint | None
    sharp_indices: frozenset[int]
    closed: bool


@dataclass(frozen=True)
class SharpenedGeometry:
    width: int
    height: int
    polylines: tuple[Polyline, ...]
    junction_vertices: tuple[Point, ...]


def _arclengths(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points))
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([np.zeros(1), np.cumsum(steps)])


def _polyline_length(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 0.0
    array = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def _fit_line(points: np.ndarray) -> LineFit | None:
    """Total-least-squares line fit; handles vertical branches exactly."""
    if len(points) < 2:
        return None
    center = points.mean(axis=0)
    _, singular, right = np.linalg.svd(points - center, full_matrices=False)
    if singular[0] <= 1e-9:
        return None
    residual_ratio = float(singular[1] / singular[0])
    direction = right[0]
    return LineFit(
        origin=(float(center[0]), float(center[1])),
        direction=(float(direction[0]), float(direction[1])),
        residual_ratio=residual_ratio,
    )


def _intersect_lines(first: LineFit, second: LineFit, minimum_sine: float) -> Point | None:
    d1 = np.array(first.direction)
    d2 = np.array(second.direction)
    denominator = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denominator) < minimum_sine:
        return None
    m1 = np.array(first.origin)
    m2 = np.array(second.origin)
    offset = m2 - m1
    t = (offset[0] * d2[1] - offset[1] * d2[0]) / denominator
    vertex = m1 + t * d1
    return (float(vertex[0]), float(vertex[1]))


def _least_squares_vertex(fits: Sequence[LineFit]) -> Point | None:
    """The point minimizing summed squared perpendicular distance to every fit."""
    if len(fits) < 2:
        return None
    accumulated_matrix = np.zeros((2, 2))
    accumulated_vector = np.zeros(2)
    for fit in fits:
        direction = np.array(fit.direction)
        origin = np.array(fit.origin)
        projector = np.eye(2) - np.outer(direction, direction)
        accumulated_matrix += projector
        accumulated_vector += projector @ origin
    trace = float(np.trace(accumulated_matrix))
    determinant = float(np.linalg.det(accumulated_matrix))
    if abs(determinant) < 1e-9 * max(trace, 1.0) ** 2:
        return None
    vertex = np.linalg.solve(accumulated_matrix, accumulated_vector)
    return (float(vertex[0]), float(vertex[1]))


def _segment_intersection(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> float | None:
    r = a1 - a0
    s = b1 - b0
    denominator = r[0] * s[1] - r[1] * s[0]
    if abs(denominator) < 1e-12:
        return None
    offset = b0 - a0
    t = (offset[0] * s[1] - offset[1] * s[0]) / denominator
    u = (offset[0] * r[1] - offset[1] * r[0]) / denominator
    if not (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0):
        return None
    return float(t)


def _intersect_polylines(
    first: Sequence[Point], second: Sequence[Point]
) -> tuple[int, float, Point] | None:
    """The crossing of `first` with `second` that comes earliest along `first`."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    for i in range(len(a) - 1):
        for j in range(len(b) - 1):
            t = _segment_intersection(a[i], a[i + 1], b[j], b[j + 1])
            if t is None:
                continue
            point = a[i] + t * (a[i + 1] - a[i])
            return i, t, (float(point[0]), float(point[1]))
    return None


def _circle_exit_index(oriented: np.ndarray, center: Point, radius: float) -> int:
    """First index outside the junction disc; a branch dipping back in is not re-cut."""
    distances = np.hypot(oriented[:, 0] - center[0], oriented[:, 1] - center[1])
    outside = np.nonzero(distances >= radius)[0]
    return int(outside[0]) if len(outside) else len(oriented) - 1


def _fit_extension(
    window: np.ndarray,
    point: Point,
    direction: Point,
    stroke_width: float,
    length: float,
    parameters: SharpeningParameters,
) -> tuple[Point, ...]:
    """Continue a branch into the junction along its own curvature.

    The window is fitted with `y = a*x^2 + b*x` in a frame whose origin is the
    stub and whose x-axis points into the junction, so the curve passes through
    the stub exactly and degenerates to a straight ray when the branch is
    straight. The sampled deviation is then rescaled to stay within
    `junction_curvature_deviation_factor * stroke_width`, which is what stops a
    short, noisy window from swinging the extrapolation across the junction.
    """
    origin = np.array(point, dtype=np.float64)
    axis = np.array(direction, dtype=np.float64)
    normal = np.array([-axis[1], axis[0]])
    samples = np.linspace(0.0, length, parameters.junction_extension_samples)

    deviations = np.zeros_like(samples)
    if len(window) >= parameters.minimum_fit_points:
        offsets = window - origin
        x = offsets @ axis
        design = np.stack([x * x, x], axis=1)
        solution = np.linalg.lstsq(design, offsets @ normal, rcond=None)[0]
        deviations = solution[0] * samples**2 + solution[1] * samples
        limit = parameters.junction_curvature_deviation_factor * stroke_width
        peak = float(np.abs(deviations).max())
        if peak > limit:
            deviations = deviations * (limit / peak)

    curve = origin + np.outer(samples, axis) + np.outer(deviations, normal)
    return tuple((float(x), float(y)) for x, y in curve)


def _extension_fill(stub: BranchStub, target: Point) -> tuple[Point, ...]:
    """The stub's extension samples strictly between it and `target`, target first."""
    origin = np.array(stub.point)
    axis = np.array(stub.direction)
    reach = float(np.dot(np.array(target) - origin, axis))
    interior = [
        sample
        for sample in stub.extension[1:]
        if 0.0 < float(np.dot(np.array(sample) - origin, axis)) < reach
    ]
    return tuple(reversed(interior))


def build_polylines(graph: SkeletonGraph) -> tuple[Polyline, ...]:
    """Turn traced graph edges into polylines, tagging junction-attached ends."""
    junction_of: dict[PixelPoint, int] = {}
    for index, intersection in enumerate(graph.intersections):
        for pixel in intersection.pixels:
            junction_of[pixel] = index

    polylines = []
    for edge in graph.edges:
        start, end = edge.pixels[0], edge.pixels[-1]
        start_junction = junction_of.get(start)
        end_junction = junction_of.get(end)
        if start != end and start_junction is not None and start_junction == end_junction:
            continue
        points = tuple((float(x), float(y)) for x, y in edge.pixels)
        is_loop = start == end and len(edge.pixels) > 2
        closed = is_loop and start_junction is None
        polylines.append(
            Polyline(
                points=points,
                start_node=None if closed else start,
                end_node=None if closed else end,
                sharp_indices=frozenset(),
                closed=closed,
            )
        )
    return tuple(polylines)


def prune_short_branches(
    polylines: Sequence[Polyline], minimum_length: float
) -> tuple[Polyline, ...]:
    endpoint_counts: dict[PixelPoint, int] = {}
    for polyline in polylines:
        if polyline.closed:
            continue
        if polyline.start_node is not None:
            endpoint_counts[polyline.start_node] = endpoint_counts.get(polyline.start_node, 0) + 1
        if polyline.end_node is not None:
            endpoint_counts[polyline.end_node] = endpoint_counts.get(polyline.end_node, 0) + 1

    kept = []
    for polyline in polylines:
        if polyline.closed:
            kept.append(polyline)
            continue
        free_start = polyline.start_node is None or endpoint_counts.get(polyline.start_node, 0) == 1
        free_end = polyline.end_node is None or endpoint_counts.get(polyline.end_node, 0) == 1
        if free_start != free_end and _polyline_length(polyline.points) < minimum_length:
            continue
        kept.append(polyline)
    return tuple(kept)


@dataclass(frozen=True)
class _PairFill:
    """Geometry closing the gap between two opposite stubs."""

    vertex: Point
    first_head: tuple[Point, ...]
    second_head: tuple[Point, ...]
    sharp: bool
    chain: tuple[Point, ...]


def _hermite_bridge(
    first: BranchStub, second: BranchStub, samples: int
) -> tuple[Point, ...]:
    start = np.array(first.point)
    end = np.array(second.point)
    tangent_scale = float(np.linalg.norm(end - start))
    start_tangent = np.array(first.direction) * tangent_scale
    end_tangent = -np.array(second.direction) * tangent_scale

    t = np.linspace(0.0, 1.0, samples)
    curve = (
        np.outer(2 * t**3 - 3 * t**2 + 1, start)
        + np.outer(t**3 - 2 * t**2 + t, start_tangent)
        + np.outer(-2 * t**3 + 3 * t**2, end)
        + np.outer(t**3 - t**2, end_tangent)
    )
    return tuple((float(x), float(y)) for x, y in curve)


def _resolve_pair(
    first: BranchStub,
    second: BranchStub,
    center: Point,
    scale: float,
    parameters: SharpeningParameters,
) -> _PairFill | None:
    """Close the gap between two opposite stubs.

    A stroke running through the junction is bridged smoothly rather than met
    at a shared vertex: two near-collinear fits are almost always laterally
    offset, and any single point on neither line kinks both branches by half
    that offset. Only a pair that genuinely turns is met at the crossing of
    the two extensions, where a corner is what the ink actually does.
    """
    incoming = np.array(first.direction)
    outgoing = -np.array(second.direction)
    cross = abs(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
    turn = degrees(atan2(cross, float(np.dot(incoming, outgoing))))

    if turn < parameters.junction_sharp_minimum_angle_degrees:
        bridge = _hermite_bridge(first, second, parameters.junction_extension_samples)
        middle = len(bridge) // 2
        vertex = bridge[middle]
        if hypot(vertex[0] - center[0], vertex[1] - center[1]) > parameters.maximum_vertex_offset_factor * scale:
            return None
        return _PairFill(
            vertex=vertex,
            first_head=tuple(reversed(bridge[1 : middle + 1])),
            second_head=bridge[middle:-1],
            sharp=False,
            chain=bridge,
        )

    hit = _intersect_polylines(first.extension, second.extension)
    vertex = (
        hit[2]
        if hit is not None
        else _intersect_lines(first.fit, second.fit, parameters.minimum_pair_sine)
    )
    if vertex is None:
        return None
    if hypot(vertex[0] - center[0], vertex[1] - center[1]) > parameters.maximum_vertex_offset_factor * scale:
        return None

    first_head = (vertex, *_extension_fill(first, vertex))
    second_head = (vertex, *_extension_fill(second, vertex))
    chain = (
        (first.point,)
        + tuple(reversed(first_head))
        + second_head[1:]
        + (second.point,)
    )
    return _PairFill(
        vertex=vertex,
        first_head=first_head,
        second_head=second_head,
        sharp=True,
        chain=chain,
    )


def _first_crossing(
    extension: Sequence[Point], others: Sequence[Sequence[Point]]
) -> Point | None:
    best: tuple[int, float, Point] | None = None
    for other in others:
        hit = _intersect_polylines(extension, other)
        if hit is not None and (best is None or hit[:2] < best[:2]):
            best = hit
    return best[2] if best is not None else None


def _fallback_vertex(
    stubs: Sequence[BranchStub],
    center: Point,
    scale: float,
    parameters: SharpeningParameters,
) -> Point | None:
    """One shared least-squares vertex, for junctions with no opposite pair."""
    fits = [stub.fit for stub in stubs]
    maximum_sine = 0.0
    for first in range(len(fits)):
        for second in range(first + 1, len(fits)):
            d1 = np.array(fits[first].direction)
            d2 = np.array(fits[second].direction)
            maximum_sine = max(maximum_sine, abs(d1[0] * d2[1] - d1[1] * d2[0]))
    if maximum_sine < parameters.minimum_pair_sine:
        return None

    vertex = _least_squares_vertex(fits)
    if vertex is None:
        return None
    if hypot(vertex[0] - center[0], vertex[1] - center[1]) > parameters.maximum_vertex_offset_factor * scale:
        return None

    for first in range(len(fits)):
        for second in range(first + 1, len(fits)):
            pairwise = _intersect_lines(fits[first], fits[second], parameters.minimum_pair_sine)
            if pairwise is None:
                continue
            if hypot(pairwise[0] - vertex[0], pairwise[1] - vertex[1]) > parameters.maximum_pair_spread_factor * scale:
                return None
    return vertex


@dataclass(frozen=True)
class _SpliceHead:
    """Replacement geometry for one end of a polyline, junction-side first."""

    points: tuple[Point, ...]
    cut: int
    sharp: bool
    node: PixelPoint


def _splice_ends(
    polyline: Polyline, start: _SpliceHead | None, end: _SpliceHead | None
) -> Polyline:
    """Replace both ends of a polyline at once.

    A branch that is attached to one junction at *both* ends — the loop of a 6
    or a 3 — is resolved as two independent stubs, so its two replacements have
    to be applied together; splicing them one at a time would rebuild the second
    from a snapshot taken before the first.
    """
    count = len(polyline.points)
    keep_low = start.cut if start is not None else 0
    keep_high = count - end.cut if end is not None else count
    if keep_high <= keep_low:
        return polyline

    head = list(start.points) if start is not None else []
    tail = list(reversed(end.points)) if end is not None else []
    points = head + list(polyline.points[keep_low:keep_high]) + tail

    offset = len(head) - keep_low
    sharp = {i + offset for i in polyline.sharp_indices if keep_low <= i < keep_high}
    if start is not None and start.sharp:
        sharp.add(0)
    if end is not None and end.sharp:
        sharp.add(len(points) - 1)

    return replace(
        polyline,
        points=tuple(points),
        sharp_indices=frozenset(sharp),
        start_node=start.node if start is not None else polyline.start_node,
        end_node=end.node if end is not None else polyline.end_node,
    )


def sharpen_junctions(
    polylines: Sequence[Polyline],
    intersections: Sequence[SkeletonIntersection],
    stroke_width: float,
    parameters: SharpeningParameters,
) -> tuple[tuple[Polyline, ...], tuple[Point, ...]]:
    """Fill each junction by pairing opposite branches and extending the rest.

    Branches are cut where they leave the junction's inscribed disc — the
    polyline analogue of `graph.remove_intersection_neighborhoods` — and every
    stub is continued into the emptied disc along its own fitted curve.
    Near-collinear stubs pair into a stroke that passes through the junction;
    each remaining stub is then extended until it cuts that through-stroke.
    Collapsing every branch onto one shared vertex instead bends a T's bar and
    turns an X into a triangle, so that survives only as the fallback for
    junctions with no opposite pair at all.

    Every rejection guard is a no-op on the affected branches, so the worst
    case reproduces the unsharpened input.
    """
    for intersection in intersections:
        if intersection.radius is None:
            raise ValueError(
                "Intersection radii are unavailable; build the graph with binary"
            )

    junction_of_radius: dict[PixelPoint, float] = {}
    for intersection in intersections:
        for pixel in intersection.pixels:
            junction_of_radius[pixel] = cast(float, intersection.radius)

    state = {index: polyline for index, polyline in enumerate(polylines)}
    vertices: list[Point] = []
    pairing_threshold = -cos(radians(parameters.junction_pairing_maximum_turn_degrees))

    # Which branches continue into each other is decided here, by geometry; the
    # original junction pixels cannot express it, since every branch of a
    # crossing ends on the same blob. Each resolution therefore re-labels the
    # ends it touches: paired branches share one synthetic node so
    # merge_degree_two_nodes stitches exactly them, everything else gets a node
    # of its own so it stitches to nothing. Without this an X is rebuilt as two
    # U-turns whenever the blob's pixels happen to pair up the other way.
    issued_nodes = 0

    def next_node() -> PixelPoint:
        nonlocal issued_nodes
        issued_nodes += 1
        return (-1, -issued_nodes)

    for intersection in intersections:
        radius = cast(float, intersection.radius)
        junction_pixels = set(intersection.pixels)
        center = (float(intersection.center[0]), float(intersection.center[1]))
        scale = max(radius, 0.5 * stroke_width)
        maximum_offset = parameters.maximum_vertex_offset_factor * scale
        cut_radius = max(
            parameters.junction_skip_factor * radius,
            parameters.minimum_skip_width_factor * stroke_width,
        )
        span = max(
            parameters.junction_window_factor * radius,
            parameters.minimum_window_width_factor * stroke_width,
        )

        incident: list[tuple[int, bool]] = []
        for index, polyline in state.items():
            if polyline.closed:
                continue
            if polyline.start_node in junction_pixels:
                incident.append((index, True))
            if polyline.end_node in junction_pixels:
                incident.append((index, False))
        if len(incident) < 2:
            continue

        branch_oriented: dict[tuple[int, bool], np.ndarray] = {}
        stubs: list[BranchStub] = []

        for key in incident:
            index, attached_at_start = key
            polyline = state[index]
            points = np.asarray(polyline.points, dtype=np.float64)
            oriented = points if attached_at_start else points[::-1]
            lengths = _arclengths(oriented)
            cut_index = max(
                1,
                min(_circle_exit_index(oriented, center, cut_radius), len(oriented) - 1),
            )
            branch_oriented[key] = oriented

            other_node = polyline.end_node if attached_at_start else polyline.start_node
            other_radius = junction_of_radius.get(other_node, 0.0) if other_node is not None else 0.0
            limit = float(lengths[-1]) - parameters.junction_skip_factor * other_radius
            cut_length = float(lengths[cut_index])
            window_end = min(cut_length + span, limit)
            if window_end <= cut_length:
                continue

            end_position = int(np.searchsorted(lengths, window_end))
            window = oriented[cut_index:end_position]
            if len(window) < parameters.minimum_fit_points:
                continue
            if lengths[end_position - 1] - cut_length < parameters.minimum_fit_span_factor * stroke_width:
                continue
            candidate_fit = _robust_fit_line(window, stroke_width)
            if (
                candidate_fit is None
                or candidate_fit.residual_ratio > parameters.maximum_fit_residual_ratio
            ):
                continue

            direction = np.array(candidate_fit.direction)
            origin = np.array(candidate_fit.origin)
            if np.dot(direction, origin - np.array(center)) < 0:
                direction = -direction
            fit = LineFit(
                origin=(float(origin[0]), float(origin[1])),
                direction=(float(direction[0]), float(direction[1])),
                residual_ratio=candidate_fit.residual_ratio,
            )
            inward = (-fit.direction[0], -fit.direction[1])
            stub_point = (float(oriented[cut_index][0]), float(oriented[cut_index][1]))
            stubs.append(
                BranchStub(
                    key=key,
                    cut_index=cut_index,
                    point=stub_point,
                    direction=inward,
                    fit=fit,
                    extension=_fit_extension(
                        window,
                        stub_point,
                        inward,
                        stroke_width,
                        parameters.junction_extension_factor * scale,
                        parameters,
                    ),
                )
            )

        if len(stubs) < 2:
            continue

        pair_candidates: list[tuple[float, int, int]] = []
        for first in range(len(stubs)):
            for second in range(first + 1, len(stubs)):
                opposition = float(
                    np.dot(
                        np.array(stubs[first].direction),
                        np.array(stubs[second].direction),
                    )
                )
                if opposition <= pairing_threshold:
                    pair_candidates.append((opposition, first, second))
        pair_candidates.sort()

        heads: dict[tuple[int, bool], _SpliceHead] = {}
        resolved_vertices: list[Point] = []
        chains: list[tuple[Point, ...]] = []
        paired: set[int] = set()

        for _opposition, first, second in pair_candidates:
            if first in paired or second in paired:
                continue
            fill = _resolve_pair(stubs[first], stubs[second], center, scale, parameters)
            if fill is None:
                continue
            paired.update((first, second))
            chains.append(fill.chain)
            resolved_vertices.append(fill.vertex)
            shared_node = next_node()
            for stub, head in (
                (stubs[first], fill.first_head),
                (stubs[second], fill.second_head),
            ):
                heads[stub.key] = _SpliceHead(
                    points=head, cut=stub.cut_index, sharp=fill.sharp, node=shared_node
                )

        unpaired = [position for position in range(len(stubs)) if position not in paired]
        for position in unpaired:
            stub = stubs[position]
            vertex = _first_crossing(stub.extension, chains)
            if vertex is None:
                vertex = _first_crossing(
                    stub.extension,
                    [stubs[other].extension for other in unpaired if other != position],
                )
            if vertex is None:
                continue
            if hypot(vertex[0] - center[0], vertex[1] - center[1]) > maximum_offset:
                continue
            resolved_vertices.append(vertex)
            heads[stub.key] = _SpliceHead(
                points=(vertex, *_extension_fill(stub, vertex)),
                cut=stub.cut_index,
                sharp=True,
                node=next_node(),
            )

        if not heads:
            fallback = _fallback_vertex(stubs, center, scale, parameters)
            if fallback is None:
                continue
            resolved_vertices.append(fallback)
            for stub in stubs:
                heads[stub.key] = _SpliceHead(
                    points=(fallback, *_extension_fill(stub, fallback)),
                    cut=stub.cut_index,
                    sharp=True,
                    node=next_node(),
                )

        # Branches too short or too noisy to fit keep their shape; only their
        # tip is pulled onto the nearest vertex so the junction stays joined.
        for key in incident:
            if key in heads:
                continue
            tip = branch_oriented[key][0]
            nearest = min(
                resolved_vertices,
                key=lambda vertex: hypot(vertex[0] - tip[0], vertex[1] - tip[1]),
            )
            if hypot(nearest[0] - tip[0], nearest[1] - tip[1]) > maximum_offset:
                continue
            heads[key] = _SpliceHead(
                points=(nearest,), cut=1, sharp=True, node=next_node()
            )

        vertices.extend(resolved_vertices)

        for index in sorted({key[0] for key in heads}):
            state[index] = _splice_ends(
                state[index], heads.get((index, True)), heads.get((index, False))
            )

    return tuple(state[i] for i in sorted(state)), tuple(vertices)


def _perpendicular_distance(point: np.ndarray, fit: LineFit) -> float:
    origin = np.array(fit.origin)
    direction = np.array(fit.direction)
    offset = point - origin
    return abs(offset[0] * direction[1] - offset[1] * direction[0])


def _robust_fit_line(points: np.ndarray, stroke_width: float) -> LineFit | None:
    """Fit a line while excluding a near-junction outlier cluster (e.g. the short
    near-vertical bridge the medial axis grows at a reflex/concave corner).

    A window can be roughly half contaminated, which defeats a median-based
    outlier test (the median itself shifts toward the contamination). Instead,
    fit a reference line from the window's far half only — the half furthest
    from the junction, which is never part of that bridge — and drop points
    that deviate from that trusted reference before the real refit.
    """
    if len(points) < 2:
        return None
    reference_count = max(2, len(points) // 2)
    reference_fit = _fit_line(points[-reference_count:])
    if reference_fit is None:
        return _fit_line(points)
    tolerance = 0.1 * stroke_width
    deviations = np.array([_perpendicular_distance(point, reference_fit) for point in points])
    inliers = points[deviations <= tolerance]
    if len(inliers) < 2:
        return reference_fit
    refit = _fit_line(inliers)
    return refit if refit is not None else reference_fit


def _merge_two_polylines(first: Polyline, node: PixelPoint, second: Polyline) -> Polyline | None:
    if first.start_node == node:
        first_points = list(reversed(first.points))
        first_sharp = {len(first.points) - 1 - i for i in first.sharp_indices}
        first_far_node = first.end_node
    else:
        first_points = list(first.points)
        first_sharp = set(first.sharp_indices)
        first_far_node = first.start_node

    if second.start_node == node:
        second_points = list(second.points)
        second_sharp = set(second.sharp_indices)
        second_far_node = second.end_node
    else:
        second_points = list(reversed(second.points))
        second_sharp = {len(second.points) - 1 - i for i in second.sharp_indices}
        second_far_node = second.start_node

    join_index = len(first_points) - 1
    combined_points = first_points + second_points[1:]
    combined_sharp = set(first_sharp)
    if join_index in first_sharp or 0 in second_sharp:
        combined_sharp.add(join_index)
    combined_sharp |= {join_index + i for i in second_sharp if i != 0}

    closed = first_far_node is not None and first_far_node == second_far_node

    return Polyline(
        points=tuple(combined_points),
        start_node=None if closed else first_far_node,
        end_node=None if closed else second_far_node,
        sharp_indices=frozenset(combined_sharp),
        closed=closed,
    )


def merge_degree_two_nodes(polylines: Sequence[Polyline]) -> tuple[Polyline, ...]:
    """Stitch polylines that meet at an unsharpened or already-sharpened degree-2 node."""
    remaining = list(polylines)
    changed = True
    while changed:
        changed = False
        node_members: dict[PixelPoint, list[int]] = {}
        for index, polyline in enumerate(remaining):
            if polyline.closed:
                continue
            for node in (polyline.start_node, polyline.end_node):
                if node is not None:
                    node_members.setdefault(node, []).append(index)

        for node, members in node_members.items():
            if len(members) != 2 or members[0] == members[1]:
                continue
            first_index, second_index = members
            merged = _merge_two_polylines(remaining[first_index], node, remaining[second_index])
            if merged is None:
                continue
            remaining = [
                polyline
                for index, polyline in enumerate(remaining)
                if index != first_index and index != second_index
            ] + [merged]
            changed = True
            break

    return tuple(remaining)


def _merge_close_intersections(
    intersections: Sequence[SkeletonIntersection], polylines: Sequence[Polyline]
) -> tuple[tuple[SkeletonIntersection, ...], tuple[Polyline, ...]]:
    """Fold junction components linked by a short, unfittable bridge into one.

    A thick self-crossing stroke often thins into two separate degree-3
    pixel clusters a short distance apart rather than one clean point. If the
    connecting branch is shorter than the two junctions' own inscribed radii
    combined, they almost certainly are the same physical crossing — fitting
    each side to its own independent vertex otherwise leaves a visible spike
    where the short, still-rounded bridge jumps between the two.
    """
    node_to_intersection: dict[PixelPoint, int] = {}
    for index, intersection in enumerate(intersections):
        for pixel in intersection.pixels:
            node_to_intersection[pixel] = index

    parent = list(range(len(intersections)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    bridge_indices: set[int] = set()
    for polyline_index, polyline in enumerate(polylines):
        if polyline.closed or polyline.start_node is None or polyline.end_node is None:
            continue
        start_intersection = node_to_intersection.get(polyline.start_node)
        end_intersection = node_to_intersection.get(polyline.end_node)
        if (
            start_intersection is None
            or end_intersection is None
            or start_intersection == end_intersection
        ):
            continue
        radius_a = intersections[start_intersection].radius or 0.0
        radius_b = intersections[end_intersection].radius or 0.0
        if _polyline_length(polyline.points) < radius_a + radius_b:
            union(start_intersection, end_intersection)
            bridge_indices.add(polyline_index)

    groups: dict[int, list[int]] = {}
    for index in range(len(intersections)):
        groups.setdefault(find(index), []).append(index)

    merged = []
    for members in groups.values():
        pixels = tuple(
            sorted(
                {pixel for member in members for pixel in intersections[member].pixels},
                key=lambda point: (point[1], point[0]),
            )
        )
        centers = np.array([intersections[member].center for member in members], dtype=np.float64)
        center = (round(float(centers[:, 0].mean())), round(float(centers[:, 1].mean())))
        radius = max(intersections[member].radius or 0.0 for member in members)
        merged.append(SkeletonIntersection(center=center, pixels=pixels, radius=radius))

    remaining_polylines = tuple(
        polyline for index, polyline in enumerate(polylines) if index not in bridge_indices
    )
    return tuple(merged), remaining_polylines


def sharpen_medial_axis(
    axis: np.ndarray,
    binary: np.ndarray,
    stroke_width: float,
    parameters: SharpeningParameters = DEFAULT_SHARPENING_PARAMETERS,
    graph: SkeletonGraph | None = None,
) -> SharpenedGeometry:
    if graph is None:
        graph = build_skeleton_graph(axis, binary)

    polylines = build_polylines(graph)
    polylines = prune_short_branches(polylines, minimum_length=stroke_width * 0.5)
    intersections, polylines = _merge_close_intersections(graph.intersections, polylines)

    polylines, junction_vertices = sharpen_junctions(polylines, intersections, stroke_width, parameters)
    polylines = merge_degree_two_nodes(polylines)

    return SharpenedGeometry(
        width=graph.width,
        height=graph.height,
        polylines=polylines,
        junction_vertices=junction_vertices,
    )
