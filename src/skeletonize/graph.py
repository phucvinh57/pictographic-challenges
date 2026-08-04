from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise
from math import cos, hypot, radians
from typing import cast

import numpy as np

from pictographic.curves import AxisPoint, BezierCurve, catmull_rom, resample
from pictographic.graph import (
    PixelPoint,
    SkeletonEdge,
    SkeletonEdgeData,
    SkeletonGraph,
    SkeletonGraphData,
    SkeletonIntersection,
    SkeletonIntersectionData,
    SkeletonNodeData,
    build_skeleton_graph,
    neighboring_pixels,
)

from .medial_axis import OverlapSpur

__all__ = [
    "SkeletonEdgeData",
    "SkeletonGraphData",
    "SkeletonIntersectionData",
    "SkeletonNodeData",
    "build_skeleton_graph",
    "neighboring_pixels",
]


@dataclass(frozen=True)
class AxisCut:
    """A bend of the axis removed so the strokes around it are left straight.

    A cut that swallowed an overlap spur already knows where its strokes meet
    and how they run into it, so it carries the spur's tip as ``focus`` and its
    unfolded ``rails`` instead of leaving both to be crossed out of tangents.
    """

    center: PixelPoint
    radius: float
    pixels: frozenset[PixelPoint]
    junction: bool
    focus: AxisPoint | None = None
    rails: tuple[tuple[AxisPoint, ...], ...] = ()


@dataclass(frozen=True)
class EdgeTangent:
    edge_id: int
    point: AxisPoint
    direction: AxisPoint


@dataclass(frozen=True)
class IntersectionTangents:
    center: PixelPoint
    radius: float
    junction: bool
    tangents: tuple[EdgeTangent, ...]
    crossings: tuple[AxisPoint, ...]
    focus: AxisPoint | None
    rails: tuple[tuple[AxisPoint, ...], ...] = ()


@dataclass(frozen=True)
class SampledGraph:
    points: tuple[AxisPoint, ...]
    segments: tuple[tuple[int, int], ...]
    foci: tuple[int, ...]
    junctions: tuple[int, ...]


def sample_axis_lines(
    edges: tuple[SkeletonEdge, ...], spacing: float
) -> tuple[tuple[AxisPoint, ...], ...]:
    """Sample every traced axis line at an even arc-length spacing."""
    return tuple(
        resample([(float(x), float(y)) for x, y in edge.pixels], spacing)
        for edge in edges
    )


def _adjacent_cut(
    point: PixelPoint, removed: dict[PixelPoint, PixelPoint]
) -> PixelPoint | None:
    """The junction whose cut left this stroke end free."""
    x, y = point
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            center = removed.get((x + dx, y + dy))
            if center is not None:
                return center
    return None


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
    """Fuse a group into one junction sitting at the mean of their foci.

    A member holding rails already knows the cap its strokes meet at, so it
    speaks for the group rather than being averaged away by a neighbour that
    only guessed at a crossing.
    """
    foci = [
        member.focus
        for member in group
        if member.rails and member.focus is not None
    ] or [member.focus for member in group if member.focus is not None]
    widest = max(group, key=lambda member: member.radius)
    return IntersectionTangents(
        center=widest.center,
        radius=widest.radius,
        junction=True,
        tangents=tuple(
            tangent for member in group for tangent in member.tangents
        ),
        crossings=tuple(
            crossing for member in group for crossing in member.crossings
        ),
        focus=_centroid(foci) if foci else None,
        rails=tuple(rail for member in group for rail in member.rails),
    )


def intersection_tangents(
    cuts: tuple[AxisCut, ...],
    edges: tuple[SkeletonEdge, ...],
    spacing: float,
    parallel_tolerance: float = 0.2,
    reach: float = 3.0,
) -> tuple[IntersectionTangents, ...]:
    """Take a tangent at every stroke end a cut left behind.

    The tangent is fitted to the pixels a stroke end spends inside its first
    sample interval, so ``spacing`` sets the arc length the fit spans. Each
    cut's tangents are extended until they cross one another; the focus is the
    centroid of those crossings, which is the single crossing of a corner and
    the triangle centre for the three of a T. Crossings between near-parallel
    tangents run off to infinity and crossings beyond ``reach`` radii belong to
    no stroke, so both are discarded.

    A cut that swallowed an overlap spur brings its own focus, the cap the
    spur points to, and keeps it: crossing the tangents there would put the
    meeting point most of an axis width off the cap the strokes really share.

    Junctions whose discs overlap are then fused, and the fused junction sits
    at the mean of their foci. A corner has only ever been one cut, so it takes
    no part in that.
    """
    removed = {pixel: cut.center for cut in cuts for pixel in cut.pixels}
    collected: dict[PixelPoint, list[EdgeTangent]] = {cut.center: [] for cut in cuts}
    for edge in edges:
        if len(edge.pixels) < 2:
            continue
        for pixels in (edge.pixels, edge.pixels[::-1]):
            run = _tail_run(pixels, spacing)
            center = _adjacent_cut(run[0], removed)
            direction = _tangent_direction(run)
            if center is None or direction is None:
                continue
            collected[center].append(
                EdgeTangent(edge.id, (float(run[0][0]), float(run[0][1])), direction)
            )

    results = []
    for cut in cuts:
        tangents = tuple(collected[cut.center])
        crossings = []
        for first, second in combinations(tangents, 2):
            crossing = _line_crossing(first, second, parallel_tolerance)
            if crossing is None:
                continue
            distance = hypot(
                crossing[0] - cut.center[0],
                crossing[1] - cut.center[1],
            )
            if distance <= reach * cut.radius:
                crossings.append(crossing)
        results.append(
            IntersectionTangents(
                center=cut.center,
                radius=cut.radius,
                junction=cut.junction,
                tangents=tangents,
                crossings=tuple(crossings),
                focus=(
                    cut.focus
                    if cut.focus is not None
                    else _centroid(crossings) if crossings else None
                ),
                rails=cut.rails,
            )
        )

    junctions = [result for result in results if result.junction]
    return tuple(
        [_merge_junctions(group) for group in _overlapping_groups(junctions)]
        + [result for result in results if not result.junction]
    )


def _claim_rail(
    point: AxisPoint, spare: list[tuple[AxisPoint, ...]]
) -> tuple[AxisPoint, ...]:
    """Take the unfolded rail that runs out to a stroke end, if one is left."""
    if not spare:
        return ()
    rail = min(
        spare, key=lambda rail: hypot(rail[0][0] - point[0], rail[0][1] - point[1])
    )
    spare.remove(rail)
    return rail


def merge_tangent_foci(
    samples: tuple[tuple[AxisPoint, ...], ...],
    junctions: tuple[IntersectionTangents, ...],
    spacing: float,
) -> SampledGraph:
    """Add each junction's focus to the sampled strokes and rejoin them there.

    Every stroke keeps its own chain of samples; the ends the cut left behind
    gain a segment back to their junction's focus, so the strokes a junction
    separated meet again at one shared node. A junction holding rails hands each
    end the one that runs out to it, so the stroke follows the centre line the
    overlap hid instead of jumping the gap straight.
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
    crossings = []
    for junction in junctions:
        if junction.focus is None:
            continue
        focus = node(junction.focus)
        foci.append(focus)
        if junction.junction:
            crossings.append(focus)
        spare = list(junction.rails)
        for tangent in junction.tangents:
            rail = _claim_rail(tangent.point, spare)
            if not rail:
                segments.append((focus, node(tangent.point)))
                continue
            walk = resample(
                [junction.focus, *reversed(rail), tangent.point], spacing
            )
            segments.extend((node(start), node(end)) for start, end in pairwise(walk))
    return SampledGraph(
        tuple(points), tuple(segments), tuple(foci), tuple(crossings)
    )


def _straight_branches(
    graph: SampledGraph,
    adjacency: dict[int, list[int]],
    max_turn: float,
) -> dict[int, dict[int, int]]:
    """Pair the branches one stroke runs straight on through at each junction.

    A junction is where strokes cross, not where they end: the stem of an H
    passes through both of its T-junctions and stays the single stroke a pen
    would draw. Two branches carry the same stroke when the axis turns by less
    than ``max_turn`` degrees between them, so a junction's branches are paired
    off straightest first and whatever is left over — the bar of the T — starts
    a stroke of its own.
    """
    limit = -cos(radians(max_turn))
    passing: dict[int, dict[int, int]] = {}
    for focus in graph.junctions:
        directions: dict[int, AxisPoint] = {}
        for neighbor in adjacency[focus]:
            step_x = graph.points[neighbor][0] - graph.points[focus][0]
            step_y = graph.points[neighbor][1] - graph.points[focus][1]
            length = hypot(step_x, step_y)
            if length:
                directions[neighbor] = (step_x / length, step_y / length)

        paired: dict[int, int] = {}
        turns = sorted(
            (
                first[1][0] * second[1][0] + first[1][1] * second[1][1],
                first[0],
                second[0],
            )
            for first, second in combinations(sorted(directions.items()), 2)
        )
        for straightness, first_branch, second_branch in turns:
            if (
                straightness > limit
                or first_branch in paired
                or second_branch in paired
            ):
                continue
            paired[first_branch] = second_branch
            paired[second_branch] = first_branch
        passing[focus] = paired
    return passing


def _graph_chains(graph: SampledGraph, max_turn: float) -> list[list[int]]:
    """Split the merged graph into the runs of points a single curve may span.

    A chain ends wherever the curve must not carry a tangent across: a stroke
    end, a focus, or any point more than two segments meet at. The focus of a
    corner joins exactly two strokes, so without breaking there the fit would
    round the very point the cut exists to keep sharp. A junction's focus is the
    exception: the branches that run straight through it are one stroke crossing
    another, and the chain carries on from one into the other rather than ending
    there.
    """
    adjacency: dict[int, list[int]] = {index: [] for index in range(len(graph.points))}
    linked: set[frozenset[int]] = set()
    for start, end in graph.segments:
        link = frozenset((start, end))
        if start == end or link in linked:
            continue
        linked.add(link)
        adjacency[start].append(end)
        adjacency[end].append(start)

    passing = _straight_branches(graph, adjacency, max_turn)
    breaks = set(graph.foci) | {
        index for index, neighbors in adjacency.items() if len(neighbors) != 2
    }
    walked: set[frozenset[int]] = set()

    def onward(current: int, previous: int) -> list[int]:
        if current not in breaks:
            return adjacency[current]
        crossed = passing.get(current, {}).get(previous)
        return [] if crossed is None else [crossed]

    def walk(start: int, following: int) -> list[int]:
        chain = [start, following]
        walked.add(frozenset((start, following)))
        previous, current = start, following
        while current != start:
            candidates = [
                neighbor
                for neighbor in onward(current, previous)
                if frozenset((current, neighbor)) not in walked
            ]
            if not candidates:
                break
            previous, current = current, candidates[0]
            walked.add(frozenset((previous, current)))
            chain.append(current)
        return chain

    chains = []
    for index in sorted(breaks):
        for neighbor in adjacency[index]:
            if passing.get(index, {}).get(neighbor) is not None:
                continue
            if frozenset((index, neighbor)) not in walked:
                chains.append(walk(index, neighbor))
    for index in range(len(graph.points)):
        for neighbor in adjacency[index]:
            if frozenset((index, neighbor)) not in walked:
                chains.append(walk(index, neighbor))
    return chains


def smooth_sampled_graph(
    graph: SampledGraph, alpha: float = 0.5, max_turn: float = 50.0
) -> tuple[tuple[BezierCurve, ...], ...]:
    """Replace the merged graph's straight segments with smooth Bezier chains.

    The samples are an arc-length walk of a thinned axis, so the polyline
    through them is faceted even where the stroke is not. Every chain between
    the graph's corners and junctions is refitted as a Catmull-Rom spline and
    returned as its cubic Bezier spans, which carries the smoothing across the
    whole image while the corners stay the sharp points they were solved for.
    A chain runs on through a junction whenever two of its branches turn by less
    than ``max_turn`` degrees into one another, so a stroke crossed by another is
    returned whole rather than as the pieces the crossing left.
    """
    return tuple(
        curves
        for chain in _graph_chains(graph, max_turn)
        if (curves := catmull_rom([graph.points[index] for index in chain], alpha))
    )


def _branch_cut(
    pixels: tuple[PixelPoint, ...],
    intersection: SkeletonIntersection,
    transitions: frozenset[PixelPoint],
    max_reach: float,
) -> tuple[PixelPoint, ...]:
    """Walk one branch out of a junction and stop on its transition point."""
    center = intersection.center
    radius = cast(float, intersection.radius)
    walked = [pixels[0]]
    for pixel in pixels[1:]:
        if hypot(pixel[0] - center[0], pixel[1] - center[1]) > max_reach * radius:
            break
        walked.append(pixel)
        if pixel in transitions:
            return tuple(walked)
    return tuple(
        pixel
        for pixel in walked
        if hypot(pixel[0] - center[0], pixel[1] - center[1]) <= radius
    )


def junction_cuts(
    graph: SkeletonGraph,
    transitions: frozenset[PixelPoint],
    spurs: Sequence[OverlapSpur] = (),
    max_reach: float = 3.0,
) -> tuple[AxisCut, ...]:
    """Claim for each junction the axis out to its branches' transition points.

    A junction's disc is pinned by the corners of the notches around it and
    stays pinned while the axis is inside their fans, so the transition point on
    a branch is where the junction ends and the stroke proper begins. Cutting
    there removes the bend the junction actually owns instead of whatever a
    circle covers. A branch with no transition point within ``max_reach`` radii
    falls back to the maximal-inscribed disc.

    A branch that is an overlap spur has no transition point to stop on because
    none of it is a stroke, so the whole of it is claimed and the junction takes
    the spur's tip and rails: the strokes meet at the cap the spur points to,
    not at a crossing of tangents an axis width away from it.
    """
    for intersection in graph.intersections:
        if intersection.radius is None:
            raise ValueError(
                "Intersection radii are unavailable; build the graph with binary"
            )

    owner = {
        pixel: index
        for index, intersection in enumerate(graph.intersections)
        for pixel in intersection.pixels
    }
    by_branch = {spur.pixels: spur for spur in spurs}
    claimed = [set(intersection.pixels) for intersection in graph.intersections]
    unfolded: list[OverlapSpur | None] = [None] * len(graph.intersections)
    for edge in graph.edges:
        for pixels in (edge.pixels, edge.pixels[::-1]):
            index = owner.get(pixels[0])
            if index is None:
                continue
            spur = by_branch.get(pixels)
            if spur is not None:
                claimed[index].update(pixels)
                unfolded[index] = spur
                continue
            claimed[index].update(
                _branch_cut(
                    pixels, graph.intersections[index], transitions, max_reach
                )
            )

    return tuple(
        AxisCut(
            center=intersection.center,
            radius=cast(float, intersection.radius),
            pixels=frozenset(pixels),
            junction=True,
            focus=None if spur is None else spur.tip,
            rails=() if spur is None else spur.rails,
        )
        for intersection, pixels, spur in zip(graph.intersections, claimed, unfolded)
    )


def corner_cuts(
    runs: Sequence[tuple[Sequence[PixelPoint], float]],
    junctions: tuple[AxisCut, ...],
) -> tuple[AxisCut, ...]:
    """Cut away each corner's fan, the axis between its two transition points.

    A corner drives the axis to bend through its whole fan, so the fan is the
    rounding the corner put there and none of it belongs to either stroke;
    removing it leaves the two straight pieces to be rejoined at the crossing
    of their tangents, which is the corner itself. A fan a junction already
    claimed is left alone, that bend belongs to the junction.
    """
    claimed = {pixel for cut in junctions for pixel in cut.pixels}
    cuts = []
    for pixels, radius in runs:
        if not pixels or any(pixel in claimed for pixel in pixels):
            continue
        cuts.append(
            AxisCut(
                center=pixels[len(pixels) // 2],
                radius=radius,
                pixels=frozenset(pixels),
                junction=False,
            )
        )
    return tuple(cuts)


def remove_axis_cuts(axis: np.ndarray, cuts: tuple[AxisCut, ...]) -> np.ndarray:
    """Remove every axis pixel a cut claimed."""
    cleaned = axis.copy()
    for cut in cuts:
        for x, y in cut.pixels:
            cleaned[y, x] = 255
    return cleaned
