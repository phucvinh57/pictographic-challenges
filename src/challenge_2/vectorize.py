"""Vectorize a 1px skeleton raster into SVG centerline paths.

The skeleton (as produced by `skeletonize.thin_zhang_suen`) is a graph of
foreground pixels, 8-connected, at most 2px wide. This module walks that
pixel graph into polylines (one per edge between endpoints/junctions, plus
one per isolated cycle or dot), simplifies each polyline with
Ramer-Douglas-Peucker, and renders the result as an SVG where every path
shares a single fixed `stroke-width`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

Point = tuple[float, float]
Pixel = tuple[int, int]  # (row, col)

_NEIGHBOR_OFFSETS = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
]


def _neighbors(pixel: Pixel, fg: set[Pixel]) -> list[Pixel]:
    """8-connected foreground neighbors, with redundant diagonals dropped.

    A staircase-shaped 1px line steps through 2x2 blocks where both
    orthogonal cells and the diagonal cell between them are foreground. Left
    as plain 8-connectivity, that diagonal is a second, redundant path
    between the same two orthogonal pixels, which inflates their degree and
    manufactures a false branch point at every step of the staircase. Since
    the two orthogonal pixels are already connected through their shared
    neighbors, the diagonal is dropped whenever both bridging cells are set.
    """
    r, c = pixel
    result = []
    for dr, dc in _NEIGHBOR_OFFSETS:
        q = (r + dr, c + dc)
        if q not in fg:
            continue
        if dr != 0 and dc != 0 and (r + dr, c) in fg and (r, c + dc) in fg:
            continue
        result.append(q)
    return result


def _chain_length(chain: list[Pixel]) -> float:
    total = 0.0
    for (r0, c0), (r1, c1) in zip(chain, chain[1:]):
        total += ((r1 - r0) ** 2 + (c1 - c0) ** 2) ** 0.5
    return total


def prune_spurs(fg: set[Pixel], min_length: float) -> set[Pixel]:
    """Remove short spurious branches left by thinning.

    Zhang-Suen (and similar) thinning routinely leaves 1-3px "hairs"
    sticking off a junction pixel. Repeatedly walk inward from every
    degree-1 pixel (leaf); if the walk reaches a real junction
    (degree >= 3) within `min_length` pixels, the leaf side is noise and
    gets removed. Leaves that terminate in another leaf (an isolated short
    stroke, not a spur off a larger shape) are left untouched.
    """
    if min_length <= 0:
        return fg

    fg = set(fg)
    changed = True
    while changed:
        changed = False
        degree = {p: len(_neighbors(p, fg)) for p in fg}
        leaves = [p for p, d in degree.items() if d == 1]
        to_remove: set[Pixel] = set()
        for leaf in leaves:
            if leaf in to_remove:
                continue
            chain = [leaf]
            prev, curr = None, leaf
            while True:
                candidates = [
                    q
                    for q in _neighbors(curr, fg)
                    if q != prev and q not in to_remove
                ]
                if not candidates:
                    break
                nxt = candidates[0]
                chain.append(nxt)
                if degree.get(nxt, 0) != 2:
                    break
                prev, curr = curr, nxt
            end = chain[-1]
            if degree.get(end, 0) >= 3 and _chain_length(chain) < min_length:
                to_remove.update(chain[:-1])
        if to_remove:
            fg -= to_remove
            changed = True

    return fg


def _merge_close_junctions(
    chains: list[list[Pixel]], degree: dict[Pixel, int], threshold: float
) -> list[list[Pixel]]:
    """Collapse clusters of junction pixels connected by very short edges.

    Near-45-degree strokes come out of Zhang-Suen thinning as a 2px-wide
    staircase rather than a single-pixel line (a known limitation of the
    algorithm for diagonals). Graphed with correct 8-connectivity, each step
    of that staircase still reads as a real branch, so a single true vertex
    ends up as a tight cluster of dozens of false junction pixels linked by
    1-3px "rungs". This unions junction pixels joined by an edge shorter
    than `threshold` and replaces each cluster with one node at its pixels'
    centroid; edges that were entirely internal to a cluster (the rungs
    themselves) are dropped.
    """
    if threshold <= 0:
        return chains

    parent: dict[Pixel, Pixel] = {}

    def find(x: Pixel) -> Pixel:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Pixel, b: Pixel) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def is_short_junction_edge(chain: list[Pixel]) -> bool:
        a, b = chain[0], chain[-1]
        return (
            degree.get(a, 0) >= 3
            and degree.get(b, 0) >= 3
            and _chain_length(chain) < threshold
        )

    for chain in chains:
        if is_short_junction_edge(chain):
            union(chain[0], chain[-1])

    groups: dict[Pixel, list[Pixel]] = {}
    for chain in chains:
        for endpoint in (chain[0], chain[-1]):
            groups.setdefault(find(endpoint), []).append(endpoint)

    centroids = {
        root: (
            sum(r for r, _ in members) / len(members),
            sum(c for _, c in members) / len(members),
        )
        for root, members in groups.items()
        if len(members) > 1
    }

    result = []
    for chain in chains:
        if is_short_junction_edge(chain):
            continue
        a, b = chain[0], chain[-1]
        new_a = centroids.get(find(a), a)
        new_b = centroids.get(find(b), b)
        result.append([new_a, *chain[1:-1], new_b])
    return result


def trace_skeleton(
    skeleton: np.ndarray, min_spur_length: float = 0.0
) -> list[list[Pixel]]:
    """Walk a binary skeleton image into a list of pixel chains.

    `skeleton` must have foreground=0 (black) / background=255 (white),
    matching `skeletonize.thin_zhang_suen`'s output. Each returned chain is a
    list of (row, col) pixels; branch points and endpoints delimit chains,
    and pixels that form a closed loop with no branch/endpoint (e.g. an "O")
    are returned as a single closed chain. `min_spur_length` removes short
    thinning artifacts before tracing; see `prune_spurs`.
    """
    fg = {(r, c) for r, c in zip(*np.nonzero(skeleton < 128))}
    fg = prune_spurs(fg, min_spur_length)
    degree = {p: len(_neighbors(p, fg)) for p in fg}
    nodes = {p for p, d in degree.items() if d != 2}

    chains: list[list[Pixel]] = []
    visited_steps: set[tuple[Pixel, Pixel]] = set()
    visited_pixels: set[Pixel] = set()

    for node in nodes:
        if degree[node] == 0:
            chains.append([node, node])
            visited_pixels.add(node)
            continue
        for nb in _neighbors(node, fg):
            if (node, nb) in visited_steps:
                continue
            chain = [node, nb]
            prev, curr = node, nb
            while curr not in nodes:
                nxt_candidates = [q for q in _neighbors(curr, fg) if q != prev]
                nxt = nxt_candidates[0]
                chain.append(nxt)
                prev, curr = curr, nxt
            for a, b in zip(chain, chain[1:]):
                visited_steps.add((a, b))
                visited_steps.add((b, a))
            visited_pixels.update(chain)
            chains.append(chain)

    chains = _merge_close_junctions(chains, degree, min_spur_length)

    # Pure cycles: no endpoints/junctions anywhere along the loop.
    remaining = fg - visited_pixels
    while remaining:
        start = next(iter(remaining))
        chain = [start]
        prev, curr = None, start
        while True:
            nxt_candidates = [q for q in _neighbors(curr, fg) if q != prev]
            nxt = nxt_candidates[0]
            chain.append(nxt)
            visited_pixels.add(curr)
            remaining.discard(curr)
            if nxt == start:
                break
            prev, curr = curr, nxt
        remaining.discard(start)
        chains.append(chain)

    return chains


def _perp_distance(p: Point, a: Point, b: Point) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    proj_x, proj_y = ax + t * dx, ay + t * dy
    return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5


def rdp(points: list[Point], epsilon: float) -> list[Point]:
    """Ramer-Douglas-Peucker polyline simplification (iterative)."""
    if len(points) < 3:
        return list(points)

    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]

    while stack:
        start, end = stack.pop()
        a, b = points[start], points[end]
        max_dist = -1.0
        max_index = -1
        for i in range(start + 1, end):
            dist = _perp_distance(points[i], a, b)
            if dist > max_dist:
                max_dist, max_index = dist, i
        if max_dist > epsilon:
            keep[max_index] = True
            stack.append((start, max_index))
            stack.append((max_index, end))

    return [p for p, k in zip(points, keep) if k]


def skeleton_to_polylines(
    skeleton: np.ndarray, epsilon: float = 1.5, min_spur_length: float = 0.0
) -> list[list[Point]]:
    """Trace + simplify a skeleton into polylines of (x, y) points."""
    polylines = []
    for chain in trace_skeleton(skeleton, min_spur_length=min_spur_length):
        points = [(float(c), float(r)) for r, c in chain]
        polylines.append(rdp(points, epsilon))
    return polylines


def polylines_to_svg(
    polylines: list[list[Point]],
    width: int,
    height: int,
    stroke_width: float = 45.0,
) -> str:
    """Render polylines as an SVG document with a single fixed stroke width."""
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">']
    for points in polylines:
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
            for i, (x, y) in enumerate(points)
        )
        lines.append(
            f'  <path d="{d}" fill="none" stroke="black" '
            f'stroke-width="{stroke_width:g}" stroke-linecap="round" '
            f'stroke-linejoin="round"/>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def skeleton_to_svg(
    skeleton: np.ndarray,
    stroke_width: float = 45.0,
    epsilon: float = 1.5,
    min_spur_length: float = 0.0,
) -> str:
    """Vectorize a skeleton raster into an SVG with a fixed stroke width."""
    height, width = skeleton.shape
    polylines = skeleton_to_polylines(
        skeleton, epsilon=epsilon, min_spur_length=min_spur_length
    )
    return polylines_to_svg(polylines, width, height, stroke_width=stroke_width)


def write_svg(svg: str, output_path: Path) -> Path:
    output_path.write_text(svg)
    return output_path
