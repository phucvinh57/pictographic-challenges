from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import KDTree

Point = tuple[float, float]
PixelPoint = tuple[int, int]


def rasterize_midpoints(
    midpoints: list[Point], shape: tuple[int, int], radius: int = 2
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for x, y in midpoints:
        cv2.circle(mask, (int(round(x)), int(round(y))), radius, 255, -1)
    return mask


def thin_mask(mask: np.ndarray) -> np.ndarray:
    return cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_GUOHALL)


def _neighbors8(point: PixelPoint, points_set: set[PixelPoint]) -> list[PixelPoint]:
    x, y = point
    return [
        (x + dx, y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx, dy) != (0, 0) and (x + dx, y + dy) in points_set
    ]


def trace_skeleton(
    mask: np.ndarray,
) -> tuple[list[list[PixelPoint]], dict[PixelPoint, int]]:
    """Walk a 1px-wide skeleton mask into ordered point chains between its
    endpoints/junctions (or, for a closed loop with no junction, all the way
    back to the start)."""
    ys, xs = np.nonzero(mask)
    points_set: set[PixelPoint] = set(zip(xs.tolist(), ys.tolist()))
    if not points_set:
        return [], {}

    degree = {p: len(_neighbors8(p, points_set)) for p in points_set}
    visited_edges: set[frozenset[PixelPoint]] = set()
    lines: list[list[PixelPoint]] = []

    def walk(p0: PixelPoint, p1: PixelPoint) -> list[PixelPoint]:
        path = [p0, p1]
        visited_edges.add(frozenset((p0, p1)))
        cur = p1
        while degree[cur] == 2:
            candidates = [
                n
                for n in _neighbors8(cur, points_set)
                if frozenset((cur, n)) not in visited_edges
            ]
            if not candidates:
                break
            nxt = candidates[0]
            visited_edges.add(frozenset((cur, nxt)))
            path.append(nxt)
            cur = nxt
            if cur == p0:
                break
        return path

    branch_points = [p for p in points_set if degree[p] != 2]
    for p in branch_points:
        for n in _neighbors8(p, points_set):
            if frozenset((p, n)) in visited_edges:
                continue
            lines.append(walk(p, n))

    for p in points_set:
        if degree[p] != 2:
            continue
        if any(frozenset((p, n)) in visited_edges for n in _neighbors8(p, points_set)):
            continue
        neighbors = _neighbors8(p, points_set)
        if not neighbors:
            continue
        lines.append(walk(p, neighbors[0]))

    return lines, degree


def _path_length(path: list[PixelPoint]) -> float:
    pts = np.asarray(path, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def prune_spurs(
    lines: list[list[PixelPoint]],
    degree: dict[PixelPoint, int],
    min_length: float,
    min_junction_gap: float,
) -> list[list[PixelPoint]]:
    """Drop thinning artifacts that aren't real strokes: short dangling
    branches hanging off a junction (spurs), and the tiny sliver edges
    connecting the handful of pixels that a single wide junction gets
    fragmented into (junction clusters). Lines with two free ends are kept
    regardless of length, since they're a genuine separate component."""
    kept = []
    for path in lines:
        if path[0] == path[-1]:
            kept.append(path)
            continue
        is_free_start = degree.get(path[0], 1) == 1
        is_free_end = degree.get(path[-1], 1) == 1
        length = _path_length(path)
        if is_free_start != is_free_end and length < min_length:
            continue
        if not is_free_start and not is_free_end and length < min_junction_gap:
            continue
        kept.append(path)
    return kept


def _outward_direction(
    path: list[PixelPoint], at_start: bool, span: int
) -> np.ndarray:
    pts = np.asarray(path, dtype=np.float64)
    span = min(span, len(pts) - 1) or 1
    vec = (pts[0] - pts[span]) if at_start else (pts[-1] - pts[-1 - span])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def connect_opposite_lines(
    lines: list[list[PixelPoint]],
    max_gap: float,
    angle_tolerance_deg: float = 40.0,
    tangent_span: int = 4,
) -> list[list[Point]]:
    """Bridge dangling line ends that face each other head-on: close in
    distance, and whose outward tangents point at one another (i.e. the same
    stroke continuing across a gap left by missing centerline points).

    An end counts as dangling only if it's the sole surviving line touching
    that point. Junction degree from the raw skeleton graph goes stale as
    soon as `prune_spurs` drops the sliver edges at a junction cluster, which
    can leave a lone remaining edge whose endpoint still looks (by its old
    degree) like part of a multi-way junction; recomputing per-point line
    counts from the current lines is what lets that lone edge be treated as
    dangling and get reconnected."""
    paths: list[list[PixelPoint] | None] = [list(line) for line in lines]
    cos_tolerance = np.cos(np.radians(angle_tolerance_deg))

    while True:
        endpoint_counts: dict[PixelPoint, int] = {}
        for path in paths:
            if path is None or path[0] == path[-1]:
                continue
            endpoint_counts[path[0]] = endpoint_counts.get(path[0], 0) + 1
            endpoint_counts[path[-1]] = endpoint_counts.get(path[-1], 0) + 1

        ends: list[tuple[int, bool, np.ndarray, np.ndarray]] = []
        for idx, path in enumerate(paths):
            if path is None or path[0] == path[-1]:
                continue
            if endpoint_counts[path[0]] == 1:
                ends.append(
                    (idx, True, np.asarray(path[0], dtype=np.float64),
                     _outward_direction(path, True, tangent_span))
                )
            if endpoint_counts[path[-1]] == 1:
                ends.append(
                    (idx, False, np.asarray(path[-1], dtype=np.float64),
                     _outward_direction(path, False, tangent_span))
                )

        if len(ends) < 2:
            break

        points = np.stack([e[2] for e in ends])
        tree = KDTree(points)
        best: tuple[float, int, int] | None = None
        for a, b in tree.query_pairs(r=max_gap):
            idx_a, start_a, point_a, dir_a = ends[a]
            idx_b, start_b, point_b, dir_b = ends[b]
            if idx_a == idx_b:
                continue
            connecting = point_b - point_a
            dist = float(np.linalg.norm(connecting))
            if dist == 0:
                continue
            unit = connecting / dist
            if np.dot(dir_a, unit) < cos_tolerance:
                continue
            if np.dot(dir_b, -unit) < cos_tolerance:
                continue
            if best is None or dist < best[0]:
                best = (dist, a, b)

        if best is None:
            break

        _, a, b = best
        idx_a, start_a, _, _ = ends[a]
        idx_b, start_b, _, _ = ends[b]
        path_a = paths[idx_a]
        path_b = paths[idx_b]
        assert path_a is not None and path_b is not None
        if start_a:
            path_a = path_a[::-1]
        if not start_b:
            path_b = path_b[::-1]
        paths[idx_a] = path_a + path_b
        paths[idx_b] = None

    return [
        [(float(x), float(y)) for x, y in path] for path in paths if path is not None
    ]


def simplify_polyline(points: list[Point], epsilon: float = 1.5) -> list[Point]:
    if len(points) < 3:
        return points
    closed = points[0] == points[-1]
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(arr, epsilon, closed)
    return [(float(x), float(y)) for x, y in simplified.reshape(-1, 2)]


def _catmull_rom_bezier_segments(
    points: list[Point], closed: bool
) -> list[tuple[Point, Point, Point, Point]]:
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return []
    if n == 2:
        p0, p1 = pts
        return [(tuple(p0), tuple(p0), tuple(p1), tuple(p1))]

    def at(i: int) -> np.ndarray:
        return pts[i % n] if closed else pts[min(max(i, 0), n - 1)]

    segments = []
    count = n if closed else n - 1
    for i in range(count):
        p0, p1, p2, p3 = at(i - 1), at(i), at(i + 1), at(i + 2)
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        segments.append((tuple(p1), tuple(c1), tuple(c2), tuple(p2)))
    return segments


def polyline_to_svg_path(points: list[Point], closed: bool = False) -> str:
    segments = _catmull_rom_bezier_segments(points, closed)
    if not segments:
        return ""
    start = segments[0][0]
    parts = [f"M {start[0]:.2f} {start[1]:.2f}"]
    for _, c1, c2, end in segments:
        parts.append(
            f"C {c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {end[0]:.2f} {end[1]:.2f}"
        )
    if closed:
        parts.append("Z")
    return " ".join(parts)


def build_svg(
    svg_paths: list[str],
    width: int,
    height: int,
    stroke_width: float,
    stroke_color: str = "black",
) -> str:
    body = "\n".join(
        f'  <path d="{d}" fill="none" stroke="{stroke_color}" '
        f'stroke-width="{stroke_width:.2f}" stroke-linecap="round" '
        f'stroke-linejoin="round" />'
        for d in svg_paths
        if d
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n{body}\n</svg>\n'
    )


def overlay_vector_paths(
    edges: np.ndarray,
    paths: list[list[Point]],
    edge_color: tuple[int, int, int] = (0, 0, 255),
    path_color: tuple[int, int, int] = (0, 160, 0),
    thickness: int = 2,
) -> np.ndarray:
    height, width = edges.shape
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[edges < 128] = edge_color
    for path in paths:
        if len(path) < 2:
            continue
        pts = np.asarray(path, dtype=np.int32).reshape(-1, 1, 2)
        closed = len(path) > 2 and path[0] == path[-1]
        cv2.polylines(canvas, [pts], closed, path_color, thickness, cv2.LINE_AA)
    return canvas


def vectorize_midpoints(
    midpoints: list[Point],
    shape: tuple[int, int],
    stroke_width: float,
    raster_radius: int = 2,
    spur_length_ratio: float = 0.5,
    junction_gap_ratio: float = 0.08,
    connect_gap_ratio: float = 2.0,
    angle_tolerance_deg: float = 40.0,
    simplify_epsilon: float = 1.5,
) -> tuple[str, list[list[Point]]]:
    """Turn the raw centerline midpoints into a smooth SVG: rasterize +
    thin into a 1px skeleton, drop thinning spurs, bridge gaps between
    opposing dangling ends, then fit a Catmull-Rom vector curve of the given
    stroke width through each resulting line."""
    if not midpoints or not stroke_width:
        height, width = shape
        return build_svg([], width, height, stroke_width or 0.0), []

    mask = rasterize_midpoints(midpoints, shape, raster_radius)
    skeleton = thin_mask(mask)
    lines, degree = trace_skeleton(skeleton)
    lines = prune_spurs(
        lines,
        degree,
        min_length=stroke_width * spur_length_ratio,
        min_junction_gap=stroke_width * junction_gap_ratio,
    )
    connected = connect_opposite_lines(
        lines,
        max_gap=stroke_width * connect_gap_ratio,
        angle_tolerance_deg=angle_tolerance_deg,
    )

    svg_paths = []
    simplified_paths = []
    for path in connected:
        if len(path) < 2:
            continue
        closed = len(path) > 2 and path[0] == path[-1]
        simplified = simplify_polyline(path, epsilon=simplify_epsilon)
        if len(simplified) < 2:
            continue
        simplified_paths.append(simplified)
        svg_paths.append(polyline_to_svg_path(simplified, closed=closed))

    height, width = shape
    svg = build_svg(svg_paths, width, height, stroke_width)
    return svg, simplified_paths
