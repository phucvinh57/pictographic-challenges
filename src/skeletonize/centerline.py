from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import KDTree

Point = tuple[float, float]  # (x, y)


def extract_contours(binary: np.ndarray) -> list[np.ndarray]:
    mask = (binary < 128).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1, 2).astype(np.float64) for c in contours if len(c) >= 8]


def _tangent_normals(points: np.ndarray, span: int) -> np.ndarray:
    n = len(points)
    span = min(span, n // 2) or 1
    tangent = np.roll(points, -span, axis=0) - np.roll(points, span, axis=0)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    length[length == 0] = 1.0
    return normal / length


def _sample_distance(distance: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = distance.shape
    xs = np.clip(np.rint(points[..., 0]).astype(int), 0, width - 1)
    ys = np.clip(np.rint(points[..., 1]).astype(int), 0, height - 1)
    return distance[ys, xs]


def _orient_normals_inward(
    points: np.ndarray, normals: np.ndarray, distance: np.ndarray, eps: float
) -> np.ndarray:
    pos_value = _sample_distance(distance, points + eps * normals)
    neg_value = _sample_distance(distance, points - eps * normals)
    oriented = normals.copy()
    oriented[neg_value > pos_value] *= -1.0
    return oriented


def _estimate_stroke_width(distance: np.ndarray, foreground: np.ndarray) -> float:
    dilated = cv2.dilate(distance, np.ones((3, 3), np.uint8))
    ridge = foreground & (distance >= dilated) & (distance > 0)
    if not np.any(ridge):
        return 0.0
    return 2.0 * float(np.median(distance[ridge]))


def find_centerline_points(
    binary: np.ndarray,
    stroke_width: float | None = None,
    tangent_span: int = 6,
    exclusion_span: int | None = None,
    search_radius: float | None = None,
    angle_tolerance_deg: float = 25.0,
    width_min_ratio: float = 0.9,
    width_max_ratio: float = 1.1,
    medial_tolerance: float = 0.85,
    normal_eps: float = 1.5,
    search_margin: float = 1.6,
) -> tuple[list[Point], float]:
    contours = extract_contours(binary)
    if not contours:
        return [], stroke_width or 0.0

    foreground = binary < 128
    if not np.any(foreground):
        return [], stroke_width or 0.0
    distance = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 5)

    auto_width = stroke_width is None
    if auto_width:
        stroke_width = _estimate_stroke_width(distance, foreground)
    if not stroke_width or stroke_width <= 0:
        return [], stroke_width or 0.0

    if search_radius is None:
        search_radius = stroke_width * search_margin
    if search_radius <= 0:
        return [], stroke_width

    if exclusion_span is None:
        exclusion_span = max(8, round(1.2 * stroke_width))

    all_points = np.concatenate(contours, axis=0)
    contour_id = np.concatenate([np.full(len(c), i) for i, c in enumerate(contours)])
    local_index = np.concatenate([np.arange(len(c)) for c in contours])
    contour_length = np.concatenate([np.full(len(c), len(c)) for c in contours])
    normals = np.concatenate([_tangent_normals(c, tangent_span) for c in contours])
    normals = _orient_normals_inward(all_points, normals, distance, normal_eps)

    tree = KDTree(all_points)
    cos_tolerance = np.cos(np.radians(angle_tolerance_deg))
    neighbor_lists = tree.query_ball_point(all_points, r=search_radius)

    def select_pairs(width: float) -> list[tuple[int, int, float]]:
        width_min = width * width_min_ratio
        width_max = width * width_max_ratio
        seen_pairs: set[tuple[int, int]] = set()
        pairs: list[tuple[int, int, float]] = []
        for i, neighbors in enumerate(neighbor_lists):
            best: tuple[float, float, int] | None = None
            for j in neighbors:
                if j == i:
                    continue
                if contour_id[j] == contour_id[i]:
                    diff = abs(int(local_index[j]) - int(local_index[i]))
                    diff = min(diff, int(contour_length[i]) - diff)
                    if diff < exclusion_span:
                        continue
                direction = all_points[j] - all_points[i]
                dist = float(np.linalg.norm(direction))
                if dist < width_min or dist > width_max:
                    continue
                unit = direction / dist
                if np.dot(unit, normals[i]) < cos_tolerance:
                    continue
                if np.dot(-unit, normals[j]) < cos_tolerance:
                    continue
                midpoint = (all_points[i] + all_points[j]) / 2.0
                if _sample_distance(distance, midpoint) < medial_tolerance * dist / 2.0:
                    continue
                score = abs(dist - width)
                if best is None or score < best[0] or (score == best[0] and dist < best[1]):
                    best = (score, dist, j)
            if best is None:
                continue
            _, dist, j = best
            pair_key = (min(i, j), max(i, j))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                pairs.append((i, j, dist))
        return pairs

    pairs = select_pairs(stroke_width)
    if auto_width and pairs:
        refined_width = float(np.median([d for _, _, d in pairs]))
        if abs(refined_width - stroke_width) > 0.02 * stroke_width:
            stroke_width = refined_width
            pairs = select_pairs(stroke_width)

    midpoints = [
        (
            (all_points[i, 0] + all_points[j, 0]) / 2.0,
            (all_points[i, 1] + all_points[j, 1]) / 2.0,
        )
        for i, j, _ in pairs
    ]
    return midpoints, stroke_width

def overlay_edges_midpoints(
    edges: np.ndarray,
    midpoints: list[Point],
    edge_color: tuple[int, int, int] = (0, 0, 255),
    midpoint_color: tuple[int, int, int] = (255, 0, 0),
    point_radius: int = 2,
) -> np.ndarray:
    height, width = edges.shape
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[edges < 128] = edge_color
    for x, y in midpoints:
        cv2.circle(
            canvas, (int(round(x)), int(round(y))), point_radius, midpoint_color, -1
        )
    return canvas
