from pathlib import Path

import cv2
import numpy as np

from common import debug

from .args import get_args
from .constants import IMAGE_SUFFIXES
from .contour import extract_ink_mask, read_image_in_gray_scale
from .graph import Graph, NodeType, RadiusProfile, build_graph
from .junction import JunctionRewriteReport, rewrite_junctions
from .skeleton import MedialAxis, skeletonize
from .smoothing import SmoothedEdge, smooth_edge
from .svg import draw_svg

_EDGE_GREY = (225, 225, 225)
_AXIS_GREY = (170, 170, 170)
_RAW_GREY = (190, 190, 190)
_FILTERED_GREY = (55, 55, 55)
_FITTED_GREEN = (40, 155, 40)
_JUNCTION_BLUE = (220, 90, 25)
_ENDPOINT_ORANGE = (20, 145, 220)
_TRANSITION_RED = (30, 30, 230)
_NEW_JUNCTION_GREEN = (45, 150, 45)
_FINAL_AXIS_GREY = (70, 70, 70)
_SMOOTHED_BLACK = (20, 20, 20)
_SAMPLE_PURPLE = (180, 70, 180)


def _draw_marker(
    canvas: np.ndarray,
    position: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    cv2.circle(canvas, position, radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, position, radius, color, -1, cv2.LINE_AA)


def write_debug_image(
    medial: MedialAxis,
    edges: np.ndarray,
    graph: Graph,
    report: JunctionRewriteReport,
    smoothed_edges: list[SmoothedEdge],
    output_path: Path,
) -> None:
    overlay = np.full((*medial.shape, 3), 255, dtype=np.uint8)
    overlay[edges > 0] = _EDGE_GREY
    overlay[medial.axis] = _AXIS_GREY

    for edge in graph.edges:
        pixels = np.asarray(edge.pixels, dtype=np.int32)
        cv2.polylines(overlay, [pixels], False, _FINAL_AXIS_GREY, 1, cv2.LINE_8)

    for edge in smoothed_edges:
        pixels = np.round(edge.polyline).astype(np.int32)
        cv2.polylines(overlay, [pixels], False, _SMOOTHED_BLACK, 1, cv2.LINE_AA)
        for point in np.round(edge.samples).astype(np.int32):
            cv2.circle(overlay, (int(point[0]), int(point[1])), 2, _SAMPLE_PURPLE, -1)

    for rewrite in report.junctions:
        for old_position in rewrite.old_positions:
            _draw_marker(overlay, old_position, _JUNCTION_BLUE, 5)
        for transition in rewrite.transitions:
            _draw_marker(overlay, transition.position, _TRANSITION_RED, 6)
            cv2.putText(
                overlay,
                transition.label,
                (transition.position[0] + 9, transition.position[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                _TRANSITION_RED,
                1,
                cv2.LINE_AA,
            )

    for junction in graph.junctions:
        _draw_marker(overlay, junction.pos, _NEW_JUNCTION_GREEN, 4)

    if not cv2.imwrite(str(output_path), overlay):
        raise ValueError(f"Failed to write image: {output_path}")


def _profile_points(
    profile: RadiusProfile,
    values: np.ndarray,
    left: int,
    right: int,
    top: int,
    bottom: int,
    minimum_radius: float,
    maximum_radius: float,
) -> np.ndarray:
    distance_range = max(float(profile.distances[-1]), 1.0)
    radius_range = max(maximum_radius - minimum_radius, 1.0)
    x = left + profile.distances / distance_range * (right - left)
    y = bottom - (values - minimum_radius) / radius_range * (bottom - top)
    return np.round(np.column_stack((x, y))).astype(np.int32)


def _node_color(node_type: NodeType) -> tuple[int, int, int]:
    return _JUNCTION_BLUE if node_type is NodeType.JUNCTION else _ENDPOINT_ORANGE


def _array_point(point: np.ndarray) -> tuple[int, int]:
    return int(point[0]), int(point[1])


def write_radius_debug_image(graph: Graph, output_path: Path) -> None:
    width = 1200
    header_height = 46
    row_height = 180
    row_count = max(len(graph.edges), 1)
    canvas = np.full(
        (header_height + row_height * row_count, width, 3), 255, dtype=np.uint8
    )
    cv2.putText(
        canvas,
        "raw radius",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _RAW_GREY,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "median radius",
        (150, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _FILTERED_GREY,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "selected fit",
        (310, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _FITTED_GREEN,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "transition",
        (440, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _TRANSITION_RED,
        2,
        cv2.LINE_AA,
    )

    for edge_index, edge in enumerate(graph.edges):
        profile = edge.profile
        if profile is None or len(profile.radii) == 0:
            continue
        row_top = header_height + edge_index * row_height
        plot_left = 90
        plot_right = width - 30
        plot_top = row_top + 34
        plot_bottom = row_top + row_height - 28
        finite_fit = profile.fitted[np.isfinite(profile.fitted)]
        displayed = [profile.radii, profile.filtered]
        if len(finite_fit) > 0:
            displayed.append(finite_fit)
        minimum_radius = min(float(np.min(values)) for values in displayed)
        maximum_radius = max(float(np.max(values)) for values in displayed)
        padding = max((maximum_radius - minimum_radius) * 0.12, 0.5)
        minimum_radius -= padding
        maximum_radius += padding

        cv2.line(
            canvas,
            (plot_left, plot_top),
            (plot_left, plot_bottom),
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (plot_left, plot_bottom),
            (plot_right, plot_bottom),
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )

        raw_points = _profile_points(
            profile,
            profile.radii,
            plot_left,
            plot_right,
            plot_top,
            plot_bottom,
            minimum_radius,
            maximum_radius,
        )
        filtered_points = _profile_points(
            profile,
            profile.filtered,
            plot_left,
            plot_right,
            plot_top,
            plot_bottom,
            minimum_radius,
            maximum_radius,
        )
        cv2.polylines(canvas, [raw_points], False, _RAW_GREY, 1, cv2.LINE_AA)
        cv2.polylines(canvas, [filtered_points], False, _FILTERED_GREY, 2, cv2.LINE_AA)
        if len(finite_fit) == len(profile.fitted):
            fitted_points = _profile_points(
                profile,
                profile.fitted,
                plot_left,
                plot_right,
                plot_top,
                plot_bottom,
                minimum_radius,
                maximum_radius,
            )
            cv2.polylines(canvas, [fitted_points], False, _FITTED_GREEN, 2, cv2.LINE_AA)

        _draw_marker(
            canvas, _array_point(raw_points[0]), _node_color(edge.start.type), 4
        )
        _draw_marker(
            canvas, _array_point(raw_points[-1]), _node_color(edge.end.type), 4
        )
        for transition_index in profile.transition_indices:
            point = _array_point(filtered_points[transition_index])
            cv2.line(
                canvas,
                (point[0], plot_top),
                (point[0], plot_bottom),
                _TRANSITION_RED,
                1,
                cv2.LINE_AA,
            )
            _draw_marker(canvas, point, _TRANSITION_RED, 4)
        cv2.putText(
            canvas,
            f"E{edge_index}  {edge.start.type.name} -> {edge.end.type.name}  "
            f"[{profile.decision}]",
            (plot_left, row_top + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            _FILTERED_GREY,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{maximum_radius - padding:.1f}",
            (8, plot_top + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            _FILTERED_GREY,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{minimum_radius + padding:.1f}",
            (8, plot_bottom),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            _FILTERED_GREY,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{profile.distances[-1]:.1f} px",
            (plot_right - 65, plot_bottom + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            _FILTERED_GREY,
            1,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(str(output_path), canvas):
        raise ValueError(f"Failed to write image: {output_path}")


def convert_to_skeleton(image_path: Path) -> None:
    args = get_args()
    debug.begin(image_path.name)
    with debug.timed("read image"):
        image = read_image_in_gray_scale(image_path)
    with debug.timed("extract Canny contours"):
        mask, edges = extract_ink_mask(image)
    with debug.timed("skeletonize"):
        medial_axis = skeletonize(mask)
    with debug.timed("build graph"):
        graph = build_graph(
            medial_axis,
            transition_delta=args.transition_delta,
            transition_flatness=args.transition_flatness,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        write_radius_debug_image(
            graph, args.output_dir / f"{image_path.stem}_radius_debug.png"
        )
    transition_count = len(graph.transitions)
    with debug.timed("rewrite junctions"):
        report = rewrite_junctions(graph)
    with debug.timed("sample and smooth edges"):
        smoothed_edges = [
            smooth_edge(np.asarray(edge.pixels, dtype=np.float64), args.sample_spacing)
            for edge in graph.edges
        ]
    with debug.timed("draw SVG"):
        svg = draw_svg(medial_axis.shape, smoothed_edges, args.stroke_width)
    output_path = args.output_dir / f"{image_path.stem}.svg"
    with debug.timed("write SVG"):
        output_path.write_text(svg)
    if args.debug:
        write_debug_image(
            medial_axis,
            edges,
            graph,
            report,
            smoothed_edges,
            args.output_dir / f"{image_path.stem}_debug.png",
        )
    debug.count("svg bytes", len(svg))
    debug.count("transitions", transition_count)
    debug.count("rewritten junctions", len(report.junctions))
    debug.count("sampled points", sum(len(edge.samples) for edge in smoothed_edges))
    debug.count("Bézier curves", sum(len(edge.curves) for edge in smoothed_edges))
    debug.report()


def main() -> None:
    debug.configure(get_args("debug"))

    input_dir = get_args("input_dir")
    images = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in images:
        convert_to_skeleton(image_path)
