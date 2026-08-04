from __future__ import annotations

import argparse
from math import isfinite
from pathlib import Path
from random import randint

import cv2
import numpy as np

from pictographic.curves import (
    AxisPoint,
    BezierCurve,
    flatten_curves,
    smooth_closed_contour,
)
from pictographic.svg import bezier_svg, filled_bezier_svg

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_ANGLE_THRESHOLD = 80
DEFAULT_SMOOTH_TOLERANCE = 1.5
EDGE_LOW_THRESHOLD = 50.0
EDGE_HIGH_THRESHOLD = 150.0


def binarize(gray: np.ndarray, threshold: int | None) -> np.ndarray:
    threshold_type = cv2.THRESH_BINARY
    if threshold is None:
        threshold = 0
        threshold_type += cv2.THRESH_OTSU
    _, binary = cv2.threshold(gray, threshold, 255, threshold_type)
    return binary


def extract_edges(binary: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(
        cv2.Canny(binary, EDGE_LOW_THRESHOLD, EDGE_HIGH_THRESHOLD)
    )


def random_line_colors(count: int) -> tuple[str, ...]:
    colors = []
    used = set()
    while len(colors) < count:
        red, green, blue = (randint(32, 223) for _ in range(3))
        if max(red, green, blue) - min(red, green, blue) < 64:
            continue
        color = f"#{red:02x}{green:02x}{blue:02x}"
        if color in used:
            continue
        used.add(color)
        colors.append(color)
    return tuple(colors)


def _bgr(color: str) -> tuple[int, int, int]:
    return int(color[5:7], 16), int(color[3:5], 16), int(color[1:3], 16)


def draw_curves(
    shape: tuple[int, int],
    contours: tuple[tuple[BezierCurve, ...], ...],
    colors: tuple[str, ...] | None = None,
) -> np.ndarray:
    image = np.full((*shape, 3), 255, dtype=np.uint8) if colors else np.full(
        shape, 255, dtype=np.uint8
    )
    for index, curves in enumerate(contours):
        flattened = flatten_curves(curves)
        if len(flattened) < 2:
            continue
        points = np.rint(np.asarray(flattened) * 256).astype(np.int32)
        color = _bgr(colors[index]) if colors else 0
        cv2.polylines(
            image,
            [points],
            False,
            color,
            1,
            cv2.LINE_AA,
            shift=8,
        )
    return image


def _inside(mask: np.ndarray, point: tuple[int, int]) -> bool:
    x, y = point
    return bool(
        0 <= y < mask.shape[0]
        and 0 <= x < mask.shape[1]
        and mask[y, x]
    )


def _ahead(point: tuple[int, int], direction: int) -> tuple[int, int]:
    x, y = point
    dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[direction]
    return x + dx, y + dy


def _side_pixels(
    point: tuple[int, int], direction: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    x, y = point
    first, second = (
        ((-1, -1), (0, -1)),
        ((0, 0), (0, -1)),
        ((-1, 0), (0, 0)),
        ((-1, 0), (-1, -1)),
    )[direction]
    return (x + first[0], y + first[1]), (x + second[0], y + second[1])


def _boundary_start(mask: np.ndarray) -> tuple[int, int] | None:
    padded = np.pad(mask, 1, mode="constant")
    interior = (
        mask
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    rows, columns = np.nonzero(mask & ~interior)
    return (int(columns[0]), int(rows[0])) if len(columns) else None


def _trace_boundary(mask: np.ndarray, clockwise: bool) -> tuple[AxisPoint, ...]:
    start = _boundary_start(mask)
    if start is None:
        return ()
    current = previous = before_previous = start
    length = 0
    path = [start]
    directions = (0, 1, 2, 3) if clockwise else (3, 2, 1, 0)
    while current != start or length == 0:
        direction: int | None = None
        while True:
            following = None
            for candidate in directions:
                target = _ahead(current, candidate)
                if target == previous or target == before_previous:
                    continue
                first, second = _side_pixels(current, candidate)
                if _inside(mask, first) != _inside(mask, second):
                    following = candidate
                    break
            if following is None:
                raise ValueError("Could not trace a closed foreground boundary")
            if direction is not None and direction != following:
                break
            direction = following
            before_previous, previous = previous, current
            current = _ahead(current, following)
            length += 1
        path.append(current)
    return tuple((float(x), float(y)) for x, y in path)


def foreground_contours(binary: np.ndarray) -> tuple[tuple[AxisPoint, ...], ...]:
    contours = []
    fields = ((binary < 128, True, False), (binary >= 128, False, True))
    for field, clockwise, exclude_border in fields:
        count, labels = cv2.connectedComponents(
            field.astype(np.uint8), connectivity=4
        )
        for label in range(1, count):
            mask = labels == label
            if exclude_border and (
                mask[0].any()
                or mask[-1].any()
                or mask[:, 0].any()
                or mask[:, -1].any()
            ):
                continue
            contour = _trace_boundary(mask, clockwise)
            if len(contour) >= 4:
                contours.append(contour)
    return tuple(contours)


def smooth_contours(
    contours: tuple[tuple[AxisPoint, ...], ...],
    angle_threshold: float,
    smooth_tolerance: float,
) -> tuple[tuple[BezierCurve, ...], ...]:
    return tuple(
        curves
        for contour in contours
        if (
            curves := smooth_closed_contour(
                contour, angle_threshold, smooth_tolerance
            )
        )
    )


def curve_anchors(
    contours: tuple[tuple[BezierCurve, ...], ...],
) -> tuple[tuple[AxisPoint, ...], ...]:
    return tuple(
        (curves[0].start, *(curve.end for curve in curves))
        for curves in contours
    )


def process_image(
    input_path: Path,
    binary_path: Path,
    edges_path: Path,
    lines_path: Path,
    vector_path: Path,
    filled_path: Path,
    threshold: int | None,
    angle_threshold: float,
    smooth_tolerance: float,
) -> None:
    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {input_path}")

    binary = binarize(gray, threshold)
    curves = smooth_contours(
        foreground_contours(binary), angle_threshold, smooth_tolerance
    )
    colors = random_line_colors(len(curves))
    edges = extract_edges(binary)
    lines_debug = draw_curves(binary.shape, curves, colors)
    vector = bezier_svg(
        binary.shape,
        curves,
        1,
        stroke_colors=colors,
        sample_chains=curve_anchors(curves),
    )
    filled = filled_bezier_svg(binary.shape, curves)

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    edges_path.parent.mkdir(parents=True, exist_ok=True)
    lines_path.parent.mkdir(parents=True, exist_ok=True)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    filled_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(binary_path), binary):
        raise OSError(f"Could not write image: {binary_path}")
    if not cv2.imwrite(str(edges_path), edges):
        raise OSError(f"Could not write image: {edges_path}")
    if not cv2.imwrite(str(lines_path), lines_debug):
        raise OSError(f"Could not write image: {lines_path}")
    vector_path.write_text(vector)
    filled_path.write_text(filled)


def output_paths_for(
    image_path: Path, input_root: Path, output_root: Path
) -> tuple[Path, Path, Path, Path, Path]:
    relative_path = image_path.relative_to(input_root)
    output_dir = output_root / relative_path.parent
    return (
        output_dir / f"{relative_path.stem}-binarize.png",
        output_dir / f"{relative_path.stem}-edges.png",
        output_dir / f"{relative_path.stem}-lines.png",
        output_dir / f"{relative_path.stem}-vector.svg",
        output_dir / f"{relative_path.stem}-filled.svg",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Binarize challenge images, remove pixel staircases from their "
            "contours, preserve corners, and fit smooth cubic Bezier curves."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("input/challenge_1"),
        help="image or directory to binarize (default: input/challenge_1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/challenge_1"),
        help="output directory (default: output/challenge_1)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        metavar="LEVEL",
        help="fixed threshold from 0 to 255 (default: Otsu thresholding)",
    )
    parser.add_argument(
        "-a",
        "--angle-threshold",
        type=float,
        default=DEFAULT_ANGLE_THRESHOLD,
        metavar="DEGREES",
        help=(
            "minimum direction change preserved as a corner "
            f"(default: {DEFAULT_ANGLE_THRESHOLD:g} degrees)"
        ),
    )
    parser.add_argument(
        "-s",
        "--smooth-tolerance",
        type=float,
        default=DEFAULT_SMOOTH_TOLERANCE,
        metavar="PIXELS",
        help=(
            "maximum Bezier fitting error "
            f"(default: {DEFAULT_SMOOTH_TOLERANCE:g})"
        ),
    )
    args = parser.parse_args()

    if args.threshold is not None and not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255")
    if not isfinite(args.angle_threshold) or not 0 <= args.angle_threshold <= 180:
        parser.error("--angle-threshold must be between 0 and 180 degrees")
    if not isfinite(args.smooth_tolerance) or args.smooth_tolerance <= 0:
        parser.error("--smooth-tolerance must be positive")

    input_is_dir = args.input.is_dir()
    image_paths: list[Path]
    if input_is_dir:
        image_paths = [
            path
            for path in sorted(args.input.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not image_paths:
            parser.error(f"no supported images found in {args.input}")
    elif not args.input.is_file():
        parser.error(f"input does not exist or is not a file: {args.input}")
    elif args.input.suffix.lower() not in IMAGE_SUFFIXES:
        parser.error(f"unsupported image type: {args.input.suffix}")
    else:
        image_paths = [args.input]

    input_root = args.input if input_is_dir else args.input.parent
    for image_path in image_paths:
        output_paths = output_paths_for(image_path, input_root, args.output)
        process_image(
            image_path,
            *output_paths,
            args.threshold,
            args.angle_threshold,
            args.smooth_tolerance,
        )


if __name__ == "__main__":
    main()
