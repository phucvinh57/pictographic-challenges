from pathlib import Path

import cv2

from pictographic.svg import bezier_svg, filled_bezier_svg

from .contours import (
    curve_anchors,
    extract_contours,
    preprocess_contours,
    smooth_contours,
)
from .raster import (
    binarize,
    draw_contours,
    random_contour_colors,
    threshold_level,
)


def process_image(
    input_path: Path,
    binary_path: Path,
    contours_path: Path,
    vector_path: Path,
    filled_path: Path,
    threshold: int | None,
    angle_threshold: float,
    smooth_tolerance: float,
) -> None:
    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {input_path}")

    level = threshold_level(gray, threshold)
    binary = binarize(gray, level)
    contours, straights = preprocess_contours(extract_contours(gray, level))
    contours_debug = draw_contours(
        binary.shape, contours, random_contour_colors(len(contours))
    )
    curves = smooth_contours(
        contours, straights, angle_threshold, smooth_tolerance
    )
    colors = random_contour_colors(len(curves))
    vector = bezier_svg(
        binary.shape,
        curves,
        1,
        stroke_colors=colors,
        sample_chains=curve_anchors(curves),
    )
    filled = filled_bezier_svg(binary.shape, curves)

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    contours_path.parent.mkdir(parents=True, exist_ok=True)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    filled_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(binary_path), binary):
        raise OSError(f"Could not write image: {binary_path}")
    if not cv2.imwrite(str(contours_path), contours_debug):
        raise OSError(f"Could not write image: {contours_path}")

    vector_path.write_text(vector)
    filled_path.write_text(filled)


def output_paths_for(
    image_path: Path, input_root: Path, output_root: Path
) -> tuple[Path, Path, Path, Path]:
    relative_path = image_path.relative_to(input_root)
    output_dir = output_root / relative_path.parent
    return (
        output_dir / f"{relative_path.stem}-1-binarize.png",
        output_dir / f"{relative_path.stem}-2-contours.png",
        output_dir / f"{relative_path.stem}-vector.svg",
        output_dir / f"{relative_path.stem}-filled.svg",
    )
