from pathlib import Path

import cv2

from pictographic.svg import bezier_svg, filled_bezier_svg

from .contours import (
    curve_anchors,
    extract_contours,
    preprocess_contours,
    smooth_contours,
)
from .raster import grayscale_on_white, random_contour_colors, threshold_level


def process_image(
    input_path: Path,
    vector_path: Path,
    filled_path: Path,
    threshold: int | None,
    angle_threshold: float,
    smooth_tolerance: float,
) -> None:
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")
    gray = grayscale_on_white(image)

    shape = (int(gray.shape[0]), int(gray.shape[1]))
    level = threshold_level(gray, threshold)
    contours, straights = preprocess_contours(extract_contours(gray, level))
    curves = smooth_contours(
        contours, straights, angle_threshold, smooth_tolerance
    )
    vector = bezier_svg(
        shape,
        curves,
        1,
        stroke_colors=random_contour_colors(len(curves)),
        sample_chains=curve_anchors(curves),
    )
    filled = filled_bezier_svg(shape, curves)

    vector_path.parent.mkdir(parents=True, exist_ok=True)
    filled_path.parent.mkdir(parents=True, exist_ok=True)

    vector_path.write_text(vector)
    filled_path.write_text(filled)


def output_paths_for(
    image_path: Path, input_root: Path, output_root: Path
) -> tuple[Path, Path]:
    relative_path = image_path.relative_to(input_root)
    output_dir = output_root / relative_path.parent
    return (
        output_dir / f"{relative_path.stem}-vector.svg",
        output_dir / f"{relative_path.stem}-filled.svg",
    )
