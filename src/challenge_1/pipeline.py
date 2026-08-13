from pathlib import Path

import cv2

from .contours import (
    extract_contours,
    preprocess_contours,
    smooth_contours,
)
from .raster import grayscale_on_white, threshold_level
from .svg import filled_bezier_svg


def process_image(
    input_path: Path,
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
    filled = filled_bezier_svg(shape, curves)

    filled_path.parent.mkdir(parents=True, exist_ok=True)
    filled_path.write_text(filled)


def output_path_for(image_path: Path, input_root: Path, output_root: Path) -> Path:
    relative_path = image_path.relative_to(input_root)
    return output_root / relative_path.parent / f"{relative_path.stem}-filled.svg"
