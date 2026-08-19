from pathlib import Path

import cv2
import numpy as np

from common import debug

from .args import get_args
from .constants import IMAGE_SUFFIXES
from .contour import extract_ink_mask, read_image_in_gray_scale
from .skeleton import MedialAxis, skeletonize

_EDGE_GREY = (225, 225, 225)
_AXIS_GREY = (170, 170, 170)
def write_debug_image(medial: MedialAxis, edges: np.ndarray, output_path: Path) -> None:
    overlay = np.full((*medial.shape, 3), 255, dtype=np.uint8)
    overlay[edges > 0] = _EDGE_GREY
    overlay[medial.axis] = _AXIS_GREY

    if not cv2.imwrite(str(output_path), overlay):
        raise ValueError(f"Failed to write image: {output_path}")


def convert_to_skeleton(image_path: Path) -> None:
    args = get_args()
    debug.begin(image_path.name)
    with debug.timed("read image"):
        image = read_image_in_gray_scale(image_path)
    with debug.timed("extract Canny contours"):
        mask, edges = extract_ink_mask(image)
    with debug.timed("thin skeleton"):
        medial_axis = skeletonize(mask)
    raster = np.where(medial_axis.axis, 0, 255).astype(np.uint8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{image_path.stem}.png"
    with debug.timed("write PNG"):
        if not cv2.imwrite(str(output_path), raster):
            raise ValueError(f"Failed to write image: {output_path}")
        if args.debug:
            write_debug_image(
                medial_axis, edges, args.output_dir / f"{image_path.stem}_debug.png"
            )
    debug.count("skeleton pixels", int((raster == 0).sum()))
    debug.report()


def main() -> None:
    debug.configure(get_args("debug"))

    input_dir = get_args("input_dir")
    images = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in images:
        convert_to_skeleton(image_path)
