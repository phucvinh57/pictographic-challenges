from pathlib import Path

import cv2

from common import debug

from .args import get_args
from .constants import IMAGE_SUFFIXES
from .contour import extract_ink_mask, read_image_in_gray_scale
from .skeleton import skeletonize


def convert_to_skeleton(image_path: Path) -> None:
    output_dir = get_args("output_dir")
    debug.begin(image_path.name)
    with debug.timed("read image"):
        image = read_image_in_gray_scale(image_path)
    with debug.timed("extract Canny contours"):
        mask = extract_ink_mask(image)
    with debug.timed("thin skeleton"):
        skeleton = skeletonize(mask)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}.png"
    with debug.timed("write PNG"):
        if not cv2.imwrite(str(output_path), skeleton):
            raise ValueError(f"Failed to write image: {output_path}")
    debug.count("skeleton pixels", int((skeleton == 0).sum()))
    debug.report()


def main() -> None:
    debug.configure(get_args("debug"))
    input_dir = get_args("input_dir")
    images = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in images:
        convert_to_skeleton(image_path)
