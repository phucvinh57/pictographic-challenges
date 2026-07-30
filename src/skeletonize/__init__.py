import argparse
from pathlib import Path

from .skeletonize import process_image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Binarize an image and estimate its stroke centerline directly "
            "from its boundary contours."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("input/skeletonize"),
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("output/skeletonize"),
    )
    parser.add_argument(
        "--width-min-ratio",
        type=float,
        default=0.9,
        help=(
            "Minimum accepted boundary-pair distance, as a ratio of the "
            "detected stroke width. Default: 0.9."
        ),
    )
    parser.add_argument(
        "--width-max-ratio",
        type=float,
        default=1.1,
        help=(
            "Maximum accepted boundary-pair distance, as a ratio of the "
            "detected stroke width. Default: 1.1."
        ),
    )
    args = parser.parse_args()

    if args.input.is_dir():
        for image_path in sorted(args.input.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative_dir = image_path.parent.relative_to(args.input)
            out_dir = args.output / relative_dir
            process_image(
                image_path,
                out_dir,
                width_min_ratio=args.width_min_ratio,
                width_max_ratio=args.width_max_ratio,
            )
    else:
        process_image(
            args.input,
            args.output,
            width_min_ratio=args.width_min_ratio,
            width_max_ratio=args.width_max_ratio,
        )
