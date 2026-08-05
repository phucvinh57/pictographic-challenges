import argparse
from math import isfinite
from pathlib import Path

from .pipeline import process_image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binarize an image and extract its medial axis."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("input/challenge_2"),
        help="image or directory to process (default: input/challenge_2)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/challenge_2"),
        help="output directory (default: output/challenge_2)",
    )
    parser.add_argument(
        "-s",
        "--sample-spacing",
        type=float,
        default=50,
        metavar="PIXELS",
        help="arc-length spacing between axis samples (default: 50)",
    )
    parser.add_argument(
        "-w",
        "--stroke-width",
        type=float,
        default=None,
        metavar="PIXELS",
        help=(
            "width the smoothed curves are stroked at (default: measured from "
            "the ink, twice the median maximal-inscribed radius)"
        ),
    )
    args = parser.parse_args()

    if not isfinite(args.sample_spacing) or args.sample_spacing <= 0:
        parser.error("--sample-spacing must be positive")
    if args.stroke_width is not None and (
        not isfinite(args.stroke_width) or args.stroke_width <= 0
    ):
        parser.error("--stroke-width must be positive")

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
        relative_dir = image_path.relative_to(input_root).parent
        output_dir = args.output / relative_dir
        process_image(
            image_path, output_dir, args.sample_spacing, args.stroke_width
        )
