import argparse
from pathlib import Path

from .skeletonize import process_image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binarize an image and extract its medial axis."
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
        "--sample-spacing",
        type=float,
        default=50,
        metavar="PIXELS",
        help="arc-length spacing between axis samples (default: 10)",
    )
    parser.add_argument(
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

    if args.input.is_dir():
        for image_path in sorted(args.input.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative_dir = image_path.parent.relative_to(args.input)
            out_dir = args.output / relative_dir
            process_image(
                image_path, out_dir, args.sample_spacing, args.stroke_width
            )
    else:
        process_image(
            args.input, args.output, args.sample_spacing, args.stroke_width
        )
