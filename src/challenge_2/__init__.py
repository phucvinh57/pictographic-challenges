import argparse
from pathlib import Path

from .skeletonize import process_image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binarize an image and thin it with the Zhang-Suen algorithm."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("input/challenge_2"),
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("output/challenge_2"),
    )
    parser.add_argument(
        "--stroke-width",
        type=float,
        default=45.0,
        help="Fixed stroke width used when vectorizing the skeleton to SVG.",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        for image_path in sorted(args.input.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative_dir = image_path.parent.relative_to(args.input)
            out_dir = args.output / relative_dir
            process_image(image_path, out_dir, stroke_width=args.stroke_width)
    else:
        process_image(args.input, args.output, stroke_width=args.stroke_width)
