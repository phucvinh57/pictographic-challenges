import argparse
from math import isfinite
from pathlib import Path

from .pipeline import output_paths_for, process_image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_ANGLE_THRESHOLD = 60
DEFAULT_SMOOTH_TOLERANCE = 0.75


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
