from .contour import extract_contours, process_contour

from .args import Args
from .constants import IN_DIR, IMAGE_SUFFIXES
from pathlib import Path


def convert_to_svg(image_path: Path, angle_threshold: float, debug: bool) -> None:
    contours = extract_contours(image_path)

    simplified_contours = []
    straights = []

    for c in contours:
        simplified, straight_flags = process_contour(c)
        simplified_contours.append(simplified)
        straights.append(straight_flags)

    


def main() -> None:
    args = Args.parse()

    images = [p for p in IN_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    for image in images:
        convert_to_svg(image, args.angle_threshold, args.debug)
