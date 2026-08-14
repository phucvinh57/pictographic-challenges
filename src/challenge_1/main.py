from challenge_1.svg import draw_bezier_svg

from .curve_fitting import fit_closed_contour
from .contour import extract_contours, process_contour, read_image
from .args import Args
from .constants import IN_DIR, IMAGE_SUFFIXES
from pathlib import Path


def convert_to_svg(image_path: Path, angle_threshold: float, debug: bool) -> None:
    image = read_image(image_path)
    contours = extract_contours(image)

    curves_list = []
    for c in contours:
        simplified_contour, straight_flags = process_contour(c)
        curves = fit_closed_contour(
            simplified_contour,
            straight_flags,
            tolerance=1.0,
            angle_threshold=angle_threshold,
        )
        curves_list.append(curves)
    shape = (int(image.shape[0]), int(image.shape[1]))
    draw_bezier_svg(shape, curves_list)

def main() -> None:
    args = Args.parse()
    images = [p for p in IN_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    for image in images:
        convert_to_svg(image, args.angle_threshold, args.debug)
