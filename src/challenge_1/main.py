from pathlib import Path

from challenge_1.svg import draw_bezier_svg

from .args import Args
from .constants import IMAGE_SUFFIXES, IN_DIR, OUT_DIR, SMOOTH_TOLERANCE
from .contour import extract_contours, process_contour, read_image
from .curve_fitting import fit_closed_contour


def convert_to_svg(image_path: Path, angle_threshold: float, debug: bool) -> None:
    image = read_image(image_path)
    contours = extract_contours(image)

    curves_list = []
    for c in contours:
        simplified_contour, straight_flags = process_contour(c)
        curves = fit_closed_contour(
            simplified_contour,
            straight_flags,
            tolerance=SMOOTH_TOLERANCE,
            angle_threshold=angle_threshold,
        )
        curves_list.append(curves)
    shape = (int(image.shape[0]), int(image.shape[1]))
    svg = draw_bezier_svg(shape, curves_list)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{image_path.stem}.svg").write_text(svg)

def main() -> None:
    args = Args.parse()
    images = [p for p in IN_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    for image in images:
        convert_to_svg(image, args.angle_threshold, args.debug)
