from pathlib import Path

from .args import Args, get_args, set_args
from .constants import IMAGE_SUFFIXES
from .contour import extract_contours, process_contour, read_image
from .curve_fitting import fit_closed_contour
from .svg import draw_bezier_svg


def convert_to_svg(image_path: Path) -> None:
    args = get_args()
    image = read_image(image_path)
    contours = extract_contours(image)

    curves_list = []
    for c in contours:
        simplified_contour, straight_flags = process_contour(c)
        curves_list.append(fit_closed_contour(simplified_contour, straight_flags))
    shape = (int(image.shape[0]), int(image.shape[1]))
    svg = draw_bezier_svg(shape, curves_list)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{image_path.stem}.svg").write_text(svg)


def main() -> None:
    set_args(Args.parse())
    images = [
        p for p in get_args().input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    ]
    for image in images:
        convert_to_svg(image)
