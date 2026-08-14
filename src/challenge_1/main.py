from pathlib import Path

from . import debug
from .args import Args, get_args, set_args
from .constants import IMAGE_SUFFIXES
from .contour import extract_contours, process_contour, read_image_in_gray_scale
from .curve_fitting import fit_closed_contour
from .svg import draw_bezier_svg


def convert_to_svg(image_path: Path) -> None:
    args = get_args()
    debug.begin(image_path.name)
    with debug.timed("read image"):
        image = read_image_in_gray_scale(image_path)
    with debug.timed("extract contours"):
        contours = extract_contours(image)
    debug.count("contours", len(contours))

    curves_list = []
    for c in contours:
        with debug.timed("process contours"):
            simplified_contour, straight_flags = process_contour(c)
        with debug.timed("fit curves"):
            curves_list.append(fit_closed_contour(simplified_contour, straight_flags))
    shape = (int(image.shape[0]), int(image.shape[1]))
    with debug.timed("draw svg"):
        svg = draw_bezier_svg(shape, curves_list)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with debug.timed("write svg"):
        (args.output_dir / f"{image_path.stem}.svg").write_text(svg)
    debug.count("svg bytes", len(svg))
    debug.report()


def main() -> None:
    set_args(Args.parse())
    images = [
        p for p in get_args().input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    ]
    for image in images:
        convert_to_svg(image)
