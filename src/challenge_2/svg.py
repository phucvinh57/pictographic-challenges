from __future__ import annotations

from collections.abc import Sequence

from common.vectorization import AxisPoint, BezierCurve


def _number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _point(point: AxisPoint) -> str:
    return f"{_number(point[0])},{_number(point[1])}"


def _contour_path(curves: Sequence[BezierCurve]) -> str:
    if not curves:
        return ""
    commands = [f"M{_point(curves[0].start)}"]
    for curve in curves:
        commands.append(
            "C"
            f"{_point(curve.first_control)} "
            f"{_point(curve.second_control)} "
            f"{_point(curve.end)}"
        )
    if curves[-1].end == curves[0].start:
        commands.append("Z")
    return " ".join(commands)


def draw_svg(
    shape: tuple[int, int],
    contours: Sequence[Sequence[BezierCurve]],
    stroke_width: float,
) -> str:
    height, width = shape
    path_data = " ".join(
        path for curves in contours if (path := _contour_path(curves))
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'<path d="{path_data}" fill="none" stroke="black" '
        f'stroke-width="{_number(stroke_width)}" stroke-linecap="round" '
        f'stroke-linejoin="round"/>'
        f"</svg>\n"
    )


__all__ = ["draw_svg"]
