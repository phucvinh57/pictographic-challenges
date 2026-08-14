from __future__ import annotations

from math import hypot

from . import debug
from .args import get_args
from .geometry import AxisPoint, BezierCurve


def _number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _point(point: AxisPoint) -> str:
    return f"{_number(point[0])},{_number(point[1])}"


def _straight(curve: BezierCurve) -> bool:
    tolerance = get_args().line_tolerance
    dx = curve.end[0] - curve.start[0]
    dy = curve.end[1] - curve.start[1]
    length = hypot(dx, dy)
    if not length:
        return True
    return all(
        abs(dx * (point[1] - curve.start[1]) - dy * (point[0] - curve.start[0]))
        <= tolerance * length
        for point in (curve.first_control, curve.second_control)
    )


def _same_direction(first: BezierCurve, second: BezierCurve) -> bool:
    first_dx = first.end[0] - first.start[0]
    first_dy = first.end[1] - first.start[1]
    second_dx = second.end[0] - second.start[0]
    second_dy = second.end[1] - second.start[1]
    cross = first_dx * second_dy - first_dy * second_dx
    lengths = hypot(first_dx, first_dy) * hypot(second_dx, second_dy)
    dot = first_dx * second_dx + first_dy * second_dy
    return bool(lengths and abs(cross) <= 1e-9 * lengths and dot > 0)


def _path(curves: list[BezierCurve]) -> str:
    commands = [f"M{_point(curves[0].start)}"]
    previous: BezierCurve | None = None
    command = ""
    for curve in curves:
        if _straight(curve):
            line = f"L{_point(curve.end)}"
            if command == "L" and previous is not None and _same_direction(previous, curve):
                commands[-1] = line
            else:
                commands.append(line)
            command = "L"
        else:
            controls = " ".join(
                (
                    _point(curve.first_control),
                    _point(curve.second_control),
                    _point(curve.end),
                )
            )
            commands.append(f'{"" if command == "C" else "C"}{controls}')
            command = "C"
        previous = curve
        debug.count("svg lines" if command == "L" else "svg cubics")
    if curves[-1].end == curves[0].start:
        commands.append("Z")
    return " ".join(commands)


def draw_bezier_svg(
    shape: tuple[int, int],
    contours: list[list[BezierCurve]],
) -> str:
    height, width = shape
    path = " ".join(_path(curves) for curves in contours if curves)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'<path d="{path}" fill="black" fill-rule="evenodd"/>'
        f"</svg>\n"
    )