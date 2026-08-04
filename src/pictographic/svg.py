from __future__ import annotations

from collections.abc import Sequence
from math import hypot

from .curves import AxisPoint, BezierCurve


def _number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _point(point: AxisPoint) -> str:
    return f"{_number(point[0])},{_number(point[1])}"


def _straight(curve: BezierCurve, tolerance: float = 0.005) -> bool:
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


def _path(curves: tuple[BezierCurve, ...]) -> str:
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
    if curves[-1].end == curves[0].start:
        commands.append("Z")
    return " ".join(commands)


def bezier_svg(
    shape: tuple[int, int],
    chains: tuple[tuple[BezierCurve, ...], ...],
    stroke_width: float,
    stroke_colors: Sequence[str] | None = None,
    sample_chains: Sequence[Sequence[AxisPoint]] | None = None,
    sample_radius: float = 2,
) -> str:
    height, width = shape
    if stroke_colors is not None and len(stroke_colors) != len(chains):
        raise ValueError("stroke_colors must contain one color per Bezier chain")
    if sample_chains is not None and len(sample_chains) != len(chains):
        raise ValueError("sample_chains must contain one point chain per Bezier chain")
    paths = []
    for index, curves in enumerate(chains):
        if not curves:
            continue
        color = "" if stroke_colors is None else f' stroke="{stroke_colors[index]}"'
        paths.append(f'    <path d="{_path(curves)}"{color} />')
    samples = []
    for index, points in enumerate(sample_chains or ()):
        color = "black" if stroke_colors is None else stroke_colors[index]
        unique_points = points[:-1] if len(points) > 1 and points[0] == points[-1] else points
        samples.extend(
            f'    <circle cx="{_number(point[0])}" cy="{_number(point[1])}" '
            f'r="{_number(sample_radius)}" fill="{color}" />'
            for point in unique_points
        )
    sample_group = (
        "  <g>\n" + "\n".join(samples) + "\n  </g>\n" if samples else ""
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'  <rect width="{width}" height="{height}" fill="white" />\n'
        f'  <g fill="none" stroke="black" stroke-width="{_number(stroke_width)}" '
        f'stroke-linecap="round" stroke-linejoin="round">\n'
        f'{"\n".join(paths)}\n'
        f"  </g>\n"
        f"{sample_group}"
        f"</svg>\n"
    )


def filled_bezier_svg(
    shape: tuple[int, int],
    contours: Sequence[tuple[BezierCurve, ...]],
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
