from __future__ import annotations

from .graph import AxisPoint, BezierCurve


def _number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _point(point: AxisPoint) -> str:
    return f"{_number(point[0])},{_number(point[1])}"


def _path(curves: tuple[BezierCurve, ...]) -> str:
    """One chain as a polybezier, the spans sharing the endpoints they meet at."""
    commands = [f"M{_point(curves[0].start)}", "C"]
    commands.extend(
        " ".join(
            (
                _point(curve.first_control),
                _point(curve.second_control),
                _point(curve.end),
            )
        )
        for curve in curves
    )
    return " ".join(commands)


def bezier_svg(
    shape: tuple[int, int],
    chains: tuple[tuple[BezierCurve, ...], ...],
    stroke_width: float,
) -> str:
    """Write the smoothed graph as stroked SVG paths.

    The curves are the axis, so the drawing is recovered by stroking them at the
    ink's own width rather than by outlining them: round caps and joins put back
    the disc the axis carried, which is what every end and every meeting of the
    strokes was inscribed in. An SVG has no background of its own, so the ink is
    laid over an opaque white rectangle to match the rasterized outputs.
    """
    height, width = shape
    paths = "\n".join(f'    <path d="{_path(curves)}" />' for curves in chains if curves)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'  <rect width="{width}" height="{height}" fill="white" />\n'
        f'  <g fill="none" stroke="black" stroke-width="{_number(stroke_width)}" '
        f'stroke-linecap="round" stroke-linejoin="round">\n'
        f"{paths}\n"
        f"  </g>\n"
        f"</svg>\n"
    )
