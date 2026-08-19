from __future__ import annotations

import numpy as np

from .smoothing import SmoothedEdge


def _number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _point(point: np.ndarray) -> str:
    return f"{_number(float(point[0]))},{_number(float(point[1]))}"


def _edge_path(edge: SmoothedEdge) -> str:
    if not edge.curves:
        if len(edge.samples) == 0:
            return ""
        return f"M{_point(edge.samples[0])} l0,0"

    commands = [f"M{_point(edge.curves[0].start)}"]
    for curve in edge.curves:
        commands.append(
            "C"
            f"{_point(curve.first_control)} "
            f"{_point(curve.second_control)} "
            f"{_point(curve.end)}"
        )
    if np.allclose(edge.curves[-1].end, edge.curves[0].start):
        commands.append("Z")
    return " ".join(commands)


def draw_svg(
    shape: tuple[int, int],
    edges: list[SmoothedEdge],
    stroke_width: float,
) -> str:
    height, width = shape
    path_data = " ".join(path for edge in edges if (path := _edge_path(edge)))
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
