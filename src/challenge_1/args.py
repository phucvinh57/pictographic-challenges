from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError
from dataclasses import dataclass
from pathlib import Path

from .constants import (
    BREAK_ANGLE_THRESHOLD,
    BREAK_SPAN_LENGTH,
    COLLINEAR_EPSILON,
    CORNER_ANGLE_THRESHOLD,
    DOMINANT_LENGTH,
    FIT_TOLERANCE,
    IN_DIR,
    LINE_TOLERANCE,
    OUT_DIR,
    SIMPLIFY_TOLERANCE,
    STRAIGHT_MIN_LENGTH,
    STRAIGHT_RADIUS,
    STRAIGHT_TOLERANCE,
    TANGENT_SPAN,
)


def _positive(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise ArgumentTypeError(f"must be greater than 0, got {value}")
    return number


def _non_negative(value: str) -> float:
    number = float(value)
    if number < 0:
        raise ArgumentTypeError(f"must be 0 or greater, got {value}")
    return number


def _angle(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 180:
        raise ArgumentTypeError(f"must be between 0 and 180 degrees, got {value}")
    return number


@dataclass(frozen=True)
class Args:
    input_dir: Path
    output_dir: Path

    simplify_tolerance: float
    collinear_epsilon: float

    break_span_length: float
    break_angle_threshold: float

    straight_min_length: float
    straight_tolerance: float
    straight_radius: float
    dominant_length: float

    corner_angle_threshold: float
    fit_tolerance: float
    tangent_span: float

    line_tolerance: float

    debug: bool

    @classmethod
    def parse(cls, argv: list[str] | None = None) -> Args:
        parser = ArgumentParser(description="Challenge 1: raster to SVG")

        paths = parser.add_argument_group("paths")
        paths.add_argument("-i", "--input-dir", type=Path, default=IN_DIR)
        paths.add_argument("-o", "--output-dir", type=Path, default=OUT_DIR)

        simplify = parser.add_argument_group("contour simplification")
        simplify.add_argument(
            "--simplify-tolerance",
            type=_positive,
            default=SIMPLIFY_TOLERANCE,
            help="Penalty budget when dropping points from a contour",
        )
        simplify.add_argument(
            "--collinear-epsilon",
            type=_positive,
            default=COLLINEAR_EPSILON,
            help="Cross product below which three points count as collinear",
        )

        breaks = parser.add_argument_group("corner detection")
        breaks.add_argument(
            "--break-span-length",
            type=_positive,
            default=BREAK_SPAN_LENGTH,
            help="Arc length looked at on each side of a vertex",
        )
        breaks.add_argument(
            "--break-angle-threshold",
            type=_angle,
            default=BREAK_ANGLE_THRESHOLD,
            help="Total turning over that span that marks a break point",
        )

        straight = parser.add_argument_group("straight runs")
        straight.add_argument(
            "--straight-min-length",
            type=_positive,
            default=STRAIGHT_MIN_LENGTH,
            help="Shortest span that may be called straight",
        )
        straight.add_argument(
            "--straight-tolerance",
            type=_positive,
            default=STRAIGHT_TOLERANCE,
            help="Largest bow allowed inside a straight span",
        )
        straight.add_argument(
            "--straight-radius",
            type=_positive,
            default=STRAIGHT_RADIUS,
            help="Smallest curvature radius still treated as straight",
        )
        straight.add_argument(
            "--dominant-length",
            type=_positive,
            default=DOMINANT_LENGTH,
            help="Edge length that always forces a break at both ends",
        )

        fitting = parser.add_argument_group("curve fitting")
        fitting.add_argument(
            "-a",
            "--corner-angle-threshold",
            type=_angle,
            default=CORNER_ANGLE_THRESHOLD,
            help="Turn at a vertex that makes it a corner",
        )
        fitting.add_argument(
            "-t",
            "--fit-tolerance",
            type=_positive,
            default=FIT_TOLERANCE,
            help="Largest deviation allowed between a curve and the contour",
        )
        fitting.add_argument(
            "--tangent-span",
            type=_positive,
            default=TANGENT_SPAN,
            help="Arc length used to estimate a tangent at a cut",
        )

        output = parser.add_argument_group("svg output")
        output.add_argument(
            "--line-tolerance",
            type=_non_negative,
            default=LINE_TOLERANCE,
            help="Flatness under which a cubic is written as a line",
        )

        parser.add_argument(
            "-d", "--debug", action="store_true", help="Enable debug mode"
        )

        return cls(**vars(parser.parse_args(argv)))


_args: Args | None = None


def get_args() -> Args:
    """The parsed arguments, parsed once on first use and shared after that."""
    global _args
    if _args is None:
        _args = Args.parse()
    return _args


def set_args(args: Args) -> None:
    global _args
    _args = args


__all__ = ["Args", "get_args", "set_args"]
