from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

from .constants import (
    BREAK_ANGLE_THRESHOLD,
    BREAK_SPAN_RATIO,
    CORNER_ANGLE_THRESHOLD,
    DOMINANT_RATIO,
    FIT_RATIO,
    IN_DIR,
    LINE_TOLERANCE,
    OUT_DIR,
    SIMPLIFY_RATIO,
    STRAIGHT_BOW_RATIO,
    STRAIGHT_MIN_RATIO,
    STRAIGHT_TOLERANCE_RATIO,
    TANGENT_SPAN_RATIO,
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


def _ratio(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise ArgumentTypeError(f"must be a fraction in (0, 1], got {value}")
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

    simplify_ratio: float

    break_span_ratio: float
    break_angle_threshold: float

    straight_min_ratio: float
    straight_tolerance_ratio: float
    straight_bow_ratio: float
    dominant_ratio: float

    corner_angle_threshold: float
    fit_ratio: float
    tangent_span_ratio: float

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
            "--simplify-ratio",
            type=_ratio,
            default=SIMPLIFY_RATIO,
            help="How far a dropped point may stray from the chord replacing it, "
            "as a fraction of perimeter",
        )

        breaks = parser.add_argument_group("corner detection")
        breaks.add_argument(
            "--break-span-ratio",
            type=_ratio,
            default=BREAK_SPAN_RATIO,
            help="Arc looked at on each side of a vertex, as a fraction of perimeter",
        )
        breaks.add_argument(
            "--break-angle-threshold",
            type=_angle,
            default=BREAK_ANGLE_THRESHOLD,
            help="Total turning over that span that marks a break point",
        )

        straight = parser.add_argument_group("straight runs")
        straight.add_argument(
            "--straight-min-ratio",
            type=_ratio,
            default=STRAIGHT_MIN_RATIO,
            help="Shortest straight span, as a fraction of perimeter",
        )
        straight.add_argument(
            "--straight-tolerance-ratio",
            type=_ratio,
            default=STRAIGHT_TOLERANCE_RATIO,
            help="Largest bow inside a straight span, as a fraction of perimeter",
        )
        straight.add_argument(
            "--straight-bow-ratio",
            type=_ratio,
            default=STRAIGHT_BOW_RATIO,
            help="Largest bow allowed inside a straight span, relative to its length",
        )
        straight.add_argument(
            "--dominant-ratio",
            type=_ratio,
            default=DOMINANT_RATIO,
            help="Edge that forces a break at both ends, as a fraction of perimeter",
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
            "--fit-ratio",
            type=_ratio,
            default=FIT_RATIO,
            help="Largest deviation between a curve and the contour, "
            "as a fraction of perimeter",
        )
        fitting.add_argument(
            "--tangent-span-ratio",
            type=_ratio,
            default=TANGENT_SPAN_RATIO,
            help="Arc used to estimate a tangent at a cut, as a fraction of perimeter",
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


PathField = Literal["input_dir", "output_dir"]
FloatField = Literal[
    "simplify_ratio",
    "break_span_ratio",
    "break_angle_threshold",
    "straight_min_ratio",
    "straight_tolerance_ratio",
    "straight_bow_ratio",
    "dominant_ratio",
    "corner_angle_threshold",
    "fit_ratio",
    "tangent_span_ratio",
    "line_tolerance",
]
BoolField = Literal["debug"]

_args: Args | None = None


@overload
def get_args() -> Args: ...
@overload
def get_args(field: PathField) -> Path: ...
@overload
def get_args(field: FloatField) -> float: ...
@overload
def get_args(field: BoolField) -> bool: ...
def get_args(
    field: PathField | FloatField | BoolField | None = None,
) -> Args | Path | float | bool:
    """The parsed arguments, parsed once on first use and shared after that."""
    global _args
    if _args is None:
        _args = Args.parse()
    return _args if field is None else getattr(_args, field)


def set_args(args: Args) -> None:
    global _args
    _args = args


__all__ = ["Args", "get_args", "set_args"]
