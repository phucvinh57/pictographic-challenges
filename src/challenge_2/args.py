from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

from common.validation import non_negative_int, positive_float, positive_int, ratio

from .constants import (
    CANNY_HIGH,
    CANNY_LOW,
    IN_DIR,
    MIN_AREA,
    OUT_DIR,
    SAMPLE_SPACING,
    STROKE_WIDTH,
    TRANSITION_DELTA,
    TRANSITION_FLATNESS,
)


@dataclass(frozen=True)
class Args:
    input_dir: Path
    output_dir: Path
    canny_low: int
    canny_high: int
    min_area: int
    transition_delta: float
    transition_flatness: float
    sample_spacing: int
    stroke_width: float
    debug: bool

    @classmethod
    def parse(cls, argv: list[str] | None = None) -> Args:
        parser = ArgumentParser(description="Challenge 2: glyph skeleton extraction")

        paths = parser.add_argument_group("paths")
        paths.add_argument("-i", "--input-dir", type=Path, default=IN_DIR)
        paths.add_argument("-o", "--output-dir", type=Path, default=OUT_DIR)

        borders = parser.add_argument_group("Canny border extraction")
        borders.add_argument("--canny-low", type=non_negative_int, default=CANNY_LOW)
        borders.add_argument("--canny-high", type=positive_int, default=CANNY_HIGH)
        borders.add_argument("--min-area", type=positive_int, default=MIN_AREA)

        transitions = parser.add_argument_group("radius transition detection")
        transitions.add_argument(
            "--transition-delta",
            type=ratio,
            default=TRANSITION_DELTA,
            help="Minimum junction-to-stroke radius drop ratio",
        )
        transitions.add_argument(
            "--transition-flatness",
            type=ratio,
            default=TRANSITION_FLATNESS,
            help="Maximum plateau slope relative to the incoming slope",
        )

        smoothing = parser.add_argument_group("Bézier smoothing")
        smoothing.add_argument(
            "--sample-spacing",
            type=positive_int,
            default=SAMPLE_SPACING,
            help="Arc distance in pixels between points sampled before smoothing",
        )
        smoothing.add_argument(
            "--stroke-width",
            type=positive_float,
            default=STROKE_WIDTH,
            help="Fixed SVG edge width in pixels",
        )

        parser.add_argument(
            "-d", "--debug", action="store_true", help="Enable debug mode"
        )
        args = cls(**vars(parser.parse_args(argv)))
        if args.canny_high <= args.canny_low:
            parser.error("--canny-high must be greater than --canny-low")
        return args


PathField = Literal["input_dir", "output_dir"]
IntField = Literal["canny_low", "canny_high", "min_area", "sample_spacing"]
FloatField = Literal["transition_delta", "transition_flatness", "stroke_width"]
BoolField = Literal["debug"]

_args: Args | None = None


@overload
def get_args() -> Args: ...
@overload
def get_args(field: PathField) -> Path: ...
@overload
def get_args(field: IntField) -> int: ...
@overload
def get_args(field: FloatField) -> float: ...
@overload
def get_args(field: BoolField) -> bool: ...
def get_args(
    field: PathField | IntField | FloatField | BoolField | None = None,
) -> Args | Path | int | float | bool:
    global _args
    if _args is None:
        _args = Args.parse()
    return _args if field is None else getattr(_args, field)


def set_args(args: Args) -> None:
    global _args
    _args = args


__all__ = ["Args", "get_args", "set_args"]
