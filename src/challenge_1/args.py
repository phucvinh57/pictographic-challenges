from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

from common.validation import non_negative_float
from common.vectorization import (
    VectorizationSettings,
    add_vectorization_arguments,
    remove_vectorization_arguments,
)

from .constants import IN_DIR, LINE_TOLERANCE, OUT_DIR


@dataclass(frozen=True)
class Args:
    input_dir: Path
    output_dir: Path
    vectorization: VectorizationSettings
    line_tolerance: float
    debug: bool

    @classmethod
    def parse(cls, argv: list[str] | None = None) -> Args:
        parser = ArgumentParser(description="Challenge 1: raster to SVG")

        paths = parser.add_argument_group("paths")
        paths.add_argument("-i", "--input-dir", type=Path, default=IN_DIR)
        paths.add_argument("-o", "--output-dir", type=Path, default=OUT_DIR)
        add_vectorization_arguments(parser)

        output = parser.add_argument_group("svg output")
        output.add_argument(
            "--line-tolerance",
            type=non_negative_float,
            default=LINE_TOLERANCE,
            help="Flatness under which a cubic is written as a line",
        )
        parser.add_argument(
            "-d", "--debug", action="store_true", help="Enable debug mode"
        )

        namespace = parser.parse_args(argv)
        vectorization = VectorizationSettings.from_namespace(namespace)
        values = vars(namespace)
        remove_vectorization_arguments(values)
        return cls(vectorization=vectorization, **values)


PathField = Literal["input_dir", "output_dir"]
FloatField = Literal["line_tolerance"]
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
    global _args
    if _args is None:
        _args = Args.parse()
    return _args if field is None else getattr(_args, field)


def set_args(args: Args) -> None:
    global _args
    _args = args


__all__ = ["Args", "get_args", "set_args"]
