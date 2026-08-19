from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, fields

from common.validation import angle, ratio

SIMPLIFY_FLOOR = 0.35
BREAK_SPAN_FLOOR = 4.0
STRAIGHT_MIN_FLOOR = 4.0
STRAIGHT_TOLERANCE_FLOOR = 1.0
DOMINANT_FLOOR = 16.0
FIT_FLOOR = 0.5
TANGENT_SPAN_FLOOR = 2.0


@dataclass(frozen=True)
class VectorizationSettings:
    simplify_ratio: float = 0.0002
    break_span_ratio: float = 0.01
    break_angle_threshold: float = 30.0
    straight_min_ratio: float = 0.0067
    straight_tolerance_ratio: float = 0.00083
    straight_bow_ratio: float = 0.01
    dominant_ratio: float = 0.053
    corner_angle_threshold: float = 60.0
    fit_ratio: float = 0.000625
    tangent_span_ratio: float = 0.0025

    @classmethod
    def from_namespace(cls, namespace: Namespace) -> VectorizationSettings:
        return cls(
            **{
                item.name: getattr(namespace, item.name)
                for item in fields(cls)
            }
        )


DEFAULT_VECTORIZATION_SETTINGS = VectorizationSettings()
VECTOR_ARGUMENT_NAMES = tuple(item.name for item in fields(VectorizationSettings))


def add_vectorization_arguments(parser: ArgumentParser) -> None:
    defaults = DEFAULT_VECTORIZATION_SETTINGS
    simplify = parser.add_argument_group("contour simplification")
    simplify.add_argument(
        "--simplify-ratio",
        type=ratio,
        default=defaults.simplify_ratio,
        help="How far a dropped point may stray from its replacement chord, "
        "as a fraction of contour length",
    )

    breaks = parser.add_argument_group("corner detection")
    breaks.add_argument(
        "--break-span-ratio",
        type=ratio,
        default=defaults.break_span_ratio,
        help="Arc examined on each side of a vertex, as a fraction of contour length",
    )
    breaks.add_argument(
        "--break-angle-threshold",
        type=angle,
        default=defaults.break_angle_threshold,
        help="Total turning over that span that marks a break point",
    )

    straight = parser.add_argument_group("straight runs")
    straight.add_argument(
        "--straight-min-ratio",
        type=ratio,
        default=defaults.straight_min_ratio,
        help="Shortest straight span, as a fraction of contour length",
    )
    straight.add_argument(
        "--straight-tolerance-ratio",
        type=ratio,
        default=defaults.straight_tolerance_ratio,
        help="Largest bow inside a straight span, as a fraction of contour length",
    )
    straight.add_argument(
        "--straight-bow-ratio",
        type=ratio,
        default=defaults.straight_bow_ratio,
        help="Largest bow allowed inside a straight span, relative to its length",
    )
    straight.add_argument(
        "--dominant-ratio",
        type=ratio,
        default=defaults.dominant_ratio,
        help="Edge that forces a break at both ends, as a fraction of contour length",
    )

    fitting = parser.add_argument_group("curve fitting")
    fitting.add_argument(
        "-a",
        "--corner-angle-threshold",
        type=angle,
        default=defaults.corner_angle_threshold,
        help="Turn at a vertex that makes it a corner",
    )
    fitting.add_argument(
        "-t",
        "--fit-ratio",
        type=ratio,
        default=defaults.fit_ratio,
        help="Largest curve deviation, as a fraction of contour length",
    )
    fitting.add_argument(
        "--tangent-span-ratio",
        type=ratio,
        default=defaults.tangent_span_ratio,
        help="Arc used to estimate a tangent, as a fraction of contour length",
    )


def remove_vectorization_arguments(values: dict[str, object]) -> None:
    for name in VECTOR_ARGUMENT_NAMES:
        values.pop(name)
