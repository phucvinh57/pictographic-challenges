from dataclasses import dataclass

AxisPoint = tuple[float, float]


@dataclass(frozen=True)
class Contour:
    points: tuple[AxisPoint, ...]
    closed: bool


@dataclass(frozen=True)
class ProcessedContour:
    points: tuple[AxisPoint, ...]
    straight_flags: tuple[bool, ...]
    closed: bool


@dataclass(frozen=True)
class BezierCurve:
    start: AxisPoint
    first_control: AxisPoint
    second_control: AxisPoint
    end: AxisPoint
