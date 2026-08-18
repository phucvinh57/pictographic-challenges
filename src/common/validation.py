from argparse import ArgumentTypeError


def non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise ArgumentTypeError(f"must be 0 or greater, got {value}")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise ArgumentTypeError(f"must be greater than 0, got {value}")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise ArgumentTypeError(f"must be 0 or greater, got {value}")
    return number


def ratio(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise ArgumentTypeError(f"must be a fraction in (0, 1], got {value}")
    return number


def angle(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 180:
        raise ArgumentTypeError(f"must be between 0 and 180 degrees, got {value}")
    return number


__all__ = [
    "angle",
    "non_negative_float",
    "non_negative_int",
    "positive_int",
    "ratio",
]
