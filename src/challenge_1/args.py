from argparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class Args:
    angle_threshold: float
    debug: bool

    @classmethod
    def parse(cls) -> "Args":
        parser = ArgumentParser(description="Challenge 1")

        parser.add_argument("-a", "--angle-threshold", type=float, default=60.0)
        parser.add_argument(
            "-d", "--debug", action="store_true", help="Enable debug mode"
        )

        args = parser.parse_args()
        if not 0 <= args.angle_threshold <= 180:
            parser.error("--angle-threshold must be between 0 and 180 degrees")

        return cls(
            angle_threshold=args.angle_threshold,
            debug=args.debug,
        )
