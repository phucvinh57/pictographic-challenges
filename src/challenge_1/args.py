from argparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class Args:
    angle_threshold: int
    debug: bool

    @classmethod
    def parse(cls) -> "Args":
        parser = ArgumentParser(description="Challenge 1")

        parser.add_argument("-a", "--angle-threshold", type=int, default=60)
        parser.add_argument(
            "-d", "--debug", action="store_true", help="Enable debug mode"
        )

        args = parser.parse_args()
        if args.angle_threshold <= 0 or args.angle_threshold > 90:
            parser.error("--angle-threshold must be between 1 and 90")

        return cls(
            angle_threshold=args.angle_threshold,
            debug=args.debug,
        )
