from __future__ import annotations

from .args import get_args

_title = ""
_steps: dict[str, int] = {}


def enabled() -> bool:
    return get_args().debug


def begin(title: str) -> None:
    global _title
    if not enabled():
        return
    _title = title
    _steps.clear()


def count(step: str, amount: int = 1) -> None:
    if not enabled():
        return
    _steps[step] = _steps.get(step, 0) + amount


def report() -> None:
    if not enabled():
        return
    print(f"[debug] {_title}")
    width = max(map(len, _steps), default=0)
    for step, amount in _steps.items():
        print(f"  {step.ljust(width)} : {amount:,}")
    _steps.clear()


__all__ = ["begin", "count", "enabled", "report"]
