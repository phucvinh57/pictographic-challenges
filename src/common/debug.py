from collections.abc import Generator
from contextlib import contextmanager
from time import perf_counter

_enabled = False
_title = ""
_steps: dict[str, int] = {}
_times: dict[str, float] = {}
_started = 0.0


def configure(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def begin(title: str) -> None:
    global _title, _started
    if not _enabled:
        return
    _title = title
    _steps.clear()
    _times.clear()
    _started = perf_counter()


def count(step: str, amount: int = 1) -> None:
    if not _enabled:
        return
    _steps[step] = _steps.get(step, 0) + amount


@contextmanager
def timed(step: str) -> Generator[None]:
    if not _enabled:
        yield
        return
    start = perf_counter()
    try:
        yield
    finally:
        _times[step] = _times.get(step, 0.0) + perf_counter() - start


def report() -> None:
    if not _enabled:
        return
    elapsed = perf_counter() - _started
    rows = [(step, f"{amount:,}") for step, amount in _steps.items()]
    rows += [(step, f"{seconds * 1000:,.1f} ms") for step, seconds in _times.items()]
    rows += [("total time", f"{elapsed * 1000:,.1f} ms")]

    print(f"[debug] {_title}")
    width = max(len(step) for step, _ in rows)
    for step, value in rows:
        print(f"  {step.ljust(width)} : {value}")
    _steps.clear()
    _times.clear()


__all__ = ["begin", "configure", "count", "report", "timed"]
