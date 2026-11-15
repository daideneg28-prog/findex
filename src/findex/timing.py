"""The @timed decorator: log how long a call took."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable

log = logging.getLogger("findex.timing")


def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Log `<qualname> took N ms` at INFO level after every call (even if it raises).

    functools.wraps copies __name__, __qualname__, __doc__, __module__ and __wrapped__
    from `func` onto the wrapper; without it every decorated function would be named
    "wrapper" in tracebacks, help() and logs.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            log.info("%s took %.2f ms", func.__qualname__, (time.perf_counter() - t0) * 1000)

    return wrapper
