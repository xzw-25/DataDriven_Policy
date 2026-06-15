"""Small timing utilities."""

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def elapsed_ms(result: list[float]) -> Iterator[None]:
    start = perf_counter()
    yield
    result.append((perf_counter() - start) * 1000.0)

