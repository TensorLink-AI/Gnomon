"""Behavioral input-immutability check for engine invocations.

The engines publish a ``primary_forecast_unchanged`` diagnostic, but a
self-reported constant cannot gate immutability — a gate over it is true
by construction whatever the engine does to its inputs.  The benchmarks
instead snapshot every argument, hand the engine the originals, and
compare element-wise afterwards; their gate is "no engine call mutated
its inputs across all cases", observed rather than attested.
"""

from __future__ import annotations

import copy
from typing import Any, Callable


def call_preserving_inputs(fn: Callable[..., Any], /, *args: Any,
                           **kwargs: Any) -> tuple[Any, bool]:
    """Invoke ``fn`` on the original arguments; report input integrity.

    Returns ``(result, unmutated)`` where ``unmutated`` is True only when
    every positional and keyword argument compares equal, element-wise,
    to a deep copy taken before the call.  Arguments must have value
    equality (numbers, strings, lists, dicts, tuples); NaN would compare
    unequal to itself, so callers pass finite series — every generator in
    these benches does.
    """
    snapshot = copy.deepcopy((args, kwargs))
    result = fn(*args, **kwargs)
    return result, (args, kwargs) == snapshot
