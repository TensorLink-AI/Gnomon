"""Suite registry. Importing the package registers every built-in suite."""

from __future__ import annotations

from .base import (
    LoadOptions,
    ScoringContext,
    Suite,
    available,
    describe_all,
    get_suite,
    normalise_conditions,
    register,
)
from . import timeseriesexam as _timeseriesexam  # noqa: F401  (registration)
from . import tsaia as _tsaia                    # noqa: F401  (registration)
from . import tsqa as _tsqa                      # noqa: F401  (registration)

__all__ = [
    "LoadOptions",
    "ScoringContext",
    "Suite",
    "available",
    "describe_all",
    "get_suite",
    "normalise_conditions",
    "register",
]
