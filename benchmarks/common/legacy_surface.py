"""Explicit serial access for historical evaluated-tool benchmarks.

These instruments measure the retained full legacy protocol, not the current
default session. Preserve this distinction in reports; never infer agent uplift
for the new default from these results. Legacy selection uses process environment,
so these calls serialize and restore it. Use subprocess sessions for parallel arms.
"""

from contextlib import contextmanager
import os
from threading import RLock

_lock = RLock()


@contextmanager
def full_profile():
    with _lock:
        previous = os.environ.get("GNOMON_MCP_PROFILE")
        os.environ["GNOMON_MCP_PROFILE"] = "full"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("GNOMON_MCP_PROFILE", None)
            else:
                os.environ["GNOMON_MCP_PROFILE"] = previous


def runner_for(name):
    from gnomon.toolspec import runner_for as legacy_runner
    with full_profile():
        if legacy_runner(name) is None:
            return None

    def call(arguments):
        with full_profile():
            return legacy_runner(name)(arguments)
    return call
