"""Two reference implementations, and the evidence contract they satisfy.

A benchmark that grades exactly one system, written by that system's authors,
is internal validation whatever the quality of its instruments. What makes
this a benchmark of *point-in-time correctness* rather than a test of Gnomon
is that the thing being graded is a contract any implementation can meet, and
that at least one implementation meeting it is not Gnomon.

The contract is small enough to state in full. An implementation is graded on
two artefacts:

1. a forecast for the horizon, and
2. an **access record** in the shape below, declaring the latest publication
   date it consulted and the fence it applied.

::

    {"kind": "snapshot_access",
     "payload": {"as_of": "<the fence, ISO, or 'latest'>",
                 "known_time_provenance": "recorded" | "assumed",
                 "accesses": [{"entity": ..., "variable": ...,
                               "max_known_time": "<latest publication read>"}]}}

`grade.structural_assertion` reads that record and nothing else, so an
implementation in any language qualifies by emitting it. The record is a
*declaration*: it is only as good as the implementation's honesty about its
own reads, which is why the benchmark also grades the forecast against a
ceiling and why the two reference arms below are useful — one declares a
correct fence and forecasts accordingly, the other declares the fence it
really used, which is no fence at all.

Neither implementation imports Gnomon. Between them they establish that the
instrument discriminates *implementations of point-in-time access*, and that
a correct one is not hard to write — which is the honest framing of what
Gnomon provides: not a capability nobody else has, but one that is easy to
get wrong silently and that this benchmark makes visible.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from . import baselines

#: The strategy both reference arms forecast with. Identical across the two,
#: so the only difference between them is which rows they were willing to
#: read — which is the variable under study.
REFERENCE_STRATEGY = "holt_winters(a=0.5,b=0.05,g=0.05,p=1.0)"


def _forecast_from(history: list[float], horizon: int,
                   season: int) -> list[float]:
    try:
        return baselines.STRATEGIES[REFERENCE_STRATEGY](
            list(history), horizon, season)
    except (ValueError, ZeroDivisionError, ArithmeticError, IndexError):
        return baselines.last_value(list(history), horizon, season)


def _access_record(max_known_time: str, as_of: str) -> list[dict[str, Any]]:
    return [{
        "kind": "snapshot_access",
        "payload": {
            "as_of": as_of,
            "known_time_assumed": False,
            "known_time_provenance": "recorded",
            "accesses": [{"entity": "__default__", "variable": "value",
                          "max_known_time": max_known_time}],
        },
    }]


def point_in_time(rows: list[dict[str, Any]], cutoff: datetime, horizon: int,
                  season: int) -> tuple[list[float], list[dict[str, Any]]]:
    """A correct point-in-time read, in about ten lines.

    Keep only rows published on or before the cutoff; for each timestamp take
    the latest such publication. This is what a bitemporal store does, done
    by hand, and it is here to show that the benchmark's pass condition is
    reachable without the system under test.
    """
    latest: dict[str, tuple[str, float]] = {}
    read: list[str] = []
    for row in rows:
        published = row["published"]
        if datetime.fromisoformat(published) > cutoff:
            continue                      # the fence
        read.append(published)
        stamp = row["timestamp"]
        if stamp not in latest or published > latest[stamp][0]:
            latest[stamp] = (published, float(row["value"]))
    history = [latest[key][1] for key in sorted(latest)]
    evidence = _access_record(max(read), cutoff.isoformat())
    return _forecast_from(history, horizon, season), evidence


def latest_value_join(rows: list[dict[str, Any]], cutoff: datetime,
                      horizon: int, season: int,
                      ) -> tuple[list[float], list[dict[str, Any]]]:
    """The ordinary bug: filter on the timestamp, forget the publication date.

    Restricting to ``timestamp <= cutoff`` looks like a point-in-time read and
    is not one — every revision published after the cutoff is still taken,
    because nothing filtered on when the value became knowable. This is the
    single most common way a backtest leaks, it produces a forecast over
    exactly the right window, and it is invisible to any check that only
    looks at timestamps.

    Its access record is honest about what it did, so the structural
    assertion catches it. An implementation that *lied* in its record would
    not be caught by that instrument — which is the contract's limit, stated
    here rather than discovered later.
    """
    latest: dict[str, tuple[str, float]] = {}
    read: list[str] = []
    for row in rows:
        if datetime.fromisoformat(row["timestamp"]) > cutoff:
            continue                      # the wrong fence: valid time, not known time
        published = row["published"]
        read.append(published)
        stamp = row["timestamp"]
        if stamp not in latest or published > latest[stamp][0]:
            latest[stamp] = (published, float(row["value"]))
    history = [latest[key][1] for key in sorted(latest)]
    evidence = _access_record(max(read), "latest")
    return _forecast_from(history, horizon, season), evidence
