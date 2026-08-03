from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

from .contracts import GnomonError
from .data import Observation, timezone_name


FREQUENCIES: dict[str, timedelta] = {
    "s": timedelta(seconds=1),
    "min": timedelta(minutes=1),
    "5min": timedelta(minutes=5),
    "10min": timedelta(minutes=10),
    "15min": timedelta(minutes=15),
    "30min": timedelta(minutes=30),
    "h": timedelta(hours=1),
    "D": timedelta(days=1),
    "W": timedelta(weeks=1),
}
# Dominant cycle per frequency: a minute cycle for 1-second data and an
# hourly cycle for 1-minute data (a daily cycle of 86400 or 1440 would
# demand days of history), daily for the coarser intraday steps.
SEASONS = {"s": 60, "min": 60, "5min": 288, "10min": 144, "15min": 96,
           "30min": 48, "h": 24, "D": 7, "W": 52, "MS": 12}

FREQUENCY_DESCRIPTIONS = {
    "s": "1 second", "min": "1 minute", "5min": "5 minutes",
    "10min": "10 minutes", "15min": "15 minutes", "30min": "30 minutes",
    "h": "hourly", "D": "daily", "W": "weekly", "MS": "month start",
}

#: General frequency codes: any whole-second step shorter than one day,
#: written ``<N>s``/``<N>min``/``<N>h`` (``T`` is the pandas minute alias).
#: Steps of a day or more stay named-only: a fixed 48-hour duration and
#: "every second calendar day" diverge at the first DST transition, and
#: Gnomon refuses to guess which one the caller meant.
_GENERAL_CODE = re.compile(r"^([1-9]\d*)(s|min|h|T)$")
_UNIT_SECONDS = {"s": 1, "min": 60, "T": 60, "h": 3600}
_DAY_SECONDS = 86400


def canonical_code(step: timedelta) -> str | None:
    """The canonical frequency code for a fixed step, or ``None``.

    Named codes win (60 seconds is ``min``, never ``60s``), then the largest
    unit that divides the step evenly, so every representable duration has
    exactly one spelling. ``None`` means the step is not representable: below
    one second, not a whole number of seconds, or a day or more without
    being exactly the calendar step ``D`` or ``W``.
    """
    for code, duration in FREQUENCIES.items():
        if step == duration:
            return code
    total = step.total_seconds()
    if total < 1 or total != int(total) or total >= _DAY_SECONDS:
        return None
    seconds = int(total)
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"


def frequency_step(frequency: str) -> timedelta | None:
    """The fixed duration of one grid step; ``None`` for the calendar-only
    code ``MS``, whose step has no fixed duration."""
    if frequency in FREQUENCIES:
        return FREQUENCIES[frequency]
    found = _GENERAL_CODE.match(frequency)
    if found:
        return timedelta(seconds=int(found.group(1)) * _UNIT_SECONDS[found.group(2)])
    return None


#: Natural cycles a general step's default season is derived from, in the
#: preference order the curated SEASONS table encodes: the daily cycle when
#: it fits (5min -> 288), else hourly (min -> 60, because 1440 daily lags
#: would demand days of history), else a minute cycle (s -> 60). The cap is
#: the largest curated value; a cycle needing more lags falls through to
#: the next shorter one.
_SEASON_CYCLES = (_DAY_SECONDS, 3600, 60)
_MAX_DEFAULT_SEASON = 288


def default_season(frequency: str) -> int:
    """The fallback seasonal period for any supported frequency code.

    Curated codes read the SEASONS table; general codes apply the rule the
    table itself follows. A step that fits no natural cycle (nothing between
    2 and 288 lags) gets 1 — no seasonality assumed — and season detection
    then rests on measured autocorrelation alone.
    """
    if frequency in SEASONS:
        return SEASONS[frequency]
    step = frequency_step(frequency)
    if step is not None:
        for cycle in _SEASON_CYCLES:
            ratio = round(cycle / step.total_seconds())
            if 2 <= ratio <= _MAX_DEFAULT_SEASON:
                return ratio
    return 1


def detect_season(values: list[float], frequency: str) -> tuple[int, float, str]:
    """Detect a repeat period from autocorrelation, falling back to frequency.

    Peaks must be both locally maximal and materially correlated.  Restricting
    the search to at least two observed cycles avoids choosing unsupported
    long lags, while lag 1 is excluded because trend commonly dominates it.
    """
    fallback = default_season(frequency)
    if len(values) < 8:
        return fallback, 0.0, "frequency_default"
    # Remove a least-squares line first: otherwise a trend creates large,
    # slowly decaying autocorrelation that masquerades as seasonality.
    x_mean = (len(values) - 1) / 2
    y_mean = sum(values) / len(values)
    x_var = sum((i - x_mean) ** 2 for i in range(len(values)))
    slope = sum((i - x_mean) * (value - y_mean) for i, value in enumerate(values)) / x_var
    centred = [value - (y_mean + slope * (i - x_mean)) for i, value in enumerate(values)]
    denominator = sum(value * value for value in centred)
    if denominator <= 1e-12:
        return fallback, 0.0, "frequency_default"
    maximum = min(len(values) // 2, max(fallback * 2, 2))
    acf = [0.0]
    for lag in range(1, maximum + 1):
        acf.append(sum(centred[i] * centred[i - lag] for i in range(lag, len(values))) / denominator)
    threshold = max(0.3, 2.0 / len(values) ** 0.5)
    peaks = [lag for lag in range(2, maximum) if acf[lag] >= threshold and acf[lag] > acf[lag - 1] and acf[lag] >= acf[lag + 1]]
    if not peaks:
        return fallback, 0.0, "frequency_default"
    lag = peaks[0]
    return lag, acf[lag], "autocorrelation"


def normalise_frequency(value: str) -> str:
    aliases = {"H": "h", "hour": "h", "hourly": "h", "1h": "h",
               "day": "D", "daily": "D", "1d": "D", "1D": "D",
               "week": "W", "weekly": "W", "1w": "W",
               "month": "MS", "monthly": "MS", "M": "MS", "1M": "MS",
               "S": "s", "1s": "s", "sec": "s", "second": "s",
               "T": "min", "1T": "min", "1min": "min", "minute": "min", "1m": "min",
               "5T": "5min", "5m": "5min",
               "10T": "10min", "10m": "10min",
               "15T": "15min", "15m": "15min",
               "30T": "30min", "30m": "30min"}
    result = aliases.get(value, value)
    if result in SEASONS:
        return result
    step = frequency_step(result)
    if step is not None:
        code = canonical_code(step)
        if code is not None:
            return code
    raise GnomonError(
        "UNSUPPORTED_FREQUENCY",
        f"Unsupported frequency: {value}. Supported: "
        + ", ".join(f"{code} ({FREQUENCY_DESCRIPTIONS[code]})" for code in SEASONS)
        + "; or any regular sub-daily step as <N>s, <N>min, or <N>h "
          "(e.g. 90s, 7min, 2h).",
        {"supported": sorted(SEASONS), "general_pattern": _GENERAL_CODE.pattern},
    )


def _month_step(left: datetime, right: datetime) -> bool:
    expected_year = left.year + (1 if left.month == 12 else 0)
    expected_month = 1 if left.month == 12 else left.month + 1
    return left.day == right.day == 1 and (right.year, right.month) == (expected_year, expected_month)


def infer_frequency(timestamps: list[datetime]) -> str:
    unique = sorted(set(timestamps))
    if len(unique) < 3:
        raise GnomonError("AMBIGUOUS_FREQUENCY", "At least three timestamps are required.")
    if all(_month_step(left, right) for left, right in zip(unique, unique[1:])):
        return "MS"
    counts = Counter(right - left for left, right in zip(unique, unique[1:]))
    step, _ = counts.most_common(1)[0]
    for code, duration in FREQUENCIES.items():
        if step == duration:
            return code
    # Beyond the named grid, inference accepts exactly one shape: a strictly
    # regular series — one unique spacing — at a representable step. A named
    # step tolerates gaps here because the grid validator names them next
    # and repair can fill them; an unusual step with gaps is indistinguishable
    # from a heavier grid with jitter, so it stays a refusal.
    regular = len(counts) == 1
    if regular:
        code = canonical_code(step)
        if code is not None:
            return code
        raise GnomonError(
            "AMBIGUOUS_FREQUENCY",
            f"The series is perfectly regular but its step ({step}) is not a "
            "supported grid: steps must be a whole number of seconds shorter "
            "than one day, or exactly the calendar codes D, W, or MS. "
            "Resample the data to a representable step.",
            {"supported": sorted(SEASONS), "observed_step": str(step),
             "observed_step_seconds": step.total_seconds(), "regular": True},
        )
    raise GnomonError(
        "AMBIGUOUS_FREQUENCY",
        "Could not infer a regular frequency: the spacing between "
        f"consecutive timestamps varies (most common step: {step}, "
        f"{len(counts)} distinct spacings). Pass an explicit frequency — a "
        "named code or any whole-second sub-daily step as <N>s/<N>min/<N>h — "
        "or repair the grid upstream.",
        {"supported": sorted(SEASONS), "observed_step": str(step),
         "distinct_spacings": len(counts), "regular": False},
    )


#: Frequencies whose step is a *calendar* step rather than a fixed
#: duration. On a timezone-aware series in a DST zone, a "day" is 23 or 25
#: hours across a transition, and adding `timedelta(days=1)` lands an hour
#: off — which the grid check then reported as an irregular period, with
#: two repairs that were both wrong (nothing is missing, and snapping would
#: shift every post-transition timestamp).
CALENDAR_FREQUENCIES = frozenset({"D", "W", "MS"})


def next_timestamp(value: datetime, frequency: str) -> datetime:
    """The next point on the grid.

    For `D`/`W`/`MS` on an aware timestamp this is calendar-aware: add the
    step to the wall-clock time and re-normalise the offset, so midnight
    stays midnight across a DST transition instead of drifting by an hour.
    """
    if frequency in CALENDAR_FREQUENCIES and frequency in FREQUENCIES:
        if value.tzinfo is None:
            return value + FREQUENCIES[frequency]
        naive = value.replace(tzinfo=None) + FREQUENCIES[frequency]
        return _relocalise(naive, value)
    step = frequency_step(frequency)
    if step is not None:
        return value + step
    year, month = value.year, value.month + 1
    if month == 13:
        year, month = year + 1, 1
    if value.tzinfo is None:
        return value.replace(year=year, month=month, day=1)
    return _relocalise(
        value.replace(tzinfo=None, year=year, month=month, day=1), value,
    )


def _relocalise(naive: datetime, reference: datetime) -> datetime:
    """Re-attach ``reference``'s zone to a wall-clock time.

    With a `ZoneInfo` this re-derives the offset for the new date, which is
    the whole point: the same wall-clock hour on either side of a DST
    transition has different offsets. With a fixed-offset zone it is a
    no-op, which is also correct — a fixed offset has no transitions.
    """
    zone = reference.tzinfo
    localised = naive.replace(tzinfo=zone)
    # `fold` disambiguates the repeated hour in an autumn transition;
    # normalising through UTC and back re-derives the correct offset.
    try:
        return localised.astimezone(zone)
    except Exception:  # pragma: no cover - exotic tzinfo implementations
        return localised


def is_regular_step(left: datetime, right: datetime, frequency: str) -> bool:
    """Whether ``right`` is the next point after ``left`` on the grid.

    For a calendar frequency on timezone-aware timestamps this compares
    *wall clocks*, not elapsed time. A day in a DST zone is 23 or 25 hours
    across a transition, and the timestamps arrive as fixed offsets — a
    parsed `+00:00` carries no knowledge that the zone is about to change —
    so an instant comparison cannot tell a DST step from a missing row.
    The wall clock can: midnight to midnight is one day either way.
    """
    if frequency == "MS":
        naive_left = left.replace(tzinfo=None)
        naive_right = right.replace(tzinfo=None)
        return _month_step(naive_left, naive_right)
    step = frequency_step(frequency)
    if step is None:
        return next_timestamp(left, frequency) == right
    if frequency in CALENDAR_FREQUENCIES and left.tzinfo is not None:
        return right.replace(tzinfo=None) - left.replace(tzinfo=None) == step
    return right - left == step


def _modal_step_description(timestamps: list[datetime]) -> str:
    """The most common gap between consecutive timestamps, in words."""
    if len(timestamps) < 2:
        return "unknown"
    steps = Counter(right - left for left, right in zip(timestamps, timestamps[1:]))
    step, _ = steps.most_common(1)[0]
    code = canonical_code(step)
    if code is not None:
        return f"{FREQUENCY_DESCRIPTIONS.get(code, code)} ({step})"
    return str(step)


def validate_and_group(
    observations: list[Observation], requested_frequency: str | None,
) -> tuple[dict[str, list[Observation]], str, str | None]:
    frequency = normalise_frequency(requested_frequency) if requested_frequency else infer_frequency(
        [item.timestamp for item in observations]
    )
    groups: dict[str, list[Observation]] = defaultdict(list)
    for item in observations:
        groups[item.series].append(item)
    # Per-series inference is a *consistency* check on a frequency Gnomon
    # chose. When the caller named one, the grid check below is the check,
    # and running inference first meant an explicit `--frequency MS` could
    # never rescue a series whose raw step is not in FREQUENCIES —
    # month-end data failed with AMBIGUOUS_FREQUENCY before the requested
    # frequency was ever applied.
    inferred: dict[str, str] = {}
    if requested_frequency is None:
        for name, values in groups.items():
            timestamps = sorted({item.timestamp for item in values})
            if len(timestamps) >= 3:
                inferred[name] = infer_frequency(timestamps)
        distinct = set(inferred.values())
        if len(distinct) > 1:
            # Blaming the minority series for being irregular describes the
            # symptom; the file mixes frequencies, and that is the finding.
            raise GnomonError(
                "MIXED_SERIES_FREQUENCIES",
                "The input mixes frequencies across series: "
                + "; ".join(
                    f"{name} is {FREQUENCY_DESCRIPTIONS.get(code, code)}"
                    for name, code in sorted(inferred.items())
                )
                + ". Split the file, or pass --frequency to state which grid "
                  "every series is on.",
                {"per_series": dict(sorted(inferred.items())),
                 "distinct": sorted(distinct)},
            )
    for name, values in groups.items():
        values.sort(key=lambda item: item.timestamp)
        timestamps = [item.timestamp for item in values]
        if len(timestamps) != len(set(timestamps)):
            raise GnomonError("DUPLICATE_TIMESTAMPS", f"Series {name} contains duplicate timestamps.")
        for left, right in zip(timestamps, timestamps[1:]):
            if not is_regular_step(left, right, frequency):
                observed = _modal_step_description(timestamps)
                raise GnomonError(
                    "IRREGULAR_TIME_GRID",
                    f"Series {name} has a missing or irregular period after "
                    f"{left.isoformat()}: expected "
                    f"{next_timestamp(left, frequency).isoformat()}, found "
                    f"{right.isoformat()}. The most common step in this series "
                    f"is {observed}.",
                    {"series": name, "after": left.isoformat(),
                     "expected": next_timestamp(left, frequency).isoformat(),
                     "found": right.isoformat(),
                     "frequency": frequency,
                     "modal_step": observed},
                )
    if inferred and set(inferred.values()) != {frequency}:
        raise GnomonError(
            "FREQUENCY_MISMATCH", "Requested frequency does not match every series.",
            {"requested": frequency, "inferred": sorted(set(inferred.values())),
             "per_series": dict(sorted(inferred.items()))},
        )
    zone = timezone_name([item.timestamp for item in observations])
    return dict(groups), frequency, zone
