from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta

from .contracts import HeadwaterError
from .data import Observation, timezone_name


FREQUENCIES: dict[str, timedelta] = {
    "h": timedelta(hours=1),
    "D": timedelta(days=1),
    "W": timedelta(weeks=1),
}
SEASONS = {"h": 24, "D": 7, "W": 52, "MS": 12}


def normalise_frequency(value: str) -> str:
    aliases = {"H": "h", "hour": "h", "hourly": "h", "day": "D", "daily": "D",
               "week": "W", "weekly": "W", "month": "MS", "monthly": "MS"}
    result = aliases.get(value, value)
    if result not in {"h", "D", "W", "MS"}:
        raise HeadwaterError("UNSUPPORTED_FREQUENCY", f"Unsupported frequency: {value}")
    return result


def _month_step(left: datetime, right: datetime) -> bool:
    expected_year = left.year + (1 if left.month == 12 else 0)
    expected_month = 1 if left.month == 12 else left.month + 1
    return left.day == right.day == 1 and (right.year, right.month) == (expected_year, expected_month)


def infer_frequency(timestamps: list[datetime]) -> str:
    unique = sorted(set(timestamps))
    if len(unique) < 3:
        raise HeadwaterError("AMBIGUOUS_FREQUENCY", "At least three timestamps are required.")
    if all(_month_step(left, right) for left, right in zip(unique, unique[1:])):
        return "MS"
    counts = Counter(right - left for left, right in zip(unique, unique[1:]))
    step, _ = counts.most_common(1)[0]
    for code, duration in FREQUENCIES.items():
        if step == duration:
            return code
    raise HeadwaterError("AMBIGUOUS_FREQUENCY", "Could not infer a supported regular frequency.")


def next_timestamp(value: datetime, frequency: str) -> datetime:
    if frequency in FREQUENCIES:
        return value + FREQUENCIES[frequency]
    year, month = value.year, value.month + 1
    if month == 13:
        year, month = year + 1, 1
    return value.replace(year=year, month=month, day=1)


def validate_and_group(
    observations: list[Observation], requested_frequency: str | None
) -> tuple[dict[str, list[Observation]], str, str | None]:
    frequency = normalise_frequency(requested_frequency) if requested_frequency else infer_frequency(
        [item.timestamp for item in observations]
    )
    groups: dict[str, list[Observation]] = defaultdict(list)
    for item in observations:
        groups[item.series].append(item)
    inferred: set[str] = set()
    for name, values in groups.items():
        values.sort(key=lambda item: item.timestamp)
        timestamps = [item.timestamp for item in values]
        if len(timestamps) != len(set(timestamps)):
            raise HeadwaterError("DUPLICATE_TIMESTAMPS", f"Series {name} contains duplicate timestamps.")
        if len(timestamps) >= 3:
            inferred.add(infer_frequency(timestamps))
        for left, right in zip(timestamps, timestamps[1:]):
            if next_timestamp(left, frequency) != right:
                raise HeadwaterError(
                    "IRREGULAR_TIME_GRID", f"Series {name} has a missing or irregular period after {left.isoformat()}.",
                    {"series": name, "after": left.isoformat(), "expected": next_timestamp(left, frequency).isoformat()},
                )
    if inferred and inferred != {frequency}:
        raise HeadwaterError(
            "FREQUENCY_MISMATCH", "Requested frequency does not match every series.",
            {"requested": frequency, "inferred": sorted(inferred)},
        )
    zone = timezone_name([item.timestamp for item in observations])
    return dict(groups), frequency, zone

