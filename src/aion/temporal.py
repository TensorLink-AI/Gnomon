from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta

from .contracts import AionError
from .data import Observation, timezone_name


FREQUENCIES: dict[str, timedelta] = {
    "min": timedelta(minutes=1),
    "5min": timedelta(minutes=5),
    "15min": timedelta(minutes=15),
    "30min": timedelta(minutes=30),
    "h": timedelta(hours=1),
    "D": timedelta(days=1),
    "W": timedelta(weeks=1),
}
# Dominant cycle per frequency: hourly for 1-minute data (a daily cycle of
# 1440 would demand days of history), daily for the coarser intraday steps.
SEASONS = {"min": 60, "5min": 288, "15min": 96, "30min": 48,
           "h": 24, "D": 7, "W": 52, "MS": 12}

FREQUENCY_DESCRIPTIONS = {
    "min": "1 minute", "5min": "5 minutes", "15min": "15 minutes",
    "30min": "30 minutes", "h": "hourly", "D": "daily", "W": "weekly",
    "MS": "month start",
}


def detect_season(values: list[float], frequency: str) -> tuple[int, float, str]:
    """Detect a repeat period from autocorrelation, falling back to frequency.

    Peaks must be both locally maximal and materially correlated.  Restricting
    the search to at least two observed cycles avoids choosing unsupported
    long lags, while lag 1 is excluded because trend commonly dominates it.
    """
    fallback = SEASONS[frequency]
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
               "T": "min", "1T": "min", "1min": "min", "minute": "min", "1m": "min",
               "5T": "5min", "5m": "5min",
               "15T": "15min", "15m": "15min",
               "30T": "30min", "30m": "30min"}
    result = aliases.get(value, value)
    if result not in SEASONS:
        raise AionError(
            "UNSUPPORTED_FREQUENCY",
            f"Unsupported frequency: {value}. Supported: "
            + ", ".join(f"{code} ({FREQUENCY_DESCRIPTIONS[code]})" for code in SEASONS),
            {"supported": sorted(SEASONS)},
        )
    return result


def _month_step(left: datetime, right: datetime) -> bool:
    expected_year = left.year + (1 if left.month == 12 else 0)
    expected_month = 1 if left.month == 12 else left.month + 1
    return left.day == right.day == 1 and (right.year, right.month) == (expected_year, expected_month)


def infer_frequency(timestamps: list[datetime]) -> str:
    unique = sorted(set(timestamps))
    if len(unique) < 3:
        raise AionError("AMBIGUOUS_FREQUENCY", "At least three timestamps are required.")
    if all(_month_step(left, right) for left, right in zip(unique, unique[1:])):
        return "MS"
    counts = Counter(right - left for left, right in zip(unique, unique[1:]))
    step, _ = counts.most_common(1)[0]
    for code, duration in FREQUENCIES.items():
        if step == duration:
            return code
    raise AionError(
        "AMBIGUOUS_FREQUENCY",
        "Could not infer a supported regular frequency. Supported: "
        + ", ".join(f"{code} ({FREQUENCY_DESCRIPTIONS[code]})" for code in SEASONS),
        {"supported": sorted(SEASONS)},
    )


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
            raise AionError("DUPLICATE_TIMESTAMPS", f"Series {name} contains duplicate timestamps.")
        if len(timestamps) >= 3:
            inferred.add(infer_frequency(timestamps))
        for left, right in zip(timestamps, timestamps[1:]):
            if next_timestamp(left, frequency) != right:
                raise AionError(
                    "IRREGULAR_TIME_GRID", f"Series {name} has a missing or irregular period after {left.isoformat()}.",
                    {"series": name, "after": left.isoformat(), "expected": next_timestamp(left, frequency).isoformat()},
                )
    if inferred and inferred != {frequency}:
        raise AionError(
            "FREQUENCY_MISMATCH", "Requested frequency does not match every series.",
            {"requested": frequency, "inferred": sorted(inferred)},
        )
    zone = timezone_name([item.timestamp for item in observations])
    return dict(groups), frequency, zone
