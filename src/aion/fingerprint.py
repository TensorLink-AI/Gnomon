"""Series fingerprints: the shape of the data, made comparable.

A fingerprint is a small, deterministic summary of a series' character —
length, seasonality, trend, intermittency, noise level — recorded with
every tracked run so realised performance can transfer across datasets:
"which model wins on data *like this*" instead of "which model won on
this exact project". All values are rounded and unit-free (scaled by the
series' own level or robust scale), so fingerprints from different
domains are comparable.
"""

from __future__ import annotations

import json
from statistics import median
from typing import Any

from .operators import _robust_scale


def series_fingerprint(values: list[float], frequency: str | None = None) -> dict[str, Any]:
    """Deterministic, unit-free summary of one series' character."""
    n = len(values)
    fingerprint: dict[str, Any] = {"length": n, "frequency": frequency}
    if n < 4:
        return fingerprint
    level = median(values)
    scale = _robust_scale([value - level for value in values])
    # Trend: OLS slope per period, standardised by the robust scale.
    mean_index = (n - 1) / 2.0
    mean_value = sum(values) / n
    denominator = sum((index - mean_index) ** 2 for index in range(n))
    slope = sum((index - mean_index) * (value - mean_value)
                for index, value in enumerate(values)) / denominator
    fingerprint["trend"] = round(slope / scale, 4)
    # Noise ratio: robust scale relative to the level (scale-free volatility).
    fingerprint["noise_ratio"] = round(scale / abs(level), 4) if abs(level) > 1e-12 else None
    # Intermittency: share of exact zeros (demand-style series).
    fingerprint["intermittency"] = round(sum(1 for value in values if value == 0.0) / n, 4)
    # Direction-change rate: how often the series reverses (1.0 = sawtooth).
    changes = sum(
        1 for index in range(2, n)
        if (values[index] - values[index - 1]) * (values[index - 1] - values[index - 2]) < 0
    )
    fingerprint["direction_change_rate"] = round(changes / max(1, n - 2), 4)
    if frequency is not None:
        from .temporal import detect_season
        period, strength, source = detect_season(values, frequency)
        fingerprint["season_period"] = period
        fingerprint["season_strength"] = round(strength, 4)
        fingerprint["season_source"] = source
    return fingerprint


def fingerprint_json(values: list[float], frequency: str | None = None) -> str:
    return json.dumps(series_fingerprint(values, frequency), sort_keys=True)


def fingerprint_distance(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    """Comparable-shape distance between two fingerprints (lower = nearer).

    Uses the unit-free numeric dimensions plus a log-length term; returns
    None when either fingerprint lacks the numeric dimensions (too short).
    """
    import math
    dimensions = ("trend", "noise_ratio", "intermittency", "direction_change_rate",
                  "season_strength")
    terms = []
    for key in dimensions:
        left, right = a.get(key), b.get(key)
        if left is None or right is None:
            continue
        terms.append((float(left) - float(right)) ** 2)
    if not terms:
        return None
    length_a, length_b = a.get("length"), b.get("length")
    if length_a and length_b:
        terms.append((math.log10(length_a) - math.log10(length_b)) ** 2)
    return round(sum(terms) ** 0.5, 4)
