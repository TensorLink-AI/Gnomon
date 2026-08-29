"""Strict deterministic compilation of cited linear relationship text.

This module intentionally understands only complete, mechanically checkable
linear lag specifications. It is a product boundary, not a natural-language
forecaster: anything partial or ambiguous returns ``None`` for a typed caller
rejection or an LLM interpretation lane.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"


def _schedule(text: str, series: str, cutoff: str,
              future_timestamps: list[str]) -> tuple[list[dict[str, Any]],
                                                       list[float]] | None:
    ranges = []
    pattern = re.compile(
        rf"({_NUMBER})\s+from\s+(\d{{4}}-\d{{2}}-\d{{2}})"
        rf"(?:T[^\s,;]+)?\s+to\s+(\d{{4}}-\d{{2}}-\d{{2}})"
        rf"(?:T[^\s,;]+)?", re.I)
    for line in text.splitlines():
        if series.casefold() not in line.casefold():
            continue
        for match in pattern.finditer(line):
            value, start, end = float(match.group(1)), match.group(2), match.group(3)
            if start <= end:
                ranges.append({"start": start, "end": end, "value": value,
                               "source_claim_ids": ["claim-1"]})
    if not ranges:
        return None
    cutoff_day = cutoff[:10]
    historical = [row for row in ranges if row["end"] <= cutoff_day]
    prospective = [row for row in ranges if row["end"] > cutoff_day]
    future = []
    for timestamp in sorted(future_timestamps):
        matches = [row for row in prospective
                   if row["start"] <= timestamp[:10] <= row["end"]]
        if len(matches) != 1:
            return None
        future.append(float(matches[0]["value"]))
    return historical, future


def _exact_terms(text: str, target: str) \
        -> list[tuple[float, str, int]] | None:
    equations = re.finditer(
        r"(?m)^\s*([A-Za-z_]\w*)\s*\^\s*\{\s*t\s*\}\s*=\s*(.+?)\s*$",
        text)
    equation = next((row for row in equations if row.group(1) == target), None)
    lagged = re.compile(
        rf"({_NUMBER})\s*\*\s*([A-Za-z_]\w*)\s*\^\s*"
        r"\{\s*t\s*-\s*(\d+)\s*\}")
    if equation is not None:
        rhs = equation.group(2)
        matches = list(lagged.finditer(rhs))
        remainder = lagged.sub("", rhs)
        remainder = re.sub(
            r"\\?epsilon(?:_\{?\w+\}?|_\w+)?\s*\^\s*\{\s*t\s*\}",
            "", remainder)
        if not matches or re.sub(r"[+\s]", "", remainder):
            return None
        return [(float(row.group(1)), row.group(2), int(row.group(3)))
                for row in matches]
    prose = re.compile(
        r"(?mi)^\s*Parents\s+for\s+(?:variable\s+)?" + re.escape(target)
        + r"\s+at\s+lag\s+(\d+)\s*:\s*\[([^\]]+)\]\s*"
          r"affect\s+the\s+forecast\s+variable\s+as\s+(.+?)\.?\s*$")
    coefficient = re.compile(rf"({_NUMBER})\s*\*\s*([A-Za-z_]\w*)")
    output = []
    rows = list(prose.finditer(text))
    if not rows:
        return None
    for row in rows:
        parents = {item.strip().strip("'\"") for item in row.group(2).split(",")
                   if item.strip()}
        matches = list(coefficient.finditer(row.group(3)))
        if ({item.group(2) for item in matches} != parents
                or re.sub(r"[+\s]", "", coefficient.sub("", row.group(3)))):
            return None
        output.extend((float(item.group(1)), item.group(2), int(row.group(1)))
                      for item in matches)
    return output


def _parent_lags(text: str, target: str) -> list[tuple[str, int]] | None:
    pattern = re.compile(
        r"(?mi)^\s*Parents\s+for\s+variable\s+" + re.escape(target)
        + r"\s+at\s+lag\s+(\d+)\s*:\s*([^\n.]+)\.?\s*$")
    output = []
    for row in pattern.finditer(text):
        lag = int(row.group(1))
        parents = [item.strip() for item in row.group(2).split(",")
                   if item.strip()]
        if lag < 1 or not parents:
            return None
        output.extend((parent, lag) for parent in parents)
    return output or None


def compile_linear_relationship_text(
    text: str, *, target_name: str, cutoff: str,
    future_timestamps: list[str], allowed_driver_names: set[str] | None = None,
) -> tuple[dict[str, Any], str] | None:
    """Return a raw dossier and compilation kind, or ``None`` if ambiguous."""
    source = str(text or "")
    if not source or not future_timestamps or not target_name:
        return None
    try:
        cutoff_dt = datetime.fromisoformat(cutoff)
        normalized_future = []
        for value in future_timestamps:
            parsed = datetime.fromisoformat(str(value))
            if parsed.tzinfo is None and cutoff_dt.tzinfo is not None:
                parsed = parsed.replace(tzinfo=cutoff_dt.tzinfo)
            normalized_future.append(parsed.isoformat())
    except ValueError:
        return None
    future_timestamps = normalized_future
    exact = _exact_terms(source, target_name)
    expression: dict[str, Any]
    drivers: set[str]
    if exact is not None:
        if any(not math.isfinite(value) or lag < 1
               for value, _, lag in exact):
            return None
        unknown = {series for _, series, _ in exact
                   if series != target_name and allowed_driver_names is not None
                   and series not in allowed_driver_names}
        if unknown:
            return None
        ar = [{"lag": lag, "coefficient": value}
              for value, series, lag in exact if series == target_name]
        driver_terms = [{"series": series, "lag": lag, "coefficient": value}
                        for value, series, lag in exact if series != target_name]
        drivers = {row["series"] for row in driver_terms}
        if not ar or not drivers:
            return None
        expression = {
            "op": "recursive_linear", "output_unit": "target_units",
            "intercept": 0.0,
            "autoregressive_terms": sorted(ar, key=lambda row: row["lag"]),
            "driver_terms": sorted(driver_terms,
                                   key=lambda row: (row["series"], row["lag"])),
        }
        kind = "exact_coefficients"
    else:
        topology = _parent_lags(source, target_name)
        if topology is None:
            return None
        unknown = {series for series, _ in topology
                   if series != target_name and allowed_driver_names is not None
                   and series not in allowed_driver_names}
        if unknown:
            return None
        ar_lags = sorted({lag for series, lag in topology if series == target_name})
        grouped: dict[str, set[int]] = {}
        for series, lag in topology:
            if series != target_name:
                grouped.setdefault(series, set()).add(lag)
        drivers = set(grouped)
        if not ar_lags or not drivers:
            return None
        expression = {
            "op": "fit_recursive_linear", "output_unit": "target_units",
            "autoregressive_lags": ar_lags,
            "driver_lags": [{"series": name, "lags": sorted(lags)}
                            for name, lags in sorted(grouped.items())],
        }
        kind = "fitted_topology"
    historical, values = {}, {}
    for driver in sorted(drivers):
        schedule = _schedule(source, driver, cutoff, future_timestamps)
        if schedule is None:
            return None
        historical[driver], future = schedule
        values[driver] = {"values": future, "known_at": cutoff,
                          "source_claim_ids": ["claim-1"]}
    dossier = {
        "events": [], "claims": [{
            "source_span": source, "relation": "unknown",
            "effective_start": future_timestamps[0],
            "effective_end": future_timestamps[-1],
            "mechanism": "complete deterministic linear relationship specification",
            "confidence": 1.0,
        }], "hypotheses": [], "covariate_tables": [],
        "transformations": [{
            "transformation": {
                "known_at": cutoff, "claim_ids": ["claim-1"],
                "lane": "historically_testable", "output_unit": "target_units",
                "expression": expression,
            },
            "units": {"primary": "target_units", **{
                driver: "target_units" for driver in drivers}},
            "series_values": values,
            "historical_series_segments": historical,
        }], "observation_interpretations": [], "effect_proposal": None,
        "forecast_candidate": None,
    }
    return dossier, kind
