"""Source-determined counterfactuals for repaired additive sensor drift."""

from __future__ import annotations

import math
import re
import statistics
from datetime import datetime
from typing import Any

from .models import (ets, last_value, seasonal_naive, theta,
                     window_average)


_ADDITIVE_DRIFT = re.compile(
    r"(?:calibration|measurement) (?:problem|issue) starting from\s+"
    r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}).*?"
    r"additive (?:trend|drift).*?increases by\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+(?:at every hour|per hour)",
    re.I | re.S)
_REPAIR_BOUNDARY = re.compile(
    r"At timestep\s+([^,]+),\s*the sensor was repaired and (?:this )?"
    r"additive (?:trend|drift) will (?:disappear|be removed)", re.I)


def deterministic_additive_drift_claim(
    context_text: str, *, history_start: str, cutoff: str,
) -> dict[str, Any] | None:
    """Extract a fully specified additive measurement-drift claim.

    This is a transcription boundary, not an inference rule.  Both the exact
    rate-bearing mechanism and the exact repair boundary must be present; the
    ordinary counterfactual executable still validates the grid and fits its
    candidate from a copy of pre-cutoff observations.
    """
    match = _ADDITIVE_DRIFT.search(str(context_text or ""))
    repair = _REPAIR_BOUNDARY.search(str(context_text or ""))
    if not match or not repair:
        return None
    start = _timestamp(match.group(1).replace(" ", "T"))
    history_start_dt = _timestamp(history_start)
    cutoff_dt = _timestamp(cutoff)
    if start is not None and start.tzinfo is None and cutoff_dt is not None:
        start = start.replace(tzinfo=cutoff_dt.tzinfo)
    if (start is None or history_start_dt is None or cutoff_dt is None
            or not history_start_dt <= start <= cutoff_dt):
        return None
    return {
        "source_span": str(context_text),
        "relation": "unknown",
        "effective_start": start.isoformat(),
        "effective_end": cutoff_dt.isoformat(),
        "mechanism": (
            "deterministic transcription of an explicit additive measurement "
            "drift and repair boundary; no forecast value inferred"),
        "confidence": 1.0,
        "compiler_binding": "deterministic_additive_drift_fallback",
    }


def _timestamp(value: str, *, timezone=None) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None and timezone is not None:
        parsed = parsed.replace(tzinfo=timezone)
    return parsed


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compile_additive_drift_repair(
    *, context_text: str, claims: list[dict[str, Any]], history: list[float],
    history_timestamps: list[str], future_timestamps: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Compile an exact stated measurement correction into a sealed path.

    This executable is intentionally narrow. It requires an additive rate per
    hour, an exact start, and an explicit statement that repair removes the
    accumulated trend at the forecast boundary. The corrected history is a
    copy; the primary and raw observations are never modified.
    """
    match = _ADDITIVE_DRIFT.search(context_text)
    repair = _REPAIR_BOUNDARY.search(context_text)
    evidence: dict[str, Any] = {
        "status": "not_applicable", "selection_eligible": False,
        "automation_eligible": False, "primary_forecast_unchanged": True,
    }
    if not match or not repair or not claims or not history_timestamps \
            or len(history) != len(history_timestamps) or not future_timestamps:
        return None, evidence
    parsed_history = [_timestamp(value) for value in history_timestamps]
    if any(value is None for value in parsed_history):
        return None, {**evidence, "status": "invalid_history_grid"}
    timezone = parsed_history[0].tzinfo
    start = _timestamp(match.group(1).replace(" ", "T"), timezone=timezone)
    repaired_at = _timestamp(repair.group(1).strip(), timezone=timezone)
    future_start = _timestamp(future_timestamps[0])
    if start is None or repaired_at is None or future_start is None \
            or repaired_at != future_start:
        return None, {**evidence, "status": "repair_boundary_not_exact"}
    deltas = [(right - left).total_seconds() for left, right in
              zip(parsed_history, parsed_history[1:])]
    if not deltas or any(delta <= 0 for delta in deltas):
        return None, {**evidence, "status": "irregular_history_grid"}
    step = statistics.median(deltas)
    if abs(step - 3600.0) > 1.0 or any(abs(delta - step) > 1.0 for delta in deltas):
        return None, {**evidence, "status": "rate_unit_grid_mismatch"}
    rate = float(match.group(2))
    if not math.isfinite(rate) or rate == 0:
        return None, {**evidence, "status": "invalid_stated_rate"}
    corrected = []
    corrected_count = 0
    for value, timestamp in zip(history, parsed_history):
        elapsed = int(round((timestamp - start).total_seconds() / step)) + 1
        adjustment = rate * max(0, elapsed)
        corrected.append(float(value) - adjustment)
        corrected_count += elapsed > 0
    if corrected_count < 8:
        return None, {**evidence, "status": "insufficient_corrected_history"}

    period = 24
    models = {
        "last_value": last_value,
        "seasonal_naive": seasonal_naive,
        "window_average": window_average,
        "theta": theta,
        "ets": ets,
    }
    origins = list(range(max(24, len(corrected) // 3), len(corrected)))
    losses = {name: [] for name in models}
    points = {name: [] for name in models}
    actuals = []
    for origin in origins:
        actuals.append(corrected[origin])
        for name, model in models.items():
            try:
                point = float(model(corrected[:origin], 1, period)[0])
            except (ValueError, ArithmeticError, OverflowError):
                point = math.nan
            points[name].append(point)
            losses[name].append(abs(corrected[origin] - point)
                                if math.isfinite(point) else math.nan)
    eligible = [name for name in models if len(losses[name]) >= 12
                and all(math.isfinite(value) for value in losses[name])]
    if not eligible:
        return None, {**evidence, "status": "insufficient_model_replay"}
    family = min(eligible, key=lambda name: statistics.mean(losses[name]))
    horizon = len(future_timestamps)
    forecast = models[family](corrected, horizon, period)
    residuals = [actual - point for actual, point in
                 zip(actuals, points[family])]
    q10, q50, q90 = (_quantile(residuals, probability)
                     for probability in (.10, .50, .90))
    rows = [{
        "timestamp": timestamp,
        "q10": float(point) + q10,
        "q50": float(point) + q50,
        "q90": float(point) + q90,
    } for timestamp, point in zip(future_timestamps, forecast)]
    evidence = {
        "status": "source_determined_prior_assisted",
        "selection_eligible": False,
        "human_recommendation_eligible": True,
        "automation_eligible": False,
        "primary_forecast_unchanged": True,
        "correction": "subtract_stated_accumulated_additive_drift",
        "stated_rate_per_hour": rate,
        "drift_start": start.isoformat(),
        "repair_boundary": repaired_at.isoformat(),
        "corrected_observations": corrected_count,
        "family": family,
        "families_compared": list(models),
        "expanding_origins": len(origins),
        "candidate_mae": statistics.mean(losses[family]),
        "raw_observations_mutated": False,
        "knowledge_note": (
            "The correction is conditional on the source-stated calibration "
            "mechanism and is not treated as historically known at earlier origins."),
    }
    claim_ids = [str(claim["claim_id"]) for claim in claims
                 if claim.get("claim_id")]
    return {
        "quantiles": rows,
        "rationale": (
            f"Removed the source-stated additive calibration drift from a "
            f"copy of history, then fit {family} on the corrected series."),
        "claim_ids": claim_ids,
        "calibration_replay": evidence,
    }, evidence
