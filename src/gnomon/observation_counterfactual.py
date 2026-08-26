"""Fitted counterfactuals for cited historical observation corruption.

The caller supplies a validated exclusion mask.  This module never decides
which observations are corrupt and never edits the source series.  It fits a
small, fixed family on earlier retained observations and evaluates one-step
conditional predictions only at retained origins.  The protocol is generic to
availability, stockout, censoring, and reporting-gap problems; it has no task,
domain, or benchmark labels.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from .models import drift, ets, last_value, linear_trend, theta, window_average
from .models import croston_sba

MIN_REPLAY_ORIGINS = 12
ADMISSION_MARGIN = 0.10


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _family_point(
    family: str, retained: list[float], excluded: list[float],
) -> float | None:
    if len(retained) < 3:
        return None
    if family == "robust_level":
        return statistics.median(retained)
    if family == "rebased_croston_sba":
        if len(excluded) < 2:
            return None
        zero_location = statistics.median(excluded)
        rebased = [max(0.0, value - zero_location) for value in retained]
        try:
            return zero_location + croston_sba(rebased, 1, 1)[0]
        except ValueError:
            return None
    raise ValueError(f"unknown counterfactual family {family!r}")


def fit_observation_counterfactual(
    history: list[float], exclusion_mask: list[bool],
    future_timestamps: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Fit and replay a fixed two-family conditional counterfactual.

    Replay targets are observations the validated mask considers unaffected.
    At every origin, candidates see only earlier unaffected observations and
    the baseline sees only the immediately preceding raw observation.  This
    tests the actual product question: once the disruption ends, is filtering
    the contaminated history better than carrying the raw last value forward?
    """
    if len(history) != len(exclusion_mask):
        raise ValueError("exclusion_mask must align with history")
    retained_all = [float(value) for value, excluded in
                    zip(history, exclusion_mask) if not excluded]

    def empirical_fallback(evidence: dict[str, Any]):
        if len(retained_all) < 3:
            return None, evidence
        q10 = _quantile(retained_all, .10)
        q50 = _quantile(retained_all, .50)
        q90 = _quantile(retained_all, .90)
        return {
            "quantiles": [{"timestamp": timestamp, "q10": q10,
                           "q50": q50, "q90": q90}
                          for timestamp in future_timestamps],
            "rationale": (
                "Visible empirical counterfactual from retained pre-cutoff "
                "observations; replay did not earn recommendation authority."),
            "conditional_replay": evidence,
        }, evidence

    families = ("robust_level", "rebased_croston_sba")
    raw_comparators = {
        "last_value": last_value,
        "window_average": window_average,
        "drift": drift,
        "linear_trend": linear_trend,
        "theta": theta,
        "ets": ets,
    }
    losses: dict[str, list[float]] = {family: [] for family in families}
    errors: dict[str, list[float]] = {family: [] for family in families}
    baseline_losses: list[float] = []
    comparator_losses: dict[str, list[float]] = {
        name: [] for name in raw_comparators}
    origin_positions: list[int] = []
    for origin in range(1, len(history)):
        if exclusion_mask[origin]:
            continue
        retained = [float(value) for value, excluded in
                    zip(history[:origin], exclusion_mask[:origin])
                    if not excluded]
        excluded_values = [float(value) for value, excluded in
                           zip(history[:origin], exclusion_mask[:origin])
                           if excluded]
        if len(retained) < 8:
            continue
        actual = float(history[origin])
        predictions = {
            family: _family_point(family, retained, excluded_values)
            for family in families}
        if all(value is None for value in predictions.values()):
            continue
        origin_positions.append(origin)
        baseline_losses.append(abs(actual - float(history[origin - 1])))
        for name, model in raw_comparators.items():
            try:
                prediction = float(model(
                    [float(value) for value in history[:origin]], 1, 1)[0])
                comparator_losses[name].append(abs(actual - prediction))
            except (ValueError, ArithmeticError, OverflowError):
                comparator_losses[name].append(math.nan)
        for family, prediction in predictions.items():
            if prediction is None:
                losses[family].append(math.nan)
                errors[family].append(math.nan)
            else:
                losses[family].append(abs(actual - prediction))
                errors[family].append(actual - prediction)

    complete = {
        family: [value for value in values if math.isfinite(value)]
        for family, values in losses.items()}
    eligible = [family for family in families
                if len(complete[family]) == len(baseline_losses)
                and len(complete[family]) >= MIN_REPLAY_ORIGINS]
    if not eligible or len(baseline_losses) < MIN_REPLAY_ORIGINS:
        return empirical_fallback({
            "status": "insufficient_replay",
            "origins": len(baseline_losses),
            "minimum_origins": MIN_REPLAY_ORIGINS,
            "selection_eligible": False,
        })
    family = min(eligible, key=lambda name: statistics.mean(complete[name]))
    candidate_mae = statistics.mean(complete[family])
    baseline_mae = statistics.mean(baseline_losses)
    complete_comparators = {
        name: values for name, values in comparator_losses.items()
        if len(values) == len(baseline_losses)
        and all(math.isfinite(value) for value in values)}
    strongest_name = min(
        complete_comparators,
        key=lambda name: statistics.mean(complete_comparators[name]))
    strongest_losses = complete_comparators[strongest_name]
    strongest_mae = statistics.mean(strongest_losses)
    block_wins = 0
    boundaries = [0, len(baseline_losses) // 3,
                  2 * len(baseline_losses) // 3, len(baseline_losses)]
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            continue
        candidate_block = statistics.mean(losses[family][left:right])
        comparator_block = statistics.mean(strongest_losses[left:right])
        if candidate_block < comparator_block:
            block_wins += 1
    admitted = (
        strongest_mae > 0
        and candidate_mae <= strongest_mae * (1 - ADMISSION_MARGIN)
        and block_wins >= 2)

    retained = [float(value) for value, excluded in zip(history, exclusion_mask)
                if not excluded]
    excluded_values = [float(value) for value, excluded in
                       zip(history, exclusion_mask) if excluded]
    point = _family_point(family, retained, excluded_values)
    if point is None:
        return empirical_fallback({
            "status": "fit_failed", "origins": len(baseline_losses),
            "selection_eligible": False,
        })
    finite_errors = [value for value in errors[family] if math.isfinite(value)]
    q10_error = _quantile(finite_errors, .10)
    q50_error = _quantile(finite_errors, .50)
    q90_error = _quantile(finite_errors, .90)
    rows = [{
        "timestamp": timestamp,
        "q10": point + q10_error,
        "q50": point + q50_error,
        "q90": point + q90_error,
    } for timestamp in future_timestamps]
    evidence = {
        "status": "admitted" if admitted else "not_admitted",
        "scheme": "expanding_origin_unaffected_targets",
        "family": family,
        "families_compared": list(families),
        "origins": len(baseline_losses),
        "candidate_mae": candidate_mae,
        "raw_last_value_mae": baseline_mae,
        "strongest_raw_comparator": strongest_name,
        "strongest_raw_mae": strongest_mae,
        "relative_improvement": (
            (strongest_mae - candidate_mae) / strongest_mae
            if strongest_mae > 0 else None),
        "required_margin": ADMISSION_MARGIN,
        "chronological_block_wins": block_wins,
        "required_block_wins": 2,
        "selection_eligible": admitted,
        "uses_future_observations": False,
        "knowledge_note": (
            "Retrospective conditional replay validates a fixed interpretation "
            "at the receipt cutoff; it does not backdate when the source was known."),
    }
    return {
        "quantiles": rows,
        "rationale": (
            f"{family} fitted on {len(retained)} retained pre-cutoff "
            "observations with uncertainty from expanding-origin residuals."),
        "conditional_replay": evidence,
    }, evidence
