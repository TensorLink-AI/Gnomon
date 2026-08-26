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
from datetime import datetime
from typing import Any

from .models import drift, ets, last_value, linear_trend, theta, window_average
from .models import croston_sba

MIN_REPLAY_ORIGINS = 12
ADMISSION_MARGIN = 0.10
MIN_CALIBRATION_ERRORS = 8


def _pinball(actual: float, forecast: float, probability: float) -> float:
    error = actual - forecast
    return max(probability * error, (probability - 1.0) * error)


def _replayed_distribution_losses(
    actuals: list[float], points: list[float],
) -> list[float]:
    """Score fold-safe q10/q50/q90 paths using only earlier residuals."""
    losses: list[float] = []
    prior_errors: list[float] = []
    for actual, point in zip(actuals, points):
        if len(prior_errors) >= MIN_CALIBRATION_ERRORS:
            forecasts = (
                point + _quantile(prior_errors, .10),
                point + _quantile(prior_errors, .50),
                point + _quantile(prior_errors, .90),
            )
            losses.append(2.0 * sum(
                _pinball(actual, forecast, probability)
                for forecast, probability in zip(forecasts, (.10, .50, .90))) / 3.0)
        prior_errors.append(actual - point)
    return losses


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


def _daily_period(timestamps: list[str] | None) -> int | None:
    if not timestamps or len(timestamps) < 3:
        return None
    try:
        parsed = [datetime.fromisoformat(value) for value in timestamps]
    except ValueError:
        return None
    deltas = [(right - left).total_seconds()
              for left, right in zip(parsed, parsed[1:])]
    if not deltas or any(delta <= 0 for delta in deltas):
        return None
    step = statistics.median(deltas)
    if any(abs(delta - step) > max(1e-6, step * 1e-6) for delta in deltas):
        return None
    candidate = round(86400.0 / step)
    if not 2 <= candidate <= 168 or abs(candidate * step - 86400.0) > 1.0:
        return None
    return candidate


def _seasonal_phase_point(
    history: list[float], exclusion_mask: list[bool], *, period: int,
    next_index: int, estimator: str = "median",
) -> float | None:
    retained = [float(value) for value, excluded in
                zip(history, exclusion_mask) if not excluded]
    if len(retained) < 3:
        return None
    phase = next_index % period
    phase_values = [float(value) for index, (value, excluded) in enumerate(
        zip(history, exclusion_mask)) if not excluded and index % period == phase]
    # Fold-safe cold start: the family is still one fixed executable, using
    # the earlier retained level until this phase has been observed twice.
    if len(phase_values) >= 2:
        return (phase_values[-1] if estimator == "last"
                else statistics.median(phase_values))
    # A recurring outage can leave a future clock phase wholly unobserved.
    # Interpolate the disclosed daily profile from the nearest observed phases;
    # this is visible prior assistance, and replay on observed phases still
    # determines whether the executable may lead.
    phase_levels: dict[int, float] = {}
    for candidate_phase in range(period):
        values = [float(value) for index, (value, excluded) in enumerate(
                  zip(history, exclusion_mask))
                  if not excluded and index % period == candidate_phase]
        if len(values) >= 2:
            phase_levels[candidate_phase] = (
                values[-1] if estimator == "last" else statistics.median(values))
    if len(phase_levels) < 2:
        return statistics.median(retained)
    before = min(phase_levels, key=lambda item: (phase - item) % period)
    after = min(phase_levels, key=lambda item: (item - phase) % period)
    left_distance = (phase - before) % period
    right_distance = (after - phase) % period
    if left_distance + right_distance == 0:
        return phase_levels[before]
    weight = left_distance / (left_distance + right_distance)
    return phase_levels[before] + weight * (
        phase_levels[after] - phase_levels[before])


def fit_observation_counterfactual(
    history: list[float], exclusion_mask: list[bool],
    future_timestamps: list[str], history_timestamps: list[str] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Fit and replay a small fixed-family conditional counterfactual.

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

    daily_period = _daily_period(history_timestamps)
    families = ("robust_level", "rebased_croston_sba", *(
        ("seasonal_phase_median", "seasonal_phase_last")
        if daily_period is not None else ()))
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
    candidate_points: dict[str, list[float]] = {family: [] for family in families}
    comparator_points: dict[str, list[float]] = {
        name: [] for name in raw_comparators}
    replay_actuals: list[float] = []
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
            family: (_seasonal_phase_point(
                [float(value) for value in history[:origin]],
                list(exclusion_mask[:origin]), period=daily_period,
                next_index=origin,
                estimator="last" if family == "seasonal_phase_last" else "median")
                if family in {"seasonal_phase_median", "seasonal_phase_last"}
                and daily_period is not None
                else _family_point(family, retained, excluded_values))
            for family in families}
        if all(value is None for value in predictions.values()):
            continue
        origin_positions.append(origin)
        replay_actuals.append(actual)
        baseline_losses.append(abs(actual - float(history[origin - 1])))
        for name, model in raw_comparators.items():
            try:
                prediction = float(model(
                    [float(value) for value in history[:origin]], 1, 1)[0])
                comparator_losses[name].append(abs(actual - prediction))
                comparator_points[name].append(prediction)
            except (ValueError, ArithmeticError, OverflowError):
                comparator_losses[name].append(math.nan)
                comparator_points[name].append(math.nan)
        for family, prediction in predictions.items():
            if prediction is None:
                losses[family].append(math.nan)
                errors[family].append(math.nan)
            else:
                losses[family].append(abs(actual - prediction))
                errors[family].append(actual - prediction)
                candidate_points[family].append(prediction)
            if prediction is None:
                candidate_points[family].append(math.nan)

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
    probabilistic_losses = {
        family: _replayed_distribution_losses(
            replay_actuals, candidate_points[family])
        for family in eligible
    }
    eligible = [family for family in eligible
                if len(probabilistic_losses[family]) >= MIN_REPLAY_ORIGINS
                and all(math.isfinite(value)
                        for value in probabilistic_losses[family])]
    if not eligible:
        return empirical_fallback({
            "status": "insufficient_probabilistic_replay",
            "origins": len(baseline_losses),
            "probabilistic_origins": max(
                (len(values) for values in probabilistic_losses.values()),
                default=0),
            "minimum_origins": MIN_REPLAY_ORIGINS,
            "selection_eligible": False,
        })
    family = min(eligible, key=lambda name: statistics.mean(
        probabilistic_losses[name]))
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
    probabilistic_comparators = {
        name: _replayed_distribution_losses(replay_actuals, points)
        for name, points in comparator_points.items()
        if all(math.isfinite(value) for value in points)}
    probabilistic_comparators = {
        name: values for name, values in probabilistic_comparators.items()
        if len(values) == len(probabilistic_losses[family])
        and all(math.isfinite(value) for value in values)}
    strongest_probabilistic_name = min(
        probabilistic_comparators,
        key=lambda name: statistics.mean(probabilistic_comparators[name]))
    strongest_probabilistic_losses = probabilistic_comparators[
        strongest_probabilistic_name]
    candidate_probabilistic_loss = statistics.mean(
        probabilistic_losses[family])
    strongest_probabilistic_loss = statistics.mean(
        strongest_probabilistic_losses)
    block_wins = 0
    distribution_origins = len(probabilistic_losses[family])
    boundaries = [0, distribution_origins // 3,
                  2 * distribution_origins // 3, distribution_origins]
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            continue
        candidate_block = statistics.mean(
            probabilistic_losses[family][left:right])
        comparator_block = statistics.mean(
            strongest_probabilistic_losses[left:right])
        if candidate_block < comparator_block:
            block_wins += 1
    admitted = (
        strongest_mae > 0
        and candidate_mae <= strongest_mae * (1 - ADMISSION_MARGIN)
        and strongest_probabilistic_loss > 0
        and candidate_probabilistic_loss <= (
            strongest_probabilistic_loss * (1 - ADMISSION_MARGIN))
        and block_wins >= 2)
    human_recommendation_eligible = (
        strongest_mae > 0 and candidate_mae < strongest_mae
        and strongest_probabilistic_loss > 0
        and candidate_probabilistic_loss < strongest_probabilistic_loss
        and block_wins >= 2)

    retained = [float(value) for value, excluded in zip(history, exclusion_mask)
                if not excluded]
    excluded_values = [float(value) for value, excluded in
                       zip(history, exclusion_mask) if excluded]
    points = ([
        _seasonal_phase_point(
            [float(value) for value in history], list(exclusion_mask),
            period=daily_period, next_index=len(history) + step,
            estimator="last" if family == "seasonal_phase_last" else "median")
        for step in range(len(future_timestamps))
    ] if family in {"seasonal_phase_median", "seasonal_phase_last"}
    and daily_period is not None else [
        _family_point(family, retained, excluded_values)
        for _ in future_timestamps])
    if any(point is None for point in points):
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
        "q10": float(point) + q10_error,
        "q50": float(point) + q50_error,
        "q90": float(point) + q90_error,
    } for timestamp, point in zip(future_timestamps, points)]
    evidence = {
        "status": "admitted" if admitted else "not_admitted",
        "scheme": "expanding_origin_unaffected_targets",
        "family": family,
        "families_compared": list(families),
        "daily_period_steps": daily_period,
        "origins": len(baseline_losses),
        "candidate_mae": candidate_mae,
        "raw_last_value_mae": baseline_mae,
        "strongest_raw_comparator": strongest_name,
        "strongest_raw_mae": strongest_mae,
        "probabilistic_metric": "mean_q10_q50_q90_pinball_v1",
        "probabilistic_origins": distribution_origins,
        "candidate_probabilistic_loss": candidate_probabilistic_loss,
        "strongest_probabilistic_comparator": strongest_probabilistic_name,
        "strongest_probabilistic_loss": strongest_probabilistic_loss,
        "probabilistic_relative_improvement": (
            (strongest_probabilistic_loss - candidate_probabilistic_loss)
            / strongest_probabilistic_loss
            if strongest_probabilistic_loss > 0 else None),
        "relative_improvement": (
            (strongest_mae - candidate_mae) / strongest_mae
            if strongest_mae > 0 else None),
        "required_margin": ADMISSION_MARGIN,
        "chronological_block_wins": block_wins,
        "required_block_wins": 2,
        "selection_eligible": admitted,
        "human_recommendation_eligible": human_recommendation_eligible,
        "authority_note": (
            "The lower human-facing gate requires directional wins under both "
            "metrics and two chronological blocks. It never upgrades support "
            "or automation; strict admission still requires the 10% margin."),
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
