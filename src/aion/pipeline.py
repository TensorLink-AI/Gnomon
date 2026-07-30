"""Named stages of the forecast pipeline.

``runtime.forecast()`` is an orchestrator over these stages; each stage has
explicit inputs and outputs so later phases can rebind them (snapshot-backed
loading, plan execution) without touching the numerics. This module is a
behaviour-preserving extraction — the golden-artifact tests pin its output
byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .context import ContextEvent
from .context_eval import CONTEXT_MODEL_NAME, assess_context
from .contracts import DataSchema, Evidence
from .covariates import CovariateDataset, assess_covariates
from .data import Observation, load_observations
from .evaluation import Evaluation, evaluate, interval_bounds, quantile
from .models import MODELS, predict
from .temporal import detect_season, next_timestamp, validate_and_group


@dataclass(frozen=True)
class LoadedDataset:
    """Output of the load stage: validated observations grouped by series."""

    source_fingerprint: str
    columns: list[str]
    groups: dict[str, list[Observation]]
    frequency: str
    timezone: str | None
    schema: DataSchema


@dataclass
class SeriesState:
    """Working state for one series as it moves through the stages."""

    name: str
    values: list[float]
    timestamps: list[datetime]
    future_timestamps: list[datetime]
    season: int
    assessment: Evaluation | None = None
    selected_model: str | None = None
    points: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    coverage: float | None = None
    warnings: list[str] = field(default_factory=list)
    context_public: dict[str, object] | None = None
    covariate_public: dict[str, object] | None = None
    evidence: list[Evidence] = field(default_factory=list)


def load_stage(
    input_path: str,
    *,
    time_column: str,
    target_column: str,
    series_column: str | None,
    frequency: str | None,
) -> LoadedDataset:
    observations, source_fingerprint, columns = load_observations(
        input_path, time_column, target_column, series_column
    )
    groups, resolved_frequency, zone = validate_and_group(observations, frequency)
    schema = DataSchema(time_column, target_column, series_column, resolved_frequency, zone)
    return LoadedDataset(source_fingerprint, columns, groups, resolved_frequency, zone, schema)


def horizon_stage(
    name: str,
    items: list[Observation],
    *,
    horizon: int,
    frequency: str,
    seasonal_period: int | None,
) -> SeriesState:
    """Fix the future timestamp grid and the seasonal period for one series."""
    values = [item.value for item in items]
    timestamps = [item.timestamp for item in items]
    future_timestamps: list[datetime] = []
    timestamp = timestamps[-1]
    for _ in range(horizon):
        timestamp = next_timestamp(timestamp, frequency)
        future_timestamps.append(timestamp)
    detected_season, _, _ = detect_season(values, frequency)
    season = seasonal_period or detected_season
    return SeriesState(name, values, timestamps, future_timestamps, season)


def evaluate_stage(
    state: SeriesState,
    *,
    horizon: int,
    minimum_baseline_improvement: float,
    frequency: str,
    config: Any,
    strict_abstention: bool,
) -> None:
    """Separated rolling evaluation: selection folds, calibration fold, test fold."""
    assessment = evaluate(
        state.values, horizon, state.season, minimum_baseline_improvement,
        frequency=frequency,
        config=config,
        strict_abstention=strict_abstention,
    )
    state.assessment = assessment
    state.selected_model = assessment.selected_model
    state.coverage = assessment.coverage
    state.warnings = list(assessment.warnings)
    state.residuals = assessment.residuals


def predict_stage(
    state: SeriesState,
    *,
    horizon: int,
    frequency: str,
    selection_strategy: str,
) -> None:
    """Produce the final point forecast from the selected model, with the
    ensemble path and the TSFM sandbox/in-process fallback chain."""
    assessment = state.assessment
    if not (assessment and assessment.supported and assessment.selected_model):
        return
    values, season = state.values, state.season
    if assessment.selected_model == "ensemble" or selection_strategy == "ensemble":
        from .ensemble import compute_ensemble_forecast
        forecasts = {}
        for name in MODELS:
            try:
                forecasts[name] = predict(name, values, horizon, season)
            except ValueError:
                pass
        state.points = compute_ensemble_forecast(forecasts, assessment.selection_scores,
                                                 strategy="weighted_mean", last_observed=values[-1])
        state.selected_model = "ensemble"
        # Pool out-of-sample residuals from every eligible model so
        # ensemble intervals reflect both data noise and model spread.
        holdout = min(horizon, max(1, len(values) // 4))
        origin = len(values) - holdout
        pooled: list[float] = []
        for name in MODELS:
            try:
                prediction = predict(name, values[:origin], holdout, season)
                pooled.extend(a - p for a, p in zip(values[origin:], prediction))
            except ValueError:
                pass
        if pooled:
            state.residuals = pooled
    elif assessment.selected_model in MODELS:
        state.points = predict(assessment.selected_model, values, horizon, season)
    else:
        # TSFM selected — use the adapter for the final forecast.
        # Try sandbox first, then in-process.
        from .tsfm import get_tsfm, TSFMError, TSFMUnavailable
        from .tsfm_sandbox import sandbox_tsfm_candidates, sandbox_available_tsfms
        try:
            if assessment.selected_model in sandbox_available_tsfms():
                adapters = sandbox_tsfm_candidates(
                    requested=[assessment.selected_model],
                    frequency=frequency,
                )
            else:
                adapters = []
            if adapters:
                adapter = adapters[0]
            else:
                adapter = get_tsfm(assessment.selected_model)
                if hasattr(adapter, '_frequency'):
                    adapter._frequency = frequency
            state.points = adapter.predict(values, horizon, season)
        except (TSFMError, TSFMUnavailable, Exception) as exc:
            import logging
            logging.getLogger(__name__).warning(
                "TSFM %s failed during final forecast, falling back to %s: %s",
                assessment.selected_model, assessment.strongest_baseline, exc,
            )
            state.selected_model = assessment.strongest_baseline
            state.points = predict(state.selected_model, values, horizon, season)


def multivariate_stage(
    state: SeriesState,
    multivariate_points: dict[str, list[float]],
    multivariate_warnings: dict[str, str],
) -> None:
    """Override the univariate forecast with the VAR forecast when one exists."""
    if state.name in multivariate_points:
        state.points = multivariate_points[state.name]
        state.selected_model = "var"
        state.warnings.append(multivariate_warnings[state.name])


def context_stage(
    state: SeriesState,
    context_events: list[ContextEvent],
    *,
    horizon: int,
    minimum_baseline_improvement: float,
) -> None:
    """Leakage-safe context-event ablation; admitted events replace the forecast."""
    context = assess_context(
        state.values, state.timestamps, state.future_timestamps, context_events,
        state.name, horizon, state.season, minimum_baseline_improvement,
        state.assessment,
    )
    state.context_public = context.to_public_dict()
    state.evidence.append(Evidence(
        f"context_ablation:{state.name}", "context_ablation", state.name,
        state.context_public,
    ))
    if context.admitted:
        state.selected_model = CONTEXT_MODEL_NAME
        state.points = context.points
        state.residuals = context.residuals
        state.coverage = context.coverage
        state.warnings = list(context.warnings)


def covariate_stage(
    state: SeriesState,
    covariates: CovariateDataset,
    *,
    horizon: int,
    minimum_baseline_improvement: float,
) -> None:
    """Leakage-safe covariate ablation; admitted covariates replace the forecast."""
    covariate_assessment = assess_covariates(
        state.values, state.timestamps, state.future_timestamps, covariates,
        state.name, horizon, state.season, minimum_baseline_improvement,
        state.assessment,
    )
    state.covariate_public = covariate_assessment.to_public_dict()
    state.evidence.append(Evidence(
        f"covariate_ablation:{state.name}", "covariate_ablation", state.name,
        {**state.covariate_public, "source_path": covariates.path,
         "source_fingerprint": covariates.fingerprint},
    ))
    if covariate_assessment.admitted:
        state.selected_model = "covariate_linear"
        state.points = covariate_assessment.points
        state.residuals = covariate_assessment.residuals
        state.coverage = covariate_assessment.coverage
        state.warnings = list(covariate_assessment.warnings)


def threshold_analysis_stage(
    threshold: float,
    rows: list[dict[str, object]],
    points: list[float],
    residuals: list[float],
    residual_quantiles: dict[float, float],
) -> dict[str, object]:
    """Empirical threshold-crossing analysis from the pooled backtest
    residuals, recentred and widened exactly like the published intervals."""
    centre_shift = residual_quantiles[0.5]
    probabilities: list[float] = []
    for step, point in enumerate(points, 1):
        scale = step ** 0.5
        above = sum(
            1 for residual in residuals
            if point + centre_shift + (residual - centre_shift) * scale > threshold
        )
        probabilities.append(round(above / len(residuals), 4))

    def first_timestamp(condition) -> str | None:
        for row in rows:
            if condition(row):
                return str(row["timestamp"])
        return None

    return {
        "value": threshold,
        "probability_above": probabilities,
        "first_timestamp_point_above": first_timestamp(lambda row: row["point"] > threshold),
        "first_timestamp_interval_above": first_timestamp(lambda row: row["q90"] > threshold),
        "first_timestamp_point_below": first_timestamp(lambda row: row["point"] < threshold),
        "first_timestamp_interval_below": first_timestamp(lambda row: row["q10"] < threshold),
        "basis": "empirical probabilities from pooled backtest residuals with sqrt-horizon widening",
    }


def interval_stage(
    state: SeriesState,
    *,
    threshold: float | None,
) -> tuple[list[dict[str, object]], str, dict[str, object] | None]:
    """Residual-quantile intervals, the support status, and threshold analysis."""
    assessment = state.assessment
    rows: list[dict[str, object]] = []
    threshold_analysis: dict[str, object] | None = None
    support = "unsupported"
    if assessment and assessment.supported and state.points:
        residual_quantiles = {probability: quantile(state.residuals, probability) for probability in (0.1, 0.5, 0.9)}
        for step, (timestamp, point) in enumerate(zip(state.future_timestamps, state.points), 1):
            q10, q50, q90 = interval_bounds(point, residual_quantiles, step)
            rows.append({
                "timestamp": timestamp.isoformat(), "point": point,
                "q10": q10, "q50": q50, "q90": q90,
            })
        support = "degraded" if assessment.degraded else ("supported_ensemble" if state.selected_model == "ensemble" else ("weakly_supported" if state.warnings else "supported"))
        if threshold is not None:
            threshold_analysis = threshold_analysis_stage(
                threshold, rows, state.points, state.residuals, residual_quantiles,
            )
    return rows, support, threshold_analysis
