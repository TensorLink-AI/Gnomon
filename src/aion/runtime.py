from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .artifacts import write_artifact
from .context import ContextEvent
from .context_eval import CONTEXT_MODEL_NAME, assess_context
from .contracts import DataSchema, Evidence, ForecastArtifact, ForecastTask, SeriesResult
from .data import load_observations
from .evaluation import evaluate, interval_bounds, quantile
from .models import BASELINES, MODELS, predict
from .temporal import FREQUENCY_DESCRIPTIONS, SEASONS, next_timestamp, validate_and_group


def inspect_dataset(
    input_path: str,
    *,
    time_column: str,
    target_column: str,
    series_column: str | None = None,
    frequency: str | None = None,
) -> dict[str, object]:
    observations, source_fingerprint, columns = load_observations(
        input_path, time_column, target_column, series_column
    )
    groups, resolved_frequency, zone = validate_and_group(observations, frequency)
    return {
        "schema_version": "0.1",
        "status": "valid",
        "input_path": str(Path(input_path).expanduser().resolve()),
        "source_fingerprint": source_fingerprint,
        "columns": columns,
        "schema": {
            "time_column": time_column,
            "target_column": target_column,
            "series_column": series_column,
            "frequency": resolved_frequency,
            "timezone": zone,
            "missing_policy": "reject",
            "duplicate_policy": "reject",
        },
        "series": [
            {
                "name": name,
                "observations": len(items),
                "start": items[0].timestamp.isoformat(),
                "end": items[-1].timestamp.isoformat(),
            }
            for name, items in sorted(groups.items())
        ],
        "suggested_next": (
            f"aion forecast {Path(input_path).expanduser().resolve()} "
            f"--time {time_column} --target {target_column}"
            + (f" --series {series_column}" if series_column else "")
            + f" --frequency {resolved_frequency} --horizon <periods>"
        ),
    }


def _assess_threshold(
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


def forecast(
    input_path: str,
    *,
    time_column: str,
    target_column: str,
    horizon: int,
    series_column: str | None = None,
    frequency: str | None = None,
    output: str = "aion-output",
    minimum_baseline_improvement: float = 0.02,
    context_events: list[ContextEvent] | None = None,
    threshold: float | None = None,
    config: Any = None,
) -> tuple[ForecastArtifact, Path]:
    if horizon < 1:
        from .contracts import AionError
        raise AionError("INVALID_HORIZON", "Horizon must be at least one period.")
    observations, source_fingerprint, _ = load_observations(
        input_path, time_column, target_column, series_column
    )
    groups, resolved_frequency, zone = validate_and_group(observations, frequency)
    schema = DataSchema(time_column, target_column, series_column, resolved_frequency, zone)
    task = ForecastTask(
        str(Path(input_path).expanduser().resolve()), schema, horizon,
        minimum_baseline_improvement=minimum_baseline_improvement,
    )
    results: list[SeriesResult] = []
    evidence: list[Evidence] = []
    season = SEASONS[resolved_frequency]
    for series_name, items in sorted(groups.items()):
        values = [item.value for item in items]
        timestamps = [item.timestamp for item in items]
        future_timestamps = []
        timestamp = timestamps[-1]
        for _ in range(horizon):
            timestamp = next_timestamp(timestamp, resolved_frequency)
            future_timestamps.append(timestamp)

        assessment = evaluate(
            values, horizon, season, minimum_baseline_improvement,
            frequency=resolved_frequency,
            config=config,
        )
        context_public: dict[str, object] | None = None
        selected_model = assessment.selected_model
        coverage = assessment.coverage
        warnings = list(assessment.warnings)
        points: list[float] = []
        residuals = assessment.residuals
        if assessment.supported and assessment.selected_model:
            if assessment.selected_model in MODELS:
                points = predict(assessment.selected_model, values, horizon, season)
            else:
                # TSFM selected — use the adapter for the final forecast.
                # Try sandbox first, then in-process.
                from .tsfm import get_tsfm, TSFMError, TSFMUnavailable
                from .tsfm_sandbox import sandbox_tsfm_candidates, sandbox_available_tsfms
                try:
                    if assessment.selected_model in sandbox_available_tsfms():
                        adapters = sandbox_tsfm_candidates(
                            requested=[assessment.selected_model],
                            frequency=resolved_frequency,
                        )
                    else:
                        adapters = []
                    if adapters:
                        adapter = adapters[0]
                    else:
                        adapter = get_tsfm(assessment.selected_model)
                        if hasattr(adapter, '_frequency'):
                            adapter._frequency = resolved_frequency
                    points = adapter.predict(values, horizon, season)
                except (TSFMError, TSFMUnavailable, Exception) as exc:
                    import logging
                    logging.getLogger(__name__).warning(
                        "TSFM %s failed during final forecast, falling back to %s: %s",
                        assessment.selected_model, assessment.strongest_baseline, exc,
                    )
                    selected_model = assessment.strongest_baseline
                    points = predict(selected_model, values, horizon, season)
        if context_events:
            context = assess_context(
                values, timestamps, future_timestamps, context_events, series_name,
                horizon, season, minimum_baseline_improvement, assessment,
            )
            context_public = context.to_public_dict()
            evidence.append(Evidence(
                f"context_ablation:{series_name}", "context_ablation", series_name,
                context_public,
            ))
            if context.admitted:
                selected_model = CONTEXT_MODEL_NAME
                points = context.points
                residuals = context.residuals
                coverage = context.coverage
                warnings = list(context.warnings)

        rows: list[dict[str, object]] = []
        threshold_analysis: dict[str, object] | None = None
        support = "unsupported"
        if assessment.supported and points:
            residual_quantiles = {probability: quantile(residuals, probability) for probability in (0.1, 0.5, 0.9)}
            for step, (timestamp, point) in enumerate(zip(future_timestamps, points), 1):
                q10, q50, q90 = interval_bounds(point, residual_quantiles, step)
                rows.append({
                    "timestamp": timestamp.isoformat(), "point": point,
                    "q10": q10, "q50": q50, "q90": q90,
                })
            support = "weakly_supported" if warnings else "supported"
            if threshold is not None:
                threshold_analysis = _assess_threshold(
                    threshold, rows, points, residuals, residual_quantiles,
                )
        result = SeriesResult(
            series_name, support, selected_model, assessment.strongest_baseline,
            assessment.selection_scores, assessment.test_scores, assessment.improvement,
            coverage, warnings, rows, context_public, threshold_analysis,
        )
        results.append(result)
        evidence.extend([
            Evidence(f"evaluation:{series_name}", "rolling_evaluation", series_name, {
                "partitioning": "selection folds, then calibration fold, then final test fold",
                "selection_scores": assessment.selection_scores,
                "test_scores": assessment.test_scores,
            }),
            Evidence(f"support:{series_name}", "support_assessment", series_name, {
                "support": support, "warnings": warnings,
            }),
        ])
    artifact = ForecastArtifact(
        "0.1", f"forecast_{uuid4().hex}", datetime.now(timezone.utc).isoformat(),
        "complete", task, source_fingerprint, results, evidence,
    )
    return artifact, write_artifact(artifact, output)


def capabilities() -> dict[str, object]:
    try:
        import pyarrow  # type: ignore[import-not-found]  # noqa: F401
        parquet = True
    except ImportError:
        parquet = False
    from .tsfm import available_tsfms, installed_tsfms
    from .tsfm_sandbox import list_sandboxes
    return {
        "schema_version": "0.1",
        "runtime_version": "0.2.0",
        "interfaces": {"cli": True, "python": True, "mcp": True, "http": False},
        "inputs": {"csv": True, "parquet": parquet},
        "frequencies": sorted(SEASONS),
        "frequency_descriptions": dict(FREQUENCY_DESCRIPTIONS),
        "models": {
            "baselines": sorted(BASELINES),
            "statistical": sorted(name for name in MODELS if name not in BASELINES),
            "context": ["event_adjusted"],
            "tsfm": installed_tsfms(),
            "tsfm_available": available_tsfms(),
            "tsfm_sandboxes": list_sandboxes(),
        },
        "features": {
            "inspection": True, "forecasting": True, "separated_evaluation": True,
            "residual_intervals": True, "horizon_widened_intervals": True,
            "threshold_analysis": True, "degraded_evaluation": True,
            "project_mode": True, "actual_scoring": True,
            "decision_outcomes": True, "agent_treatment_control_eval": True,
            "context_events": True, "llm_workflow_prompts": True, "sharing": False,
        },
    }
