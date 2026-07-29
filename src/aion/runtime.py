from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .artifacts import write_artifact
from .context import ContextEvent
from .context_eval import CONTEXT_MODEL_NAME, assess_context
from .contracts import DataSchema, Evidence, ForecastArtifact, ForecastTask, SeriesResult
from .data import load_observations
from .evaluation import evaluate, quantile
from .models import predict
from .temporal import SEASONS, next_timestamp, validate_and_group


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

        assessment = evaluate(values, horizon, season, minimum_baseline_improvement)
        context_public: dict[str, object] | None = None
        selected_model = assessment.selected_model
        coverage = assessment.coverage
        warnings = list(assessment.warnings)
        points: list[float] = []
        residuals = assessment.residuals
        if assessment.supported and assessment.selected_model:
            points = predict(assessment.selected_model, values, horizon, season)
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
        support = "unsupported"
        if assessment.supported and points:
            residual_quantiles = {probability: quantile(residuals, probability) for probability in (0.1, 0.5, 0.9)}
            for timestamp, point in zip(future_timestamps, points):
                q10 = point + residual_quantiles[0.1]
                q50 = point + residual_quantiles[0.5]
                q90 = point + residual_quantiles[0.9]
                rows.append({
                    "timestamp": timestamp.isoformat(), "point": point,
                    "q10": min(q10, q50, q90), "q50": q50, "q90": max(q10, q50, q90),
                })
            support = "weakly_supported" if warnings else "supported"
        result = SeriesResult(
            series_name, support, selected_model, assessment.strongest_baseline,
            assessment.selection_scores, assessment.test_scores, assessment.improvement,
            coverage, warnings, rows, context_public,
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
    return {
        "schema_version": "0.1",
        "runtime_version": "0.1.0",
        "interfaces": {"cli": True, "python": True, "mcp": True, "http": False},
        "inputs": {"csv": True, "parquet": parquet},
        "frequencies": ["h", "D", "W", "MS"],
        "models": {
            "baselines": ["last_value", "seasonal_naive"],
            "statistical": ["drift"],
            "context": ["event_adjusted"],
            "tsfm": [],
        },
        "features": {
            "inspection": True, "forecasting": True, "separated_evaluation": True,
            "residual_intervals": True, "project_mode": False, "actual_scoring": False,
            "context_events": True, "llm_workflow_prompts": True, "sharing": False,
        },
    }

