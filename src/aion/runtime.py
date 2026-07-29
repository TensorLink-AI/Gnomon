from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .artifacts import write_artifact
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
        assessment = evaluate(values, horizon, season, minimum_baseline_improvement)
        rows: list[dict[str, object]] = []
        support = "unsupported"
        if assessment.supported and assessment.selected_model:
            points = predict(assessment.selected_model, values, horizon, season)
            residuals = assessment.residuals
            residual_quantiles = {probability: quantile(residuals, probability) for probability in (0.1, 0.5, 0.9)}
            timestamp = items[-1].timestamp
            for point in points:
                timestamp = next_timestamp(timestamp, resolved_frequency)
                q10 = point + residual_quantiles[0.1]
                q50 = point + residual_quantiles[0.5]
                q90 = point + residual_quantiles[0.9]
                rows.append({
                    "timestamp": timestamp.isoformat(), "point": point,
                    "q10": min(q10, q50, q90), "q50": q50, "q90": max(q10, q50, q90),
                })
            support = "weakly_supported" if assessment.warnings else "supported"
        result = SeriesResult(
            series_name, support, assessment.selected_model, assessment.strongest_baseline,
            assessment.selection_scores, assessment.test_scores, assessment.improvement,
            assessment.coverage, assessment.warnings, rows,
        )
        results.append(result)
        evidence.extend([
            Evidence(f"evaluation:{series_name}", "rolling_evaluation", series_name, {
                "partitioning": "selection folds, then calibration fold, then final test fold",
                "selection_scores": assessment.selection_scores,
                "test_scores": assessment.test_scores,
            }),
            Evidence(f"support:{series_name}", "support_assessment", series_name, {
                "support": support, "warnings": assessment.warnings,
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
        "interfaces": {"cli": True, "python": True, "mcp": False, "http": False},
        "inputs": {"csv": True, "parquet": parquet},
        "frequencies": ["h", "D", "W", "MS"],
        "models": {
            "baselines": ["last_value", "seasonal_naive"],
            "statistical": ["drift"],
            "tsfm": [],
        },
        "features": {
            "inspection": True, "forecasting": True, "separated_evaluation": True,
            "residual_intervals": True, "project_mode": False, "actual_scoring": False,
            "context_events": False, "sharing": False,
        },
    }

