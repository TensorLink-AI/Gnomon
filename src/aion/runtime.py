from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifacts import write_artifact
from .context import ContextEvent
from .contracts import Evidence, ForecastArtifact, ForecastTask, SeriesResult
from .covariates import CovariateDataset
from .ids import SYSTEM_CLOCK, Clock, content_id
from .models import BASELINES, MODELS
from .pipeline import (
    LoadedDataset,
    context_stage,
    covariate_stage,
    evaluate_stage,
    horizon_stage,
    interval_stage,
    load_stage,
    multivariate_stage,
    predict_stage,
)
from .temporal import FREQUENCY_DESCRIPTIONS, SEASONS, detect_season


def inspect_dataset(
    input_path: str,
    *,
    time_column: str,
    target_column: str,
    series_column: str | None = None,
    frequency: str | None = None,
    seasonal_period: int | None = None,
) -> dict[str, object]:
    loaded = load_stage(
        input_path, time_column=time_column, target_column=target_column,
        series_column=series_column, frequency=frequency,
    )
    from .multivariate import correlation_report
    return {
        "schema_version": "0.1",
        "status": "valid",
        "input_path": str(Path(input_path).expanduser().resolve()),
        "source_fingerprint": loaded.source_fingerprint,
        "columns": loaded.columns,
        "schema": {
            "time_column": time_column,
            "target_column": target_column,
            "series_column": series_column,
            "frequency": loaded.frequency,
            "timezone": loaded.timezone,
            "missing_policy": "reject",
            "duplicate_policy": "reject",
        },
        "series": [
            {
                "name": name,
                "observations": len(items),
                "start": items[0].timestamp.isoformat(),
                "end": items[-1].timestamp.isoformat(),
                "seasonality": dict(zip(("period", "strength", "source"),
                    (seasonal_period, 1.0, "override") if seasonal_period else detect_season([item.value for item in items], loaded.frequency))),
            }
            for name, items in sorted(loaded.groups.items())
        ],
        "cross_series_correlations": correlation_report(loaded.groups),
        "suggested_next": (
            f"aion forecast {Path(input_path).expanduser().resolve()} "
            f"--time {time_column} --target {target_column}"
            + (f" --series {series_column}" if series_column else "")
            + f" --frequency {loaded.frequency} --horizon <periods>"
        ),
    }


def _config_fingerprint(config: Any) -> dict[str, object] | None:
    """The behaviour-relevant subset of the config, for content addressing.

    A missing config and the built-in defaults fingerprint identically, so
    the same task yields the same artifact ID regardless of which interface
    invoked it."""
    if config is None:
        return None
    ensemble = getattr(config, "ensemble", None)
    meta_model = getattr(config, "meta_model", None)
    backends = getattr(config, "backends", None)
    api = getattr(backends, "api", None) if backends else None
    models = getattr(config, "models", None)
    payload: dict[str, object] = {
        "ensemble": asdict(ensemble) if ensemble is not None and is_dataclass(ensemble) and ensemble.enabled else None,
        "meta_model": asdict(meta_model) if meta_model is not None and is_dataclass(meta_model) and meta_model.enabled else None,
        "api_providers": sorted(api.providers) if api is not None and api.enabled else None,
        "tsfm_candidates": sorted(getattr(models, "tsfm_candidates", None) or []) or None,
    }
    if all(value is None for value in payload.values()):
        return None
    return payload


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
    covariates: CovariateDataset | None = None,
    threshold: float | None = None,
    config: Any = None,
    strict_abstention: bool = False,
    seasonal_period: int | None = None,
    selection_strategy: str = "best",
    multivariate: bool = False,
    clock: Clock | None = None,
    as_of: datetime | None = None,
    store_path: str | None = None,
) -> tuple[ForecastArtifact, Path]:
    clock = clock or SYSTEM_CLOCK
    if horizon < 1:
        from .contracts import AionError
        raise AionError("INVALID_HORIZON", "Horizon must be at least one period.")
    if context_events and covariates:
        from .contracts import AionError
        raise AionError(
            "COMBINED_ENRICHMENT_UNSUPPORTED",
            "Context events and covariates cannot yet be admitted in the same run; evaluate them separately.",
        )
    loaded: LoadedDataset = load_stage(
        input_path, time_column=time_column, target_column=target_column,
        series_column=series_column, frequency=frequency,
        as_of=as_of, store_path=store_path,
    )
    task = ForecastTask(
        input_path if input_path.startswith("store:")
        else str(Path(input_path).expanduser().resolve()),
        loaded.schema, horizon,
        minimum_baseline_improvement=minimum_baseline_improvement,
        as_of=as_of.isoformat() if as_of else None,
    )
    results: list[SeriesResult] = []
    evidence: list[Evidence] = []
    multivariate_points: dict[str, list[float]] = {}
    multivariate_warnings: dict[str, str] = {}
    if multivariate and len(loaded.groups) > 1:
        from .multivariate import forecast_var
        multivariate_points, multivariate_warnings = forecast_var(loaded.groups, horizon)
    for series_name, items in sorted(loaded.groups.items()):
        state = horizon_stage(
            series_name, items, horizon=horizon, frequency=loaded.frequency,
            seasonal_period=seasonal_period,
        )
        evaluate_stage(
            state, horizon=horizon,
            minimum_baseline_improvement=minimum_baseline_improvement,
            frequency=loaded.frequency, config=config,
            strict_abstention=strict_abstention,
            snapshot=loaded.snapshot, variable=loaded.variable,
        )
        predict_stage(
            state, horizon=horizon, frequency=loaded.frequency,
            selection_strategy=selection_strategy,
        )
        multivariate_stage(state, multivariate_points, multivariate_warnings)
        if context_events:
            context_stage(
                state, context_events, horizon=horizon,
                minimum_baseline_improvement=minimum_baseline_improvement,
            )
        if covariates:
            covariate_stage(
                state, covariates, horizon=horizon,
                minimum_baseline_improvement=minimum_baseline_improvement,
            )
        rows, support, threshold_analysis = interval_stage(state, threshold=threshold)
        assessment = state.assessment
        from .support import assess_forecast_support
        support_assessment = assess_forecast_support(
            support, state.warnings, assessment,
            known_time_assumed=loaded.snapshot.assumed_known_time,
        )
        result = SeriesResult(
            series_name, support, state.selected_model, assessment.strongest_baseline,
            assessment.selection_scores, assessment.test_scores, assessment.improvement,
            state.coverage, state.warnings, rows, state.context_public,
            state.covariate_public, threshold_analysis,
            support_assessment.to_dict(),
        )
        results.append(result)
        evidence.extend(state.evidence)
        evidence.extend([
            Evidence(f"evaluation:{series_name}", "rolling_evaluation", series_name, {
                "partitioning": "selection folds, then calibration fold, then final test fold",
                "selection_scores": assessment.selection_scores,
                "test_scores": assessment.test_scores,
            }),
            Evidence(f"support:{series_name}", "support_assessment", series_name, {
                "support": support, "warnings": state.warnings,
            }),
        ])
    evidence.append(Evidence(
        "snapshot", "snapshot_access", "__all__", loaded.snapshot.access_summary(),
    ))
    forecast_id = content_id("forecast", {
        "source": loaded.source_fingerprint,
        "as_of": as_of.isoformat() if as_of else None,
        "schema": {
            "time": time_column, "target": target_column, "series": series_column,
            "frequency": loaded.frequency, "timezone": loaded.timezone,
        },
        "horizon": horizon,
        "minimum_baseline_improvement": minimum_baseline_improvement,
        "threshold": threshold,
        "seasonal_period": seasonal_period,
        "selection_strategy": selection_strategy,
        "multivariate": multivariate,
        "strict_abstention": strict_abstention,
        "context_events": [event.__dict__ for event in context_events] if context_events else None,
        "covariates": {"source": covariates.fingerprint, "specs": [str(spec) for spec in covariates.specs]} if covariates else None,
        "config": _config_fingerprint(config),
    })
    artifact = ForecastArtifact(
        "0.1", forecast_id, clock.now().isoformat(),
        "complete", task, loaded.source_fingerprint, results, evidence,
    )
    from .contracts import forecast_task
    from .lineage import build_forecast_lineage
    from .verifier import verify_or_raise
    temporal_task = forecast_task(
        task.input_path, time_column=time_column, target_column=target_column,
        horizon=horizon, series_column=series_column, frequency=loaded.frequency,
        threshold=threshold, minimum_baseline_improvement=minimum_baseline_improvement,
        as_of=task.as_of,
    )
    lineage = build_forecast_lineage(artifact, temporal_task)
    # No response leaves the process unverified — including our own.
    verify_or_raise(lineage, as_of=task.as_of)
    return artifact, write_artifact(artifact, output, lineage=lineage.to_dict())


def capabilities() -> dict[str, object]:
    try:
        import pyarrow  # type: ignore[import-not-found]  # noqa: F401
        parquet = True
    except ImportError:
        parquet = False
    from .tsfm import available_tsfms, capability_matrix, installed_tsfms
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
            "tsfm_capabilities": capability_matrix(),
        },
        "features": {
            "inspection": True, "forecasting": True, "separated_evaluation": True,
            "residual_intervals": True, "horizon_widened_intervals": True,
            "threshold_analysis": True, "degraded_evaluation": True,
            "project_mode": True, "actual_scoring": True,
            "decision_outcomes": True, "agent_treatment_control_eval": True,
            "context_events": True, "llm_workflow_prompts": True, "sharing": False,
            "future_known_covariates": True, "point_in_time_covariates": True,
            "covariate_ablation": True,
            "season_detection": True, "ensemble_forecasting": True,
            "multivariate_var": True, "strict_abstention": True,
        },
    }
