from __future__ import annotations

import os
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
    adjudicate_enrichments_stage,
    conditional_stage,
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
    as_of: datetime | None = None,
    store_path: str | None = None,
    clock: Clock | None = None,
) -> dict[str, object]:
    # Diagnose, don't just reject: try the strict path, then each repair
    # level, and report what the file needs to become forecastable.
    from .contracts import GnomonError
    from .repair import REPAIR_LEVELS, RepairLog
    loaded = None
    repair_level_used = None
    log = RepairLog()
    errors: dict[str, GnomonError] = {}
    for level in REPAIR_LEVELS:
        log = RepairLog()
        try:
            loaded = load_stage(
                input_path, time_column=time_column, target_column=target_column,
                series_column=series_column, frequency=frequency,
                as_of=as_of, store_path=store_path,
                repair=level, repair_log=log,
            )
            repair_level_used = level
            break
        except GnomonError as error:
            errors[level] = error
    if loaded is None:
        # The safe-level error is the diagnosis; an aggressive-level failure
        # (e.g. EXCESSIVE_REPAIR) is a consequence of forcing assumptions.
        raise errors.get("safe") or errors["aggressive"]
    data_quality: dict[str, object] = {
        "status": {
            "off": "clean",
            "safe": "repaired_safe",
            "aggressive": "repaired_aggressive",
        }[repair_level_used],
        "repairs": log.summary()["actions"],
        "note": {
            "off": "The file passes strict validation untouched.",
            "safe": "The default repair level (safe) reads this file; every "
                    "normalisation is listed under repairs.",
            "aggressive": "This file needs --repair aggressive; the structural "
                          "fixes it would apply are listed under repairs and "
                          "will be disclosed as warnings on the forecast.",
        }[repair_level_used],
    }
    repair_flag = " --repair aggressive" if repair_level_used == "aggressive" else ""
    # A harness built on knowledge time should remark when every observation
    # postdates now. It is not an error — synthetic and planning data are
    # legitimate — but `status: valid` with no note read as endorsement.
    now = (clock or SYSTEM_CLOCK).now()
    latest = max(
        (item.timestamp for items in loaded.groups.values() for item in items),
        default=None,
    )
    if latest is not None:
        # Most datasets are naive and the clock is aware, so compare wall
        # clocks — the same alignment the context path makes, for the same
        # reason: without it the check never fires on real input.
        from .constraints import _align

        latest, now = _align(latest, now)
        if latest > now:
            earliest = min(
                item.timestamp for items in loaded.groups.values() for item in items
            )
            earliest, _ = _align(earliest, now)
            data_quality["temporal_position"] = (
                "entirely_in_the_future" if earliest > now else "extends_into_the_future"
            )
            data_quality["note"] += (
                f" Every observation is dated after the current instant "
                f"({now.isoformat()}); the series runs to {latest.isoformat()}. "
                f"That is legitimate for synthetic or planning data and "
                f"unusual otherwise."
                if earliest > now else
                f" The series extends past the current instant "
                f"({now.isoformat()}) to {latest.isoformat()}."
            )
    from .multivariate import correlation_report
    return {
        "schema_version": "0.1",
        "status": "valid",
        "input_path": (
            input_path if input_path.startswith("store:")
            else str(Path(input_path).expanduser().resolve())
        ),
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
        "data_quality": data_quality,
        "suggested_next": (
            f"gnomon forecast {Path(input_path).expanduser().resolve()} "
            f"--time {time_column} --target {target_column}"
            + (f" --series {series_column}" if series_column else "")
            + f" --frequency {loaded.frequency} --horizon <periods>"
            + repair_flag
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


def _restrict_candidates(config: Any, candidates: list[str]):
    """A copy of ``config`` whose candidate pool is the named models.

    Baselines are added back unconditionally: a candidate is selected by
    beating them, so a pool without them has nothing to select against.
    """
    import copy as copy_module

    from .config import load_config
    from .contracts import GnomonError
    from .models import BASELINES, MODELS
    from .tsfm import available_tsfms

    known_tsfms = set(available_tsfms())
    unknown = [
        name for name in candidates
        if name not in MODELS and name not in known_tsfms
    ]
    if unknown:
        raise GnomonError(
            "UNKNOWN_MODEL",
            f"candidates names models that do not exist: "
            f"{', '.join(sorted(unknown))}.",
            {"unknown": sorted(unknown),
             "available": sorted(set(MODELS) | known_tsfms)},
        )
    resolved = copy_module.deepcopy(config) if config is not None else load_config()
    statistical = [
        name for name in candidates if name in MODELS and name not in BASELINES
    ]
    resolved.models.statistical_candidates = statistical or None
    resolved.models.tsfm_candidates = [
        name for name in candidates if name in known_tsfms
    ]
    return resolved


def forecast(
    input_path: str,
    *,
    time_column: str,
    target_column: str,
    horizon: int,
    series_column: str | None = None,
    frequency: str | None = None,
    output: str = "gnomon-output",
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
    repair: str = "safe",
    candidates: list[str] | None = None,
) -> tuple[ForecastArtifact, Path]:
    clock = clock or SYSTEM_CLOCK
    if candidates:
        # `gnomon route`'s output, made actionable. The router answered "which
        # method for this task?" and nothing consumed the answer: forecast
        # had no model parameter at all, so even a confident recommendation
        # could not be acted on. Restricting the pool is advisory in the
        # right way — the named candidates still backtest against the
        # mandatory baselines, which are never removable.
        config = _restrict_candidates(config, candidates)
    from .repair import REPAIR_LEVELS, REPAIR_SAFE, RepairLog
    if repair not in REPAIR_LEVELS:
        from .contracts import GnomonError
        raise GnomonError(
            "INVALID_REPAIR_LEVEL",
            f"repair must be one of {', '.join(REPAIR_LEVELS)}.",
            {"requested": repair, "supported": list(REPAIR_LEVELS)},
        )
    if horizon < 1:
        from .contracts import GnomonError
        raise GnomonError("INVALID_HORIZON", "Horizon must be at least one period.")
    if selection_strategy == "ensemble":
        # Asking for the ensemble has to enter it in the evaluation, not just
        # swap the final forecast. Otherwise it is never scored on the folds
        # and has no fold-separated residuals to build an interval from.
        import copy as _copy
        from .config import load_config as _load_config
        config = _copy.deepcopy(config) if config is not None else _load_config()
        config.ensemble.enabled = True
    # When both enrichment kinds are supplied, neither ablation stage applies
    # its own winner; the adjudication ladder owns the choice.
    adjudicating = bool(context_events) and covariates is not None
    # `evaluation.uncertainty.target_coverage`, previously parsed and never
    # read: every run published an 80% interval whatever the config said.
    from .evaluation import DEFAULT_TARGET_COVERAGE
    target_coverage = DEFAULT_TARGET_COVERAGE
    if config is not None and getattr(config, "evaluation", None) is not None:
        target_coverage = float(
            getattr(config.evaluation, "target_coverage", DEFAULT_TARGET_COVERAGE)
        )
    if covariates is not None:
        # Bind the run's boundary to the covariate snapshot before anything
        # reads it, so leakage control is a property of the object rather
        # than of every call site remembering to pass a cutoff.
        covariates.bind_as_of(as_of)
    repair_log = RepairLog()
    loaded: LoadedDataset = load_stage(
        input_path, time_column=time_column, target_column=target_column,
        series_column=series_column, frequency=frequency,
        as_of=as_of, store_path=store_path,
        repair=repair, repair_log=repair_log,
    )
    task = ForecastTask(
        input_path if input_path.startswith("store:")
        else str(Path(input_path).expanduser().resolve()),
        loaded.schema, horizon,
        minimum_baseline_improvement=minimum_baseline_improvement,
        as_of=as_of.isoformat() if as_of else None,
    )
    # The textual-verifiability lane for future-dated context events.
    # Resolved once, here, so the flag has exactly one meaning per run and
    # the ID payload below can state it.
    future_events_enabled = bool(
        getattr(getattr(config, "context", None), "future_events", False)
    )
    future_context_admitted: dict[str, list[dict[str, object]]] = {}
    results: list[SeriesResult] = []
    evidence: list[Evidence] = []
    var_frame = None
    var_ineligible: str | None = None
    if multivariate:
        from .multivariate import VarFrame
        var_frame, var_ineligible = VarFrame.build(loaded.groups)
    for series_name, items in sorted(loaded.groups.items()):
        state = horizon_stage(
            series_name, items, horizon=horizon, frequency=loaded.frequency,
            seasonal_period=seasonal_period,
        )
        extra_candidates: dict[str, Any] = {}
        if var_frame is not None and series_name in var_frame.names:
            from .multivariate import MULTIVARIATE_MODEL_NAME
            extra_candidates[MULTIVARIATE_MODEL_NAME] = var_frame.predictor(series_name)
        evaluate_stage(
            state, horizon=horizon,
            minimum_baseline_improvement=minimum_baseline_improvement,
            frequency=loaded.frequency, config=config,
            strict_abstention=strict_abstention,
            snapshot=loaded.snapshot, variable=loaded.variable,
            extra_candidates=extra_candidates,
        )
        predict_stage(
            state, horizon=horizon, frequency=loaded.frequency,
            selection_strategy=selection_strategy,
            extra_candidates=extra_candidates,
        )
        if multivariate:
            multivariate_stage(
                state,
                eligible=var_frame is not None,
                minimum_baseline_improvement=minimum_baseline_improvement,
                ineligibility_reason=var_ineligible,
                strongest_correlation=(
                    round(var_frame.strongest_correlation, 4) if var_frame else None
                ),
                series_count=len(var_frame.names) if var_frame else len(loaded.groups),
            )
        if context_events:
            context_stage(
                state, context_events, horizon=horizon,
                minimum_baseline_improvement=minimum_baseline_improvement,
                apply=not adjudicating,
            )
        if covariates:
            covariate_stage(
                state, covariates, horizon=horizon,
                minimum_baseline_improvement=minimum_baseline_improvement,
                apply=not adjudicating,
            )
        if adjudicating:
            adjudicate_enrichments_stage(
                state, context_events, covariates, horizon=horizon,
            )
        if context_events:
            # After every stage that can change the point forecast and its
            # calibration: a conditional answer is conditioned on the
            # forecast that was actually selected.
            conditional_stage(state, context_events, horizon=horizon)
        repair_warnings = repair_log.warnings_for(series_name)
        if repair_warnings:
            state.warnings.extend(repair_warnings)
        rows, support, threshold_analysis = interval_stage(
            state, threshold=threshold, context_events=context_events,
            target_coverage=target_coverage,
            future_events=future_events_enabled,
        )
        if state.future_context_public and state.future_context_public.get("admitted"):
            future_context_admitted[series_name] = list(
                state.future_context_public["admitted"]  # type: ignore[arg-type]
            )
        assessment = state.assessment
        from .support import assess_forecast_support
        support_assessment = assess_forecast_support(
            support, state.warnings, assessment,
            known_time_assumed=loaded.snapshot.assumed_known_time,
            disclosures=state.disclosures,
            measured_coverage=state.coverage,
        )
        result = SeriesResult(
            series_name, support, state.selected_model, assessment.strongest_baseline,
            assessment.selection_scores, assessment.test_scores, assessment.improvement,
            state.coverage, state.warnings, rows, state.context_public,
            state.covariate_public, threshold_analysis,
            support_assessment.to_dict(),
            notes=state.notes,
            conditional_forecasts=state.conditional_forecasts,
            future_context=state.future_context_public,
        )
        results.append(result)
        evidence.extend(state.evidence)
        evidence.extend([
            Evidence(f"evaluation:{series_name}", "rolling_evaluation", series_name, {
                "partitioning": "selection folds, then calibration fold, then final test fold",
                "selection_scores": assessment.selection_scores,
                "test_scores": assessment.test_scores,
                # The verifier gates probability-bearing claims on these,
                # so they have to be *in* the calibration record rather
                # than only in the result beside it.
                "measured_interval_coverage": state.coverage,
                "baseline_improvement": assessment.improvement,
                "strongest_baseline": assessment.strongest_baseline,
                "selected_model": state.selected_model,
                "residuals_pooled_across_selection":
                    assessment.residuals_pooled_across_selection,
                "residual_fold_count": assessment.residual_fold_count,
            }),
            Evidence(f"support:{series_name}", "support_assessment", series_name, {
                "support": support, "warnings": state.warnings,
            }),
        ])
    if repair_log.has_actions():
        evidence.append(Evidence(
            "data_repair", "data_repair", "__all__",
            {"level": repair, **repair_log.summary()},
        ))
    snapshot_access: dict[str, object] = dict(loaded.snapshot.access_summary())
    if covariates is not None:
        # Merge the covariate reads in, so the `max_known_time` the
        # verifier's leakage check reads covers every source the run
        # consulted rather than the target series alone.
        covariate_access = covariates.access_summary()
        snapshot_access["accesses"] = list(snapshot_access.get("accesses", [])) + [
            {**entry, "source": "covariates"}
            for entry in covariate_access.get("accesses", [])
        ]
        snapshot_access["covariate_as_of"] = covariate_access.get("as_of")
    evidence.append(Evidence(
        "snapshot", "snapshot_access", "__all__", snapshot_access,
    ))
    # A selected TSFM's weights are part of what produced the numbers, so
    # they belong in the id and in the evidence. Without this the id covers
    # the model *name* only, and two runs at different Hub revisions could
    # publish different forecasts under one id.
    from .tsfm import resolved_weights
    selected_weights = {
        model: weights
        for model in sorted({
            item.selected_model for item in results if item.selected_model
        })
        if (weights := resolved_weights(model))
    }
    if selected_weights:
        evidence.append(Evidence(
            "model_weights", "model_weights", "__all__",
            {"pinned_revisions": selected_weights},
        ))
    id_payload: dict[str, object] = {
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
    }
    if repair != REPAIR_SAFE:
        # The default level is absent from the payload so IDs predating the
        # repair layer are unchanged.
        id_payload["repair"] = repair
    if selected_weights:
        # Absent when no TSFM was selected, so ids for baseline and
        # statistical selections are unchanged by this addition.
        id_payload["model_weights"] = selected_weights
    if future_events_enabled:
        # Same pattern as model_weights: the key exists only when the flag
        # is on, so every flag-off ID — including all pre-existing ones —
        # is byte-identical. When on, the ID covers both the flag and the
        # events that actually influenced the numbers.
        id_payload["future_context"] = {
            "enabled": True,
            "admitted": future_context_admitted,
        }
    forecast_id = content_id("forecast", id_payload)
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
    return artifact, write_artifact(
        artifact, output, lineage=lineage.to_dict(),
        output_config=getattr(config, "output", None),
    )


def _has_module(name: str) -> bool:
    from importlib.util import find_spec
    return find_spec(name) is not None


def capabilities() -> dict[str, object]:
    try:
        import pyarrow  # type: ignore[import-not-found]  # noqa: F401
        parquet = True
    except ImportError:
        parquet = False
    from .registry import registry_capabilities
    from .tsfm import available_tsfms, capability_matrix, installed_tsfms
    from .tsfm_sandbox import list_sandboxes
    try:
        from .config import load_config
        future_events_on = bool(load_config().context.future_events)
    except Exception:
        # A malformed config file must not make capabilities unreportable;
        # the flag reads as its default.
        future_events_on = False
    return {
        "schema_version": "0.1",
        "runtime_version": "0.5.0",
        "interfaces": {"cli": True, "python": True, "mcp": True, "http": False},
        "inputs": {
            "csv": True, "tsv": True, "json": True, "jsonl": True,
            "gzip": True, "parquet": parquet, "excel": _has_module("openpyxl"),
        },
        "frequencies": sorted(SEASONS),
        "frequency_descriptions": dict(FREQUENCY_DESCRIPTIONS),
        "context_events": {
            "fold_validated": {
                "model": "event_adjusted",
                "effect_shapes": ["level", "decay", "ramp"],
                "admission": "identical-fold ablation with known-at gating",
            },
            "future_events": {
                "flag": "context.future_events",
                "default": "off",
                "enabled": future_events_on,
                "event_classes": ["constraint", "deterministic_override"],
                "admission": (
                    "textual verifiability: a quoted source span, "
                    "deterministic re-parsing of its numbers, and a "
                    "recent-history consistency check — only for windows "
                    "with no overlap with the observed history"
                ),
                "disclosure": (
                    "influenced forecasts report support "
                    "'context_trusted' and carry the history-only "
                    "counterfactual in evidence"
                ),
            },
        },
        "models": {
            "baselines": sorted(BASELINES),
            "statistical": sorted(name for name in MODELS if name not in BASELINES),
            "context": ["event_adjusted"],
            "tsfm": installed_tsfms(),
            "tsfm_available": available_tsfms(),
            "tsfm_sandboxes": list_sandboxes(),
            "tsfm_capabilities": capability_matrix(),
            "tsfm_install_command": "gnomon tsfm install <name>",
            "tsfm_install_note": (
                "Sandboxed TSFMs are pulled per model into isolated venvs "
                "(requires uv; weights download on first inference). "
                "Installed models join forecast selection automatically; "
                "moment_small also adds a reconstruction candidate to "
                "detect_anomalies. Installation is a shell step, not a tool."
            ),
        },
        **registry_capabilities(),
        "experimental": {"planner": os.environ.get("GNOMON_EXPERIMENTAL_PLANNER") == "1"},
        "features": {
            "inspection": True, "forecasting": True, "separated_evaluation": True,
            "investigate_change": True, "decide": True, "monitor": True,
            "anomaly_detection": True, "graded_detector_selection": True,
            "series_fingerprints": True, "task_conditioned_leaderboard": True,
            "task_routing": True,
            "bitemporal_store": True, "as_of_replay": True, "typed_lineage": True,
            "claim_verifier": True,
            "residual_intervals": True, "horizon_widened_intervals": True,
            "threshold_analysis": True, "degraded_evaluation": True,
            "project_mode": True, "actual_scoring": True,
            "decision_outcomes": True, "agent_treatment_control_eval": True,
            "context_events": True, "future_context_events": True,
            "llm_workflow_prompts": True, "sharing": False,
            "future_known_covariates": True, "point_in_time_covariates": True,
            "covariate_ablation": True, "enrichment_adjudication": True,
            "season_detection": True, "ensemble_forecasting": True,
            "multivariate_var": True, "strict_abstention": True,
        },
    }
