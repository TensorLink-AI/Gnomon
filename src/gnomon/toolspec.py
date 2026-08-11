"""Canonical machine-facing tool specifications.

One definition of Gnomon's agent-facing tools — names, JSON Schemas, and
in-process runners over the runtime — consumed by the MCP server and any
future adapter. The Hermes plugin carries its own copy of the schemas
because it must remain standalone-installable; this module is the source
those copies are checked against.
"""

from __future__ import annotations

from typing import Any, Callable

from .context import load_events_file
from .contracts import ForecastArtifact
from .runtime import capabilities, forecast, inspect_dataset

_INPUT_PROPERTIES: dict[str, Any] = {
    "input": {"type": "string", "description": "Path to a local CSV, TSV, JSON, JSONL, Parquet, or Excel file of time-series observations, or `store:<dataset>` to read a dataset from the bitemporal store (see gnomon_list_datasets)."},
    "time_column": {"type": "string", "description": "Name of the timestamp column."},
    "target_column": {"type": "string", "description": "Name of the numeric column to forecast."},
    "series_column": {"type": "string", "description": "Optional column identifying independent series."},
    "frequency": {
        "type": "string",
        "pattern": "^([1-9][0-9]*)?(s|min|h)$|^(D|W|MS)$",
        "description": (
            "Observation frequency: s (seconds), min (minutes), h (hourly), "
            "D (daily), W (weekly), MS (month start), or any whole-second "
            "sub-daily step as <N>s, <N>min, or <N>h (e.g. 5min, 90s, 2h). "
            "Omit to infer; ambiguity fails loudly."
        ),
    },
}

#: Replay controls, shared by every verb that reads data. `gnomon_forecast`
#: and `gnomon_inspect` were the two that lacked them, which made the
#: bitemporal store and `--as-of` replay CLI-only — invisible to the agents
#: the MCP server exists to serve.
_REPLAY_PROPERTIES: dict[str, Any] = {
    "as_of": {
        "type": "string",
        "description": (
            "Replay instant (ISO-8601). Only data whose known_time is at or "
            "before this is visible; the artifact's snapshot_access evidence "
            "proves what was served. Requires a `store:<dataset>` input to "
            "mean anything, since a plain file carries one vintage."
        ),
    },
    "store_path": {
        "type": "string",
        "description": (
            "Override the temporal-store path for `store:<dataset>` inputs "
            "(default ~/.local/share/gnomon)."
        ),
    },
}

#: Covariate parameters, shared by every tool that takes them so the four
#: covariate entry points describe the same grammar the parser enforces.
_COVARIATE_PROPERTIES: dict[str, Any] = {
    "covariates_file": {"type": "string", "description": (
        "Local CSV of point-in-time covariate vintages; call "
        "gnomon_covariate_guide first for the exact format and cutoffs "
        "this dataset and horizon require."
    )},
    "covariate_mapping": {"type": "string", "description": (
        "Comma-separated name:type:availability entries, e.g. "
        "`promo:binary:future_known,temperature:continuous:future_known`. "
        "type is continuous or binary; availability must be future_known "
        "in this release — a value not knowable ahead of time cannot be "
        "backtested without leakage."
    )},
    "covariate_time_column": {"type": "string", "description": (
        "Valid-at column in the covariates file (default timestamp)."
    )},
    "covariate_known_at_column": {"type": "string", "description": (
        "Publication-time column (default known_at): when each value "
        "became knowable, so each backtest fold sees only what was "
        "published by its cutoff."
    )},
    "covariate_series_column": {"type": "string", "description": (
        "Series column in the covariates file, for panel data."
    )},
}

_HORIZON_PROPERTY: dict[str, Any] = {
    "type": "integer",
    "description": "Future periods to forecast, in units of the data frequency.",
}

#: Tools that read state without writing artifacts, stores, or the registry.
#: Surfaced to hosts as MCP `readOnlyHint` annotations so an agent can tell
#: an inspection from a mutation without guessing from the name.
READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "gnomon_capabilities",
    "gnomon_inspect",
    "gnomon_covariate_guide",
    "gnomon_validate_covariates",
    "gnomon_list_open_forecasts",
    "gnomon_model_performance",
    "gnomon_list_datasets",
    "gnomon_get_artifact",
    "gnomon_status",
    "gnomon_explain_run",
    "gnomon_proposer_skill",
    "gnomon_compile_task",
    "gnomon_validate_plan",
    "gnomon_get_run",
})

FORECAST_PREVIEW_ROWS = 12


def forecast_summary(artifact: ForecastArtifact, path: Any) -> dict[str, Any]:
    """The compact forecast payload shared by the CLI and every adapter.

    The first forecast rows are inlined so an agent can quote numbers without
    a second read; the full series always lives in forecast.csv."""
    return {
        "schema_version": "0.1",
        "status": "complete",
        "forecast_id": artifact.forecast_id,
        "artifact_path": str(path),
        "results": [
            {
                "series": item.series, "support": item.support,
                "support_assessment": item.support_assessment,
                "selected_model": item.selected_model,
                "interval_coverage": item.interval_coverage,
                "warnings": item.warnings,
                "forecast_preview": item.forecast[:FORECAST_PREVIEW_ROWS],
                "forecast_rows": len(item.forecast),
                "threshold": item.threshold,
                "context": item.context,
                "covariates": item.covariates,
            }
            for item in artifact.results
        ],
    }


def brief_summary(artifact: ForecastArtifact, path: Any) -> dict[str, Any]:
    """The compact forecast payload: q50 path, one q10–q90 interval, the
    selection, and every disclosure — roughly summary.md as JSON.

    What it drops is bulk only: the extra quantile levels, the raw
    `point` path beside its bias correction, and the context/covariate
    gate detail (all still in the artifact on disk, which is written
    unchanged). What it may never drop is epistemics: the support state,
    every warning, every abstention reason, every recovery action, and
    every disclosure ride along verbatim — an abstention serialises the
    same structured support assessment full mode carries. Hiding
    disclosures is the one thing this codebase exists to not do.
    """
    results = []
    for item in artifact.results:
        results.append({
            "series": item.series,
            "support": item.support,
            "selected_model": item.selected_model,
            "interval_coverage": item.interval_coverage,
            # Verbatim, never summarised: the same objects full mode carries.
            "warnings": item.warnings,
            "support_assessment": item.support_assessment,
            "notes": item.notes,
            "forecast": [
                {"timestamp": row["timestamp"], "q50": row["q50"],
                 "q10": row["q10"], "q90": row["q90"]}
                for row in item.forecast
            ],
            **({"threshold": item.threshold} if item.threshold else {}),
        })
    return {
        "schema_version": "0.1",
        "status": "complete",
        "format": "brief",
        "forecast_id": artifact.forecast_id,
        "artifact_path": str(path),
        "note": (
            "Brief output: q50 with the q10-q90 interval per step. The full "
            "artifact (all quantile levels, evidence, lineage) is on disk at "
            "artifact_path, unchanged."
        ),
        "results": results,
    }


def _run_capabilities(arguments: dict[str, Any]) -> dict[str, Any]:
    return capabilities()


def _run_inspect(arguments: dict[str, Any]) -> dict[str, Any]:
    return inspect_dataset(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
        store_path=arguments.get("store_path"),
    )


def _run_ingest(arguments: dict[str, Any]) -> dict[str, Any]:
    """Append a file's observations to the bitemporal store as vintages."""
    from .ids import SYSTEM_CLOCK
    from .temporal_store import TemporalStore

    store = TemporalStore(arguments.get("store_path"))
    report = store.ingest_csv(
        str(arguments["input"]),
        dataset=str(arguments["dataset"]),
        time_column=str(arguments["time_column"]),
        target_column=str(arguments["target_column"]),
        series_column=arguments.get("series_column"),
        known_at_column=arguments.get("known_at_column"),
        variable=arguments.get("variable"),
        clock=SYSTEM_CLOCK,
    )
    return report.to_dict()


def _run_list_datasets(arguments: dict[str, Any]) -> dict[str, Any]:
    from .temporal_store import TemporalStore

    store = TemporalStore(arguments.get("store_path"))
    datasets = store.list_datasets()
    return {
        "schema_version": "0.1",
        "status": "ok",
        "datasets": [
            {**item,
             "input_ref": f"store:{item['dataset']}",
             "known_time_provenance": store.known_time_provenance(str(item["dataset"]))}
            for item in datasets
        ],
    }


def _run_covariate_guide(arguments: dict[str, Any]) -> dict[str, Any]:
    from .covariates import covariate_guide
    return covariate_guide(
        arguments["input"], time_column=arguments["time_column"],
        target_column=arguments["target_column"], horizon=int(arguments["horizon"]),
        series_column=arguments.get("series_column"), frequency=arguments.get("frequency"),
    )


def _run_validate_covariates(arguments: dict[str, Any]) -> dict[str, Any]:
    from .covariates import validate_covariate_file
    return validate_covariate_file(
        arguments["input"], arguments["covariates_file"], arguments["covariate_mapping"],
        time_column=arguments["time_column"], target_column=arguments["target_column"],
        horizon=int(arguments["horizon"]), series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        covariate_time_column=arguments.get("covariate_time_column", "timestamp"),
        covariate_known_at_column=arguments.get("covariate_known_at_column", "known_at"),
        covariate_series_column=arguments.get("covariate_series_column"),
    )


def _run_forecast(arguments: dict[str, Any]) -> dict[str, Any]:
    target_spec = str(arguments["target_column"])
    if "," in target_spec or target_spec.strip().lower() == "auto":
        return _run_forecast_multi(arguments, target_spec)
    events = None
    if arguments.get("context_events_file"):
        events = load_events_file(arguments["context_events_file"])
    covariates = None
    if arguments.get("covariates_file"):
        from .covariates import load_covariates
        covariates = load_covariates(
            arguments["covariates_file"], arguments["covariate_mapping"],
            time_column=arguments.get("covariate_time_column", "timestamp"),
            known_at_column=arguments.get("covariate_known_at_column", "known_at"),
            series_column=arguments.get("covariate_series_column"),
        )
    artifact, path = forecast(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        horizon=int(arguments["horizon"]),
        as_of=_parse_as_of(arguments.get("as_of")),
        store_path=arguments.get("store_path"),
        output=arguments.get("output_dir") or "gnomon-output",
        minimum_baseline_improvement=float(arguments.get("minimum_baseline_improvement", 0.02)),
        context_events=events,
        covariates=covariates,
        threshold=float(arguments["threshold"]) if arguments.get("threshold") is not None else None,
        repair=arguments.get("repair", "safe"),
        candidates=arguments.get("candidates"),
    )
    payload = (brief_summary(artifact, path)
               if arguments.get("format") == "brief"
               else forecast_summary(artifact, path))
    if arguments.get("project"):
        from .tracking import register_artifact
        payload["tracking_ids"] = register_artifact(
            artifact, str(arguments["project"]), str(path),
            context_events=events,
        )
        payload["project"] = str(arguments["project"])
    return payload


def _run_forecast_multi(arguments: dict[str, Any], target_spec: str) -> dict[str, Any]:
    """The multi-target branch of gnomon_forecast: a comma list or `auto`
    in target_column batches several columns into one run and one
    combined artifact — same numbers per channel as separate calls."""
    from .contracts import GnomonError
    from .data import resolve_target_spec
    from .runtime import forecast_multi

    targets = resolve_target_spec(
        str(arguments["input"]), target_spec,
        time_column=arguments.get("time_column"),
        series_column=arguments.get("series_column"),
    )
    if len(targets) == 1:
        return _run_forecast({**arguments, "target_column": targets[0]})
    unsupported = [
        name for name in (
            "series_column", "context_events_file", "covariates_file", "project",
        ) if arguments.get(name)
    ]
    if unsupported:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            f"{', '.join(unsupported)} cannot be combined with a "
            f"multi-target target_column yet; run those channels one "
            f"target at a time.",
            {"unsupported_with_multi_target": unsupported, "targets": targets},
        )
    artifact, path = forecast_multi(
        str(arguments["input"]),
        time_column=arguments["time_column"],
        target_columns=targets,
        frequency=arguments.get("frequency"),
        horizon=int(arguments["horizon"]),
        as_of=_parse_as_of(arguments.get("as_of")),
        output=arguments.get("output_dir") or "gnomon-output",
        minimum_baseline_improvement=float(arguments.get("minimum_baseline_improvement", 0.02)),
        threshold=float(arguments["threshold"]) if arguments.get("threshold") is not None else None,
        repair=arguments.get("repair", "safe"),
        candidates=arguments.get("candidates"),
    )
    if arguments.get("format") == "brief":
        return brief_summary(artifact, path)
    return forecast_summary(artifact, path)


def _run_submit_actuals(arguments: dict[str, Any]) -> dict[str, Any]:
    import csv as csv_module

    from .tracking import TrackingStore
    store = TrackingStore()
    project = str(arguments["project"])
    path = str(arguments["actuals_file"])
    time_column = arguments.get("time_column")
    target_column = arguments.get("target_column")
    series_column = arguments.get("series_column")
    results = store.submit_actuals_csv(
        project, path, time_column=time_column,
        target_column=target_column, series_column=series_column,
    )
    if not results:
        # A bare `scored: 0` reads as "nothing was due" whether or not
        # anything was due. Return the diagnosis instead.
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv_module.DictReader(handle)
            columns = reader.fieldnames or []
            rows = list(reader)
        resolved_time, _, _ = store._resolve_actuals_columns(
            columns, time_column, target_column, series_column,
        )
        return {
            "schema_version": "0.1", "status": "ok", "project": project,
            **store.explain_unscored(project, [row[resolved_time] for row in rows]),
        }
    return {"schema_version": "0.1", "status": "ok", "scored": len(results),
            "results": [item.__dict__ for item in results]}


def _run_open_forecasts(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    rows = TrackingStore(create=False).due_forecasts(arguments.get("project"))
    return {"status": "ok", "forecasts": rows}


def _run_model_performance(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    store = TrackingStore(create=False)
    if arguments.get("model"):
        rows: Any = store.model_performance(
            str(arguments["project"]), str(arguments["model"]),
        )
    else:
        rows = [item.__dict__ for item in store.leaderboard(str(arguments["project"]))]
    return {"status": "ok", "performance": rows,
            "warning": "Historical telemetry is observational, not causal."}


def _run_proposer_skill(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    rows = TrackingStore(create=False).proposer_skill(
        str(arguments["project"]),
        proposer_id=(str(arguments["proposer_id"])
                     if arguments.get("proposer_id") else None),
        event_type=(str(arguments["event_type"])
                    if arguments.get("event_type") else None),
    )
    return {"status": "ok", "proposers": rows,
            "warning": ("Observational, set-level attribution; shrunk "
                        "toward no-skill priors. No proposal currently "
                        "earns forecast influence from these numbers.")}


def _run_record_decision(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    item = TrackingStore().record_decision(
        str(arguments["decision_id"]), str(arguments["project"]),
        str(arguments["forecast_id"]), str(arguments["action"]),
        str(arguments["expected_outcome"]),
    )
    return {"status": "ok", "decision": item.__dict__}


def _run_resolve_decision(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    item = TrackingStore().resolve_decision(
        str(arguments["decision_id"]), str(arguments["actual_outcome"]),
        bool(arguments["correct"]),
    )
    return {"status": "ok", "decision": item.__dict__}


TOOLS: list[dict[str, Any]] = [
    {
        "name": "gnomon_capabilities",
        "description": (
            "Report what the installed Gnomon runtime actually supports. Use for "
            "feature detection instead of assuming a capability exists."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "runner": _run_capabilities,
    },
    {
        "name": "gnomon_inspect",
        "description": (
            "Validate a temporal dataset before forecasting: schema mapping, "
            "frequency, duplicates, missing periods. Prefer this before "
            "gnomon_forecast when mappings or data quality are uncertain."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {**_INPUT_PROPERTIES, **_REPLAY_PROPERTIES},
            "required": ["input", "time_column", "target_column"],
        },
        "runner": _run_inspect,
    },
    {
        "name": "gnomon_forecast",
        "description": (
            "Run Gnomon's evaluated forecast: baselines and candidates are "
            "backtested on rolling folds; each series gets a selected model or "
            "an unsupported abstention. Context events (from `gnomon context "
            "validate`) and covariates are admitted only when they demonstrate "
            "stable lift on identical folds; when both are supplied, an "
            "adjudication ladder picks the best of base, context, covariates, "
            "or their combination and records the comparison as evidence. "
            "Read forecast.csv / summary.md in the returned "
            "artifact directory for the numbers and quote them verbatim; never "
            "invent values for an unsupported series."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **_INPUT_PROPERTIES,
                **_REPLAY_PROPERTIES,
                "target_column": {"type": "string", "description": (
                    "Name of the numeric column to forecast. Also accepts a "
                    "comma list (`hr,spo2,resp`) or `auto` (every numeric "
                    "non-time column) to batch several columns of a wide "
                    "file into one run — one shared load pass, channels "
                    "evaluated concurrently, one combined artifact with a "
                    "result per column. Each channel's numbers are identical "
                    "to a single-target run; a channel that abstains is "
                    "disclosed in its own result and never blocks the others."
                )},
                "horizon": {"type": "integer", "description": "Future periods to forecast, in units of the data frequency."},
                "format": {"type": "string", "enum": ["full", "brief"], "description": (
                    "Response verbosity (default full). `brief` returns per "
                    "target the q50 path with one q10-q90 interval, the "
                    "selected model, and — verbatim, never summarised — the "
                    "support state, every warning, abstention reason, "
                    "recovery action, and disclosure. The full artifact is "
                    "written to disk unchanged either way; only the response "
                    "payload shrinks."
                )},
                "candidates": {"type": "array", "items": {"type": "string"}, "description": "Restrict the model pool to these names — pass `gnomon_route`'s `candidates` or its `recommendation` to act on a routing decision. The mandatory baselines always compete regardless, so a named candidate still has to beat them."},
                "output_dir": {"type": "string", "description": "Directory for the immutable artifact. Defaults to ./gnomon-output relative to the *server's* working directory, which is often inside the user's repository — pass an explicit path when that matters."},
                "minimum_baseline_improvement": {"type": "number", "minimum": 0, "description": "Minimum relative improvement over the strongest baseline to select a candidate (default 0.02). Must be >= 0; a negative value would let a model that lost the backtest be selected."},
                "context_events_file": {"type": "string", "description": "Optional validated context-events JSON file (the output of `gnomon context validate`)."},
                "threshold": {"type": "number", "description": "Optional decision threshold: the result reports when and how likely the forecast crosses this value."},
                "project": {"type": "string", "description": "Optional tracking project. When set, register the forecast for realised scoring."},
                **_COVARIATE_PROPERTIES,
                "repair": {"type": "string", "enum": ["off", "safe", "aggressive"], "description": "Messy-data handling (default safe): off rejects anything non-strict; safe normalises cell text with disclosure; aggressive additionally fills gaps, snaps timestamps, and resolves conflicts — capped, and every fix is reported in evidence and warnings."},
            },
            "required": ["input", "time_column", "target_column", "horizon"],
        },
        "runner": _run_forecast,
    },
    {
        "name": "gnomon_covariate_guide",
        "description": "Return point-in-time format, forecast dates, and fold cutoffs. Gnomon does not suggest what data to fetch.",
        "inputSchema": {"type": "object", "properties": {
            **_INPUT_PROPERTIES,
            "horizon": {**_HORIZON_PROPERTY, "description": (
                "Future periods the eventual forecast will cover; determines "
                "the fold cutoffs and forecast dates returned."
            )},
        }, "required": ["input", "time_column", "target_column", "horizon"]},
        "runner": _run_covariate_guide,
    },
    {
        "name": "gnomon_validate_covariates",
        "description": "Validate local covariate vintages for format, final-horizon coverage, and availability at every selection cutoff.",
        "inputSchema": {"type": "object", "properties": {
            **_INPUT_PROPERTIES,
            "horizon": _HORIZON_PROPERTY,
            **_COVARIATE_PROPERTIES,
        }, "required": ["input", "time_column", "target_column", "horizon", "covariates_file", "covariate_mapping"]},
        "runner": _run_validate_covariates,
    },
    {
        "name": "gnomon_propose_covariates",
        "description": (
            "Evaluate a local point-in-time covariate proposal through "
            "leakage-safe ablation on identical folds. This is the "
            "covariate-admission entry to the same evaluated forecast as "
            "gnomon_forecast (one shared runner): the response is always a "
            "full forecast payload, and whether the covariates were admitted "
            "or rejected is disclosed in the result — a rejected proposal "
            "still returns the covariate-free forecast."
        ),
        "inputSchema": {"type": "object", "properties": {
            **_INPUT_PROPERTIES,
            "horizon": _HORIZON_PROPERTY,
            "output_dir": {"type": "string", "description": (
                "Directory for the immutable artifact (default "
                "./gnomon-output relative to the server's working directory)."
            )},
            "minimum_baseline_improvement": {"type": "number", "minimum": 0, "description": (
                "Minimum relative improvement over the strongest baseline to "
                "select a candidate (default 0.02). Must be >= 0."
            )},
            **_COVARIATE_PROPERTIES,
        }, "required": ["input", "time_column", "target_column", "horizon", "covariates_file", "covariate_mapping"]},
        "runner": _run_forecast,
    },
    {
        "name": "gnomon_submit_actuals",
        "description": "Score all due forecasts in a project from complete realised actuals. Panel actuals must include series,timestamp,value. A forecast scores only when every period in its horizon has an actual; when nothing scores, the result explains which window was missing rather than returning a bare zero.",
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string"}, "actuals_file": {"type": "string"},
            "time_column": {"type": "string", "description": "Timestamp column in the actuals file. Inferred from a conventional name or a two-column layout when omitted."},
            "target_column": {"type": "string", "description": "Realised value column. Inferred when unambiguous."},
            "series_column": {"type": "string", "description": "Series column, required for multi-series projects."},
        }, "required": ["project", "actuals_file"]},
        "runner": _run_submit_actuals,
    },
    {
        "name": "gnomon_list_open_forecasts",
        "description": "List unscored forecasts and distinguish horizons that are due from those still awaiting observations.",
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string"},
        }, "required": []},
        "runner": _run_open_forecasts,
    },
    {
        "name": "gnomon_model_performance",
        "description": "Read descriptive realised model performance for a project. Do not treat observational rankings as causal evidence.",
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string"}, "model": {"type": "string"},
        }, "required": ["project"]},
        "runner": _run_model_performance,
    },
    {
        "name": "gnomon_record_decision",
        "description": ("DEPRECATED (v0.2 lifecycle) — prefer `gnomon_decide`, which produces a DecisionArtifact that `gnomon_resolve_outcome` scores against realised utility and regret. This pair records a free-text action and a later yes/no verdict, which cannot express \"a costly precaution was rational even though the adverse event never occurred\". Kept for v0.2 compatibility. Link an agent decision and expected outcome to a tracked forecast."),
        "inputSchema": {"type": "object", "properties": {
            "decision_id": {"type": "string", "description": (
                "Caller-minted stable identifier for this decision; pass the "
                "same value to gnomon_resolve_decision later."
            )},
            "project": {"type": "string", "description": "Tracking project the forecast was registered in."},
            "forecast_id": {"type": "string", "description": "forecast_id returned by gnomon_forecast."},
            "action": {"type": "string", "description": "Free-text description of the action taken."},
            "expected_outcome": {"type": "string", "description": "Free-text expected outcome, judged later by resolve."},
        }, "required": ["decision_id", "project", "forecast_id", "action", "expected_outcome"]},
        "runner": _run_record_decision,
    },
    {
        "name": "gnomon_resolve_decision",
        "description": ("DEPRECATED (v0.2 lifecycle) — resolves records made by `gnomon_record_decision` only. Bare `correct` is retired: prefer `gnomon_decide` + `gnomon_resolve_outcome`, which report realised utility, regret against the best feasible action in hindsight, and ex-ante optimality separately. Record the realised business outcome and whether a previously recorded agent decision was correct."),
        "inputSchema": {"type": "object", "properties": {
            "decision_id": {"type": "string", "description": "The id passed to gnomon_record_decision."},
            "actual_outcome": {"type": "string", "description": "Free-text realised outcome."},
            "correct": {"type": "boolean", "description": "Whether the recorded decision proved correct."},
        }, "required": ["decision_id", "actual_outcome", "correct"]},
        "runner": _run_resolve_decision,
    },
    {
        "name": "gnomon_ingest",
        "description": (
            "Append a file's observations to the bitemporal store as vintages. "
            "Supply known_at_column when the source records when each value "
            "became knowable — that is what makes `as_of` replay meaningful. "
            "Without it Gnomon records known_time = valid_time and says so, "
            "which asserts every value was knowable the moment it applied. "
            "Re-ingesting a corrected file appends revisions; it never "
            "overwrites, so the vintage history accumulates."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Path to the CSV to ingest."},
                "dataset": {"type": "string", "description": "Dataset name; read it back as `store:<dataset>`."},
                "time_column": {"type": "string", "description": "Valid-time column: when the value applies."},
                "target_column": {"type": "string", "description": "Numeric value column."},
                "known_at_column": {"type": "string", "description": "Known-time column: when the value became knowable. Omit only if the source genuinely has no publication lag."},
                "series_column": {"type": "string", "description": "Optional column identifying independent series."},
                "variable": {"type": "string", "description": "Name to store the measure under (defaults to target_column)."},
                "store_path": {"type": "string", "description": "Override the temporal-store path."},
            },
            "required": ["input", "dataset", "time_column", "target_column"],
        },
        "runner": _run_ingest,
    },
    {
        "name": "gnomon_list_datasets",
        "description": (
            "List datasets in the bitemporal store with their observation and "
            "revision counts, their valid- and known-time ranges, and whether "
            "their known times were recorded or assumed. Each carries the "
            "`store:<dataset>` reference to pass as an input."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "store_path": {"type": "string", "description": "Override the temporal-store path."},
            },
            "required": [],
        },
        "runner": _run_list_datasets,
    },
]


def _parse_as_of(raw: Any):
    if not raw:
        return None
    from .data import _parse_timestamp
    return _parse_timestamp(str(raw), 0)


def _run_investigate_change(arguments: dict[str, Any]) -> dict[str, Any]:
    from .context import load_events_file
    from .macros import investigate_change
    events = None
    if arguments.get("context_events_file"):
        events = load_events_file(arguments["context_events_file"])
    payload, path = investigate_change(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
        context_events=events,
        output=arguments.get("output_dir") or "gnomon-output",
    )
    return {**payload, "artifact_path": str(path)}


def _run_route(arguments: dict[str, Any]) -> dict[str, Any]:
    from .pipeline import load_stage
    from .router import route
    from .tracking import TrackingStore
    loaded = load_stage(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
    )
    project = arguments.get("project")
    store = TrackingStore() if project else None
    decisions = [
        route(arguments.get("task") or "forecast",
              [item.value for item in items], loaded.frequency,
              horizon=int(arguments.get("horizon") or 1),
              series=name, project=project, store=store)
        for name, items in sorted(loaded.groups.items())
    ]
    return {"schema_version": "0.1", "decisions": decisions}


def _run_detect_anomalies(arguments: dict[str, Any]) -> dict[str, Any]:
    from .macros import detect_anomalies
    payload, path = detect_anomalies(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
        threshold=(float(arguments["threshold"])
                   if arguments.get("threshold") is not None else None),
        labels=arguments.get("labels"),
        output=arguments.get("output_dir") or "gnomon-output",
    )
    return {**payload, "artifact_path": str(path)}


def _run_decide(arguments: dict[str, Any]) -> dict[str, Any]:
    from .macros import decide
    payload, path = decide(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        horizon=int(arguments["horizon"]),
        threshold=float(arguments["threshold"]),
        actions=list(arguments["actions"]),
        utilities=arguments.get("utilities"),
        max_acceptable_risk=(
            float(arguments["max_acceptable_risk"])
            if arguments.get("max_acceptable_risk") is not None else None
        ),
        series_column=arguments.get("series_column"),
        series_name=arguments.get("series_name"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
        project=arguments.get("project"),
        output=arguments.get("output_dir") or "gnomon-output",
    )
    return {**payload, "artifact_path": str(path)}


def _run_status(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    return TrackingStore(create=False).status(arguments.get("project"))


def _run_resolve_outcome(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    artifact = TrackingStore().resolve_decision_outcome(
        str(arguments["decision_id"]),
        realised_scenario=arguments.get("realised_scenario"),
        realised_utilities=arguments.get("realised_utilities"),
        constraint_violations=arguments.get("constraint_violations"),
        note=arguments.get("note"),
    )
    return {"status": "ok", "decision": artifact.to_dict()}


def _run_monitor(arguments: dict[str, Any]) -> dict[str, Any]:
    from .macros import monitor
    payload, path = monitor(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        horizon=int(arguments["horizon"]),
        threshold=float(arguments["threshold"]),
        alert_cost=float(arguments["alert_cost"]) if arguments.get("alert_cost") is not None else None,
        miss_cost=float(arguments["miss_cost"]) if arguments.get("miss_cost") is not None else None,
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
        project=arguments.get("project"),
        output=arguments.get("output_dir") or "gnomon-output",
    )
    return {**payload, "artifact_path": str(path)}


def _run_get_artifact(arguments: dict[str, Any]) -> dict[str, Any]:
    from pathlib import Path
    from .artifacts import read_artifact
    directory = Path(arguments["artifact_path"])
    payload: dict[str, Any] = {
        "schema_version": "0.1",
        "artifact": read_artifact(directory),
    }
    lineage_path = directory / "lineage.json"
    if arguments.get("include_lineage") and lineage_path.is_file():
        import json as _json
        payload["lineage"] = _json.loads(lineage_path.read_text(encoding="utf-8"))
    return payload


def _run_explain_run(arguments: dict[str, Any]) -> dict[str, Any]:
    """Compact explanation of a stored run: claims, support, warnings.
    Statements come verbatim from the verified lineage — nothing is composed."""
    import json as _json
    from pathlib import Path
    from .artifacts import read_artifact
    directory = Path(arguments["artifact_path"])
    artifact = read_artifact(directory)
    explanation: dict[str, Any] = {
        "schema_version": "0.1",
        "artifact_id": (
            artifact.get("investigation_id") or artifact.get("decision_id")
            or artifact.get("monitor_id") or artifact.get("forecast_id")
        ),
        "created_at": artifact.get("created_at"),
        "support_assessments": {},
        "warnings": {},
        "claims": [],
    }
    for result in artifact.get("results", []):
        name = result.get("series", "__default__")
        if result.get("support_assessment") is not None:
            explanation["support_assessments"][name] = result["support_assessment"]
        if result.get("warnings"):
            explanation["warnings"][name] = result["warnings"]
    if artifact.get("support_assessment") is not None:
        explanation["support_assessments"]["__task__"] = artifact["support_assessment"]
    for trigger in artifact.get("triggers", []):
        explanation["support_assessments"][trigger.get("series", "__default__")] = (
            trigger.get("support_assessment")
        )
    lineage_path = directory / "lineage.json"
    if lineage_path.is_file():
        lineage = _json.loads(lineage_path.read_text(encoding="utf-8"))
        explanation["claims"] = [
            {"claim_id": claim["claim_id"], "claim_class": claim["claim_class"],
             "statement": claim["statement"], "evidence_ids": claim["evidence_ids"]}
            for claim in lineage.get("claims", [])
        ]
    summary = directory / "summary.md"
    if summary.is_file():
        explanation["summary_md"] = summary.read_text(encoding="utf-8")
    return explanation


def _registry_tools() -> list[dict[str, Any]]:
    """Agent tools generated from the macro registry — one source of truth
    for schemas across CLI, Python API, and MCP."""
    from .registry import MACROS
    runners = {
        "gnomon_investigate_change": _run_investigate_change,
        "gnomon_detect_anomalies": _run_detect_anomalies,
        "gnomon_decide": _run_decide,
        "gnomon_monitor": _run_monitor,
    }
    tools = []
    for spec in MACROS.values():
        if spec.tool_name not in runners:
            continue  # gnomon_forecast keeps its frozen v0.2 definition above
        tools.append({
            "name": spec.tool_name,
            "description": spec.summary,
            "inputSchema": spec.input_schema,
            "runner": runners[spec.tool_name],
        })
    return tools


TOOLS.extend(_registry_tools())
TOOLS.extend([
    {
        "name": "gnomon_get_artifact",
        "description": (
            "Read a stored artifact directory: full artifact.json and, "
            "optionally, the typed lineage. All numbers live here; quote them "
            "verbatim."
        ),
        "inputSchema": {"type": "object", "properties": {
            "artifact_path": {"type": "string", "description": "Artifact directory returned by a macro."},
            "include_lineage": {"type": "boolean", "description": "Include lineage.json (artifacts/evidence/claims)."},
        }, "required": ["artifact_path"]},
        "runner": _run_get_artifact,
    },
    {
        "name": "gnomon_status",
        "description": (
            "Pollable status: open forecasts, due horizons, unresolved "
            "decisions, and realised-performance summaries. Descriptive "
            "evidence an agent can cite — never causal."
        ),
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string", "description": "Optional project filter."},
        }, "required": []},
        "runner": _run_status,
    },
    {
        "name": "gnomon_resolve_outcome",
        "description": (
            "The current decision resolver, for DecisionArtifacts produced by "
            "`gnomon_decide`. (The v0.2 `gnomon_record_decision` / "
            "`gnomon_resolve_decision` pair is deprecated and resolves only its "
            "own records.) "
            "Resolve a recorded DecisionArtifact with what actually happened: "
            "realised scenario and/or per-action realised utilities. Returns "
            "realised utility, regret vs the best feasible action in "
            "hindsight, ex-ante optimality, and risk calibration — bare "
            "'correct' is retired."
        ),
        "inputSchema": {"type": "object", "properties": {
            "decision_id": {"type": "string", "description": (
                "decision_id from a gnomon_decide response (not one minted "
                "for the deprecated gnomon_record_decision)."
            )},
            "realised_scenario": {"type": "string", "description": (
                "Which scenario actually happened. Must be one of the "
                "scenario keys the decision was computed over — for a "
                "threshold decision, `exceed` or `no_exceed`."
            )},
            "realised_utilities": {"type": "object", "description": "Optional per-action realised payoff."},
            "constraint_violations": {"type": "array", "items": {"type": "string"}, "description": "Names of constraints that turned out violated."},
            "note": {"type": "string", "description": "Optional free-text context for the resolution."},
        }, "required": ["decision_id"]},
        "runner": _run_resolve_outcome,
    },
    {
        "name": "gnomon_route",
        "description": (
            "Which method for this task on this data? A disclosed, advisory "
            "routing decision: verified capability filter, then a realised-"
            "performance prior from the tracking store when enough scored "
            "history exists (never claimed cold), with the series fingerprint "
            "and every exclusion reason in the output. Feed `candidates` (or "
            "`recommendation`) to `gnomon_forecast`'s `candidates` parameter to "
            "act on the answer. Evaluated runs still backtest whatever pool "
            "they are given against the mandatory baselines, so routing "
            "narrows the contest but never decides it."
        ),
        "inputSchema": {"type": "object", "properties": {
            **_INPUT_PROPERTIES,
            "task": {"type": "string", "enum": ["forecast", "detect_anomalies"],
                     "description": "Task to route (default forecast)."},
            "horizon": {"type": "integer", "description": "Forecast horizon (default 1)."},
            "project": {"type": "string", "description": (
                "Tracking project: consults the realised-performance prior and "
                "records the routing decision for replay."
            )},
        }, "required": ["input", "time_column", "target_column"]},
        "runner": _run_route,
    },
    {
        "name": "gnomon_explain_run",
        "description": (
            "Compact explanation of a stored run: verified claim statements, "
            "per-series support assessments, and warnings. Statements come "
            "from the verified lineage; never paraphrase abstentions away."
        ),
        "inputSchema": {"type": "object", "properties": {
            "artifact_path": {"type": "string", "description": "Artifact directory returned by a macro."},
        }, "required": ["artifact_path"]},
        "runner": _run_explain_run,
    },
    {
        "name": "gnomon_proposer_skill",
        "description": ("How good has each context-event proposer been? "
                        "Read a per-proposer, per-event-type skill estimate "
                        "from resolved context-event proposals (realised lift "
                        "vs the history-only counterfactual, in WAPE, "
                        "shrunk toward zero at low sample counts). "
                        "Observational; attribution is set-level because the "
                        "admission gate decides on event sets."),
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string"},
            "proposer_id": {"type": "string"},
            "event_type": {"type": "string"},
        }, "required": ["project"]},
        "runner": _run_proposer_skill,
    },
])


# --- Experimental planner surface (advanced; macros remain the default) ---

def planner_enabled() -> bool:
    import os
    return os.environ.get("GNOMON_EXPERIMENTAL_PLANNER") == "1"


def _run_compile_task(arguments: dict[str, Any]) -> dict[str, Any]:
    from dataclasses import asdict
    from .plan import compile_task
    plan = compile_task(str(arguments["task_type"]), dict(arguments.get("params") or {}))
    return {"schema_version": "0.1", "plan_id": plan.plan_id(),
            "plan": {"task": plan.task, "steps": [asdict(step) for step in plan.steps]}}


def _run_validate_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    from .plan import plan_from_dict, validate_plan
    plan = plan_from_dict(dict(arguments["plan"]))
    violations = validate_plan(plan)
    return {"schema_version": "0.1", "plan_id": plan.plan_id(),
            "valid": not violations, "violations": violations}


def _run_execute_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    from .execution import execute_plan
    from .plan import plan_from_dict
    payload, path = execute_plan(
        plan_from_dict(dict(arguments["plan"])),
        output=arguments.get("output_dir") or "gnomon-output",
        as_of=_parse_as_of(arguments.get("as_of")),
        store_path=arguments.get("store_path"),
    )
    return {**payload, "artifact_path": str(path)}


def _run_get_run(arguments: dict[str, Any]) -> dict[str, Any]:
    from .artifacts import read_artifact
    return {"schema_version": "0.1", "run": read_artifact(arguments["run_path"])}


_PLAN_PROPERTY = {"type": "object", "description": "A TemporalPlan: {task, steps: [{step_id, operator, inputs, produces, failure_policy}]}."}

PLANNER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "gnomon_compile_task",
        "description": "Compile one of the four canonical task types into a validated TemporalPlan (experimental).",
        "inputSchema": {"type": "object", "properties": {
            "task_type": {"type": "string", "enum": ["forecast", "investigate_change", "decide", "monitor"]},
            "params": {"type": "object", "description": "The macro's parameters."},
        }, "required": ["task_type", "params"]},
        "runner": _run_compile_task,
    },
    {
        "name": "gnomon_validate_plan",
        "description": "Deterministically validate a TemporalPlan: operators, references, leakage, claim-class feasibility, budget, duplicates (experimental).",
        "inputSchema": {"type": "object", "properties": {"plan": _PLAN_PROPERTY},
                        "required": ["plan"]},
        "runner": _run_validate_plan,
    },
    {
        "name": "gnomon_execute_plan",
        "description": "Execute a validated TemporalPlan with step checkpointing, content-addressed caching, and deterministic replay (experimental).",
        "inputSchema": {"type": "object", "properties": {
            "plan": _PLAN_PROPERTY,
            "output_dir": {"type": "string"},
            "as_of": {"type": "string"},
            "store_path": {"type": "string"},
        }, "required": ["plan"]},
        "runner": _run_execute_plan,
    },
    {
        "name": "gnomon_get_run",
        "description": "Read a stored plan-run artifact: step provenance and outputs (experimental).",
        "inputSchema": {"type": "object", "properties": {
            "run_path": {"type": "string", "description": "Run artifact directory."},
        }, "required": ["run_path"]},
        "runner": _run_get_run,
    },
]


def visible_tools() -> list[dict[str, Any]]:
    """The tool surface as gated for this process: macros always; the raw
    planner only behind GNOMON_EXPERIMENTAL_PLANNER=1."""
    return TOOLS + (PLANNER_TOOLS if planner_enabled() else [])


def runner_for(name: str) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    for tool in visible_tools():
        if tool["name"] == name:
            return tool["runner"]
    return None
