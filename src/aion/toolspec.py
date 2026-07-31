"""Canonical machine-facing tool specifications.

One definition of Aion's agent-facing tools — names, JSON Schemas, and
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
    "input": {"type": "string", "description": "Path to a local CSV or Parquet file of time-series observations."},
    "time_column": {"type": "string", "description": "Name of the timestamp column."},
    "target_column": {"type": "string", "description": "Name of the numeric column to forecast."},
    "series_column": {"type": "string", "description": "Optional column identifying independent series."},
    "frequency": {
        "type": "string",
        "enum": ["min", "5min", "15min", "30min", "h", "D", "W", "MS"],
        "description": (
            "Observation frequency: min/5min/15min/30min (minutes), h (hourly), "
            "D (daily), W (weekly), MS (month start). Omit to infer; ambiguity "
            "fails loudly."
        ),
    },
}

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


def _run_capabilities(arguments: dict[str, Any]) -> dict[str, Any]:
    return capabilities()


def _run_inspect(arguments: dict[str, Any]) -> dict[str, Any]:
    return inspect_dataset(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
    )


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
        output=arguments.get("output_dir") or "aion-output",
        minimum_baseline_improvement=float(arguments.get("minimum_baseline_improvement", 0.02)),
        context_events=events,
        covariates=covariates,
        threshold=float(arguments["threshold"]) if arguments.get("threshold") is not None else None,
        repair=arguments.get("repair", "safe"),
    )
    payload = forecast_summary(artifact, path)
    if arguments.get("project"):
        from .tracking import register_artifact
        payload["tracking_ids"] = register_artifact(
            artifact, str(arguments["project"]), str(path),
        )
        payload["project"] = str(arguments["project"])
    return payload


def _run_submit_actuals(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    results = TrackingStore().submit_actuals_csv(
        str(arguments["project"]), str(arguments["actuals_file"]),
    )
    return {"status": "ok", "scored": len(results),
            "results": [item.__dict__ for item in results]}


def _run_open_forecasts(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    rows = TrackingStore().due_forecasts(arguments.get("project"))
    return {"status": "ok", "forecasts": rows}


def _run_model_performance(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    store = TrackingStore()
    if arguments.get("model"):
        rows: Any = store.model_performance(
            str(arguments["project"]), str(arguments["model"]),
        )
    else:
        rows = [item.__dict__ for item in store.leaderboard(str(arguments["project"]))]
    return {"status": "ok", "performance": rows,
            "warning": "Historical telemetry is observational, not causal."}


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
        "name": "aion_capabilities",
        "description": (
            "Report what the installed Aion runtime actually supports. Use for "
            "feature detection instead of assuming a capability exists."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "runner": _run_capabilities,
    },
    {
        "name": "aion_inspect",
        "description": (
            "Validate a temporal dataset before forecasting: schema mapping, "
            "frequency, duplicates, missing periods. Prefer this before "
            "aion_forecast when mappings or data quality are uncertain."
        ),
        "inputSchema": {
            "type": "object",
            "properties": dict(_INPUT_PROPERTIES),
            "required": ["input", "time_column", "target_column"],
        },
        "runner": _run_inspect,
    },
    {
        "name": "aion_forecast",
        "description": (
            "Run Aion's evaluated forecast: baselines and candidates are "
            "backtested on rolling folds; each series gets a selected model or "
            "an unsupported abstention. Context events (from `aion context "
            "validate`) are admitted only when they demonstrate stable lift on "
            "identical folds. Read forecast.csv / summary.md in the returned "
            "artifact directory for the numbers and quote them verbatim; never "
            "invent values for an unsupported series."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **_INPUT_PROPERTIES,
                "horizon": {"type": "integer", "description": "Future periods to forecast, in units of the data frequency."},
                "output_dir": {"type": "string", "description": "Directory for the immutable artifact (default ./aion-output)."},
                "minimum_baseline_improvement": {"type": "number", "description": "Minimum relative improvement over the strongest baseline to select a candidate (default 0.02)."},
                "context_events_file": {"type": "string", "description": "Optional validated context-events JSON file (the output of `aion context validate`)."},
                "threshold": {"type": "number", "description": "Optional decision threshold: the result reports when and how likely the forecast crosses this value."},
                "project": {"type": "string", "description": "Optional tracking project. When set, register the forecast for realised scoring."},
                "covariates_file": {"type": "string", "description": "Local CSV containing point-in-time covariate vintages."},
                "covariate_mapping": {"type": "string", "description": "Comma-separated name:type:future_known entries."},
                "covariate_time_column": {"type": "string", "description": "Valid-at column (default timestamp)."},
                "covariate_known_at_column": {"type": "string", "description": "Availability timestamp column (default known_at)."},
                "covariate_series_column": {"type": "string", "description": "Optional series column in the covariate CSV."},
                "repair": {"type": "string", "enum": ["off", "safe", "aggressive"], "description": "Messy-data handling (default safe): off rejects anything non-strict; safe normalises cell text with disclosure; aggressive additionally fills gaps, snaps timestamps, and resolves conflicts — capped, and every fix is reported in evidence and warnings."},
            },
            "required": ["input", "time_column", "target_column", "horizon"],
        },
        "runner": _run_forecast,
    },
    {
        "name": "aion_covariate_guide",
        "description": "Return point-in-time format, forecast dates, and fold cutoffs. Aion does not suggest what data to fetch.",
        "inputSchema": {"type": "object", "properties": {
            **_INPUT_PROPERTIES,
            "horizon": {"type": "integer"},
        }, "required": ["input", "time_column", "target_column", "horizon"]},
        "runner": _run_covariate_guide,
    },
    {
        "name": "aion_validate_covariates",
        "description": "Validate local covariate vintages for format, final-horizon coverage, and availability at every selection cutoff.",
        "inputSchema": {"type": "object", "properties": {
            **_INPUT_PROPERTIES,
            "horizon": {"type": "integer"},
            "covariates_file": {"type": "string"},
            "covariate_mapping": {"type": "string"},
            "covariate_time_column": {"type": "string"},
            "covariate_known_at_column": {"type": "string"},
            "covariate_series_column": {"type": "string"},
        }, "required": ["input", "time_column", "target_column", "horizon", "covariates_file", "covariate_mapping"]},
        "runner": _run_validate_covariates,
    },
    {
        "name": "aion_propose_covariates",
        "description": "Evaluate a local point-in-time covariate proposal through leakage-safe ablation and produce a forecast only when it earns admission.",
        "inputSchema": {"type": "object", "properties": {
            **_INPUT_PROPERTIES,
            "horizon": {"type": "integer"},
            "output_dir": {"type": "string"},
            "minimum_baseline_improvement": {"type": "number"},
            "covariates_file": {"type": "string"},
            "covariate_mapping": {"type": "string"},
            "covariate_time_column": {"type": "string"},
            "covariate_known_at_column": {"type": "string"},
            "covariate_series_column": {"type": "string"},
        }, "required": ["input", "time_column", "target_column", "horizon", "covariates_file", "covariate_mapping"]},
        "runner": _run_forecast,
    },
    {
        "name": "aion_submit_actuals",
        "description": "Score all due forecasts in a project from complete realised actuals. Panel actuals must include series,timestamp,value.",
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string"}, "actuals_file": {"type": "string"},
        }, "required": ["project", "actuals_file"]},
        "runner": _run_submit_actuals,
    },
    {
        "name": "aion_list_open_forecasts",
        "description": "List unscored forecasts and distinguish horizons that are due from those still awaiting observations.",
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string"},
        }, "required": []},
        "runner": _run_open_forecasts,
    },
    {
        "name": "aion_model_performance",
        "description": "Read descriptive realised model performance for a project. Do not treat observational rankings as causal evidence.",
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string"}, "model": {"type": "string"},
        }, "required": ["project"]},
        "runner": _run_model_performance,
    },
    {
        "name": "aion_record_decision",
        "description": "Link an agent decision and expected outcome to a tracked forecast.",
        "inputSchema": {"type": "object", "properties": {
            "decision_id": {"type": "string"}, "project": {"type": "string"},
            "forecast_id": {"type": "string"}, "action": {"type": "string"},
            "expected_outcome": {"type": "string"},
        }, "required": ["decision_id", "project", "forecast_id", "action", "expected_outcome"]},
        "runner": _run_record_decision,
    },
    {
        "name": "aion_resolve_decision",
        "description": "Record the realised business outcome and whether a previously recorded agent decision was correct.",
        "inputSchema": {"type": "object", "properties": {
            "decision_id": {"type": "string"}, "actual_outcome": {"type": "string"},
            "correct": {"type": "boolean"},
        }, "required": ["decision_id", "actual_outcome", "correct"]},
        "runner": _run_resolve_decision,
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
        output=arguments.get("output_dir") or "aion-output",
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
        output=arguments.get("output_dir") or "aion-output",
    )
    return {**payload, "artifact_path": str(path)}


def _run_status(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    return TrackingStore().status(arguments.get("project"))


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
        output=arguments.get("output_dir") or "aion-output",
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
        "aion_investigate_change": _run_investigate_change,
        "aion_decide": _run_decide,
        "aion_monitor": _run_monitor,
    }
    tools = []
    for spec in MACROS.values():
        if spec.tool_name not in runners:
            continue  # aion_forecast keeps its frozen v0.2 definition above
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
        "name": "aion_get_artifact",
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
        "name": "aion_status",
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
        "name": "aion_resolve_outcome",
        "description": (
            "Resolve a recorded DecisionArtifact with what actually happened: "
            "realised scenario and/or per-action realised utilities. Returns "
            "realised utility, regret vs the best feasible action in "
            "hindsight, ex-ante optimality, and risk calibration — bare "
            "'correct' is retired."
        ),
        "inputSchema": {"type": "object", "properties": {
            "decision_id": {"type": "string"},
            "realised_scenario": {"type": "string", "description": "e.g. exceed / no_exceed."},
            "realised_utilities": {"type": "object", "description": "Optional per-action realised payoff."},
            "constraint_violations": {"type": "array", "items": {"type": "string"}},
            "note": {"type": "string"},
        }, "required": ["decision_id"]},
        "runner": _run_resolve_outcome,
    },
    {
        "name": "aion_explain_run",
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
])


# --- Experimental planner surface (advanced; macros remain the default) ---

def planner_enabled() -> bool:
    import os
    return os.environ.get("AION_EXPERIMENTAL_PLANNER") == "1"


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
        output=arguments.get("output_dir") or "aion-output",
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
        "name": "aion_compile_task",
        "description": "Compile one of the four canonical task types into a validated TemporalPlan (experimental).",
        "inputSchema": {"type": "object", "properties": {
            "task_type": {"type": "string", "enum": ["forecast", "investigate_change", "decide", "monitor"]},
            "params": {"type": "object", "description": "The macro's parameters."},
        }, "required": ["task_type", "params"]},
        "runner": _run_compile_task,
    },
    {
        "name": "aion_validate_plan",
        "description": "Deterministically validate a TemporalPlan: operators, references, leakage, claim-class feasibility, budget, duplicates (experimental).",
        "inputSchema": {"type": "object", "properties": {"plan": _PLAN_PROPERTY},
                        "required": ["plan"]},
        "runner": _run_validate_plan,
    },
    {
        "name": "aion_execute_plan",
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
        "name": "aion_get_run",
        "description": "Read a stored plan-run artifact: step provenance and outputs (experimental).",
        "inputSchema": {"type": "object", "properties": {
            "run_path": {"type": "string", "description": "Run artifact directory."},
        }, "required": ["run_path"]},
        "runner": _run_get_run,
    },
]


def visible_tools() -> list[dict[str, Any]]:
    """The tool surface as gated for this process: macros always; the raw
    planner only behind AION_EXPERIMENTAL_PLANNER=1."""
    return TOOLS + (PLANNER_TOOLS if planner_enabled() else [])


def runner_for(name: str) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    for tool in visible_tools():
        if tool["name"] == name:
            return tool["runner"]
    return None
