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
                "selected_model": item.selected_model,
                "interval_coverage": item.interval_coverage,
                "warnings": item.warnings,
                "forecast_preview": item.forecast[:FORECAST_PREVIEW_ROWS],
                "forecast_rows": len(item.forecast),
                "threshold": item.threshold,
                "context": item.context,
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


def _run_forecast(arguments: dict[str, Any]) -> dict[str, Any]:
    events = None
    if arguments.get("context_events_file"):
        events = load_events_file(arguments["context_events_file"])
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
        threshold=float(arguments["threshold"]) if arguments.get("threshold") is not None else None,
    )
    return forecast_summary(artifact, path)


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
            },
            "required": ["input", "time_column", "target_column", "horizon"],
        },
        "runner": _run_forecast,
    },
]


def runner_for(name: str) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    for tool in TOOLS:
        if tool["name"] == name:
            return tool["runner"]
    return None
