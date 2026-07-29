"""OpenAI-style function schemas for the Aion tools exposed to Hermes.

Deliberately three tools, mirroring the v0.1 CLI exactly. Richer surfaces
(explain, score, compare, context evaluation) arrive only after these prove
themselves in agent hands — every extra tool competes for the
orchestrator's attention.
"""

from __future__ import annotations

_INPUT_PROPERTIES = {
    "input": {
        "type": "string",
        "description": "Path to a local CSV (or Parquet, if installed) file of time-series observations.",
    },
    "time_column": {
        "type": "string",
        "description": "Name of the timestamp column.",
    },
    "target_column": {
        "type": "string",
        "description": "Name of the numeric column to forecast.",
    },
    "series_column": {
        "type": "string",
        "description": "Optional column identifying independent series in a panel file.",
    },
    "frequency": {
        "type": "string",
        "enum": ["h", "D", "W", "MS"],
        "description": "Observation frequency: hourly, daily, weekly, or month-start. Omit to let Aion infer it; ambiguity fails loudly rather than guessing.",
    },
}

AION_CAPABILITIES_SCHEMA = {
    "name": "aion_capabilities",
    "description": (
        "Report what the installed Aion runtime actually supports (inputs, "
        "frequencies, models, features). Call this for feature detection "
        "instead of assuming a capability exists."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

AION_INSPECT_SCHEMA = {
    "name": "aion_inspect",
    "description": (
        "Validate a temporal dataset before forecasting: schema mapping, "
        "frequency, duplicates, missing periods, and history sufficiency. "
        "Prefer this before aion_forecast when column mappings or data "
        "quality are uncertain."
    ),
    "parameters": {
        "type": "object",
        "properties": dict(_INPUT_PROPERTIES),
        "required": ["input", "time_column", "target_column"],
    },
}

AION_FORECAST_SCHEMA = {
    "name": "aion_forecast",
    "description": (
        "Run Aion's evaluated forecast: baselines and candidates are "
        "backtested on rolling folds, a model is selected per series or the "
        "series is marked unsupported (abstention). Returns a compact result "
        "with support status, selected model, warnings, and the artifact "
        "directory; read forecast.csv / summary.md there for the numbers and "
        "quote them verbatim. Never invent values for an unsupported series."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **_INPUT_PROPERTIES,
            "horizon": {
                "type": "integer",
                "description": "Number of future periods to forecast, in units of the data frequency.",
            },
            "output_dir": {
                "type": "string",
                "description": "Directory for the immutable forecast artifact (default: ./aion-output).",
            },
            "minimum_baseline_improvement": {
                "type": "number",
                "description": "Minimum relative improvement over the strongest baseline required to select a candidate model (default 0.02).",
            },
        },
        "required": ["input", "time_column", "target_column", "horizon"],
    },
}
