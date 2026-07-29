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
        "enum": ["min", "5min", "15min", "30min", "h", "D", "W", "MS"],
        "description": "Observation frequency: min/5min/15min/30min (minutes), h (hourly), D (daily), W (weekly), MS (month-start). Omit to let Aion infer it; ambiguity fails loudly rather than guessing.",
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
        "quote them verbatim. Never invent values for an unsupported series. "
        "Before first use, load the aion:forecasting skill for the full "
        "workflow and safe-use rules."
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
            "context_events_file": {
                "type": "string",
                "description": (
                    "Optional validated context-events JSON file produced by "
                    "aion_propose_context_events. Events are admitted into the "
                    "forecast only if they demonstrate stable improvement on "
                    "identical backtest folds."
                ),
            },
            "threshold": {
                "type": "number",
                "description": (
                    "Optional decision threshold: the result reports when and "
                    "how likely the forecast crosses this value."
                ),
            },
            "project": {
                "type": "string",
                "description": "Optional tracking project for realised scoring and decision outcomes.",
            },
            "covariates_file": {"type": "string", "description": "Local point-in-time covariate CSV."},
            "covariate_mapping": {"type": "string", "description": "Comma-separated name:type:future_known entries."},
            "covariate_time_column": {"type": "string", "description": "Valid-at column (default timestamp)."},
            "covariate_known_at_column": {"type": "string", "description": "Availability timestamp column (default known_at)."},
            "covariate_series_column": {"type": "string", "description": "Optional covariate series column."},
        },
        "required": ["input", "time_column", "target_column", "horizon"],
    },
}

AION_COVARIATE_GUIDE_SCHEMA = {
    "name": "aion_covariate_guide",
    "description": "Return temporal constraints and the point-in-time CSV contract. You decide what to fetch; Aion defines how to represent it.",
    "parameters": {"type": "object", "properties": {
        **_INPUT_PROPERTIES, "horizon": {"type": "integer"},
    }, "required": ["input", "time_column", "target_column", "horizon"]},
}

AION_VALIDATE_COVARIATES_SCHEMA = {
    "name": "aion_validate_covariates",
    "description": "Validate local covariate vintages for alignment, horizon coverage, and historical availability without fetching URLs.",
    "parameters": {"type": "object", "properties": {
        **_INPUT_PROPERTIES, "horizon": {"type": "integer"},
        "covariates_file": {"type": "string"},
        "covariate_mapping": {"type": "string"},
        "covariate_time_column": {"type": "string"},
        "covariate_known_at_column": {"type": "string"},
        "covariate_series_column": {"type": "string"},
    }, "required": ["input", "time_column", "target_column", "horizon", "covariates_file", "covariate_mapping"]},
}

AION_PROPOSE_COVARIATES_SCHEMA = {
    **AION_FORECAST_SCHEMA,
    "name": "aion_propose_covariates",
    "description": "Evaluate a local covariate proposal on identical rolling folds and retain only features with stable material lift.",
    "parameters": {
        **AION_FORECAST_SCHEMA["parameters"],
        "required": [
            "input", "time_column", "target_column", "horizon",
            "covariates_file", "covariate_mapping",
        ],
    },
}

AION_SUBMIT_ACTUALS_SCHEMA = {
    "name": "aion_submit_actuals",
    "description": "Score due forecasts from a complete actuals CSV.",
    "parameters": {"type": "object", "properties": {
        "project": {"type": "string"}, "actuals_file": {"type": "string"},
    }, "required": ["project", "actuals_file"]},
}

AION_LIST_OPEN_SCHEMA = {
    "name": "aion_list_open_forecasts",
    "description": "List unscored forecasts and whether each complete horizon is due.",
    "parameters": {"type": "object", "properties": {
        "project": {"type": "string"},
    }, "required": []},
}

AION_MODEL_PERFORMANCE_SCHEMA = {
    "name": "aion_model_performance",
    "description": "Read descriptive realised performance; rankings are observational, not causal.",
    "parameters": {"type": "object", "properties": {
        "project": {"type": "string"}, "model": {"type": "string"},
    }, "required": ["project"]},
}

AION_RECORD_DECISION_SCHEMA = {
    "name": "aion_record_decision",
    "description": "Link an agent action and expected outcome to a tracked forecast.",
    "parameters": {"type": "object", "properties": {
        "decision_id": {"type": "string"}, "project": {"type": "string"},
        "forecast_id": {"type": "string"}, "action": {"type": "string"},
        "expected_outcome": {"type": "string"},
    }, "required": ["decision_id", "project", "forecast_id", "action", "expected_outcome"]},
}

AION_RESOLVE_DECISION_SCHEMA = {
    "name": "aion_resolve_decision",
    "description": "Record the realised outcome and correctness of an agent decision.",
    "parameters": {"type": "object", "properties": {
        "decision_id": {"type": "string"}, "actual_outcome": {"type": "string"},
        "correct": {"type": "boolean"},
    }, "required": ["decision_id", "actual_outcome", "correct"]},
}

AION_PROPOSE_CONTEXT_SCHEMA = {
    "name": "aion_propose_context_events",
    "description": (
        "Extract candidate context events (launches, promotions, outages, "
        "holidays) from explicitly permitted local documents using an "
        "Aion-owned prompt run on the host LLM, then validate them "
        "deterministically. Returns typed events plus rejected proposals with "
        "reasons, and writes an events file for aion_forecast. Events without "
        "a verifiable dated source are never used in backtests. Only use "
        "documents the user has allowed you to read. The aion:forecasting "
        "skill documents how to report admission decisions honestly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Paths of permitted local documents to scan (planning notes, calendars, release notes).",
            },
            "series_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Series names from the dataset the events may scope to (from aion_inspect).",
            },
            "output_file": {
                "type": "string",
                "description": "Where to write the validated events JSON (default ./aion-context-events.json).",
            },
        },
        "required": ["files"],
    },
}
