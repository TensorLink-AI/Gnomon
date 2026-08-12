"""OpenAI-style function schemas for the Gnomon tools exposed to Hermes.

These schemas mirror the stable Hermes-facing CLI views. Gnomon's MCP server
has a broader expert surface; the plugin keeps a smaller, task-oriented set so
every additional tool earns its place in the agent prompt.
"""

from __future__ import annotations

_INPUT_PROPERTIES = {
    "input": {
        "type": "string",
        "description": "Path to a local CSV (or Parquet, if installed) file of time-series observations.",
    },
    "time_column": {
        "type": "string",
        "description": (
            "Name of the timestamp column. Omit to let Gnomon infer it when "
            "exactly one column parses as timestamps; the inference is "
            "disclosed in the result's assumptions, and ambiguity fails "
            "loudly with the candidate columns."
        ),
    },
    "target_column": {
        "type": "string",
        "description": (
            "Name of the numeric column to forecast. Omit to let Gnomon "
            "infer it when exactly one non-time column parses as numbers; "
            "disclosed and refused the same way as time_column."
        ),
    },
    "series_column": {
        "type": "string",
        "description": "Optional column identifying independent series in a panel file.",
    },
    "frequency": {
        "type": "string",
        "pattern": "^([1-9][0-9]*)?(s|min|h)$|^(D|W|MS)$",
        "description": "Observation frequency: s (seconds), min (minutes), h (hourly), D (daily), W (weekly), MS (month-start), or any whole-second sub-daily step as <N>s, <N>min, or <N>h (e.g. 5min, 90s, 2h). Omit to let Gnomon infer it; ambiguity fails loudly rather than guessing.",
    },
}

GNOMON_CAPABILITIES_SCHEMA = {
    "name": "gnomon_capabilities",
    "description": (
        "Report what the installed Gnomon runtime actually supports (inputs, "
        "frequencies, models, features). Call this for feature detection "
        "instead of assuming a capability exists."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

GNOMON_INSPECT_SCHEMA = {
    "name": "gnomon_inspect",
    "description": (
        "Validate a temporal dataset before forecasting: schema mapping, "
        "frequency, duplicates, missing periods, and history sufficiency. "
        "Prefer this before gnomon_forecast when column mappings or data "
        "quality are uncertain."
    ),
    "parameters": {
        "type": "object",
        "properties": dict(_INPUT_PROPERTIES),
        "required": ["input"],
    },
}

GNOMON_FORECAST_SCHEMA = {
    "name": "gnomon_forecast",
    "description": (
        "Run Gnomon's evaluated forecast: baselines and candidates are "
        "backtested on rolling folds, a model is selected per series or the "
        "series is marked unsupported (abstention). Returns a compact result "
        "with support status, selected model, warnings, and the artifact "
        "directory; read forecast.csv / summary.md there for the numbers and "
        "quote them verbatim. Never invent values for an unsupported series. "
        "Before first use, load the gnomon:forecasting skill for the full "
        "workflow and safe-use rules."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **_INPUT_PROPERTIES,
            "horizon": {
                "type": "integer",
                "description": (
                    "Number of future periods to forecast, in units of the "
                    "data frequency. Omit to default to one seasonal period "
                    "of the inferred grid, disclosed as an assumption."
                ),
            },
            "output_dir": {
                "type": "string",
                "description": "Directory for the immutable forecast artifact (default: ./gnomon-output).",
            },
            "minimum_baseline_improvement": {
                "type": "number",
                "description": "Minimum relative improvement over the strongest baseline required to select a candidate model (default 0.02).",
            },
            "context_events_file": {
                "type": "string",
                "description": (
                    "Optional validated context-events JSON file produced by "
                    "gnomon_propose_context_events. Events are admitted into the "
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
            "repair": {
                "type": "string",
                "enum": ["off", "safe", "aggressive"],
                "description": (
                    "Messy-data handling (default safe): safe normalises cell "
                    "text with disclosure; aggressive additionally fills gaps, "
                    "snaps timestamps, and resolves conflicts — capped, and "
                    "every fix is reported in evidence and warnings."
                ),
            },
        },
        "required": ["input"],
    },
}

GNOMON_COVARIATE_GUIDE_SCHEMA = {
    "name": "gnomon_covariate_guide",
    "description": "Return temporal constraints and the point-in-time CSV contract. You decide what to fetch; Gnomon defines how to represent it.",
    "parameters": {"type": "object", "properties": {
        **_INPUT_PROPERTIES, "horizon": {"type": "integer"},
    }, "required": ["input", "horizon"]},
}

GNOMON_VALIDATE_COVARIATES_SCHEMA = {
    "name": "gnomon_validate_covariates",
    "description": "Validate local covariate vintages for alignment, horizon coverage, and historical availability without fetching URLs.",
    "parameters": {"type": "object", "properties": {
        **_INPUT_PROPERTIES, "horizon": {"type": "integer"},
        "covariates_file": {"type": "string"},
        "covariate_mapping": {"type": "string"},
        "covariate_time_column": {"type": "string"},
        "covariate_known_at_column": {"type": "string"},
        "covariate_series_column": {"type": "string"},
    }, "required": ["input", "horizon", "covariates_file", "covariate_mapping"]},
}

GNOMON_PROPOSE_COVARIATES_SCHEMA = {
    **GNOMON_FORECAST_SCHEMA,
    "name": "gnomon_propose_covariates",
    "description": "Evaluate a local covariate proposal on identical rolling folds and retain only features with stable material lift.",
    "parameters": {
        **GNOMON_FORECAST_SCHEMA["parameters"],
        "required": [
            "input", "horizon", "covariates_file", "covariate_mapping",
        ],
    },
}

GNOMON_SUBMIT_ACTUALS_SCHEMA = {
    "name": "gnomon_submit_actuals",
    "description": "Score due forecasts from a complete actuals CSV.",
    "parameters": {"type": "object", "properties": {
        "project": {"type": "string"}, "actuals_file": {"type": "string"},
    }, "required": ["project", "actuals_file"]},
}

GNOMON_LIST_OPEN_SCHEMA = {
    "name": "gnomon_list_open_forecasts",
    "description": "List unscored forecasts and whether each complete horizon is due.",
    "parameters": {"type": "object", "properties": {
        "project": {"type": "string"},
    }, "required": []},
}

GNOMON_MODEL_PERFORMANCE_SCHEMA = {
    "name": "gnomon_model_performance",
    "description": "Read descriptive realised performance; rankings are observational, not causal.",
    "parameters": {"type": "object", "properties": {
        "project": {"type": "string"}, "model": {"type": "string"},
    }, "required": ["project"]},
}

GNOMON_RECORD_DECISION_SCHEMA = {
    "name": "gnomon_record_decision",
    "description": "Link an agent action and expected outcome to a tracked forecast.",
    "parameters": {"type": "object", "properties": {
        "decision_id": {"type": "string"}, "project": {"type": "string"},
        "forecast_id": {"type": "string"}, "action": {"type": "string"},
        "expected_outcome": {"type": "string"},
    }, "required": ["decision_id", "project", "forecast_id", "action", "expected_outcome"]},
}

GNOMON_RESOLVE_DECISION_SCHEMA = {
    "name": "gnomon_resolve_decision",
    "description": "Record the realised outcome and correctness of an agent decision.",
    "parameters": {"type": "object", "properties": {
        "decision_id": {"type": "string"}, "actual_outcome": {"type": "string"},
        "correct": {"type": "boolean"},
    }, "required": ["decision_id", "actual_outcome", "correct"]},
}

GNOMON_PROPOSE_CONTEXT_SCHEMA = {
    "name": "gnomon_propose_context_events",
    "description": (
        "Extract candidate context events (launches, promotions, outages, "
        "holidays) from explicitly permitted local documents using an "
        "Gnomon-owned prompt run on the host LLM, then validate them "
        "deterministically. Returns typed events plus rejected proposals with "
        "reasons, and writes an events file for gnomon_forecast. Events without "
        "a verifiable dated source are never used in backtests. Only use "
        "documents the user has allowed you to read. The gnomon:forecasting "
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
                "description": "Series names from the dataset the events may scope to (from gnomon_inspect).",
            },
            "output_file": {
                "type": "string",
                "description": "Where to write the validated events JSON (default ./gnomon-context-events.json).",
            },
        },
        "required": ["files"],
    },
}


GNOMON_INVESTIGATE_SCHEMA = {
    "name": "gnomon_investigate_change",
    "description": (
        "What changed? Detect changepoints in a series, locate the onset, "
        "classify regime shift versus transient anomaly, and rank concurrent "
        "events and cross-series precedence as associational explanations "
        "with residual uncertainty. Never returns a cause; report the "
        "ranking as associational, exactly as Gnomon states it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **_INPUT_PROPERTIES,
            "context_events_file": {
                "type": "string",
                "description": "Optional validated context-events JSON (output of gnomon_propose_context_events) to rank as concurrent events.",
            },
            "as_of": {
                "type": "string",
                "description": "Optional ISO instant: investigate using only data known at or before this moment.",
            },
            "output_dir": {
                "type": "string",
                "description": "Directory for the immutable artifact (default ./gnomon-output).",
            },
        },
        "required": ["input"],
    },
}

GNOMON_DETECT_SCHEMA = {
    "name": "gnomon_detect_anomalies",
    "description": (
        "What is abnormal? Grade competing anomaly detectors using supplied "
        "labels or injected anomalies, then return the winning detector's "
        "flags with every candidate grade and the support assessment. Report "
        "the selection basis and support; never turn a flag into a cause."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **_INPUT_PROPERTIES,
            "threshold": {
                "type": "number",
                "description": "Optional threshold on standardised anomaly scores (runtime default 3.5).",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional ISO timestamps of known anomalies; detector selection then uses label F1.",
            },
            "as_of": {
                "type": "string",
                "description": "Optional ISO instant: detect using only data known at or before this moment.",
            },
            "output_dir": {
                "type": "string",
                "description": "Directory for the immutable artifact (default ./gnomon-output).",
            },
        },
        "required": ["input"],
    },
}

GNOMON_DECIDE_SCHEMA = {
    "name": "gnomon_decide",
    "description": (
        "What should we do? Builds exceedance scenarios from an evaluated "
        "forecast, checks feasibility and constraints over candidate "
        "actions, and chooses by expected utility. Without utilities it "
        "returns the feasible-action comparison and probabilities as "
        "conditionally_supported (missing utility inputs) — report that "
        "honestly and ask the user for utilities if they want a choice. "
        "Never infer business thresholds or payoffs yourself."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **_INPUT_PROPERTIES,
            "horizon": {"type": "integer", "description": "Future periods to forecast, in units of the data frequency."},
            "threshold": {"type": "number", "description": "Decision threshold on the target metric, supplied by the user."},
            "actions": {
                "type": "array", "items": {"type": "object"},
                "description": "Candidate actions: [{name, feasible?, residual_risk?}].",
            },
            "utilities": {
                "type": "object",
                "description": "Optional payoff per action per scenario: {action: {exceed: x, no_exceed: y}}. Omit for the degraded comparison.",
            },
            "max_acceptable_risk": {"type": "number", "description": "Optional constraint on each action's residual_risk."},
            "series_name": {"type": "string", "description": "Series to decide for (required when the file holds several)."},
            "project": {"type": "string", "description": "Optional tracking project: records a DecisionArtifact for realised-outcome scoring."},
            "as_of": {"type": "string", "description": "Optional ISO instant for historical replay."},
            "output_dir": {"type": "string", "description": "Directory for the immutable artifact (default ./gnomon-output)."},
        },
        "required": ["input", "horizon", "threshold", "actions"],
    },
}

GNOMON_MONITOR_SCHEMA = {
    "name": "gnomon_monitor",
    "description": (
        "When should we intervene? Sequential exceedance risk per horizon "
        "step and an alert rule. With alert_cost and miss_cost the rule is "
        "cost-optimal; without them it uses a flagged 0.5 default and the "
        "result is conditionally_supported — say so, and ask for costs "
        "rather than inventing them."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **_INPUT_PROPERTIES,
            "horizon": {"type": "integer", "description": "Future periods to monitor."},
            "threshold": {"type": "number", "description": "Trigger threshold on the target metric."},
            "alert_cost": {"type": "number", "description": "Cost of an unnecessary alert."},
            "miss_cost": {"type": "number", "description": "Cost of a missed exceedance."},
            "project": {"type": "string", "description": "Optional tracking project for the underlying forecast."},
            "as_of": {"type": "string", "description": "Optional ISO instant for historical replay."},
            "output_dir": {"type": "string", "description": "Directory for the immutable artifact (default ./gnomon-output)."},
        },
        "required": ["input", "horizon", "threshold"],
    },
}
