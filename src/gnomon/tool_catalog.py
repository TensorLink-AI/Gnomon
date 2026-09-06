"""Legacy tool descriptions and schemas. Runner names are data, not imported execution code."""

from __future__ import annotations


from typing import Any

from .tool_schema import (
    CONTEXT_EVENTS_PROPERTY as _CONTEXT_EVENTS_PROPERTY,
    COVARIATE_MAPPING_PROPERTY as _COVARIATE_MAPPING_PROPERTY,
    COVARIATES_PROPERTY as _COVARIATES_PROPERTY,
    INPUT_PROPERTIES as _INPUT_PROPERTIES,
    OBSERVATIONS_PROPERTY as _OBSERVATIONS_PROPERTY,
    REPLAY_PROPERTIES as _REPLAY_PROPERTIES,
    TEMPORAL_QUESTIONS_PROPERTY as _TEMPORAL_QUESTIONS_PROPERTY,
)


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "gnomon_capabilities",
        "description": (
            "Report what the installed Gnomon runtime supports. Use only for "
            "explicit feature discovery, never as a prerequisite to forecast, "
            "describe, or inspect. The "
            "default view is brief (every section and capability name, long "
            "prose elided); pass format 'full' or sections for the verbatim "
            "detail."
        ),
        "inputSchema": {"type": "object", "properties": {
            "format": {"type": "string", "enum": ["brief", "full"],
                       "description": (
                "brief (default): every section with long prose elided, "
                "within the response budget; full: the complete payload "
                "verbatim."
            )},
            "sections": {"type": "array", "items": {"type": "string"},
                         "description": (
                "Return only these top-level sections, verbatim. Any "
                "response's view.sections_available lists the names."
            )},
        }, "required": []},
        "runner": "_run_capabilities",
    },
    {
        "name": "gnomon_inspect",
        "description": (
            "Validate a temporal dataset before forecasting: schema mapping, "
            "frequency, duplicates, missing periods. Prefer this before "
            "gnomon_forecast when mappings or data quality are uncertain. "
            "target_column takes a comma list or \"auto\" to inspect every "
            "channel of a wide file in one call (the default when several "
            "columns qualify)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **_INPUT_PROPERTIES,
                "target_column": {"type": "string", "description": (
                    "Numeric column to inspect, a comma list "
                    "(`\"cpu,mem\"`), or `\"auto\"` (every numeric "
                    "non-time column) — one report per channel. Omit to "
                    "infer: a lone qualifying column is chosen, several "
                    "are all inspected; either is disclosed as an "
                    "assumption. Required for store:<dataset> inputs."
                )},
                **_REPLAY_PROPERTIES,
            },
            "required": [],
            "anyOf": [
                {"required": ["input"]},
                {"required": ["observations"]},
                {"required": ["data_ref"]},
            ],
        },
        "runner": "_run_inspect",
    },
    {
        "name": "gnomon_describe",
        "description": (
            "Answer typed temporal questions without changing a primary: "
            "description, stationarity, fixed-period decomposition, or "
            "exogenous regression. Unsupported methods fail typed; semantic "
            "substitution is forbidden."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **_INPUT_PROPERTIES,
                "target_column": {"type": "string", "description": (
                    "Numeric column, comma list, or auto for every numeric channel."
                )},
                **_REPLAY_PROPERTIES,
                **_TEMPORAL_QUESTIONS_PROPERTY,
                "format": {"type": "string", "enum": ["brief", "full"],
                           "description": ("brief returns compact typed "
                               "answers and per-series diagnostics; full "
                               "returns complete reasoning receipts.")},
            },
            "required": [],
            "anyOf": [
                {"required": ["input"]},
                {"required": ["observations"]},
                {"required": ["data_ref"]},
            ],
        },
        "runner": "_run_describe",
    },
    {
        "name": "gnomon_forecast",
        "description": (
            "Forecast columns (`\"cpu,mem,requests\"` or `\"auto\"`). "
            "Infer schema; backtest candidates; disclose weak support."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **_INPUT_PROPERTIES,
                **_REPLAY_PROPERTIES,
                "target_column": {"type": "string", "description": (
                    "Column, `\"cpu,mem,requests\"`, or `\"auto\"`; omit only "
                    "when unambiguous."
                )},
                "horizon": {"type": "integer", "description": (
                    "Future periods, in units of the data frequency. "
                    "Default: one seasonal period, disclosed as an "
                    "assumption."
                )},
                "format": {"type": "string", "enum": ["full", "brief"], "description": (
                    "brief (default): q50, q10-q90 and disclosures; full "
                    "adds quantiles. The artifact is identical."
                )},
                "candidates": {
                    "type": "array", "items": {"type": "string"},
                    "description": (
                        "Exact allowlist; baselines compete and may win."
                    ),
                },
                "model_admission": {"type": "string", "enum": ["strict", "evidence_weighted"], "description": "Default: strict."},
                "model_evidence_registry": {"type": "string", "description": "Registry for evidence_weighted."},
                "output_dir": {"type": "string", "description": (
                    "Artifact directory; default from gnomon_capabilities."
                )},
                "minimum_baseline_improvement": {"type": "number", "minimum": 0, "description": "Required relative gain over baseline (default 0.02)."},
                "context_events_file": {"type": "string", "description": "Validated context-events JSON."},
                **_CONTEXT_EVENTS_PROPERTY,
                "threshold": {"type": "number", "description": "Optional decision threshold: the result reports when and how likely the forecast crosses this value."},
                "project": {"type": "string", "description": "Optional tracking project. When set, register the forecast for realised scoring."},
                "covariates_file": {"type": "string", "description": (
                    "Point-in-time CSV keyed by timestamp and known_at; "
                    "folds cannot see later vintages."
                )},
                **_COVARIATES_PROPERTY,
                **_COVARIATE_MAPPING_PROPERTY,
                "covariate_time_column": {"type": "string", "description": "Valid-at column (default timestamp)."},
                "covariate_known_at_column": {"type": "string", "description": "Availability timestamp column (default known_at)."},
                **_TEMPORAL_QUESTIONS_PROPERTY,
                "covariate_series_column": {"type": "string", "description": "Optional series column in the covariate CSV."},
                "repair": {"type": "string", "enum": ["off", "safe", "aggressive"], "description": "Repair: off strict; safe aligns bounded jitter; aggressive also fills gaps/conflicts. All disclosed."},
                "best_effort": {"type": "boolean", "description": (
                    "Deprecated alias for minimum_support=best_effort."
                )},
                "minimum_support": {"type": "string",
                                    "enum": ["supported",
                                             "conditionally_supported",
                                             "best_effort"],
                                    "description": (
                    "Floor (default best_effort); supported refuses weaker results."
                )},
                "publication_mode": {"type": "string",
                    "enum": ["strict", "best_effort", "scenario"],
                    "description": (
                        "strict=evidence-only; best_effort may recommend context; "
                        "scenario lists alternatives.")},
                "temporal_dossiers": {"type": "array", "items": {"type": "object"},
                    "description": "Sealed temporal dossiers."},
                "context_submission": {
                    "type": "object", "additionalProperties": False,
                    "description": (
                        "Context or cited human-only prior; never changes "
                        "primary or automation."),
                    "properties": {
                        "text": {"type": "string"},
                        "known_at": {"type": "string"},
                        "compiler": {"type": "string"},
                        "compile": {"type": "string",
                                    "enum": ["deterministic_linear"]},
                        "allow_prior_compromise": {"type": "boolean"},
                        "proposal": {"type": "object"},
                        "transformations": {"type": "array",
                                            "items": {"type": "object"}},
                        "rejections": {"type": "array"},
                        "model_candidate": {
                            "type": "object", "additionalProperties": False,
                            "description": (
                                "Cited prior: source_spans plus quantiles xor "
                                "3-16 full-grid sample_paths."),
                            "properties": {
                                "source_spans": {"type": "array", "minItems": 1,
                                    "maxItems": 8,
                                    "items": {"type": "string"}},
                                "quantiles": {"type": "array",
                                    "items": {"type": "object"}},
                                "sample_paths": {"type": "array", "minItems": 3,
                                    "maxItems": 16,
                                    "items": {"type": "array",
                                              "items": {"type": "number"}}},
                                "rationale": {"type": "string"},
                                "temperature": {"type": "number", "minimum": 0},
                            },
                            "required": ["source_spans"],
                        },
                    },
                },
                "scenario_selection": {"type": "object",
                    "description": "Number-free governed ranking of scenario ids."},
                "automation_policy": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "authorize": {"type": "boolean", "description": (
                            "Request automation; omission stays advisory.")},
                        "policy_id": {"type": "string", "minLength": 1,
                            "description": "Caller policy id."},
                        "minimum_support": {"type": "string",
                            "enum": ["supported", "context_trusted"],
                            "description": "Required evidence tier."},
                        "action_tier": {"type": "string", "enum": [
                            "advisory", "reversible_low_impact", "high_impact"],
                            "description": (
                                "Impact boundary. Advisory and high-impact "
                                "never actuate; reversible low-impact also "
                                "requires exact artifact-local calibration.")},
                    },
                    "required": ["authorize", "policy_id", "minimum_support"],
                    "description": (
                        "Explicit automation policy; recommendations alone grant no authority.")},
                "future_events": {"type": "boolean", "description": (
                    "Admit verified future constraints (default false); "
                    "retains a history-only counterfactual."
                )},
                "structural_events": {"type": "boolean", "description": (
                    "Recognize typed closed-menu structural events. Quantities "
                    "stay engine-derived; unvalidated effects remain scenarios."
                )},
            },
            "required": [],
            "anyOf": [
                {"required": ["input"]},
                {"required": ["observations"]},
                {"required": ["data_ref"]},
            ],
        },
        "runner": "_run_forecast",
    },
    {
        "name": "gnomon_validate_covariates",
        "description": (
            "Validate covariate vintages for format, coverage, and "
            "availability at every selection cutoff. Format: one row per "
            "(timestamp, known_at); a fold only uses rows known at or "
            "before its cutoff. Mapping grammar: name:type:future_known "
            "entries. Failures name the empty cutoffs; pass the same "
            "arguments to gnomon_forecast for the leakage-safe ablation."
        ),
        "inputSchema": {"type": "object", "properties": {
            **_INPUT_PROPERTIES,
            "horizon": {"type": "integer"},
            "covariates_file": {"type": "string"},
            **_COVARIATES_PROPERTY,
            **_COVARIATE_MAPPING_PROPERTY,
            "covariate_time_column": {"type": "string"},
            "covariate_known_at_column": {"type": "string"},
            "covariate_series_column": {"type": "string"},
        }, "required": ["horizon", "covariate_mapping"]},
        "runner": "_run_validate_covariates",
    },
    {
        "name": "gnomon_submit_actuals",
        "description": "Score all due forecasts in a project from complete realised actuals. Panel actuals must include series,timestamp,value. A forecast scores only when every period in its horizon has an actual; when nothing scores, the result explains which window was missing rather than returning a bare zero.",
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string"},
            "actuals_file": {"type": "string", "description": "CSV of realised values. Callers without a filesystem pass `actuals` inline instead."},
            "actuals": {"type": "array", "items": {"type": "object"}, "description": (
                "Realised values supplied inline: objects of "
                "{timestamp, value, series?, known_at?}. `known_at` (ISO) "
                "backfills when the outcome became knowable; rows without "
                "it became knowable at this submission. Mutually exclusive "
                "with actuals_file (which accepts a known_at column)."
            )},
            "time_column": {"type": "string", "description": "Timestamp column in the actuals file. Inferred from a conventional name or a two-column layout when omitted."},
            "target_column": {"type": "string", "description": "Realised value column. Inferred when unambiguous."},
            "series_column": {"type": "string", "description": "Series column, required for multi-series projects."},
            "effect_occurrences": {"type": "array", "description": (
                "Optional confirmations for tracked context scenarios. Actual "
                "values do not prove an event happened; each item supplies "
                "effect_id, status (confirmed/cancelled/revised), known_at, "
                "and optional note."
            ), "items": {"type": "object", "properties": {
                "effect_id": {"type": "string"},
                "status": {"type": "string", "enum": [
                    "confirmed", "cancelled", "revised"]},
                "known_at": {"type": "string"},
                "note": {"type": "string"},
            }, "required": ["effect_id", "status", "known_at"]}},
        }, "required": ["project"]},
        "runner": "_run_submit_actuals",
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
                "input": {"type": "string", "description": "Path to the CSV to ingest. Callers without a filesystem pass `observations` inline instead."},
                **_OBSERVATIONS_PROPERTY,
                "dataset": {"type": "string", "description": "Dataset name; read it back as `store:<dataset>`."},
                "time_column": {"type": "string", "description": "Valid-time column: when the value applies."},
                "target_column": {"type": "string", "description": "Numeric value column."},
                "known_at_column": {"type": "string", "description": "Known-time column: when the value became knowable. Omit only if the source genuinely has no publication lag."},
                "series_column": {"type": "string", "description": "Optional column identifying independent series."},
                "variable": {"type": "string", "description": "Name to store the measure under (defaults to target_column)."},
                "store_path": {"type": "string", "description": "Override the temporal-store path."},
            },
            "required": ["dataset", "time_column", "target_column"],
        },
        "runner": "_run_ingest",
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
        "runner": "_run_list_datasets",
    },
    {
        "name": "gnomon_preflight_context",
        "description": (
            "Dry-run the admission checks for proposed context events "
            "against the actual data, before spending a forecast. Returns "
            "one verdict per event — would_influence, rejected (with the "
            "typed reason), or ablation_gated (fold admission is measured, "
            "not predictable) — plus the span grammar the parser accepts, "
            "so a rejected proposal can be repaired and resubmitted in one "
            "step. Deterministic verdicts here are the verdicts the "
            "forecast will reach on the same data; nothing is written."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **_INPUT_PROPERTIES,
                "horizon": {"type": "integer", "description": "Future periods the events would apply to, in units of the data frequency."},
                "context_events_file": {"type": "string", "description": "Context-events JSON file to preflight (the output of `gnomon context validate`, or the same shape)."},
                **_CONTEXT_EVENTS_PROPERTY,
                "repair": {"type": "string", "enum": ["off", "safe", "aggressive"], "description": "Messy-data handling, matched to what the forecast will use (default safe)."},
            },
            "required": ["horizon"],
        },
        "runner": "_run_preflight_context",
    },
]


def _registry_tools() -> list[dict[str, Any]]:
    """Agent tools generated from the macro registry — one source of truth
    for schemas across CLI, Python API, and MCP."""
    from .registry import MACROS
    runners = {
        "gnomon_investigate_change": "_run_investigate_change",
        "gnomon_detect_anomalies": "_run_detect_anomalies",
        "gnomon_decide": "_run_decide",
        "gnomon_monitor": "_run_monitor",
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


TOOL_SCHEMAS.extend(_registry_tools())


TOOL_SCHEMAS.extend([
    {
        "name": "gnomon_select_scenario",
        "description": (
            "Choose sealed path. Cannot change numbers, support, primary, "
            "automation."
        ),
        "inputSchema": {"type": "object", "properties": {
            "publication_path": {"type": "string", "description": (
                "publication_path returned by gnomon_forecast.")},
            "scenario_selection": {"type": "object", "properties": {
                "selected_scenario_id": {"type": "string"},
                "ranking": {"type": "array", "items": {"type": "string"}},
                "cited_claim_ids": {"type": "array", "items": {"type": "string"}},
                "counterevidence_claim_ids": {"type": "array", "items": {"type": "string"}},
                "counterevidence_hypothesis_ids": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
                "what_would_change_selection": {"type": "string"},
            }, "required": [
                "selected_scenario_id", "ranking", "cited_claim_ids",
                "counterevidence_claim_ids", "confidence", "rationale",
                "what_would_change_selection",
            ]},
        }, "required": ["publication_path", "scenario_selection"]},
        "runner": "_run_select_scenario",
    },
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
            "series": {"type": ["string", "array"], "description": "Series name or names to return from artifact results."},
            "fields": {"type": "array", "items": {"type": "string"}, "description": "Result fields to return; series is always retained."},
            "where": {"type": "object", "description": "Exact-match filters over result fields."},
            "order_by": {"type": "string", "enum": ["notability", "series"], "description": "Deterministic result ordering."},
            "limit": {"type": "integer", "minimum": 1, "description": "Maximum selected results to return."},
        }, "required": ["artifact_path"]},
        "runner": "_run_get_artifact",
    },
    {
        "name": "gnomon_status",
        "description": (
            "The one tracking read: open forecasts, due horizons, "
            "unresolved decisions, and realised-performance summaries. "
            "Narrow with `section` (open_forecasts / performance / "
            "decisions / effects / effect_prior) to get exactly the tracking evidence "
            "returned. Descriptive evidence an agent can cite — never "
            "causal; do not treat observational rankings as causal "
            "evidence."
        ),
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string", "description": (
                "Optional project filter; required for "
                "section='performance'."
            )},
            "section": {"type": "string",
                        "enum": ["open_forecasts", "performance",
                                 "decisions", "effects", "effect_prior", "all"],
                        "description": (
                            "Slice to return (default all). open_forecasts: "
                            "unscored forecasts with due horizons; "
                            "performance: realised per-model performance "
                            "for a project; decisions: unresolved decisions "
                            "and the resolution summary; effects: frozen "
                            "context scenarios and realised effect estimates; "
                            "effect_prior: resolve the governed evidence ladder."
                        )},
            "model": {"type": "string", "description": (
                "With section='performance': narrow to one model's "
                "realised runs."
            )},
            "event_type": {"type": "string", "description": (
                "With section='effects': restrict to one event type."
            )},
            "series": {"type": "string", "description": (
                "With section='effects': restrict to one series."
            )},
            "resolved_only": {"type": "boolean", "description": (
                "With section='effects': omit scenarios awaiting outcomes."
            )},
            "as_of": {"type": "string", "description": (
                "With section='effect_prior': timezone-aware knowledge cutoff."
            )},
            "target": {"type": "string"},
            "domain": {"type": "string"},
            "population": {"type": "string"},
            "unit": {"type": "string"},
            "external_priors": {"type": "array", "items": {"type": "object"},
                                "description": (
                                    "Versioned external effect priors; each needs known_at."
                                )},
            "human_assumption": {"type": "object", "description": (
                "Explicit sensitivity assumption with location and known_at; "
                "never treated as learned probabilistic evidence."
            )},
        }, "required": []},
        "runner": "_run_status",
    },
    {
        "name": "gnomon_resolve_outcome",
        "description": (
            "Resolve DecisionArtifacts produced by `gnomon_decide` with what "
            "actually happened: "
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
        "runner": "_run_resolve_outcome",
    },
    {
        "name": "gnomon_route",
        "description": (
            "Which method for this task on this data? A disclosed, advisory "
            "structural starting point with a verified capability filter. "
            "Mutable tracking scores are not historical routing evidence. "
            "The execution profile supports cutoff-bound ledger studies. Includes the series fingerprint "
            "and every exclusion reason in the output. Feed `candidates` (or "
            "`recommendation`) to `gnomon_forecast`'s `candidates` parameter to "
            "act on the answer. Evaluated runs still backtest whatever pool "
            "they are given against the mandatory baselines, so routing "
            "narrows the contest but never decides it."
        ),
        "inputSchema": {"type": "object", "properties": {
            "input": {"type": "string", "description": "Path to a CSV/Parquet file or store:<dataset>. Callers without a filesystem pass `observations` inline instead."},
            **_OBSERVATIONS_PROPERTY,
            "time_column": {"type": "string"},
            "target_column": {"type": "string"},
            "series_column": {"type": "string"},
            "frequency": {"type": "string"},
            "task": {"type": "string", "enum": ["forecast", "detect_anomalies"],
                     "description": "Task to route (default forecast)."},
            "horizon": {"type": "integer", "description": "Forecast horizon (default 1)."},
            "project": {"type": "string", "description": (
                "Tracking project: records the structural recommendation; "
                "mutable scores never select a model."
            )},
        }, "required": []},
        "runner": "_run_route",
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
        "runner": "_run_explain_run",
    },
    {
        "name": "gnomon_install_tsfm",
        "description": (
            "Install a time-series foundation model into its isolated "
            "sandbox venv, without blocking: the install runs as a "
            "detached process and each call reports the current state "
            "(absent / installing / ready / failed). Eligible names are "
            "in gnomon_capabilities under models.tsfm_available; when "
            "state is ready, pass the name in gnomon_forecast's "
            "`candidates`. Packages come from the pinned per-model spec "
            "via uv — expect minutes on first install. TSFMs remain "
            "candidates: they compete against the baselines on identical "
            "folds and never win by default."
        ),
        "inputSchema": {"type": "object", "properties": {
            "name": {"type": "string", "description": (
                "TSFM adapter name (e.g. chronos_bolt_mini); the "
                "installable set is in gnomon_capabilities under "
                "models.tsfm_available."
            )},
            "status_only": {"type": "boolean", "description": (
                "Report the sandbox state without starting an install."
            )},
        }, "required": ["name"]},
        "runner": "_run_install_tsfm",
    },
])
