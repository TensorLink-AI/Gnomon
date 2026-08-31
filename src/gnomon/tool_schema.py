"""Reusable JSON Schema fragments for Gnomon's canonical tool surface."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CONTEXT_EVENTS_PROPERTY",
    "COVARIATE_MAPPING_PROPERTY",
    "COVARIATES_PROPERTY",
    "DATA_REF_PROPERTY",
    "INPUT_PROPERTIES",
    "OBSERVATIONS_PROPERTY",
    "REPLAY_PROPERTIES",
    "TEMPORAL_QUESTIONS_PROPERTY",
]

#: Inline context shared by tools that accept dated claims.
CONTEXT_EVENTS_PROPERTY: dict[str, Any] = {
    "context_source_text": {
        "type": "string",
        "description": "Source containing each verbatim source_span.",
    },
    "context_events": {
        "type": "array",
        "description": "Dated quoted min/max/exact claims.",
        "items": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "claim_kind": {"type": "string",
                               "enum": ["min", "max", "exact"]},
                "entity_scope": {"type": "array", "items": {"type": "string"},
                                 "description": "Optional target names."},
                "effective_start": {"type": "string"},
                "effective_end": {"type": "string"},
                "known_at": {"type": "string"},
                "source_span": {"type": "string"},
                "source_reference": {"type": "string"},
            },
            "required": ["event_id", "effective_start", "effective_end", "known_at"],
        },
    },
    "context_ref": {
        "type": "string",
        "description": "Prior context receipt; timing is rechecked.",
    },
    "qualitative_context_events": {
        "type": "array",
        "description": "Quoted unknown-magnitude event; scenario-only.",
        "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "event_id": {"type": "string"},
                "effective_start": {"type": "string"},
                "effective_end": {"type": "string"},
                "known_at": {"type": "string"},
                "source_span": {"type": "string",
                                "description": "Quote the start date."},
                "direction": {"type": "string",
                              "enum": ["increase", "decrease", "unknown"]},
                "effect_family": {"type": "string", "enum": [
                    "level_shift", "temporary_pulse", "variance_change",
                    "seasonal_regime_change"]},
                "duration": {"type": "string",
                             "enum": ["temporary", "persistent", "unknown"]},
                "entity_scope": {"type": "array", "items": {"type": "string"}},
                "source_reference": {"type": "string"},
            },
            "required": ["event_id", "effective_start", "effective_end",
                         "known_at", "source_span", "direction",
                         "effect_family", "duration"],
        },
    },
    "context_rejections": {
        "type": "array", "items": {"type": "object"},
        "description": "Rejected context with id, reason_code, reason, quote.",
    },
}

#: Inline point-in-time covariates, mutually exclusive with a covariate file.
COVARIATES_PROPERTY: dict[str, Any] = {
    "covariates": {
        "type": "array",
        "description": (
            "Inline point-in-time rows: timestamp, known_at, and mapped "
            "covariates. Mutually exclusive with covariates_file."
        ),
        "items": {"type": "object"},
    },
}

COVARIATE_MAPPING_PROPERTY: dict[str, Any] = {
    "covariate_mapping": {
        "type": ["string", "array"],
        "items": {"type": ["string", "object"]},
        "description": (
            "Covariates as name:type:future_known strings or objects with "
            "name, type, and availability."
        ),
    },
}

#: Inline observations let MCP clients without a filesystem reach data tools.
OBSERVATIONS_PROPERTY: dict[str, Any] = {
    "observations": {
        "type": "array",
        "maxItems": 500,
        "description": (
            "Inline row objects; exclusive with input, maximum 500. Reuse "
            "the returned data_ref."
        ),
        "items": {"type": "object"},
    },
}

DATA_REF_PROPERTY: dict[str, Any] = {
    "data_ref": {
        "type": "string",
        "description": (
            "Prior call reference; replaces input/observations and binds "
            "schema and cutoff."
        ),
    },
}

INPUT_PROPERTIES: dict[str, Any] = {
    "input": {"type": "string", "description": "Local table, store:<dataset>, or allowlisted read-only prom:// range query. Otherwise pass observations."},
    **OBSERVATIONS_PROPERTY,
    **DATA_REF_PROPERTY,
    "time_column": {"type": "string", "description": (
        "Timestamp column. Omit for unambiguous inference; required for "
        "store:<dataset>."
    )},
    "target_column": {"type": "string", "description": (
        "Numeric column to operate on. Omit to infer when exactly one "
        "non-time column qualifies (disclosed as an assumption); "
        "ambiguity fails loudly. Required for store:<dataset> inputs."
    )},
    "series_column": {"type": "string", "description": "Optional column identifying independent series."},
    "frequency": {
        "type": "string",
        "pattern": "^([1-9][0-9]*)?(s|min|h)$|^(D|W|MS)$",
        "description": (
            "Grid: s|min|h|D|W|MS or <N>s|<N>min|<N>h. Omit to infer; "
            "ambiguity fails loudly."
        ),
    },
    "regrid": {
        "type": "string", "enum": ["business_daily", "month_start"],
        "description": (
            "Calendar: business_daily fills non-business days (D); "
            "month_start restamps months (MS). Changes are disclosed."
        ),
    },
}

#: Replay controls shared by every verb that reads bitemporal data.
REPLAY_PROPERTIES: dict[str, Any] = {
    "as_of": {
        "type": "string",
        "description": (
            "ISO-8601 replay instant; only earlier-known data is visible. "
            "Useful for store:<dataset>; files carry one vintage."
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

TEMPORAL_QUESTIONS_PROPERTY: dict[str, Any] = {
    # Kept intentionally tiny: detailed validation is returned by the
    # compiler, while every schema byte is repaid on every agent turn.
    "questions": {"type": "array", "items": {"type": "object"}},
}
