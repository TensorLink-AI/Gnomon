"""Operator-owned execution context shared by Python, CLI and MCP.

The session is dependency injection, not another model protocol. Tool requests
select registered names; only startup configuration can load Python plugins,
resolve service URLs, select authentication or open a writable ledger.
"""

from __future__ import annotations

from dataclasses import asdict
from collections import OrderedDict
from copy import deepcopy
import importlib
import json
import os
import sys
from pathlib import Path
import tomllib
from typing import Any
from functools import wraps

from .contracts import GnomonError
from .forecast_adapter import AdapterCapabilities, ForecastAdapterError, ForecastRequest, StatisticalAdapter
from .inference import InferenceEngine
from .ledger import TemporalLedger
from .decision_memory import (MEMORY_PARAMETERS, MEMORY_REQUIRED, MEMORY_WRITES,
                              MEMORY_PROPERTIES, MEMORY_DESCRIPTIONS)
from .ephemeris import EphemerisProvider
from .product_contract import __version__, product_claims
from .build_info import build_info
from .recovery import argument_recovery
from .repair import REPAIR_HELP

EXIT_SEMANTICS = {'0': 'Operation executed; inspect semantic_completion, routing_status, scoring_status and evidence completeness.',
                  '2': 'Rejected, invalid, unscored or failed execution.', '3': 'Partial evaluation.', '130': 'Interrupted.'}


def _measured(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        before = self._execution_counts()
        try:
            result = method(self, *args, **kwargs)
        except (GnomonError, ForecastAdapterError) as exc:
            exc.details['execution_diagnostics'] = self._execution_delta(before)
            raise
        return {**result, 'execution_diagnostics': self._execution_delta(before)}
    return invoke


def resolved_configuration(path=None, *, ledger_path=None):
    """Inspect operator settings without importing providers, opening databases or resolving secrets."""
    location = Path(path).expanduser().resolve() if path is not None else None
    config = {}
    if location:
        try:
            with location.open('rb') as handle:
                config = tomllib.load(handle)
        except tomllib.TOMLDecodeError:
            raise ForecastAdapterError('Provider configuration must be valid TOML.') from None
    _strict(config, configuration_schema()['properties'], label='operator TOML configuration')
    from .recovery import _matches
    if not _matches(config, configuration_schema()):
        _configuration_error(config)
    directory = location.parent if location else Path.cwd()
    configured = (directory / config['ledger_path']).resolve() if config.get('ledger_path') else None
    if ledger_path is not None:
        explicit = Path(ledger_path).expanduser().resolve()
        if configured is not None and configured != explicit:
            raise ForecastAdapterError('--ledger-path conflicts with ledger_path in provider configuration; select one ledger.')
        configured = explicit
    providers = {}
    for name, spec in config.get('providers', {}).items():
        providers[name] = {k: spec[k] for k in ('kind', 'entrypoint', 'revision', 'deterministic', 'lifecycle') if k in spec}
        providers[name]['entrypoint_imported'] = False
        providers[name]['remote_endpoint_configured'] = bool(spec.get('base_url') or spec.get('base_url_env'))
    return {'schema_version': '1', 'status': 'ok', 'configuration_file': str(location) if location else None,
        'configuration_file_exists': location.is_file() if location else None,
        'relative_path_base': str(directory), 'path_resolution': 'Relative TOML paths resolve against the configuration file directory; CLI paths resolve against cwd.',
        'ledger_path': str(configured) if configured else None, 'ledger_exists': configured.is_file() if configured else None,
        'ledger_parent_exists': configured.parent.is_dir() if configured else None,
        'cache_size': config.get('cache_size', 0), 'allow_outcome_writes': config.get('allow_outcome_writes', False),
        'enable_temporal': config.get('enable_temporal', False), 'providers': providers,
        'provider_files_checked': False, 'guidance': 'Entrypoints are shown without importing code. Existence and dependency checks happen when starting the configured session. Secrets and remote endpoints are omitted.'}


def _configuration_error(config):
    """Name invalid paths and public constraints, never echo operator values."""
    from .recovery import _matches
    failures = []
    def visit(value, schema, path):
        if _matches(value, schema):
            return
        if isinstance(value, dict) and schema.get('type') == 'object':
            props, extra = schema.get('properties', {}), schema.get('additionalProperties', True)
            for key in schema.get('required', []):
                if key not in value:
                    failures.append({'field': path + '.' + key if path else key, 'reason': 'required field missing'})
            for key, item in value.items():
                child = path + '.' + key if path else key
                spec = props.get(key, extra)
                if isinstance(spec, dict):
                    visit(item, spec, child)
                elif spec is False:
                    failures.append({'field': child, 'reason': 'unknown field'})
        else:
            failures.append({'field': path, 'reason': 'unsupported value or type',
                'expected': {k: schema[k] for k in ('type', 'enum', 'minimum', 'maximum', 'const') if k in schema}})
    visit(config, configuration_schema(), '')
    raise ForecastAdapterError('Invalid configuration fields: ' + '; '.join(f['field'] + ': ' + f['reason'] +
        ('; expected ' + json.dumps(f['expected']) if f.get('expected') else '') for f in failures) + '; use gnomon capabilities --config-schema.',
        details={'invalid_fields': failures, 'rejected_fields': [f['field'] for f in failures]})

_NUMBER_ARRAY = {"type": "array", "items": {"type": "number"}}
_STRING_ARRAY = {"type": "array", "items": {"type": "string"}}
REQUEST_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["history", "horizon"],
    "properties": {
        "history": {**_NUMBER_ARRAY, "minItems": 1},
        "horizon": {"type": "integer", "minimum": 1},
        "season": {"type": "integer", "minimum": 1, "default": 1,
                   "description": "Seasonal period in observations (CLI: --season), not season_length. "
                                  "seasonal_naive requires at least season history values; 1 repeats the last value. "
                                  "Non-seasonal providers may ignore this field."},
        "samples": {"type": "integer", "minimum": 0},
        "quantiles": _NUMBER_ARRAY,
        **{name: {"type": ["string", "null"]} for name in (
            "frequency", "cutoff", "known_time_cutoff", "recorded_time_cutoff",
            "series_id", "unit", "snapshot_id")},
        **{name: _STRING_ARRAY for name in (
            "timestamps", "future_timestamps", "past_covariate_names", "future_covariate_names")},
        **{name: {"type": "array", "items": _NUMBER_ARRAY} for name in (
            "past_covariates", "future_covariates", "related_series")},
    },
}

def provider_request_schema(capabilities: dict) -> dict:
    """Describe the common request fields with this provider's declared limits."""
    schema = deepcopy(REQUEST_SCHEMA)
    # Shared scalar-array templates must not couple quantile and covariate limits.
    properties = schema["properties"] = {name: deepcopy(value) for name, value in REQUEST_SCHEMA["properties"].items()}
    properties["history"]["minItems"] = capabilities["min_history"] or 1
    if capabilities["max_horizon"] is not None:
        properties["horizon"]["maximum"] = capabilities["max_horizon"]
    if capabilities["frequencies"]:
        properties["frequency"]["enum"] = [None, *capabilities["frequencies"]]
    for feature, field in (("quantiles", "quantiles"), ("past_covariates", "past_covariates"),
                           ("future_covariates", "future_covariates"), ("panel", "related_series")):
        if not capabilities[feature]:
            properties[field]["maxItems"] = 0
    if not capabilities["sample_paths"]:
        properties["samples"]["maximum"] = 0
    return schema


INSPECT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["input"],
                  "properties": {**{name: {"type": "string"} for name in (
                      "input", "time_column", "target_column", "series_column", "frequency", "as_of",
                      "recorded_as_of", "store_path", "unit", "regrid", "timezone")},
                      "purpose": {"enum": ["infer", "evaluate", "route"]},
                      "window": {"enum": ["latest_contiguous"]},
                      "repair": {"enum": ["off", "safe", "aggressive"], "default": "safe", "description": REPAIR_HELP}}}
_SERIES_SELECTOR = {"type": "string", "description":
    "Select an existing inspected series, not a new label. Unlabeled input uses __default__; "
    "read labels from a column using inspect series_column (CLI: --series-column)."}
DESCRIBE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["data_ref", "statistic"],
                   "properties": {**{name: {"type": "string"} for name in ("data_ref", "series_id", "start", "end")},
                       "statistic": {"enum": ["mean", "median", "latest", "minimum", "maximum", "sum"]}}}
_FORECAST_COMMON = {"provider": {"type": "string"}, "use_cache": {"type": "boolean"},
                    'verify': {'type': 'boolean', 'default': False, 'description': 'Include independent deterministic built-in arithmetic verification.'}}
FORECAST_SCHEMA = {"type": "object", "oneOf": [
    {"type": "object", "additionalProperties": False, "required": ["provider", "request"],
     "properties": {**_FORECAST_COMMON, "request": REQUEST_SCHEMA}},
    {"type": "object", "additionalProperties": False, "required": ["provider", "data_ref", "horizon"],
     "properties": {**_FORECAST_COMMON, "data_ref": {"type": "string"}, "series_id": {"type": "string"},
                    **{name: REQUEST_SCHEMA["properties"][name] for name in ("horizon", "season", "quantiles")}}},
]}
_BUDGET_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    **{name: {"type": "integer", "minimum": 0 if name == "max_calls" else 1}
       for name in ("max_providers", "max_folds", "max_calls")},
    "max_seconds": {"type": ["number", "null"], "exclusiveMinimum": 0}}}
EVALUATE_SCHEMA = {"type": "object", "oneOf": [
    {"type": "object", "additionalProperties": False, "required": ["data_ref", "candidates", "baseline", "horizon"],
     "properties": {"data_ref": {"type": "string"}, "series_id": {"type": "string"},
                    "candidates": {**_STRING_ARRAY, "minItems": 1, "uniqueItems": True}, "baseline": {"type": "string"},
                    **{name: {"type": "integer", "minimum": 1} for name in ("horizon", "folds", "min_history", "stride", "season")},
                    "budget": _BUDGET_SCHEMA, "verify": {'type': 'boolean', 'default': False},
                    "preflight": {'type': 'boolean', 'default': False, 'description': 'Plan folds and report visibility/capability causes without forecast calls or saving a study.'},
                    "replay": {"enum": ["recorded", "source_available"], 'description': 'Defaults to recorded for a recording-bounded snapshot, otherwise source_available. Explicit source replay changes the temporal question; it does not attest local recording at each origin.'}}},
    {"type": "object", "additionalProperties": False, "required": ["study_id"],
     "description": "Retrieve complete saved study evidence, including fold requests, predictions and actuals. Retrieval makes no provider calls and does not rescore revised observations.",
     "properties": {"study_id": {"type": "string"}}},
]}
EVALUATE_SCHEMA['oneOf'].extend([
    {'type': 'object', 'additionalProperties': False,
     'required': ['operation', 'study_id', 'data_ref', 'source_as_of', 'recorded_as_of'],
     'description': 'Rescore recorded predictions at later actual availability cutoffs without changing original origins or calling providers. Saves a new immutable study.',
     'properties': {'operation': {'const': 'rescore'}, **{k: {'type': 'string'} for k in ('study_id', 'data_ref', 'source_as_of', 'recorded_as_of')},
                    'allow_partial': {'type': 'boolean', 'default': True}}},
    {'type': 'object', 'additionalProperties': False,
     'required': ['operation', 'original_study_id', 'rescored_study_id'],
     'properties': {'operation': {'const': 'compare_studies'}, **{k: {'type': 'string'} for k in ('original_study_id', 'rescored_study_id')}}},
])
ROUTE_SCHEMA = {"type": "object", "additionalProperties": False,
               "required": ["data_ref", "study_id", "source_as_of", "recorded_as_of"],
               "description": "Omitted candidates, baseline, horizon, season and series_id are loaded from the study visible at recorded_as_of. Mismatched overrides or insufficient evidence return a disclosed baseline fallback by default (status ok). Set require_evidence=true to reject any fallback. Routing makes zero provider calls; successful matched routing saves a new immutable rescore and leaves the original study unchanged.",
               "properties": {**{key: {"type": "string"} for key in (
                   "data_ref", "study_id", "baseline", "source_as_of", "recorded_as_of", "series_id")},
                   "candidates": {**_STRING_ARRAY, "minItems": 1, "uniqueItems": True},
                   "horizon": {"type": "integer", "minimum": 1}, "season": {"type": "integer", "minimum": 1},
                   "min_folds": {"type": "integer", "minimum": 3},
                   "min_improvement": {"type": "number", "minimum": 0, "maximum": 1},
                   "require_evidence": {"type": "boolean", "default": False,
                       "description": "Reject baseline fallback with ROUTING_EVIDENCE_REQUIRED (CLI exit 2). An evidence-supported baseline, including an exact tie, remains a successful selection."}}}
for _series_schema in (DESCRIBE_SCHEMA, FORECAST_SCHEMA["oneOf"][1],
                       EVALUATE_SCHEMA["oneOf"][0], ROUTE_SCHEMA):
    _series_schema["properties"]["series_id"] = deepcopy(_SERIES_SELECTOR)
INSPECT_SCHEMA['properties']['diagnose'] = {'type': 'boolean', 'default': False,
    'description': 'Bounded local-file dry run of all three repair policies; no source changes, providers or reusable data reference. Do not combine with repair.'}
for _schema in (INSPECT_SCHEMA, ROUTE_SCHEMA, EVALUATE_SCHEMA['oneOf'][2]):
    for _key in ('as_of', 'source_as_of', 'recorded_as_of'):
        if _key in _schema['properties']:
            _schema['properties'][_key]['description'] = (
                'Inclusive local recording cutoff; distinct from valid time and source availability.' if _key == 'recorded_as_of' else
                'Inclusive source availability cutoff. For routing, the prospective forecast must also begin after this instant.' if _schema is ROUTE_SCHEMA else
                'Inclusive source availability cutoff (known_time); rescore keeps original forecast origins unchanged.' if _schema is EVALUATE_SCHEMA['oneOf'][2] else
                'Freeze observations visible by source availability (known_time), not local recording time. File known times are assumed from valid timestamps.')
for _key in ('cutoff', 'known_time_cutoff', 'recorded_time_cutoff'):
    REQUEST_SCHEMA['properties'][_key]['description'] = {
        'cutoff': 'Forecast origin boundary: history ends at or before it; future timestamps follow it.',
        'known_time_cutoff': 'Source availability boundary of the history used for this prediction.',
        'recorded_time_cutoff': 'Local recording visibility boundary of the history used for this prediction.'}[_key]
READ_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["result_ref"],
               "properties": {"result_ref": {"type": "string"}, "pointer": {"type": "string", "maxLength": 1024},
                              "offset": {"type": "integer", "minimum": 0},
                              "max_chars": {"type": "integer", "minimum": 1, "maximum": 4096}}}

_LEDGER_PARAMETERS = {
    "search": ("series_id", "horizon", "provider", "unit", "start", "end", "status", "source_as_of", "recorded_as_of", "limit", "cursor"),
    "compare_history": ("series_id", "horizon", "providers", "unit", "start", "end", "source_as_of", "recorded_as_of"),
    "study": ("study_id", "recorded_as_of"),
    "execution": ("execution_id",),
    "actuals_as_of": ("series_id", "source_as_of", "recorded_as_of", "unit"),
    "evaluations": ("execution_id", "recorded_as_of"),
    "pending": ("source_as_of", "recorded_as_of"),
    "compare": ("execution_ids", "source_as_of", "recorded_as_of"),
    "decision": ("decision_id", "source_as_of", "recorded_as_of"),
    "append_actual": ("series_id", "valid_time", "value", "source_available_at", "unit", "source_ref", "actuals"),
    "evaluate": ("execution_id", "execution_ids", "source_as_of", "recorded_as_of", "allow_partial"),
    "record_decision": ("execution_ids", "policy", "inputs", "action", "authorization_ref"),
    "append_decision_outcome": ("decision_id", "outcome", "source_available_at"),
}
_LEDGER_REQUIRED = {
    "search": (),
    "compare_history": ("series_id", "horizon", "providers", "start", "end", "source_as_of", "recorded_as_of"),
    "study": ("study_id",),
    "execution": ("execution_id",), "actuals_as_of": ("series_id",),
    "evaluations": ("execution_id",), "pending": (), "compare": ("execution_ids",),
    "decision": ("decision_id",),
    "append_actual": (),
    "evaluate": (),
    "record_decision": ("execution_ids", "policy", "inputs", "action"),
    "append_decision_outcome": ("decision_id", "outcome", "source_available_at"),
}
_LEDGER_PARAMETERS.update(MEMORY_PARAMETERS)
_LEDGER_REQUIRED.update(MEMORY_REQUIRED)
_OUTCOME_WRITES = {"append_actual", "record_decision", "append_decision_outcome"} | MEMORY_WRITES


def configuration_schema():
    """Discover operator TOML keys without opening a ledger or loading providers."""
    from .backtesting import EvaluationBudget
    from .result_refs import ResultLimits
    fields = {
        "schema_version": {"const": 1, "default": 1},
        "ledger_path": {"type": "string", "description": "SQLite path relative to this TOML file. Relative paths resolve relative to the configuration file, not the current working directory."},
        "cache_size": {"type": "integer", "minimum": 0, "default": 0,
                       "description": "Maximum cached results per persistent session; 0 disables. "
                                      "Set cache_size = 8 in TOML, then GnomonSession.from_config('providers.toml'). "
                                      "Repeat the same deterministic, versioned provider request in that session to get a hit."},
        "allow_outcome_writes": {"type": "boolean", "default": False},
        "enable_temporal": {"type": "boolean", "default": False},
        'compact_errors': {'type': 'boolean', 'default': True, 'description': 'Replace the legacy duplicate rejection payload with an /error reference. False restores expanded legacy errors.'},
        "max_data_refs": {"type": "integer", "minimum": 1, "default": 16},
        "max_data_rows": {"type": "integer", "minimum": 1, "default": 100000},
        "evaluation_limits": {**deepcopy(_BUDGET_SCHEMA), "default": asdict(EvaluationBudget(max_seconds=30))},
        "result_limits": {"type": "object", "additionalProperties": False, "properties": {
            k: {"type": "integer", "minimum": 2048 if k == "max_response_bytes" else 1, "default": v}
            for k, v in asdict(ResultLimits()).items()},
            "description": "Response <= individual result <= retained bytes."},
        "providers": {"type": "object", "additionalProperties": {"type": "object",
            "required": ["kind"], "properties": {"kind": {"enum": ["ephemeris", "callable", "factory"]}},
            "description": "ephemeris: exactly one base_url/base_url_env; optional token_env, model, mode, combine, timeout, discover. "
                           "callable/factory: entrypoint=module:attribute required; optional capabilities, revision, deterministic, lifecycle. "
                           "These operator fields load trusted Python or configure network providers."}},
    }
    return {"type": "object", "additionalProperties": False, "properties": fields,
            "description": "Operator configuration is TOML, not JSON; this schema describes the parsed keys. "
                           "Example TOML: schema_version = 1\\nledger_path = \"ledger.db\"\\nallow_outcome_writes = true"}


def _strict(arguments, allowed, required=(), *, label="arguments"):
    if not isinstance(arguments, dict):
        raise ForecastAdapterError(f"{label} must be an object")
    if set(arguments) - set(allowed):
        raise ForecastAdapterError("unknown arguments: " + ", ".join(sorted(set(arguments) - set(allowed)))
                                   + "; accepted fields: " + ", ".join(sorted(allowed)),
                                   details={'rejected_fields': [k[:128] for k in sorted(set(arguments) - set(allowed))],
                                            'rejected_field_names_truncated': any(len(k) > 128 for k in arguments), 'expected_fields': sorted(allowed)})
    missing = set(required) - set(arguments)
    if missing:
        raise ForecastAdapterError("missing arguments: " + ", ".join(sorted(missing)), details={'missing_fields': sorted(missing), 'expected_fields': sorted(allowed)})


class GnomonSession:
    """An explicit registry, optional ledger and small tool dispatch boundary.

    Start with GnomonSession.from_config() for built-in providers; no TOML file
    is required. Bare GnomonSession() starts empty for custom registries.
    """

    def __init__(self, engine: InferenceEngine | None = None, *, ledger: TemporalLedger | None = None,
                 allow_outcome_writes: bool = False, max_data_refs: int = 16, max_data_rows: int = 100_000,
                 evaluation_limits: dict | None = None, result_limits: dict | None = None,
                 enable_temporal: bool = False, compact_errors: bool = True):
        if type(compact_errors) is not bool:
            raise ForecastAdapterError('compact_errors must be a boolean')
        self.compact_errors = compact_errors
        if type(allow_outcome_writes) is not bool:
            raise ForecastAdapterError("allow_outcome_writes must be a boolean")
        if type(enable_temporal) is not bool:
            raise ForecastAdapterError("enable_temporal must be a boolean")
        self.enable_temporal = enable_temporal
        self.ledger = ledger if ledger is not None else (engine.ledger if engine is not None else None)
        self.engine = engine if engine is not None else InferenceEngine(ledger=ledger)
        if engine is not None and ledger is not None and engine.ledger is not ledger:
            raise ForecastAdapterError("session and engine must share the same ledger")
        self.allow_outcome_writes = allow_outcome_writes
        self._ephemeris_configs: list[dict] = []
        from .data_refs import DataReferences
        self.data = DataReferences(max_refs=max_data_refs, max_rows=max_data_rows)
        from .backtesting import EvaluationBudget
        self.evaluation_limits = EvaluationBudget.from_dict(
            {"max_seconds": 30, **(evaluation_limits if evaluation_limits is not None else {})})
        self._studies = OrderedDict()
        from .result_refs import ResultLimits, ResultReferences
        if result_limits is not None:
            _strict(result_limits, ResultLimits.__dataclass_fields__)
        self.results = ResultReferences(ResultLimits(**(result_limits or {})))

    @classmethod
    def from_config(cls, path: str | Path | None = None, *, ledger_path: str | Path | None = None,
                    create_ledger: bool = True, ledger: TemporalLedger | None = None,
                    discovery_only: bool = False) -> "GnomonSession":
        """Load built-ins and an explicitly supplied TOML path. No cwd config search.

        Relative paths resolve relative to the configuration file, not cwd.
        CLI ledger_path overrides resolve against cwd. Inspect settings without
        importing providers: gnomon capabilities --show-resolved-config
        --providers-config providers.toml.

        To enable built-in caching, put ``cache_size = 8`` in providers.toml::

            with GnomonSession.from_config("providers.toml") as session:
                request = {"history": [1, 2, 3], "horizon": 2, "series_id": "sales", "unit": "widgets"}
                assert not session.forecast("last_value", request)["cache_hit"]
                assert session.forecast("last_value", request)["cache_hit"]

        cache_size is a TOML key, not a from_config keyword. Discover all keys
        with ``gnomon capabilities --config-schema``. Separate CLI processes
        cannot share this cache. Custom registries use InferenceEngine(cache_size=8).
        To use built-ins with a controlled recording clock, pass
        ledger=TemporalLedger('evidence.db', clock=your_clock). Do not also
        configure a ledger path. discovery_only avoids opening the ledger;
        configured provider discovery may still initialize provider code.
        """
        config, directory = {}, Path.cwd()
        if path is not None:
            location = Path(path).expanduser().resolve()
            directory = location.parent
            with location.open("rb") as handle:
                try:
                    config = tomllib.load(handle)
                except tomllib.TOMLDecodeError:
                    raise ForecastAdapterError(
                        'Provider configuration must be valid TOML, not JSON. '
                        'Example: schema_version = 1\nledger_path = "ledger.db"'
                    ) from None
        _strict(config, configuration_schema()["properties"], label="operator TOML configuration")
        from .recovery import _matches
        if not _matches(config, configuration_schema()):
            _configuration_error(config)
        if config.get("schema_version", 1) != 1:
            raise ForecastAdapterError("unsupported provider configuration schema")
        configured_ledger = directory / config["ledger_path"] if config.get("ledger_path") else None
        if ledger_path is not None:
            explicit_ledger = Path(ledger_path).expanduser().resolve()
            if configured_ledger is not None and configured_ledger.resolve() != explicit_ledger:
                raise ForecastAdapterError("--ledger-path conflicts with ledger_path in provider configuration; select one ledger.")
            configured_ledger = explicit_ledger
        if ledger is not None and (configured_ledger is not None or discovery_only):
            raise ForecastAdapterError('An explicit ledger cannot be combined with configured ledger paths or discovery_only.')
        engine = InferenceEngine(ledger=ledger, cache_size=config.get("cache_size", 0))
        session = cls(engine, ledger=ledger, allow_outcome_writes=config.get("allow_outcome_writes", False),
                      max_data_refs=config.get("max_data_refs", 16), max_data_rows=config.get("max_data_rows", 100_000),
                      evaluation_limits=config.get("evaluation_limits"), result_limits=config.get("result_limits"),
                      enable_temporal=config.get("enable_temporal", False), compact_errors=config.get('compact_errors', True))
        from .models import BASELINES, predict
        for name in sorted(BASELINES):
            adapter = StatisticalAdapter(name, predict)
            engine.register(name, adapter, revision=f"gnomon/{build_info()['build_id']}/{name}", deterministic=True)
        try:
            for name, spec in config.get("providers", {}).items():
                try:
                    session._configure_provider(name, spec)
                except ForecastAdapterError as exc:
                    exc.details.update(provider=name)
                    exc.details.setdefault('rejected_fields', exc.details.get('unknown_fields', []))
                    raise
            if configured_ledger is not None and not discovery_only:
                session.ledger = engine._ledger = TemporalLedger(configured_ledger, create=create_ledger)
            session._configured_ledger_path = configured_ledger
            session._discovery_only = discovery_only
        except Exception:
            engine.close()
            raise
        return session

    def _configure_provider(self, name, spec):
        _strict(spec, {"kind", "base_url", "base_url_env", "token_env", "model", "mode", "combine", "timeout",
                       "discover", "entrypoint", "capabilities", "revision", "deterministic", "lifecycle"}, {"kind"})
        kind = spec["kind"]
        if kind == "ephemeris":
            _strict(spec, {"kind", "base_url", "base_url_env", "token_env", "model", "mode", "combine", "timeout", "discover"})
            if bool(spec.get("base_url")) == bool(spec.get("base_url_env")):
                raise ForecastAdapterError("configure exactly one base_url or base_url_env")
            url = spec.get("base_url") or os.environ.get(spec["base_url_env"])
            if not url:
                raise ForecastAdapterError("provider base URL environment variable is not set")
            provider = EphemerisProvider(url, **{key: spec[key] for key in (
                "model", "mode", "combine", "token_env", "timeout") if key in spec})
            self.engine.register(name, provider, lifecycle="pretrained")
            discovered = provider.register_models(self.engine, prefix=name + "/") if spec.get("discover", False) else []
            self._ephemeris_configs.append({"name": name, "base_url_env": spec.get("base_url_env"),
                                            "discovered_models": len(discovered)})
        elif kind in {"callable", "factory"}:
            _strict(spec, {"kind", "entrypoint", "capabilities", "revision", "deterministic", "lifecycle"}, {"entrypoint"})
            entrypoint = spec["entrypoint"]
            if not isinstance(entrypoint, str) or entrypoint.count(":") != 1:
                raise ForecastAdapterError("entrypoint must be module:attribute")
            module, attribute = entrypoint.split(":")
            if not module or not attribute:
                raise ForecastAdapterError("entrypoint must contain a nonempty module and attribute")
            try:
                imported = importlib.import_module(module)
            except ModuleNotFoundError as exc:
                missing = exc.name or module
                raise GnomonError(
                    "PROVIDER_LOAD_FAILED",
                    f"Provider {name!r} could not import its configured entrypoint.",
                    details={"provider": name, "entrypoint": entrypoint,
                             "stage": "import_module", "missing_module": missing},
                    repair_options=[{
                        "action": "install_provider_dependency",
                        "description": f"Install {missing!r} in the same Python environment as Gnomon, or correct the entrypoint.",
                    }],
                ) from None
            except ImportError:
                raise GnomonError(
                    "PROVIDER_LOAD_FAILED",
                    f"Provider {name!r} raised ImportError while loading its configured entrypoint.",
                    details={"provider": name, "entrypoint": entrypoint,
                             "stage": "import_module", "exception_type": "ImportError"},
                    repair_options=[{
                        "action": "check_provider_environment",
                        "description": "Install the provider and its dependencies in the same Python environment as Gnomon, then verify the entrypoint.",
                    }],
                ) from None
            try:
                target = getattr(imported, attribute)
            except AttributeError:
                raise GnomonError(
                    "PROVIDER_LOAD_FAILED",
                    f"Provider {name!r} could not resolve its configured entrypoint attribute.",
                    details={"provider": name, "entrypoint": entrypoint,
                             "stage": "resolve_attribute", "attribute": attribute},
                    repair_options=[{
                        "action": "correct_provider_entrypoint",
                        "description": "Set entrypoint to an importable module:attribute exposed by the provider package.",
                    }],
                ) from None
            caps = spec.get("capabilities")
            capabilities = AdapterCapabilities(**caps) if caps is not None else None
            kwargs = {"capabilities": capabilities, "revision": spec.get("revision"),
                      "deterministic": spec.get("deterministic", False)}
            if kind == "factory":
                if "lifecycle" in spec:
                    raise ForecastAdapterError("factory lifecycle is always fresh_per_request")
                self.engine.register_factory(name, target, **kwargs)
            else:
                self.engine.register(name, target, lifecycle=spec.get("lifecycle", "stateless"), **kwargs)
        else:
            raise ForecastAdapterError("provider kind must be ephemeris, callable or factory")

    def _ephemeris_status(self) -> dict:
        """Additive discovery field: is a hosted-model provider configured in this session?"""
        configs = list(getattr(self, "_ephemeris_configs", []))
        if not configs:
            with self.engine._lock:
                manual = [name for name, p in self.engine._providers.items() if isinstance(p.target, EphemerisProvider)]
            if manual:
                configs = [{"name": manual[0], "base_url_env": None,
                            "discovered_models": sum(1 for n in manual if "/" in n)}]
        if not configs:
            return {"configured": False,
                    "setup": "Add [providers.ephemeris] kind = \"ephemeris\" with base_url_env/token_env to operator TOML, "
                             "set those environment variables and pass --providers-config; see "
                             "https://github.com/TensorLink-AI/Gnomon/blob/main/docs/production/INFERENCE.md#ephemeris-hosted-models"}
        return {"configured": True, "base_url_env": configs[0]["base_url_env"],
                "discovered_models": sum(c["discovered_models"] for c in configs)}

    @_measured
    def capabilities(self, *, brief: bool = True) -> dict:
        if type(brief) is not bool:
            raise ForecastAdapterError('brief must be a boolean')
        if brief:
            from .diagnostics import shared_provider_schemas
            return shared_provider_schemas(self.capabilities(brief=False))
        from .diagnostics import CUTOFF_SEMANTICS
        providers = self.engine.capabilities()
        for provider in providers.values():
            provider["request_schema"] = provider_request_schema(provider["capabilities"])
        return {"schema_version": "1", "status": "ok", "runtime_version": __version__,
                "build": build_info(),
                "product_contract": product_claims(),
                "interfaces": {"python": True, "cli": True, "mcp": True},
                "operation_interfaces": {"routing": {'cli': True, 'python': True, 'mcp': self.ledger is not None},
                    "ledger": {'cli': True, 'python': True, 'mcp': self.ledger is not None},
                    "rescore": {'cli': True, 'python': True, 'mcp': True, 'requires_ledger': True}},
                "exit_semantics": EXIT_SEMANTICS,
                'cutoff_semantics': CUTOFF_SEMANTICS,
                "version_semantics": {'distribution_version': __version__, 'runtime_build_id': build_info()['build_id']},
                "mcp_protocol": {'negotiated_version': getattr(self, '_mcp_protocol_version', None),
                    'supported_versions': ['2025-06-18'], 'transport': 'stdio', 'general_client_compatibility': 'not_claimed'},
                "python_environment": {"executable": sys.executable, "distribution": "gnomon-forecast",
                                       "import_name": "gnomon", "command": "gnomon python",
                                       "details_command": "gnomon environment"},
                "tools": {"visible": [tool["name"] for tool in self.tools()]},
                "ephemeris": self._ephemeris_status(),
                "providers": providers,
                "cache": {**self.engine.cache_policy(),
                          "enable": "Set cache_size = 8 in TOML; use GnomonSession.from_config('providers.toml'). "
                                    "Repeat the same request in one session; see help(GnomonSession.from_config).",
                          "schema_command": "gnomon capabilities --config-schema"},
                "ledger": {"enabled": self.ledger is not None, "outcome_writes": self.allow_outcome_writes,
                           "decision_memory": {"operations": list(MEMORY_PARAMETERS),
                               "discovery": "gnomon ledger --schema", "example": "python -m gnomon.examples.decision_memory",
                               "background_review": False, "external_memory_writes": "explicit caller-owned adapter only"},
                    'configured': self.ledger is not None or getattr(self, '_configured_ledger_path', None) is not None,
                    'exists': self.ledger.path.is_file() if self.ledger else Path(self._configured_ledger_path).is_file() if getattr(self, '_configured_ledger_path', None) else False,
                    'opened': self.ledger is not None},
                "temporal": {"enabled": self.enable_temporal, "semantics": "explicit_facts_not_natural_language",
                             "scope": "this_session_and_its_MCP_tools", "standalone_cli": "gnomon temporal is always available"},
                "data": {"reference_scope": "session", "max_refs": self.data.max_refs, "max_retained_rows": self.data.max_rows},
                "evaluation_limits": asdict(self.evaluation_limits),
                "result_limits": {**asdict(self.results.limits), "scope": "session", "offset_unit": "unicode_codepoints",
                                  "bound": "compact_structured_payload_utf8_not_provider_memory"},
                "semantics": {"forecast": "inference_only", "calibration": "not_implied", "action_authorized": False}}

    @_measured
    def forecast(self, provider: str, request: ForecastRequest | dict[str, Any] | None = None, *, use_cache: bool = True, verify: bool = False) -> dict:
        """Forecast from a typed request or the same request dict accepted by CLI/MCP."""
        if request is None:
            raise ForecastAdapterError('forecast request must be a ForecastRequest or dict. Use session.forecast(provider, request), e.g. '
                                       'session.forecast("last_value", {"history":[1,2,3],"horizon":2}).')
        if type(verify) is not bool:
            raise ForecastAdapterError('verify must be a boolean')
        if verify and not isinstance(getattr(self.engine._providers.get(provider), 'target', None), StatisticalAdapter):
            raise ForecastAdapterError('verify requires a registered statistical built-in provider')
        run = self.engine.forecast(provider, request, use_cache=use_cache)
        from .final_selection import FINAL_SELECTION_GUIDANCE
        canonical = run.completion()
        from .diagnostics import verify_builtin
        cache = self.engine.cache_policy(provider)
        cache.update(lookup_requested=use_cache, status=(
            "bypassed" if not use_cache else "disabled" if not cache["enabled"] else
            "ineligible" if not cache["provider_eligible"] else "hit" if run.cache_hit else "miss"))
        return {"schema_version": "1", "status": "ok", "execution_id": run.execution_id,
                "fingerprint": run.fingerprint, "provider": run.provider, "revision": run.revision,
                'request_fingerprint': canonical['request_fingerprint'], 'completion': canonical,
                'final_selection': {**deepcopy(FINAL_SELECTION_GUIDANCE),
                    'example': {'provider': run.provider, 'execution_id': run.execution_id}},
                "cache_hit": run.cache_hit, "result": asdict(run.result),
                "effective_season": run.request.season, "result_contract_validated": True,
                **({'verification': verify_builtin(provider, run.request, run.result)} if verify else {}),
                **({'season_guidance': 'Season is an observation count; seasonal_naive with season=1 repeats the last value. Specify the intended period explicitly.'} if provider == 'seasonal_naive' else {}),
                "cache": cache,
                "request_provenance": {
                    "source": "caller_supplied_request",
                    **{key: getattr(run.request, key) for key in (
                        "cutoff", "known_time_cutoff", "recorded_time_cutoff", "snapshot_id", "series_id")},
                    "history_start": run.request.timestamps[0] if run.request.timestamps else None,
                    "history_end": run.request.timestamps[-1] if run.request.timestamps else None,
                },
                "evidence": run.evidence, "action_authorized": run.action_authorized,
                "recorded": self.ledger is not None}

    @_measured
    def evaluate(self, data_ref: str, *, budget: dict | None = None, **kwargs) -> dict:
        from .backtesting import EvaluationBudget, evaluate_reference
        limits = asdict(self.evaluation_limits)
        if budget is not None:
            _strict(budget, limits, label="budget")
        requested = EvaluationBudget.from_dict({**limits, **(budget or {})})
        for key, limit in limits.items():
            value = getattr(requested, key)
            if limit is not None and (value is None or value > limit):
                raise ForecastAdapterError("evaluation budget cannot exceed operator startup limits")
        report = evaluate_reference(self.engine, self.data, data_ref, budget=requested, **kwargs)
        if self.ledger is None and 'study_id' in report:
            self._studies[report["study_id"]] = self.results.put(report)
            while len(self._studies) > 3:
                self._studies.popitem(last=False)
        return report

    @_measured
    def rescore(self, data_ref: str, *, study_id: str, source_as_of: str, recorded_as_of: str, allow_partial=True):
        """Reuse original predictions; evidence cutoffs govern actuals, never forecast origins."""
        from .rescoring import rescore_study
        return rescore_study(self.ledger, self.data, data_ref, study_id=study_id, source_as_of=source_as_of,
                             recorded_as_of=recorded_as_of, allow_partial=allow_partial,
                             max_folds=self.evaluation_limits.max_folds)

    @_measured
    def compare_studies(self, *, original_study_id, rescored_study_id):
        """Read immutable original/rescore differences without model calls."""
        from .rescoring import compare_studies
        return compare_studies(self.ledger, original_study_id=original_study_id, rescored_study_id=rescored_study_id)

    @_measured
    def route(self, data_ref: str, **kwargs) -> dict:
        from .study_routing import route_study
        fields = ("candidates", "baseline", "horizon", "season", "series_id")
        if any(key not in kwargs for key in fields):
            if self.ledger is None:
                raise ForecastAdapterError("study routing requires an explicit ledger")
            if "study_id" not in kwargs or "recorded_as_of" not in kwargs:
                raise ForecastAdapterError("study_id and recorded_as_of are required to load study parameters")
            if not isinstance(kwargs["recorded_as_of"], str) or not kwargs["recorded_as_of"]:
                raise ForecastAdapterError("recorded_as_of must be an explicit timestamp to load study parameters")
            # Never derive a past recommendation's task from future evidence.
            try:
                report = self.ledger.study(kwargs["study_id"], recorded_as_of=kwargs["recorded_as_of"])
            except ForecastAdapterError:
                if any(key not in kwargs for key in ("candidates", "baseline", "horizon")):
                    raise
                # An explicitly supplied task can still return its usual safe
                # fallback when the study cannot be read at this cutoff.
            else:
                defaults = {key: report[key] for key in fields if key != "candidates" and key in report}
                defaults["candidates"] = [key for key in report.get('provider_order', report["providers"]) if key != report["baseline"]]
                kwargs = {**defaults, **kwargs}
        return route_study(self.engine, self.data, data_ref, max_folds=self.evaluation_limits.max_folds, **kwargs)

    def call(self, name: str, arguments: dict[str, Any], *, compact: bool = True) -> dict:
        """Shared tool dispatch. Default compact=True bounds responses and may retain
        session-local result references. Pass compact=False for full Python results.
        """
        if type(compact) is not bool:
            raise GnomonError("INVALID_ARGUMENTS", "compact must be a boolean")
        before = self._execution_counts()
        try:
            result = self._call(name, arguments)
        except GnomonError as exc:
            exc.details['execution_diagnostics'] = self._execution_delta(before)
            raise
        except Exception as exc:
            # Exception text and arbitrary class names may contain credentials.
            from .contracts import execution_failure
            error = execution_failure(exc, provider=arguments.get('provider') if isinstance(arguments, dict) else None)
            error.details['execution_diagnostics'] = self._execution_delta(before)
            raise error from None
        result['execution_diagnostics'] = self._execution_delta(before)
        if compact and name == 'gnomon_capabilities':
            result.pop('cutoff_semantics', None)
            result['cutoff_semantics_command'] = 'gnomon capabilities'
            result['cache']['enable'] = 'Set cache_size in TOML; repeat within one session.'
            result['cache'].pop('statistics_scope', None)
            for provider in result['providers'].values():
                for field, spec in provider.get('request_schema', {}).get('properties', {}).items():
                    if field != 'season':
                        spec.pop('description', None)
                    else:
                        spec['description'] = 'Seasonal observations (--season); requires season history values. Default 1 repeats last value.'
        if compact and name == "gnomon_evaluate" and "study_id" not in arguments and 'folds' in result:
            from .backtesting import compact_study
            result = compact_study(result)
        if name != 'gnomon_read':
            from .diagnostics import completion
            result = completion(result)
        return self.results.project(result) if compact else result

    def _execution_counts(self):
        return {**self.engine._execution_stats,
                'ledger_writes': getattr(self.ledger, '_committed_row_writes', 0)}

    def _execution_delta(self, before):
        from .diagnostics import EXECUTION_SCOPE
        after = self._execution_counts()
        return {**{k: after[k] - before[k] for k in before}, 'source_mutations': 0,
                'scope': EXECUTION_SCOPE}

    def _call(self, name: str, arguments: dict[str, Any]) -> dict:
        try:
            if not isinstance(arguments, dict):
                raise ForecastAdapterError("arguments must be an object")
            if name == "gnomon_read":
                _strict(arguments, READ_SCHEMA["properties"], READ_SCHEMA["required"])
                return self.results.read(**arguments, _execution_diagnostics=self._execution_delta(self._execution_counts()))
            if name == "gnomon_temporal":
                if not self.enable_temporal:
                    raise GnomonError("UNKNOWN_TOOL", "Temporal operations require enable_temporal=true at session startup.")
                from .temporal_ops import temporal_operation
                from .recovery import temporal_recovery
                try:
                    return temporal_operation(**arguments)
                except (ValueError, TypeError) as exc:
                    raise GnomonError("INVALID_ARGUMENTS", str(exc), details=temporal_recovery(arguments)) from None
            if name == "gnomon_capabilities":
                _strict(arguments, ('brief',))
                return self.capabilities(**arguments)
            if name == "gnomon_forecast":
                if 'selected_provider' in arguments and 'provider' not in arguments and isinstance(arguments['selected_provider'], str):
                    corrected = {k: v for k, v in arguments.items() if k != 'selected_provider'}
                    corrected['provider'] = arguments['selected_provider']
                    from .recovery import _matches
                    runnable = any(set(v['required']) <= corrected.keys() <= v['properties'].keys() and
                        all(_matches(value, v['properties'][key]) for key, value in corrected.items()) for v in FORECAST_SCHEMA['oneOf'])
                    if runnable and 'request' in corrected:
                        try:
                            ForecastRequest.from_dict(corrected['request'])
                        except (ForecastAdapterError, ValueError, TypeError):
                            runnable = False
                    raise ForecastAdapterError("Missing required field 'provider'; 'selected_provider' is not a tool argument. Retry gnomon_forecast with provider using the same value.",
                        details={'cause_code': 'INVALID_TOOL_ARGUMENTS', 'received_fields': sorted(arguments), 'missing_fields': ['provider'],
                            'rejected_fields': ['selected_provider'], 'supplied_arguments': deepcopy(arguments),
                            'example_arguments': corrected, 'example_kind': 'task_correction' if runnable else 'task_template', 'example_runnable': runnable,
                            'changed_fields': ['selected_provider', 'provider'], 'preserved_fields': sorted(set(arguments) - {'selected_provider'}),
                            'next_call': {'tool': 'gnomon_forecast', 'arguments': corrected, 'admissible': None},
                            'provider_calls': 0, 'guidance': 'Only the argument name changed; no provider was executed. Request shape, provider availability and capabilities are checked on retry.'})
                variants = FORECAST_SCHEMA["oneOf"]
                forms = [{"required": v["required"], "accepted_fields": sorted(v["properties"])} for v in variants]
                if ("request" in arguments and "data_ref" in arguments) or not any(
                        set(v["required"]) <= arguments.keys() and arguments.keys() <= v["properties"].keys()
                        for v in variants):
                    selected = variants[1] if "data_ref" in arguments or ("horizon" in arguments and "request" not in arguments) else variants[0]
                    missing = sorted(set(selected["required"]) - arguments.keys())
                    unknown = sorted(arguments.keys() - selected["properties"].keys())
                    problems = (["unknown arguments: " + ", ".join(unknown)] if unknown else [])
                    problems += ["missing arguments: " + ", ".join(missing)] if missing else []
                    raise ForecastAdapterError(
                        "; ".join(problems) + ". Forecast accepts exactly one form: provider, request[, use_cache]; or provider, data_ref, horizon "
                        "with optional series_id, season, quantiles, use_cache. See argument_forms for both contracts.",
                        details={"argument_forms": forms, "missing_fields": missing, "unknown_fields": unknown})
                if type(arguments.get("use_cache", True)) is not bool:
                    raise ForecastAdapterError("use_cache must be a boolean")
                if "data_ref" in arguments:
                    _strict(arguments, {"provider", "data_ref", "horizon", "series_id", "season", "quantiles", "use_cache", "verify"},
                            {"provider", "data_ref", "horizon"})
                    request = self.data.request(**{k: v for k, v in arguments.items() if k not in {"provider", "use_cache", "verify"}})
                    snapshot = self.data.snapshot_summary(arguments["data_ref"])
                else:
                    _strict(arguments, {"provider", "request", "use_cache", "verify"}, {"provider", "request"})
                    request = ForecastRequest.from_dict(arguments["request"])
                result = self.forecast(arguments["provider"], request, use_cache=arguments.get("use_cache", True), verify=arguments.get('verify', False))
                result['season_defaulted'] = 'season' not in (arguments['request'] if 'request' in arguments else arguments)
                if "data_ref" in arguments:
                    result["data_ref"] = arguments["data_ref"]
                    result["snapshot"] = snapshot
                    result["request_provenance"]["source"] = "frozen_snapshot"
                return result
            if name in {"gnomon_inspect", "gnomon_describe"}:
                schema = INSPECT_SCHEMA if name == "gnomon_inspect" else DESCRIBE_SCHEMA
                _strict(arguments, schema["properties"], schema["required"])
                if name == 'gnomon_inspect' and 'diagnose' in arguments:
                    if type(arguments['diagnose']) is not bool:
                        raise ForecastAdapterError('diagnose must be a boolean')
                    clean = {k: v for k, v in arguments.items() if k != 'diagnose'}
                    if arguments['diagnose']:
                        return self.data.diagnose(**clean)
                    arguments = clean
                if name == 'gnomon_inspect' and str(arguments.get('input', '')).endswith('.gnomon'):
                    rejected = set(arguments) - {'input', 'purpose'}
                    if rejected:
                        from .recovery import frozen_recovery
                        raise ForecastAdapterError('No preparation options are accepted with .gnomon input, even identical values. Inspect the original source to prepare a different snapshot.',
                            details={**frozen_recovery(arguments, rejected), 'rejected_arguments': {k: arguments[k] for k in sorted(rejected)}})
                return getattr(self.data, "inspect" if name == "gnomon_inspect" else "describe")(**arguments)
            if name == "gnomon_ledger":
                return self._ledger_call(arguments)
            if name == "gnomon_route":
                _strict(arguments, ROUTE_SCHEMA["properties"], ROUTE_SCHEMA["required"])
                return self.route(**arguments)
            if name == "gnomon_evaluate":
                if 'operation' in arguments:
                    operation = arguments['operation']
                    if operation not in ('rescore', 'compare_studies'):
                        raise ForecastAdapterError('evaluate operation must be rescore or compare_studies; omit it for a new evaluation or retrieval')
                    schema = next(v for v in EVALUATE_SCHEMA['oneOf'] if v['properties'].get('operation', {}).get('const') == operation)
                    _strict(arguments, schema['properties'], schema['required'])
                    return (self.rescore if operation == 'rescore' else self.compare_studies)(**{k: v for k, v in arguments.items() if k != 'operation'})
                if "study_id" in arguments:
                    _strict(arguments, {"study_id"}, {"study_id"})
                    study_id = arguments["study_id"]
                    if study_id in self._studies:
                        return {**self.results.value(self._studies[study_id]), "study_evidence_scope": "full_folds"}
                    if self.ledger is not None:
                        return {**self.ledger.study(study_id), "study_evidence_scope": "full_folds"}
                    raise ForecastAdapterError("study is no longer retained; configure a ledger for durable retrieval")
                schema = EVALUATE_SCHEMA["oneOf"][0]
                _strict(arguments, schema["properties"], schema["required"])
                return self.evaluate(**arguments)
            raise GnomonError("UNKNOWN_TOOL", "This execution session does not expose that tool.")
        except GnomonError as exc:
            from .recovery import _example_copy
            exc.details.setdefault('supplied_arguments', _example_copy(arguments))
            if exc.code == 'MISSING_COLUMNS':
                from .recovery import column_recovery
                exc.details.update(column_recovery(exc.details, arguments))
                exc.details['argument_basis'] = 'explicit_tool_arguments'
            if name == "gnomon_inspect":
                from .recovery import _example_copy
                exc.details.setdefault("input_options", _example_copy(arguments))
            if exc.code == "INVALID_ARGUMENTS":
                for key, value in argument_recovery(name, arguments).items():
                    exc.details.setdefault(key, value)
            raise
        except (ForecastAdapterError, TypeError, KeyError) as exc:
            details = {**argument_recovery(name, arguments), **getattr(exc, "details", {})}
            from .recovery import _example_copy
            if name != 'gnomon_capabilities':
                details.setdefault('supplied_arguments', _example_copy(arguments))
            if name in {"gnomon_inspect", "gnomon_describe", "gnomon_evaluate", "gnomon_route"}:
                from .recovery import _example_copy
                details.setdefault("supplied_arguments", _example_copy(arguments))
            if "execution_recorded_at" in details and "example_arguments" in details:
                details["example_arguments"]["recorded_as_of"] = details["execution_recorded_at"]
                from .recovery import example_changes
                details["changed_fields"] = example_changes(arguments, details["example_arguments"])
            if "required_history" in details:
                details.pop("example_arguments", None)
                details.pop("example_kind", None)
                details.pop("changed_fields", None)
                details["request_parameters"] = {"season": details["season"], "horizon": details["horizon"]}
            repairs = getattr(exc, "repair_options", None)
            if repairs is None and "guidance" in details:
                repairs = [{"action": "repair_operation", "description": details["guidance"]}]
            raise GnomonError("INVALID_ARGUMENTS", str(exc), details=details, repair_options=repairs) from None

    def _ledger_call(self, arguments):
        if self.ledger is None:
            raise GnomonError("LEDGER_NOT_CONFIGURED", "Configure the ledger at session startup.")
        operation = arguments.get("operation")
        if not isinstance(operation, str) or operation not in _LEDGER_PARAMETERS:
            raise ForecastAdapterError("ledger operation must be one of: " + ", ".join(
                op for op in _LEDGER_PARAMETERS if self.allow_outcome_writes or op not in _OUTCOME_WRITES))
        if operation in _OUTCOME_WRITES and not self.allow_outcome_writes:
            raise GnomonError("OUTCOME_WRITES_DISABLED", "Outcome writes require operator startup authorization.",
                             details={"config_setting": "allow_outcome_writes = true", "cli_option": "--providers-config"},
                             repair_options=[{"action": "configure_outcome_writes", "description":
                                 "An authorized operator can set allow_outcome_writes = true in TOML and pass "
                                 "--providers-config providers.toml, or set it when constructing GnomonSession."}])
        _strict(arguments, {"operation", *_LEDGER_PARAMETERS[operation]}, {"operation", *_LEDGER_REQUIRED[operation]})
        parameters = {k: v for k, v in arguments.items() if k != "operation"}
        if operation == "pending":
            now = self.ledger._now()
            for field in ("source_as_of", "recorded_as_of"):
                if parameters.get(field) is None:
                    parameters[field] = now
        result = getattr(self.ledger, operation)(**parameters, **(
            {"include_current_coverage": True} if operation == "evaluate" else {}))
        query = {"source_as_of": parameters.get("source_as_of"), "recorded_as_of": parameters.get("recorded_as_of"),
                 "cutoff_default": "current_clock" if operation in {"search", "pending"} else
                                    "required" if operation in {"compare_history", *MEMORY_PARAMETERS} else "unbounded",
                 "unit": arguments.get("unit"), "unit_default": "all_units" if operation == "search" else
                         "execution_unit" if operation in {"evaluate", "compare"} else "unitless"}
        for field in ("source_as_of", "recorded_as_of"):
            if field not in _LEDGER_PARAMETERS[operation]:
                query.pop(field)
            else:
                query[field + "_defaulted"] = arguments.get(field) is None
        if not any(field in query for field in ("source_as_of", "recorded_as_of")):
            query.pop("cutoff_default")
        else:
            query["omitted_cutoffs"] = query["cutoff_default"] if any(
                query.get(field + "_defaulted", False) for field in ("source_as_of", "recorded_as_of")) else None
        if operation in {"evaluate", "compare"}:
            query.pop("unit")
        query.update({k: parameters[k] for k in ("series_id", "horizon", "provider", "start", "end", "status", "execution_id", "execution_ids", "study_id", "decision_id") if k in parameters})
        if "unit" not in _LEDGER_PARAMETERS[operation] and operation not in {"evaluate", "compare"}:
            query.pop("unit")
            query.pop("unit_default")
        else:
            query["unit_defaulted"] = arguments.get("unit") is None
            query["omitted_unit"] = query["unit_default"] if query["unit_defaulted"] else None
        if operation == "append_actual" and "actuals" in arguments:
            for key in ("unit", "unit_default", "unit_defaulted", "omitted_unit"):
                query.pop(key, None)
            query["unit_scope"] = "per_actual"
            query["actual_units"] = [{"index": i, "unit": item.get("unit"),
                "unit_defaulted": item.get("unit") is None,
                "omitted_unit": "unitless" if item.get("unit") is None else None}
                for i, item in enumerate(arguments["actuals"])]
        # Search already reports its exact effective clock, including cursor reuse.
        if operation == "search":
            query.update({k: result[k] for k in ("source_as_of", "recorded_as_of") if k in result})
        answer = {"schema_version": "1", "status": "ok", "operation": operation, "result": result,
                  "query": query}
        if operation == "evaluate":
            scores = result if isinstance(result, list) else [result]
            answer.update(scoring_status=("complete" if all(s["complete"] for s in scores) else
                                          "partial" if any(s["n"] for s in scores) else "pending"),
                          complete=all(s["complete"] for s in scores), allow_partial=arguments.get("allow_partial", True))
        if operation == 'review_decision':
            answer.update(scoring_status=result['scoring_status'], complete=result['review_ready'],
                          review_ready=result['review_ready'])
        return answer

    def tools(self) -> list[dict]:
        tools = [
            {"name": "gnomon_read", "description": "Read exact retained result JSON text pages, optionally at a JSON pointer. Concatenate pages at next_offset; no provider calls. References expire with the session or LRU eviction.",
             "inputSchema": READ_SCHEMA},
            {"name": "gnomon_capabilities", "description": "List this session's registered providers and storage capabilities.",
             "inputSchema": {"type": "object", "properties": {'brief': {'type': 'boolean', 'default': True,
                'description': 'Deduplicate provider request schemas using shared JSON references.'}}, "additionalProperties": False}},
            {"name": "gnomon_forecast", "description": "Execute a registered provider. Preserve completion as typed evidence; final_selection shows the provider/execution_id selection to return. No implicit backtest, calibration claim or action permission.",
             "inputSchema": FORECAST_SCHEMA},
            {"name": "gnomon_inspect", "description": "Validate file/store data and freeze a session-local data_ref. Discloses cutoffs and repairs.",
             "inputSchema": INSPECT_SCHEMA},
            {"name": "gnomon_describe", "description": "Compute an exact observed statistic over one frozen series and optional inclusive timestamp window.",
             "inputSchema": DESCRIBE_SCHEMA},
            {"name": "gnomon_evaluate", "description": "Create a matched backtest; retrieve full saved evidence by study_id; operation=rescore reuses original executions with revised actuals and explicit cutoffs; operation=compare_studies compares immutable evidence. Rescore/comparison require a ledger and make zero provider calls.",
             "inputSchema": EVALUATE_SCHEMA},
        ]
        if self.enable_temporal:
            from .temporal_ops import TEMPORAL_SCHEMA
            tools.append({"name": "gnomon_temporal", "description": "Calculate explicit dates/instants, half-open interval relations and stable event order. Local times require zones and ambiguous folds; no implicit now or causal inference.",
                          "inputSchema": TEMPORAL_SCHEMA})
        if self.ledger is not None:
            tools.append({"name": "gnomon_route", "description": "Recommend from one immutable matched study with explicit source/recorded cutoffs; rescore without model calls or action permission.",
                          "inputSchema": ROUTE_SCHEMA})
            tools.append({"name": "gnomon_ledger", "description": "Find forecasts and feedback status, compare matched production history, score named runs or record authorized actuals. Reads never run models; exact scoring retries reuse evidence.",
                          "inputSchema": ledger_schema(allow_outcome_writes=self.allow_outcome_writes)})
        return tools

    def close(self):
        self._studies.clear()
        self.data.clear()
        self.results.close()
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_json_argument(value: str) -> dict:
    """CLI-only @file materialization. The tool boundary never imports code."""
    text = Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ForecastAdapterError("JSON argument must be an object")
    return parsed


def ledger_schema(*, allow_outcome_writes=False):
    variants = []
    for operation, parameters in _LEDGER_PARAMETERS.items():
        if operation in _OUTCOME_WRITES and not allow_outcome_writes:
            continue
        properties = {"operation": {"const": operation}}
        for p in parameters:
            properties[p] = ({"type": "number"} if p == "value" else
                             {"type": ["string", "null"], "minLength": 1} if p == "unit" else
                             {"type": "boolean", "default": True, "description":
                              "True scores available matching-unit actuals, including partial/pending horizons. "
                              "False rejects incomplete horizons. Top-level status ok and CLI exit 0 mean the operation succeeded; "
                              "check scoring_status/complete and result.status/result.coverage (each result for batches). "
                              "Check result.coverage_basis for saved versus reconstructed legacy coverage; "
                              "result.current_coverage refreshes diagnostics at the query cutoffs. "
                              "Scoring persists evidence without actual-write opt-in; exact retries reuse scores."} if p == "allow_partial" else
                             {"type": "integer", "minimum": 1, "maximum": 100 if p == "limit" else 1_000_000} if p in {"limit", "horizon"} else
                             {"enum": ["waiting", "ready", "scored", "stale", "unscorable", "scored_in_study"]} if p == "status" else
                             {"type": "object", "minProperties": 2, "maxProperties": 8,
                              "additionalProperties": {"type": "string", "minLength": 1}} if p == "providers" else
                             {"type": "object"} if p in {"policy", "inputs", "action", "outcome"} else
                             {**_STRING_ARRAY, "minItems": 2 if operation == "compare" else 1,
                              "maxItems": 100, "uniqueItems": True} if p == "execution_ids" else
                             {"type": "string"})
            if operation in MEMORY_PARAMETERS and p in MEMORY_PROPERTIES:
                properties[p] = deepcopy(MEMORY_PROPERTIES[p])
            if p == "unit":
                properties[p]["description"] = (
                    "Exact unit label; omitted/null searches all units." if operation == "search" else
                    "Unit label to record; omitted/null records unitless data." if operation == "append_actual" else
                    "Exact unit label; omitted/null selects only unitless data. Units are never converted.")
            if p in {"source_as_of", "recorded_as_of"}:
                clock = "source availability (source_available_at), not valid_time" if p == "source_as_of" else "local recording time"
                default = ("Required for this operation." if operation in {"compare_history", *MEMORY_PARAMETERS} else
                           "Omitted means the current clock." if operation in {"search", "pending"} else
                           "Omitted means unbounded.")
                properties[p]["description"] = f"Inclusive {clock} cutoff. {default} Supply explicit cutoffs for reproducible queries."
        if operation == "append_actual":
            scalar = {k: v for k, v in properties.items() if k != "actuals"}
            required = ["series_id", "valid_time", "value", "source_available_at"]
            variants.append({"type": "object", "properties": scalar, "additionalProperties": False,
                             "required": ["operation", *required]})
            variants.append({"type": "object", "additionalProperties": False, "required": ["operation", "actuals"],
                             "properties": {"operation": properties["operation"], "actuals": {
                                 "type": "array", "minItems": 1, "maxItems": 1000, "items": {
                                     "type": "object", "additionalProperties": False, "required": required,
                                     "properties": {k: v for k, v in scalar.items() if k != "operation"}}}}})
            continue
        if operation == "evaluate":
            for field, other in (("execution_id", "execution_ids"), ("execution_ids", "execution_id")):
                variants.append({"type": "object", "additionalProperties": False, "required": ["operation", field],
                                 "properties": {k: v for k, v in properties.items() if k != other}})
            continue
        variants.append({"type": "object", "properties": properties, "additionalProperties": False,
                         "required": ["operation", *_LEDGER_REQUIRED[operation]],
                         **({"description": MEMORY_DESCRIPTIONS[operation]} if operation in MEMORY_DESCRIPTIONS else {})})
    return {"type": "object", "oneOf": variants,
            'defaults_matrix': {op: {'source_as_of': 'not_applicable' if 'source_as_of' not in params else 'required' if op in {'compare_history', *MEMORY_PARAMETERS} else 'current_clock' if op in {'search', 'pending'} else 'unbounded',
                'recorded_as_of': 'not_applicable' if 'recorded_as_of' not in params else 'required' if op in {'compare_history', *MEMORY_PARAMETERS} else 'current_clock' if op in {'search', 'pending'} else 'unbounded',
                'unit': 'all_units' if op == 'search' else 'execution_unit' if op == 'evaluate' else 'unitless' if 'unit' in params else 'not_applicable'}
                for op, params in _LEDGER_PARAMETERS.items() if allow_outcome_writes or op not in _OUTCOME_WRITES}}
