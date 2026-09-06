"""Operator-owned execution context shared by Python, CLI and MCP.

The session is dependency injection, not another model protocol. Tool requests
select registered names; only startup configuration can load Python plugins,
resolve service URLs, select authentication or open a writable ledger.
"""

from __future__ import annotations

from dataclasses import asdict
from collections import OrderedDict
import importlib
import json
import os
from pathlib import Path
import tomllib
from typing import Any

from .contracts import GnomonError
from .forecast_adapter import AdapterCapabilities, ForecastAdapterError, ForecastRequest, StatisticalAdapter
from .inference import InferenceEngine
from .ledger import TemporalLedger
from .ephemeris import EphemerisProvider
from .product_contract import __version__, product_claims

_NUMBER_ARRAY = {"type": "array", "items": {"type": "number"}}
_STRING_ARRAY = {"type": "array", "items": {"type": "string"}}
REQUEST_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["history", "horizon"],
    "properties": {
        "history": {**_NUMBER_ARRAY, "minItems": 1},
        "horizon": {"type": "integer", "minimum": 1},
        "season": {"type": "integer", "minimum": 1},
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

INSPECT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["input"],
                  "properties": {**{name: {"type": "string"} for name in (
                      "input", "time_column", "target_column", "series_column", "frequency", "as_of",
                      "recorded_as_of", "store_path", "unit", "regrid")},
                      "repair": {"enum": ["off", "safe", "aggressive"]}}}
DESCRIBE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["data_ref", "statistic"],
                   "properties": {**{name: {"type": "string"} for name in ("data_ref", "series_id", "start", "end")},
                       "statistic": {"enum": ["mean", "median", "latest", "minimum", "maximum", "sum"]}}}
_FORECAST_COMMON = {"provider": {"type": "string"}, "use_cache": {"type": "boolean"}}
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
                    "budget": _BUDGET_SCHEMA, "replay": {"enum": ["recorded", "source_available"]}}},
    {"type": "object", "additionalProperties": False, "required": ["study_id"],
     "properties": {"study_id": {"type": "string"}}},
]}
ROUTE_SCHEMA = {"type": "object", "additionalProperties": False,
               "required": ["data_ref", "study_id", "candidates", "baseline", "horizon", "source_as_of", "recorded_as_of"],
               "properties": {**{key: {"type": "string"} for key in (
                   "data_ref", "study_id", "baseline", "source_as_of", "recorded_as_of", "series_id")},
                   "candidates": {**_STRING_ARRAY, "minItems": 1, "uniqueItems": True},
                   "horizon": {"type": "integer", "minimum": 1}, "season": {"type": "integer", "minimum": 1},
                   "min_folds": {"type": "integer", "minimum": 3},
                   "min_improvement": {"type": "number", "minimum": 0, "maximum": 1}}}
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
    "import_artifact": ("artifact_path", "project", "naive_timezone"),
    "import_tracking": ("registry_path", "project", "naive_timezone"),
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
    "import_artifact": ("artifact_path",), "import_tracking": ("registry_path",),
    "append_actual": (),
    "evaluate": (),
    "record_decision": ("execution_ids", "policy", "inputs", "action"),
    "append_decision_outcome": ("decision_id", "outcome", "source_available_at"),
}
_OUTCOME_WRITES = {"append_actual", "record_decision", "append_decision_outcome", "import_artifact", "import_tracking"}


def _strict(arguments, allowed, required=()):
    if not isinstance(arguments, dict):
        raise ForecastAdapterError("arguments must be an object")
    if set(arguments) - set(allowed):
        raise ForecastAdapterError("unknown arguments: " + ", ".join(sorted(set(arguments) - set(allowed))))
    missing = set(required) - set(arguments)
    if missing:
        raise ForecastAdapterError("missing arguments: " + ", ".join(sorted(missing)))


class GnomonSession:
    """An explicit registry, optional ledger and small tool dispatch boundary."""

    def __init__(self, engine: InferenceEngine | None = None, *, ledger: TemporalLedger | None = None,
                 allow_outcome_writes: bool = False, max_data_refs: int = 16, max_data_rows: int = 100_000,
                 evaluation_limits: dict | None = None, result_limits: dict | None = None,
                 enable_temporal: bool = False):
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
    def from_config(cls, path: str | Path | None = None) -> "GnomonSession":
        """Load only an explicitly supplied TOML path. No cwd config search."""
        config, directory = {}, Path.cwd()
        if path is not None:
            location = Path(path).expanduser().resolve()
            directory = location.parent
            with location.open("rb") as handle:
                config = tomllib.load(handle)
        _strict(config, {"schema_version", "ledger_path", "cache_size", "allow_outcome_writes", "providers",
                         "max_data_refs", "max_data_rows", "evaluation_limits", "result_limits", "enable_temporal"})
        if config.get("schema_version", 1) != 1:
            raise ForecastAdapterError("unsupported provider configuration schema")
        ledger_path = config.get("ledger_path")
        ledger = TemporalLedger(directory / ledger_path) if ledger_path else None
        engine = InferenceEngine(ledger=ledger, cache_size=config.get("cache_size", 0))
        session = cls(engine, ledger=ledger, allow_outcome_writes=config.get("allow_outcome_writes", False),
                      max_data_refs=config.get("max_data_refs", 16), max_data_rows=config.get("max_data_rows", 100_000),
                      evaluation_limits=config.get("evaluation_limits"), result_limits=config.get("result_limits"),
                      enable_temporal=config.get("enable_temporal", False))
        from .models import BASELINES, predict
        for name in sorted(BASELINES):
            adapter = StatisticalAdapter(name, predict)
            engine.register(name, adapter, revision=f"gnomon/{__version__}/{name}", deterministic=True)
        try:
            for name, spec in config.get("providers", {}).items():
                session._configure_provider(name, spec)
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
            if spec.get("discover", False):
                provider.register_models(self.engine, prefix=name + "/")
        elif kind in {"callable", "factory"}:
            _strict(spec, {"kind", "entrypoint", "capabilities", "revision", "deterministic", "lifecycle"}, {"entrypoint"})
            entrypoint = spec["entrypoint"]
            if not isinstance(entrypoint, str) or entrypoint.count(":") != 1:
                raise ForecastAdapterError("entrypoint must be module:attribute")
            module, attribute = entrypoint.split(":")
            target = getattr(importlib.import_module(module), attribute)
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

    def capabilities(self) -> dict:
        return {"schema_version": "1", "status": "ok", "runtime_version": __version__,
                "product_contract": product_claims(),
                "interfaces": {"python": True, "cli": True, "mcp": True},
                "mcp_profile": {"active": "execution", "available": ["execution"],
                                "visible_tools": [tool["name"] for tool in self.tools()]},
                "providers": self.engine.capabilities(),
                "ledger": {"enabled": self.ledger is not None, "outcome_writes": self.allow_outcome_writes},
                "temporal": {"enabled": self.enable_temporal, "semantics": "explicit_facts_not_natural_language"},
                "data": {"reference_scope": "session", "max_refs": self.data.max_refs, "max_retained_rows": self.data.max_rows},
                "evaluation_limits": asdict(self.evaluation_limits),
                "result_limits": {**asdict(self.results.limits), "scope": "session", "offset_unit": "unicode_codepoints",
                                  "bound": "compact_structured_payload_utf8_not_provider_memory"},
                "semantics": {"forecast": "inference_only", "calibration": "not_implied", "action_authorized": False}}

    def forecast(self, provider: str, request: ForecastRequest, *, use_cache: bool = True) -> dict:
        run = self.engine.forecast(provider, request, use_cache=use_cache)
        return {"schema_version": "1", "status": "ok", "execution_id": run.execution_id,
                "fingerprint": run.fingerprint, "provider": run.provider, "revision": run.revision,
                "cache_hit": run.cache_hit, "result": asdict(run.result),
                "evidence": run.evidence, "action_authorized": run.action_authorized,
                "recorded": self.ledger is not None}

    def evaluate(self, data_ref: str, *, budget: dict | None = None, **kwargs) -> dict:
        from .backtesting import EvaluationBudget, evaluate_reference
        limits = asdict(self.evaluation_limits)
        if budget is not None:
            _strict(budget, limits)
        requested = EvaluationBudget.from_dict({**limits, **(budget or {})})
        for key, limit in limits.items():
            value = getattr(requested, key)
            if limit is not None and (value is None or value > limit):
                raise ForecastAdapterError("evaluation budget cannot exceed operator startup limits")
        report = evaluate_reference(self.engine, self.data, data_ref, budget=requested, **kwargs)
        if self.ledger is None:
            self._studies[report["study_id"]] = self.results.put(report)
            while len(self._studies) > 3:
                self._studies.popitem(last=False)
        return report

    def route(self, data_ref: str, **kwargs) -> dict:
        from .study_routing import route_study
        return route_study(self.engine, self.data, data_ref, max_folds=self.evaluation_limits.max_folds, **kwargs)

    def call(self, name: str, arguments: dict[str, Any], *, compact: bool = True) -> dict:
        """Shared tool dispatch; full CLI/Python mode does not leave ephemeral references."""
        if type(compact) is not bool:
            raise GnomonError("INVALID_ARGUMENTS", "compact must be a boolean")
        result = self._call(name, arguments)
        if compact and name == "gnomon_evaluate" and "study_id" not in arguments:
            from .backtesting import compact_study
            result = compact_study(result)
        return self.results.project(result) if compact else result

    def _call(self, name: str, arguments: dict[str, Any]) -> dict:
        try:
            if not isinstance(arguments, dict):
                raise ForecastAdapterError("arguments must be an object")
            if name == "gnomon_read":
                _strict(arguments, READ_SCHEMA["properties"], READ_SCHEMA["required"])
                return self.results.read(**arguments)
            if name == "gnomon_temporal":
                if not self.enable_temporal:
                    raise GnomonError("UNKNOWN_TOOL", "Temporal operations require enable_temporal=true at session startup.")
                from .temporal_ops import temporal_operation
                try:
                    return temporal_operation(**arguments)
                except ValueError as exc:
                    raise ForecastAdapterError(str(exc)) from None
            if name == "gnomon_capabilities":
                _strict(arguments, ())
                return self.capabilities()
            if name == "gnomon_forecast":
                if type(arguments.get("use_cache", True)) is not bool:
                    raise ForecastAdapterError("use_cache must be a boolean")
                if "data_ref" in arguments:
                    _strict(arguments, {"provider", "data_ref", "horizon", "series_id", "season", "quantiles", "use_cache"},
                            {"provider", "data_ref", "horizon"})
                    request = self.data.request(**{k: v for k, v in arguments.items() if k not in {"provider", "use_cache"}})
                else:
                    _strict(arguments, {"provider", "request", "use_cache"}, {"provider", "request"})
                    request = ForecastRequest.from_dict(arguments["request"])
                result = self.forecast(arguments["provider"], request, use_cache=arguments.get("use_cache", True))
                if "data_ref" in arguments:
                    result["data_ref"] = arguments["data_ref"]
                return result
            if name in {"gnomon_inspect", "gnomon_describe"}:
                schema = INSPECT_SCHEMA if name == "gnomon_inspect" else DESCRIBE_SCHEMA
                _strict(arguments, schema["properties"], schema["required"])
                return getattr(self.data, "inspect" if name == "gnomon_inspect" else "describe")(**arguments)
            if name == "gnomon_ledger":
                return self._ledger_call(arguments)
            if name == "gnomon_route":
                _strict(arguments, ROUTE_SCHEMA["properties"], ROUTE_SCHEMA["required"])
                return self.route(**arguments)
            if name == "gnomon_evaluate":
                if "study_id" in arguments:
                    _strict(arguments, {"study_id"}, {"study_id"})
                    study_id = arguments["study_id"]
                    if study_id in self._studies:
                        return self.results.value(self._studies[study_id])
                    if self.ledger is not None:
                        return self.ledger.study(study_id)
                    raise ForecastAdapterError("study is no longer retained; configure a ledger for durable retrieval")
                schema = EVALUATE_SCHEMA["oneOf"][0]
                _strict(arguments, schema["properties"], schema["required"])
                return self.evaluate(**arguments)
            raise GnomonError("UNKNOWN_TOOL", "This execution session does not expose that tool.")
        except (ForecastAdapterError, TypeError, KeyError) as exc:
            raise GnomonError("INVALID_ARGUMENTS", str(exc)) from None

    def _ledger_call(self, arguments):
        if self.ledger is None:
            raise GnomonError("LEDGER_NOT_CONFIGURED", "Configure the ledger at session startup.")
        operation = arguments.get("operation")
        if operation not in _LEDGER_PARAMETERS:
            raise ForecastAdapterError("unknown ledger operation")
        if operation in _OUTCOME_WRITES and not self.allow_outcome_writes:
            raise GnomonError("OUTCOME_WRITES_DISABLED", "Outcome writes require operator startup authorization.")
        _strict(arguments, {"operation", *_LEDGER_PARAMETERS[operation]}, {"operation", *_LEDGER_REQUIRED[operation]})
        result = getattr(self.ledger, operation)(**{k: v for k, v in arguments.items() if k != "operation"})
        return {"schema_version": "1", "status": "ok", "operation": operation, "result": result}

    def tools(self) -> list[dict]:
        tools = [
            {"name": "gnomon_read", "description": "Read exact retained result JSON text pages, optionally at a JSON pointer. Concatenate pages at next_offset; no provider calls. References expire with the session or LRU eviction.",
             "inputSchema": READ_SCHEMA},
            {"name": "gnomon_capabilities", "description": "List this session's registered providers and storage capabilities.",
             "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
            {"name": "gnomon_forecast", "description": "Execute a registered provider. No implicit backtest, calibration claim or action permission.",
             "inputSchema": FORECAST_SCHEMA},
            {"name": "gnomon_inspect", "description": "Validate file/store data and freeze a session-local data_ref. Discloses cutoffs and repairs.",
             "inputSchema": INSPECT_SCHEMA},
            {"name": "gnomon_describe", "description": "Compute an exact observed statistic over one frozen series and optional inclusive timestamp window.",
             "inputSchema": DESCRIBE_SCHEMA},
            {"name": "gnomon_evaluate", "description": "Optional matched vintage-aware backtest with explicit baseline and dispatch budgets, or retrieve a full study.",
             "inputSchema": EVALUATE_SCHEMA},
        ]
        if self.enable_temporal:
            from .temporal_ops import TEMPORAL_SCHEMA
            tools.append({"name": "gnomon_temporal", "description": "Calculate explicit dates/instants, half-open interval relations and stable event order. Local times require zones and ambiguous folds; no implicit now or causal inference.",
                          "inputSchema": TEMPORAL_SCHEMA})
        if self.ledger is not None:
            tools.append({"name": "gnomon_route", "description": "Recommend from one immutable matched study with explicit source/recorded cutoffs; rescore without model calls or action permission.",
                          "inputSchema": ROUTE_SCHEMA})
            variants = []
            for operation, parameters in _LEDGER_PARAMETERS.items():
                if operation in _OUTCOME_WRITES and not self.allow_outcome_writes:
                    continue
                properties = {"operation": {"const": operation}}
                for p in parameters:
                    properties[p] = ({"type": "number"} if p == "value" else
                                     {"type": ["string", "null"], "minLength": 1} if p == "unit" else
                                     {"type": "boolean"} if p == "allow_partial" else
                                     {"type": "integer", "minimum": 1, "maximum": 100 if p == "limit" else 1_000_000} if p in {"limit", "horizon"} else
                                     {"enum": ["waiting", "ready", "scored", "stale", "unscorable"]} if p == "status" else
                                     {"type": "object", "minProperties": 2, "maxProperties": 8,
                                      "additionalProperties": {"type": "string", "minLength": 1}} if p == "providers" else
                                     {"type": "object"} if p in {"policy", "inputs", "action", "outcome"} else
                                     {**_STRING_ARRAY, "minItems": 1, "maxItems": 100, "uniqueItems": True} if p == "execution_ids" else
                                     {"type": "string"})
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
                                 "required": ["operation", *_LEDGER_REQUIRED[operation]]})
            tools.append({"name": "gnomon_ledger", "description": "Find forecasts and feedback status, compare matched production history, score named runs or record authorized actuals. Reads never run models; exact scoring retries reuse evidence.",
                          "inputSchema": {"type": "object", "oneOf": variants}})
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
