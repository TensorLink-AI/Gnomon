"""Direct, model-neutral inference. No implicit backtests, routing or action authority.

Register a callable over ForecastRequest/ForecastResult, or a provider object
with a forecast method. Library-specific fitting and conversion stays in the
user's callable. Registries are explicit instances, never process-global plugin
imports controlled by an agent's request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import threading
from typing import Callable, Any
from uuid import uuid4

from .forecast_adapter import (
    AdapterCapabilities, ForecastAdapterError, ForecastRequest, ForecastResult,
    validate_capabilities,
)

Predictor = Callable[[ForecastRequest], ForecastResult]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _freeze_request(request: ForecastRequest | dict[str, Any]) -> ForecastRequest:
    """Copy user containers before invoking an extension; keep the recorded input."""
    if isinstance(request, dict):
        request = ForecastRequest.from_dict(request)
    if not isinstance(request, ForecastRequest):
        raise ForecastAdapterError(
            'forecast request must be a ForecastRequest or dict, e.g. {"history": [1, 2, 3], "horizon": 2}')
    validated = ForecastRequest.from_dict(json.loads(_json(asdict(request))))
    values = asdict(validated)
    def number(value):
        # Preserve large integers exactly; float(value) would merge distinct
        # integers beyond binary64 precision. Integral floats have exact int equivalents.
        return int(value) if isinstance(value, float) and value.is_integer() else value
    for key in ("history", "quantiles"):
        values[key] = tuple(number(v) for v in values[key])
    for key in ("past_covariates", "future_covariates", "related_series"):
        values[key] = tuple(tuple(number(v) for v in row) for row in values[key])
    def timestamp(value):
        parsed = datetime.fromisoformat(value)
        # Calendar frequency grids retain their local offsets: a daily step
        # across DST is not necessarily 24 elapsed hours. Without a frequency,
        # the request describes explicit instants only and UTC is canonical.
        return (parsed.astimezone(timezone.utc) if parsed.tzinfo is not None and not validated.frequency else parsed).isoformat()
    for key in ("timestamps", "future_timestamps"):
        values[key] = tuple(timestamp(v) for v in values[key])
    for key in ("cutoff", "known_time_cutoff", "recorded_time_cutoff"):
        if values[key] is not None:
            values[key] = timestamp(values[key])
    return ForecastRequest.from_dict(values)


def _copy_result(result: ForecastResult) -> ForecastResult:
    # JSON-safe metadata is required for portable provenance, not arbitrary
    # Python objects whose repr could contain credentials or change on replay.
    return replace(result, point=tuple(result.point),
                   quantiles=None if result.quantiles is None else tuple(dict(row) for row in result.quantiles),
                   metadata=json.loads(_json(result.metadata)),
                   timestamps=tuple(result.timestamps),
                   sample_paths=None if result.sample_paths is None else tuple(tuple(row) for row in result.sample_paths))


@dataclass(frozen=True)
class ForecastExecution:
    execution_id: str
    fingerprint: str
    provider: str
    revision: str | None
    request: ForecastRequest
    result: ForecastResult
    cache_hit: bool = False
    evidence: str = "inference_only"
    action_authorized: bool = False
    provider_identity: dict | None = None
    cache: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "request_provenance": request_provenance(self.request),
                "effective_season": self.request.season, "result_contract_validated": True}


def request_provenance(request):
    return {"source": "caller_supplied_request", **{key: getattr(request, key) for key in (
        "cutoff", "known_time_cutoff", "recorded_time_cutoff", "snapshot_id", "series_id")},
        "history_start": request.timestamps[0] if request.timestamps else None,
        "history_end": request.timestamps[-1] if request.timestamps else None}


@dataclass
class _Provider:
    target: Any
    capabilities: AdapterCapabilities
    revision: str | None
    lifecycle: str
    factory: bool = False
    deterministic: bool = False


class InferenceEngine:
    """Explicit provider registry with optional bounded, versioned result reuse.

    A factory creates a fresh callable/provider for *each* request (including
    each evaluation fold). A pretrained provider can be retained across calls.
    Callables must not inspect future observations or external mutable training
    state; Gnomon cannot sandbox arbitrary user Python.
    """

    def __init__(self, *, ledger: Any = None, cache_size: int = 0):
        if type(cache_size) is not int or cache_size < 0:
            raise ForecastAdapterError("cache_size must be a nonnegative integer")
        self._providers: dict[str, _Provider] = {}
        self._cache: dict[str, ForecastResult] = {}
        self._cache_size = cache_size
        self._ledger = ledger
        self._lock = threading.RLock()

    @property
    def ledger(self):
        return self._ledger

    def register(self, name: str, predictor: Any, *,
                 capabilities: AdapterCapabilities | None = None,
                 revision: str | None = None, lifecycle: str = "stateless",
                 deterministic: bool = False) -> None:
        """Register a ForecastRequest -> ForecastResult callable or provider object.

        Results must echo request series_id and unit, and any supplied future
        timestamps. Example (import ForecastResult from gnomon first)::

            engine.register("custom", lambda r: ForecastResult(
                point=(r.history[-1],) * r.horizon,
                series_id=r.series_id, unit=r.unit, timestamps=r.future_timestamps),
                revision="custom-v1", deterministic=True)

        Return ForecastResult, not a dict. An object may expose forecast(request).
        """
        self._register(name, predictor, capabilities, revision, lifecycle, False, deterministic)

    def register_factory(self, name: str, factory: Callable[[], Any], *,
                         capabilities: AdapterCapabilities | None = None,
                         revision: str | None = None,
                         deterministic: bool = False) -> None:
        """Create fresh user-owned fitting state for every request/fold."""
        self._register(name, factory, capabilities, revision, "fresh_per_request", True, deterministic)

    def _register(self, name, target, capabilities, revision, lifecycle, factory, deterministic):
        if type(deterministic) is not bool:
            raise ForecastAdapterError("deterministic must be a boolean")
        if not isinstance(name, str) or not name.strip():
            raise ForecastAdapterError("provider name must be nonempty")
        if lifecycle not in {"stateless", "pretrained", "fresh_per_request"} or (lifecycle == "fresh_per_request" and not factory):
            raise ForecastAdapterError("fresh_per_request requires register_factory")
        if (factory and not callable(target)) or (not callable(target) and not callable(getattr(target, "forecast", None))):
            raise ForecastAdapterError("provider must be callable or implement forecast(request)")
        caps = capabilities if capabilities is not None else getattr(target, "capabilities", AdapterCapabilities())
        if not isinstance(caps, AdapterCapabilities):
            raise ForecastAdapterError("capabilities must be AdapterCapabilities")
        revision = revision if revision is not None else getattr(target, "revision", None)
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ForecastAdapterError("revision must be a nonempty string or None")
        with self._lock:
            if name in self._providers:
                raise ForecastAdapterError(f"provider {name!r} is already registered")
            self._providers[name] = _Provider(target, caps, revision, lifecycle, factory, deterministic)

    def capabilities(self) -> dict[str, Any]:
        with self._lock:
            return {name: {"capabilities": asdict(p.capabilities), "revision": p.revision,
                           "lifecycle": p.lifecycle, "deterministic": p.deterministic,
                           "batch": not p.factory and callable(getattr(p.target, "forecast_batch", None))}
                    for name, p in self._providers.items()}

    def cache_policy(self, provider: str | None = None) -> dict[str, Any]:
        policy = {"enabled": self._cache_size > 0, "max_entries": self._cache_size,
                  "scope": "session", "persistent": False}
        if provider is not None:
            with self._lock:
                registered = self._providers[provider]
                policy["provider_eligible"] = (
                    registered.deterministic and registered.revision not in {None, "latest", "unversioned"})
        return policy

    def cache_diagnostic(self, name, request, *, use_cache=True):
        """Validate/canonicalize and inspect a lookup without executing a provider."""
        if type(use_cache) is not bool:
            raise ForecastAdapterError('use_cache must be a boolean')
        _, canonical, fingerprint = self._prepare(name, request)
        policy = self.cache_policy(name)
        with self._lock:
            present = fingerprint in self._cache
        status = ("bypassed" if not use_cache else "disabled" if not policy["enabled"] else
                  "ineligible" if not policy["provider_eligible"] else "hit" if present else "miss")
        return {**policy, "status": status, "lookup_requested": use_cache, "fingerprint": fingerprint,
                "canonical_request": asdict(canonical), "provider_calls": 0,
                "reason": {"bypassed": "lookup_disabled_by_caller", "disabled": "cache_size_is_zero",
                           "ineligible": "provider_requires_determinism_and_explicit_revision",
                           "hit": "matching_entry_in_this_session", "miss": "no_matching_entry_in_this_session"}[status],
                "cli_processes_share_entries": False}

    def _prepare(self, name: str, request: ForecastRequest):
        with self._lock:
            provider = self._providers.get(name)
        if provider is None:
            raise ForecastAdapterError(f"unknown provider {name!r}; available providers: "
                                       + (", ".join(sorted(self._providers)) or "none registered")
                                       + (". InferenceEngine starts empty: register a provider with engine.register(name, predictor), "
                                          "or use GnomonSession.from_config() for built-in providers" if not self._providers else "")
                                       + ". Run gnomon capabilities with the same --providers-config to discover providers, "
                                       "or register a custom provider at startup.")
        request = _freeze_request(request)
        validate_capabilities(provider.capabilities, request)
        identity = {"provider": name, "revision": provider.revision,
                    "lifecycle": provider.lifecycle, "request": asdict(request)}
        fingerprint = hashlib.sha256(_json(identity).encode()).hexdigest()
        return provider, request, fingerprint

    def _finish(self, name, provider, request, fingerprint, result, cache_hit=False, use_cache=True):
        if not isinstance(result, ForecastResult):
            raise ForecastAdapterError("provider must return ForecastResult, not a dict or other value. "
                "Import ForecastResult from gnomon; e.g. return ForecastResult(point=(request.history[-1],) * request.horizon, "
                "series_id=request.series_id, unit=request.unit, timestamps=request.future_timestamps).")
        result = _copy_result(result).validate(request)
        execution = ForecastExecution(str(uuid4()), fingerprint, name, provider.revision,
                                      request, result, cache_hit, cache={**self.cache_policy(name),
                                          "lookup_requested": use_cache, "status": "bypassed" if not use_cache else
                                          "disabled" if not self._cache_size else "ineligible" if not self.cache_policy(name)["provider_eligible"] else
                                          "hit" if cache_hit else "miss"}, provider_identity={
                                          "lifecycle": provider.lifecycle, "capabilities": asdict(provider.capabilities)})
        # Recording failure is not silently ignored; callers should not claim
        # that an unrecorded invocation is auditable.
        if self._ledger is not None:
            self._ledger.record_execution(execution)
        if self._cache_size and provider.deterministic and provider.revision not in {None, "latest", "unversioned"}:
            with self._lock:
                self._cache[fingerprint] = _copy_result(result)
                while len(self._cache) > self._cache_size:
                    self._cache.pop(next(iter(self._cache)))
        return execution

    def forecast(self, name: str, request: ForecastRequest | dict[str, Any], *, use_cache: bool = True) -> ForecastExecution:
        if type(use_cache) is not bool:
            raise ForecastAdapterError('use_cache must be a boolean')
        provider, request, fingerprint = self._prepare(name, request)
        with self._lock:
            cached = self._cache.get(fingerprint) if use_cache else None
        if cached is not None:
            return self._finish(name, provider, request, fingerprint, cached, True)
        target = provider.target() if provider.factory else provider.target
        method = getattr(target, "forecast", target)
        try:
            result = method(request)
        finally:
            if provider.factory and callable(getattr(target, "close", None)):
                target.close()
        return self._finish(name, provider, request, fingerprint, result, use_cache=use_cache)

    def forecast_batch(self, name: str, requests: list[ForecastRequest | dict[str, Any]]) -> list[ForecastExecution]:
        """Validate the whole batch before dispatch. Results preserve input order.

        Provider batching is optional. Factories always execute independently,
        and batch requests bypass the cache to avoid hidden partial dispatch.
        """
        if not requests:
            return []
        prepared = [self._prepare(name, request) for request in requests]
        provider = prepared[0][0]
        batch = None if provider.factory else getattr(provider.target, "forecast_batch", None)
        if not callable(batch):
            return [self.forecast(name, request, use_cache=False) for _, request, _ in prepared]
        results = list(batch([request for _, request, _ in prepared]))
        if len(results) != len(prepared):
            raise ForecastAdapterError("provider returned the wrong batch size")
        for result, (_, request, _) in zip(results, prepared):
            if not isinstance(result, ForecastResult):
                raise ForecastAdapterError("provider must return ForecastResult, not a dict or other value. "
                    "Import ForecastResult from gnomon; e.g. return ForecastResult(point=(request.history[-1],) * request.horizon, "
                    "series_id=request.series_id, unit=request.unit, timestamps=request.future_timestamps).")
            _copy_result(result).validate(request)
        return [self._finish(name, provider, request, fingerprint, result)
                for result, (_, request, fingerprint) in zip(results, prepared)]

    def close(self) -> None:
        """Close retained provider resources. The caller owns the ledger."""
        with self._lock:
            providers = list(self._providers.values())
            self._providers.clear()
            self._cache.clear()
        errors = []
        for provider in providers:
            try:
                if not provider.factory and callable(getattr(provider.target, "close", None)):
                    provider.target.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
