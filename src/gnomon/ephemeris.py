"""Optional Ephemeris inference connector using its quantile wire protocol.

Contract traced through service schema, router and deployed inference routes
at revision f5d3f53b17e56d8c7ed0cbae61e043b8667f236a. The deployment URL
is not fixed. This client does not import service code, torch or a model catalogue.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .forecast_adapter import (
    AdapterCapabilities, ForecastAdapterError, ForecastRequest, ForecastResult,
    validate_capabilities,
)
from .http_transport import JSONTransport


class EphemerisProvider:
    kind = "api"
    # Neither /models nor /forecast attests a weights revision. A deployment
    # alias or source commit is not a substitute for a served-model revision.
    revision = None

    def __init__(self, base_url: str, *, model: str | None = None,
                 mode: str | None = None, combine: str = "vincentize",
                 token_env: str | None = None, timeout: float = 30.0,
                 transport: JSONTransport | None = None):
        mode = mode or ("explicit" if model else "route")
        if mode not in {"explicit", "route", "ensemble"} or (mode == "explicit") != bool(model):
            raise ForecastAdapterError("explicit mode requires a model; route/ensemble must not specify one")
        if combine not in {"vincentize", "mixture"}:
            raise ForecastAdapterError("unsupported Ephemeris combination method")
        self.model, self.mode, self.combine = model, mode, combine
        self.name = f"ephemeris/{model or mode}"
        self.transport = transport or JSONTransport(base_url, token_env=token_env, timeout=timeout)
        self.capabilities = AdapterCapabilities(quantiles=True, min_history=2,
                                               past_covariates=True, future_covariates=True)

    def models(self) -> list[dict[str, Any]]:
        """Read deployment discovery; never substitute the local TSFM list."""
        response = self.transport.call("/models")
        rows = response.get("models")
        if not isinstance(rows, list) or any(not isinstance(row, dict)
                or not isinstance(row.get("name"), str) for row in rows):
            raise ForecastAdapterError("invalid Ephemeris models response")
        return rows

    def register_models(self, engine, *, prefix: str = "ephemeris/") -> list[str]:
        """Register currently enabled, healthy explicit models on an engine."""
        registered = []
        for row in self.models():
            if row.get("enabled") is not True or row.get("healthy") is not True:
                continue
            provider = EphemerisProvider(self.transport.base_url, model=row["name"], transport=self.transport)
            provider.capabilities = replace(provider.capabilities,
                                            past_covariates=row.get("covariates") is True,
                                            future_covariates=row.get("covariates") is True)
            name = prefix + row["name"]
            engine.register(name, provider, lifecycle="pretrained")
            registered.append(name)
        return registered

    @staticmethod
    def _covariates(req: ForecastRequest) -> dict | None:
        tables = {}
        for kind in ("past", "future"):
            rows = getattr(req, kind + "_covariates")
            names = getattr(req, kind + "_covariate_names")
            if rows:
                names = names or tuple(f"covariate_{i}" for i in range(len(rows[0])))
                tables[kind] = {name: [row[i] for row in rows] for i, name in enumerate(names)}
            else:
                tables[kind] = {}
        if set(tables["future"]) - set(tables["past"]):
            raise ForecastAdapterError("Ephemeris future covariates require matching named past histories")
        return tables if any(tables.values()) else None

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        return self.forecast_batch([request])[0]

    def forecast_batch(self, requests: list[ForecastRequest]) -> list[ForecastResult]:
        if not requests:
            return []
        first = requests[0]
        levels = sorted({.5, *(q for req in requests for q in req.quantiles)})
        # The service formats probability keys to six significant digits.
        # Refuse silent quantile rounding/collision at this protocol boundary.
        if any(float(f"{q:.6g}") != q for q in levels):
            raise ForecastAdapterError("Ephemeris quantiles support at most six significant digits")
        for req in requests:
            validate_capabilities(self.capabilities, req)
            if req.season != 1:
                raise ForecastAdapterError("Ephemeris accepts a frequency hint, not an explicit seasonal period")
            if (req.horizon, req.frequency) != (first.horizon, first.frequency):
                raise ForecastAdapterError("Ephemeris batches require the same horizon and frequency")
        covariates = [self._covariates(req) for req in requests]
        payload = {"mode": self.mode, "series": [list(req.history) for req in requests],
                   "horizon": first.horizon, "freq": first.frequency,
                   "quantiles": levels, "combine": self.combine}
        if self.model:
            payload["model"] = self.model
        if any(covariates):
            payload["covariates"] = covariates
        response = self.transport.call("/forecast", payload)
        rows, meta = response.get("forecasts"), response.get("meta")
        if not isinstance(rows, list) or len(rows) != len(requests) or not isinstance(meta, dict):
            raise ForecastAdapterError("invalid Ephemeris forecast response shape")
        models = meta.get("models_used")
        if not isinstance(models, list) or not models or any(not isinstance(m, str) or not m for m in models):
            raise ForecastAdapterError("Ephemeris must report models_used")
        if meta.get("mode") != self.mode or (self.model and models != [self.model]):
            raise ForecastAdapterError("Ephemeris response model/mode differs from request")
        results = []
        for req, row in zip(requests, rows):
            try:
                quantiles = {float(q): values for q, values in row["quantiles"].items()}
                if len(quantiles) != len(row["quantiles"]) or set(levels) - set(quantiles) or any(not isinstance(values, list) or len(values) != req.horizon
                                                     for values in quantiles.values()):
                    raise ValueError("missing or malformed quantiles")
                points = tuple(float(v) for v in quantiles[.5])
                marginal = tuple({q: float(values[i]) for q, values in quantiles.items()}
                                 for i in range(req.horizon))
            except (KeyError, TypeError, ValueError, AttributeError):
                raise ForecastAdapterError("invalid Ephemeris quantile response") from None
            result = ForecastResult(points, marginal, metadata={
                "provider": "ephemeris", "point_definition": "median",
                "model_revision": None, "revision_attested": False,
                "alignment_basis": "request_order", "service": meta,
                "season_handling": "service uses frequency hint; no fixed-period control",
            }, timestamps=req.future_timestamps, series_id=req.series_id, unit=req.unit)
            # Check *all* returned uncertainty, even when caller asked only for a point.
            result.validate(req)
            results.append(result)
        return results
