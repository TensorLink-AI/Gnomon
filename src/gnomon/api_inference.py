"""API inference backend for TSFMs.

When TSFMs are served via HTTP (hosted, managed, or self-served via
Triton/vLLM/TorchServe), Gnomon can call them without installing any
model dependencies locally. This is the ``api`` backend alternative
to the ``sandbox`` backend (isolated venvs).

Protocol — Gnomon sends a POST with::

    {
        "model": "<model_name_on_server>",
        "history": [float, ...],
        "horizon": int,
        "quantiles": [0.1, 0.5, 0.9]    // optional
    }

And expects a response::

    {
        "point": [float, ...],           // horizon point forecasts
        "quantiles": [                    // optional, per-step
            {"0.1": float, "0.5": float, "0.9": float},
            ...
        ]
    }

This is intentionally a simple JSON protocol — not tied to any specific
serving framework. A thin adapter on the server side translates to
whatever the TSFM library expects.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit
from typing import Any

from .config import APIProviderConfig
from .tsfm import (
    TSFMError,
    TSFMUnavailable,
    tsfm_parameter_count,
    tsfm_supports_quantiles,
)
from .forecast_adapter import (
    PROTOCOL_VERSION, AdapterCapabilities, ForecastAdapterError,
    ForecastRequest, ForecastResult,
)
from .http_transport import InferenceHTTPError, JSONTransport


class APIAdapter:
    """A TSFM adapter that calls a remote HTTP endpoint for inference.

    Implements the same ``TSFMAdapter`` protocol as the in-process and
    sandbox adapters, but delegates all computation to a remote server.
    No torch, no model weights, no local deps — just HTTP.
    """

    backend = "api"
    def __init__(
        self,
        name: str,
        provider: APIProviderConfig,
        timeout: int = 60,
        retry: int = 2,
    ):
        self.name = name
        self._provider = provider
        self.revision = provider.revision or None
        self._timeout = provider.timeout if provider.timeout is not None else timeout
        self._retry = provider.retry if provider.retry is not None else retry

        self._params_m = tsfm_parameter_count(name)
        self._supports_quantiles = tsfm_supports_quantiles(name)
        self.capabilities = AdapterCapabilities(
            quantiles=self._supports_quantiles)

    @property
    def params_m(self) -> float:
        return self._params_m

    @property
    def supports_quantiles(self) -> bool:
        return self._supports_quantiles

    def _build_request(
        self,
        history: list[float],
        horizon: int,
        want_quantiles: bool,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "model": self._provider.model or self.name,
            "history": history,
            "horizon": horizon,
            "season": 1,
        }
        if self.revision:
            payload["revision"] = self.revision
        if want_quantiles and self._supports_quantiles:
            payload["quantiles"] = list(quantiles)
        return payload

    def _call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Compatibility wire format, shared bounded transport; no POST replay."""
        url = self._provider.url
        if not url:
            raise TSFMUnavailable(f"No API URL configured for TSFM {self.name}")
        auth = self._provider.auth
        if auth.type not in {"none", "bearer", "header"}:
            raise TSFMUnavailable("unsupported API authentication type")
        parsed = urlsplit(url)
        try:
            transport = JSONTransport(
                urlunsplit((parsed.scheme, parsed.netloc, "", parsed.query, parsed.fragment)),
                token_env=auth.token_env if auth.type != "none" else None,
                auth_header=auth.header if auth.type == "header" else "Authorization",
                auth_prefix="" if auth.type == "header" else "Bearer ",
                timeout=self._timeout,
            )
            response = transport.call(parsed.path or "/", payload)
        except InferenceHTTPError as exc:
            if exc.status in {401, 403, 404}:
                raise TSFMUnavailable(str(exc)) from None
            raise TSFMError(str(exc)) from None
        except ForecastAdapterError as exc:
            raise TSFMUnavailable(str(exc)) from None
        if "error" in response:
            raise TSFMError("inference service returned an error object")
        return response

    def predict(self, history: list[float], horizon: int, season: int) -> list[float]:
        payload = self._build_request(history, horizon, want_quantiles=False)
        payload["season"] = season
        response = self._call_api(payload)
        point = response.get("point")
        if point is None:
            raise TSFMError(f"API for {self.name} returned no point forecast")
        try:
            request = ForecastRequest.from_values(history, horizon, season)
            return ForecastResult(tuple(float(value) for value in point)).validate(
                request).points()
        except (TypeError, ValueError, ForecastAdapterError) as exc:
            raise TSFMError(
                f"API for {self.name} violated the forecast contract: {exc}") from exc

    def predict_quantiles(
        self,
        history: list[float],
        horizon: int,
        season: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[dict[str, float]] | None:
        if not self._supports_quantiles:
            return None
        payload = self._build_request(
            history, horizon, want_quantiles=True, quantiles=quantiles,
        )
        payload["season"] = season
        response = self._call_api(payload)
        raw = response.get("quantiles")
        if raw is None:
            return None
        try:
            rows = tuple({float(key): float(value) for key, value in row.items()}
                         for row in raw)
            request = ForecastRequest.from_values(
                history, horizon, season, quantiles=quantiles)
            point = response.get("point") or [row.get(.5, 0.0) for row in rows]
            return [dict(row) for row in ForecastResult(
                tuple(float(value) for value in point), rows).validate(
                    request).quantiles or ()]
        except (TypeError, ValueError, ForecastAdapterError) as exc:
            raise TSFMError(
                f"API for {self.name} violated the quantile contract: {exc}") from exc
