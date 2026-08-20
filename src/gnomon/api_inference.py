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

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from .config import APIProviderConfig
from .tsfm import TSFMAdapter, TSFMError, TSFMUnavailable
from .forecast_adapter import (
    PROTOCOL_VERSION, AdapterCapabilities, ForecastAdapterError,
    ForecastRequest, ForecastResult,
)

logger = logging.getLogger(__name__)


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
        self._timeout = provider.timeout or timeout
        self._retry = provider.retry or retry

        # Infer metadata from name
        self._params_m = self._lookup_params(name)
        self._supports_quantiles = self._lookup_supports_quantiles(name)
        self.capabilities = AdapterCapabilities(
            quantiles=self._supports_quantiles)

    @staticmethod
    def _lookup_params(name: str) -> float:
        params_map = {
            "chronos_bolt_mini": 21.0,
            "chronos_bolt_small": 48.0,
            "toto2_4m": 4.14,
            "toto2_22m": 22.0,
            "flowstate": 18.5,
            "ttm": 3.0,
            "moirai2_small": 14.0,
            "moment_small": 38.0,
        }
        return params_map.get(name, 0.0)

    @staticmethod
    def _lookup_supports_quantiles(name: str) -> bool:
        no_quantiles = {"ttm", "moment_small"}
        return name not in no_quantiles

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
        """Make the HTTP request with retry logic."""
        url = self._provider.url
        if not url:
            raise TSFMUnavailable(f"No API URL configured for TSFM {self.name}")

        # Build auth headers
        headers = {"Content-Type": "application/json"}
        auth = self._provider.auth
        if auth.type == "bearer":
            token = os.environ.get(auth.token_env, "")
            if not token:
                raise TSFMUnavailable(
                    f"API token env var '{auth.token_env}' not set for {self.name}"
                )
            headers["Authorization"] = f"Bearer {token}"
        elif auth.type == "header":
            token = os.environ.get(auth.token_env, "")
            if not token:
                raise TSFMUnavailable(
                    f"API token env var '{auth.token_env}' not set for {self.name}"
                )
            headers[auth.header] = token

        body = json.dumps(payload).encode("utf-8")
        last_error: str = ""

        for attempt in range(self._retry + 1):
            try:
                req = urllib.request.Request(
                    url, data=body, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    response_data = json.loads(resp.read().decode("utf-8"))
                    if "error" in response_data:
                        raise TSFMError(
                            f"API for {self.name} returned error: {response_data['error']}"
                        )
                    return response_data

            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
                if e.code in (401, 403):
                    raise TSFMUnavailable(
                        f"Auth failed for {self.name} API: {last_error}"
                    ) from e
                if e.code == 404:
                    raise TSFMUnavailable(
                        f"API endpoint not found for {self.name}: {url}"
                    ) from e
                # 5xx — retry
                logger.warning(
                    "API call for %s failed (attempt %d): %s",
                    self.name, attempt + 1, last_error,
                )

            except urllib.error.URLError as e:
                last_error = f"Connection error: {e.reason}"
                logger.warning(
                    "API call for %s failed (attempt %d): %s",
                    self.name, attempt + 1, last_error,
                )

            except json.JSONDecodeError as e:
                raise TSFMError(
                    f"API for {self.name} returned invalid JSON: {e}"
                ) from e

        raise TSFMError(
            f"API for {self.name} failed after {self._retry + 1} attempts: {last_error}"
        )

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
