"""Model-neutral forecasting protocol and conformance checks.

Statistical functions, local foundation models, subprocess sandboxes, and
remote inference APIs all cross this boundary before evaluation may score or
publish their values. Existing TSFM adapters remain valid through a bridge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Callable, Protocol, runtime_checkable


PROTOCOL_VERSION = "0.1"


class ForecastAdapterError(ValueError):
    """An adapter violated the model-neutral forecast contract."""


@dataclass(frozen=True)
class AdapterCapabilities:
    """Features an adapter explicitly promises to honor."""
    quantiles: bool = False
    past_covariates: bool = False
    future_covariates: bool = False
    panel: bool = False
    min_history: int | None = None
    max_horizon: int | None = None
    frequencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForecastRequest:
    history: tuple[float, ...]
    horizon: int
    season: int = 1
    quantiles: tuple[float, ...] = ()
    frequency: str | None = None
    cutoff: str | None = None
    known_time_cutoff: str | None = None
    past_covariates: tuple[tuple[float, ...], ...] = ()
    future_covariates: tuple[tuple[float, ...], ...] = ()
    related_series: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.history or any(not math.isfinite(float(value))
                                   for value in self.history):
            raise ForecastAdapterError("history must contain finite observations")
        if isinstance(self.horizon, bool) or self.horizon < 1:
            raise ForecastAdapterError("horizon must be a positive integer")
        if isinstance(self.season, bool) or self.season < 1:
            raise ForecastAdapterError("season must be a positive integer")
        if any(not 0 < value < 1 for value in self.quantiles) \
                or tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ForecastAdapterError("quantiles must be unique and increasing")
        self._validate_rows("past_covariates", self.past_covariates,
                            len(self.history))
        self._validate_rows("future_covariates", self.future_covariates,
                            self.horizon)
        for series in self.related_series:
            if len(series) != len(self.history) or any(
                    not math.isfinite(float(value)) for value in series):
                raise ForecastAdapterError(
                    "related_series must be finite and history-aligned")

    @staticmethod
    def _validate_rows(name: str, rows: tuple[tuple[float, ...], ...],
                       expected: int) -> None:
        if not rows:
            return
        widths = {len(row) for row in rows}
        if len(rows) != expected or len(widths) != 1 or 0 in widths or any(
                not math.isfinite(float(value)) for row in rows for value in row):
            raise ForecastAdapterError(
                f"{name} must be finite, rectangular, and length {expected}")

    @classmethod
    def from_values(cls, history: list[float], horizon: int, season: int,
                    *, quantiles: tuple[float, ...] = ()) -> "ForecastRequest":
        numeric = tuple(float(value) for value in history)
        if not numeric or any(not math.isfinite(value) for value in numeric):
            raise ForecastAdapterError("history must contain finite observations")
        if isinstance(horizon, bool) or int(horizon) < 1:
            raise ForecastAdapterError("horizon must be a positive integer")
        if isinstance(season, bool) or int(season) < 1:
            raise ForecastAdapterError("season must be a positive integer")
        ordered = tuple(float(value) for value in quantiles)
        if any(not 0 < value < 1 for value in ordered) \
                or tuple(sorted(set(ordered))) != ordered:
            raise ForecastAdapterError("quantiles must be unique and increasing")
        return cls(numeric, int(horizon), int(season), ordered)


@dataclass(frozen=True)
class ForecastResult:
    point: tuple[float, ...]
    quantiles: tuple[dict[float, float], ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, request: ForecastRequest) -> "ForecastResult":
        if len(self.point) != request.horizon:
            raise ForecastAdapterError(
                f"adapter returned {len(self.point)} points for horizon {request.horizon}")
        if any(not math.isfinite(float(value)) for value in self.point):
            raise ForecastAdapterError("adapter returned a non-finite point forecast")
        if self.quantiles is not None:
            if len(self.quantiles) != request.horizon:
                raise ForecastAdapterError("quantile rows do not match the horizon")
            for row in self.quantiles:
                values = []
                for quantile in request.quantiles:
                    if quantile not in row or not math.isfinite(float(row[quantile])):
                        raise ForecastAdapterError(
                            f"quantile row is missing finite q={quantile}")
                    values.append(float(row[quantile]))
                if values != sorted(values):
                    raise ForecastAdapterError("forecast quantiles are not monotone")
        return self

    def points(self) -> list[float]:
        return list(self.point)


@runtime_checkable
class ForecastAdapter(Protocol):
    name: str
    kind: str
    revision: str | None
    capabilities: AdapterCapabilities

    def forecast(self, request: ForecastRequest) -> ForecastResult: ...


class StatisticalAdapter:
    kind = "statistical"
    revision: str | None = None
    capabilities = AdapterCapabilities()

    def __init__(self, name: str,
                 predictor: Callable[[str, list[float], int, int], list[float]]):
        self.name, self._predictor = name, predictor

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        validate_capabilities(self.capabilities, request)
        point = self._predictor(
            self.name, list(request.history), request.horizon, request.season)
        return ForecastResult(tuple(float(value) for value in point), metadata={
            "adapter_kind": self.kind, "protocol_version": PROTOCOL_VERSION,
        }).validate(request)


class LegacyModelAdapter:
    """Bridge an existing local, sandbox, or API TSFM adapter."""
    kind = "model"

    def __init__(self, adapter: Any):
        self.adapter = adapter
        self.name = str(adapter.name)
        self.revision = getattr(adapter, "revision", None)
        self.capabilities = AdapterCapabilities(
            quantiles=bool(getattr(adapter, "supports_quantiles", False)),
            past_covariates=bool(getattr(adapter, "supports_past_covariates", False)),
            future_covariates=bool(getattr(adapter, "supports_future_covariates", False)),
            panel=bool(getattr(adapter, "supports_panel", False)),
            min_history=getattr(adapter, "min_history", None),
            max_horizon=getattr(adapter, "max_horizon", None),
            frequencies=tuple(getattr(adapter, "supported_frequencies", ()) or ()),
        )

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        validate_capabilities(self.capabilities, request)
        original = tuple(request.history)
        point = self.adapter.predict(
            list(request.history), request.horizon, request.season)
        quantile_rows = None
        if request.quantiles and bool(getattr(self.adapter,
                                               "supports_quantiles", False)):
            raw = self.adapter.predict_quantiles(
                list(request.history), request.horizon, request.season,
                request.quantiles)
            if raw is not None:
                quantile_rows = tuple({float(key): float(value)
                                       for key, value in row.items()}
                                      for row in raw)
        if request.history != original:
            raise ForecastAdapterError("adapter mutated immutable request history")
        return ForecastResult(
            tuple(float(value) for value in point), quantile_rows,
            {"adapter_kind": self.kind, "protocol_version": PROTOCOL_VERSION,
             "backend": str(getattr(self.adapter, "backend", "in_process")),
             "revision": self.revision or "unversioned"},
        ).validate(request)


def validate_capabilities(capabilities: AdapterCapabilities,
                          request: ForecastRequest) -> None:
    """Reject unsupported inputs instead of silently discarding them."""
    unsupported = []
    if capabilities.min_history is not None \
            and len(request.history) < capabilities.min_history:
        unsupported.append("history")
    if request.past_covariates and not capabilities.past_covariates:
        unsupported.append("past_covariates")
    if request.future_covariates and not capabilities.future_covariates:
        unsupported.append("future_covariates")
    if request.related_series and not capabilities.panel:
        unsupported.append("related_series")
    if capabilities.max_horizon is not None \
            and request.horizon > capabilities.max_horizon:
        unsupported.append("horizon")
    if capabilities.frequencies and request.frequency \
            and request.frequency not in capabilities.frequencies:
        unsupported.append("frequency")
    if unsupported:
        raise ForecastAdapterError(
            "adapter does not support request features: "
            + ", ".join(unsupported))


def predict_checked(adapter: ForecastAdapter, history: list[float],
                    horizon: int, season: int) -> list[float]:
    request = ForecastRequest.from_values(history, horizon, season)
    return adapter.forecast(request).validate(request).points()


def conformance_report(adapter: ForecastAdapter, *,
                       history: list[float] | None = None,
                       horizon: int = 4, season: int = 2) -> dict[str, Any]:
    """Exercise behavior every adapter must satisfy before admission."""
    minimum = int(getattr(getattr(adapter, "capabilities", None),
                          "min_history", 0) or 0)
    values = list(history or ([1, 2] * max(6, (minimum + 1) // 2)))
    request = ForecastRequest.from_values(values, horizon, season)
    checks: dict[str, bool] = {}
    failures: dict[str, str] = {}
    before = list(values)
    try:
        first = adapter.forecast(request).validate(request)
        checks["finite_exact_horizon"] = True
        checks["input_immutable"] = values == before
        second = adapter.forecast(request).validate(request)
        checks["deterministic_replay"] = first == second
    except Exception as error:
        failures["forecast"] = f"{type(error).__name__}: {error}"[:500]
        checks.setdefault("finite_exact_horizon", False)
        checks.setdefault("input_immutable", values == before)
        checks.setdefault("deterministic_replay", False)
    try:
        short = ForecastRequest.from_values(values, 1, season)
        checks["variable_horizon"] = len(adapter.forecast(short).validate(short).point) == 1
    except Exception as error:
        failures["variable_horizon"] = \
            f"{type(error).__name__}: {error}"[:500]
        checks["variable_horizon"] = False
    capabilities = getattr(adapter, "capabilities", AdapterCapabilities())
    return {"protocol_version": PROTOCOL_VERSION, "adapter": adapter.name,
            "kind": adapter.kind, "checks": checks,
            "capabilities": asdict(capabilities),
            **({"failures": failures} if failures else {}),
            "conformant": all(checks.values())}
