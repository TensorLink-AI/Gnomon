"""Model-neutral forecasting protocol and conformance checks.

Statistical functions, local foundation models, subprocess sandboxes, and
remote inference APIs all cross this boundary before results are recorded or
scored. Providers implement the same request/result contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
import math
from typing import Any, Callable, Protocol, runtime_checkable


PROTOCOL_VERSION = "0.1"


class ForecastAdapterError(ValueError):
    """An adapter violated the model-neutral forecast contract."""

    def __init__(self, message, *, details=None, repair_options=None):
        super().__init__(message)
        self.details = details or {}
        self.repair_options = repair_options


def point_error_metrics(pairs) -> dict:
    """Finite point losses shared by backtests and durable ledger scoring."""
    errors = [point - actual for point, actual in pairs]
    if not errors:
        return {"n": 0, "mae": None, "rmse": None, "bias": None}
    if not all(math.isfinite(error) for error in errors):
        raise ForecastAdapterError("forecast error exceeds finite numeric range")
    n = len(errors)
    try:
        metrics = {"mae": math.fsum(abs(error) / n for error in errors),
                   "rmse": math.hypot(*(error / math.sqrt(n) for error in errors)),
                   "bias": math.fsum(error / n for error in errors)}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise OverflowError
    except (OverflowError, ValueError):
        raise ForecastAdapterError("forecast metrics exceed finite numeric range") from None
    return {"n": n, **metrics}


@dataclass(frozen=True)
class AdapterCapabilities:
    """Features an adapter explicitly promises to honor."""
    quantiles: bool = False
    past_covariates: bool = False
    future_covariates: bool = False
    panel: bool = False
    sample_paths: bool = False
    min_history: int | None = None
    max_horizon: int | None = None
    frequencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("quantiles", "past_covariates", "future_covariates", "panel", "sample_paths"):
            if type(getattr(self, name)) is not bool:
                raise ForecastAdapterError(f"{name} capability must be a boolean")
        for name in ("min_history", "max_horizon"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ForecastAdapterError(f"{name} must be a positive integer")


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
    timestamps: tuple[str, ...] = ()
    future_timestamps: tuple[str, ...] = ()
    series_id: str | None = None
    unit: str | None = None
    snapshot_id: str | None = None
    recorded_time_cutoff: str | None = None
    past_covariate_names: tuple[str, ...] = ()
    future_covariate_names: tuple[str, ...] = ()
    samples: int = 0

    def __post_init__(self) -> None:
        for name in ("frequency", "cutoff", "known_time_cutoff", "series_id", "unit", "snapshot_id", "recorded_time_cutoff"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ForecastAdapterError(f"{name} must be a nonempty string or None")
        try:
            finite_history = bool(self.history) and all(math.isfinite(float(value)) for value in self.history)
        except (ValueError, TypeError, OverflowError):
            finite_history = False
        if not finite_history:
            raise ForecastAdapterError("history must contain finite observations")
        if type(self.horizon) is not int or self.horizon < 1:
            raise ForecastAdapterError("horizon must be a positive integer")
        if type(self.season) is not int or self.season < 1:
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
        if type(self.samples) is not int or self.samples < 0:
            raise ForecastAdapterError("samples must be a nonnegative integer")
        for name in ("past", "future"):
            names = getattr(self, f"{name}_covariate_names")
            rows = getattr(self, f"{name}_covariates")
            if names and (not rows or len(names) != len(rows[0])
                          or len(set(names)) != len(names)
                          or any(not isinstance(n, str) or not n for n in names)):
                raise ForecastAdapterError(f"{name}_covariate_names must match columns")
        history_times = self._validate_times("timestamps", self.timestamps, len(self.history))
        future_times = self._validate_times("future_timestamps", self.future_timestamps, self.horizon)
        times = [*history_times, *future_times]
        cutoffs = [self._parse_time(value) for value in
                   (self.cutoff, self.known_time_cutoff, self.recorded_time_cutoff)
                   if value is not None]
        if len({value.tzinfo is not None for value in [*times, *cutoffs]}) > 1:
            raise ForecastAdapterError("timestamps and cutoffs cannot mix naive and aware time")
        if history_times and future_times and history_times[-1] >= future_times[0]:
            raise ForecastAdapterError("future timestamps must follow history")
        if self.cutoff is not None:
            cutoff = self._parse_time(self.cutoff)
            if history_times and history_times[-1] > cutoff:
                raise ForecastAdapterError("history extends beyond cutoff")
            if future_times and future_times[0] <= cutoff:
                raise ForecastAdapterError("future timestamps must follow cutoff")

    @staticmethod
    def _parse_time(value: str) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            raise ForecastAdapterError("timestamps must be ISO 8601 strings") from exc

    @classmethod
    def _validate_times(cls, name: str, values: tuple[str, ...], expected: int) -> list[datetime]:
        if not values:
            return []
        parsed = [cls._parse_time(value) for value in values]
        if len({value.tzinfo is not None for value in parsed}) > 1:
            raise ForecastAdapterError(f"{name} cannot mix naive and aware time")
        if len(values) != expected or any(a >= b for a, b in zip(parsed, parsed[1:])):
            raise ForecastAdapterError(f"{name} must be strictly increasing and length {expected}")
        return parsed

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
    def from_dict(cls, value: dict[str, Any]) -> "ForecastRequest":
        if not isinstance(value, dict):
            raise ForecastAdapterError("forecast request must be an object")
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ForecastAdapterError("unknown forecast request fields: " + ", ".join(sorted(unknown)))
        missing = {"history", "horizon"} - set(value)
        if missing:
            raise ForecastAdapterError("missing forecast request fields: " + ", ".join(sorted(missing)))
        converted = dict(value)
        arrays = {"history", "quantiles", "timestamps", "future_timestamps",
                  "past_covariate_names", "future_covariate_names"}
        matrices = {"past_covariates", "future_covariates", "related_series"}
        try:
            for name in (arrays | matrices) & converted.keys():
                if not isinstance(converted[name], (list, tuple)):
                    raise ForecastAdapterError(f"{name} must be an array")
                converted[name] = (tuple(tuple(row) for row in converted[name])
                                   if name in matrices else tuple(converted[name]))
            for name in {"history", "quantiles", *matrices} & converted.keys():
                numbers = (v for row in converted[name] for v in row) if name in matrices else converted[name]
                if any(type(v) not in (float, int) for v in numbers):
                    raise ForecastAdapterError(f"{name} must contain numbers, not strings or booleans")
            return cls(**converted)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ForecastAdapterError):
                raise
            raise ForecastAdapterError("invalid forecast request shape or values") from None

    @classmethod
    def from_values(cls, history: list[float], horizon: int, season: int,
                    *, quantiles: tuple[float, ...] = ()) -> "ForecastRequest":
        numeric = tuple(float(value) for value in history)
        if not numeric or any(not math.isfinite(value) for value in numeric):
            raise ForecastAdapterError("history must contain finite observations")
        if type(horizon) is not int or horizon < 1:
            raise ForecastAdapterError("horizon must be a positive integer")
        if type(season) is not int or season < 1:
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
    timestamps: tuple[str, ...] = ()
    series_id: str | None = None
    unit: str | None = None
    sample_paths: tuple[tuple[float, ...], ...] | None = None

    def validate(self, request: ForecastRequest) -> "ForecastResult":
        if len(self.point) != request.horizon:
            raise ForecastAdapterError(
                f"adapter returned {len(self.point)} points for horizon {request.horizon}")
        if any(not math.isfinite(float(value)) for value in self.point):
            raise ForecastAdapterError("adapter returned a non-finite point forecast")
        if request.quantiles and self.quantiles is None:
            raise ForecastAdapterError("adapter returned no requested quantiles")
        if self.quantiles is not None:
            if len(self.quantiles) != request.horizon:
                raise ForecastAdapterError("quantile rows do not match the horizon")
            for row in self.quantiles:
                if any(type(q) not in (float, int) or not 0 < q < 1 for q in row):
                    raise ForecastAdapterError("quantile keys must be numeric probabilities")
                values = []
                for quantile in sorted(set(row) | set(request.quantiles)):
                    if quantile not in row or not math.isfinite(float(row[quantile])):
                        raise ForecastAdapterError(
                            f"quantile row is missing finite q={quantile}")
                    values.append(float(row[quantile]))
                if values != sorted(values):
                    raise ForecastAdapterError("forecast quantiles are not monotone")
        if self.timestamps != request.future_timestamps:
            raise ForecastAdapterError("result timestamps do not match requested future timestamps")
        if self.series_id != request.series_id or self.unit != request.unit:
            raise ForecastAdapterError("result series_id and unit must match the request")
        if request.samples and (self.sample_paths is None or len(self.sample_paths) != request.samples):
            raise ForecastAdapterError("adapter returned the wrong number of sample paths")
        if self.sample_paths is not None:
            if not self.sample_paths or any(len(path) != request.horizon or any(
                    not math.isfinite(float(value)) for value in path) for path in self.sample_paths):
                raise ForecastAdapterError("sample paths must be finite and horizon-aligned")
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
        }, timestamps=request.future_timestamps, series_id=request.series_id,
            unit=request.unit).validate(request)


def validate_capabilities(capabilities: AdapterCapabilities,
                          request: ForecastRequest) -> None:
    """Reject unsupported inputs instead of silently discarding them."""
    unsupported = []
    if request.quantiles and not capabilities.quantiles:
        unsupported.append("quantiles")
    if request.samples and not capabilities.sample_paths:
        unsupported.append("sample_paths")
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


def conformance_report(adapter: ForecastAdapter, *,
                       history: list[float] | None = None,
                       horizon: int = 4, season: int = 1,
                       related_series: list[list[float]] | None = None,
                       require_deterministic: bool = False,
                       ) -> dict[str, Any]:
    """Exercise protocol behavior with three explicit provider calls.

    Repeatability is diagnostic unless required. Varying service request IDs
    are not stochastic numeric output, and stochastic forecasts are not by
    themselves a protocol violation. This is not an accuracy/calibration test.
    """
    if type(require_deterministic) is not bool:
        raise ForecastAdapterError("require_deterministic must be a boolean")
    minimum = int(getattr(getattr(adapter, "capabilities", None),
                          "min_history", 0) or 0)
    values = list(history if history is not None else ([1, 2] * max(6, (minimum + 1) // 2)))
    request = ForecastRequest(
        tuple(float(value) for value in values), horizon, season,
        related_series=tuple(
            tuple(float(value) for value in series)
            for series in (related_series or [])
        ),
    )
    checks: dict[str, bool] = {}
    failures: dict[str, str] = {}
    before = asdict(request)
    capabilities = getattr(adapter, "capabilities", AdapterCapabilities())
    try:
        validate_capabilities(capabilities, request)
        first = adapter.forecast(request).validate(request)
        checks["finite_exact_horizon"] = True
        checks["input_immutable"] = asdict(request) == before
        second = adapter.forecast(request).validate(request)
        checks["input_immutable"] = checks["input_immutable"] and asdict(request) == before
        checks["deterministic_replay"] = replace(first, metadata={}) == replace(second, metadata={})
    except Exception as error:
        failures["forecast"] = type(error).__name__
        checks.setdefault("finite_exact_horizon", False)
        checks.setdefault("input_immutable", asdict(request) == before)
        checks.setdefault("deterministic_replay", False)
    try:
        short = ForecastRequest(
            tuple(float(value) for value in values), 1, season,
            related_series=request.related_series,
        )
        validate_capabilities(capabilities, short)
        short_before = asdict(short)
        checks["variable_horizon"] = len(adapter.forecast(short).validate(short).point) == 1
        checks["input_immutable"] = checks["input_immutable"] and asdict(short) == short_before
    except Exception as error:
        failures["variable_horizon"] = \
            type(error).__name__
        checks["variable_horizon"] = False
    return {"protocol_version": PROTOCOL_VERSION, "adapter": adapter.name,
            "kind": adapter.kind, "checks": checks,
            "capabilities": asdict(capabilities),
            "determinism_required": require_deterministic,
            **({"failures": failures} if failures else {}),
            "conformant": all(value for key, value in checks.items() if key != "deterministic_replay" or require_deterministic)}
