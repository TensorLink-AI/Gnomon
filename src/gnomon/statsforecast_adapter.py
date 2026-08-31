"""Optional StatsForecast candidates behind Gnomon's evidence boundary.

The third-party package supplies point estimators, not publication authority.
Every adapter returned here enters the same prefix-only rolling contest as the
dependency-free baselines. Imports are lazy so the base installation remains
zero-dependency and a broken optional install is a disclosed soft skip.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import math
import warnings
from typing import Any, Callable

from .forecast_adapter import AdapterCapabilities


SUPPORTED_VERSION = ">=2.1,<3"
DEFAULT_CANDIDATES = (
    "statsforecast_autoets",
    "statsforecast_autoarima",
    "statsforecast_autotheta",
    "statsforecast_croston_optimized",
)
OUTER_CANDIDATE = "statsforecast_auto"
MAX_OUTER_FOLDS = 16
MAX_FIT_HISTORY = 256
MAX_SEASON_LENGTH = 64


class StatsForecastUnavailable(RuntimeError):
    """The optional package is absent or cannot provide its model API."""


def installed_version() -> str | None:
    try:
        return version("statsforecast")
    except PackageNotFoundError:
        return None


def _major_minor_eligible(revision: str) -> bool:
    try:
        major, minor, *_ = revision.split(".")
        return int(major) == 2 and int(minor) >= 1
    except (ValueError, TypeError):
        return False


def installation_status() -> dict[str, Any]:
    revision = installed_version()
    if revision is None:
        return {
            "installed": False,
            "compatible": False,
            "required": SUPPORTED_VERSION,
            "models": list(DEFAULT_CANDIDATES),
            "outer_candidate": OUTER_CANDIDATE,
        }
    compatible = _major_minor_eligible(revision)
    return {
        "installed": True,
        "compatible": compatible,
        "version": revision,
        "required": SUPPORTED_VERSION,
        "models": list(DEFAULT_CANDIDATES),
        "outer_candidate": OUTER_CANDIDATE,
    }


@dataclass(frozen=True)
class _ModelDefinition:
    class_name: str
    min_history: int
    seasonal: bool = True
    intermittent: bool = False


_DEFINITIONS = {
    "statsforecast_autoets": _ModelDefinition("AutoETS", 8),
    "statsforecast_autoarima": _ModelDefinition("AutoARIMA", 10),
    "statsforecast_autotheta": _ModelDefinition("AutoTheta", 8),
    "statsforecast_croston_optimized": _ModelDefinition(
        "CrostonOptimized", 8, seasonal=False, intermittent=True),
}


class StatsForecastAdapter:
    """A single preregistered StatsForecast family representative."""

    kind = "statistical_plugin"
    backend = "in_process"
    supports_quantiles = False
    supports_past_covariates = False
    supports_future_covariates = False
    supports_panel = False
    supports_sample_paths = False

    def __init__(self, name: str, revision: str):
        if name not in _DEFINITIONS:
            raise ValueError(f"unknown StatsForecast candidate: {name}")
        self.name = name
        self.revision = revision
        self._definition = _DEFINITIONS[name]
        self.model_class = self._definition.class_name
        self.min_history = self._definition.min_history
        self.max_horizon = None
        self.capabilities = AdapterCapabilities(min_history=self.min_history)
        self.last_warnings: list[str] = []

    def _model(self, season: int) -> Any:
        try:
            models = import_module("statsforecast.models")
            model_class: Callable[..., Any] = getattr(
                models, self._definition.class_name)
        except (ImportError, AttributeError) as exc:
            raise StatsForecastUnavailable(
                "StatsForecast model API is unavailable"
            ) from exc
        if not self._definition.seasonal:
            return model_class()
        return model_class(season_length=max(1, int(season)))

    def _check_domain(self, history: list[float]) -> None:
        if len(history) < self.min_history:
            raise ValueError(
                f"{self.name} requires at least {self.min_history} observations")
        if not self._definition.intermittent:
            return
        if any(value < 0 for value in history):
            raise ValueError("CrostonOptimized requires non-negative demand")
        positive = sum(value > 0 for value in history)
        zeros = sum(value == 0 for value in history)
        if positive < 2 or zeros / len(history) < 0.20:
            raise ValueError(
                "CrostonOptimized is eligible only for visibly intermittent demand")

    def predict(self, history: list[float], horizon: int,
                season: int) -> list[float]:
        values = [float(value) for value in history]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("history must contain finite observations")
        self._check_domain(values)
        if horizon < 1:
            raise ValueError("horizon must be positive")
        try:
            numpy = import_module("numpy")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                raw = self._model(season).forecast(
                    y=numpy.asarray(values, dtype=numpy.float64),
                    h=int(horizon))
            self.last_warnings = list(dict.fromkeys(
                f"{item.category.__name__}: {item.message}"[:300]
                for item in caught))
            points = raw.get("mean") if isinstance(raw, dict) else None
            result = [float(value) for value in points]
        except StatsForecastUnavailable:
            raise
        except Exception as exc:
            raise ValueError(
                f"{self.name} failed to fit the visible prefix: "
                f"{type(exc).__name__}: {exc}") from exc
        if len(result) != horizon or any(not math.isfinite(value)
                                         for value in result):
            raise ValueError(
                f"{self.name} returned a non-finite or misaligned forecast")
        return result


class StatsForecastPortfolioAdapter:
    """One outer candidate with a training-internal component choice.

    Gnomon's outer folds compare one StatsForecast family representative with
    the controls. Component search therefore cannot multiply the number of
    outer admissions. The trailing validation slice is entirely inside the
    prefix supplied for the current outer origin.
    """

    name = OUTER_CANDIDATE
    kind = "statistical_plugin"
    backend = "in_process"
    model_class = "TrainingInternalStatsForecastPortfolio"
    min_history = 12
    max_horizon = None
    fit_history_limit = MAX_FIT_HISTORY
    capabilities = AdapterCapabilities(min_history=min_history)
    supports_quantiles = False
    supports_past_covariates = False
    supports_future_covariates = False
    supports_panel = False
    supports_sample_paths = False

    def __init__(self, components: list[str], revision: str):
        if not components:
            raise ValueError("StatsForecast portfolio needs a component")
        self.components = tuple(components)
        self.revision = revision
        self.selection_trace: list[dict[str, Any]] = []
        self._last_component_warnings: dict[str, list[str]] = {}

    def _select(self, history: list[float], horizon: int,
                season: int) -> tuple[StatsForecastAdapter, dict[str, float]]:
        holdout = max(3, min(int(horizon), len(history) // 4, 12))
        train, actual = history[:-holdout], history[-holdout:]
        scores: dict[str, float] = {}
        component_warnings: dict[str, list[str]] = {}
        adapters: dict[str, StatsForecastAdapter] = {}
        for name in self.components:
            adapter = StatsForecastAdapter(name, self.revision)
            try:
                predicted = adapter.predict(train, holdout, season)
            except (ValueError, StatsForecastUnavailable):
                continue
            loss = sum(abs(observed - estimated)
                       for observed, estimated in zip(actual, predicted)) \
                / holdout
            if math.isfinite(loss):
                scores[name] = loss
                adapters[name] = adapter
                if adapter.last_warnings:
                    component_warnings[name] = list(adapter.last_warnings)
        if not scores:
            raise ValueError(
                "no StatsForecast component completed internal validation")
        selected = min(scores, key=lambda name: (scores[name], name))
        self._last_component_warnings = component_warnings
        return adapters[selected], scores

    def predict(self, history: list[float], horizon: int,
                season: int) -> list[float]:
        visible_length = len(history)
        if season > MAX_SEASON_LENGTH:
            raise ValueError(
                f"{self.name} soft-skipped: season_length {season} exceeds "
                f"the interactive bound {MAX_SEASON_LENGTH}")
        values = [float(value) for value in history[-MAX_FIT_HISTORY:]]
        if len(values) < self.min_history:
            raise ValueError(
                f"{self.name} requires at least {self.min_history} observations")
        selected, scores = self._select(values, horizon, season)
        trace = {
            "history_length": len(values),
            "visible_history_length": visible_length,
            "fit_history_limit": self.fit_history_limit,
            "horizon": int(horizon),
            "season": int(season),
            "selected_component": selected.name,
            "component_validation_mae": dict(sorted(scores.items())),
            "future_observations_used": 0,
            **({"component_warnings": self._last_component_warnings}
               if self._last_component_warnings else {}),
        }
        points = selected.predict(values, horizon, season)
        if selected.last_warnings:
            trace["selected_fit_warnings"] = list(selected.last_warnings)
        self.selection_trace.append(trace)
        return points


def statsforecast_candidates(
    requested: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[StatsForecastAdapter], dict[str, Any]]:
    """Return compatible adapters plus a machine-readable discovery receipt."""
    names = list(DEFAULT_CANDIDATES if requested is None else requested)
    if names == [OUTER_CANDIDATE]:
        names = list(DEFAULT_CANDIDATES)
    unknown = sorted(set(names) - set(DEFAULT_CANDIDATES))
    if unknown:
        raise ValueError(
            "unknown StatsForecast candidates: " + ", ".join(unknown))
    status = installation_status()
    status["requested"] = list(names)
    if not status["installed"]:
        status["status"] = "soft_skip_missing_dependency"
        return [], status
    if not status["compatible"]:
        status["status"] = "soft_skip_incompatible_version"
        return [], status
    revision = str(status["version"])
    try:
        import_module("statsforecast.models")
    except Exception as exc:
        status["status"] = "soft_skip_broken_import"
        status["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return [], status
    status["status"] = "available"
    status["outer_candidate"] = OUTER_CANDIDATE
    status["components"] = list(names)
    return [StatsForecastPortfolioAdapter(names, revision)], status
