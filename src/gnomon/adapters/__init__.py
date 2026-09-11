"""Thin adapters for third-party forecasting packages, one operator ``kind`` each.

Gnomon's core stays dependency-free. A third-party package is imported only
when a provider of its kind appears in operator TOML, and each kind has a
matching pip extra, for example ``pip install 'gnomon-forecast[statsforecast]'``.
An adapter converts ``ForecastRequest`` into the package's own call and the
package's output back into ``ForecastResult``. It makes no accuracy claim and
does not choose a model for the user: ``model`` is an explicit operator setting.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass

from ._base import Registration


@dataclass(frozen=True)
class AdapterSpec:
    kind: str
    extra: str
    import_names: tuple[str, ...]
    distributions: tuple[str, ...]
    summary: str
    default_model: str
    lifecycle: str  # "fresh_per_request" for fit-per-call packages, "stateless" otherwise


ADAPTERS: dict[str, AdapterSpec] = {
    "statsforecast": AdapterSpec(
        "statsforecast", "statsforecast", ("statsforecast", "pandas"), ("statsforecast",),
        "Nixtla statistical models (AutoARIMA, AutoETS, Theta, MSTL) with prediction intervals.",
        "AutoARIMA", "fresh_per_request"),
    "statsmodels": AdapterSpec(
        "statsmodels", "statsmodels", ("statsmodels",), ("statsmodels",),
        "statsmodels ETS, SARIMAX and Theta reference implementations.",
        "ETS", "fresh_per_request"),
    "prophet": AdapterSpec(
        "prophet", "prophet", ("prophet", "pandas"), ("prophet",),
        "Prophet additive model; requires request timestamps.",
        "Prophet", "fresh_per_request"),
    "mlforecast": AdapterSpec(
        "mlforecast", "mlforecast", ("mlforecast", "pandas"), ("mlforecast",),
        "Nixtla lag-feature regression with conformal prediction intervals.",
        "lightgbm", "fresh_per_request"),
    "skforecast": AdapterSpec(
        "skforecast", "skforecast", ("skforecast", "pandas", "sklearn"), ("skforecast",),
        "scikit-learn regressors as recursive forecasters with bootstrapped quantiles.",
        "sklearn.linear_model:Ridge", "fresh_per_request"),
    "sktime": AdapterSpec(
        "sktime", "sktime", ("sktime", "pandas"), ("sktime",),
        "Any sktime forecaster class path; predict_quantiles where the estimator supports it.",
        "sktime.forecasting.theta:ThetaForecaster", "fresh_per_request"),
    "darts": AdapterSpec(
        "darts", "darts", ("darts", "pandas"), ("darts",),
        "Any darts model class; probabilistic models return sampled quantiles and paths.",
        "ExponentialSmoothing", "fresh_per_request"),
    "gluonts": AdapterSpec(
        "gluonts", "gluonts", ("gluonts", "pandas"), ("gluonts",),
        "GluonTS predictors and estimators; sample forecasts map to quantiles and paths.",
        "gluonts.model.npts:NPTSPredictor", "fresh_per_request"),
    "neuralforecast": AdapterSpec(
        "neuralforecast", "neuralforecast", ("neuralforecast", "torch", "pandas"), ("neuralforecast",),
        "Nixtla neural models (NHITS, NBEATS, PatchTST, TFT) trained per request.",
        "NHITS", "fresh_per_request"),
}

PROVIDER_FIELDS = frozenset({"kind", "model", "options", "revision", "deterministic", "seed"})


def install_command(spec: AdapterSpec) -> str:
    return f"python -m pip install 'gnomon-forecast[{spec.extra}]'"


def is_installed(spec: AdapterSpec) -> bool:
    """Check import availability without importing the package."""
    try:
        return all(importlib.util.find_spec(name) is not None for name in spec.import_names)
    except (ImportError, ValueError):
        return False


def adapter_kinds() -> dict[str, dict]:
    """Describe adapter kinds for configuration discovery; imports nothing heavy."""
    return {kind: {"extra": spec.extra, "install": install_command(spec), "summary": spec.summary,
                   "default_model": spec.default_model, "lifecycle": spec.lifecycle,
                   "installed": is_installed(spec)} for kind, spec in ADAPTERS.items()}


def build_provider(kind: str, name: str, spec: dict) -> Registration:
    """Import the adapter module for ``kind`` and build a registration.

    Raises ModuleNotFoundError when the third-party package is absent so the
    session can report the pip extra; adapter option errors raise
    ForecastAdapterError.
    """
    adapter = ADAPTERS[kind]
    module = importlib.import_module(f"gnomon.adapters.{kind}")
    options = spec.get("options", {})
    if not isinstance(options, dict):
        from ..forecast_adapter import ForecastAdapterError
        raise ForecastAdapterError("options must be a TOML table of model keyword arguments")
    registration = module.build(name=name, model=spec.get("model", adapter.default_model),
                                options=dict(options), seed=spec.get("seed"))
    revision = spec.get("revision", registration.revision)
    deterministic = spec.get("deterministic", registration.deterministic)
    target = registration.target
    if registration.factory:
        inner = target

        def target():
            instance = inner()
            # Marks the instance so the engine lets Gnomon-authored ForecastAdapterError
            # messages (missing timestamps, one-sided covariates, short history) reach the
            # caller instead of the redacted execution failure used for arbitrary user code.
            instance.gnomon_adapter_kind = kind
            return instance
    else:
        target.gnomon_adapter_kind = kind
    return Registration(target, registration.factory, registration.capabilities, revision, deterministic)
