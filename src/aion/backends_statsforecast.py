"""First-party statsforecast model backend (the ``statistical`` extra).

Contributes Nixtla's optimized classical models — AutoETS, AutoARIMA, and
AutoTheta — as plugin candidates named ``statsforecast:auto_ets`` etc.
They compete on the same rolling folds against the same mandatory
baselines as every other candidate; installing the extra
(``pip install "aion-forecast[statistical]"``) is the only step, and
uninstalling it removes the candidates.

The entry-point factory returns ``None`` when statsforecast is not
importable, so the core package never gains a hard dependency. First call
pays numba's JIT warm-up; that latency is the backend's, never the
harness's.
"""

from __future__ import annotations

from .backends import ModelBackend, ModelFn


def _model_fn(constructor_name: str) -> ModelFn:
    def run(history: list[float], horizon: int, season: int) -> list[float]:
        import numpy as np
        from statsforecast import models as sf_models

        constructor = getattr(sf_models, constructor_name)
        model = constructor(season_length=max(1, season))
        fitted = model.forecast(
            y=np.asarray(history, dtype=float), h=horizon,
        )
        mean = fitted["mean"]
        if len(mean) != horizon:
            raise ValueError(
                f"{constructor_name} returned {len(mean)} values for horizon {horizon}"
            )
        return [float(value) for value in mean]

    return run


def backend() -> ModelBackend | None:
    """Entry-point factory: the backend, or None when the extra is absent."""
    try:
        import statsforecast  # noqa: F401  (availability probe)
        from importlib.metadata import version
    except Exception:
        return None
    return ModelBackend(
        name="statsforecast",
        version=version("statsforecast"),
        models={
            "auto_ets": _model_fn("AutoETS"),
            "auto_arima": _model_fn("AutoARIMA"),
            "auto_theta": _model_fn("AutoTheta"),
        },
    )
