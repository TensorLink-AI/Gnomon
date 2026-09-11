"""statsmodels: ETS, SARIMAX or Theta, fit fresh on every request.

TOML::

    [providers.ets]
    kind = "statsmodels"
    model = "ETS"                 # ETS | SARIMAX | Theta
    [providers.ets.options]       # constructor keyword arguments
    trend = "add"
    seasonal = "add"              # seasonal_periods defaults to the request season

ETS uses exact intervals for additive-error models and simulation otherwise;
set ``seed`` for a reproducible simulated interval. SARIMAX ``order`` and
``seasonal_order`` are operator choices, never inferred.
"""

from __future__ import annotations

import warnings

from ..forecast_adapter import AdapterCapabilities, ForecastAdapterError, ForecastRequest, validate_capabilities
from . import _base

KIND = "statsmodels"
MODELS = ("ETS", "SARIMAX", "Theta")
CAPABILITIES = AdapterCapabilities(quantiles=True, min_history=3)


def build(*, name: str, model: str, options: dict, seed) -> _base.Registration:
    import statsmodels.api  # noqa: F401
    if model not in MODELS:
        raise ForecastAdapterError(f"statsmodels model must be one of {', '.join(MODELS)}")
    seed = _base.check_seed(seed)
    deterministic = model != "ETS" or seed is not None
    return _base.Registration(lambda: _Provider(model, options, seed), True, CAPABILITIES,
                              _base.revision(KIND, "statsmodels"), deterministic)


def _tuples(options: dict) -> dict:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in options.items()}


class _Provider:
    capabilities = CAPABILITIES

    def __init__(self, model, options, seed):
        self._model, self._options, self._seed, self._used = model, _tuples(options), seed, False

    def forecast(self, request: ForecastRequest):
        if self._used:
            raise RuntimeError("statsmodels fitting object must not be reused")
        self._used = True
        validate_capabilities(self.capabilities, request)
        import numpy as np
        y, h = np.asarray(request.history, dtype=float), request.horizon
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            point, interval = getattr(self, "_" + self._model.lower())(y, h, request)

        def at(q):
            if q == 0.5:
                return point
            lower, upper = interval(1 - _base.level_for(q) / 100)
            return lower if q < 0.5 else upper

        rows, fixed = _base.make_rows(request, at)
        return _base.finish(request, point, kind=KIND, quantiles=rows, fixed=fixed, model=self._model,
                            index="integer_index_statsmodels_ignores_timestamps",
                            median_quantile_is_point=bool(request.quantiles and 0.5 in request.quantiles))

    def _ets(self, y, h, request):
        import pandas as pd
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        kwargs = {"error": "add", **self._options}
        if kwargs.get("seasonal") and "seasonal_periods" not in kwargs:
            kwargs["seasonal_periods"] = request.season
        fitted = ETSModel(pd.Series(y, index=pd.RangeIndex(len(y))), **kwargs).fit(disp=False)
        simulate = {"random_state": self._seed} if self._seed is not None else {}
        prediction = fitted.get_prediction(start=len(y), end=len(y) + h - 1, **simulate)
        point = np.asarray(prediction.predicted_mean, dtype=float)

        def interval(alpha):
            frame = prediction.summary_frame(alpha=alpha)
            return frame["pi_lower"].to_numpy(), frame["pi_upper"].to_numpy()
        return point, interval

    def _sarimax(self, y, h, request):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        kwargs = {"order": (1, 1, 1), **self._options}
        fitted = SARIMAX(y, **kwargs).fit(disp=False)
        forecast = fitted.get_forecast(h)
        point = np.asarray(forecast.predicted_mean, dtype=float)

        def interval(alpha):
            bounds = np.asarray(forecast.conf_int(alpha=alpha), dtype=float)
            return bounds[:, 0], bounds[:, 1]
        return point, interval

    def _theta(self, y, h, request):
        from statsmodels.tsa.forecasting.theta import ThetaModel
        seasonal = request.season > 1
        kwargs = {"period": request.season if seasonal else None, "deseasonalize": seasonal, **self._options}
        fitted = ThetaModel(y, **kwargs).fit()
        point = np.asarray(fitted.forecast(h), dtype=float)

        def interval(alpha):
            bounds = fitted.prediction_intervals(h, alpha=alpha)
            return bounds["lower"].to_numpy(), bounds["upper"].to_numpy()
        return point, interval


import numpy as np  # noqa: E402  imported after the class for the lazy-import pattern above
