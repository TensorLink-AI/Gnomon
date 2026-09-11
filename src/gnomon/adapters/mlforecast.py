"""mlforecast: lag-feature regression with conformal intervals, fit fresh per request.

TOML::

    [providers.boost]
    kind = "mlforecast"
    model = "lightgbm"                 # or module:Class of any scikit-learn regressor
    seed = 7
    [providers.boost.options]
    lags = [1, 2, 3, 7]                # default: 1..season, or 1..3 when season is 1
    n_windows = 2                      # conformal windows used for quantiles
    [providers.boost.options.regressor]
    n_estimators = 200                 # keyword arguments for the regressor

Quantiles need enough history for ``n_windows`` backtest windows of the
requested horizon on top of the longest lag; shorter history is rejected with
the required length rather than padded.
"""

from __future__ import annotations

from ..forecast_adapter import AdapterCapabilities, ForecastAdapterError, ForecastRequest, validate_capabilities
from . import _base

KIND = "mlforecast"
CAPABILITIES = AdapterCapabilities(quantiles=True, past_covariates=True, future_covariates=True, min_history=4)


def build(*, name: str, model: str, options: dict, seed) -> _base.Registration:
    import mlforecast  # noqa: F401
    seed = _base.check_seed(seed)
    if model == "lightgbm":
        from lightgbm import LGBMRegressor
        cls = LGBMRegressor
    else:
        cls = _base.resolve(model, label="regressor")
    lags = options.get("lags")
    if lags is not None and (not isinstance(lags, list) or not lags or any(type(v) is not int or v < 1 for v in lags)):
        raise ForecastAdapterError("lags must be a nonempty list of positive integers")
    n_windows = options.get("n_windows", 2)
    if type(n_windows) is not int or n_windows < 1:
        raise ForecastAdapterError("n_windows must be a positive integer")
    regressor = dict(options.get("regressor", {}))
    if cls.__name__ == "LGBMRegressor":
        regressor.setdefault("verbosity", -1)
    if seed is not None and _base.accepts(cls, "random_state"):
        regressor.setdefault("random_state", seed)
    return _base.Registration(lambda: _Provider(cls, regressor, lags, n_windows), True, CAPABILITIES,
                              _base.revision(KIND, "mlforecast"), seed is not None)


class _Provider:
    capabilities = CAPABILITIES

    def __init__(self, cls, regressor, lags, n_windows):
        self._cls, self._regressor, self._lags, self._n_windows, self._used = cls, regressor, lags, n_windows, False

    def forecast(self, request: ForecastRequest):
        if self._used:
            raise RuntimeError("mlforecast fitting object must not be reused")
        self._used = True
        validate_capabilities(self.capabilities, request)
        import pandas as pd
        from mlforecast import MLForecast
        from mlforecast.utils import PredictionIntervals
        h = request.horizon
        lags = self._lags or list(range(1, (request.season if request.season > 1 else 3) + 1))
        required = max(lags) + 1 + (self._n_windows * h if request.quantiles else 0)
        if len(request.history) < required:
            raise ForecastAdapterError(
                f"mlforecast needs at least {required} observations for these lags"
                + (" and conformal windows" if request.quantiles else "") + f"; observed={len(request.history)}",
                details={"required_history": required, "observed_history": len(request.history), "lags": lags})
        index, freq, note = _base.history_index(request)
        if isinstance(index, pd.DatetimeIndex):
            if freq is None:
                raise ForecastAdapterError("mlforecast needs frequency with dated history",
                                           details={"missing_fields": ["frequency"]})
            ds, ml_freq = _base.naive_utc(index), freq
        else:
            ds, ml_freq = index, 1
        frame = pd.DataFrame({"unique_id": "y", "ds": ds, "y": list(request.history)})
        hist, fut, names = _base.paired_covariates(request)
        x_df = None
        if hist is not None:
            for column, name in enumerate(names):
                frame[name] = hist[:, column]
            future = _base.naive_utc(_base.future_index(request, index, freq))
            x_df = pd.DataFrame({"unique_id": "y", "ds": future,
                                 **{name: fut[:, column] for column, name in enumerate(names)}})
        forecaster = MLForecast(models={"m": self._cls(**self._regressor)}, freq=ml_freq, lags=lags)
        intervals = PredictionIntervals(n_windows=self._n_windows, h=h) if request.quantiles else None
        forecaster.fit(frame, static_features=[], prediction_intervals=intervals)
        levels = _base.levels(request.quantiles)
        result = forecaster.predict(h=h, level=levels or None, X_df=x_df)
        point = result["m"].to_numpy()

        def at(q):
            if q == 0.5:
                return point
            return result[f"m-{'lo' if q < 0.5 else 'hi'}-{_base.level_for(q)}"].to_numpy()

        rows, fixed = _base.make_rows(request, at)
        return _base.finish(request, point, kind=KIND, quantiles=rows, fixed=fixed, model=self._cls.__name__,
                            lags=lags, index=note, interval_method="conformal" if request.quantiles else None,
                            median_quantile_is_point=bool(request.quantiles and 0.5 in request.quantiles))
