"""skforecast: a scikit-learn regressor as a recursive forecaster, fit fresh per request.

TOML::

    [providers.ridge]
    kind = "skforecast"
    model = "sklearn.linear_model:Ridge"   # any scikit-learn compatible regressor
    seed = 7
    [providers.ridge.options]
    lags = 7                                # int or list; default: season, or 3
    n_boot = 250                            # bootstrap draws for quantiles
    [providers.ridge.options.regressor]
    alpha = 1.0

Quantiles are bootstrapped from in-sample residuals, so they are stochastic
unless ``seed`` is set.
"""

from __future__ import annotations

from ..forecast_adapter import AdapterCapabilities, ForecastAdapterError, ForecastRequest, validate_capabilities
from . import _base

KIND = "skforecast"
CAPABILITIES = AdapterCapabilities(quantiles=True, past_covariates=True, future_covariates=True, min_history=4)


def build(*, name: str, model: str, options: dict, seed) -> _base.Registration:
    import skforecast  # noqa: F401
    seed = _base.check_seed(seed)
    cls = _base.resolve(model, label="regressor")
    lags = options.get("lags")
    if lags is not None and not (type(lags) is int and lags > 0
                                 or isinstance(lags, list) and lags and all(type(v) is int and v > 0 for v in lags)):
        raise ForecastAdapterError("lags must be a positive integer or a list of positive integers")
    n_boot = options.get("n_boot", 250)
    if type(n_boot) is not int or n_boot < 1:
        raise ForecastAdapterError("n_boot must be a positive integer")
    regressor = dict(options.get("regressor", {}))
    if seed is not None and _base.accepts(cls, "random_state"):
        regressor.setdefault("random_state", seed)
    return _base.Registration(lambda: _Provider(cls, regressor, lags, n_boot, seed), True, CAPABILITIES,
                              _base.revision(KIND, "skforecast"), seed is not None)


class _Provider:
    capabilities = CAPABILITIES

    def __init__(self, cls, regressor, lags, n_boot, seed):
        self._cls, self._regressor, self._lags, self._n_boot, self._seed = cls, regressor, lags, n_boot, seed
        self._used = False

    def forecast(self, request: ForecastRequest):
        if self._used:
            raise RuntimeError("skforecast fitting object must not be reused")
        self._used = True
        validate_capabilities(self.capabilities, request)
        import numpy as np
        import pandas as pd
        from skforecast.recursive import ForecasterRecursive
        h = request.horizon
        lags = self._lags if self._lags is not None else (request.season if request.season > 1 else 3)
        longest = lags if type(lags) is int else max(lags)
        if len(request.history) <= longest:
            raise ForecastAdapterError(
                f"skforecast needs more than {longest} observations for these lags; observed={len(request.history)}",
                details={"required_history": longest + 1, "observed_history": len(request.history)})
        index, freq, note = _base.history_index(request)
        if isinstance(index, pd.DatetimeIndex):
            if freq is None:
                raise ForecastAdapterError("skforecast needs frequency with dated history",
                                           details={"missing_fields": ["frequency"]})
            y_index = pd.DatetimeIndex(_base.naive_utc(index), freq=freq)
            future = pd.DatetimeIndex(_base.naive_utc(_base.future_index(request, index, freq)), freq=freq)
        else:
            y_index, future = index, _base.future_index(request, index, freq)
        y = pd.Series(np.asarray(request.history, dtype=float), index=y_index)
        hist, fut, names = _base.paired_covariates(request)
        exog = pd.DataFrame(hist, index=y_index, columns=list(names)) if hist is not None else None
        exog_future = pd.DataFrame(fut, index=future, columns=list(names)) if fut is not None else None
        estimator = self._cls(**self._regressor)
        key = "estimator" if _base.accepts(ForecasterRecursive, "estimator") else "regressor"  # renamed in 0.17
        forecaster = ForecasterRecursive(**{key: estimator}, lags=lags)
        forecaster.fit(y=y, exog=exog, store_in_sample_residuals=bool(request.quantiles))
        point = forecaster.predict(steps=h, exog=exog_future).to_numpy()
        rows, fixed = None, 0
        if request.quantiles:
            kwargs = {"steps": h, "quantiles": list(request.quantiles), "n_boot": self._n_boot, "exog": exog_future,
                      "use_in_sample_residuals": True, "random_state": self._seed if self._seed is not None else 123}
            accepted = {k: v for k, v in kwargs.items() if _base.accepts(forecaster.predict_quantiles, k)}
            table = forecaster.predict_quantiles(**accepted)
            columns = [table.iloc[:, i].to_numpy() for i in range(len(request.quantiles))]
            rows, fixed = _base.make_rows(request, lambda q: columns[request.quantiles.index(q)])
        return _base.finish(request, point, kind=KIND, quantiles=rows, fixed=fixed, model=self._cls.__name__,
                            lags=lags, index=note, interval_method="bootstrap" if request.quantiles else None,
                            seed=self._seed)
