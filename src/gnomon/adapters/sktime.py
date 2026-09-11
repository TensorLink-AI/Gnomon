"""sktime: any forecaster class path, fit fresh on every request.

TOML::

    [providers.theta]
    kind = "sktime"
    model = "sktime.forecasting.theta:ThetaForecaster"   # module:Class
    [providers.theta.options]                            # constructor keyword arguments
    # sp defaults to the request season when the class accepts it

The series is modelled on an integer index; timestamps are echoed but not
used for calendar features. Quantiles use ``predict_quantiles`` and are
rejected up front when the estimator lacks the ``capability:pred_int`` tag.
"""

from __future__ import annotations

from ..forecast_adapter import AdapterCapabilities, ForecastAdapterError, ForecastRequest, validate_capabilities
from . import _base

KIND = "sktime"
CAPABILITIES = AdapterCapabilities(quantiles=True, past_covariates=True, future_covariates=True, min_history=3)


def build(*, name: str, model: str, options: dict, seed) -> _base.Registration:
    import sktime  # noqa: F401
    cls = _base.resolve(model, label="forecaster")
    seed = _base.check_seed(seed)
    kwargs = dict(options)
    if seed is not None and _base.accepts(cls, "random_state"):
        kwargs.setdefault("random_state", seed)
    return _base.Registration(lambda: _Provider(cls, kwargs), True, CAPABILITIES,
                              _base.revision(KIND, "sktime"), False)


class _Provider:
    capabilities = CAPABILITIES

    def __init__(self, cls, options):
        self._cls, self._options, self._used = cls, options, False

    def forecast(self, request: ForecastRequest):
        if self._used:
            raise RuntimeError("sktime fitting object must not be reused")
        self._used = True
        validate_capabilities(self.capabilities, request)
        import numpy as np
        import pandas as pd
        from sktime.forecasting.base import ForecastingHorizon
        n, h = len(request.history), request.horizon
        kwargs = dict(self._options)
        if "sp" not in kwargs and _base.accepts(self._cls, "sp"):
            kwargs["sp"] = request.season
        forecaster = self._cls(**kwargs)
        if request.quantiles and not forecaster.get_tag("capability:pred_int", False):
            raise ForecastAdapterError(f"{self._cls.__name__} does not support predict_quantiles; "
                                       "choose an sktime forecaster with the capability:pred_int tag")
        y = pd.Series(np.asarray(request.history, dtype=float), index=pd.RangeIndex(n))
        fh = ForecastingHorizon(np.arange(1, h + 1), is_relative=True)
        hist, fut, names = _base.paired_covariates(request)
        x = pd.DataFrame(hist, index=pd.RangeIndex(n), columns=list(names)) if hist is not None else None
        x_future = pd.DataFrame(fut, index=pd.RangeIndex(n, n + h), columns=list(names)) if fut is not None else None
        forecaster.fit(y, X=x, fh=fh)
        point = np.asarray(forecaster.predict(fh, X=x_future), dtype=float).reshape(-1)
        rows, fixed = None, 0
        if request.quantiles:
            table = forecaster.predict_quantiles(fh, X=x_future, alpha=list(request.quantiles))
            alphas = [float(column[-1]) for column in table.columns]

            def at(q):
                position = min(range(len(alphas)), key=lambda i: abs(alphas[i] - q))
                if abs(alphas[position] - q) > 1e-9:
                    raise ForecastAdapterError(f"sktime returned no quantile column for q={q}")
                return table.iloc[:, position].to_numpy()
            rows, fixed = _base.make_rows(request, at)
        return _base.finish(request, point, kind=KIND, quantiles=rows, fixed=fixed, model=self._cls.__name__,
                            index="integer_index_sktime_adapter")
