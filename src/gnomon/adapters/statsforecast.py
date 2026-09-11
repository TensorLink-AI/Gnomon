"""statsforecast: one Nixtla statistical model, fit fresh on every request.

TOML::

    [providers.arima]
    kind = "statsforecast"
    model = "AutoARIMA"          # any class in statsforecast.models
    [providers.arima.options]    # keyword arguments for that class
    # season_length defaults to the request season when the class accepts it

Quantiles come from statsforecast prediction-interval levels; the 0.5 quantile
is the model's point forecast. Regressors need paired past/future covariates.
"""

from __future__ import annotations

from ..forecast_adapter import AdapterCapabilities, ForecastAdapterError, ForecastRequest, validate_capabilities
from . import _base

KIND = "statsforecast"
CAPABILITIES = AdapterCapabilities(quantiles=True, past_covariates=True, future_covariates=True, min_history=2)


def build(*, name: str, model: str, options: dict, seed) -> _base.Registration:
    import statsforecast  # noqa: F401  fail at configuration time, not first forecast
    cls = _base.resolve(model, default_module="statsforecast.models")
    _base.check_seed(seed)
    return _base.Registration(lambda: _Provider(cls, options), True, CAPABILITIES,
                              _base.revision(KIND, "statsforecast"), True)


class _Provider:
    capabilities = CAPABILITIES

    def __init__(self, cls, options):
        self._cls, self._options, self._used = cls, options, False

    def forecast(self, request: ForecastRequest):
        if self._used:
            raise RuntimeError("statsforecast fitting object must not be reused")
        self._used = True
        validate_capabilities(self.capabilities, request)
        import pandas as pd
        from statsforecast import StatsForecast
        index, freq, note = _base.history_index(request)
        if isinstance(index, pd.DatetimeIndex):
            if freq is None:
                raise ForecastAdapterError("statsforecast needs frequency with dated history",
                                           details={"missing_fields": ["frequency"]})
            ds, sf_freq = _base.naive_utc(index), freq
        else:
            ds, sf_freq = index, 1
        kwargs = dict(self._options)
        if "season_length" not in kwargs and _base.accepts(self._cls, "season_length"):
            kwargs["season_length"] = request.season
        model = self._cls(**kwargs)
        alias = getattr(model, "alias", type(model).__name__)
        frame = pd.DataFrame({"unique_id": "y", "ds": ds, "y": list(request.history)})
        hist, fut, names = _base.paired_covariates(request)
        x_df = None
        if hist is not None:
            for column, name in enumerate(names):
                frame[name] = hist[:, column]
            future = _base.naive_utc(_base.future_index(request, index, freq))
            x_df = pd.DataFrame({"unique_id": "y", "ds": future,
                                 **{name: fut[:, column] for column, name in enumerate(names)}})
        forecaster = StatsForecast(models=[model], freq=sf_freq, n_jobs=1)
        levels = _base.levels(request.quantiles)
        result = forecaster.forecast(df=frame, h=request.horizon, level=levels or None, X_df=x_df)
        point = result[alias].to_numpy()

        def at(q):
            if q == 0.5:
                return point
            side = "lo" if q < 0.5 else "hi"
            return result[f"{alias}-{side}-{_base.level_for(q)}"].to_numpy()

        rows, fixed = _base.make_rows(request, at)
        return _base.finish(request, point, kind=KIND, quantiles=rows, fixed=fixed, model=alias,
                            index=note, median_quantile_is_point=bool(request.quantiles and 0.5 in request.quantiles))
