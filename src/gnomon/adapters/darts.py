"""darts: any model class in darts.models, fit fresh on every request.

TOML::

    [providers.ets]
    kind = "darts"
    model = "ExponentialSmoothing"     # class in darts.models, or module:Class
    seed = 7
    [providers.ets.options]            # constructor keyword arguments
    num_samples = 200                  # draws for quantiles and sample paths
    # seasonal_periods defaults to the request season when the class accepts it

Probabilistic darts models return sampled forecasts, which supply quantiles
and sample paths. Deterministic models reject quantile requests instead of
inventing an interval. Deep-learning models need the ``darts[torch]`` extra.
"""

from __future__ import annotations

from ..forecast_adapter import AdapterCapabilities, ForecastAdapterError, ForecastRequest, validate_capabilities
from . import _base

KIND = "darts"
CAPABILITIES = AdapterCapabilities(quantiles=True, sample_paths=True, min_history=2)


def build(*, name: str, model: str, options: dict, seed) -> _base.Registration:
    import darts  # noqa: F401
    cls = _base.resolve(model, default_module="darts.models")
    seed = _base.check_seed(seed)
    kwargs = dict(options)
    num_samples = kwargs.pop("num_samples", 200)
    if type(num_samples) is not int or num_samples < 1:
        raise ForecastAdapterError("num_samples must be a positive integer")
    if seed is not None and _base.accepts(cls, "random_state"):
        kwargs.setdefault("random_state", seed)
    return _base.Registration(lambda: _Provider(cls, kwargs, num_samples, seed), True, CAPABILITIES,
                              _base.revision(KIND, "darts"), seed is not None)


class _Provider:
    capabilities = CAPABILITIES

    def __init__(self, cls, options, num_samples, seed):
        self._cls, self._options, self._num_samples, self._seed, self._used = cls, options, num_samples, seed, False

    def forecast(self, request: ForecastRequest):
        if self._used:
            raise RuntimeError("darts fitting object must not be reused")
        self._used = True
        validate_capabilities(self.capabilities, request)
        import numpy as np
        import pandas as pd
        from darts import TimeSeries
        values = np.asarray(request.history, dtype=float)
        index, freq, note = _base.history_index(request)
        if isinstance(index, pd.DatetimeIndex) and freq is not None:
            series = TimeSeries.from_times_and_values(pd.DatetimeIndex(_base.naive_utc(index), freq=freq), values)
        else:
            series, note = TimeSeries.from_values(values), "integer_index_darts_adapter"
        kwargs = dict(self._options)
        if "seasonal_periods" not in kwargs and request.season > 1 and _base.accepts(self._cls, "seasonal_periods"):
            kwargs["seasonal_periods"] = request.season
        model = self._cls(**kwargs)
        probabilistic = bool(getattr(model, "supports_probabilistic_prediction", getattr(model, "_is_probabilistic", False)))
        wants_samples = bool(request.quantiles or request.samples)
        if wants_samples and not probabilistic:
            raise ForecastAdapterError(f"{self._cls.__name__} is not probabilistic in darts; "
                                       "quantiles and sample paths need a probabilistic model")
        _base.seed_numpy(self._seed)
        model.fit(series)
        draws = max(self._num_samples, request.samples) if wants_samples else 1
        prediction = model.predict(request.horizon, num_samples=draws)
        array = np.asarray(prediction.all_values(copy=False), dtype=float)[:, 0, :].T  # (samples, horizon)
        rows, fixed, paths = None, 0, None
        if wants_samples:
            point = array.mean(axis=0)
            rows, fixed = _base.rows_from_samples(request, array)
            paths = _base.take_paths(request, array)
        else:
            point = array[0]
        return _base.finish(request, point, kind=KIND, quantiles=rows, fixed=fixed, sample_paths=paths,
                            model=self._cls.__name__, index=note, num_samples=draws if wants_samples else None,
                            seed=self._seed)
