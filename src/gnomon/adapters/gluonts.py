"""GluonTS: a predictor used directly, or an estimator trained on the request.

TOML::

    [providers.npts]
    kind = "gluonts"
    model = "gluonts.model.npts:NPTSPredictor"    # module:Class of a Predictor or Estimator
    seed = 7
    [providers.npts.options]
    num_samples = 100                             # draws per forecast
    # other keys are constructor keyword arguments; prediction_length and freq
    # are filled from the request when the class accepts them

GluonTS needs a calendar start and frequency. Dated requests use their own;
undated requests get a synthetic daily index that is disclosed in metadata
and affects only models with calendar features. Estimators such as
``gluonts.torch:DeepAREstimator`` need the ``gluonts[torch]`` extra and are
trained on every request, so keep ``trainer_kwargs`` small.
"""

from __future__ import annotations

from ..forecast_adapter import AdapterCapabilities, ForecastAdapterError, ForecastRequest, validate_capabilities
from . import _base

KIND = "gluonts"
CAPABILITIES = AdapterCapabilities(quantiles=True, sample_paths=True, min_history=2)


def build(*, name: str, model: str, options: dict, seed) -> _base.Registration:
    import gluonts  # noqa: F401
    cls = _base.resolve(model, label="model")
    seed = _base.check_seed(seed)
    kwargs = dict(options)
    num_samples = kwargs.pop("num_samples", 100)
    if type(num_samples) is not int or num_samples < 1:
        raise ForecastAdapterError("num_samples must be a positive integer")
    return _base.Registration(lambda: _Provider(cls, kwargs, num_samples, seed), True, CAPABILITIES,
                              _base.revision(KIND, "gluonts"), seed is not None)


class _Provider:
    capabilities = CAPABILITIES

    def __init__(self, cls, options, num_samples, seed):
        self._cls, self._options, self._num_samples, self._seed, self._used = cls, options, num_samples, seed, False

    def forecast(self, request: ForecastRequest):
        if self._used:
            raise RuntimeError("gluonts fitting object must not be reused")
        self._used = True
        validate_capabilities(self.capabilities, request)
        import numpy as np
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        index, freq, note = _base.history_index(request)
        if isinstance(index, pd.DatetimeIndex):
            if freq is None:
                raise ForecastAdapterError("gluonts needs frequency with dated history",
                                           details={"missing_fields": ["frequency"]})
            start = pd.Period(_base.naive_utc(index)[0], freq=freq)
        else:
            freq, note = "D", "synthetic_daily_index_no_timestamps_supplied"
            start = pd.Period("2000-01-01", freq=freq)
        dataset = ListDataset([{"start": start, "target": np.asarray(request.history, dtype=float)}], freq=freq)
        kwargs = dict(self._options)
        if _base.accepts(self._cls, "prediction_length"):
            kwargs.setdefault("prediction_length", request.horizon)
        if _base.accepts(self._cls, "freq"):
            kwargs.setdefault("freq", freq)
        instance = self._cls(**kwargs)
        _base.seed_numpy(self._seed)
        if self._seed is not None:
            try:
                import torch
                torch.manual_seed(self._seed)
            except ImportError:
                pass
        predictor = instance.train(dataset) if hasattr(instance, "train") else instance
        if getattr(predictor, "prediction_length", request.horizon) != request.horizon:
            raise ForecastAdapterError("gluonts predictor prediction_length does not match the request horizon")
        draws = max(self._num_samples, request.samples)
        forecast = next(iter(predictor.predict(dataset, num_samples=draws)))
        rows, fixed, paths = None, 0, None
        if hasattr(forecast, "samples"):
            array = np.asarray(forecast.samples, dtype=float).reshape(draws, -1)[:, :request.horizon]
            point = array.mean(axis=0)
            if request.quantiles:
                rows, fixed = _base.rows_from_samples(request, array)
            paths = _base.take_paths(request, array)
        else:
            point = np.asarray(forecast.mean, dtype=float)
            if request.samples:
                raise ForecastAdapterError("this gluonts model returns quantiles, not sample paths")
            rows, fixed = _base.make_rows(request, lambda q: np.asarray(forecast.quantile(q), dtype=float))
        return _base.finish(request, point, kind=KIND, quantiles=rows, fixed=fixed, sample_paths=paths,
                            model=self._cls.__name__, index=note, num_samples=draws, seed=self._seed)
