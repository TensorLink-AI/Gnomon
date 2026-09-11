"""Prophet: additive trend and seasonality, fit fresh on every request.

TOML::

    [providers.prophet]
    kind = "prophet"
    seed = 7                       # optional; seeds the predictive samples
    [providers.prophet.options]    # Prophet(...) keyword arguments
    yearly_seasonality = "auto"
    uncertainty_samples = 1000

Prophet models calendar effects, so requests must carry real timestamps and
either a frequency or explicit future timestamps. Quantiles and sample paths
come from Prophet's predictive samples, not from a single interval width.
"""

from __future__ import annotations

import logging

from ..forecast_adapter import AdapterCapabilities, ForecastRequest, validate_capabilities
from . import _base

KIND = "prophet"
CAPABILITIES = AdapterCapabilities(quantiles=True, past_covariates=True, future_covariates=True,
                                   sample_paths=True, min_history=2)


def build(*, name: str, model: str, options: dict, seed) -> _base.Registration:
    import prophet  # noqa: F401
    if model != "Prophet":
        from ..forecast_adapter import ForecastAdapterError
        raise ForecastAdapterError("prophet supports model = \"Prophet\" only")
    seed = _base.check_seed(seed)
    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.ERROR)
    return _base.Registration(lambda: _Provider(options, seed), True, CAPABILITIES,
                              _base.revision(KIND, "prophet"), seed is not None)


class _Provider:
    capabilities = CAPABILITIES

    def __init__(self, options, seed):
        self._options, self._seed, self._used = options, seed, False

    def forecast(self, request: ForecastRequest):
        if self._used:
            raise RuntimeError("prophet fitting object must not be reused")
        self._used = True
        validate_capabilities(self.capabilities, request)
        import numpy as np
        import pandas as pd
        from prophet import Prophet
        index, freq = _base.require_dates(KIND, request)
        kwargs = {"uncertainty_samples": 1000, **self._options}
        if request.samples:
            kwargs["uncertainty_samples"] = max(int(kwargs["uncertainty_samples"]), request.samples)
        model = Prophet(**kwargs)
        hist, fut, names = _base.paired_covariates(request)
        frame = pd.DataFrame({"ds": _base.naive_utc(index), "y": list(request.history)})
        future = pd.DataFrame({"ds": _base.naive_utc(_base.future_index(request, index, freq))})
        for column, name in enumerate(names):
            model.add_regressor(name)
            frame[name], future[name] = hist[:, column], fut[:, column]
        model.fit(frame)
        point = model.predict(future)["yhat"].to_numpy()
        rows, fixed, paths = None, 0, None
        if request.quantiles or request.samples:
            _base.seed_numpy(self._seed)
            samples = np.asarray(model.predictive_samples(future)["yhat"], dtype=float).T
            rows, fixed = _base.rows_from_samples(request, samples)
            paths = _base.take_paths(request, samples)
        return _base.finish(request, point, kind=KIND, quantiles=rows, fixed=fixed, sample_paths=paths,
                            model="Prophet", index="timestamps_from_request", seed=self._seed)
