"""neuralforecast: one Nixtla neural model trained on every request.

TOML::

    [providers.nhits]
    kind = "neuralforecast"
    model = "NHITS"                    # class in neuralforecast.models, or module:Class
    seed = 7
    [providers.nhits.options]          # constructor keyword arguments
    max_steps = 100                    # training steps; small by default
    input_size = 28                    # default: twice the horizon, capped by history

Training happens on each call, including every evaluation fold, so this is
the most expensive adapter. Quantile requests switch the loss to MQLoss over
exactly the requested quantiles; the point forecast is then the median.
"""

from __future__ import annotations

import logging
import warnings

from ..forecast_adapter import AdapterCapabilities, ForecastAdapterError, ForecastRequest, validate_capabilities
from . import _base

KIND = "neuralforecast"
CAPABILITIES = AdapterCapabilities(quantiles=True, min_history=4)


def build(*, name: str, model: str, options: dict, seed) -> _base.Registration:
    import neuralforecast  # noqa: F401
    import torch  # noqa: F401
    cls = _base.resolve(model, default_module="neuralforecast.models")
    seed = _base.check_seed(seed)
    for logger in ("pytorch_lightning", "lightning", "lightning.pytorch", "neuralforecast"):
        logging.getLogger(logger).setLevel(logging.ERROR)
    return _base.Registration(lambda: _Provider(cls, options, seed), True, CAPABILITIES,
                              _base.revision(KIND, "neuralforecast"), seed is not None)


class _Provider:
    capabilities = CAPABILITIES

    def __init__(self, cls, options, seed):
        self._cls, self._options, self._seed, self._used = cls, options, seed, False

    def forecast(self, request: ForecastRequest):
        if self._used:
            raise RuntimeError("neuralforecast fitting object must not be reused")
        self._used = True
        validate_capabilities(self.capabilities, request)
        import pandas as pd
        from neuralforecast import NeuralForecast
        n, h = len(request.history), request.horizon
        index, freq, note = _base.history_index(request)
        if isinstance(index, pd.DatetimeIndex):
            if freq is None:
                raise ForecastAdapterError("neuralforecast needs frequency with dated history",
                                           details={"missing_fields": ["frequency"]})
            ds, nf_freq = _base.naive_utc(index), freq
        else:
            ds, nf_freq = index, 1
        kwargs = {"max_steps": 100, "enable_progress_bar": False, "logger": False,
                  "enable_model_summary": False, **self._options}
        input_size = kwargs.pop("input_size", None)
        if input_size is None:
            input_size = min(max(2 * h, request.season if request.season > 1 else 0, 1), max(n - h, 1))
        if n < input_size + h:
            raise ForecastAdapterError(
                f"neuralforecast needs at least input_size + horizon = {input_size + h} observations; observed={n}",
                details={"required_history": input_size + h, "observed_history": n, "input_size": input_size})
        levels = _base.levels(request.quantiles)
        if request.quantiles:
            from neuralforecast.losses.pytorch import MQLoss
            kwargs["loss"] = MQLoss(level=levels) if levels else MQLoss(quantiles=[0.5])
        if self._seed is not None and _base.accepts(self._cls, "random_seed"):
            kwargs.setdefault("random_seed", self._seed)
        model = self._cls(h=h, input_size=input_size, **kwargs)
        alias = getattr(model, "alias", None) or type(model).__name__
        frame = pd.DataFrame({"unique_id": "y", "ds": ds, "y": list(request.history)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster = NeuralForecast(models=[model], freq=nf_freq)
            forecaster.fit(frame)
            result = forecaster.predict()
        columns = {c: result[c].to_numpy() for c in result.columns if c.startswith(alias)}
        rows, fixed = None, 0
        if request.quantiles:
            # MQLoss names columns "<alias>-median", "<alias>-lo-<level>", "<alias>-hi-<level>".
            bounds = {}
            for column, values in columns.items():
                suffix = column[len(alias):]
                if suffix == "-median":
                    bounds[("median", 0)] = values
                elif suffix.startswith(("-lo-", "-hi-")):
                    bounds[(suffix[1:3], round(float(suffix[4:]), 6))] = values

            def at(q):
                key = ("median", 0) if q == 0.5 else ("lo" if q < 0.5 else "hi", round(float(_base.level_for(q)), 6))
                if key not in bounds:
                    raise ForecastAdapterError(f"neuralforecast returned no column for quantile {q}")
                return bounds[key]
            rows, fixed = _base.make_rows(request, at)
            point = bounds.get(("median", 0), next(iter(columns.values())))
        else:
            point = columns[alias] if alias in columns else next(iter(columns.values()))
        return _base.finish(request, point, kind=KIND, quantiles=rows, fixed=fixed, model=alias, index=note,
                            input_size=input_size, max_steps=kwargs["max_steps"], seed=self._seed,
                            point_is_median=bool(request.quantiles))
