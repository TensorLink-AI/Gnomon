"""User-owned examples, not additional Gnomon built-in model adapters."""

from statistics import fmean

from gnomon import ForecastRequest, ForecastResult


def _result(request: ForecastRequest, value: float) -> ForecastResult:
    return ForecastResult(
        point=(value,) * request.horizon,
        timestamps=request.future_timestamps,
        series_id=request.series_id,
        unit=request.unit,
        metadata={"example_package": "gnomon-example-provider/0.0.0"},
    )


def last_value(request: ForecastRequest) -> ForecastResult:
    """A stateless callable; no fit object needs to survive between requests."""
    return _result(request, request.history[-1])


class FittedMean:
    """Stand-in for a library-owned mutable fitting object, used once only.

    Replace this fit/predict block with your own library configuration and output
    conversion. This example does not install or execute StatsForecast, Darts or
    NeuralForecast and makes no accuracy claim about those libraries.
    """

    def __init__(self):
        self.fitted = False

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        if self.fitted:
            raise RuntimeError("this fitting object must not be reused")
        self.fitted = True
        return _result(request, fmean(request.history))


def fresh_mean() -> FittedMean:
    return FittedMean()
