from __future__ import annotations

import pytest

from gnomon.forecast_adapter import (
    AdapterCapabilities, ForecastAdapterError, ForecastRequest, ForecastResult,
    StatisticalAdapter, conformance_report,
)
from gnomon.models import predict


def test_statistical_models_cross_the_same_validated_protocol() -> None:
    adapter = StatisticalAdapter("last_value", predict)
    report = conformance_report(adapter)
    assert report["conformant"] is True
    request = ForecastRequest.from_values([1, 2, 3], 3, 1)
    assert adapter.forecast(request).points() == [3, 3, 3]


def test_point_metrics_handle_missing_values_large_values_and_cancellation():
    from gnomon.forecast_adapter import point_error_metrics
    assert point_error_metrics([]) == {"n": 0, "mae": None, "rmse": None, "bias": None}
    score = point_error_metrics([(1e308, 0), (-1e308, 0)])
    assert score["mae"] == 1e308 and score["bias"] == 0
    assert score["rmse"] == pytest.approx(1e308)
    assert point_error_metrics([(2, 3), (2, 4)]) == {
        "n": 2, "mae": 1.5, "rmse": pytest.approx(2.5 ** 0.5), "bias": -1.5,
    }


def test_stochastic_outputs_are_conformant_unless_repeatability_is_required():
    class Stochastic:
        name, kind = "stochastic", "callable"
        count = 0
        def forecast(self, request):
            self.count += 1
            return ForecastResult((float(self.count),) * request.horizon)
    report = conformance_report(Stochastic())
    assert report["conformant"] and report["checks"]["deterministic_replay"] is False
    assert conformance_report(Stochastic(), require_deterministic=True)["conformant"] is False


def test_request_ids_do_not_make_deterministic_forecasts_stochastic():
    class Provider:
        name, kind = "request-ids", "api"
        count = 0
        def forecast(self, request):
            self.count += 1
            return ForecastResult((1.0,) * request.horizon, metadata={"request_id": str(self.count)})
    assert conformance_report(Provider(), require_deterministic=True)["conformant"] is True


def test_conformance_checks_actual_request_mutation_not_an_unused_source_list():
    class Mutating:
        name, kind = "mutating", "callable"
        def forecast(self, request):
            object.__setattr__(request, "history", (999.0,))
            return ForecastResult((1.0,) * request.horizon)
    report = conformance_report(Mutating())
    assert not report["conformant"] and not report["checks"]["input_immutable"]


def test_request_rejects_invalid_history_and_quantiles() -> None:
    with pytest.raises(ForecastAdapterError):
        ForecastRequest.from_values([1, float("nan")], 2, 1)
    with pytest.raises(ForecastAdapterError):
        ForecastRequest.from_values([1, 2], 2, 1, quantiles=(.9, .1))


def test_capability_contract_can_admit_declared_features() -> None:
    class CovariateAdapter:
        name = "covariate"
        kind = "api"
        revision = "r1"
        capabilities = AdapterCapabilities(future_covariates=True)

        def forecast(self, request):
            return ForecastResult((request.future_covariates[0][0],))

    request = ForecastRequest(
        history=(1.0, 2.0), horizon=1,
        future_covariates=((9.0,),))
    assert CovariateAdapter().forecast(request).validate(request).point == (9.0,)
