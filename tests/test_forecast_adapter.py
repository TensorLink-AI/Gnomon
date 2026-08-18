from __future__ import annotations

import pytest

from gnomon.forecast_adapter import (
    AdapterCapabilities, ForecastAdapterError, ForecastRequest, ForecastResult,
    LegacyModelAdapter, StatisticalAdapter, conformance_report,
)
from gnomon.models import predict
from gnomon.api_inference import APIAdapter
from gnomon.config import APIAuthConfig, APIProviderConfig
from gnomon.tsfm import TSFMError
from gnomon.tsfm_sandbox import SubprocessAdapter


def test_statistical_models_cross_the_same_validated_protocol() -> None:
    adapter = StatisticalAdapter("last_value", predict)
    report = conformance_report(adapter)
    assert report["conformant"] is True
    request = ForecastRequest.from_values([1, 2, 3], 3, 1)
    assert adapter.forecast(request).points() == [3, 3, 3]


class _Short:
    name = "short"
    supports_quantiles = False
    def predict(self, history, horizon, season):
        return [history[-1]] * max(0, horizon - 1)


class _NonFinite:
    name = "nonfinite"
    supports_quantiles = False
    def predict(self, history, horizon, season):
        return [float("nan")] * horizon


class _CrossedQuantiles:
    name = "crossed"
    supports_quantiles = True
    def predict(self, history, horizon, season):
        return [history[-1]] * horizon
    def predict_quantiles(self, history, horizon, season, quantiles):
        return [{0.1: 3, 0.5: 2, 0.9: 1} for _ in range(horizon)]


@pytest.mark.parametrize("raw", [_Short(), _NonFinite()])
def test_malformed_adapter_outputs_fail_loudly(raw) -> None:
    adapter = LegacyModelAdapter(raw)
    request = ForecastRequest.from_values([1, 2, 3], 3, 1)
    with pytest.raises(ForecastAdapterError):
        adapter.forecast(request)


def test_crossed_quantiles_fail_loudly() -> None:
    request = ForecastRequest.from_values(
        [1, 2, 3], 2, 1, quantiles=(.1, .5, .9))
    with pytest.raises(ForecastAdapterError, match="not monotone"):
        LegacyModelAdapter(_CrossedQuantiles()).forecast(request)


def test_request_rejects_invalid_history_and_quantiles() -> None:
    with pytest.raises(ForecastAdapterError):
        ForecastRequest.from_values([1, float("nan")], 2, 1)
    with pytest.raises(ForecastAdapterError):
        ForecastRequest.from_values([1, 2], 2, 1, quantiles=(.9, .1))


def _provider() -> APIProviderConfig:
    return APIProviderConfig(
        url="https://example.invalid/forecast", model="remote-model",
        revision="sha256:abc",
        auth=APIAuthConfig(type="none"))


def test_api_adapter_sends_protocol_and_season_and_validates_response(monkeypatch) -> None:
    adapter = APIAdapter("remote", _provider())
    seen = {}
    def call(payload):
        seen.update(payload)
        return {"point": [4.0, 5.0]}
    monkeypatch.setattr(adapter, "_call_api", call)
    assert adapter.predict([1, 2, 3], 2, 7) == [4.0, 5.0]
    assert seen["protocol_version"] == "0.1"
    assert seen["season"] == 7
    assert seen["revision"] == "sha256:abc"


def test_api_adapter_rejects_short_remote_response(monkeypatch) -> None:
    adapter = APIAdapter("remote", _provider())
    monkeypatch.setattr(adapter, "_call_api", lambda payload: {"point": [4.0]})
    with pytest.raises(TSFMError, match="forecast contract"):
        adapter.predict([1, 2, 3], 2, 1)


def test_adapter_rejects_covariates_it_does_not_claim() -> None:
    adapter = StatisticalAdapter("last_value", predict)
    request = ForecastRequest(
        history=(1.0, 2.0, 3.0), horizon=1,
        future_covariates=((4.0,),))
    with pytest.raises(ForecastAdapterError, match="future_covariates"):
        adapter.forecast(request)


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


def test_sandbox_adapter_identity_carries_pinned_weight_revision() -> None:
    adapter = SubprocessAdapter("chronos_bolt_mini")
    assert adapter.revision is not None
    assert "amazon/chronos-bolt-mini@" in adapter.revision
