from dataclasses import replace
import subprocess
import sys

import pytest

from gnomon import AdapterCapabilities, ForecastRequest, ForecastResult, InferenceEngine
from gnomon.forecast_adapter import ForecastAdapterError, LegacyModelAdapter


def echo(request):
    return ForecastResult((request.history[-1],) * request.horizon,
                          timestamps=request.future_timestamps,
                          series_id=request.series_id, unit=request.unit)


def test_callable_without_catalogue_and_without_backtests():
    engine = InferenceEngine()
    engine.register("user/anything", echo)
    run = engine.forecast("user/anything", ForecastRequest((1, 9), 2))
    assert run.result.point == (9, 9)
    assert run.evidence == "inference_only"
    assert run.action_authorized is False
    assert run.revision is None
    with pytest.raises(ForecastAdapterError, match="already registered"):
        engine.register("user/anything", echo)


def test_timestamp_series_and_unit_roundtrip():
    req = ForecastRequest((1, 2), 1, timestamps=("2025-01-01", "2025-01-02"),
                          future_timestamps=("2025-01-03",), cutoff="2025-01-02",
                          series_id="shop/one", unit="USD", snapshot_id="sha256:abc")
    engine = InferenceEngine()
    engine.register("user", echo)
    assert engine.forecast("user", req).result.validate(req)
    engine.register("bad", lambda r: ForecastResult((3,)))
    with pytest.raises(ForecastAdapterError, match="timestamps"):
        engine.forecast("bad", req)


@pytest.mark.parametrize("kwargs", [
    {"horizon": 1.9}, {"horizon": True}, {"season": 1.1}, {"samples": -1},
    {"timestamps": ("2025-01-02", "2025-01-01")},
    {"timestamps": ("2025-01-01", "2025-01-02"), "cutoff": "2025-01-01"},
    {"timestamps": ("2025-01-01", "2025-01-02"), "future_timestamps": ("2025-01-02",)},
    {"timestamps": ("2025-01-01", "2025-01-02T00:00:00+00:00")},
    {"timestamps": ("2025-01-01", "2025-01-02"), "known_time_cutoff": "2025-01-03T00:00:00Z"},
    {"past_covariates": ((1,), (2,)), "past_covariate_names": ("a", "b")},
])
def test_request_rejects_ambiguous_or_misaligned_values(kwargs):
    with pytest.raises(ForecastAdapterError):
        ForecastRequest(**{"history": (1, 2), "horizon": 1, **kwargs})


def test_from_values_does_not_truncate_horizon():
    with pytest.raises(ForecastAdapterError):
        ForecastRequest.from_values([1, 2], 1.9, 1)


def test_capabilities_fail_before_callable_runs():
    def unexpected(req):
        pytest.fail("unsupported input must not execute")
    engine = InferenceEngine()
    engine.register("user", unexpected)
    for req in (ForecastRequest((1, 2), 1, quantiles=(.5,)),
                ForecastRequest((1, 2), 1, samples=3),
                ForecastRequest((1, 2), 1, related_series=((3, 4),))):
        with pytest.raises(ForecastAdapterError, match="does not support"):
            engine.forecast("user", req)


def test_requested_uncertainty_must_be_returned():
    engine = InferenceEngine()
    engine.register("user", echo, capabilities=AdapterCapabilities(quantiles=True, sample_paths=True))
    with pytest.raises(ForecastAdapterError, match="no requested quantiles"):
        engine.forecast("user", ForecastRequest((1, 2), 1, quantiles=(.5,)))
    with pytest.raises(ForecastAdapterError, match="number of sample paths"):
        engine.forecast("user", ForecastRequest((1, 2), 1, samples=3))


def test_extra_uncertainty_outputs_are_also_validated():
    request = ForecastRequest((1, 2), 1)
    for result in (ForecastResult((3,), ({.1: 4, .9: 2},)),
                   ForecastResult((3,), ({1.1: 4},)),
                   ForecastResult((3,), sample_paths=((float("nan"),),))):
        with pytest.raises(ForecastAdapterError):
            result.validate(request)


def test_uncertainty_is_data_not_calibration_or_action_authority():
    engine = InferenceEngine()
    engine.register("user", lambda r: ForecastResult((2,), ({.1: 1, .5: 2, .9: 3},),
                                                     sample_paths=((1,), (3,))),
                    capabilities=AdapterCapabilities(quantiles=True, sample_paths=True))
    run = engine.forecast("user", ForecastRequest((1, 2), 1, quantiles=(.1, .5, .9), samples=2))
    assert run.result.sample_paths == ((1,), (3,))
    assert run.evidence == "inference_only" and not run.action_authorized


def test_factory_is_fresh_for_each_fold_and_closes():
    states = []
    class Fitted:
        def __init__(self):
            self.histories = []
            self.closed = False
            states.append(self)
        def forecast(self, r):
            self.histories.append(r.history)
            return echo(r)
        def close(self):
            self.closed = True
    engine = InferenceEngine()
    engine.register_factory("fitted", Fitted)
    result = engine.forecast_batch("fitted", [ForecastRequest((1, 2), 1), ForecastRequest((1, 2, 3), 1)])
    assert [r.result.point for r in result] == [(2,), (3,)]
    assert len(states) == 2 and all(s.closed and len(s.histories) == 1 for s in states)
    assert engine.capabilities()["fitted"]["lifecycle"] == "fresh_per_request"


def test_native_batch_preserves_identity_and_validates_before_dispatch():
    class Batch:
        calls = 0
        def forecast(self, r):
            pytest.fail("native batch should be used")
        def forecast_batch(self, requests):
            self.calls += 1
            return [echo(r) for r in requests]
    batch = Batch()
    engine = InferenceEngine()
    engine.register("batch", batch)
    requests = [ForecastRequest((1, 2), 1, series_id="one"), ForecastRequest((3, 4), 1, series_id="two")]
    assert [r.result.series_id for r in engine.forecast_batch("batch", requests)] == ["one", "two"]
    with pytest.raises(ForecastAdapterError):
        engine.forecast_batch("batch", [requests[0], replace(requests[1], quantiles=(.5,))])
    assert batch.calls == 1
    batch.forecast_batch = lambda rs: [echo(rs[0])]
    with pytest.raises(ForecastAdapterError, match="batch size"):
        engine.forecast_batch("batch", requests)


def test_cache_requires_version_and_determinism_and_preserves_execution_identity():
    calls = []
    def counted(r):
        calls.append(r)
        return replace(echo(r), metadata={"nested": [1]})
    engine = InferenceEngine(cache_size=1)
    for name, revision, deterministic in (("safe", "v1", True), ("unversioned", None, True), ("random", "v1", False)):
        engine.register(name, counted, revision=revision, deterministic=deterministic)
        first = engine.forecast(name, ForecastRequest((1, 2), 1))
        first.result.metadata["nested"].append(99)
        second = engine.forecast(name, ForecastRequest((1, 2), 1))
        assert first.execution_id != second.execution_id
        assert first.fingerprint == second.fingerprint
        assert second.cache_hit == (name == "safe")
        assert second.result.metadata["nested"] == [1]
    assert len(calls) == 5


def test_snapshot_identity_and_cutoffs_partition_cache():
    engine = InferenceEngine(cache_size=5)
    engine.register("user", echo, revision="v1", deterministic=True)
    request = ForecastRequest((1, 2), 1, snapshot_id="one")
    a = engine.forecast("user", request)
    b = engine.forecast("user", replace(request, snapshot_id="two"))
    c = engine.forecast("user", replace(request, known_time_cutoff="2025-01-01"))
    assert len({a.fingerprint, b.fingerprint, c.fingerprint}) == 3


def test_legacy_bridge_cannot_claim_then_drop_covariates():
    class Legacy:
        name = "legacy"
        supports_past_covariates = True
        def predict(self, *args):
            pytest.fail("must not drop covariates")
    with pytest.raises(ForecastAdapterError, match="cannot forward"):
        LegacyModelAdapter(Legacy()).forecast(ForecastRequest((1, 2), 1, past_covariates=((3,), (4,))))


def test_direct_inference_does_not_load_optional_context_or_evaluation_stack():
    result = subprocess.run([sys.executable, "-c", """
import sys
from gnomon import InferenceEngine, ForecastRequest, ForecastResult
engine = InferenceEngine()
engine.register('user', lambda r: ForecastResult((2,)))
engine.forecast('user', ForecastRequest((1, 2), 1))
for module in ('runtime', 'evaluation', 'publication', 'context_intelligence', 'llm_dossier'):
    assert 'gnomon.' + module not in sys.modules, module
from gnomon import forecast, TemporalStore
assert callable(forecast)
assert TemporalStore.__name__ == 'TemporalStore'
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
