from __future__ import annotations

import math
from types import SimpleNamespace
import warnings

import pytest

from gnomon.config import GnomonConfig
from gnomon.evaluation import evaluate
from gnomon.forecast_adapter import (
    AdapterCapabilities, LegacyModelAdapter, conformance_report,
)
from gnomon.statsforecast_adapter import (
    DEFAULT_CANDIDATES, StatsForecastAdapter, installation_status,
    statsforecast_candidates,
)


def test_missing_optional_dependency_is_a_structured_soft_skip(monkeypatch):
    monkeypatch.setattr(
        "gnomon.statsforecast_adapter.installed_version", lambda: None)
    adapters, receipt = statsforecast_candidates()
    assert adapters == []
    assert receipt["status"] == "soft_skip_missing_dependency"
    assert receipt["requested"] == list(DEFAULT_CANDIDATES)


def test_incompatible_optional_dependency_is_a_structured_soft_skip(monkeypatch):
    monkeypatch.setattr(
        "gnomon.statsforecast_adapter.installed_version", lambda: "3.0.0")
    adapters, receipt = statsforecast_candidates()
    assert adapters == []
    assert receipt["status"] == "soft_skip_incompatible_version"
    assert receipt["version"] == "3.0.0"


def test_unknown_plugin_candidate_fails_loudly():
    with pytest.raises(ValueError, match="unknown StatsForecast"):
        statsforecast_candidates(["statsforecast_not_a_model"])


def test_backend_numerical_warnings_are_receipted_not_leaked(
        monkeypatch, capsys):
    class _Numpy:
        float64 = float

        @staticmethod
        def asarray(values, dtype=None):
            return list(values)

    class _Model:
        def __init__(self, **kwargs):
            pass

        def forecast(self, *, y, h):
            warnings.warn("unstable fit", RuntimeWarning)
            return {"mean": [float(y[-1])] * h}

    def fake_import(name):
        if name == "numpy":
            return _Numpy
        if name == "statsforecast.models":
            return SimpleNamespace(AutoETS=_Model)
        raise ImportError(name)

    monkeypatch.setattr(
        "gnomon.statsforecast_adapter.import_module", fake_import)
    adapter = StatsForecastAdapter("statsforecast_autoets", "2.1.1")
    assert adapter.predict([float(value) for value in range(12)], 2, 1) \
        == [11.0, 11.0]
    assert adapter.last_warnings == ["RuntimeWarning: unstable fit"]
    assert capsys.readouterr().err == ""


class _PerfectTrendPlugin:
    name = "statsforecast_test_trend"
    kind = "statistical_plugin"
    backend = "in_process"
    revision = "2.1.1-test"
    model_class = "TestTrend"
    min_history = 8
    max_horizon = None
    capabilities = AdapterCapabilities(min_history=8)
    supports_quantiles = False
    supports_past_covariates = False
    supports_future_covariates = False
    supports_panel = False
    supports_sample_paths = False

    def predict(self, history, horizon, season):
        slope = history[-1] - history[-2]
        return [history[-1] + slope * step
                for step in range(1, horizon + 1)]


def test_plugin_uses_governed_selection_and_binds_executable(monkeypatch):
    adapter = _PerfectTrendPlugin()
    monkeypatch.setattr(
        "gnomon.statsforecast_adapter.statsforecast_candidates",
        lambda requested=None: ([adapter], {
            "status": "available", "version": adapter.revision,
            "required": ">=2.1,<3", "requested": [adapter.name],
        }),
    )
    config = GnomonConfig()
    config.models.statistical_enabled = False
    config.models.statsforecast_enabled = True
    config.models.statsforecast_candidates = [adapter.name]
    values = [10.0 + 2.0 * index for index in range(60)]
    result = evaluate(
        values, 4, 1, .02, config=config, tsfm_names=[],
        strict_abstention=False,
    )
    assert result.selected_model == adapter.name
    assert result.final_candidate is not None
    assert result.final_candidate.identity.kind == "statistical_plugin"
    assert result.final_candidate.identity.revisions[adapter.name] == \
        "statsforecast@2.1.1-test"
    assert result.final_candidate.fit(values, 1).predict(4) == \
        [130.0, 132.0, 134.0, 136.0]
    receipt = result.adapter_receipts["statsforecast"]["candidates"][adapter.name]
    assert receipt["completed_folds"] == receipt["required_folds"]
    assert receipt["failures"] == []
    assert result.tsfm_scores == {}
    assert result.statistical_plugin_scores[adapter.name] is not None


def test_installed_models_conform_when_extra_is_available():
    if not installation_status()["compatible"]:
        pytest.skip("optional StatsForecast extra is not installed")
    adapters, receipt = statsforecast_candidates()
    assert receipt["status"] == "available"
    seasonal = [
        20 + .3 * index + 2 * math.sin(2 * math.pi * index / 4)
        for index in range(32)
    ]
    intermittent = [0, 0, 4, 0, 0, 6, 0, 0, 5, 0, 0, 7, 0, 0, 6, 0]
    for adapter in adapters:
        history = intermittent if "croston" in adapter.name else seasonal
        report = conformance_report(
            LegacyModelAdapter(adapter), history=history, horizon=4, season=4)
        assert report["conformant"], report
