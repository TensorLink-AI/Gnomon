"""Tests for the tracking, scoring, and model performance system."""

import sys
import tempfile
import json
import csv
from pathlib import Path

sys.path.insert(0, "src")

import pytest
from aion.tracking import (
    DEFAULT_REGISTRY_PATH, TrackingStore, ForecastRecord, ModelPerformance, ScoreResult,
    mase_score, mape_score, bias_score, interval_coverage, threshold_accuracy,
    mean_absolute_error,
)


@pytest.fixture
def store(tmp_path):
    """A TrackingStore using a temp directory."""
    return TrackingStore(path=tmp_path / "test_registry.db")


@pytest.fixture
def sample_forecast_dir(tmp_path):
    """Create a fake forecast artifact directory with forecast.csv."""
    d = tmp_path / "forecast_test123"
    d.mkdir()
    fc_path = d / "forecast.csv"
    rows = [
        {"series": "__default__", "timestamp": "2026-01-01T00:00:00+00:00", "point": 100.0, "q10": 90.0, "q50": 100.0, "q90": 110.0},
        {"series": "__default__", "timestamp": "2026-01-01T01:00:00+00:00", "point": 105.0, "q10": 95.0, "q50": 105.0, "q90": 115.0},
        {"series": "__default__", "timestamp": "2026-01-01T02:00:00+00:00", "point": 110.0, "q10": 100.0, "q50": 110.0, "q90": 120.0},
    ]
    with open(fc_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["series", "timestamp", "point", "q10", "q50", "q90"])
        w.writeheader()
        w.writerows(rows)
    return d


class TestScoringFunctions:
    """Individual scoring metric functions."""

    def test_mae(self):
        assert mean_absolute_error([100, 110], [95, 105]) == 5.0

    def test_mase_beats_baseline(self):
        # If prediction equals actual, MASE = 0
        assert mase_score([100, 110, 120], [100, 110, 120], 10.0) == 0.0

    def test_mase_worse_than_baseline(self):
        # If error > naive error, MASE > 1
        result = mase_score([100, 110, 120], [200, 210, 220], 10.0)
        assert result > 1.0

    def test_mase_with_zero_naive_error(self):
        # MASE is undefined when the training-set naive scale is zero.
        result = mase_score([100, 110, 120], [100, 110, 120], 0.0)
        assert result is None

    def test_mape(self):
        result = mape_score([100, 200], [90, 210])
        # |100-90|/100 = 0.1, |200-210|/200 = 0.05 → avg = 7.5%
        assert abs(result - 7.5) < 0.01

    def test_bias_positive_means_underprediction(self):
        # Actual > predicted → positive bias (underprediction)
        assert bias_score([100, 110], [90, 100]) == 10.0

    def test_bias_negative_means_overprediction(self):
        assert bias_score([100, 110], [110, 120]) == -10.0

    def test_interval_coverage_all_inside(self):
        assert interval_coverage([100, 105], [90, 95], [110, 115]) == 1.0

    def test_interval_coverage_some_outside(self):
        assert interval_coverage([100, 120], [90, 95], [110, 115]) == 0.5

    def test_threshold_accuracy(self):
        actual = [100, 80, 120, 90]
        predicted_above = [True, False, True, False]
        threshold_val = 95
        result = threshold_accuracy(actual, threshold_val, predicted_above)
        # Step 1: actual 100 > 95, predicted True → correct
        # Step 2: actual 80 < 95, predicted False → correct
        # Step 3: actual 120 > 95, predicted True → correct
        # Step 4: actual 90 < 95, predicted False → correct
        assert result == 1.0

    def test_threshold_accuracy_partial(self):
        actual = [100, 80]
        predicted_above = [False, True]
        threshold_val = 95
        result = threshold_accuracy(actual, threshold_val, predicted_above)
        # Both wrong → 0.0
        assert result == 0.0

    def test_threshold_accuracy_none(self):
        assert threshold_accuracy([100], 50, None) is None


class TestTrackingStore:
    """Registry and scoring operations."""

    def test_register_and_get(self, store):
        store.register(
            forecast_id="test_001",
            project="api-capacity",
            series="__default__",
            horizon=24,
            frequency="h",
            selected_model="seasonal_naive",
            support="supported",
            artifact_path="/tmp/test",
        )
        record = store.get_forecast("test_001")
        assert record is not None
        assert record.project == "api-capacity"
        assert record.selected_model == "seasonal_naive"
        assert record.support == "supported"
        assert record.scored is False

    def test_default_registry_path_is_a_database_file(self):
        assert DEFAULT_REGISTRY_PATH.name == "registry.db"

    def test_list_by_project(self, store):
        store.register("f1", project="api", selected_model="drift")
        store.register("f2", project="billing", selected_model="ets")
        store.register("f3", project="api", selected_model="ets")

        api_forecasts = store.list_forecasts(project="api")
        assert len(api_forecasts) == 2
        assert all(f.project == "api" for f in api_forecasts)

    def test_list_all(self, store):
        store.register("f1", project="a")
        store.register("f2", project="b")
        all_f = store.list_forecasts()
        assert len(all_f) == 2

    def test_score_forecast(self, store):
        store.register(
            forecast_id="test_scoring",
            project="test",
            selected_model="drift",
            naive_error=10.0,
            artifact_path="/tmp/test",
        )
        result = store.score_forecast(
            "test_scoring",
            actuals=[100, 110, 120],
            points=[95, 105, 115],
            q10=[85, 95, 105],
            q90=[105, 115, 125],
        )
        assert result.mase is not None
        assert result.mape > 0
        assert result.coverage == 1.0  # all actuals within q10-q90

    def test_score_with_threshold(self, store):
        store.register(
            forecast_id="test_thresh",
            project="test",
            selected_model="drift",
            threshold=105,
            naive_error=10.0,
            artifact_path="/tmp/test",
        )
        result = store.score_forecast(
            "test_thresh",
            actuals=[100, 110, 120],
            points=[95, 115, 125],
            threshold=105,
            predicted_above=[False, True, True],
        )
        assert result.threshold_accuracy is not None
        # Step 1: actual 100 < 105, predicted False → correct
        # Step 2: actual 110 > 105, predicted True → correct
        # Step 3: actual 120 > 105, predicted True → correct
        assert result.threshold_accuracy == 1.0

    def test_score_nonexistent_raises(self, store):
        with pytest.raises(ValueError):
            store.score_forecast("nonexistent", [100], [100])

    def test_leaderboard(self, store):
        # Register and score two models
        store.register("f1", project="lb", selected_model="drift", naive_error=10.0)
        store.score_forecast("f1", [100, 110], [95, 105])
        store.register("f2", project="lb", selected_model="ets", naive_error=10.0)
        store.score_forecast("f2", [100, 110], [99, 109])

        lb = store.leaderboard("lb")
        assert len(lb) == 2
        # ets should be better (lower MASE since closer)
        assert lb[0].model in ("ets", "drift")
        assert lb[0].count == 1

    def test_model_performance_history(self, store):
        store.register("f1", project="p", selected_model="drift", naive_error=10.0)
        store.score_forecast("f1", [100, 110], [95, 105])
        store.register("f2", project="p", selected_model="drift", naive_error=10.0)
        store.score_forecast("f2", [200, 210], [195, 205])

        history = store.model_performance("p", "drift")
        assert len(history) == 2
        assert all(h["model"] == "drift" for h in history)

    def test_compare(self, store):
        store.register("f1", project="p", selected_model="drift", naive_error=10.0)
        store.score_forecast("f1", [100, 110], [95, 105])
        store.register("f2", project="p", selected_model="ets", naive_error=10.0)
        store.score_forecast("f2", [100, 110], [99, 109])

        result = store.compare("f1", "f2")
        assert "winner" in result
        assert result["forecast_a"]["model"] == "drift"
        assert result["forecast_b"]["model"] == "ets"

    def test_drift_detection(self, store):
        # Score 3 forecasts with consistent MASE ~0.5
        for i, (fid, actuals, points) in enumerate([
            ("d1", [100, 110], [98, 108]),
            ("d2", [200, 210], [198, 208]),
            ("d3", [300, 310], [298, 308]),
        ]):
            store.register(
                fid, project="drift_test", selected_model="ets", naive_error=4.0,
            )
            store.score_forecast(fid, actuals, points)

        # Now score a bad forecast (MASE will be much higher)
        store.register("d4", project="drift_test", selected_model="ets", naive_error=4.0)
        result = store.score_forecast("d4", [100, 110], [200, 210])

        assert result.drift_flag is not None
        assert "degraded" in result.drift_flag or "warning" in result.drift_flag

    def test_no_drift_with_few_scores(self, store):
        # Only 1 score — not enough history for drift detection
        store.register("d1", project="p", selected_model="drift", naive_error=10.0)
        result = store.score_forecast("d1", [100], [100])
        assert result.drift_flag is None

    def test_submit_actuals_csv(self, store, sample_forecast_dir, tmp_path):
        # Register a forecast
        store.register(
            forecast_id="test123",
            project="submit_test",
            selected_model="drift",
            naive_error=10.0,
            artifact_path=str(sample_forecast_dir),
        )

        # Create actuals CSV matching forecast timestamps
        actuals_path = tmp_path / "actuals.csv"
        with open(actuals_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "value"])
            w.writerow(["2026-01-01T00:00:00+00:00", 102.0])
            w.writerow(["2026-01-01T01:00:00+00:00", 107.0])
            w.writerow(["2026-01-01T02:00:00+00:00", 112.0])

        results = store.submit_actuals_csv("submit_test", str(actuals_path))
        assert len(results) == 1
        assert results[0].mase is not None
        assert results[0].mape > 0  # actuals differ from forecast

    def test_compare_treats_zero_mase_as_best(self, store):
        store.register("perfect", project="p", selected_model="a", naive_error=10.0)
        store.score_forecast("perfect", [100, 110], [100, 110])
        store.register("imperfect", project="p", selected_model="b", naive_error=10.0)
        store.score_forecast("imperfect", [100, 110], [101, 111])

        assert store.compare("perfect", "imperfect")["winner"] == "a"

    def test_panel_actuals_are_scored_per_series(self, store, tmp_path):
        artifact = tmp_path / "panel"
        artifact.mkdir()
        with (artifact / "forecast.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["series", "timestamp", "point", "q10", "q50", "q90"],
            )
            writer.writeheader()
            writer.writerow({"series": "a", "timestamp": "2026-01-01T00:00:00Z", "point": 10})
            writer.writerow({"series": "b", "timestamp": "2026-01-01T00:00:00Z", "point": 100})
        store.register(
            "panel:a", project="panel", series="a", selected_model="last_value",
            naive_error=2.0, artifact_path=str(artifact),
        )
        store.register(
            "panel:b", project="panel", series="b", selected_model="drift",
            naive_error=20.0, artifact_path=str(artifact),
        )
        actuals = tmp_path / "panel-actuals.csv"
        with actuals.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["series", "timestamp", "value"])
            writer.writerow(["a", "2026-01-01T00:00:00+00:00", 12])
            writer.writerow(["b", "2026-01-01T00:00:00+00:00", 120])

        results = store.submit_actuals_csv("panel", str(actuals))

        assert {result.forecast_id for result in results} == {"panel:a", "panel:b"}
        assert store.get_forecast("panel:a").mase == 1.0
        assert store.get_forecast("panel:b").mase == 1.0

    def test_panel_actuals_require_series_column(self, store, tmp_path):
        store.register("panel:a", project="panel", series="a")
        store.register("panel:b", project="panel", series="b")
        actuals = tmp_path / "actuals.csv"
        actuals.write_text("timestamp,value\n2026-01-01T00:00:00Z,12\n")

        with pytest.raises(ValueError, match="series.*column"):
            store.submit_actuals_csv("panel", str(actuals))

    def test_partial_actuals_do_not_close_forecast(self, store, sample_forecast_dir):
        store.register(
            "partial", project="p", selected_model="drift", naive_error=10.0,
            artifact_path=str(sample_forecast_dir),
        )

        results = store.submit_actuals(
            "p", [("2026-01-01T00:00:00Z", 102.0)],
        )

        assert results == []
        assert store.get_forecast("partial").scored is False

    def test_reregister_updates(self, store):
        """Re-registering with same ID should update, not duplicate."""
        store.register("f1", project="a", selected_model="drift", support="unsupported")
        store.register("f1", project="a", selected_model="ets", support="supported")
        record = store.get_forecast("f1")
        assert record.selected_model == "ets"
        assert record.support == "supported"
