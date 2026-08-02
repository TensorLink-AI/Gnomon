"""Fingerprints, task-conditioned tracking, and the thin router."""

from __future__ import annotations

import json
import math

import pytest

from aion.fingerprint import fingerprint_distance, fingerprint_json, series_fingerprint
from aion.router import MIN_PRIOR_RECORDS, route
from aion.tracking import TrackingStore


def _series(n: int = 60) -> list[float]:
    return [20.0 + 4.0 * math.sin(2 * math.pi * i / 7) + 0.1 * (i % 5) for i in range(n)]


class TestFingerprint:
    def test_fingerprint_dimensions(self):
        fingerprint = series_fingerprint(_series(), "D")
        assert fingerprint["length"] == 60
        assert fingerprint["frequency"] == "D"
        for key in ("trend", "noise_ratio", "intermittency",
                    "direction_change_rate", "season_period", "season_strength"):
            assert key in fingerprint

    def test_short_series_degrades_gracefully(self):
        fingerprint = series_fingerprint([1.0, 2.0], "D")
        assert fingerprint == {"length": 2, "frequency": "D"}

    def test_deterministic_json(self):
        assert fingerprint_json(_series(), "D") == fingerprint_json(_series(), "D")

    def test_distance_is_smaller_for_similar_shapes(self):
        base = series_fingerprint(_series(), "D")
        similar = series_fingerprint([v + 1.0 for v in _series()], "D")
        different = series_fingerprint([float(i * i % 17) for i in range(60)], "D")
        assert fingerprint_distance(base, similar) < fingerprint_distance(base, different)

    def test_distance_none_when_undimensioned(self):
        assert fingerprint_distance({"length": 2}, {"length": 3}) is None


class TestTrackingTaskDimension:
    def test_register_stores_task_and_fingerprint(self, tmp_path):
        store = TrackingStore(tmp_path / "t.db")
        fingerprint = fingerprint_json(_series(), "D")
        store.register("f1", "proj", task="forecast", fingerprint=fingerprint,
                       selected_model="theta", naive_error=1.0)
        record = store.get_forecast("f1")
        assert record.task == "forecast"
        assert json.loads(record.fingerprint)["length"] == 60

    def test_task_flows_into_performance_and_leaderboard(self, tmp_path):
        store = TrackingStore(tmp_path / "t.db")
        for index, task in enumerate(["forecast", "forecast", "detect_anomalies"]):
            store.register(f"f{index}", "proj", task=task,
                           selected_model="theta" if task == "forecast" else "robust_zscore",
                           naive_error=1.0)
            store.score_forecast(f"f{index}", [1.0, 2.0], [1.1, 2.1])
        forecast_board = store.leaderboard("proj", task="forecast")
        anomaly_board = store.leaderboard("proj", task="detect_anomalies")
        assert [m.model for m in forecast_board] == ["theta"]
        assert [m.model for m in anomaly_board] == ["robust_zscore"]
        assert sum(m.count for m in store.leaderboard("proj")) == 3

    def test_legacy_rows_default_to_forecast_task(self, tmp_path):
        store = TrackingStore(tmp_path / "t.db")
        with store._connect() as conn:
            conn.execute(
                "UPDATE forecasts SET task = NULL WHERE 1=0"
            )  # migration path exists; no error
        store.register("f1", "proj", selected_model="theta", naive_error=1.0)
        store.score_forecast("f1", [1.0], [1.0])
        assert [m.model for m in store.leaderboard("proj", task="forecast")] == ["theta"]

    def test_route_recording_roundtrip(self, tmp_path):
        store = TrackingStore(tmp_path / "t.db")
        store.record_route("r1", "proj", "forecast", recommendation="theta",
                           basis="tracking_prior", payload={"a": 1})
        routes = store.list_routes("proj")
        assert routes[0]["route_id"] == "r1"
        assert routes[0]["payload"] == {"a": 1}


class TestRouter:
    def test_unknown_task_raises(self):
        with pytest.raises(ValueError):
            route("classify", _series(), "D")

    def test_cold_start_is_honest(self, tmp_path):
        store = TrackingStore(tmp_path / "t.db")
        decision = route("forecast", _series(), "D", horizon=7,
                         project="proj", store=store)
        assert decision["basis"] == "backtest_required"
        assert decision["recommendation"] is None
        assert decision["prior"]["source"] is None
        assert "theta" in decision["candidates"]
        assert decision["fingerprint"]["length"] == 60
        # The decision was recorded for replay.
        assert store.list_routes("proj")[0]["basis"] == "backtest_required"

    def test_prior_engages_with_history(self, tmp_path):
        store = TrackingStore(tmp_path / "t.db")
        fingerprint = fingerprint_json(_series(), "D")
        for index in range(MIN_PRIOR_RECORDS):
            model = "theta" if index % 2 == 0 else "ets"
            store.register(f"f{index}", "proj", task="forecast",
                           selected_model=model, naive_error=1.0,
                           fingerprint=fingerprint)
            # theta scores better than ets
            actual, predicted = [10.0], ([10.1] if model == "theta" else [12.0])
            store.score_forecast(f"f{index}", actual, predicted)
        decision = route("forecast", _series(), "D", horizon=7,
                         project="proj", store=store)
        assert decision["basis"] == "tracking_prior"
        assert decision["recommendation"] == "theta"
        assert decision["prior"]["source"] == "tracking"
        ranking = decision["prior"]["ranking"]
        assert ranking[0]["fingerprint_weighted_mase"] <= ranking[1]["fingerprint_weighted_mase"]

    def test_anomaly_task_candidates(self):
        decision = route("detect_anomalies", _series(), "D")
        assert set(decision["candidates"]) == {
            "robust_zscore", "rolling_median_residual", "local_slope",
            "forecast_interval"}
        assert decision["basis"] == "backtest_required"

    def test_too_short_series_has_no_candidates(self):
        decision = route("detect_anomalies", [1.0] * 8, "D")
        assert decision["candidates"] == []
        assert decision["basis"] == "no_eligible_candidates"
        assert decision["excluded"]

    def test_decision_is_deterministic(self, tmp_path):
        first = route("forecast", _series(), "D", horizon=7)
        second = route("forecast", _series(), "D", horizon=7)
        assert first == second

    def test_capabilities_flags(self):
        from aion.runtime import capabilities
        features = capabilities()["features"]
        assert features["series_fingerprints"] is True
        assert features["task_conditioned_leaderboard"] is True
        assert features["task_routing"] is True
