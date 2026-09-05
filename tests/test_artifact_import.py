from datetime import datetime, timezone
import json
import sqlite3

import pytest

from gnomon import TemporalLedger
from gnomon.config import GnomonConfig
from gnomon.contracts import GnomonError
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock
from gnomon.runtime import forecast
from gnomon.temporal_store import TemporalObservation, TemporalStore
from gnomon.tracking import TrackingStore, register_artifact


def legacy_artifact(tmp_path):
    directory = tmp_path / "legacy-artifact"
    directory.mkdir()
    payload = {"forecast_id": "old-forecast", "created_at": "2025-01-03T00:00:00Z",
               "task": {"schema": {"frequency": "D"}},
               "results": [{"series": "shop", "selected_model": "old-model", "support": "supported",
                            "forecast": [{"timestamp": "2025-01-04T00:00:00", "point": 3,
                                          "q10": 2, "q50": 3, "q90": 4}]}]}
    (directory / "artifact.json").write_text(json.dumps(payload))
    return directory


def test_legacy_import_preserves_forecast_but_does_not_invent_history_or_execution_time(tmp_path):
    directory = legacy_artifact(tmp_path)
    ledger = TemporalLedger(tmp_path / "ledger.db")
    ids = ledger.import_artifact(str(directory), project="business")
    assert ids == ledger.import_artifact(str(directory), project="business")
    run = ledger.execution(ids[0])
    assert run["request"]["history"] is None
    assert run["result"]["point"] == [3]
    assert run["result"]["metadata"]["source_created_at"] == "2025-01-03T00:00:00Z"
    assert run["result"]["metadata"]["integrity"] == "legacy_unsealed"
    assert run["evidence"] == "imported_forecast" and run["revision"] is None
    assert run["recorded_at"] != run["result"]["metadata"]["source_created_at"]
    assert ledger.pending()[0]["reason"] == "timezone_unresolved"
    with pytest.raises(ForecastAdapterError, match="explicit timezone"):
        ledger.evaluate(ids[0])
    bound = ledger.import_artifact(str(directory), project="business", naive_timezone="UTC")[0]
    assert bound != ids[0]
    ledger.append_actual(series_id="business:shop", valid_time="2025-01-04T00:00:00Z", value=4,
                         source_available_at="2025-01-05T00:00:00Z")
    assert ledger.evaluate(bound)["mae"] == 1
    with pytest.raises(ForecastAdapterError, match="complete input history"):
        ledger.compare([ids[0], bound])


def test_read_only_tracking_migration_keeps_legacy_scores_in_original_registry(tmp_path):
    directory = legacy_artifact(tmp_path)
    registry = tmp_path / "registry.db"
    tracker = TrackingStore(registry)
    tracker.register("old", "project", series="shop", artifact_path=str(directory))
    tracker.register("missing", "project", series="other", artifact_path=str(tmp_path / "missing"))
    before = registry.read_bytes()
    ledger = TemporalLedger(tmp_path / "ledger.db")
    report = ledger.import_tracking(str(registry), project="project", naive_timezone="UTC")
    assert len(report["execution_ids"]) == 1
    assert report["skipped"][0]["forecast_id"] == "missing"
    assert report["legacy_registry_unchanged"] is True
    assert "not_reconstructed" in report["score_history"]
    assert registry.read_bytes() == before
    assert ledger.evaluations(report["execution_ids"][0]) == []


def test_schema_one_upgrade_preserves_existing_records(tmp_path):
    ledger = TemporalLedger(tmp_path / "ledger.db")
    actual_id = ledger.append_actual(series_id="shop", valid_time="2025-01-04T00:00:00Z", value=4,
                                    source_available_at="2025-01-05T00:00:00Z")
    # Model the previous schema: data is unchanged; only the imports table and
    # associated guards were added in v2.
    with sqlite3.connect(ledger.path) as conn:
        conn.execute("DROP TABLE imports")
        conn.execute("PRAGMA user_version=1")
    upgraded = TemporalLedger(ledger.path)
    assert upgraded.actuals_as_of("shop")[0]["actual_id"] == actual_id
    assert upgraded.import_artifact(str(legacy_artifact(tmp_path)))


def test_registration_uses_sealed_history_after_source_changes(tmp_path, monkeypatch):
    import gnomon.tracking as tracking
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "registry.db"))
    source = tmp_path / "input.csv"
    source.write_text("timestamp,value\n2025-01-01,1\n2025-01-02,2\n2025-01-03,4\n")
    artifact, path = forecast(str(source), time_column="timestamp", target_column="value", horizon=1,
                              frequency="D", output=str(tmp_path / "out"))
    source.write_text("timestamp,value\n2030-01-01,10000\n2030-01-02,90000\n")
    [tracking_id] = register_artifact(artifact, "project", str(path))
    record = tracking.TrackingStore().get_forecast(tracking_id, "project")
    assert record.cutoff_time == "2025-01-03T00:00:00"
    assert record.naive_error == 1.5
    ledger = TemporalLedger(tmp_path / "ledger.db")
    [execution_id] = ledger.import_artifact(str(path), naive_timezone="UTC")
    assert ledger.execution(execution_id)["request"]["history"] == [1, 2, 4]
    assert ledger.execution(execution_id)["result"]["metadata"]["integrity"] == "verified"


def test_store_snapshots_register_without_reopening_store_uri_as_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "registry.db"))
    temporal = TemporalStore(tmp_path / "temporal.db")
    rows = [TemporalObservation("shop", "sales", datetime(2025, 1, day), datetime(2025, 1, day), day)
            for day in range(1, 8)]
    temporal.ingest_rows("sales", rows, source_fingerprint="test")
    artifact, path = forecast("store:sales", time_column="timestamp", target_column="sales", horizon=1,
                              store_path=str(temporal.path), as_of=datetime(2025, 1, 5), output=str(tmp_path / "out"))
    [tracking_id] = register_artifact(artifact, "project", str(path))
    assert TrackingStore().get_forecast(tracking_id, "project").cutoff_time == "2025-01-05T00:00:00"


def test_scoring_works_when_optional_forecast_csv_is_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "registry.db"))
    source = tmp_path / "input.csv"
    source.write_text("timestamp,value\n2025-01-01,1\n2025-01-02,2\n2025-01-03,4\n")
    config = GnomonConfig()
    config.output.write_forecast_csv = False
    artifact, path = forecast(str(source), time_column="timestamp", target_column="value", horizon=1,
                              frequency="D", output=str(tmp_path / "out"), config=config)
    register_artifact(artifact, "project", str(path))
    assert not (path / "forecast.csv").exists()
    row = artifact.results[0].forecast[0]
    scores = TrackingStore().submit_actuals("project", [(row["timestamp"], row["point"])])
    assert len(scores) == 1 and scores[0].wape == 0


def test_decision_replay_filters_source_availability_and_local_recording(tmp_path):
    clock = FixedClock(datetime(2025, 1, 3, tzinfo=timezone.utc))
    ledger = TemporalLedger(tmp_path / "ledger.db", clock=clock)
    decision_id = ledger.record_decision(execution_ids=[], policy={"version": "1"}, inputs={}, action={})
    ledger.append_decision_outcome(decision_id, outcome={"value": 1}, source_available_at="2025-01-03T00:00:00Z")
    ledger.clock = FixedClock(datetime(2025, 1, 10, tzinfo=timezone.utc))
    ledger.append_decision_outcome(decision_id, outcome={"value": 9}, source_available_at="2025-01-04T00:00:00Z")
    assert len(ledger.decision(decision_id, source_as_of="2025-01-05T00:00:00Z")["outcomes"]) == 2
    assert len(ledger.decision(decision_id, source_as_of="2025-01-05T00:00:00Z", recorded_as_of="2025-01-05T00:00:00Z")["outcomes"]) == 1


def test_tampered_frozen_history_is_rejected_before_registration_or_import(tmp_path, monkeypatch):
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "registry.db"))
    source = tmp_path / "input.csv"
    source.write_text("timestamp,value\n2025-01-01,1\n2025-01-02,2\n2025-01-03,4\n")
    artifact, path = forecast(str(source), time_column="timestamp", target_column="value", horizon=1,
                              frequency="D", output=str(tmp_path / "out"))
    (path / "history.json").write_text('{}')
    with pytest.raises(GnomonError, match="integrity"):
        register_artifact(artifact, "project", str(path))
    with pytest.raises(GnomonError, match="integrity"):
        TemporalLedger(tmp_path / "ledger.db").import_artifact(str(path))
