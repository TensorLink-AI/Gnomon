from datetime import datetime, timezone
import json
import sqlite3

import pytest

from gnomon import TemporalLedger
from gnomon.contracts import GnomonError
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock


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
    # Freeze the historical table contract; migration must not need its old writer.
    with sqlite3.connect(registry) as conn:
        conn.execute("CREATE TABLE forecasts (forecast_id TEXT, project TEXT, artifact_path TEXT)")
        conn.executemany("INSERT INTO forecasts VALUES (?,?,?)",
                         [("old", "project", str(directory)), ("missing", "project", str(tmp_path / "missing"))])
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


def test_decision_replay_filters_source_availability_and_local_recording(tmp_path):
    clock = FixedClock(datetime(2025, 1, 3, tzinfo=timezone.utc))
    ledger = TemporalLedger(tmp_path / "ledger.db", clock=clock)
    decision_id = ledger.record_decision(execution_ids=[], policy={"version": "1"}, inputs={}, action={})
    ledger.append_decision_outcome(decision_id, outcome={"value": 1}, source_available_at="2025-01-03T00:00:00Z")
    ledger.clock = FixedClock(datetime(2025, 1, 10, tzinfo=timezone.utc))
    ledger.append_decision_outcome(decision_id, outcome={"value": 9}, source_available_at="2025-01-04T00:00:00Z")
    assert len(ledger.decision(decision_id, source_as_of="2025-01-05T00:00:00Z")["outcomes"]) == 2
    assert len(ledger.decision(decision_id, source_as_of="2025-01-05T00:00:00Z", recorded_as_of="2025-01-05T00:00:00Z")["outcomes"]) == 1


def sealed_history(tmp_path):
    import hashlib
    directory = legacy_artifact(tmp_path)
    history = {"series": {"shop": [{"timestamp": f"2025-01-0{day}T00:00:00", "value": value}
                                 for day, value in enumerate([1, 2, 4], 1)]}}
    (directory / "history.json").write_text(json.dumps(history))
    files = {name: "sha256:" + hashlib.sha256((directory / name).read_bytes()).hexdigest()
             for name in ("artifact.json", "history.json")}
    (directory / "integrity.json").write_text(json.dumps({"algorithm": "sha256", "files": files}))
    return directory


def test_import_reads_sealed_history_without_the_original_source(tmp_path):
    directory = sealed_history(tmp_path)
    ledger = TemporalLedger(tmp_path / "ledger.db")
    [execution_id] = ledger.import_artifact(str(directory), naive_timezone="UTC")
    record = ledger.execution(execution_id)
    assert record["request"]["history"] == [1, 2, 4]
    assert record["result"]["metadata"]["integrity"] == "verified"
    assert record["result"]["point"] == [3]
    assert record["action_authorized"] is False


def test_tampered_frozen_history_is_rejected_before_import(tmp_path):
    directory = sealed_history(tmp_path)
    (directory / "history.json").write_text("{}")
    with pytest.raises(GnomonError, match="integrity"):
        TemporalLedger(tmp_path / "ledger.db").import_artifact(str(directory))


@pytest.mark.parametrize("kind", ["parent", "absolute", "symlink"])
def test_manifest_cannot_hash_files_outside_the_import_directory(tmp_path, kind, monkeypatch):
    from gnomon import artifact_import
    directory = sealed_history(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    name = "../outside.json" if kind == "parent" else str(outside)
    if kind == "symlink":
        (directory / "link.json").symlink_to(outside)
        name = "link.json"
    (directory / "integrity.json").write_text(json.dumps({"algorithm": "sha256", "files": {name: "ignored"}}))
    def forbidden_read(path):
        pytest.fail("outside file must not be opened")
    monkeypatch.setattr(artifact_import, "_file_digest", forbidden_read)
    with pytest.raises(GnomonError, match="escapes"):
        artifact_import.read_forecast_import(str(directory))
