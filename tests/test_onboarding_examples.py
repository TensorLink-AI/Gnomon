"""Execute the operator examples, not just assertions about their prose."""

from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys

import pytest

from gnomon import TemporalLedger

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/provider_plugin"
spec = spec_from_file_location("provider_walkthrough", EXAMPLE / "walkthrough.py")
walkthrough = module_from_spec(spec)
spec.loader.exec_module(walkthrough)


@pytest.fixture
def demo(tmp_path):
    directory = tmp_path / "operator files"
    shutil.copytree(EXAMPLE, directory)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(directory / "src")])}
    output = tmp_path / "saved run"
    process = subprocess.run([sys.executable, str(directory / "walkthrough.py"), "run", "--output-dir", str(output)],
                             cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30, check=True)
    return directory, output, env, json.loads(process.stdout)


def test_walkthrough_preserves_forecasts_and_exact_vintage_scores(demo):
    _, output, _, receipt = demo
    assert receipt["mae_history"] == [None, 1, 1.5, 4, 1.5, 1.5]
    assert receipt["score_statuses"] == ["pending", "partial", "complete", "complete", "complete", "complete"]
    assert len(set(receipt["factory_execution_ids"])) == 2
    original = TemporalLedger(output / "ledger.db")
    restored = TemporalLedger(output / "restored.db")
    assert original.execution(receipt["execution_id"])["result"]["point"] == [14, 14]
    assert restored.evaluations(receipt["execution_id"]) == original.evaluations(receipt["execution_id"])
    assert restored.decision(receipt["decision_id"])["authorization_ref"] is None
    assert receipt["action_executed"] is False
    assert (output / "ledger-backup.db").stat().st_mode & 0o777 == 0o600


def test_same_operator_config_and_plugin_work_through_cli_and_real_mcp(demo):
    directory, output, env, _ = demo
    config = output / "providers.toml"
    request = json.loads((output / "request.json").read_text())
    cli = subprocess.run([sys.executable, "-m", "gnomon", "infer", "--providers-config", str(config),
                          "--provider", "example-last", "--request", "@" + str(output / "request.json")],
                         cwd=directory, env=env, text=True, capture_output=True, check=True, timeout=30)
    result = json.loads(cli.stdout)
    assert result["result"]["point"] == [14, 14]
    assert result["result"]["metadata"]["example_package"] == "gnomon-example-provider/0.0.0"
    messages = [
        {"id": 1, "method": "tools/list"},
        {"id": 2, "method": "tools/call", "params": {"name": "gnomon_forecast", "arguments": {
            "provider": "example-mean", "request": request}}},
        {"id": 3, "method": "tools/call", "params": {"name": "gnomon_ledger", "arguments": {
            "operation": "append_actual", "series_id": request["series_id"], "value": 999,
            "valid_time": request["future_timestamps"][0], "source_available_at": "2025-01-10T00:00:00Z"}}},
    ]
    process = subprocess.run([sys.executable, "-m", "gnomon", "mcp", "serve", "--profile", "execution",
                              "--providers-config", str(config)], cwd=directory, env=env, text=True,
                             input="".join(json.dumps(item) + "\n" for item in messages),
                             capture_output=True, check=True, timeout=30)
    replies = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(replies[0]["result"]["tools"]) == 8
    assert replies[1]["result"]["structuredContent"]["result"]["point"] == [12, 12]
    assert replies[2]["result"]["isError"] is True
    assert "OUTCOME_WRITES_DISABLED" in json.dumps(replies[2])
    assert all(row["value"] != 999 for row in TemporalLedger(output / "ledger.db").actuals_as_of(request["series_id"]))


def test_walkthrough_refuses_to_overwrite_an_existing_run(demo):
    directory, output, env, _ = demo
    original = (output / "receipt.json").read_bytes()
    process = subprocess.run([sys.executable, str(directory / "walkthrough.py"), "run", "--output-dir", str(output)],
                             env=env, text=True, capture_output=True, timeout=30)
    assert process.returncode != 0
    assert (output / "receipt.json").read_bytes() == original


@pytest.mark.parametrize("symlink", [False, True])
def test_backup_never_overwrites_destination_or_symlink(tmp_path, symlink):
    source, destination = tmp_path / "ledger.db", tmp_path / "backup.db"
    TemporalLedger(source)
    protected = tmp_path / "protected"
    protected.write_bytes(b"do not replace")
    if symlink:
        destination.symlink_to(protected)
    else:
        destination.write_bytes(b"do not replace")
    with pytest.raises(FileExistsError):
        walkthrough.backup(source, destination)
    assert destination.read_bytes() == protected.read_bytes() == b"do not replace"


@pytest.mark.parametrize("missing", [False, True])
def test_backup_requires_an_existing_ledger_without_creating_a_source(tmp_path, missing):
    source, destination = tmp_path / "source.db", tmp_path / "backup.db"
    if not missing:
        with sqlite3.connect(source) as conn:
            conn.execute("CREATE TABLE unrelated (id TEXT)")
    with pytest.raises((sqlite3.OperationalError, ValueError)):
        walkthrough.backup(source, destination)
    assert source.exists() is not missing
    assert not destination.exists()


def test_backup_preserves_preupgrade_schema_and_restore_can_migrate_separately(tmp_path):
    source, destination = tmp_path / "v2.db", tmp_path / "backup.db"
    original = TemporalLedger(source)
    actual_id = original.append_actual(series_id="demo", valid_time="2025-01-01T00:00:00Z", value=1,
                                       source_available_at="2025-01-02T00:00:00Z")
    with sqlite3.connect(source) as conn:
        conn.execute("DROP TABLE study_executions")
        conn.execute("DROP TABLE studies")
        conn.execute("PRAGMA user_version=2")
    before = source.read_bytes()
    walkthrough.backup(source, destination)
    assert source.read_bytes() == before
    with sqlite3.connect(destination) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    upgraded_copy = TemporalLedger(destination)
    assert upgraded_copy.actuals_as_of("demo")[0]["actual_id"] == actual_id
    assert source.read_bytes() == before
    with sqlite3.connect(destination) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3


def test_backup_includes_committed_wal_data_without_copying_sidecar_files(tmp_path):
    source, destination = tmp_path / "live.db", tmp_path / "backup.db"
    ledger = TemporalLedger(source)
    connection = sqlite3.connect(source)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        connection.execute("BEGIN")
        assert connection.execute("SELECT count(*) FROM actuals").fetchone()[0] == 0
        actual_id = ledger.append_actual(series_id="demo", valid_time="2025-01-01T00:00:00Z", value=42,
                                         source_available_at="2025-01-02T00:00:00Z")
        assert Path(str(source) + "-wal").stat().st_size > 0
        walkthrough.backup(source, destination)
        assert TemporalLedger(destination).actuals_as_of("demo")[0]["actual_id"] == actual_id
    finally:
        connection.close()


def test_documented_legacy_import_runs_without_rewriting_the_old_registry(tmp_path):
    from gnomon.tracking import TrackingStore
    artifact = tmp_path / "old-artifact-directory"
    artifact.mkdir()
    (artifact / "artifact.json").write_text(json.dumps({
        "forecast_id": "old", "created_at": "2025-01-03T00:00:00Z",
        "task": {"schema": {"frequency": "D"}},
        "results": [{"series": "shop", "selected_model": "old-model", "support": "supported",
                     "forecast": [{"timestamp": "2025-01-04T00:00:00Z", "point": 3}]}],
    }))
    registry = tmp_path / "old-registry.db"
    TrackingStore(registry).register("old", "sales", series="shop", artifact_path=str(artifact))
    before = registry.read_bytes()
    section = (ROOT / "docs/production/OPERATIONS.md").read_text().split("## Importing historical artifacts", 1)[1]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                   env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                   text=True, capture_output=True, check=True, timeout=30)
    assert registry.read_bytes() == before
    ledger = TemporalLedger(tmp_path / "new-ledger.db")
    with sqlite3.connect(ledger.path) as conn:
        [row] = conn.execute("SELECT execution_id FROM executions").fetchall()
    imported = ledger.execution(row[0])
    assert imported["result"]["point"] == [3]
    assert imported["request"]["history"] is None
    assert imported["evidence"] == "imported_forecast"


def test_installed_smoke_drops_ambient_checkout_imports_and_profile(monkeypatch):
    spec = spec_from_file_location("offline_smoke", ROOT / "scripts/offline_wheel_smoke.py")
    smoke = module_from_spec(spec)
    spec.loader.exec_module(smoke)
    monkeypatch.setenv("PYTHONPATH", "unrelated-checkout")
    monkeypatch.setenv("GNOMON_MCP_PROFILE", "full")
    child = smoke.offline_env()
    assert "PYTHONPATH" not in child and "PYTHONHOME" not in child
    assert "GNOMON_MCP_PROFILE" not in child
    assert child["PIP_NO_INDEX"] == "1"
    assert os.environ["PYTHONPATH"] == "unrelated-checkout"
